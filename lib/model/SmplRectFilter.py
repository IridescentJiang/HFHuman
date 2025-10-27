import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..net_util import init_net
import cv2

from lib.model.vol.Voxelize import Voxelization
from lib.model.vol import util as util, constant as const

BN_MOMENTUM = 0.1


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FiLM(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(FiLM, self).__init__()

        # 全连接层，用于生成γ和β参数
        self.fc_gamma = nn.Linear(condition_dim, input_dim)
        self.fc_beta = nn.Linear(condition_dim, input_dim)
        self.flatten = nn.Flatten()

    def forward(self, x, condition):
        # 根据条件特征获取缩放scale参数和移位参数shift，即计算γ和β参数
        x = self.flatten(x)
        condition = self.flatten(condition)

        gamma = self.fc_gamma(condition)
        beta = self.fc_beta(condition)

        # 对输入特征x进行缩放和偏移，实现条件特征调整输入特征
        y = gamma * x + beta
        return y


class HighResolutionNet(nn.Module):

    def __init__(self, out_channels):
        self.inplanes = 16
        super(HighResolutionNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(16, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.layer1 = self._make_layer(Bottleneck, 16, 4)
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, out_channels * 3),
            nn.Unflatten(1, (out_channels, 3))
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.flatten(x)

        return x


class SmplRectFilter(nn.Module):

    def __init__(self,
                 opt
                 ):
        super(SmplRectFilter, self).__init__()

        self.name = 'smplrectfilter'

        self.opt = opt

        self.depth_resize_ch = 64

        self.mse_loss = nn.MSELoss()

        input_dim_pose = 25 * 3
        condition_dim_pose = self.depth_resize_ch * 3
        self.film_layer = FiLM(input_dim_pose, condition_dim_pose)

        self.hrnet = HighResolutionNet(self.depth_resize_ch)

        self.smpl_vertices_rect = None
        self.smpl_shape_rect = None
        self.smpl_pose_rect = None

        self.mlp_depth_resize_to_shape = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 224 * 224, 10)
        )


        self.mlp_smpl_shape = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        self.mlp_smpl_pose = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(25 * 3 + self.depth_resize_ch * 3, 256),
            nn.Linear(25 * 3, 256),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(512, 25 * 3),
            nn.Unflatten(1, (25, 3))
        )

        self.im_feat_list = []

        init_net(self)  # initialise weights

    def filter(self, smpl_pose, smpl_shape, depth_map):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''

        # current_depth_map_resize_pose = self.mlp_depth_resize_pose(depth_map)
        current_depth_map_resize_pose = self.hrnet(depth_map)

        # fusion_depth_pose = torch.cat((smpl_pose, current_depth_map_resize_pose), dim=1)
        fusion_depth_pose = self.film_layer(smpl_pose, current_depth_map_resize_pose)

        self.smpl_pose_rect = self.mlp_smpl_pose(fusion_depth_pose)

        current_depth_map_resize_shape = self.mlp_depth_resize_to_shape(depth_map)
        fusion_depth_shape = torch.cat((smpl_shape, current_depth_map_resize_shape), dim=1)

        self.smpl_shape_rect = self.mlp_smpl_shape(fusion_depth_shape)

        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = util.read_smpl_constants('./smpl_data')
        voxelization = Voxelization(smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras,
                                    volume_res=const.vol_res,
                                    sigma=const.semantic_encoding_sigma,
                                    smooth_kernel_size=const.smooth_kernel_size,
                                    batch_size=1)
        self.smpl_vertices_rect = voxelization.para_to_smpl(pose=self.smpl_pose_rect, shape=self.smpl_shape_rect)
        self.smpl_vertices_rect *= 2

    def get_smpl_para(self):

        bs = self.smpl_pose_rect.size()[0]
        pose = self.smpl_pose_rect[:, 0:24, :]
        pose = pose.reshape(bs, 24, 3)
        transl = self.smpl_pose_rect[:, 24, :]
        transl = transl.reshape(bs, 3)

        generated_smpl_para = {'poses': pose,
                               'betas': self.smpl_shape_rect,
                               'transl': transl
                               }
        return generated_smpl_para

    def generate_smpl_para(self):
        return self.get_smpl_para()

    def get_smpl_vertices(self):
        return self.smpl_vertices_rect

    def generate_smpl_vertices(self):
        return self.get_smpl_vertices()

    def get_error(self):
        '''
        return the loss given the ground truth labels and prediction
        '''
        error = {}

        #  global_orient and transl need to be eliminated in the pose para loss
        smpl_pose_rect_only_shape = self.smpl_pose_rect[:, 1:24, :]
        smpl_pose_gt_only_shape = self.smpl_pose_gt[:, 1:24, :]
        error['Err(smpl_pose)'] = self.mse_loss(smpl_pose_rect_only_shape, smpl_pose_gt_only_shape)

        error['Err(smpl_shape)'] = self.mse_loss(self.smpl_shape_rect, self.smpl_shape_gt)

        error['Err(smpl_vertices)'] = self.mse_loss(self.smpl_vertices_rect, self.smpl_vertices_gt)

        return error

    def forward(self, smpl_shape_pred=None, smpl_pose_pred=None,
                smpl_shape_gt=None, smpl_pose_gt=None,
                depth_map=None, smpl_vertices_gt=None):
        self.filter(smpl_pose=smpl_pose_pred, smpl_shape=smpl_shape_pred, depth_map=depth_map)

        self.smpl_shape_gt = smpl_shape_gt
        self.smpl_pose_gt = smpl_pose_gt
        self.smpl_vertices_gt = smpl_vertices_gt

        err = self.get_error()

        return err
