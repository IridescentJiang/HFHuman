import torch
import re
import torch.nn as nn
import torch.nn.functional as F
from .vol import util as util, constant as const, VolumeEncoder as ve, HourglassNet as hg, Block as block
from .BasePIFuNet import BasePIFuNet
from .MLPn import MLPn
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .HGFilters import HGFilter
from lib.model.vol.Voxelize import Voxelization
from ..net_util import init_net
from .vol.Fusion import CorrFeatureFuser2D as Fuser2D
from .vol.evaluator import Evaluator
from .Transformer import ViTF,ViT3DF

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


class Conv_FiLM(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(Conv_FiLM, self).__init__()

        # 卷积层，用于生成γ和β参数
        self.conv_gamma = nn.Conv2d(condition_dim, input_dim, kernel_size=1)
        self.conv_beta = nn.Conv2d(condition_dim, input_dim, kernel_size=1)

    def forward(self, x, condition):

        gamma = self.conv_gamma(condition)
        beta = self.conv_beta(condition)

        # 对输入特征x进行缩放和偏移，实现条件特征调整输入特征
        y = gamma * x + beta

        return y


class Decoder_depthmap_smpl_rect(nn.Module):

    def __init__(self, out_channels):
        self.inplanes = 16
        super(Decoder_depthmap_smpl_rect, self).__init__()

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



class HGPIFuNetwNML(BasePIFuNet):
    '''
    HGPIFu uses stacked hourglass as an image encoder.
    '''

    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 criteria={'occ': nn.MSELoss()},
                 use_High_Res_Component=False
                 ):
        super(HGPIFuNetwNML, self).__init__(
            projection_mode=projection_mode,
            criteria=criteria)

        self.name = 'hg_pifu_low_res'

        self.opt = opt

        self.use_High_Res_Component = use_High_Res_Component

        if self.opt.no_images:
            in_ch = 0
        else:
            in_ch = 3

        if self.opt.use_front_normal:
            in_ch += 3
        if self.opt.use_back_normal:
            in_ch += 3

        if self.opt.use_depth_map and self.opt.depth_in_front:
            if not self.use_High_Res_Component:
                in_ch += 1
            elif self.use_High_Res_Component and self.opt.allow_highres_to_use_depth:
                in_ch += 1
            else:
                pass

        if self.opt.use_human_parse_maps:
            if not self.use_High_Res_Component:
                if self.opt.use_groundtruth_human_parse_maps:
                    in_ch += 6
                else:
                    in_ch += 7
            else:
                pass

        if self.use_High_Res_Component:
            from .DifferenceIntegratedHGFilters import DifferenceIntegratedHGFilter
            self.feat_ch_2D = opt.hg_dim_high_res
            self.image_filter = DifferenceIntegratedHGFilter(1, 2, in_ch, self.feat_ch_2D,
                                                             opt.norm, opt.hg_down, False)
        else:
            if self.opt.use_transformer_encoder:
                self.feat_ch_2D = opt.hg_dim_low_res
                self.image_filter = ViTF(image_size=512, channels=in_ch)
            else:
                self.feat_ch_2D = opt.hg_dim_low_res
                self.image_filter = HGFilter(opt.num_stack_low_res, opt.hg_depth_low_res, in_ch, self.feat_ch_2D, opt.norm,
                                             opt.hg_down, False)
                # self.image_filter = hg.HourglassNet(4, 3, 128, self.feat_ch_2D, in_ch)

        if self.opt.use_depth_map and not self.opt.depth_in_front:
            self.opt.mlp_dim_low_res[0] = self.opt.mlp_dim_low_res[0] + 1  # plus 1 for the depthmap.
            print("Overwriting self.opt.mlp_dim_low_res to add in 1 dim for depth map!")

        if self.opt.use_smpl or self.opt.use_smpl_x or self.opt.use_smpl_para or self.opt.use_smpl_para_gt or self.opt.smpl_para_depth_guidance_learning:
            smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = util.read_smpl_constants('./smpl_data')
            self.voxelization = Voxelization(smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras,
                                             volume_res=const.vol_res,
                                             sigma=const.semantic_encoding_sigma,
                                             smooth_kernel_size=const.smooth_kernel_size,
                                             batch_size=1)
            self.smpl_faces = smpl_faces
            self.feat_ch_3D = opt.ve_3D_dim
            if self.opt.use_transformer_encoder:
                # self.ve = ViT3DF(vol_size=256, patch_size=16, channels=3)
                self.ve = ve.VolumeEncoder(3, self.feat_ch_3D)
            else:
                self.ve = ve.VolumeEncoder(3, self.feat_ch_3D)

            if self.opt.use_transformer_encoder:
                if self.opt.film_fusion:
                    self.mlp = MLPn(32, 1, weight_norm=False)
                else:
                    self.mlp = MLPn(32 + self.feat_ch_3D + 1, 1, weight_norm=False)
            else:
                if self.opt.film_fusion:
                    self.mlp = MLPn(self.feat_ch_2D, 1, weight_norm=False)
                else:
                    self.mlp = MLPn(self.feat_ch_2D + self.feat_ch_3D + 1, 1, weight_norm=False)

            self.depth_resize_ch = 64

            input_dim_pose = 25 * 3
            condition_dim_pose = self.depth_resize_ch * 3
            self.film_layer = FiLM(input_dim_pose, condition_dim_pose)

            if self.opt.film_fusion:
                if self.opt.use_transformer_encoder:
                    input_dim_2D = 32
                    condition_dim_3D = 33
                else:
                    input_dim_2D = self.feat_ch_2D
                    condition_dim_3D = self.feat_ch_3D + 1
                self.film_layer_2D_3D = Conv_FiLM(input_dim_2D, condition_dim_3D)

            self.decoder_depth_smpl_rect = Decoder_depthmap_smpl_rect(self.depth_resize_ch)

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


            self.fuse_module_3to2_list = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.feat_ch_2D + self.feat_ch_3D + 1, self.feat_ch_2D + 1, 1),
                    nn.BatchNorm2d(self.feat_ch_2D + 1),
                    nn.ReLU()
                ), nn.Sequential(
                    nn.Conv2d(self.feat_ch_2D + self.feat_ch_3D + 1, self.feat_ch_2D + 1, 1),
                    nn.BatchNorm2d(self.feat_ch_2D + 1),
                    nn.ReLU()
                ), nn.Sequential(
                    nn.Conv2d(self.feat_ch_2D + self.feat_ch_3D + 1, self.feat_ch_2D + 1, 1),
                    nn.BatchNorm2d(self.feat_ch_2D + 1),
                    nn.ReLU()
                )])

            self.fuse_module_2to3_list = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.feat_ch_2D + self.feat_ch_3D + 1, self.feat_ch_3D, 1),
                    nn.BatchNorm2d(self.feat_ch_3D),
                    nn.ReLU()
                ), nn.Sequential(
                    nn.Conv2d(self.feat_ch_2D + self.feat_ch_3D + 1, self.feat_ch_3D, 1),
                    nn.BatchNorm2d(self.feat_ch_3D),
                    nn.ReLU()
                ), nn.Sequential(
                    nn.Conv2d(self.feat_ch_2D + self.feat_ch_3D + 1, self.feat_ch_3D, 1),
                    nn.BatchNorm2d(self.feat_ch_3D),
                    nn.ReLU()
                )])

            self.fuse_final_list = nn.ModuleList([
                nn.Conv2d(self.feat_ch_2D + self.feat_ch_3D + 1, self.feat_ch_2D, kernel_size=1),
                nn.Conv2d(self.feat_ch_2D + self.feat_ch_3D + 1, self.feat_ch_2D, kernel_size=1),
                nn.Conv2d(self.feat_ch_2D + self.feat_ch_3D + 1, self.feat_ch_2D, kernel_size=1)
            ])

            self.img_transform_list = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.feat_ch_2D + 1, self.feat_ch_2D + 1, 1),
                    nn.BatchNorm2d(self.feat_ch_2D + 1),
                ),
                nn.Sequential(
                    nn.Conv2d(self.feat_ch_2D + 1, self.feat_ch_2D + 1, 1),
                    nn.BatchNorm2d(self.feat_ch_2D + 1),
                ),
                nn.Sequential(
                    nn.Conv2d(self.feat_ch_2D + 1, self.feat_ch_2D + 1, 1),
                    nn.BatchNorm2d(self.feat_ch_2D + 1),
                )
            ])

            self.vol_transform_list = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.feat_ch_3D, self.feat_ch_3D + self.feat_ch_2D + 1, 1),
                    nn.Conv2d(self.feat_ch_3D + self.feat_ch_2D + 1, self.feat_ch_2D + 1, 1),
                    nn.BatchNorm2d(self.feat_ch_2D + 1),
                ),
                nn.Sequential(
                    nn.Conv2d(self.feat_ch_3D, self.feat_ch_3D + self.feat_ch_2D + 1, 1),
                    nn.Conv2d(self.feat_ch_3D + self.feat_ch_2D + 1, self.feat_ch_2D + 1, 1),
                    nn.BatchNorm2d(self.feat_ch_2D + 1),
                ),
                nn.Sequential(
                    nn.Conv2d(self.feat_ch_3D, self.feat_ch_3D + self.feat_ch_2D + 1, 1),
                    nn.Conv2d(self.feat_ch_3D + self.feat_ch_2D + 1, self.feat_ch_2D + 1, 1),
                    nn.BatchNorm2d(self.feat_ch_2D + 1),
                )])

            self.relu = nn.ReLU()

            self.Fuser2D_list = nn.ModuleList([
                Fuser2D(self.feat_ch_2D, self.feat_ch_3D, num_heads=4),
                Fuser2D(self.feat_ch_2D, self.feat_ch_3D, num_heads=4),
                Fuser2D(self.feat_ch_2D, self.feat_ch_3D, num_heads=4)
            ])

            # self.mlp = MLP(
            #     filter_channels=self.opt.mlp_dim_low_res_smpl,
            #     merge_layer=self.opt.merge_layer_low_res,
            #     res_layers=self.opt.mlp_res_layers_low_res,
            #     norm="no_norm",
            #     last_op=nn.Sigmoid())
        else:
            if self.opt.use_transformer_encoder:
                self.mlp = MLP(
                    filter_channels=self.opt.mlp_dim_low_res_transformer,
                    merge_layer=self.opt.merge_layer_low_res,
                    res_layers=self.opt.mlp_res_layers_low_res,
                    norm="no_norm",
                    last_op=nn.Sigmoid())
            else:
                self.mlp = MLP(
                    filter_channels=self.opt.mlp_dim_low_res,
                    merge_layer=self.opt.merge_layer_low_res,
                    res_layers=self.opt.mlp_res_layers_low_res,
                    norm="no_norm",
                    last_op=nn.Sigmoid())

        self.spatial_enc = DepthNormalizer(opt)

        self.im_feat_list = []
        self.im_feat = None
        self.tmpx = None
        self.normx = None
        self.phi = None

        self.intermediate_preds_list = []

        init_net(self)  # initialise weights

        self.netF = None
        self.netB = None

        self.nmlF = None
        self.nmlB = None

        self.gamma = None

        self.current_depth_map = None

    def filter(self, images, nmlF=None, nmlB=None, current_depth_map=None, netG_output_map=None, human_parse_map=None,
               mask_low_res_tensor=None, mask_high_res_tensor=None, smpl=None, smpl_shape=None, smpl_pose=None,
               save_path=None, rotation_matrix=None, smpl_test_opt=False, smpl_para_rect=True, calib=None, scale=None,
               depthmap_for_smpl_rect=None, depth_map_for_opt=None, mask_for_opt=None, nml_for_opt=None):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''
        if self.opt.use_depth_map and not self.opt.depth_in_front:
            self.current_depth_map = current_depth_map

        self.mask_high_res_tensor = mask_high_res_tensor
        self.mask_low_res_tensor = mask_low_res_tensor

        ori_images = images

        nmls = []
        # if you wish to train jointly, remove detach etc.
        with torch.no_grad():
            if self.opt.use_front_normal:
                if nmlF == None:
                    raise Exception("NORMAL MAPS ARE MISSING!!")

                self.nmlF = nmlF
                nmls.append(self.nmlF)
            if self.opt.use_back_normal:
                if nmlB == None:
                    raise Exception("NORMAL MAPS ARE MISSING!!")

                self.nmlB = nmlB
                nmls.append(self.nmlB)

        # Concatenate the input image with the two normals maps together
        if self.opt.no_images:
            if self.opt.use_depth_map and self.opt.depth_in_front and (current_depth_map is not None):
                images = current_depth_map
            if len(nmls) != 0:
                nmls = torch.cat(nmls, 1)
                if images.size()[2:] != nmls.size()[2:]:
                    nmls = nn.Upsample(size=images.size()[2:], mode='bilinear', align_corners=True)(nmls)
                images = torch.cat([images, nmls], 1)
            if self.opt.use_human_parse_maps and (human_parse_map is not None):
                images = torch.cat([images, human_parse_map], 1)
        else:
            if len(nmls) != 0:
                nmls = torch.cat(nmls, 1)
                if images.size()[2:] != nmls.size()[2:]:
                    nmls = nn.Upsample(size=images.size()[2:], mode='bilinear', align_corners=True)(nmls)
                images = torch.cat([images, nmls], 1)

            if self.opt.use_depth_map and self.opt.depth_in_front and (current_depth_map is not None):
                images = torch.cat([images, current_depth_map], 1)

            if self.opt.use_human_parse_maps and (human_parse_map is not None):
                images = torch.cat([images, human_parse_map], 1)

        if self.opt.use_smpl or self.opt.use_smpl_x or self.opt.use_smpl_para or self.opt.use_smpl_para_gt or self.opt.smpl_para_depth_guidance_learning:

            import trimesh
            smpl_pose_before_opt = None
            smpl_shape_before_opt = None

            if self.opt.use_smpl or self.opt.use_smpl_x:
                smpl_ori = torch.matmul(smpl, rotation_matrix)
                smpl_model = smpl_ori

                if not self.training:
                    save_smpl_ori_path = save_path[:-4] + '_smpl_ori.obj'
                    vertices = smpl_ori.squeeze(0).cpu().numpy()
                    vertices *= 2
                    smpl_mesh = trimesh.Trimesh(vertices=vertices, faces=self.smpl_faces)
                    smpl_mesh.export(save_smpl_ori_path)

            elif (self.opt.use_smpl_para or self.opt.use_smpl_para_gt or self.opt.use_smpl_para_opt) and not self.opt.smpl_para_depth_guidance_learning:
                if self.opt.use_smpl_para_gt:
                    smpl_ori = self.voxelization.para_to_smpl(pose=smpl_pose, shape=smpl_shape)
                else:
                    smpl_ori = self.voxelization.para_to_smpl_reverse(pose=smpl_pose, shape=smpl_shape)
                smpl_ori = torch.matmul(smpl_ori, rotation_matrix)
                smpl_model = smpl_ori

                smpl_pose_before_opt = smpl_pose.clone()
                smpl_shape_before_opt = smpl_shape.clone()

                if not self.training:
                    if self.opt.use_smpl_para_opt or smpl_test_opt:
                        save_smpl_ori_path = save_path[:-4] + '_smpl_ori.obj'
                        vertices = smpl_ori.squeeze(0).cpu().detach().numpy()
                        vertices *= 2
                        smpl_mesh = trimesh.Trimesh(vertices=vertices, faces=self.smpl_faces)
                        smpl_mesh.export(save_smpl_ori_path)

            else:
                if smpl_para_rect:
                    current_depth_map_resize_pose = self.decoder_depth_smpl_rect(depthmap_for_smpl_rect)
                    fusion_depth_pose = self.film_layer(smpl_pose, current_depth_map_resize_pose)

                    self.smpl_pose_rect = self.mlp_smpl_pose(fusion_depth_pose)

                    current_depth_map_resize_shape = self.mlp_depth_resize_to_shape(depthmap_for_smpl_rect)
                    fusion_depth_shape = torch.cat((smpl_shape, current_depth_map_resize_shape), dim=1)

                    self.smpl_shape_rect = self.mlp_smpl_shape(fusion_depth_shape)

                    self.smpl_vertices_rect = self.voxelization.para_to_smpl(pose=self.smpl_pose_rect,
                                                                             shape=self.smpl_shape_rect)
                else:
                    self.smpl_pose_rect = smpl_pose.clone()
                    self.smpl_shape_rect = smpl_shape.clone()
                    self.smpl_vertices_rect = self.voxelization.para_to_smpl(pose=self.smpl_pose_rect,
                                                                             shape=self.smpl_shape_rect)

                smpl_ori = self.voxelization.para_to_smpl(pose=smpl_pose, shape=smpl_shape)

                smpl_vertices_rect_rot = torch.matmul(self.smpl_vertices_rect, rotation_matrix)
                smpl_ori = torch.matmul(smpl_ori, rotation_matrix)

                smpl_model = smpl_vertices_rect_rot

                if not self.training and smpl_para_rect:
                    save_smpl_rect_path = save_path[:-4] + '_smpl_rect.obj'
                    vertices = smpl_vertices_rect_rot.squeeze(0).detach().cpu().numpy()
                    vertices *= 2
                    smpl_mesh = trimesh.Trimesh(vertices=vertices, faces=self.smpl_faces)
                    smpl_mesh.export(save_smpl_rect_path)

                    if self.opt.smpl_para_depth_guidance_learning:
                        save_smpl_ori_path = save_path[:-4] + '_smpl_ori.obj'
                        vertices = smpl_ori.squeeze(0).detach().cpu().numpy()
                        vertices *= 2
                        smpl_mesh = trimesh.Trimesh(vertices=vertices, faces=self.smpl_faces)
                        smpl_mesh.export(save_smpl_ori_path)

                smpl_pose_before_opt = self.smpl_pose_rect.clone()
                smpl_shape_before_opt = self.smpl_shape_rect.clone()

            if smpl_test_opt:
                # smpl predict optimized model
                evaluator = Evaluator(images.device)
                if netG_output_map is not None:
                    netG_output_map = netG_output_map.detach()

                pattern = r"\d{4}"
                matches = re.findall(pattern, save_path)
                name = matches[0] if len(matches) > 0 else None


                smpl_pose_depth_opt, smpl_shape_depth_opt, smpl_depth_opt = evaluator.optm_smpl_param_depth(pose_transl=smpl_pose_before_opt,
                                                                      shape=smpl_shape_before_opt,
                                                                      iter_num=100,
                                                                      refer_depth_map=depth_map_for_opt,
                                                                      nmlF=nml_for_opt,
                                                                      rotation_matrix=rotation_matrix,
                                                                      mask=mask_for_opt,
                                                                      name=name,
                                                                      calib=calib,
                                                                      scale=scale,
                                                                      opt=self.opt)

                smpl_pose_opt, smpl_shape_opt, smpl_opt = evaluator.optm_smpl_param_shrink(img=ori_images,
                                                                                           calib=calib,
                                                                                           pose=smpl_pose_depth_opt,
                                                                                           shape=smpl_shape_depth_opt,
                                                                                           iter_num=100,
                                                                                           nmlF=nmlF,
                                                                                           nmlB=nmlB,
                                                                                           netG_output_map=netG_output_map,
                                                                                           current_depth_map=current_depth_map,
                                                                                           human_parse_map=human_parse_map,
                                                                                           mask_low_res_tensor=mask_low_res_tensor,
                                                                                           mask_high_res_tensor=None,
                                                                                           smpl=smpl_depth_opt,
                                                                                           save_path=save_path,
                                                                                           rotation_matrix=rotation_matrix,
                                                                                           net=self, opt=self.opt)

                if self.opt.smpl_para_depth_guidance_learning or self.opt.use_smpl_para_gt:
                    smpl_opt = self.voxelization.para_to_smpl(pose=smpl_pose_opt, shape=smpl_shape_opt)
                else:
                    smpl_opt = self.voxelization.para_to_smpl_reverse(pose=smpl_pose_opt, shape=smpl_shape_opt)
                smpl_opt = torch.matmul(smpl_opt, rotation_matrix)
                save_smpl_opt_path = save_path[:-4] + '_smpl_opt.obj'
                vertices = smpl_opt.squeeze(0).detach().cpu().numpy()
                vertices *= 2
                smpl_mesh = trimesh.Trimesh(vertices=vertices, faces=self.smpl_faces)
                smpl_mesh.export(save_smpl_opt_path)

                smpl_model = smpl_opt

            smpl_model = smpl_model.detach()
            split_smpl = torch.split(smpl_model, split_size_or_sections=1, dim=0)
            vol_list = []
            for tensor in split_smpl:
                vol = self.voxelization(tensor)
                vol_list.append(vol)
            vol = torch.cat(vol_list, dim=0)
            self.vol_feats = self.ve(vol)

        if self.use_High_Res_Component:
            self.im_feat_list, self.normx = self.image_filter(images, netG_output_map)
        else:
            if self.opt.use_transformer_encoder:
                self.im_feat = self.image_filter(images)
            else:
                self.im_feat_list, self.normx = self.image_filter(images)

    def query(self, points, calibs, transforms=None, labels=None, update_pred=True, update_phi=True):
        '''
        given 3d points, we obtain 2d projection of these given the camera matrices.
        filter needs to be called beforehand.
        the prediction is stored to self.preds
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image. If calibs is [B,3,4], it is fine as well.
            transforms: [B, 2, 3] image space coordinate transforms
            labels: [B, C, N] ground truth labels (for supervision only)
        return:
            [B, C, N] prediction
        '''
        xyz = self.projection(points, calibs, transforms)  # [B, 3, N]
        xy = xyz[:, :2, :]  # [B, 2, N]

        if self.use_High_Res_Component and self.opt.use_mask_for_rendering_high_res and (
                self.mask_high_res_tensor is not None):
            mask_values = self.index(self.mask_high_res_tensor, xy)
        if (not self.use_High_Res_Component) and self.opt.use_mask_for_rendering_low_res and (
                self.mask_low_res_tensor is not None):
            mask_values = self.index(self.mask_low_res_tensor, xy)

            # if the point is outside bounding box, return outside.
        in_bb = (xyz >= -1) & (xyz <= 1)  # [B, 3, N]
        in_bb = in_bb[:, 0, :] & in_bb[:, 1, :] & in_bb[:, 2, :]  # [B, N]
        in_bb = in_bb[:, None, :].detach().float()  # [B, 1, N]

        is_zero_bool = (
                xyz == 0)  # [B, 3, N]; remove the (0,0,0) point that has been used to discard unwanted sample pts
        is_zero_bool = is_zero_bool[:, 0, :] & is_zero_bool[:, 1, :] & is_zero_bool[:, 2, :]  # [B, N]
        not_zero_bool = torch.logical_not(is_zero_bool)
        not_zero_bool = not_zero_bool[:, None, :].detach().float()  # [B, 1, N]

        if labels is not None:
            self.labels = in_bb * labels  # [B, 1, N]
            self.labels = not_zero_bool * self.labels

            size_of_batch = self.labels.shape[0]

        sp_feat = self.spatial_enc(xyz, calibs=calibs)  # sp_feat is the normalized z value. (x and y are removed)

        intermediate_preds_list = []

        phi = None
        if self.opt.use_smpl or self.opt.use_smpl_x or self.opt.use_smpl_para or self.opt.use_smpl_para_gt or self.opt.smpl_para_depth_guidance_learning:

            if self.opt.use_transformer_encoder:
                batch_size = points.size()[0]
                point_num = points.size()[2]
                xyz = xyz.transpose(1, 2)
                points = points.transpose(1, 2)

                h_grid = xyz[:, :, 0].view(batch_size, point_num, 1, 1)
                v_grid = xyz[:, :, 1].view(batch_size, point_num, 1, 1)
                grid_2d = torch.cat([h_grid, v_grid], dim=-1)

                x_grid = points[:, :, 0].view(batch_size, point_num, 1, 1, 1)
                y_grid = points[:, :, 1].view(batch_size, point_num, 1, 1, 1)
                z_grid = points[:, :, 2].view(batch_size, point_num, 1, 1, 1)
                grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)

                sp_feat = sp_feat.unsqueeze(-1)

                for i, vol_feat in enumerate(self.vol_feats):

                    pt_feat_2D = F.grid_sample(input=self.im_feat, grid=grid_2d, align_corners=False,
                                               mode='bilinear', padding_mode='border')

                    pt_feat_3D = F.grid_sample(input=vol_feat, grid=grid_3d, align_corners=False,
                                               mode='bilinear', padding_mode='border')

                    pt_feat_3D = pt_feat_3D.view([batch_size, -1, point_num, 1])

                    # fuse 3D and spatial feature
                    pt_feat_3D = torch.cat([pt_feat_3D, sp_feat], dim=1)

                    if self.opt.film_fusion:
                        pt_feat_fuse = self.film_layer_2D_3D(pt_feat_2D, pt_feat_3D)
                    else:
                        pt_feat_fuse = torch.cat([pt_feat_2D, pt_feat_3D], dim=1)

                    pred = self.mlp(pt_feat_fuse)  # shape = [batch_size, channels, point_num, 1]
                    pred = pred.permute([0, 3, 1, 2])
                    pred = pred.view([batch_size, 1, point_num])

                    pred = in_bb * pred
                    pred = not_zero_bool * pred

                    if self.use_High_Res_Component and self.opt.use_mask_for_rendering_high_res and (
                            self.mask_high_res_tensor is not None):
                        pred = mask_values * pred
                    if (not self.use_High_Res_Component) and self.opt.use_mask_for_rendering_low_res and (
                            self.mask_low_res_tensor is not None):
                        pred = mask_values * pred

                    intermediate_preds_list.append(pred)
            else:
                batch_size = points.size()[0]
                point_num = points.size()[2]
                self.im_feat_list = self.im_feat_list[-len(self.vol_feats):]
                xyz = xyz.transpose(1, 2)
                points = points.transpose(1, 2)

                h_grid = xyz[:, :, 0].view(batch_size, point_num, 1, 1)
                v_grid = xyz[:, :, 1].view(batch_size, point_num, 1, 1)
                grid_2d = torch.cat([h_grid, v_grid], dim=-1)

                x_grid = points[:, :, 0].view(batch_size, point_num, 1, 1, 1)
                y_grid = points[:, :, 1].view(batch_size, point_num, 1, 1, 1)
                z_grid = points[:, :, 2].view(batch_size, point_num, 1, 1, 1)
                grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)

                sp_feat = sp_feat.unsqueeze(-1)

                for i, (img_feat, vol_feat) in enumerate(zip(self.im_feat_list, self.vol_feats)):

                    pt_feat_2D = F.grid_sample(input=img_feat, grid=grid_2d, align_corners=False,
                                               mode='bilinear', padding_mode='border')

                    pt_feat_3D = F.grid_sample(input=vol_feat, grid=grid_3d, align_corners=False,
                                               mode='bilinear', padding_mode='border')

                    # pt_feat_2D = pt_feat_2D.view([batch_size, -1, point_num, 1])
                    pt_feat_3D = pt_feat_3D.view([batch_size, -1, point_num, 1])

                    # fuse 2D and spatial feature
                    # pt_feat_2D = torch.cat([pt_feat_2D, sp_feat], dim=1)

                    # fuse 3D and spatial feature
                    pt_feat_3D = torch.cat([pt_feat_3D, sp_feat], dim=1)

                    # fuse the 2D and 3D feature
                    # pt_feat_3D = vol_transform(pt_feat_3D)
                    # pt_feat_3to2 = self.fuse_feat(pt_feat_2D, pt_feat_3D, self.fuse_module_3to2_list[i])
                    # pt_feat_3to2 = pt_feat_2D + pt_feat_3to2
                    # pt_feat_fuse = self.img_transform_list[i](pt_feat_3to2)
                    #
                    # pt_feat_2to3 = self.fuse_feat(pt_feat_3D, pt_feat_2D, self.fuse_module_2to3_list[i])
                    # pt_feat_2to3 = pt_feat_3D + pt_feat_2to3
                    #
                    # pt_feat_fuse = torch.cat([pt_feat_3to2, pt_feat_2to3], dim=1)
                    # pt_feat_fuse = self.fuse_final_list[i](pt_feat_fuse)

                    if self.opt.film_fusion:
                        pt_feat_fuse = self.film_layer_2D_3D(pt_feat_2D, pt_feat_3D)
                    else:
                        pt_feat_fuse = torch.cat([pt_feat_2D, pt_feat_3D], dim=1)
                    # pt_feat_fuse = pt_feat_3D

                    # pt_feat_fuse = self.Fuser2D(pt_feat_2D, pt_feat_3D, sp_feat)

                    pred = self.mlp(pt_feat_fuse)  # shape = [batch_size, channels, point_num, 1]
                    pred = pred.permute([0, 3, 1, 2])
                    pred = pred.view([batch_size, 1, point_num])

                    pred = in_bb * pred
                    pred = not_zero_bool * pred

                    if self.use_High_Res_Component and self.opt.use_mask_for_rendering_high_res and (
                            self.mask_high_res_tensor is not None):
                        pred = mask_values * pred
                    if (not self.use_High_Res_Component) and self.opt.use_mask_for_rendering_low_res and (
                            self.mask_low_res_tensor is not None):
                        pred = mask_values * pred

                    intermediate_preds_list.append(pred)

        else:
            for i, im_feat in enumerate(self.im_feat_list):

                if self.opt.use_depth_map and not self.opt.depth_in_front:
                    point_local_feat_list = [self.index(im_feat, xy), self.index(self.current_depth_map, xy), sp_feat]
                else:
                    point_local_feat_list = [self.index(im_feat, xy),
                                             sp_feat]  # z_feat has already gone through a round of indexing. 'point_local_feat_list' should have shape of [batch_size, 272, num_of_points]
                point_local_feat = torch.cat(point_local_feat_list, 1)
                pred, phi = self.mlp(point_local_feat)  # phi is activations from an intermediate layer of the MLP
                pred = in_bb * pred
                pred = not_zero_bool * pred
                if self.use_High_Res_Component and self.opt.use_mask_for_rendering_high_res and (
                        self.mask_high_res_tensor is not None):
                    pred = mask_values * pred
                if (not self.use_High_Res_Component) and self.opt.use_mask_for_rendering_low_res and (
                        self.mask_low_res_tensor is not None):
                    pred = mask_values * pred

                intermediate_preds_list.append(pred)

        if update_phi:
            self.phi = phi

        if update_pred:
            self.intermediate_preds_list = intermediate_preds_list
            self.preds = self.intermediate_preds_list[-1]

    def calc_normal(self, points, calibs, transforms=None, labels=None, delta=0.01, fd_type='forward'):
        '''
        return surface normal in 'model' space.
        it computes normal only in the last stack.
        note that the current implementation use forward difference.
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
            delta: perturbation for finite difference
            fd_type: finite difference type (forward/backward/central) 
        '''
        pdx = points.clone()
        pdx[:, 0, :] += delta
        pdy = points.clone()
        pdy[:, 1, :] += delta
        pdz = points.clone()
        pdz[:, 2, :] += delta

        if labels is not None:
            self.labels_nml = labels

        points_all = torch.stack([points, pdx, pdy, pdz], 3)
        points_all = points_all.view(*points.size()[:2], -1)
        xyz = self.projection_orthogonal(points_all, calibs, transforms)
        xy = xyz[:, :2, :]

        im_feat = self.im_feat_list[-1]
        sp_feat = self.spatial_enc(xyz, calibs=calibs)

        point_local_feat_list = [self.index(im_feat, xy), sp_feat]
        point_local_feat = torch.cat(point_local_feat_list, 1)

        pred = self.mlp(point_local_feat)[0]

        pred = pred.view(*pred.size()[:2], -1, 4)  # (B, 1, N, 4)

        # divide by delta is omitted since it's normalized anyway
        dfdx = pred[:, :, :, 1] - pred[:, :, :, 0]
        dfdy = pred[:, :, :, 2] - pred[:, :, :, 0]
        dfdz = pred[:, :, :, 3] - pred[:, :, :, 0]

        nml = -torch.cat([dfdx, dfdy, dfdz], 1)
        nml = F.normalize(nml, dim=1, eps=1e-8)

        self.nmls = nml

    def gather_3to2(self, feat_3D, idx):
        feat_3D = torch.cat((feat_3D, torch.zeros_like(feat_3D[:1, :])), 0)
        return feat_3D[idx]

    def fuse_feat(self, feat_o, fuse_a, layer):
        fuse_a = torch.cat([feat_o, fuse_a], 1)
        fuse_a = layer(fuse_a)
        return fuse_a

    def get_im_feat(self):
        '''
        return the image filter in the last stack
        return:
            [B, C, H, W]
        '''
        return self.im_feat_list[-1]

    def get_error(self, points=None, smpl_pose_gt=None, smpl_shape_gt=None, smpl_vertices_gt=None):
        '''
        return the loss given the ground truth labels and prediction
        '''
        error = {}
        error['Err(occ)'] = 0
        for preds in self.intermediate_preds_list:
            error['Err(occ)'] += self.criteria['occ'](preds, self.labels)

        error['Err(occ)'] /= len(self.intermediate_preds_list)


        if self.nmls is not None and self.labels_nml is not None:
            error['Err(nml)'] = self.criteria['nml'](self.nmls, self.labels_nml)

        if self.opt.smpl_para_depth_guidance_learning:
            smpl_pose_rect_only_pose = self.smpl_pose_rect[:, 1:24, :]
            smpl_pose_gt_only_pose = smpl_pose_gt[:, 1:24, :]

            error['Err(smpl_pose)'] = self.criteria['occ'](smpl_pose_rect_only_pose, smpl_pose_gt_only_pose)
            error['Err(smpl_shape)'] = self.criteria['occ'](self.smpl_shape_rect, smpl_shape_gt)

            # smpl_vertices_gt match the 2 scales of self.smpl_vertices_rect
            self.smpl_vertices_rect *= 2
            error['Err(smpl_vertices)'] = self.criteria['occ'](self.smpl_vertices_rect, smpl_vertices_gt)

        return error

    def forward(self, images, points, calibs, scales, rotation_matrix, labels, points_nml=None, labels_nml=None, nmlF=None,
                nmlB=None,current_depth_map=None, netG_output_map=None, human_parse_map=None, mask_low_res_tensor=None,
                mask_high_res_tensor=None, smpl=None, smpl_shape=None, smpl_pose=None, smpl_shape_gt=None,
                smpl_pose_gt=None, smpl_vertices_gt=None, depthmap_for_smpl_rect=None,depth_map_for_opt=None):
        self.filter(images, nmlF=nmlF, nmlB=nmlB, current_depth_map=current_depth_map, netG_output_map=netG_output_map,
                    human_parse_map=human_parse_map, mask_low_res_tensor=mask_low_res_tensor,
                    mask_high_res_tensor=mask_high_res_tensor, smpl=smpl, smpl_shape=smpl_shape, smpl_pose=smpl_pose,
                    rotation_matrix=rotation_matrix, depthmap_for_smpl_rect=depthmap_for_smpl_rect, depth_map_for_opt=depth_map_for_opt,
                    calib=calibs, scale=scales)
        self.query(points, calibs, labels=labels)
        if points_nml is not None and labels_nml is not None:
            self.calc_normal(points_nml, calibs, labels=labels_nml)
        res = self.get_preds()

        err = self.get_error(smpl_pose_gt=smpl_pose_gt, smpl_shape_gt=smpl_shape_gt, smpl_vertices_gt=smpl_vertices_gt)

        return err, res
