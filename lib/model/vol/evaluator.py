"""
This file includes the full training procedure.
"""
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from . import util, constant as const
from .TetraSmpl import TetraSMPL
from .Voxelize import Voxelization
from .geometric_layers import rodrigues, orthographic_projection
from .Smpl import SMPL
from .Mesh import Mesh
from rendering_script.render_depthmap_smpl import Render
from lib.mesh_util import projection

current_path = os.path.dirname(os.path.abspath(__file__))

# 定义神经网络模型
class poseOptNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(poseOptNetwork, self).__init__()
        # 定义网络的层和参数
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Evaluator(object):
    def __init__(self, device):
        super(Evaluator, self).__init__()
        util.configure_logging(True, False, None)

        self.device = device
        mesh_downsampling_path = os.path.join(current_path, "smpl_data/mesh_downsampling.npz")
        self.graph_mesh = Mesh(filename=mesh_downsampling_path)

        # neural voxelization components
        basicModel_path = os.path.join(current_path, "smpl_data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
        tetra_smpl_path = os.path.join(current_path, "smpl_data/tetra_smpl.npz")
        self.smpl = SMPL(basicModel_path).to(self.device)
        self.tet_smpl = TetraSMPL(basicModel_path,
                                  tetra_smpl_path).to(self.device)
        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
            util.read_smpl_constants('./smpl_data')
        self.smpl_faces = smpl_faces
        self.voxelization = Voxelization(smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras,
                                         volume_res=const.vol_res,
                                         sigma=const.semantic_encoding_sigma,
                                         smooth_kernel_size=const.smooth_kernel_size,
                                         batch_size=1).to(self.device)
        self.render = Render(size=512, device=device)

    def optm_smpl_param_depth(self, pose_transl, shape, iter_num, refer_depth_map, nmlF, rotation_matrix, mask, name,
                              calib, scale, opt):
        assert iter_num > 0

        mask = F.interpolate(mask, size=(opt.loadSizeDepthOpt, opt.loadSizeDepthOpt))
        refer_depth_map = refer_depth_map.to(torch.float32)

        bs = pose_transl.size()[0]
        pose = pose_transl[:, :24, :]
        transl = pose_transl[:, 24:25, :]
        pose = pose.view(bs, 72)

        theta_new = torch.nn.Parameter(pose)
        theta_new_limb = []
        theta_new_limb.append(torch.nn.Parameter(pose[:, 0:3]))  # orient_global
        theta_new_limb.append(torch.nn.Parameter(pose[:, 3:9]))  # thigh
        theta_new_limb.append(torch.nn.Parameter(pose[:, 12:18]))  # knees
        theta_new_limb.append(torch.nn.Parameter(pose[:, 39:45]))  # shoulders
        theta_new_limb.append(torch.nn.Parameter(pose[:, 48:60]))  # arms

        beta_new = torch.nn.Parameter(shape)
        transl_new = torch.nn.Parameter(transl)

        theta_orig = theta_new.clone().detach()
        betas_orig = beta_new.clone().detach()
        transl_orig = transl_new.clone().detach()

        # parameters = [beta_new] + [transl_new]
        # parameters = [theta_new_limb[i] for i in range(5)]
        parameters_mask = [transl_new]

        optimizer_smpl_mask = torch.optim.Adam(parameters_mask, lr=1e-2, amsgrad=True)

        scheduler_smpl_mask = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl_mask,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=5,
        )

        parameters_depth = [theta_new_limb[i] for i in range(5)] + [beta_new]
        # parameters_depth = [theta_new]

        optimizer_smpl_depth = torch.optim.Adam(parameters_depth, lr=2e-3, amsgrad=True)

        scheduler_smpl_depth = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl_depth,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=5,
        )

        optm_theta = torch.optim.Adam(params=(theta_new,), lr=1e-1)
        optm_beta = torch.optim.Adam(params=(beta_new,), lr=5e-2)
        # optm_transl = torch.optim.Adam(params=(transl_new_1,), lr=2e-2)
        optm_transl = torch.optim.Adam(params=(transl_new,), lr=2e-4)

        pbar_mask = tqdm(range(int(iter_num / 2)), desc='Smpl Mask Optimizing:')
        pbar_depth = tqdm(range(int(iter_num / 2)), desc='Smpl Depth Optimizing:')

        best_loss_fitting_first = torch.inf
        best_loss_fitting_second = torch.inf
        best_i = -1

        # bs = gt_pose_transl.size()[0]
        # gt_pose = gt_pose_transl[:, :24, :]
        # gt_transl = gt_pose_transl[:, 24:25, :]
        # gt_pose = gt_pose.view(bs, 72)

        refer_mask = mask * 100

        # gt_mesh_vertices, gt_mesh_faces = self.render.load_scan(GT_mesh_file)
        # rotation_inv = np.linalg.inv(rotation_matrix.cpu().detach().numpy())
        # gt_mesh_vertices = np.dot(gt_mesh_vertices, rotation_inv)
        # refer_mask, mask_B = self.rendering_gt_mesh_mask(verts=gt_mesh_vertices, faces=gt_mesh_faces)
        # gt_mesh_depth = self.rendering_gt_mesh_depth(verts=gt_mesh_vertices, faces=gt_mesh_faces, refer_depth_map=refer_depth_map)
        # gt_mesh_depth = gt_mesh_depth.squeeze(0).squeeze(0)

        for i in pbar_mask:
            theta = theta_new

            # theta = torch.cat([theta_new_limb[0], theta_new_limb[1], theta_orig[:, 9:12],
            #                   theta_new_limb[2], theta_orig[:, 18:39],
            #                   theta_new_limb[3], theta_orig[:, 45:48],
            #                   theta_new_limb[4], theta_orig[:, 60:]], dim=1)

            # transl_new_ = torch.cat([transl_new[:,:,0:2], transl_orig[:,:,2].unsqueeze(2)], dim=2)

            with torch.no_grad():
                transl_new.data[:, :, 2] = transl_orig[:, :, 2]
            transl_new_ = transl_new


            verts, faces = self.get_smpl_verts_face(pose=theta, shape=beta_new,
                                                    transl=transl_new_,
                                                    rotation_matrix=rotation_matrix,
                                                    refer_depth_map=refer_depth_map,
                                                    mask=refer_mask, calib=calib, scale=scale)

            self.render.load_meshes(verts, faces)

            mask_F, mask_B = self.rendering_smpl_mask(verts=verts,
                                                      faces=faces,
                                                      scale=scale)

            if i % (iter_num / 10) == 0 or i == (iter_num - 1):
                save_path_diff_mask_depth = '%s/%s/opt_smpl_diff_%s_1_%s.png' % (
                    opt.results_path, opt.name, name, i)
                diff_mask = abs(mask_F.squeeze(0) - refer_mask.squeeze(0).squeeze(0))
                diff_mask_output = diff_mask.cpu().detach().numpy()
                diff_mask_output = diff_mask_output + 255 / 4
                diff_mask_output = diff_mask_output.astype(np.uint8)
                diff_mask_output = Image.fromarray(diff_mask_output, 'L')
                diff_mask_output.save(save_path_diff_mask_depth)

            loss_mask_fitting = torch.mean(torch.abs(mask_F - refer_mask))
            loss_depth_fitting = 0
            loss_bias = torch.mean((theta_orig - theta) ** 2)

            loss_depth_fitting_weight = 1
            loss_mask_fitting_weight = 1e1
            loss_bias_weight = 0
            loss = loss_depth_fitting * loss_depth_fitting_weight + loss_mask_fitting * loss_mask_fitting_weight + loss_bias * loss_bias_weight
            pbar_mask.set_postfix(loss='TL: {0:.06f}, LF: {1:.06f}, LM: {2:.06f}, LB: {3:.06f}'.format(loss,
                                                                                                       loss_depth_fitting * loss_depth_fitting_weight,
                                                                                                       loss_mask_fitting * loss_mask_fitting_weight,
                                                                                                       loss_bias * loss_bias_weight))

            # 判断是否找到更好的解
            if loss < best_loss_fitting_first:
                best_i = i
                best_loss_fitting_first = loss
                best_theta_new = theta.clone()

            optimizer_smpl_mask.zero_grad()

            loss.backward()

            optimizer_smpl_mask.step()
            scheduler_smpl_mask.step(loss)

        for i in pbar_depth:
            # theta = theta_new

            theta = torch.cat([theta_new_limb[0], theta_new_limb[1], theta_orig[:, 9:12],
                               theta_new_limb[2], theta_orig[:, 18:39],
                               theta_new_limb[3], theta_orig[:, 45:48],
                               theta_new_limb[4], theta_orig[:, 60:]], dim=1)

            with torch.no_grad():
                transl_new.data[:, :, 2] = transl_orig[:, :, 2]
            transl_new_ = transl_new
            # transl_new_ = torch.cat([transl_new[:,:,0:2], transl_orig[:,:,2].unsqueeze(2)], dim=2)

            verts, faces = self.get_smpl_verts_face(pose=theta, shape=beta_new,
                                                    transl=transl_new_,
                                                    rotation_matrix=rotation_matrix,
                                                    refer_depth_map=refer_depth_map,
                                                    mask=refer_mask, calib=calib, scale=scale)

            self.render.load_meshes(verts, faces)

            smpl_depth_map = self.rendering_smpl_depth_map(verts=verts,
                                                           faces=faces,
                                                           refer_depth_map=refer_depth_map,
                                                           scale=scale)

            refer_depth_map_new = refer_depth_map.squeeze(0).squeeze(0)

            mask_F, mask_B = self.rendering_smpl_mask(verts=verts, faces=faces, scale=scale)

            normal_F, normal_B = self.rendering_smpl_normal(verts=verts, faces=faces, scale=scale)

            # if i % (iter_num / 10) == 0 or i == (iter_num - 1):
            #     save_path_diff_depth = '%s/%s/opt_smpl_diff_%s_2_%s_diff.png' % (
            #         opt.results_path, opt.name, name, i)
            #     # diff_depth_map = abs(smpl_depth_mask.cpu().detach().numpy() - refer_depth_mask.cpu().detach().numpy())
            #     diff_depth_map = abs(smpl_depth_map.cpu().detach().numpy() - refer_depth_map_new.cpu().detach().numpy())
            #     diff_depth_map = diff_depth_map + 255 / 8
            #     diff_depth_map = diff_depth_map.astype(np.uint8)
            #     diff_depth_map = Image.fromarray(diff_depth_map, 'L')
            #     diff_depth_map.save(save_path_diff_depth)
            if i % (iter_num / 10) == 0 or i == (iter_num - 1):
                save_path_diff_normal = '%s/%s/opt_smpl_diff_%s_2_%s_diff.png' % (
                    opt.results_path, opt.name, name, i)
                diff_normal_map = abs(normal_F.cpu().detach().numpy() - nmlF.cpu().detach().numpy())
                # diff_normal_map = normal_F.cpu().detach().numpy()
                # diff_normal_map = diff_normal_map + 255 / 8
                diff_normal_map = np.transpose(np.squeeze(diff_normal_map), (1, 2, 0)).astype(np.uint8)
                diff_normal_map = Image.fromarray(diff_normal_map)
                diff_normal_map.save(save_path_diff_normal)

            loss_mask_fitting = torch.mean(torch.abs(mask_F - refer_mask))
            # loss_mask_fitting = 0
            loss_depth_fitting = torch.abs(smpl_depth_map - refer_depth_map_new).mean()
            # loss_depth_fitting = 0
            loss_normal_fitting = torch.abs(normal_F - nmlF).mean()
            # loss_normal_fitting = 0
            loss_bias = torch.mean((theta_orig - theta) ** 2)

            # smpl_para = torch.cat((theta.flatten(), transl_new_.flatten()), dim=0).unsqueeze(0)
            # loss_smpl_discriminator = self.smpl_discriminator(smpl_para).squeeze(0).squeeze(0)
            loss_smpl_discriminator = 0

            loss_depth_fitting_weight = 1
            loss_mask_fitting_weight = 1
            loss_normal_fitting_weight = 1
            loss_bias_weight = 0
            loss_smpl_discriminator_weight = 1e2

            loss = (loss_depth_fitting * loss_depth_fitting_weight +
                    loss_mask_fitting * loss_mask_fitting_weight +
                    loss_normal_fitting * loss_normal_fitting_weight +
                    loss_bias * loss_bias_weight)
            pbar_depth.set_postfix(loss='TL: {0:.06f}, LF: {1:.06f}, LM: {2:.06f}, LN: {2:.06f}'.format(loss,
                                                                                                        loss_depth_fitting * loss_depth_fitting_weight,
                                                                                                        loss_mask_fitting * loss_mask_fitting_weight,
                                                                                                        loss_normal_fitting * loss_normal_fitting_weight))

            # loss = (loss_depth_fitting * loss_depth_fitting_weight +
            #         loss_mask_fitting * loss_mask_fitting_weight +
            #         loss_normal_fitting * loss_normal_fitting_weight +
            #         loss_bias * loss_bias_weight +
            #         loss_smpl_discriminator_weight * loss_smpl_discriminator)
            # pbar_depth.set_postfix(loss='TL: {0:.06f}, LF: {1:.06f}, LM: {2:.06f}, LN: {2:.06f}, LD: {3:.06f}'.format(loss,
            #                                                                                             loss_depth_fitting * loss_depth_fitting_weight,
            #                                                                                             loss_mask_fitting * loss_mask_fitting_weight,
            #                                                                                             loss_normal_fitting * loss_normal_fitting_weight,
            #                                                                                             loss_smpl_discriminator_weight * loss_smpl_discriminator))


            # 判断是否找到更好的解
            if loss < best_loss_fitting_second:
                best_i = i
                best_loss_fitting_second = loss
                best_theta_new = theta.clone()

            optimizer_smpl_depth.zero_grad()

            loss.backward()

            optimizer_smpl_depth.step()
            scheduler_smpl_depth.step(loss)

        print('BL: {0:.06f}, {1:d}'.format(best_loss_fitting_second, best_i))
        best_theta_new = best_theta_new.reshape(bs, 72)

        vert_tetsmpl_new_cam = self.tet_smpl(best_theta_new, beta_new)
        vert_tetsmpl_new_cam = self.tet_smpl.transl_respectively(vert_tetsmpl_new_cam, transl_new_)
        vert_tetsmpl_new_cam = torch.matmul(vert_tetsmpl_new_cam, rotation_matrix)

        best_theta_new = best_theta_new.reshape(bs, 24, 3)
        best_theta_new = torch.cat((best_theta_new, transl_new_), dim=1)

        return best_theta_new, beta_new, vert_tetsmpl_new_cam


    def optm_smpl_param_shrink(self, img, pose, shape, iter_num, calib=None, nmlF=None, nmlB=None, current_depth_map=None,
                               netG_output_map=None, human_parse_map=None,
                               mask_low_res_tensor=None, mask_high_res_tensor=None, smpl=None, save_path=None,
                               rotation_matrix=None, net=None, opt=None):
        assert iter_num > 0
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        # # convert rotmat to theta
        # rotmat_host = pose.detach().cpu().numpy().squeeze()
        # theta_host = []
        # for r in rotmat_host:
        #     theta_host.append(cv.Rodrigues(r)[0])
        # theta_host = np.asarray(theta_host).reshape((1, -1))
        # theta = torch.from_numpy(theta_host).to(self.device)

        batch_size = pose.size()[0]
        theta = pose.view(batch_size, 75)


        # construct parameters
        # vert_cam = self.tet_smpl(theta, shape)
        # vol = self.voxelization(vert_cam)
        theta_new = torch.nn.Parameter(theta)
        # shape[:, 6] = 9
        betas_new = torch.nn.Parameter(shape)
        theta_orig = theta_new.clone().detach()
        betas_orig = betas_new.clone().detach()


        optm = torch.optim.Adam(params=(theta_new, ), lr=2e-2)

        # vert_tetsmpl = self.tet_smpl(theta_orig, betas_orig)
        # vert_tetsmpl = self.tet_smpl.transl(vert_tetsmpl)
        # vert_tetsmpl = torch.matmul(vert_tetsmpl, rotation_matrix)

        # keypoint = self.smpl.get_joints(vert_tetsmpl[:, :6890])
        # kp_conf = keypoint[:, :, -1:].clone()
        # kp_detection = keypoint[:, :, :-1].clone()

        for i in tqdm(range(iter_num), desc='Body Fitting Optimization'):

            # betas_new_ = torch.cat([betas_new[:, :1], betas_orig[:, 1:2], betas_new[:, 2:4], betas_orig[:, 4:]], dim=1)


            # theta_new_ = torch.cat([theta_new[:, :3], theta_orig[:, 3:]], dim=1)
            # theta_new_ = theta_new
            if i < iter_num * (1 / 2):
                # theta_new_ = torch.cat([theta_new[:, :72], theta_orig[:, 72:75]], dim=1)
                theta_new_ = torch.cat([theta_new[:, :39], theta_orig[:, 39:45], theta_new[:, 45:48], theta_orig[:, 48:60], theta_new[:, 60:72], theta_orig[:, 72:75]], dim=1)
            elif i == int(iter_num * (1 / 2)):
                # theta_new_ = torch.cat([theta_new[:, :72], theta_orig[:, 72:75]], dim=1)
                theta_new_ = torch.cat([theta_new[:, :39], theta_orig[:, 39:45], theta_new[:, 45:48], theta_orig[:, 48:60], theta_new[:, 60:72], theta_orig[:, 72:75]], dim=1)
                theta_new_1_stage = theta_new_.clone().detach()
            else:
                theta_new_ = torch.cat([theta_new_1_stage[:, :39], theta_new[:, 39:45], theta_new_1_stage[:, 45:48], theta_new[:, 48:60], theta_new_1_stage[:, 60:]], dim=1)
            # theta_new_ = torch.cat([theta_new[:, :39], theta_orig[:, 39:45], theta_new[:, 45:48], theta_orig[:, 48:]], dim=1)

            only_pose = theta_new_[:, :72]
            transl = theta_new_[:, 72:75]
            vert_tetsmpl_new_cam = self.tet_smpl(only_pose, betas_new)
            vert_tetsmpl_new_cam = self.tet_smpl.transl_respectively(vert_tetsmpl_new_cam, transl)
            vert_tetsmpl_new_cam = torch.matmul(vert_tetsmpl_new_cam, rotation_matrix)

            # keypoint_new = self.smpl.get_joints(vert_tetsmpl_new_cam[:, :6890])
            # keypoint_new_proj = self.forward_point_sample_projection(
            #     keypoint_new, cam_r, cam_t, cam_f, cam_c)

            # if i % 10 == 0:
            #     vol = self.voxelization(vert_tetsmpl_new_cam.detach())

            pred_vert_new_cam = self.graph_mesh.downsample(vert_tetsmpl_new_cam[:, :6890], n2=1)
            # pred_vert_new_proj = self.forward_point_sample_projection(
            #     pred_vert_new_cam, cam_r, cam_t, cam_f, cam_c)


            theta_new_ = theta_new_.reshape(batch_size, 25, 3)
            pred_vert_new_cam = pred_vert_new_cam.permute(0, 2, 1)

            smpl_sdf = self.forward_infer_occupancy_value(net=net, img=img, point=pred_vert_new_cam, calib_tensor=calib,
                                                          nmlF=nmlF, nmlB=nmlB,
                                                          current_depth_map=current_depth_map,
                                                          netG_output_map=netG_output_map,
                                                          human_parse_map=human_parse_map,
                                                          mask_low_res_tensor=mask_low_res_tensor,
                                                          mask_high_res_tensor=mask_high_res_tensor,
                                                          smpl=vert_tetsmpl_new_cam,
                                                          smpl_pose=theta_new_, smpl_shape=betas_new,
                                                          save_path=save_path, rotation_matrix=rotation_matrix)


            if i % 10 == 0 or i == (iter_num-1):
                try:
                    # save visualization of model performance
                    save_path_ply = '%s/%s/opt_pts_%s_%s.ply' % (opt.results_path, opt.name, save_path[-8:-4], i)
                    r = smpl_sdf.cpu()  # get only the first example in the batch (i.e. 1 CAD model or subject). [1, Num of sampled points]
                    points = pred_vert_new_cam[0].transpose(0, 1).cpu()  # note that similar to res[0], we only take sample_tensor[0] i.e. the first CAD model. Shape of [Num of sampled points, 3] after the transpose.
                    self.save_samples_truncted_prob(save_path_ply, points.detach().numpy(), r.detach().numpy())
                except:
                    print("Unable to save point cloud.")

            if i < iter_num * (1 / 2):
                loss_fitting = torch.mean(torch.abs(F.relu(0.5 - smpl_sdf)))
                # loss_fitting = torch.mean(torch.abs(F.leaky_relu(0.5 - smpl_sdf, negative_slope=0.2)))
                loss_bias = torch.mean((theta_orig - theta_new) ** 2)
                loss = loss_fitting * 1.0 + loss_bias * 1e1
            else:
                loss_fitting = torch.mean(torch.abs(F.leaky_relu(0.5 - smpl_sdf, negative_slope=0.8)))
                # loss_fitting = torch.mean(torch.abs(0.5 - smpl_sdf))
                loss_bias = torch.mean((theta_orig - theta_new) ** 2)
                loss = loss_fitting * 1.0 + loss_bias * 5e1
            # loss_kp = torch.mean((kp_conf * keypoint_new_proj - kp_conf * kp_detection) ** 2)
            # loss_bias2 = torch.mean((keypoint[:, :, 2] - keypoint_new[:, :, 2]) ** 2)

            # loss = loss_fitting * 1.0 + loss_bias * 1.0 + loss_kp * 500.0 + loss_bias2 * 1

            optm.zero_grad()
            loss.backward()
            optm.step()

            # print('Iter No.%d: loss = %f, loss_fitting = %f, loss_bias = %f, loss_kp = %f, loss_bias2 = %f' %
            #       (i, loss.item(), loss_fitting.item(), loss_bias.item(), loss_kp.item(), loss_bias2.item()))
            # print('Iter No.%d : loss = %.5f, loss_fitting = %.5f, loss_bias = %.5f' %
            #       (i, loss.item(), loss_fitting.item(), loss_bias.item()))

        theta_new = theta_new_.reshape(batch_size, 25, 3)
        return theta_new, betas_new, vert_tetsmpl_new_cam[:, :6890]

    def save_samples_truncted_prob(self, fname, points, prob):
        '''
        Save the visualization of sampling to a ply file.
        Red points represent positive predictions.
        Green points represent negative predictions.
        :param fname: File name to save
        :param points: [N, 3] array of points
        :param prob: [N, 1] array of predictions in the range [0~1]
        :return:
        '''
        r = (prob >= 0.5).reshape([-1, 1]) * 255
        g = (prob < 0.5).reshape([-1, 1]) * 255
        b = np.zeros(r.shape)

        to_save = np.concatenate([points, r, g, b], axis=-1)
        return np.savetxt(fname,
                          to_save,
                          fmt='%.6f %.6f %.6f %d %d %d',
                          comments='',
                          header=(
                              'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                              points.shape[0])
                          )

    def get_smpl_verts_face(self, pose, shape, transl, rotation_matrix, refer_depth_map, mask, calib, scale):

        vert_tetsmpl_new_cam = self.tet_smpl(pose, shape)
        # vert_tetsmpl_new_cam = self.tet_smpl.rot_vert(vert_tetsmpl_new_cam, transl.squeeze(0))
        vert_tetsmpl_new_cam = self.tet_smpl.transl_respectively(vert_tetsmpl_new_cam, transl.squeeze(0))

        verts = vert_tetsmpl_new_cam[0]
        rotation_inv = np.linalg.inv(rotation_matrix.cpu().detach().numpy())
        rotation_matrix = torch.from_numpy(rotation_inv).to(self.device)
        verts = torch.matmul(verts, rotation_matrix.t())
        calib = calib.squeeze(0)
        verts = projection(verts, calib)
        verts[:, 1] *= -1
        verts = verts.unsqueeze(0)

        faces = self.smpl_faces
        bs = verts.size()[0]
        faces = np.expand_dims(faces, axis=0).repeat(bs, axis=0)
        faces = torch.from_numpy(faces).to(self.device)

        return verts, faces

    def rendering_smpl_mask(self, verts, faces, scale):

        # self.render.load_meshes(verts, faces)
        F_mask, B_mask = self.render.get_image(type="mask", scale=scale)
        F_mask = F_mask.squeeze(0) * 100
        # F_mask = self.adjust_map_to_mask(target_image=F_mask, mask_image=mask)

        return F_mask, B_mask

    def rendering_smpl_normal(self, verts, faces, scale):

        # self.render.load_meshes(verts, faces)
        F_normal, B_normal = self.render.get_image(type="rgb", scale=scale)
        F_normal = (F_normal.squeeze(0) + 1.0) * 255.0 / 2.0
        # F_normal = F_normal[[2, 1, 0], :, :]

        return F_normal, B_normal

    def rendering_smpl_depth_map(self, verts, faces, refer_depth_map, scale):

        # render optimized mesh (normal, T_normal, image [-1,1])
        # self.render.load_meshes(verts, faces)
        depth_F_image, depth_B_image = self.render.get_image(type="depth", scale=scale)

        mask_depth = depth_F_image == -1
        camera_position = 100.0
        depth_F_image = depth_F_image - camera_position

        modified_depth_F_image = depth_F_image.clone()
        modified_depth_F_image[modified_depth_F_image < -1] = float('inf')

        min_value = torch.min(modified_depth_F_image)
        max_value = torch.max(depth_F_image)

        depth_F_image = (depth_F_image - min_value) / (max_value - min_value)

        depth_F_image[mask_depth] = 0

        depth_F_image = (depth_F_image[0, :, :]) * 255.0 / 2
        depth_F_image += 1

        depth_F_image[mask_depth[0, :, :]] = 0

        # smpl_depth_map = self.adjust_map_to_mask(target_image=depth_F_image, mask_image=mask)
        smpl_depth_map = depth_F_image

        # 将smpl depth图和预测出的depth map数值范围统一
        smpl_depth_map /= 2

        non_zero_refer_depth_map = refer_depth_map[refer_depth_map != 0]
        non_zero_smpl_depth_map = smpl_depth_map[smpl_depth_map != 0]
        refer_mean = non_zero_refer_depth_map.mean()
        smpl_mean = non_zero_smpl_depth_map.mean()
        mean_diff = smpl_mean - refer_mean
        smpl_depth_map[smpl_depth_map >= 0.2] -= mean_diff

        smpl_depth_map[smpl_depth_map < 0] = 0
        smpl_depth_map[smpl_depth_map > 255] = 255

        return smpl_depth_map



    def adjust_map_to_mask(self, target_image, mask_image):
        """使用有效像素边界框，调整深度图像，使其和掩码最大程度上吻合"""

        smpl_depth_map_np = target_image.squeeze().detach().cpu().numpy()
        mask_image_np = mask_image.squeeze().detach().cpu().numpy()

        smpl_depth_map_np = np.uint8(smpl_depth_map_np)
        mask_image_np = np.uint8(mask_image_np)
        # 进行连通组件分析
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask_image_np)
        # 找到具有最大面积的组件
        max_area_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # 忽略背景组件
        # 创建只包含最大连通域的掩码
        mask_image_np = np.uint8(labels == max_area_label) * 255

        _, smpl_depth_map_binary = cv2.threshold(smpl_depth_map_np, 0, 255, cv2.THRESH_BINARY)
        _, mask_image_binary = cv2.threshold(mask_image_np, 0, 255, cv2.THRESH_BINARY)

        # 计算深度图像的有效像素边界框
        depth_bbox = cv2.boundingRect(smpl_depth_map_binary)
        # 计算图像掩码的有效像素边界框
        mask_bbox = cv2.boundingRect(mask_image_binary)

        # 缩放深度图像使其宽度与 bbox 一致
        scale_ratio = mask_bbox[3] / depth_bbox[3]  # 计算缩放比例
        smpl_depth_image_scaled = cv2.resize(smpl_depth_map_np, None, fx=scale_ratio, fy=1)

        smpl_depth_image_adjusted = np.zeros_like(mask_image_np)

        # 确定需要裁剪的区域
        crop_width = min(smpl_depth_image_scaled.shape[1], smpl_depth_image_adjusted.shape[1])
        crop_height = min(smpl_depth_image_scaled.shape[0], smpl_depth_image_adjusted.shape[0])

        smpl_depth_image_adjusted[:crop_height, :crop_width] = smpl_depth_image_scaled[:crop_height, :crop_width]

        # 计算缩放后深度图像的有效像素边界框
        _, smpl_depth_image_adjusted_binary = cv2.threshold(smpl_depth_image_adjusted, 0, 255, cv2.THRESH_BINARY)
        depth_bbox_adjusted = cv2.boundingRect(smpl_depth_image_adjusted_binary)

        # 计算平移的偏移量
        # dx = int((mask_bbox[0] + mask_bbox[2] / 2) - (depth_bbox_adjusted[0] + depth_bbox_adjusted[2] / 2))
        # dy = int((mask_bbox[1] + mask_bbox[3] / 2) - (depth_bbox_adjusted[1] + depth_bbox_adjusted[3] / 2))
        dx = mask_bbox[0] - depth_bbox_adjusted[0]
        dy = mask_bbox[1] - depth_bbox_adjusted[1]


        target_image = target_image.squeeze()

        # 创建一个与scaled_depth_map相同形状的全零张量
        smpl_depth_image_adjusted = torch.zeros_like(target_image)

        new_size = (int(target_image.shape[1] * scale_ratio), int(target_image.shape[0] * scale_ratio))
        scaled_depth_map = torch.nn.functional.interpolate(target_image.unsqueeze(0).unsqueeze(0), size=new_size,
                                                           mode='bilinear', align_corners=False).squeeze().to(target_image.device).float()

        # 确定需要裁剪的区域
        crop_width = min(scaled_depth_map.shape[1], smpl_depth_image_adjusted.shape[1])
        crop_height = min(scaled_depth_map.shape[0], smpl_depth_image_adjusted.shape[0])

        # 将裁剪后的scaled_depth_map赋值给smpl_depth_image_adjusted
        smpl_depth_image_adjusted[:crop_height, :crop_width] = scaled_depth_map[:crop_height, :crop_width]

        # 平移
        smpl_depth_image_adjusted = torch.roll(smpl_depth_image_adjusted, shifts=dx, dims=1)
        smpl_depth_image_adjusted = torch.roll(smpl_depth_image_adjusted, shifts=dy, dims=0)

        return smpl_depth_image_adjusted

    # def generate_point_grids(self, vol_res, cam_R, cam_t, cam_f, img_res):
    #     x_coords = np.array(range(0, vol_res), dtype=np.float32)
    #     y_coords = np.array(range(0, vol_res), dtype=np.float32)
    #     z_coords = np.array(range(0, vol_res), dtype=np.float32)
    #
    #     yv, xv, zv = np.meshgrid(x_coords, y_coords, z_coords)
    #     xv = np.reshape(xv, (-1, 1))
    #     yv = np.reshape(yv, (-1, 1))
    #     zv = np.reshape(zv, (-1, 1))
    #     xv = xv / vol_res - 0.5 + 0.5 / vol_res
    #     yv = yv / vol_res - 0.5 + 0.5 / vol_res
    #     zv = zv / vol_res - 0.5 + 0.5 / vol_res
    #     pts = np.concatenate([xv, yv, zv], axis=-1)
    #     pts = np.float32(pts)
    #     pts_proj = np.dot(pts, cam_R.transpose()) + cam_t
    #     pts_proj[:, 0] = pts_proj[:, 0] * cam_f / pts_proj[:, 2] / (img_res / 2)
    #     pts_proj[:, 1] = pts_proj[:, 1] * cam_f / pts_proj[:, 2] / (img_res / 2)
    #     pts_proj = pts_proj[:, :2]
    #
    #     return pts, pts_proj

    # def forward_keypoint_projection(self, smpl_vert, cam):
    #     pred_keypoints = self.smpl.get_joints(smpl_vert)
    #     pred_keypoints_2d = orthographic_projection(pred_keypoints, cam)
    #     return pred_keypoints_2d

    # def forward_coordinate_conversion(self, cam_f, cam_tz, cam_c, pred_cam):
    #     # calculates camera parameters
    #     with torch.no_grad():
    #         scale = pred_cam[:, 0:1] * cam_c * cam_tz / cam_f
    #         trans_x = pred_cam[:, 1:2] * cam_c * cam_tz * pred_cam[:, 0:1] / cam_f
    #         trans_y = -pred_cam[:, 2:3] * cam_c * cam_tz * pred_cam[:, 0:1] / cam_f
    #         trans_z = torch.zeros_like(trans_x)
    #         scale_ = torch.cat([scale, -scale, -scale], dim=-1).detach().view((-1, 1, 3))
    #         trans_ = torch.cat([trans_x, trans_y, trans_z], dim=-1).detach().view((-1, 1, 3))
    #
    #     return scale_, trans_

    def forward_point_sample_projection(self, points, cam_r, cam_t, cam_f, cam_c):
        points_proj = points * cam_r.view((1, 1, -1)) + cam_t.view((1, 1, -1))
        points_proj = points_proj * (cam_f / cam_c) / points_proj[:, :, 2:3]
        points_proj = points_proj[:, :, :2]
        return points_proj

    # def forward_infer_occupancy_value_grid_naive(self, img, vol, test_res, group_size):
    #     pts, pts_proj = self.generate_point_grids(
    #         test_res, const.cam_R, const.cam_t, const.cam_f, img.size(2))
    #     pts_ov = self.forward_infer_occupancy_value_group(img, vol, pts, pts_proj, group_size)
    #     pts_ov = pts_ov.reshape([test_res, test_res, test_res])
    #     return pts_ov
    #
    # def forward_infer_occupancy_value_grid_octree(self, img, vol, test_res, group_size,
    #                                               init_res=64, ignore_thres=0.05):
    #     pts, pts_proj = self.generate_point_grids(
    #         test_res, const.cam_R, const.cam_t, const.cam_f, img.size(2))
    #     pts = np.reshape(pts, (test_res, test_res, test_res, 3))
    #     pts_proj = np.reshape(pts_proj, (test_res, test_res, test_res, 2))
    #
    #     pts_ov = np.zeros([test_res, test_res, test_res])
    #     dirty = np.ones_like(pts_ov, dtype=np.bool)
    #     grid_mask = np.zeros_like(pts_ov, dtype=np.bool)
    #
    #     reso = test_res // init_res
    #     while reso > 0:
    #         grid_mask[0:test_res:reso, 0:test_res:reso, 0:test_res:reso] = True
    #         test_mask = np.logical_and(grid_mask, dirty)
    #
    #         pts_ = pts[test_mask]
    #         pts_proj_ = pts_proj[test_mask]
    #         pts_ov[test_mask] = self.forward_infer_occupancy_value_group(
    #             img, vol, pts_, pts_proj_, group_size).squeeze()
    #
    #         if reso <= 1:
    #             break
    #         for x in range(0, test_res - reso, reso):
    #             for y in range(0, test_res - reso, reso):
    #                 for z in range(0, test_res - reso, reso):
    #                     # if center marked, return
    #                     if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
    #                         continue
    #                     v0 = pts_ov[x, y, z]
    #                     v1 = pts_ov[x, y, z + reso]
    #                     v2 = pts_ov[x, y + reso, z]
    #                     v3 = pts_ov[x, y + reso, z + reso]
    #                     v4 = pts_ov[x + reso, y, z]
    #                     v5 = pts_ov[x + reso, y, z + reso]
    #                     v6 = pts_ov[x + reso, y + reso, z]
    #                     v7 = pts_ov[x + reso, y + reso, z + reso]
    #                     v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
    #                     v_min = v.min()
    #                     v_max = v.max()
    #                     # this cell is all the same
    #                     if (v_max - v_min) < ignore_thres:
    #                         pts_ov[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
    #                         dirty[x:x + reso, y:y + reso, z:z + reso] = False
    #         reso //= 2
    #     return pts_ov

    # def forward_infer_occupancy_value_group(self, img, vol, pts, pts_proj, group_size):
    #     assert isinstance(pts, np.ndarray)
    #     assert len(pts.shape) == 2
    #     assert pts.shape[1] == 3
    #     pts_num = pts.shape[0]
    #     pts = torch.from_numpy(pts).unsqueeze(0).to(self.device)
    #     pts_proj = torch.from_numpy(pts_proj).unsqueeze(0).to(self.device)
    #     pts_group_num = (pts.size()[1] + group_size - 1) // group_size
    #     pts_ov = []
    #     for gi in tqdm(range(pts_group_num), desc='SDF query'):
    #         # print('Testing point group: %d/%d' % (gi + 1, pts_group_num))
    #         pts_group = pts[:, (gi * group_size):((gi + 1) * group_size), :]
    #         pts_proj_group = pts_proj[:, (gi * group_size):((gi + 1) * group_size), :]
    #         outputs = self.forward_infer_occupancy_value(
    #             img, pts_group, pts_proj_group, vol)
    #         pts_ov.append(np.squeeze(outputs.detach().cpu().numpy()))
    #     pts_ov = np.concatenate(pts_ov)
    #     pts_ov = np.array(pts_ov)
    #     return pts_ov


    def forward_infer_occupancy_value(self, net, img, point, calib_tensor, nmlF=None, nmlB=None, current_depth_map=None,
                                      netG_output_map=None, human_parse_map=None,
                                      mask_low_res_tensor=None, mask_high_res_tensor=None, smpl=None, smpl_pose=None,
                                      smpl_shape=None, save_path=None, rotation_matrix=None):
        net.filter(images=img, nmlF=nmlF, nmlB=nmlB, current_depth_map=current_depth_map, netG_output_map=netG_output_map,
                   human_parse_map=human_parse_map, mask_low_res_tensor=mask_low_res_tensor,
                   mask_high_res_tensor=mask_high_res_tensor, smpl=smpl, smpl_pose=smpl_pose, smpl_shape=smpl_shape,
                   save_path=save_path, rotation_matrix=rotation_matrix, smpl_test_opt=False, smpl_para_rect=False)  # forward-pass
        net.query(point, calib_tensor)
        return net.get_preds()[0][0]
