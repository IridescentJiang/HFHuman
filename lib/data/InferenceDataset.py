import os
import random
import time

import numpy as np
from PIL import Image, ImageOps
import cv2
import torch
import json
import trimesh
import logging
import pickle
from pysdf import SDF
from scipy import ndimage
import scipy.io as sio
import concurrent.futures
import pickle as pkl

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from numpy.linalg import inv

log = logging.getLogger('trimesh')
log.setLevel(40)


def load_trimesh(root_dir, training_subject_list=None):
    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        if f == ".DS_Store":
            continue

        if f not in training_subject_list:  # only load meshes that are in the training set
            continue

        meshs[f] = trimesh.load(os.path.join(root_dir, f, '%s.obj' % f))

    return meshs


class InferenceDataset(Dataset):

    def __init__(self, opt, projection='orthogonal', phase='train', evaluation_mode=False, validation_mode=False):
        self.opt = opt
        self.projection_mode = projection

        if self.opt.debug_mode:
            self.training_subject_list = np.loadtxt("/data/jym/IntegratedPIFu/IntegratePIFu/train_set_list.txt",
                                                    dtype=str)
        else:
            self.training_subject_list = np.loadtxt("train_set_list.txt", dtype=str)

        self.evaluation_mode = evaluation_mode

        self.validation_mode = validation_mode

        self.phase = phase
        self.is_train = (self.phase == 'train')

        if self.opt.useValidationSet:

            indices = np.arange(len(self.training_subject_list))
            np.random.seed(10)
            np.random.shuffle(indices)
            lower_split_index = round(len(self.training_subject_list) * 0.1)
            val_indices = indices[:lower_split_index]
            train_indices = indices[lower_split_index:]

            if self.validation_mode:
                self.training_subject_list = self.training_subject_list[val_indices]
                self.is_train = False
            else:
                self.training_subject_list = self.training_subject_list[train_indices]

        self.training_subject_list = self.training_subject_list.tolist()

        if evaluation_mode:
            print("Overwriting self.training_subject_list!")
            if self.opt.debug_mode:
                self.training_subject_list = np.loadtxt("/data/jym/IntegratedPIFu/IntegratePIFu/test_set_list.txt",
                                                        dtype=str).tolist()
            else:
                self.training_subject_list = np.loadtxt("test_set_list.txt", dtype=str).tolist()
            self.is_train = False

        self.root = "rendering_script/inference/buffer_fixed_full_mesh"

        self.mesh_directory = "rendering_script/THuman2.0_Release"

        self.normal_directory_high_res = "rendering_script/inference/trained_normal_maps"

        self.depth_map_directory = "rendering_script/inference/trained_depth_maps_refer"  # New version (Depth maps trained with only normal - Second Stage maps)

        if self.opt.use_SMPL_depth_map:
            self.SMPL_depth_map_directory = "rendering_script/smpl_depth"
            # self.depth_map_directory = "trained_depth_maps"
            self.depth_map_directory = "trained_coarse_depth_maps"

        self.human_parse_map_directory = "trained_parse_maps"

        if self.opt.use_smpl_x:
            self.smpl_directory = "rendering_script/smpl_x"
        else:
            self.smpl_directory = "rendering_script/inference/smpl_pred"

        if self.opt.use_smpl_para_opt:
            self.smpl_para_opt = "rendering_script/inference/smpl_para_opt"
        else:
            self.smpl_para_pred = "rendering_script/inference/smpl_para_pred"

        self.subjects = self.training_subject_list

        self.load_size = self.opt.loadSize

        self.num_sample_inout = self.opt.num_sample_inout

        if self.opt.use_sampled_points_smpl:
            self.point_output_fd = 'rendering_script/sample_points_smpl_new'
        else:
            self.point_output_fd = 'rendering_script/sample_points'

        self.img_files = []
        subject_render_folder = os.path.join(self.root, "in-the-wild")
        subject_render_paths_list = [os.path.join(subject_render_folder, f) for f in os.listdir(subject_render_folder)
                                     if "image" in f]
        self.img_files = self.img_files + subject_render_paths_list

        self.img_files = sorted(self.img_files)
        self.img_files = np.array(self.img_files, dtype=np.string_)

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # ToTensor converts input to a shape of (C x H x W) in the range [0.0, 1.0] for each dimension
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # normalise with mean of 0.5 and std_dev of 0.5 for each dimension. Finally range will be [-1,1] for each dimension
        ])

        self.to_tensor_depth = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_files)

    def get_item(self, index):

        img_path = self.img_files[index].decode()
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # get yaw
        yaw = img_name.split("_")[-1]
        yaw = int(yaw)

        # get subject
        subject = img_path.split('/')[-2]  # e.g. "0507"

        param_path = os.path.join(self.root, subject, "rendered_params_" + "{0:03d}".format(yaw) + ".npy")
        render_path = os.path.join(self.root, subject, "rendered_image_" + "{0:03d}".format(yaw) + ".png")
        mask_path = os.path.join(self.root, subject, "rendered_mask_" + "{0:03d}".format(yaw) + ".png")

        nmlF_high_res_path = os.path.join(self.normal_directory_high_res, subject,
                                          "rendered_nmlF_" + "{0:03d}".format(yaw) + ".npy")
        nmlB_high_res_path = os.path.join(self.normal_directory_high_res, subject,
                                          "rendered_nmlB_" + "{0:03d}".format(yaw) + ".npy")

        if self.opt.smpl_test_opt:
            nmlF_for_smpl_opt_high_res_path = os.path.join(self.normal_directory_high_res, subject,
                                                           "rendered_nmlF_" + "{0:03d}".format(yaw) + ".png")

        depth_map_path = os.path.join(self.depth_map_directory, subject,
                                      "rendered_depthmap_" + "{0:03d}".format(yaw) + ".npy")
        depth_map_L_path = os.path.join(self.depth_map_directory, subject,
                                        "rendered_depthmap_" + "{0:03d}".format(yaw) + ".png")

        if self.opt.use_SMPL_depth_map:
            # use smpl depth map as back and use the picture's depth map as front
            # depth_map_path_front = os.path.join(self.SMPL_depth_map_directory, subject,
            #                               "rendered_depthmap_" + "{0:03d}".format(yaw) + "_F" + ".exr")
            depth_map_path_front = os.path.join(self.depth_map_directory, subject,
                                                "rendered_depthmap_" + "{0:03d}".format(yaw) + ".npy")
            depth_map_path_back = os.path.join(self.SMPL_depth_map_directory, subject,
                                               "rendered_depthmap_" + "{0:03d}".format(yaw) + "_B" + ".exr")

        human_parse_map_path = os.path.join(self.human_parse_map_directory, subject,
                                            "rendered_parse_" + "{0:03d}".format(yaw) + ".npy")

        load_size_associated_with_scale_factor = 1024

        center = torch.tensor([0, -0.1, 0.25], dtype=torch.float32)
        R = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        scale_factor = 570

        b_range = load_size_associated_with_scale_factor / scale_factor  # e.g. 512/scale_factor
        b_center = center
        b_min = b_center - b_range / 2
        b_max = b_center + b_range / 2

        # extrinsic is used to rotate the 3D points according to our specified pitch and yaw
        translate = -center.reshape(3, 1)
        extrinsic = np.concatenate([R, translate], axis=1)
        extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

        R = torch.Tensor(R).float()

        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = 1.0 * scale_factor
        scale_intrinsic[1, 1] = -1.0 * scale_factor
        scale_intrinsic[2, 2] = 1.0 * scale_factor

        # Match image pixel space to image uv space
        uv_intrinsic = np.identity(4)
        uv_intrinsic[0, 0] = 1.0 / float(load_size_associated_with_scale_factor // 2)
        uv_intrinsic[1, 1] = 1.0 / float(load_size_associated_with_scale_factor // 2)
        uv_intrinsic[2, 2] = 1.0 / float(load_size_associated_with_scale_factor // 2)

        mask = Image.open(mask_path).convert('L')
        render = Image.open(render_path).convert('RGB')

        intrinsic = np.matmul(uv_intrinsic, scale_intrinsic)
        calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
        extrinsic = torch.Tensor(extrinsic).float()

        mask = transforms.ToTensor()(mask).float()

        render = self.to_tensor(render)  # normalize render
        render = mask.expand_as(render) * render

        render_low_pifu = F.interpolate(torch.unsqueeze(render, 0),
                                        size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
        mask_low_pifu = F.interpolate(torch.unsqueeze(mask, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
        render_low_pifu = render_low_pifu[0]
        mask_low_pifu = mask_low_pifu[0]

        nmlF = []
        nmlB = []
        nmlF_high_res = []
        nmlB_high_res = []
        nmlF_for_smpl_opt_high_res = []
        nml_for_opt = []

        if self.opt.use_front_normal or self.opt.use_back_normal:
            if self.opt.use_groundtruth_normal_maps:
                nmlF_high_res = cv2.imread(nmlF_high_res_path, cv2.IMREAD_UNCHANGED).astype(
                    np.float32)  # numpy of [1024,1024,3]
                nmlB_high_res = cv2.imread(nmlB_high_res_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                nmlB_high_res = nmlB_high_res[:, ::-1, :].copy()
                nmlF_high_res = np.transpose(nmlF_high_res, [2, 0, 1])  # change to shape of [3,1024,1024]
                nmlB_high_res = np.transpose(nmlB_high_res, [2, 0, 1])
            else:
                if self.opt.use_front_normal:
                    nmlF_high_res = np.load(nmlF_high_res_path)  # shape of [3, 1024,1024]
                    nmlF_high_res = torch.Tensor(nmlF_high_res)
                    nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res
                    nmlF = F.interpolate(torch.unsqueeze(nmlF_high_res, 0),
                                         size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
                    nmlF = nmlF[0]
                if self.opt.use_back_normal:
                    nmlB_high_res = np.load(nmlB_high_res_path)  # shape of [3, 1024,1024]
                    nmlB_high_res = torch.Tensor(nmlB_high_res)
                    nmlB_high_res = mask.expand_as(nmlB_high_res) * nmlB_high_res
                    nmlB = F.interpolate(torch.unsqueeze(nmlB_high_res, 0),
                                         size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
                    nmlB = nmlB[0]

        if self.opt.smpl_test_opt:
            nmlF_for_smpl_opt_high_res = cv2.imread(nmlF_for_smpl_opt_high_res_path)
            nmlF_for_smpl_opt_high_res = torch.from_numpy(nmlF_for_smpl_opt_high_res).permute(2, 0, 1).unsqueeze(0)
            nml_for_opt = F.interpolate(nmlF_for_smpl_opt_high_res,
                                        size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
            nml_for_opt = nml_for_opt[0]

        depthmap_for_smpl_rect = []
        depth_map_for_opt = []
        mask_for_opt = []

        if self.opt.use_depth_map or self.opt.smpl_para_depth_guidance_learning:

            depth_map = np.load(depth_map_path)
            depth_map = torch.Tensor(depth_map)

            depth_map = mask.expand_as(depth_map) * depth_map  # shape of [C,H,W]

            if self.opt.depth_in_front:
                depth_map_low_res = F.interpolate(torch.unsqueeze(depth_map, 0),
                                                  size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
                depth_map_low_res = depth_map_low_res[0]
            else:
                depth_map_low_res = 0

            if self.opt.smpl_para_depth_guidance_learning:
                depthmap_for_smpl_rect = F.interpolate(torch.unsqueeze(depth_map, 0),
                                                       size=(self.opt.loadSizeDepth, self.opt.loadSizeDepth))
                depthmap_for_smpl_rect = depthmap_for_smpl_rect[0]

            depth_map_L = cv2.imread(depth_map_L_path, cv2.IMREAD_GRAYSCALE)
            depth_map_L = torch.from_numpy(depth_map_L)
            depth_map_L = depth_map_L.unsqueeze(0)
            depth_map_L = F.interpolate(torch.unsqueeze(depth_map_L, 0),
                                        size=(self.opt.loadSizeDepthOpt, self.opt.loadSizeDepthOpt))
            depth_map_for_opt = depth_map_L[0]
            mask_for_opt = torch.where(depth_map_for_opt > 0.5, 1., 0.)

            depth_map_F = depth_map
            depth_map_F_low_res = depth_map_low_res
            depth_map_B = 0
            depth_map_B_low_res = 0


        elif self.opt.use_SMPL_depth_map:
            depth_map_F = cv2.imread(depth_map_path_front, cv2.IMREAD_UNCHANGED).astype(np.float32)
            # depth_map_back = cv2.imread(depth_map_path_back, cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth_map_F = depth_map_F[:, :, 0]
            mask_depth = depth_map_F > 110
            camera_position = 100.0
            depth_map_F = depth_map_F - camera_position  # make the center pixel to have a depth value of 0.0
            depth_map_F = depth_map_F / (
                    b_range / self.opt.resolution)  # converts the units into in terms of no. of bounding cubes
            depth_map_F = depth_map_F / (self.opt.resolution / 2)  # normalize into range of [-1,1]
            depth_map_F = depth_map_F + 1.0  # convert into range of [0,2.0] where the center pixel has value of 1.0
            depth_map_F[mask_depth] = 0  # the invalid values are set to 0.
            expanded_size = (1024, 1024)
            scale_factor = (expanded_size[0] / depth_map_F.shape[0], expanded_size[1] / depth_map_F.shape[1])
            depth_map_F = ndimage.zoom(depth_map_F, scale_factor, order=1)  # expand depth map to 1024*1024
            depth_map_F = np.expand_dims(depth_map_F, 0)  # shape of [1,1024,1024]
            depth_map_F = torch.Tensor(depth_map_F)
            depth_map_F = mask.expand_as(depth_map_F) * depth_map_F

            depth_map_B = cv2.imread(depth_map_path_back, cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth_map_B = depth_map_B[:, :, 0]
            mask_depth = depth_map_B > 110
            camera_position = 100.0
            depth_map_B = depth_map_B - camera_position  # make the center pixel to have a depth value of 0.0
            depth_map_B = depth_map_B / (
                    b_range / self.opt.resolution)  # converts the units into in terms of no. of bounding cubes
            depth_map_B = depth_map_B / (self.opt.resolution / 2)  # normalize into range of [-1,1]
            depth_map_B = depth_map_B + 1.0  # convert into range of [0,2.0] where the center pixel has value of 1.0
            depth_map_B[mask_depth] = 0  # the invalid values are set to 0.
            expanded_size = (1024, 1024)
            scale_factor = (expanded_size[0] / depth_map_B.shape[0], expanded_size[1] / depth_map_B.shape[1])
            depth_map_B = ndimage.zoom(depth_map_B, scale_factor, order=1)  # expand depth map to 1024*1024
            depth_map_B = np.expand_dims(depth_map_B, 0)  # shape of [1,1024,1024]
            depth_map_B = torch.Tensor(depth_map_B)
            depth_map_B = mask.expand_as(depth_map_B) * depth_map_B

            # downsample depth_map
            if self.opt.depth_in_front:
                depth_map_F_low_res = F.interpolate(torch.unsqueeze(depth_map_F, 0),
                                                    size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
                depth_map_F_low_res = depth_map_F_low_res[0]

                depth_map_B_low_res = F.interpolate(torch.unsqueeze(depth_map_B, 0),
                                                    size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
                depth_map_B_low_res = depth_map_B_low_res[0]
            else:
                depth_map_F_low_res = 0
                depth_map_B_low_res = 0

        else:
            depth_map_F = 0
            depth_map_B = 0
            depth_map_F_low_res = 0
            depth_map_B_low_res = 0

        if self.opt.use_human_parse_maps:
            human_parse_map = np.load(human_parse_map_path)  # shape of (1024,1024)
            human_parse_map = torch.Tensor(human_parse_map)
            human_parse_map = torch.unsqueeze(human_parse_map, 0)  # shape of (1,1024,1024)
            human_parse_map = mask.expand_as(human_parse_map) * human_parse_map  # shape of [1,H,W]

            human_parse_map_0 = (human_parse_map == 0).float()
            human_parse_map_1 = (human_parse_map == 1).float()
            human_parse_map_2 = (human_parse_map == 2).float()
            human_parse_map_3 = (human_parse_map == 3).float()
            human_parse_map_4 = (human_parse_map == 4).float()
            human_parse_map_5 = (human_parse_map == 5).float()
            human_parse_map_6 = (human_parse_map == 6).float()
            human_parse_map_list = [human_parse_map_0, human_parse_map_1, human_parse_map_2, human_parse_map_3,
                                    human_parse_map_4, human_parse_map_5, human_parse_map_6]

            human_parse_map = torch.cat(human_parse_map_list, dim=0)

            human_parse_map = F.interpolate(torch.unsqueeze(human_parse_map, 0),
                                            size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
            human_parse_map = human_parse_map[0]
        else:
            human_parse_map = 0

        if self.opt.use_smpl_x:
            smpl_npy_path = os.path.join(self.smpl_directory, subject, "smpl_" + "{0:03d}".format(yaw) + ".npy")
            smpl_npy = np.load(smpl_npy_path, allow_pickle=True)
            smpl_scale = smpl_npy.item().get('scale')
            transl = smpl_npy.item().get('transl')
            smpl_mesh_path = os.path.join(self.smpl_directory, subject, "smpl_" + "{0:03d}".format(yaw) + ".obj")
            smpl_mesh = trimesh.load(smpl_mesh_path)
            smpl_verts = torch.tensor(smpl_mesh.vertices).float()
            # smpl_faces = torch.tensor(smpl_mesh.faces).long()
            # smpl_verts = smpl_verts / smpl_scale[0]  # normalizing the scale
            smpl_verts = smpl_verts - transl[0] / 2
            smpl_verts = smpl_verts * 0.5  # corrects coordinates for torch in-network sampling
        elif self.opt.use_smpl:
            # smpl_pkl_path = os.path.join(self.smpl_directory, subject, "smpl_" + "{0:03d}".format(yaw) + ".pkl")
            # with open(smpl_pkl_path, 'rb') as fp:
            #     data = pkl.load(fp)
            #     smpl_scale = np.float32(data['body_scale'])
            smpl_mesh_path = os.path.join(self.smpl_directory, subject, "smpl_" + "{0:03d}".format(yaw) + ".obj")
            smpl_mesh = trimesh.load(smpl_mesh_path)
            smpl_verts = torch.tensor(smpl_mesh.vertices).float()
            # smpl_faces = torch.tensor(smpl_mesh.faces).long()
            # smpl_verts = smpl_verts / smpl_scale[0]  # normalizing the scale
            smpl_verts = smpl_verts * 0.5  # corrects coordinates for torch in-network sampling
        else:
            smpl_verts = 0

        smpl_shapes_pred = []
        smpl_poses_pred = []

        if self.opt.use_smpl_para or self.opt.smpl_para_depth_guidance_learning:
            if self.opt.use_smpl_para_opt:
                smpl_para_pred_path = os.path.join(self.smpl_para_opt, subject,
                                                   "smpl_" + "{0:03d}".format(yaw) + ".json")
            else:
                smpl_para_pred_path = os.path.join(self.smpl_para_pred, subject,
                                                   "smpl_" + "{0:03d}".format(yaw) + ".json")

            if os.path.exists(smpl_para_pred_path):
                with open(smpl_para_pred_path, 'r') as f:
                    data_smpl_pred = json.load(f)

                # 获取 "betas" 和 "poses" 矩阵
                smpl_shapes_pred = torch.tensor(data_smpl_pred['betas'])
                smpl_poses_pred = torch.tensor(data_smpl_pred['poses'])
                smpl_transl_pred = torch.tensor(data_smpl_pred['transl'])

                smpl_shapes_pred = smpl_shapes_pred.squeeze(0)
                smpl_poses_pred = smpl_poses_pred.squeeze(0)
                smpl_transl_pred = smpl_transl_pred.squeeze(0)

                if not self.opt.use_smpl_para_opt:
                    smpl_transl_pred = smpl_transl_pred.unsqueeze(0)

                smpl_poses_pred = torch.cat((smpl_poses_pred, smpl_transl_pred), dim=0)

        return {
            'name': subject,
            'yaw': yaw,
            'render_path': render_path,
            'render_low_pifu': render_low_pifu,
            'mask_low_pifu': mask_low_pifu,
            'original_high_res_render': render,
            'mask': mask,
            'calib': calib,
            'scale': scale_factor,
            'extrinsic': extrinsic,
            'rotation_matrix': R,
            'b_min': b_min,
            'b_max': b_max,
            'nml_for_opt': nml_for_opt,
            'nmlF': nmlF,
            'nmlB': nmlB,
            'nmlF_high_res': nmlF_high_res,
            'nmlB_high_res': nmlB_high_res,
            'depth_map_F': depth_map_F,
            'depth_map_F_low_res': depth_map_F_low_res,
            'depth_map_B': depth_map_B,
            'depth_map_B_low_res': depth_map_B_low_res,
            'human_parse_map': human_parse_map,
            'smpl_verts': smpl_verts,
            'depthmap_for_smpl_rect': depthmap_for_smpl_rect,
            'depth_map_for_opt': depth_map_for_opt,
            'mask_for_opt': mask_for_opt,
            'smpl_shapes_pred': smpl_shapes_pred,
            'smpl_poses_pred': smpl_poses_pred
        }

    def __getitem__(self, index):
        return self.get_item(index)
