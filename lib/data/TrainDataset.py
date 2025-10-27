
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


def load_trimesh(root_dir, training_subject_list = None):


    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        if f == ".DS_Store":
            continue

        if f not in training_subject_list: # only load meshes that are in the training set
            continue

        meshs[f] = trimesh.load(os.path.join(root_dir, f, '%s.obj' % f))

    return meshs



class TrainDataset(Dataset):

    def __init__(self, opt, projection='orthogonal', phase = 'train', evaluation_mode=False, validation_mode=False):
        self.opt = opt
        self.projection_mode = projection

        if self.opt.debug_mode:
            self.training_subject_list = np.loadtxt("/data/jym/IntegratedPIFu/IntegratePIFu/train_set_list.txt", dtype=str)
        else:
            self.training_subject_list = np.loadtxt("train_set_list.txt", dtype=str)


        self.evaluation_mode = evaluation_mode

        self.validation_mode = validation_mode

        self.phase = phase
        self.is_train = (self.phase == 'train')

        if self.opt.useValidationSet:

            indices = np.arange( len(self.training_subject_list) )
            np.random.seed(10)
            np.random.shuffle(indices)
            lower_split_index = round( len(self.training_subject_list )* 0.1 )
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


        self.root = "rendering_script/buffer_fixed_full_mesh"

        if self.opt.debug_mode:
            self.root = "/data/jym/IntegratedPIFu/IntegratePIFu/rendering_script/buffer_fixed_full_mesh"

        self.mesh_directory = "rendering_script/THuman2.0_Release"

        # self.mesh_dic = load_trimesh(self.mesh_directory,  training_subject_list = self.training_subject_list)  # a dict containing the meshes of all the CAD models.

        if self.opt.use_groundtruth_normal_maps:
            self.normal_directory_high_res = "rendering_script/buffer_normal_maps_of_full_mesh"
        else:
            self.normal_directory_high_res = "trained_normal_maps"

        if self.opt.useGTdepthmap:
            # self.depth_map_directory = "rendering_script/buffer_depth_maps_of_full_mesh"
            self.depth_map_directory = "rendering_script/buffer_depth_maps_of_full_mesh_png"
        else:
            self.depth_map_directory = "trained_coarse_depth_maps_refer" # New version (Depth maps trained with only normal - Second Stage maps)

        if self.opt.use_SMPL_depth_map:
            self.SMPL_depth_map_directory = "rendering_script/smpl_depth"
            # self.depth_map_directory = "trained_depth_maps"
            self.depth_map_directory = "trained_coarse_depth_maps"

        if self.opt.use_groundtruth_human_parse_maps:
            self.human_parse_map_directory = "rendering_script/render_human_parse_results"
        else:
            self.human_parse_map_directory = "trained_parse_maps"

        if self.opt.use_smpl_x:
            self.smpl_directory = "rendering_script/smpl_x"
            if self.opt.debug_mode:
                self.smpl_directory = "/data/jym/IntegratedPIFu/IntegratePIFu/rendering_script/smpl_x"
        elif self.opt.use_smpl:
            if self.opt.use_smpl_gt:
                self.smpl_directory = "rendering_script/smpl_gt"
                if self.opt.debug_mode:
                    self.smpl_directory = "/data/jym/IntegratedPIFu/IntegratePIFu/rendering_script/smpl_gt"
            else:
                self.smpl_directory = "rendering_script/smpl_pred"
                if self.opt.debug_mode:
                    self.smpl_directory = "/data/jym/IntegratedPIFu/IntegratePIFu/rendering_script/smpl_pred"
        else:
            self.smpl_directory = "rendering_script/smpl_pred"

        if self.opt.use_smpl_para or self.opt.use_smpl_para_gt or self.opt.smpl_para_depth_guidance_learning:
            self.smpl_para_gt = "rendering_script/smpl_para_gt"
            self.smpl_mesh_gt = "rendering_script/smpl_gt"
            self.smpl_para_pred = "rendering_script/smpl_para_pred"
            # Using rectified smpl for testing
            # self.smpl_para_pred = "rendering_script/smpl_para_rect"

        if self.opt.use_smpl_para_opt:
            self.smpl_para_opt = "rendering_script/smpl_para_opt"


        self.subjects = self.training_subject_list

        self.load_size = self.opt.loadSize

        self.num_sample_inout = self.opt.num_sample_inout

        if self.opt.use_sampled_points_smpl:
            self.point_output_fd = 'rendering_script/sample_points_smpl_new'
        else:
            self.point_output_fd = 'rendering_script/sample_points'

        self.img_files = []
        for training_subject in self.subjects:
            subject_render_folder = os.path.join(self.root, training_subject)
            subject_render_paths_list = [os.path.join(subject_render_folder ,f) for f in os.listdir(subject_render_folder) if "image" in f   ]
            self.img_files = self.img_files + subject_render_paths_list

        self.img_files = sorted(self.img_files)
        self.img_files = np.array(self.img_files, dtype=np.string_)


        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # ToTensor converts input to a shape of (C x H x W) in the range [0.0, 1.0] for each dimension
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalise with mean of 0.5 and std_dev of 0.5 for each dimension. Finally range will be [-1,1] for each dimension
        ])

        self.to_tensor_depth = transforms.Compose([
            transforms.ToTensor()
        ])



    def __len__(self):
        return len(self.img_files)

    def select_sampling_method(self, subject, yaw, calib, b_min, b_max, R = None):

        if self.opt.use_sampled_points:
            if self.opt.use_sampled_points_smpl:
                sample_path = os.path.join(self.point_output_fd, subject, 'samples_%s.mat' % f"{yaw:03d}")
            else:
                sample_path = os.path.join(self.point_output_fd, subject, 'samples.mat')
            try:
                points_data = sio.loadmat(sample_path, verify_compressed_data_integrity=False)
            except ValueError as e:
                print('Value error occurred when loading ' + sample_path)
                raise ValueError(str(e))

            rand_id = np.random.rand(self.num_sample_inout)
            if self.opt.useDOS:
                surface_points_with_normal_sigma_id = np.int32(
                    rand_id * len(points_data['surface_points_with_normal_sigma']))
                outside_surface_points_id = np.int32(rand_id * len(points_data['outside_surface_points']))
                way_inside_pts_id = np.int32(rand_id * len(points_data['way_inside_pts']))

                surface_points_with_normal_sigma = points_data['surface_points_with_normal_sigma'][surface_points_with_normal_sigma_id]
                way_inside_pts = points_data['way_inside_pts'][way_inside_pts_id]
                outside_surface_points = points_data['outside_surface_points'][outside_surface_points_id]

                labels_with_normal_sigma = points_data['labels_with_normal_sigma'][:, surface_points_with_normal_sigma_id]

                n_way_inside_pts = round(self.num_sample_inout * self.opt.ratio_of_way_inside_points)
                way_inside_pts = way_inside_pts[:n_way_inside_pts]

                n_outside_surface_points = round(self.num_sample_inout * self.opt.ratio_of_outside_points)
                outside_surface_points = outside_surface_points[:n_outside_surface_points]

                n_surface_points_with_normal_sigma = self.num_sample_inout
                surface_points_with_normal_sigma = surface_points_with_normal_sigma[:n_surface_points_with_normal_sigma]
                labels_with_normal_sigma = labels_with_normal_sigma[:, :n_surface_points_with_normal_sigma]

                inside_points_low_res_pifu_DOS = np.concatenate([surface_points_with_normal_sigma, way_inside_pts], 0)
                all_points_low_res_pifu = np.concatenate([inside_points_low_res_pifu_DOS, outside_surface_points], 0)

            else:
                inside_points_low_res_pifu_id = np.int32(rand_id * len(points_data['inside_points_low_res_pifu']))
                outside_points_low_res_pifu_id = np.int32(rand_id * len(points_data['outside_points_low_res_pifu']))

                inside_points_low_res_pifu = points_data['inside_points_low_res_pifu'][inside_points_low_res_pifu_id]
                outside_points_low_res_pifu = points_data['outside_points_low_res_pifu'][outside_points_low_res_pifu_id]

        else:
            compensation_factor = 4.0
            #
            # # mesh = self.mesh_dic[subject].copy() # the mesh of 1 subject/CAD
            mesh = trimesh.load(os.path.join(self.mesh_directory, subject, '%s.obj' % subject))
            #
            # # note, this is the solution for when dataset is "THuman"
            # # adjust sigma according to the mesh's size (measured using the y-coordinates)
            y_length = np.abs(np.max(mesh.vertices, axis=0)[1])  + np.abs(np.min(mesh.vertices, axis=0)[1] )
            sigma_multiplier = y_length / 188
            # sigma_multiplier = y_length / 38  # The higher the sigma, the more concentrated the points are on the model surface.

            surface_points, face_indices = trimesh.sample.sample_surface(mesh, int
                (compensation_factor * 4 * self.num_sample_inout) )# self.num_sample_inout is no. of sampling points and is default to 8000.


            # add random points within image space
            length = b_max - b_min # has shape of (3,)
            if not self.opt.useDOS:
                random_points = np.random.rand( int(compensation_factor * self.num_sample_inout // 4) , 3) * length + b_min # shape of [compensation_factor*num_sample_inout/4, 3]
                surface_points_shape = list(surface_points.shape)
                random_noise = np.random.normal(scale= self.opt.sigma_low_resolution_pifu * sigma_multiplier, size=surface_points_shape)
                sample_points_low_res_pifu = surface_points + random_noise # sample_points are points very near the surface. The sigma represents the std dev of the normal distribution
                sample_points_low_res_pifu = np.concatenate([sample_points_low_res_pifu, random_points], 0) # shape of [compensation_factor*4.25*num_sample_inout, 3]
                np.random.shuffle(sample_points_low_res_pifu)

                inside_low_res_pifu = mesh.contains \
                    (sample_points_low_res_pifu) # return a boolean 1D array of size (num of sample points,)
                inside_points_low_res_pifu = sample_points_low_res_pifu[inside_low_res_pifu]


            if self.opt.useDOS:

                num_of_pts_in_section = self.num_sample_inout // 3

                normal_vectors = mesh.face_normals[face_indices] # [num_of_sample_pts, 3]

                directional_vector = np.array([[0.0 ,0.0 ,1.0]]) # 1x3
                directional_vector = np.matmul(inv(R), directional_vector.T) # 3x1
                # get dot product
                normal_vectors_to_use = normal_vectors[ 0: self.num_sample_inout ,:]
                dot_product = np.matmul(directional_vector.T, normal_vectors_to_use.T)  # [1 x num_of_sample_pts]

                dot_product[dot_product < 0] = -1.0  # points generated from faces that are facing backwards
                dot_product[dot_product >= 0] = 1.0  # points generated from faces that are facing camera
                z_displacement = np.matmul(dot_product.T,
                                           directional_vector.T)  # [num_of_sample_pts, 3]. Will displace points facing backwards to go backwards, but points facing forward to go forward

                normal_sigma = np.random.normal(loc=0.0, scale=1.0,
                                                size=[4 * self.num_sample_inout, 1])  # shape of [num_of_sample_pts, 1]
                normal_sigma_mask = (normal_sigma[:, 0] < 1.0) & (normal_sigma[:, 0] > -1.0)
                normal_sigma = normal_sigma[normal_sigma_mask, :]
                normal_sigma = normal_sigma[0:self.num_sample_inout, :]
                surface_points_with_normal_sigma = surface_points[0:self.num_sample_inout,
                                                   :] - z_displacement * sigma_multiplier * normal_sigma * 2.0  # The minus sign means that we are getting points that are all inside the surface, rather than outside of it.
                labels_with_normal_sigma = normal_sigma.T / 2.0 * 0.8  # set range to 0.8. range from -0.4 to 0.4
                labels_with_normal_sigma = labels_with_normal_sigma + 0.5  # range from 0.1 to 0.9 . Shape of [1, self.num_sample_inout]

                # get way inside points:
                num_of_way_inside_pts = round(self.num_sample_inout * self.opt.ratio_of_way_inside_points)
                way_inside_pts = surface_points[0: num_of_way_inside_pts] - z_displacement[
                                                                            0:num_of_way_inside_pts] * sigma_multiplier * (
                                             4.0 + np.random.uniform(low=0.0, high=2.0, size=None))
                proximity = trimesh.proximity.longest_ray(mesh, way_inside_pts, -z_displacement[
                                                                                 0:num_of_way_inside_pts])  # shape of [num_of_sample_pts]

                way_inside_pts[
                    proximity < (sigma_multiplier * 4.0)] = 0  # remove points that are too near the opposite z direction

                f = SDF(mesh.vertices, mesh.faces)
                contains = f.contains(way_inside_pts)
                way_inside_pts[~contains, :] = 0  # remove pts that are actually outside the mesh

                # signed_dist, _, _ = igl.signed_distance(way_inside_pts, mesh.vertices, mesh.faces)
                # way_inside_pts[signed_dist>0, :] = 0 # remove pts that are actually outside the mesh

                # proximity = trimesh.proximity.signed_distance(mesh, way_inside_pts) # [num_of_sample_pts]
                # way_inside_pts[proximity<0, :] = 0 # remove pts that are actually outside the mesh

                inside_points_low_res_pifu = np.concatenate([surface_points_with_normal_sigma, way_inside_pts], 0)

                # get way outside points
                num_of_outside_pts = round(self.num_sample_inout * self.opt.ratio_of_outside_points)
                outside_surface_points = surface_points[0: num_of_outside_pts] + z_displacement[
                                                                                 0:num_of_outside_pts] * sigma_multiplier * (
                                                     5.0 + np.random.uniform(low=0.0, high=50.0, size=None))
                proximity = trimesh.proximity.longest_ray(mesh, outside_surface_points, z_displacement[
                                                                                        0:num_of_outside_pts])  # shape of [num_of_sample_pts]
                outside_surface_points[
                    proximity < (sigma_multiplier * 5.0)] = 0  # remove points that are too near the opposite z direction

                all_points_low_res_pifu = np.concatenate([inside_points_low_res_pifu, outside_surface_points], 0)

            else:
                outside_points_low_res_pifu = sample_points_low_res_pifu[np.logical_not(inside_low_res_pifu)]


        if self.opt.useDOS:

            samples_low_res_pifu = all_points_low_res_pifu.T  # should have shape of [3, 5000]

            labels_low_res_pifu = np.concatenate([labels_with_normal_sigma, np.ones((1, way_inside_pts.shape[0])) * 1.0,
                                                  np.ones((1, outside_surface_points.shape[0])) * 0.0],
                                                 1)  # should have shape of [1, 5000]. If element is 1, it means the point is inside. If element is 0, it means the point is outside.

        else:
            # reduce the number of inside and outside points if there are too many inside points. (it is very likely that "nin > self.num_sample_inout // 2" is true)
            nin = inside_points_low_res_pifu.shape[0]

            inside_points_low_res_pifu = inside_points_low_res_pifu[
                                         :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points_low_res_pifu  # should have shape of [2500, 3]
            outside_points_low_res_pifu = outside_points_low_res_pifu[
                                          :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points_low_res_pifu[:(self.num_sample_inout - nin)]  # should have shape of [2500, 3]

            samples_low_res_pifu = np.concatenate([inside_points_low_res_pifu, outside_points_low_res_pifu], 0).T  # should have shape of [3, 5000]

            labels_low_res_pifu = np.concatenate([np.ones((1, inside_points_low_res_pifu.shape[0])),
                                                  np.zeros((1, outside_points_low_res_pifu.shape[0]))],
                                                 1)  # should have shape of [1, 5000]. If element is 1, it means the point is inside. If element is 0, it means the point is outside.

        samples_low_res_pifu = torch.Tensor(samples_low_res_pifu.copy()).float()

        labels_low_res_pifu = torch.Tensor(labels_low_res_pifu.copy()).float()

        # del mesh

        return {
            'samples_low_res_pifu': samples_low_res_pifu,
            'labels_low_res_pifu': labels_low_res_pifu
        }

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


        if self.opt.use_groundtruth_normal_maps:
            nmlF_high_res_path = os.path.join(self.normal_directory_high_res, subject,
                                              "rendered_nmlF_" + "{0:03d}".format(yaw) + ".exr")
            nmlB_high_res_path = os.path.join(self.normal_directory_high_res, subject,
                                              "rendered_nmlB_" + "{0:03d}".format(yaw) + ".exr")
        else:
            nmlF_high_res_path = os.path.join(self.normal_directory_high_res, subject,
                                              "rendered_nmlF_" + "{0:03d}".format(yaw) + ".npy")
            nmlB_high_res_path = os.path.join(self.normal_directory_high_res, subject,
                                              "rendered_nmlB_" + "{0:03d}".format(yaw) + ".npy")

        if self.opt.smpl_test_opt:
            nmlF_for_smpl_opt_high_res_path = os.path.join(self.normal_directory_high_res, subject,
                                          "rendered_nmlF_" + "{0:03d}".format(yaw) + ".png")

        if self.opt.useGTdepthmap:
            # depth_map_path = os.path.join(self.depth_map_directory, subject,
            #                               "rendered_depthmap_" + "{0:03d}".format(yaw) + ".exr")
            depth_map_path = os.path.join(self.depth_map_directory, subject,
                                          "rendered_depthmap_" + "{0:03d}".format(yaw) + ".png")
            depth_map_L_path = os.path.join(self.depth_map_directory, subject,
                                          "rendered_depthmap_" + "{0:03d}".format(yaw) + ".png")

        else:
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

        if self.opt.inference:
            center = torch.tensor([-1.0, 100, 1.0], dtype=torch.float32)
            R = torch.tensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, -1]
            ], dtype=torch.float32)

            scale_factor = 1

        else:
            # get params
            param = np.load(param_path,
                            allow_pickle=True)  # param is a np.array that looks similar to a dict.  # ortho_ratio = 0.4 , e.g. scale or y_scale = 0.961994278, e.g. center or vmed = [-1.0486  92.56105  1.0101 ]
            center = param.item().get(
                'center')  # is camera 3D center position in the 3D World point space (without any rotation being applied).
            R = param.item().get('R')  # R is used to rotate the CAD model according to a given pitch and yaw.
            scale_factor = param.item().get(
                'scale_factor')  # is camera 3D center position in the 3D World point space (without any rotation being applied).

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
                    nmlB = F.interpolate(torch.unsqueeze(nmlB_high_res, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
                    nmlB = nmlB[0]

        if self.opt.smpl_test_opt:
            nmlF_for_smpl_opt_high_res = cv2.imread(nmlF_for_smpl_opt_high_res_path)
            nmlF_for_smpl_opt_high_res = torch.from_numpy(nmlF_for_smpl_opt_high_res).permute(2, 0, 1).unsqueeze(0)
            nml_for_opt = F.interpolate(nmlF_for_smpl_opt_high_res, size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
            nml_for_opt = nml_for_opt[0]

        depthmap_for_smpl_rect = []
        depth_map_for_opt = []
        mask_for_opt = []

        if self.opt.use_depth_map or self.opt.smpl_para_depth_guidance_learning:

            if self.opt.useGTdepthmap:
                # depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                # depth_map = depth_map[:, :, 0]
                # mask_depth = depth_map > 100
                # camera_position = 10.0
                # depth_map = depth_map - camera_position  # make the center pixel to have a depth value of 0.0
                # depth_map = depth_map / (
                #             b_range / self.opt.resolution)  # converts the units into in terms of no. of bounding cubes
                # depth_map = depth_map / (self.opt.resolution / 2)  # normalize into range of [-1,1]
                # depth_map = depth_map + 1.0  # convert into range of [0,2.0] where the center pixel has value of 1.0
                # depth_map[mask_depth] = 0  # the invalid values are set to 0.
                # depth_map = np.expand_dims(depth_map, 0)  # shape of [1,1024,1024]
                # depth_map = torch.Tensor(depth_map)
                # depth_map = mask.expand_as(depth_map) * depth_map

                depth_map = Image.open(depth_map_path).convert('L')
                depth_map = self.to_tensor_depth(depth_map)
                depth_map = mask.expand_as(depth_map) * depth_map

                # downsample depth_map
                if self.opt.depth_in_front:
                    depth_map_low_res = F.interpolate(torch.unsqueeze(depth_map, 0),
                                                      size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
                    depth_map_low_res = depth_map_low_res[0]
                else:
                    depth_map_low_res = 0

            else:
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

            if self.opt.use_groundtruth_human_parse_maps:
                human_parse_map_1 = (human_parse_map == 0.5).float()
                human_parse_map_2 = (human_parse_map == 0.6).float()
                human_parse_map_3 = (human_parse_map == 0.7).float()
                human_parse_map_4 = (human_parse_map == 0.8).float()
                human_parse_map_5 = (human_parse_map == 0.9).float()
                human_parse_map_6 = (human_parse_map == 1.0).float()
                human_parse_map_list = [human_parse_map_1, human_parse_map_2, human_parse_map_3, human_parse_map_4,
                                        human_parse_map_5, human_parse_map_6]
            else:
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

        smpl_shapes_gt = []
        smpl_poses_gt = []
        smpl_vertices_gt = []
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

                smpl_transl_pred = smpl_transl_pred.unsqueeze(0)
                smpl_poses_pred = torch.cat((smpl_poses_pred, smpl_transl_pred), dim=0)


        if self.opt.use_smpl_para_gt or self.opt.smpl_para_depth_guidance_learning:
            smpl_para_gt_path = os.path.join(self.smpl_para_gt, subject,
                                              "smpl_" + "{0:03d}".format(yaw) + ".json")

            if os.path.exists(smpl_para_gt_path):
                with open(smpl_para_gt_path, 'r') as f:
                    data_smpl_gt = json.load(f)

                # 获取 "betas" 和 "poses" 矩阵
                smpl_shapes_gt = torch.tensor(data_smpl_gt['betas'])
                smpl_poses_gt = torch.tensor(data_smpl_gt['poses'])
                smpl_transl_gt = torch.tensor(data_smpl_gt['transl'])

                smpl_transl_gt = smpl_transl_gt.unsqueeze(0)
                smpl_poses_gt = torch.cat((smpl_poses_gt, smpl_transl_gt), dim=0)

            smpl_mesh_gt_path = os.path.join(self.smpl_mesh_gt, subject, "smpl_" + "{0:03d}".format(yaw) + ".obj")
            smpl_mesh_gt = trimesh.load(smpl_mesh_gt_path)
            smpl_vertices_gt = torch.tensor(smpl_mesh_gt.vertices).float()

        if self.evaluation_mode:
            # sample_data = {'samples_low_res_pifu': 0, 'labels_low_res_pifu': 0}
            sample_data = self.select_sampling_method(subject, yaw, calib, b_min=b_min, b_max=b_max, R=R)

        else:
            if self.opt.num_sample_inout:  # opt.num_sample_inout has default of 8000
                # sample_data = {'samples_low_res_pifu': 0, 'labels_low_res_pifu': 0}
                sample_data = self.select_sampling_method(subject, yaw, calib, b_min=b_min, b_max=b_max, R=R)


        return {
            'name': subject,
            'render_path': render_path,
            'render_low_pifu': render_low_pifu,
            'mask_low_pifu': mask_low_pifu,
            'original_high_res_render': render,
            'mask': mask,
            'calib': calib,
            'scale': scale_factor,
            'extrinsic': extrinsic,
            'rotation_matrix': R,
            'samples_low_res_pifu': sample_data['samples_low_res_pifu'],
            'labels_low_res_pifu': sample_data['labels_low_res_pifu'],
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
            'smpl_shapes_gt': smpl_shapes_gt,
            'smpl_poses_gt': smpl_poses_gt,
            'smpl_vertices_gt': smpl_vertices_gt,
            'depthmap_for_smpl_rect': depthmap_for_smpl_rect,
            'depth_map_for_opt': depth_map_for_opt,
            'mask_for_opt': mask_for_opt,
            'smpl_shapes_pred': smpl_shapes_pred,
            'smpl_poses_pred': smpl_poses_pred
        }

    def __getitem__(self, index):
        return self.get_item(index)





