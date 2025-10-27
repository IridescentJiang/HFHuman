


import os
import random

import numpy as np 
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
import cv2
import torch
import json
import trimesh
import logging

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

CAMERA_TO_MESH_DISTANCE = 10.0 # This is an arbitrary value set by the rendering script. Can be modified by changing the rendering script.



class DepthDataset(Dataset):


    def __init__(self, opt, evaluation_mode=False):
        self.opt = opt
        self.training_subject_list = np.loadtxt("train_set_list.txt", dtype=str).tolist()


        if evaluation_mode:
            print("Overwriting self.training_subject_list!")
            self.training_subject_list = np.loadtxt("test_set_list.txt", dtype=str).tolist()
            self.is_train = False


        self.depth_map_directory = "rendering_script/buffer_depth_maps_of_full_mesh"
        self.render_dir = "rendering_script/buffer_fixed_full_mesh"
        self.smlp_depthmap_dir = "rendering_script/smpl_depth"
        self.reference_depthmap_dir = "rendering_script/reference_depth"
        if self.opt.second_stage_depth:
            self.coarse_depth_map_directory = "trained_coarse_depth_maps_refer"
            

        if self.opt.use_normal_map_for_depth_training:
            self.normal_directory_high_res = "trained_normal_maps"




        self.subjects = self.training_subject_list  

        self.img_files = []
        for training_subject in self.subjects:
            subject_render_folder = os.path.join(self.render_dir, training_subject)
            subject_render_paths_list = [  os.path.join(subject_render_folder,f) for f in os.listdir(subject_render_folder) if "image" in f   ]
            self.img_files = self.img_files + subject_render_paths_list
        self.img_files = sorted(self.img_files)


        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(), #  ToTensor converts input to a shape of (C x H x W) in the range [0.0, 1.0] for each dimension
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalise with mean of 0.5 and std_dev of 0.5 for each dimension. Finally range will be [-1,1] for each dimension
        ])

        self.to_tensor_depth = transforms.Compose([
            transforms.ToTensor(), #  ToTensor converts input to a shape of (C x H x W) in the range [0.0, 1.0] for each dimension
        ])

    def __len__(self):
        return len(self.img_files)






    def get_item(self, index):

        img_path = self.img_files[index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # get yaw
        yaw = img_name.split("_")[-1]
        yaw = int(yaw)

        # get subject
        subject = img_path.split('/')[-2] # e.g. "0507"
            
        param_path = os.path.join(self.render_dir, subject, "rendered_params_" + "{0:03d}".format(yaw) + ".npy")
        render_path = os.path.join(self.render_dir, subject, "rendered_image_" + "{0:03d}".format(yaw) + ".png")
        mask_path = os.path.join(self.render_dir, subject, "rendered_mask_" + "{0:03d}".format(yaw) + ".png")

        smpl_depthmap_path = os.path.join(self.smlp_depthmap_dir, subject, "smpl_depth_" + "{0:03d}".format(yaw) + "_F.png")
        reference_depthmap_path = os.path.join(self.reference_depthmap_dir, subject, "rendered_depthmap_" + "{0:03d}".format(yaw) + ".png")
        

        depth_map_path =  os.path.join(self.depth_map_directory, subject, "rendered_depthmap_" + "{0:03d}".format(yaw) + ".exr"  )
        if self.opt.second_stage_depth:
            coarse_depth_map_path =  os.path.join(self.coarse_depth_map_directory, subject, "rendered_depthmap_" + "{0:03d}".format(yaw) + ".npy"  )


        mask = Image.open(mask_path).convert('L') # convert to grayscale (it shd already be grayscale)
        render = Image.open(render_path).convert('RGB')

        mask = transforms.ToTensor()(mask).float()

        render = self.to_tensor(render)  # normalize render from [0,255] to [-1,1]
        render = mask.expand_as(render) * render

        # get params
        param = np.load(param_path, allow_pickle=True) 
        scale_factor = param.item().get('scale_factor') 

        load_size_associated_with_scale_factor = 1024
        b_range = load_size_associated_with_scale_factor / scale_factor # e.g. 512/scale_factor


        center_indicator = np.zeros([1,1024,1024])
        center_indicator[:, 511:513, 511:513] = 1.0
        # low_res_center_indicator = np.zeros([1, 512, 512])
        # low_res_center_indicator[:, 255:257, 255:257] = 1.0
        center_indicator = torch.Tensor(center_indicator).float()

        # resize the 1024 x 1024 image to 512 x 512 for the low-resolution pifu
        render_low_pifu = F.interpolate(torch.unsqueeze(render,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        mask_low_pifu = F.interpolate(torch.unsqueeze(mask,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        render_low_pifu = render_low_pifu[0]
        mask_low_pifu = mask_low_pifu[0]


        """使用有效像素边界框，调整深度图像，使其和掩码最大程度上吻合"""

        # 加载深度图像和图像掩码
        smpl_depth_map = cv2.imread(smpl_depthmap_path, cv2.IMREAD_GRAYSCALE)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 进行连通组件分析
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask_image)
        # 找到具有最大面积的组件
        max_area_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # 忽略背景组件
        # 创建只包含最大连通域的掩码
        mask_image = np.uint8(labels == max_area_label) * 255

        _, smpl_depth_map_binary = cv2.threshold(smpl_depth_map, 0, 255, cv2.THRESH_BINARY)
        _, mask_image_binary = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY)

        # 计算深度图像的有效像素边界框
        depth_bbox = cv2.boundingRect(smpl_depth_map_binary)
        # 计算图像掩码的有效像素边界框
        mask_bbox = cv2.boundingRect(mask_image_binary)

        # 缩放深度图像使其宽度与 bbox 一致
        scale_ratio = mask_bbox[2] / depth_bbox[2]  # 计算缩放比例
        smpl_depth_image_scaled = cv2.resize(smpl_depth_map, None, fx=scale_ratio, fy=1)

        smpl_depth_image_adjusted = np.zeros_like(mask_image)

        # 确定需要裁剪的区域
        crop_width = min(smpl_depth_image_scaled.shape[1], smpl_depth_image_adjusted.shape[1])
        crop_height = min(smpl_depth_image_scaled.shape[0], smpl_depth_image_adjusted.shape[0])

        smpl_depth_image_adjusted[:crop_height, :crop_width] = smpl_depth_image_scaled[:crop_height, :crop_width]

        # 计算缩放后深度图像的有效像素边界框
        _, smpl_depth_image_adjusted_binary = cv2.threshold(smpl_depth_image_adjusted, 0, 255, cv2.THRESH_BINARY)
        depth_bbox_adjusted = cv2.boundingRect(smpl_depth_image_adjusted_binary)

        # 计算平移的偏移量
        dx = mask_bbox[0] - depth_bbox_adjusted[0]
        dy = mask_bbox[1] - depth_bbox_adjusted[1]

        # 平移深度图像
        depth_image_translated = np.roll(smpl_depth_image_adjusted, dx, axis=1)
        depth_image_translated = np.roll(depth_image_translated, dy, axis=0)

        # 使用掩码进行遮罩
        smpl_depth_map = cv2.bitwise_and(depth_image_translated, depth_image_translated, mask=mask_image)

        # save_path = './tmp/' + str(subject) + '_' + str(yaw) + '_smpl_depth_map.jpg'
        # cv2.imwrite(save_path, smpl_depth_map)

        smpl_depth_map = torch.from_numpy(smpl_depth_map)
        smpl_depth_map = torch.unsqueeze(smpl_depth_map, 0)




        # reference_depth_map
        reference_depth_map = Image.open(reference_depthmap_path).convert('L')
        reference_depth_map = self.to_tensor_depth(reference_depth_map)
        reference_depth_map = mask.expand_as(reference_depth_map) * reference_depth_map

        # reference_depth_map_save = torch.squeeze(reference_depth_map)
        # reference_depth_map_np = reference_depth_map_save.detach().cpu().numpy()

        # reference_depth_map_pil = Image.fromarray((reference_depth_map_np * 255).astype('uint8'))
        # save_path = './tmp/' + str(subject) + '_' + str(yaw) + '_reference_depth_map.jpg'
        # reference_depth_map_pil.save(save_path)


        if self.opt.use_normal_map_for_depth_training:
            nmlF_high_res_path =  os.path.join(self.normal_directory_high_res, subject, "rendered_nmlF_" + "{0:03d}".format(yaw) + ".npy"  )
            nmlF_high_res = np.load(nmlF_high_res_path) # shape of [3, 1024,1024]
            nmlF_high_res = torch.Tensor(nmlF_high_res)
            nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res
        else:
            nmlF_high_res = 0



        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED).astype(np.float32) 
        depth_map = depth_map[:,:,0]
        mask_depth = depth_map > 100

        depth_map = depth_map - CAMERA_TO_MESH_DISTANCE # make the center pixel to have a depth value of 0.0
        depth_map = depth_map / (b_range/self.opt.resolution  ) # converts the units into in terms of no. of bounding cubes
        depth_map = depth_map / (self.opt.resolution/2) # normalize into range of [-1,1]

        depth_map = depth_map + 1.0 # convert into range of [0,2.0] where the center pixel has value of 1.0
        depth_map[mask_depth] = 0 # the invalid values are set to 0.

        depth_map = np.expand_dims(depth_map,0) # shape of [1,1024,1024]
        depth_map = torch.Tensor(depth_map)
        depth_map = mask.expand_as(depth_map) * depth_map

    
        if self.opt.second_stage_depth:
            coarse_depth_map = np.load(coarse_depth_map_path)

            coarse_depth_map = torch.Tensor(coarse_depth_map)
            coarse_depth_map = mask.expand_as(coarse_depth_map) * coarse_depth_map # shape of [C,H,W]

        else:
            coarse_depth_map = 0



        return {
            'name': subject,
            'render_path':render_path,
            'render_low_pifu': render_low_pifu,
            'mask_low_pifu': mask_low_pifu,
            'original_high_res_render': render,
            'center_indicator': center_indicator,
            'depth_map': depth_map,
            'smpl_depth_map': smpl_depth_map,
            'reference_depth_map': reference_depth_map,
            'coarse_depth_map':coarse_depth_map,
            'nmlF_high_res':nmlF_high_res,
            'mask':mask

                }



    def __getitem__(self, index):
        return self.get_item(index)




class DepthInferenceDataset(Dataset):


    def __init__(self, opt,):
        self.opt = opt
        self.is_train = False

        self.render_dir = "rendering_script/inference/buffer_fixed_full_mesh"
        self.reference_depthmap_dir = "rendering_script/inference/reference_depth"
        if self.opt.second_stage_depth:
            self.coarse_depth_map_directory = "rendering_script/inference/trained_coarse_depth_maps_refer"

        if self.opt.use_normal_map_for_depth_training:
            self.normal_directory_high_res = "rendering_script/inference/trained_normal_maps"

        self.img_files = []

        subject_render_folder = os.path.join(self.render_dir, "in-the-wild")
        subject_render_paths_list = [os.path.join(subject_render_folder, f) for f in
                                     os.listdir(subject_render_folder) if "image" in f]
        self.img_files = self.img_files + subject_render_paths_list
        self.img_files = sorted(self.img_files)

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # ToTensor converts input to a shape of (C x H x W) in the range [0.0, 1.0] for each dimension
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # normalise with mean of 0.5 and std_dev of 0.5 for each dimension. Finally range will be [-1,1] for each dimension
        ])

        self.to_tensor_depth = transforms.Compose([
            transforms.ToTensor(),
            # ToTensor converts input to a shape of (C x H x W) in the range [0.0, 1.0] for each dimension
        ])

    def __len__(self):
        return len(self.img_files)

    def get_item(self, index):

        img_path = self.img_files[index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # get yaw
        yaw = img_name.split("_")[-1]
        yaw = int(yaw)

        # get subject
        subject = img_path.split('/')[-2]  # e.g. "0507"

        render_path = os.path.join(self.render_dir, subject, "rendered_image_" + "{0:03d}".format(yaw) + ".png")
        mask_path = os.path.join(self.render_dir, subject, "rendered_mask_" + "{0:03d}".format(yaw) + ".png")

        reference_depthmap_path = os.path.join(self.reference_depthmap_dir, subject,
                                               "rendered_depthmap_" + "{0:03d}".format(yaw) + ".png")

        if self.opt.second_stage_depth:
            coarse_depth_map_path = os.path.join(self.coarse_depth_map_directory, subject,
                                                 "rendered_depthmap_" + "{0:03d}".format(yaw) + ".npy")

        mask = Image.open(mask_path).convert('L')  # convert to grayscale (it shd already be grayscale)
        render = Image.open(render_path).convert('RGB')

        mask = transforms.ToTensor()(mask).float()

        render = self.to_tensor(render)  # normalize render from [0,255] to [-1,1]
        render = mask.expand_as(render) * render


        center_indicator = np.zeros([1, 1024, 1024])
        center_indicator[:, 511:513, 511:513] = 1.0
        # low_res_center_indicator = np.zeros([1, 512, 512])
        # low_res_center_indicator[:, 255:257, 255:257] = 1.0
        center_indicator = torch.Tensor(center_indicator).float()

        # resize the 1024 x 1024 image to 512 x 512 for the low-resolution pifu
        render_low_pifu = F.interpolate(torch.unsqueeze(render, 0),
                                        size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
        mask_low_pifu = F.interpolate(torch.unsqueeze(mask, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
        render_low_pifu = render_low_pifu[0]
        mask_low_pifu = mask_low_pifu[0]

        # reference_depth_map
        reference_depth_map = Image.open(reference_depthmap_path).convert('L')
        reference_depth_map = self.to_tensor_depth(reference_depth_map)
        reference_depth_map = mask.expand_as(reference_depth_map) * reference_depth_map

        # reference_depth_map_save = torch.squeeze(reference_depth_map)
        # reference_depth_map_np = reference_depth_map_save.detach().cpu().numpy()

        # reference_depth_map_pil = Image.fromarray((reference_depth_map_np * 255).astype('uint8'))
        # save_path = './tmp/' + str(subject) + '_' + str(yaw) + '_reference_depth_map.jpg'
        # reference_depth_map_pil.save(save_path)

        if self.opt.use_normal_map_for_depth_training:
            nmlF_high_res_path = os.path.join(self.normal_directory_high_res, subject,
                                              "rendered_nmlF_" + "{0:03d}".format(yaw) + ".npy")
            nmlF_high_res = np.load(nmlF_high_res_path)  # shape of [3, 1024,1024]
            nmlF_high_res = torch.Tensor(nmlF_high_res)
            nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res
        else:
            nmlF_high_res = 0

        if self.opt.second_stage_depth:
            coarse_depth_map = np.load(coarse_depth_map_path)

            coarse_depth_map = torch.Tensor(coarse_depth_map)
            coarse_depth_map = mask.expand_as(coarse_depth_map) * coarse_depth_map  # shape of [C,H,W]

        else:
            coarse_depth_map = 0

        return {
            'name': subject,
            'render_path': render_path,
            'render_low_pifu': render_low_pifu,
            'mask_low_pifu': mask_low_pifu,
            'original_high_res_render': render,
            'center_indicator': center_indicator,
            'reference_depth_map': reference_depth_map,
            'coarse_depth_map': coarse_depth_map,
            'nmlF_high_res': nmlF_high_res,
            'mask': mask

        }

    def __getitem__(self, index):
        return self.get_item(index)
    












