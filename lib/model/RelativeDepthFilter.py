
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from ..net_util import init_net
import cv2




class translated_Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x) + 1.0


class widened_Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x) * 2.0



class RelativeDepthFilter(nn.Module):

    def __init__(self, 
                 opt
                 ):
        super(RelativeDepthFilter, self).__init__()

        self.name = 'depthfilter'

        self.opt = opt

        if not self.opt.second_stage_depth:
            from .UNet import UNet
            n_channels = 4
            if self.opt.use_normal_map_for_depth_training:
                n_channels = n_channels + 3
            if self.opt.use_reference_depth_map_for_depth_training:
                n_channels = n_channels + 1
            if self.opt.use_smpl_depth_map_for_depth_training:
                n_channels = n_channels + 1

            self.image_filter = UNet(n_channels=n_channels, n_classes=1, bilinear=False, last_op=translated_Tanh())
        elif self.opt.second_stage_depth:
            n_channels = 4
            if self.opt.use_normal_map_for_depth_training:
                n_channels = n_channels + 3
            if self.opt.use_reference_depth_map_for_depth_training:
                n_channels = n_channels + 1
            if self.opt.use_smpl_depth_map_for_depth_training:
                n_channels = n_channels + 1
            
            from .UNet import DifferenceUNet
            self.image_filter = DifferenceUNet( n_channels=n_channels, n_classes=1, bilinear=False, last_op=widened_Tanh() , scale_factor=2  )

        else:
            raise Exception("Incorrect config")



        self.im_feat_list = []


        init_net(self) # initialise weights  


 


    def filter(self, images, reference_depthmap):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''

        fuse_image = torch.cat((images, reference_depthmap), dim=1)
        self.im_feat_list  = self.image_filter(fuse_image)


        


    def get_im_feat(self):

        return self.im_feat_list


    def generate_depth_map(self):


        return self.get_im_feat()



    def get_error(self):
        '''
        return the loss given the ground truth labels and prediction
        '''
        error = {}

        error['Err'] = nn.SmoothL1Loss()(self.im_feat_list, self.groundtruth_depthmap)

        return error

    def forward(self, images, smpl_depthmap, reference_depthmap, groundtruth_depthmap ):

        self.filter(images, reference_depthmap)

        self.groundtruth_depthmap = groundtruth_depthmap  # [B, C, H, W] 
            
        err = self.get_error()

        return err
