# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)
import os
import sys
import json

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from lib.common.render import query_color, image2vid
from lib.renderer.mesh import compute_normal_batch
from lib.common.config import cfg
from lib.common.cloth_extraction import extract_cloth
from lib.dataset.mesh_util import (load_checkpoint,
                                   update_mesh_shape_prior_losses,
                                   get_optim_grid_image, blend_rgb_norm,
                                   unwrap, remesh, tensor2variable,
                                   rot6d_to_rotmat, rescale_smpl, projection)
from lib.dataset.mesh_util import (SMPLX, apply_vertex_mask, part_removal, poisson)
from lib.dataset.TestDataset import TestDataset
from lib.net.local_affine import LocalAffine
from pytorch3d.structures import Meshes
from apps.ICON import ICON
from trimesh.collision import CollisionManager


import os
from termcolor import colored
import argparse
import numpy as np
from PIL import Image
import trimesh
import pickle
import numpy as np
import torch
import math
import cv2

torch.backends.cudnn.benchmark = True


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['vertex_indices'], data['faces']


def load_obj(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines


def extract_vertices(obj_lines, vertex_indices):
    vertices = []
    index_mapping = {}
    valid_vertices = [line for line in obj_lines if line.startswith('v ')]
    for new_index, obj_index in enumerate(vertex_indices):
        # Adjust by subtracting 1 because vertex_indices are 1-based
        line = valid_vertices[obj_index - 1]
        if line.startswith('v '):
            # Track the new index mapping
            index_mapping[obj_index] = new_index + 1  # OBJ format requires 1-based indexing
            vertices.append(line.strip())
    return vertices, index_mapping


def extract_faces(face_lines, index_mapping):
    extracted_faces = []
    for face_str in face_lines:
        if face_str.startswith('f '):
            # Parse the face indices
            original_indices = [int(i) for i in face_str.split()[1:]]

            # Remap them to the new indices
            try:
                new_indices = [index_mapping[idx] for idx in original_indices]
                extracted_faces.append(f"f {' '.join(map(str, new_indices))}")
            except KeyError:
                # Skip faces that reference vertices not in index_mapping
                continue
    return extracted_faces


def save_obj(vertices, faces, output_file):
    with open(output_file, 'w') as file:
        for vertex in vertices:
            file.write(vertex + '\n')
        for face in faces:
            file.write(face + '\n')


base_dir = './results/hand_replacement'
gpu_device = 0


def process_hand(json_filename, obj_lines, identifier):
    json_file = os.path.join(base_dir, json_filename)
    vertex_indices, faces = load_json(json_file)

    vertices, index_mapping = extract_vertices(obj_lines, vertex_indices)
    hand_faces = extract_faces(faces, index_mapping)

    hand_output = os.path.join(base_dir, f'test_{identifier}_{json_filename.replace(".json", ".obj")}')
    save_obj(vertices, hand_faces, hand_output)

    return trimesh.load(hand_output)


def clean_mesh(mesh):
    cc = mesh.split(only_watertight=False)

    out_mesh = cc[0]
    bbox = out_mesh.bounds
    height = bbox[1, 0] - bbox[0, 0]
    for c in cc:
        bbox = c.bounds
        if height < bbox[1, 0] - bbox[0, 0]:
            height = bbox[1, 0] - bbox[0, 0]
            out_mesh = c

    return out_mesh

def get_vertex_index_map(smpl_mesh, right_hand_mesh):
    # 取得 right_hand_mesh 的所有顶点在 smpl_mesh 中的索引
    indices_map = {tuple(v): idx for idx, v in enumerate(smpl_mesh.vertices)}
    right_hand_indices = [indices_map[tuple(v)] for v in right_hand_mesh.vertices]
    return right_hand_indices


def is_inside(smpl_mesh, hand_mesh):

    # 检查hand_mesh的所有顶点是否在smpl_mesh内部
    contains = smpl_mesh.contains(hand_mesh.vertices)

    # 如果任何点都不在 smpl_mesh 内部，则返回 False
    if not np.any(contains):
        return False
    return True

def process_hand_collision(mesh_file, obj_lines, identifier, hand_mesh, collision_manager, hand_type):
    # 处理手部网格
    smpl_mesh = process_hand(mesh_file, obj_lines, identifier)

    # # 设置碰撞管理器
    # collision_manager.add_object(f'smpl_mesh_no_{hand_type}_hand', smpl_mesh)
    # collision_manager.add_object(f'{hand_type}_hand_mesh', hand_mesh)

    # 检查碰撞和内部判断
    # if not collision_manager.in_collision_internal() and not is_inside(smpl_mesh, hand_mesh):
    if not is_inside(smpl_mesh, hand_mesh):
        return hand_mesh
    return None

def refine_and_export_mesh(hand_mesh, pred_obj_mesh, refine_mesh_path, smpl_mesh, foots_mesh, face_mesh, device):

    final_mesh_part = part_removal(pred_obj_mesh, foots_mesh, 0.06, device, smpl_mesh, region="foots")
    final_mesh_part = part_removal(final_mesh_part, face_mesh, 0.06, device, smpl_mesh, region="face")

    if hand_mesh:
        final_mesh_part = part_removal(final_mesh_part, hand_mesh, 0.08, device, smpl_mesh, region="hand")

    final_mesh_part = clean_mesh(final_mesh_part)
    full_mesh = hand_mesh + foots_mesh + face_mesh + final_mesh_part

    return poisson(full_mesh, refine_mesh_path, 10)

def main():
    txt_file = os.path.join(base_dir, 'tmp.txt')
    with open(txt_file, 'r') as file:
        identifiers = [line.strip() for line in file]

    for test_id in identifiers:
        print(test_id)

        smpl_obj_file = os.path.join(base_dir, f'test_{test_id}_smpl_opt.obj')
        pred_obj_file = os.path.join(base_dir, f'test_{test_id}.obj')

        # Load inputs
        smpl_obj = load_obj(smpl_obj_file)
        device = torch.device(f"cuda:{gpu_device}")

        # Process hands
        right_hand_mesh = process_hand('right_hand.json', smpl_obj, test_id)
        left_hand_mesh = process_hand('left_hand.json', smpl_obj, test_id)
        foots_mesh = process_hand('foots.json', smpl_obj, test_id)
        face_mesh = process_hand('face.json', smpl_obj, test_id)

        # Load prediction and SMPL meshes
        pred_obj_mesh = trimesh.load(pred_obj_file)
        smpl_mesh = trimesh.load(smpl_obj_file)

        hand_mesh = None

        # 判断smpl的左手和右手有没有和身体部分重合，如果没有重合则粘贴在mesh上
        # 右手部分
        right_collision_manager = CollisionManager()
        if process_hand_collision('smpl_no_right_hand.json', smpl_obj, test_id, right_hand_mesh,
                                           right_collision_manager, "right"):
            hand_mesh += right_hand_mesh

        # 左手部分
        left_collision_manager = CollisionManager()
        if process_hand_collision('smpl_no_left_hand.json', smpl_obj, test_id, left_hand_mesh,
                                           left_collision_manager, "left"):
            hand_mesh += left_hand_mesh

        # 网格文件导出路径
        refine_mesh_path = os.path.join(base_dir, f'test_{test_id}_refine.obj')

        # 网格细化与导出
        refined_mesh = refine_and_export_mesh(hand_mesh, pred_obj_mesh, refine_mesh_path, smpl_mesh,
                                              foots_mesh, face_mesh, device)

        # Export refined mesh
        refine_mesh_path = os.path.join(base_dir, f'test_{test_id}_refine.obj')
        refined_mesh.export(refine_mesh_path)


if __name__ == "__main__":
    main()