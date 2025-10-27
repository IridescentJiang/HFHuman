import numpy as np
import os
import glob
import multiprocessing
import tqdm
import trimesh
import pickle

mesh_dir = './THuman2.0_Release'
smpl_dir = './smpl_x'


def get_data_list():
    """reads data list"""
    data_list = glob.glob(os.path.join(mesh_dir, './*/'))
    return sorted(data_list)


def get_mesh_scale_fname(mesh_folder, smpl_pkl_folder):
    obj_list = glob.glob(os.path.join(mesh_folder, '*.obj'))
    gt_smpl_pkl_path = glob.glob(os.path.join(smpl_pkl_folder, "smplx_param_gt" + ".pkl"))
    with open(gt_smpl_pkl_path[0], 'rb') as f:
        gt_smpl_pkl = pickle.load(f)
    assert len(obj_list)==1, '[ERROR] More than one obj file are found!'
    return obj_list[0], gt_smpl_pkl


def process_one_data_item(data_item):
    _, item_name = os.path.split(data_item[:-1])

    mesh_fd = os.path.join(mesh_dir, item_name)
    smpl_pkl_fd = os.path.join(smpl_dir, item_name)
    obj_fname, gt_smpl_pkl = get_mesh_scale_fname(mesh_fd, smpl_pkl_fd)
    mesh = trimesh.load(obj_fname)
    
    mesh_v = mesh.vertices
    mesh_gt_smpl_scale = gt_smpl_pkl['scale']
    mesh_v /= mesh_gt_smpl_scale

    mesh.vertices = mesh_v
    trimesh.base.export_mesh(mesh, obj_fname)

    for root, dirs, files in os.walk(mesh_fd):
        for file in files:
            if "material0" in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
    print('Processed ' + item_name)


def main(worker_num=4):
    os.makedirs(mesh_dir, exist_ok=True)

    data_list = get_data_list()
    print('Found %d data items' % len(data_list))
    pool = multiprocessing.Pool(processes=worker_num)
    try:
        r = [pool.apply_async(process_one_data_item, args=(data_item,))
             for data_item in data_list]
        pool.close()
        for item in r:
            item.wait(timeout=9999999)
    except KeyboardInterrupt:
        pool.terminate()
    finally:
        pool.join()
        print('Done. ')
    # for data_item in tqdm.tqdm(data_list, ascii=True):
    #     process_one_data_item(data_item)
    #     import pdb
    #     pdb.set_trace()
    print('Done')


if __name__ == '__main__':
    main()
