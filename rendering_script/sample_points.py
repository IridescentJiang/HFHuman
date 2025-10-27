import scipy.io as sio
import trimesh
import numpy as np
from pysdf import SDF
import glob
import os
import multiprocessing
from numpy.linalg import inv

compensation_factor = 4.0
mesh_directory = "rendering_script/THuman2.0_Release"
point_output_fd = 'rendering_script/sample_points_36'
sigma_low_resolution_pifu = 3.5
ratio_of_way_inside_points = 0.05
ratio_of_outside_points = 0.05
num_sample_inout = 80000
sigma = 0.025
sigma_small = 0.01
curv_thresh = 0.004

def get_data_list():
    """reads data list"""
    data_list = glob.glob(os.path.join(mesh_directory, './*/'))
    return sorted(data_list)


def read_data(item):
    """reads data """
    mesh = trimesh.load(os.path.join(mesh_directory, item, '%s.obj' % item))
    return mesh


def process_one_data_item(data_item):
    _, item_name = os.path.split(data_item[:-1])

    mesh = read_data(item_name)
    y_length = np.abs(np.max(mesh.vertices, axis=0)[1]) + np.abs(np.min(mesh.vertices, axis=0)[1])
    sigma_multiplier = y_length / 36

    surface_points, face_indices = trimesh.sample.sample_surface(mesh, int(compensation_factor * 4 * num_sample_inout))

    root = "rendering_script/arounding_dataset/buffer_fixed_full_mesh"
    param_path = os.path.join(root, item_name, "rendered_params_" + "000" + ".npy")
    param = np.load(param_path,
                    allow_pickle=True)  # param is a np.array that looks similar to a dict.  # ortho_ratio = 0.4 , e.g. scale or y_scale = 0.961994278, e.g. center or vmed = [-1.0486  92.56105  1.0101 ]
    load_size_associated_with_scale_factor = 1024
    R = param.item().get('R')
    center = param.item().get(
        'center')  # is camera 3D center position in the 3D World point space (without any rotation being applied).
    scale_factor = param.item().get(
        'scale_factor')  # is camera 3D center position in the 3D World point space (without any rotation being applied).

    b_range = load_size_associated_with_scale_factor / scale_factor  # e.g. 512/scale_factor
    b_center = center
    b_min = b_center - b_range / 2
    b_max = b_center + b_range / 2

    # add random points within image space
    length = b_max - b_min  # has shape of (3,)

    '''
        Sample points without DOS
    '''
    # random_points = np.random.rand(int(compensation_factor * num_sample_inout // 4),
    #                                3) * length + b_min  # shape of [compensation_factor*num_sample_inout/4, 3]
    random_points = np.random.rand(int(compensation_factor * num_sample_inout * 2.5),
                                   3) * length + b_min  # shape of [compensation_factor*num_sample_inout/4, 3]
    surface_points_shape = list(surface_points.shape)

    # random_noise of integratedPIFu
    random_noise = np.random.normal(scale=sigma_low_resolution_pifu * sigma_multiplier,
                                    size=surface_points_shape)

    sample_points_low_res_pifu = surface_points + random_noise  # sample_points are points very near the surface. The sigma represents the std dev of the normal distribution
    sample_points_low_res_pifu = np.concatenate([sample_points_low_res_pifu, random_points],
                                                0)  # shape of [compensation_factor*4.25*num_sample_inout, 3]
    np.random.shuffle(sample_points_low_res_pifu)

    inside_low_res_pifu = mesh.contains \
        (sample_points_low_res_pifu)  # return a boolean 1D array of size (num of sample points,)
    inside_points_low_res_pifu = sample_points_low_res_pifu[inside_low_res_pifu]

    outside_points_low_res_pifu = sample_points_low_res_pifu[np.logical_not(inside_low_res_pifu)]

    '''
        Sample points with DOS
    '''
    num_of_pts_in_section = num_sample_inout // 3

    normal_vectors = mesh.face_normals[face_indices]  # [num_of_sample_pts, 3]

    directional_vector = np.array([[0.0, 0.0, 1.0]])  # 1x3
    directional_vector = np.matmul(inv(R), directional_vector.T)  # 3x1
    # get dot product
    normal_vectors_to_use = normal_vectors[0: num_sample_inout, :]
    dot_product = np.matmul(directional_vector.T, normal_vectors_to_use.T)  # [1 x num_of_sample_pts]

    dot_product[dot_product < 0] = -1.0  # points generated from faces that are facing backwards
    dot_product[dot_product >= 0] = 1.0  # points generated from faces that are facing camera
    z_displacement = np.matmul(dot_product.T,
                               directional_vector.T)  # [num_of_sample_pts, 3]. Will displace points facing backwards to go backwards, but points facing forward to go forward

    normal_sigma = np.random.normal(loc=0.0, scale=1.0,
                                    size=[4 * num_sample_inout, 1])  # shape of [num_of_sample_pts, 1]
    normal_sigma_mask = (normal_sigma[:, 0] < 1.0) & (normal_sigma[:, 0] > -1.0)
    normal_sigma = normal_sigma[normal_sigma_mask, :]
    normal_sigma = normal_sigma[0:num_sample_inout, :]
    surface_points_with_normal_sigma = surface_points[0:num_sample_inout,
                                       :] - z_displacement * sigma_multiplier * normal_sigma * 2.0  # The minus sign means that we are getting points that are all inside the surface, rather than outside of it.
    labels_with_normal_sigma = normal_sigma.T / 2.0 * 0.8  # set range to 0.8. range from -0.4 to 0.4
    labels_with_normal_sigma = labels_with_normal_sigma + 0.5  # range from 0.1 to 0.9 . Shape of [1, num_sample_inout]

    # get way inside points:
    num_of_way_inside_pts = round(num_sample_inout * ratio_of_way_inside_points)
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

    inside_points_low_res_pifu_DOS = np.concatenate([surface_points_with_normal_sigma, way_inside_pts], 0)

    # get way outside points
    num_of_outside_pts = round(num_sample_inout * ratio_of_outside_points)
    outside_surface_points = surface_points[0: num_of_outside_pts] + z_displacement[
                                                                     0:num_of_outside_pts] * sigma_multiplier * (
                                     5.0 + np.random.uniform(low=0.0, high=50.0, size=None))
    proximity = trimesh.proximity.longest_ray(mesh, outside_surface_points, z_displacement[
                                                                            0:num_of_outside_pts])  # shape of [num_of_sample_pts]
    outside_surface_points[
        proximity < (sigma_multiplier * 5.0)] = 0  # remove points that are too near the opposite z direction

    random_points = np.random.rand(int(compensation_factor * num_sample_inout),
                                   3) * length + b_min  # shape of [compensation_factor*num_sample_inout/4, 3]
    outside_surface_points = np.concatenate([outside_surface_points, random_points], 0)
    np.random.shuffle(outside_surface_points)

    os.makedirs(os.path.join(point_output_fd), exist_ok=True)
    os.makedirs(os.path.join(point_output_fd, item_name), exist_ok=True)
    sio.savemat(os.path.join(point_output_fd, item_name, 'samples.mat'),
                {
                    'surface_points_with_normal_sigma': surface_points_with_normal_sigma,
                    'outside_surface_points': outside_surface_points,
                    'way_inside_pts': way_inside_pts,
                    'labels_with_normal_sigma': labels_with_normal_sigma,
                    'inside_points_low_res_pifu': inside_points_low_res_pifu,
                    'outside_points_low_res_pifu': outside_points_low_res_pifu
                })
    print(item_name)


def main(worker_num=20):
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


if __name__ == '__main__':
    main()


