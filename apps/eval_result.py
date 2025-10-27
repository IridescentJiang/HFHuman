import trimesh
import os
from evaluate_model import quick_get_chamfer_and_surface_dist
from tqdm import tqdm
from lib.data import TrainDataset
from lib.options import BaseOptions
import numpy as np
import open3d as o3d
from scipy.spatial import distance

parser = BaseOptions()
opt = parser.parse()

num_samples_to_use = 5000
gen_test_counter = 0
gt_mesh_directory = "rendering_script/THuman2.0_Release"
# gt_mesh_directory = "rendering_script/test_mesh/GT_for_ICON"
test_mesh_directory = "rendering_script/test_mesh/cleaned_mesh_Date_08_Sep_24_Time_23_48_56"
result_file_path = '%s/result.txt' % (test_mesh_directory)
result_file = open(result_file_path, 'a')


train_dataset = TrainDataset(opt, projection='orthogonal', phase='train', evaluation_mode=True)
# train_data_loader = DataLoader(train_dataset,
#                                batch_size=opt.batch_size, shuffle=not opt.serial_batches,
#                                num_workers=opt.num_threads, pin_memory=opt.pin_memory)

print('train loader size: ', len(train_dataset))

train_dataset.is_train = False
len_to_iterate = int(len(train_dataset) // 11)

total_chamfer_distance = []
total_point_to_surface_distance = []

for gen_idx in tqdm(range(len_to_iterate)):

    index_to_use = gen_test_counter % len(train_dataset)
    gen_test_counter += 11  # 11 is the number of images for each class

    train_data = train_dataset.get_item(index=index_to_use)
    subject = train_data['name']
    subject = subject.replace('.obj', '')

    # GT_mesh_path = os.path.join(gt_mesh_directory, '%s.obj' % subject)
    GT_mesh_path = os.path.join(gt_mesh_directory, subject, '%s.obj' % subject)
    GT_mesh = trimesh.load(GT_mesh_path)

    test_mesh_path = os.path.join(test_mesh_directory, 'test_%s.obj' % subject)
    source_mesh = trimesh.load(test_mesh_path)

    chamfer_distance, point_to_surface_distance = quick_get_chamfer_and_surface_dist(
        src_mesh=source_mesh, tgt_mesh=GT_mesh, num_samples=num_samples_to_use)
    print('{0} - CD: {1} P2S: {2}'.format(train_data['name'], chamfer_distance,
                                          point_to_surface_distance), file=result_file)
    print('{0} - CD: {1} P2S: {2}'.format(train_data['name'], chamfer_distance,
                                          point_to_surface_distance))
    total_chamfer_distance.append(chamfer_distance)
    total_point_to_surface_distance.append(point_to_surface_distance)

if len(total_chamfer_distance) == 0:
    average_chamfer_distance = 0
else:
    average_chamfer_distance = np.mean(total_chamfer_distance)

if len(total_point_to_surface_distance) == 0:
    average_point_to_surface_distance = 0
else:
    average_point_to_surface_distance = np.mean(total_point_to_surface_distance)

print("[Testing] Overall - Avg CD: {0}; Avg P2S: {1}".format(average_chamfer_distance,
                                                             average_point_to_surface_distance), file=result_file)
print("[Testing] Overall - Avg CD: {0}; Avg P2S: {1}".format(average_chamfer_distance,
                                                             average_point_to_surface_distance))

print("Testing is Done! Exiting...")