import numpy as np

vol_res = 256
semantic_encoding_sigma = 0.005
smooth_kernel_size = 7

cam_f = 5000
img_res = 512
cam_c = img_res/2
cam_R = np.eye(3, dtype=np.float32) * np.array([[1, -1, -1]], dtype=np.float32)
cam_tz = 10.0
cam_t = np.array([[0, 0, cam_tz]], dtype=np.float32)
