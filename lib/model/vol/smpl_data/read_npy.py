import numpy as np

# 替换为您的文件路径
file_path = "./tetrahedrons.npy"

# 设置NumPy的打印选项
np.set_printoptions(threshold=np.inf)

data = np.load(file_path, allow_pickle=True)
print(data)

