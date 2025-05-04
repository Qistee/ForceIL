import h5py
import numpy as np

# 打开文件
random_action = np.random.rand(201, 7).astype(np.float32)
with h5py.File('./data/episode_0.hdf5', 'r+') as f:
    # # 查看根目录下的所有组
    # if 'action' in f:
    #     print("Found existing 'action' dataset. Deleting and replacing...")
    #     del f['action']
    # f.create_dataset('action', data=random_action)
    print("Keys at root:", list(f.keys()))
    # data = f["/observations/gripper_pos"][()]  # shape: (201,)
    # data = data.reshape(-1, 1)  # shape: (201, 1)
    # for cam in ["image_logitech", "image_realsense"]:
    #     f.copy(f"/observations/image/{cam}", f"/observations/images/{cam}")
    #     del f[f"/observations/image/{cam}"]
    # 删除原有 dataset（HDF5 不支持直接 in-place 修改 shape）
    # del f["/observations/gripper_pos"]

    # 写入新的 dataset
    # f["/observations/gripper_pos"] = data
    # data = f['observations/force'][:]
    # print(data.shape)
    # print(data)
    
    # data = f['observations/gripper_pos'][:]
    # print(data.shape)
    # print(data)
    
    # data = f['action'][:]
    # print(data.shape)
    #print(data)
    # 递归打印整个结构
    def print_hdf5_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
                print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    f.visititems(print_hdf5_structure)
