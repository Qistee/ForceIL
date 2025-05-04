import h5py

# 打开文件
with h5py.File('../data/episode_0.hdf5', 'r') as f:
    # 查看根目录下的所有组
    print("Keys at root:", list(f.keys()))
    data = f['observations/force'][:]
    print(data.shape)
    print(data)
    
    data = f['observations/gripper_pos'][:]
    print(data.shape)
    print(data)
    
    # 递归打印整个结构
    def print_hdf5_structure(name, obj):
        print(name)
    f.visititems(print_hdf5_structure)
