#!/home/yangchao/.conda/envs/yolo/bin/python
import h5py
import numpy as np
import matplotlib.pyplot as plt  # 用于图像可视化
import cv2  

def read_hdf5_dataset(file_path):
    """
    读取HDF5数据集的完整示例
    :param file_path: .hdf5文件路径
    :return: 包含所有数据的字典
    """
    data_dict = {}
    
    try:
        with h5py.File(file_path, 'r') as hdf_file:
            # 读取观测数据
            obs_group = hdf_file['observations']
            
            # 读取关节位置数据 (N, 6)
            data_dict['joint_pos'] = np.array(obs_group['joint_pos'])
            print(f"关节位置数据形状: {data_dict['joint_pos'].shape}")
            
            # 读取夹爪位置数据 (N, 1)
            data_dict['gripper_pos'] = np.array(obs_group['gripper_pos'])
            print(f"夹爪位置数据形状: {data_dict['gripper_pos'].shape}")
            
            # 读取力传感器数据 (N, 16)
            data_dict['force'] = np.array(obs_group['force'])
            print(f"力传感器数据形状: {data_dict['force'].shape}")
            
            # 读取图像数据
            image_group = obs_group['images']
            # 罗技相机数据 (N, 3, 384, 384)
            data_dict['logitech_images'] = np.array(image_group['image_logitech'])
            # RealSense相机数据 (N, 3, 384, 384)
            data_dict['realsense_images'] = np.array(image_group['image_realsense'])
            print(f"罗技图像数据形状: {data_dict['logitech_images'].shape}")
            print(f"RealSense图像数据形状: {data_dict['realsense_images'].shape}")

            # 如果需要读取动作数据（当前被注释）
            if 'actions' in hdf_file:
                action_group = hdf_file['actions']
                # 添加动作数据读取逻辑...

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
    except KeyError as e:
        print(f"数据格式错误，缺少关键字段: {str(e)}")
    
    return data_dict

def play_camera_data(dataset, camera_type='logitech', frame_delay=25):
    """
    使用OpenCV播放摄像头数据
    :param dataset: 数据集字典（包含logitech_images/realsense_images）
    :param camera_type: 摄像头类型，'logitech' 或 'realsense'
    :param frame_delay: 帧间延迟（毫秒），控制播放速度
    """
    # 获取对应的图像数据
    key = f'{camera_type}_images'
    if key not in dataset:
        print(f"错误：数据集中不存在 {key} 数据")
        return
    
    images = dataset[key]
    
    # 转换数据维度：从 (N, C, H, W) 到 (N, H, W, C)
    images = images.transpose(0, 2, 3, 1)
    
    # 检查数据类型（如果是float且范围0-1，转换为uint8）
    if images.dtype == np.float32 or images.dtype == np.float64:
        images = (images * 255).astype(np.uint8)
    
    # 检查颜色通道是否需要转换（OpenCV使用BGR格式）
    for i in range(len(images)):
        # 获取当前帧
        frame = images[i]
        
        # 转换颜色空间 RGB -> BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 显示图像
        cv2.imshow(f'Camera View - {camera_type}', frame)
        
        # 检查退出按键（按q键退出或关闭窗口）
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

# 使用示例 ---------------------------------------------------
if __name__ == '__main__':
    dataset = read_hdf5_dataset('data/episode_0.hdf5')
    
    joint_pos = dataset['joint_pos']
    gripper_pos = dataset['gripper_pos']

    joint_pos = joint_pos.transpose(1, 0)

    for i in range(joint_pos.shape[0]):
        plt.plot(joint_pos[i, :])
    plt.plot(gripper_pos, label='gripper_pos')
    plt.legend()
    plt.show()

    # 显示摄像头数据
    play_camera_data(dataset, 'logitech')
    play_camera_data(dataset,'realsense')
