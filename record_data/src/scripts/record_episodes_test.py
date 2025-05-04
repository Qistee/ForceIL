#!/home/yangchao/.conda/envs/yolo/bin/python
from fairino import Robot
import rospy
from dynamixel_msgs.msg import JointState
from functools import partial
import time
import numpy as np
import os
from jointstate_reader import JointState_Reader
from tqdm import tqdm
import h5py
import argparse
from std_msgs.msg import String
from force import force_recorder
from camera import image_reader

JOINTSTATE_READER=JointState_Reader()
FORCE_RECORDER=force_recorder()
IMAGE_READER=image_reader()
robot = Robot.RPC('192.168.58.2')
ret = robot.SetGripperConfig(6,0)
tool = 0
user = 0

def Dynamixel2Fairino(joints_positions):
    """
    将dynamixel的关节位置转换为fairino的关节位置
    """
    fairino_joints=joints_positions*180/np.pi
    return fairino_joints

def Dynamixel2Gripper(gripper_pos):
    """
    将dynamixel的夹爪位置转换为fairino的夹爪位置
    """
    fairino_gripper=gripper_pos*60/np.pi
    return fairino_gripper

class real_env:
    """
    环境设定：
    action_space: {
                    'joint_pos': (6,),主机械臂的关节位置，单位为弧度
                    'gripper_pos': (1,),主机械臂的夹爪位置, 单位为弧度
                    }
    observation_space: {
                        'joint_pos': (6,),从机械臂的关节位置，单位为度
                        'gripper_pos': (1,),从机械臂的夹爪位置， 为百分比
                        'force': (16,),力传感器测量的力
                        'image':
                        }
    """
    def __init__(self):
        pass

    def get_joint_pos(self):
        ret, joint_pos = robot.GetActualJointPosDegree(flag=1)
        if not ret==0:
            print("GetActualJointPosDegree error")
            return None
        self.joint_pos = joint_pos
        while np.allclose(joint_pos[:3], 0, atol=1e-1):
            print("Joints are not ready, waiting...")
            ret, joint_pos = robot.GetActualJointPosDegree(flag=1)
            if not ret==0:
                print("GetActualJointPosDegree error")
                return None
        return joint_pos
    
    def get_gripper_pos(self):
        gripper_pos = 0
        return gripper_pos

    def get_force(self):
        force = FORCE_RECORDER.read_data()
        return force

    def get_image(self):
        image_logitech, image_realsense = IMAGE_READER.read_image()
        image_logitech = np.transpose(image_logitech, (2, 0, 1))
        image_realsense = np.transpose(image_realsense, (2, 0, 1))
        return {'image_logitech': image_logitech, 'image_realsense': image_realsense}

    def get_observation(self):
        joint_pos = self.get_joint_pos()
        gripper_pos = self.get_gripper_pos()
        force = self.get_force()
        image = self.get_image()
        observation = {'joint_pos': joint_pos, 'gripper_pos': gripper_pos, 'force': force, 'image': image}
        return observation

    def reset(self):
        # 重置环境
        return self.get_observation()

    def step(self,joint_pos):
        fairino_joints = joint_pos
        ret = robot.MoveJ(fairino_joints, 0, 0,vel=100,blendT=-1.0)
        if not ret==0:
            print("MoveJ error")
            return None
        observation = self.get_observation()
        return observation

def get_action():
    """
    获取主机械臂位姿
    """
    joint_pos,gripper_pos = JOINTSTATE_READER.read_joints()
    action = {'joint_pos': joint_pos, 'gripper_pos': gripper_pos}
    return action


def opening_ceremony():
    pass

def capture_one_episode(dt,max_timesteps,dataset_dir,dataset_name,overwrite=False):
    print("Dataset directory: ",dataset_dir)
    print("Dataset name: ",dataset_name)
    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()
    
    opening_ceremony()

    
    print("Waiting for robot to connect...")
    time.sleep(10)
    env=real_env()

    #重置环境
    obs = env.reset()
    joint_pos = obs['joint_pos']
    timesteps=[obs]
    #actions=[]
    actual_dt_history=[]
    time0=time.time()
    #rospy.wait_for_message('/start_record', String)

    # 开始记录数据
    print("Start recording...")
    for t in tqdm(range(max_timesteps)):
        t0=time.time()
        #action = get_action()
        t1=time.time()
        joint_pos = [joint_pos[x]-0.1 if x==0 or x==1 else joint_pos[x] for x in range(6)]
        obs=env.step(joint_pos)
        t2=time.time()
        timesteps.append(obs)
        #actions.append(action)
        actual_dt_history.append([t0,t1,t2])
        time.sleep(max(0,dt-(time.time()-t0)))
    print(f"average FPS: {max_timesteps/(time.time()-time0)}")

    

    """
    对于每一个timestep,保存:
    observatoins:
        - joint_pos: (6,)
        - gripper_pos: (1,)
        - force: (16,)
        - image:
                'image_logitech': (3,384,384),
                'image_realsense': (3,384,384),
    actions:
        - joint_pos: (6,)
        - gripper_pos: (1,)
    """
    data_dict = {
        'observations/joint_pos':[],
        'observations/gripper_pos':[],
        'observations/force':[],
        'observations/image/image_logitech':[],
        'observations/image/image_realsense':[],
        'actions/joint_pos':[],
        'actions/gripper_pos':[]
    }

    while timesteps:
        #action = actions.pop(0)
        obs = timesteps.pop(0)
        data_dict['observations/joint_pos'].append(obs['joint_pos'])
        data_dict['observations/gripper_pos'].append(obs['gripper_pos'])
        data_dict['observations/force'].append(obs['force'])
        data_dict['observations/image/image_logitech'].append(obs['image']['image_logitech'])
        data_dict['observations/image/image_realsense'].append(obs['image']['image_realsense'])
        #data_dict['actions/joint_pos'].append(action['joint_pos'])
        #data_dict['actions/gripper_pos'].append(action['gripper_pos'])
    
    # 保存数据
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        obs_group = root.create_group('observations')
        #action_group = root.create_group('actions')
        image_group = obs_group.create_group('image')

        _=obs_group.create_dataset('joint_pos', data=np.array(data_dict['observations/joint_pos']))
        _=obs_group.create_dataset('gripper_pos', data=np.array(data_dict['observations/gripper_pos']))
        _=obs_group.create_dataset('force', data=np.array(data_dict['observations/force']))
        _=image_group.create_dataset('image_logitech', data=np.array(data_dict['observations/image/image_logitech']))
        _=image_group.create_dataset('image_realsense', data=np.array(data_dict['observations/image/image_realsense']))

        #=action_group.create_dataset('joint_pos', data=np.array(data_dict['actions/joint_pos']))
        #_=action_group.create_dataset('gripper_pos', data=np.array(data_dict['actions/gripper_pos']))
    print(f"Dataset saved at \n{dataset_path}.hdf5")
    return True

def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

def main(args):
    dataset_dir = args.dataset_dir
    max_timesteps = args.max_timesteps
    
    if args.episode_idx is not None:
        episode_idx = args.episode_idx
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    capture_one_episode(args.dt, max_timesteps, dataset_dir, dataset_name, overwrite)

if __name__ == '__main__':
    rospy.init_node('record_data')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./data', help='Directory to save dataset')
    parser.add_argument('--max_timesteps', type=int, default=100, help='Maximum number of timesteps to record')
    parser.add_argument('--episode_idx', type=int, default=0, help='Episode index to record')
    parser.add_argument('--dt', type=float, default=1/10, help='Duration of each timestep to control FPS')
    args = parser.parse_args()
    main(args)