#!/home/yangchao/.conda/envs/yolo/bin/python
from fairino import Robot
import rospy
from dynamixel_msgs.msg import JointState
from functools import partial
import time
import numpy as np
import os
from tqdm import tqdm
import h5py
import argparse
from force import force_recorder
from camera2 import image_reader
from Dynamixel_arm import DynamixelArm

DYNAMIXELARM=DynamixelArm()
FORCE_RECORDER=force_recorder()
IMAGE_READER=image_reader()
robot = Robot.RPC('192.168.58.2')
ret = robot.SetGripperConfig(6,0)
tool = 0
user = 0
robot.ActGripper(1,1)

def Dynamixel2Fairino(joints_positions):
    """
    将dynamixel的关节位置转换为fairino的关节位置
    """
    fairino_joints = []
    fairino_joints.append((joints_positions[0]-1000)/4096*360+110)
    fairino_joints.append((joints_positions[1]-1000)/4096*360-90)
    fairino_joints.append((1000-joints_positions[2])/4096*360)
    fairino_joints.append((joints_positions[3])/4096*360-90)
    fairino_joints.append((1000-joints_positions[4])/4096*360-90)   
    return fairino_joints

def Dynamixel2Gripper(gripper_pos):
    """
    将dynamixel的夹爪位置转换为fairino的夹爪位置
    """
    fairino_gripper=(gripper_pos-700)*100/2100
    fairino_gripper%=100
    return fairino_gripper
class real_env:
    """
    环境设定：
    action_space: {
                    'joint_pos': (6,),主机械臂的关节位置，单位为度
                    'gripper_pos': (1,),主机械臂的夹爪位置, 单位为度
                    }
    observation_space: {
                        'joint_pos': (6,),从机械臂的关节位置，单位为度
                        'gripper_pos': (1,),从机械臂的夹爪位置， 为百分比
                        'force': (16,),力传感器测量的力
                        'image':
                                'image_logitech': (3,384,384),
                                'image_realsense': (3,384,384),
                        }
    """
    def __init__(self):
        self.joint_pos = [104,-56,40,-86,-79,-11.8]
    
    
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
        """
        获取夹爪位置,因无对于API,使用dynamixel夹爪位置代替
        """
        joint_pos = DYNAMIXELARM.read_position()
        gripper_pos = joint_pos[-1]
        return Dynamixel2Gripper(gripper_pos)

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

    def step(self, action):
        # 执行遥操作，并获取observation
        fairino_joints = action['joint_pos']
        gripper_joint = action['gripper_pos']
        if fairino_joints[0]<=-175 or fairino_joints[0]>=175:
            print("joint1 is out of range")
            print("fairino_joints:",fairino_joints)
            fairino_joints[0] = self.joint_pos[0]
        if fairino_joints[1]<=-265 or fairino_joints[1]>=85:
            print("joint2 is out of range")
            print("fairino_joints:",fairino_joints)
            fairino_joints[1] = self.joint_pos[1]
        if fairino_joints[2]<=-150 or fairino_joints[2]>=150:
            print("joint3 is out of range")
            print("fairino_joints:",fairino_joints)
            fairino_joints[2] = self.joint_pos[2]
        if fairino_joints[3]<=-265 or fairino_joints[3]>=85:
            print("joint4 is out of range")
            print("fairino_joints:",fairino_joints)
            fairino_joints[3] = self.joint_pos[3]
        if fairino_joints[4]<=-175 or fairino_joints[4]>=175:
            print("joint5 is out of range")
            print("fairino_joints:",fairino_joints)
            fairino_joints[4] = self.joint_pos[4]
        if fairino_joints[5]<=-175 or fairino_joints[5]>=175:
            print("joint6 is out of range")
            print("fairino_joints:",fairino_joints)
            fairino_joints[5] = self.joint_pos[5]
        max_djoint = np.max(np.array(fairino_joints)-np.array(self.joint_pos))
        max_djoint_index = np.argmax(np.array(fairino_joints)-np.array(self.joint_pos))
        if max_djoint>500:
            print(f"joint{max_djoint_index+1} is moving too fast, max_djoint={max_djoint}")
            return self.get_observation()
        ret = robot.MoveJ(fairino_joints, 0, 0)
        if not ret==0:
            print("MoveJ error")
            print("fairino_joints:",fairino_joints)
            return None
        ret = robot.MoveGripper(1,gripper_joint,48,46,30000,1,0,0,0,0)
        if not ret==0:
            print("MoveGripper error",'error code:',ret)
            print("gripper_joint:",gripper_joint)
            return None
        observation = self.get_observation()
        return observation

def get_action():
    """
    获取主机械臂位姿
    """
    pos = DYNAMIXELARM.read_position()
    joint_pos = pos[:5]
    gripper_pos = pos[-1]
    joint_pos = Dynamixel2Fairino(joint_pos)
    joint_pos.append(-11.8)
    gripper_pos = Dynamixel2Gripper(gripper_pos)
    action = {'joint_pos': joint_pos, 'gripper_pos': gripper_pos}
    return action

def opening_ceremony():
    ret = robot.MoveJ([104,-56,40,-86,-79,-11.8], 0, 0)
    if not ret==0:
        print("MoveJ error")
        return None
    DYNAMIXELARM.torque_on()
    DYNAMIXELARM.move_homepose()
    time.sleep(1)
    DYNAMIXELARM.torque_off()

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
    env=real_env()

    #重置环境
    obs = env.reset()
    timesteps=[obs]
    actions=[]
    actual_dt_history=[]
    time0=time.time()
    # 开始记录数据
    print("Start recording...")
    for t in tqdm(range(max_timesteps)):
        t0=time.time()
        action = get_action()
        t1=time.time()
        obs=env.step(action)
        t2=time.time()
        if t%args.frame_skip==0:
            timesteps.append(obs)
            actions.append(action)
            actual_dt_history.append([t0,t1,t2])
        time.sleep(max(0,dt-(time.time()-t0)))
    print(f"average FPS: {max_timesteps/(time.time()-time0)}")

    """
    对于每一个timestep,保存:
    observatoins:
        - joint_pos: (6,)
        - gripper_pos: (1,)
        - force: (16,)
        - images:
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
        'observations/images/image_logitech':[],
        'observations/images/image_realsense':[],
        'actions/joint_pos':[],
        'actions/gripper_pos':[]
    }

    while actions:
        action = actions.pop(0)
        obs = timesteps.pop(0)
        data_dict['observations/joint_pos'].append(obs['joint_pos'])
        data_dict['observations/gripper_pos'].append([obs['gripper_pos']])
        data_dict['observations/force'].append(obs['force'])
        data_dict['observations/images/image_logitech'].append(obs['image']['image_logitech'])
        data_dict['observations/images/image_realsense'].append(obs['image']['image_realsense'])
        data_dict['actions/joint_pos'].append(action['joint_pos'])
        data_dict['actions/gripper_pos'].append([action['gripper_pos']])
    
    # 保存数据
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        obs_group = root.create_group('observations')
        action_group = root.create_group('actions')
        image_group = obs_group.create_group('images')

        _=obs_group.create_dataset('joint_pos', data=np.array(data_dict['observations/joint_pos']))
        _=obs_group.create_dataset('gripper_pos', data=np.array(data_dict['observations/gripper_pos']))
        _=obs_group.create_dataset('force', data=np.array(data_dict['observations/force']))
        _=image_group.create_dataset('image_logitech', data=np.array(data_dict['observations/images/image_logitech']))
        _=image_group.create_dataset('image_realsense', data=np.array(data_dict['observations/images/image_realsense']))

        _=action_group.create_dataset('joint_pos', data=np.array(data_dict['actions/joint_pos']))
        _=action_group.create_dataset('gripper_pos', data=np.array(data_dict['actions/gripper_pos']))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./data', help='Directory to save dataset')
    parser.add_argument('--max_timesteps', type=int, default=50, help='Maximum number of timesteps to record')
    parser.add_argument('--episode_idx', type=int, default=1, help='Episode index to record')
    parser.add_argument('--dt', type=float, default=1/20, help='Duration of each timestep to control FPS')
    parser.add_argument('--frame_skip', type=int, default=1, help='Number of frames to skip')
    args = parser.parse_args()
    main(args)