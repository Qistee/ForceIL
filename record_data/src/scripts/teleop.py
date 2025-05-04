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
from camera import image_reader
from Dynamixel_arm import DynamixelArm

DYNAMIXELARM=DynamixelArm()
robot = Robot.RPC('192.168.58.2')
ret = robot.SetGripperConfig(6,0)
tool = 0
user = 0
robot.ActGripper(1,1)


"""
Dynamixel joints range:
joint1: 0-4095
joint2: 0-4095
joint3: 0-4095
joint4: 0-4095
joint5: 0-4095
joint6: 1085-3214
"""
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
                        'joint_pos': (5,),从机械臂的关节位置，单位为度
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
        return 0

    def get_image(self):
        return 0

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

    def step(self, action,obs):
        # 执行遥操作，并获取observation
        joint_pos = action['joint_pos']
        gripper_pos = action['gripper_pos']
        fairino_joints = Dynamixel2Fairino(joint_pos)
        fairino_joints.append(self.joint_pos[-1])
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
        gripper_joint = Dynamixel2Gripper(gripper_pos)
        max_djoint = np.max(np.array(fairino_joints)-np.array(self.joint_pos))
        max_djoint_index = np.argmax(np.array(fairino_joints)-np.array(self.joint_pos))
        if max_djoint>1000:
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
    action = {'joint_pos': joint_pos, 'gripper_pos': gripper_pos}
    return action


def opening_ceremony():
    #[110,-90,0,-90,-90,-11.8]
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
        obs=env.step(action,obs)
        t2=time.time()
        timesteps.append(obs)
        actions.append(action)
        actual_dt_history.append([t0,t1,t2])
        time.sleep(max(0,dt-(time.time()-t0)))
    print(f"average FPS: {max_timesteps/(time.time()-time0)}")

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
    parser.add_argument('--max_timesteps', type=int, default=1000, help='Maximum number of timesteps to record')
    parser.add_argument('--episode_idx', type=int, default=None, help='Episode index to record')
    parser.add_argument('--dt', type=float, default=1/20, help='Duration of each timestep to control FPS')
    args = parser.parse_args()
    main(args)