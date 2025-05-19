#!/home/yangchao/.conda/envs/yolo/bin/python
from fairino import Robot
import rospy
from std_msgs.msg import Bool,Float32MultiArray
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
import zmq
import pickle

class DataGrabber:
    def __init__(self):
        self.force_recorder = force_recorder()
        self.image_reader = image_reader()

        self.sub_ros = rospy.Subscriber('/dynamixel_pos', Float32MultiArray, self.joint_state_callback, queue_size=1)
        self.start_pub = rospy.Publisher('/start_recording', Bool, queue_size=1)
        self.stop_pub = rospy.Publisher('/stop_recording', Bool, queue_size=1)
    
    def joint_state_callback(self, msg):
        self.joint_pos = msg.data[:6]
        self.gripper_pos = msg.data[-1]
        #self.gripper_pos = 1 if self.gripper_pos > 30 else 0


    def get_force(self):
        force = self.force_recorder.read_data()
        return force

    def get_image(self):
        image_logitech, image_realsense = self.image_reader.read_image()
        #print(image_logitech.shape, image_realsense.shape)
        image_logitech = np.transpose(image_logitech, (2, 0, 1))
        image_realsense = np.transpose(image_realsense, (2, 0, 1))
        return {'image_logitech': image_logitech, 'image_realsense': image_realsense}
    
    def get_observation(self):
        
        #print(self.joint_pos, self.gripper_pos)
        force = self.get_force()
        image = self.get_image()
        return {'joint_pos': self.joint_pos, 'gripper_pos': self.gripper_pos, 'force': force, 'image': image}

    def get_action(self):
        return {'joint_pos': self.joint_pos, 'gripper_pos': self.gripper_pos}


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
    
    #opening_ceremony()
    data_grabber = DataGrabber()

    timesteps=[]
    actions=[]
    actual_dt_history=[]
    time0=time.time()
    # 开始记录数据
    time.sleep(1)
    print("Start recording...")
    data_grabber.start_pub.publish(True)
    for t in tqdm(range(max_timesteps)):
        t0=time.time()
        action = data_grabber.get_action()
        t1=time.time()
        obs=data_grabber.get_observation()
        t2=time.time()
        if t%args.frame_skip==0:
            timesteps.append(obs)
            actions.append(action)
            actual_dt_history.append([t0,t1,t2])
        time.sleep(max(0,dt-(time.time()-t0)))
    data_grabber.stop_pub.publish(True)
    print(f"average FPS: {max_timesteps/(time.time()-time0)}")

    """
    对于每一个timestep,保存:
    observatoins:
        - joint_pos: (6,)
        - gripper_pos: (1,)
        - force: (16,)
        - images:
                'image_logitech': (3,384,384),
                #'image_realsense': (3,384,384),
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
    rospy.init_node('record_data_node', anonymous=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./data', help='Directory to save dataset')
    parser.add_argument('--max_timesteps', type=int, default=100, help='Maximum number of timesteps to record')
    parser.add_argument('--episode_idx', type=int, default=None, help='Episode index to record')
    parser.add_argument('--dt', type=float, default=1/20, help='Duration of each timestep to control FPS')
    parser.add_argument('--frame_skip', type=int, default=1, help='Number of frames to skip')
    args = parser.parse_args()
    main(args)
    