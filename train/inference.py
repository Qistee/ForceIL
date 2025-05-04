from model_util import make_policy, make_optimizer
from camera import image_reader   
import cv2
import numpy as np
import torch
import argparse
from fairino import Robot
from constants import TASK_CONFIGS
import time

def preprocess_frame(img_logitech,img_realsense,arg):
        all_cam_images = [img_logitech,img_realsense]
        all_cam_images = np.stack(all_cam_images, axis=0)
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('h w c -> c h w', image_data)
        image_data = image_data / 255.0
        
        if not arg.randomize_color:
            image_data = image_data[:,[2,1,0],:, :] # BGR to RGB
        else:
            order = np.random.permutation(3)
            image_data = image_data[:,order,:,:]
            
        if arg.grayscale:
            image_data = torch.mean(image_data, dim=1, keepdim=True).repeat(1,3,1,1)
        
        return image_data


task_name='grasp_with_force_feedback'
lr_backbone = 1e-5
parser = argparse.ArgumentParser()    
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=360)
parser.add_argument('--data_aug', action='store_true')
parser.add_argument('--normalize_resnet', action='store_true') ### not used - always normalize - in the model.forward
parser.add_argument('--grayscale', action='store_true')
parser.add_argument('--randomize_color', action='store_true')
parser.add_argument('--randomize_data', action='store_true')
parser.add_argument('--randomize_data_degree', action='store', type=int, default=3)

parser.add_argument('--lr', default=1e-5, action='store', type=float, help='lr')
parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)    
parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
parser.add_argument('--no_encoder', action='store_true')
parser.add_argument('--backbone', type=str, default='resnet18')

args = parser.parse_args()


task_config = TASK_CONFIGS[task_name]
dataset_dir = task_config['dataset_dir']
camera_names = task_config['camera_names']
stats_dir = task_config.get('stats_dir', None)
sample_weights = task_config.get('sample_weights', None)
train_ratio = task_config.get('train_ratio', 0.99)
name_filter = task_config.get('name_filter', lambda n: True)
randomize_index = task_config.get('randomize_index', False)
state_dim = task_config.get('state_dim', 40)
action_dim = task_config.get('action_dim', 40)
state_idx = np.arange(state_dim).tolist()
action_idx = np.arange(action_dim).tolist()
state_mask = task_config.get('state_mask', np.ones(state_dim))
action_mask = task_config.get('action_mask', np.ones(action_dim))

robot = Robot.RPC('192.168.58.2')
ret = robot.SetGripperConfig(6,0)
tool = 0
user = 0
robot.ActGripper(1,1)

policy_class='HIT'
policy_config = {'lr': args['lr'],
                 'hidden_dim': args['hidden_dim'],
                 'dec_layers': args['dec_layers'],
                 'nheads': args['nheads'],
                 'num_queries': args['chunk_size'],
                 'camera_names': camera_names,
                 'action_dim': action_dim,
                 'state_dim': state_dim,
                 'backbone': args['backbone'],
                 'same_backbones': args['same_backbones'],
                 'lr_backbone': lr_backbone,
                 'context_len': 183+args['chunk_size'], #for 224,400
                 'num_queries': args['chunk_size'], 
                 'use_pos_embd_image': args['use_pos_embd_image'],
                 'use_pos_embd_action': args['use_pos_embd_action'],
                 'feature_loss': args['feature_loss_weight']>0,
                 'feature_loss_weight': args['feature_loss_weight'],
                 'self_attention': args['self_attention']==1,
                 'state_idx': state_idx,
                 'action_idx': action_idx,
                 'state_mask': state_mask,
                 'action_mask': action_mask,
                 }

policy = make_policy(policy_class, policy_config)
loading_status = policy.deserialize(torch.load(f'{config["pretrained_path"]}/policy_last.ckpt', map_location='cuda'))
print(f'loaded! {loading_status}')
policy.cuda()
policy.eval()



while(1) : 
        image_reader = image_reader()
        img_logitech, img_realsense = image_reader.read_image()
        image_data=preprocess_frame(img_logitech,img_realsense,args)
        
        qpos_data=robot.GetActualJointPosDegree().joint_pos
        
        image_data, qpos_data = image_data.cuda(), qpos_data.cuda()
        
        a_hat=policy.forward_inf(qpos_data, image_data)
        
        joint_pos=a_hat[:6]
        robot.MoveJ(joint_pos, tool, user, vel=80) 
        time.sleep(0.05)
        
                