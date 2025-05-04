#!/home/yangchao/.conda/envs/yolo/bin/python
import time, pickle, zmq
from Dynamixel_arm import DynamixelArm
from multiprocessing import shared_memory
from fairino import Robot
import numpy as np
from collections import deque
import threading

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
        self.gripper_joint=0
        self.action_joint_pos=[104,-56,40,-86,-79,-11.8]
        self.action_gripper_pos=0

        self.ctx = zmq.Context()
        # 观测数据发布端口
        self.obs_pub = self.ctx.socket(zmq.PUB)
        self.obs_pub.bind("tcp://*:5556")  
        # 动作数据发布端口 
        self.act_pub = self.ctx.socket(zmq.PUB)
        self.act_pub.bind("tcp://*:5557")  

    def step(self, action):
        # 执行遥操作，并获取observation
        fairino_joints = action['joint_pos']
        gripper_joint = action['gripper_pos']
        self.action_joint_pos=fairino_joints
        self.action_gripper_pos=gripper_joint
        # t1=time.time()
        # t3=time.time()
        # 定义每个关节的范围限制 (min, max)
        joint_limits = [
            (-175, 175),    # joint1
            (-110, 85),     # joint2
            (-150, 150),    # joint3
            (-265, -10),     # joint4
            (-175, 175),    # joint5
            (-175, 175)     # joint6
        ]

        for i in range(len(fairino_joints)):
            min_limit, max_limit = joint_limits[i]
            if fairino_joints[i] <= min_limit or fairino_joints[i] >= max_limit:
                print(f"joint{i+1} is out of range")
                print("fairino_joints:", fairino_joints)
                fairino_joints[i] = self.joint_pos[i]
        # t4=time.time()
        # print("if time:",t4-t3)
        max_djoint = np.max(np.array(fairino_joints)-np.array(self.joint_pos))
        max_djoint_index = np.argmax(np.array(fairino_joints)-np.array(self.joint_pos))
        if max_djoint>5:
            print(f"joint{max_djoint_index+1} is moving too fast, max_djoint={max_djoint}")
        # t5=time.time()
        # print("max time:",t5-t4)
        # t1=time.time()
        print("fairino_joints:",fairino_joints)
        ret = robot.MoveJ(fairino_joints, 0, 0,vel=80, blendT=0)
        # t2=time.time()
        # print(t2-t1)
        if not ret==0:
            print("MoveJ error",'error code:',ret)
            print("fairino_joints:",fairino_joints)
            return None
        # t6=time.time()
        # gripper_dis=gripper_joint-self.gripper_joint
        # if(gripper_dis>5):
        #     ret = robot.MoveGripper(1,gripper_joint,100,80,2000,1,0,0,0,0)
        #     t7=time.time()
        #     print("grasp time:",t7-t6)
        #     if not ret==0:
        #         print("MoveGripper error",'error code:',ret)
        #         print("gripper_joint:",gripper_joint)
        #         return None
        # self.gripper_joint=gripper_joint
        self.joint_pos=fairino_joints

    def get_joint_pos(self):
        return self.joint_pos

    def get_gripper_pos(self):
        return self.gripper_joint
    
    def get_observation(self):
        joint_pos = self.get_joint_pos()
        gripper_pos = self.get_gripper_pos()
        observation = {'joint_pos': joint_pos, 'gripper_pos': gripper_pos}
        return observation
    
    def get_action(self):
        action = {'joint_pos': self.action_joint_pos, 'gripper_pos': self.action_gripper_pos}
        return action
    
    def publish_observation(self):
        obs = self.get_observation()
        obs_data = pickle.dumps(obs, protocol=pickle.HIGHEST_PROTOCOL)
        self.obs_pub.send(obs_data) 

    def publish_action(self):
        action = self.get_action()
        act_data = pickle.dumps(action, protocol=pickle.HIGHEST_PROTOCOL)
        self.act_pub.send(act_data)

class CommandBuffer:
    def __init__(self, maxlen=3):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()
    
    def add_command(self, cmd):
        with self.lock:
            self.buffer.append(cmd)
    
    def get_latest(self):
        with self.lock:
            return self.buffer[-1] if self.buffer else None

class RealParallelController:
    def __init__(self):
        self.cmd_buffer = CommandBuffer()
        self.running = True

    def receiver_thread(self):
        while self.running:
            msg = sub.recv()
            try:
                raw_pos = pickle.loads(msg)  # 保持使用pickle反序列化
                action = {
                    'joint_pos': raw_pos[:6], 
                    'gripper_pos': raw_pos[-1]
                }
                self.cmd_buffer.add_command(action)
            except Exception as e:
                print(f"反序列化错误: {e}")
                
    def control_thread(self):
        while self.running:
            t1=time.time()
            latest_cmd = self.cmd_buffer.get_latest()
            if latest_cmd:
                env.step(latest_cmd)
                # env.publish_observation()
                # env.publish_action()   
            time.sleep(0.1)  # 控制频
            t2=time.time() 
            print("step time:",t2-t1)

    def publish_thread(self):
        while self.running:
            env.publish_observation()
            env.publish_action()
            print("pub_finish")   

    def start(self):
        # 启动双线程
        threading.Thread(target=self.receiver_thread, daemon=True).start()
        threading.Thread(target=self.control_thread, daemon=True).start()
        #threading.Thread(target=self.publish_thread, daemon=True).start()


robot = Robot.RPC('192.168.58.2')
ret = robot.SetGripperConfig(6,0)
tool = 0
user = 0
robot.ActGripper(1,1)

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://localhost:5555")
sub.setsockopt_string(zmq.SUBSCRIBE, "")
# 改为使用原始字节数组
#sub.setsockopt(zmq.CONFLATE, 1)  # 只保留最新消息
env=real_env()
t=0
controller = RealParallelController()
controller.start()
while True:
    time.sleep(0.1)
# while True:
#     msg = sub.recv()
#     # print("massage:",msg)
#     # raw_pos = pickle.loads(msg)
#     # #print("raw_pos:",raw_pos)
#     # joint_pos=raw_pos[:6]
#     raw_pos = np.frombuffer(msg, dtype=np.float32)
#     action={'joint_pos': raw_pos[:6], 'gripper_pos': raw_pos[-1]}
#     cmd_buffer.add_command(action)
#     # if(t%10== 0):
#     #     env.step(action)
#     # t+=1
#     # time.sleep(0.01)  # 20 Hz0

# while True:
#     latest_cmd = cmd_buffer.get_latest()
#     if latest_cmd:
#         env.step(latest_cmd)
#     time.sleep(0.01) 