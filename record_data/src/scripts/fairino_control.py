#!/home/yangchao/.conda/envs/yolo/bin/python
from fairino import Robot
import rospy
from functools import partial
import time
import numpy as np
from std_msgs.msg import Float32MultiArray, String, Bool
import zmq
import threading
import pickle
    

class FairinoControl:
    def __init__(self):
        self.robot = Robot.RPC('192.168.58.2')
        ret = self.robot.SetGripperConfig(6,0)
        if not ret==0:
            print("SetGripperConfig error")
            return None
        ret=self.robot.ActGripper(1,1)
        print(ret)

        self.joint_pos = [110,-90,0,-90,-90,-11.8]
        self.gripper_pos = 0.0

        self.ctx = zmq.Context()
        # 观测数据发布端口
        self.obs_pub = self.ctx.socket(zmq.PUB)
        self.obs_pub.bind("tcp://*:5556")  
        # 动作数据发布端口 
        self.act_pub = self.ctx.socket(zmq.PUB)
        self.act_pub.bind("tcp://*:5557")  

        #self.robot.GetAxleLuaGripperFunc(1,9)

    def _get_joint_pos(self):
        while not rospy.is_shutdown():
            time.sleep(0.1)
            ret, joint_pos = self.robot.GetActualJointPosDegree(flag=1)
            self.joint_pos = joint_pos
            while np.allclose(joint_pos[:3], 0, atol=1e-1):
                print("Joints are not ready, waiting...")
                ret, joint_pos = self.robot.GetActualJointPosDegree(flag=1)
            self.publish_action()
            self.publish_observation()
               
    def start_read_joint_pos(self):
        """
        开始读取机械臂的关节角度
        """
        self.joint_pos_thread = threading.Thread(target=self._get_joint_pos)
        self.joint_pos_thread.start()

    def stop_read_joint_pos(self):
        """
        停止读取机械臂的关节角度
        """
        self.joint_pos_thread.join()

    def get_observation(self):
        observation = {'joint_pos': self.joint_pos, 'gripper_pos': self.gripper_pos}
        return observation
    
    def get_action(self):
        action = {'joint_pos': self.joint_pos, 'gripper_pos': self.gripper_pos}
        return action

    def publish_observation(self):
        obs = self.get_observation()
        obs_data = pickle.dumps(obs, protocol=pickle.HIGHEST_PROTOCOL)
        self.obs_pub.send(obs_data) 

    def publish_action(self):
        action = self.get_action()
        act_data = pickle.dumps(action, protocol=pickle.HIGHEST_PROTOCOL)
        self.act_pub.send(act_data)

    def move_with_record(self,target_pos):
        self.start_read_joint_pos()
        rospy.wait_for_message('/start_recording', Bool)
        self.robot.MoveJ(target_pos, 0, 0, vel=30)
        rospy.loginfo("Move to target position finished")
        rospy.wait_for_message('/stop_recording', Bool)
        self.stop_read_joint_pos()


    
    def move_home(self):
        """
        移动到初始位置
        """
        self.robot.MoveJ([110,-90,0,-90,-90,-11.8], 0, 0, vel=30)
        rospy.loginfo("Move home finished")

    def move_init(self):
        """
        移动到初始位置
        """
        self.robot.MoveJ([153,-82,60,-97,-104,-11.8], 0, 0, vel=30)
        rospy.loginfo("Move init finished")

if __name__ == '__main__':
    rospy.init_node('fairino_control')
    
    fairino_control = FairinoControl()
    fairino_control.move_init()

