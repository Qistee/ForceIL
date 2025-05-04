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

        self.pub_ros = rospy.Publisher('/dynamixel_pos', Float32MultiArray, queue_size=10)

        #self.robot.GetAxleLuaGripperFunc(1,9)

    def _get_joint_pos(self):
        while not rospy.is_shutdown():
            time.sleep(0.05)
            ret, joint_pos = self.robot.GetActualJointPosDegree(flag=1)
            self.joint_pos = joint_pos
            print(self.joint_pos)
            while np.allclose(joint_pos[:3], 0, atol=1e-1):
                print("Joints are not ready, waiting...")
                ret, joint_pos = self.robot.GetActualJointPosDegree(flag=1)

            msg = Float32MultiArray(data=self.joint_pos+[self.gripper_pos])
            self.pub_ros.publish(msg)

               
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


    def move_with_record(self,target_pos):
        self.move_init()
        self.start_read_joint_pos()
        rospy.wait_for_message('/start_recording', Bool)
        self.robot.MoveJ(target_pos, 0, 0, vel=5)
        rospy.loginfo("Move to target position finished")
        rospy.wait_for_message('/stop_recording', Bool)
        rospy.loginfo("Recording finished")
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
        self.robot.MoveGripper(1,0,100,80,2000,1,0,0,0,0)
        rospy.loginfo("Move init finished")

if __name__ == '__main__':
    rospy.init_node('fairino_control')
    
    fairino_control = FairinoControl()
    fairino_control.move_home()
    #fairino_control.move_with_record([131.005859375, -58.359375, 63.896484375, -95.830078125, -96.15234375, -11.8])

