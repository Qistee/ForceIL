#!/home/yangchao/.conda/envs/fairno/bin/python
import rospy
from dynamixel_msgs.msg import JointState
from functools import partial
import time
import numpy as np

class JointState_Reader():
    """
    读取dynamixel关节状态的类
    """
    def __init__(self):
        rospy.on_shutdown(self.clean_up)
        self.joint_subscribers()
        self.joints_positions=[0,0,0,0,0,0,0]

    def joint_subscribers(self):
        self.sub_joint1 = rospy.Subscriber('/joint1_controller/state', JointState, partial(self.jointCallback, index=0),queue_size=1)
        self.sub_joint2 = rospy.Subscriber('/joint2_controller/state', JointState, partial(self.jointCallback, index=1),queue_size=1)
        self.sub_joint3 = rospy.Subscriber('/joint3_controller/state', JointState, partial(self.jointCallback, index=2),queue_size=1)
        self.sub_joint4 = rospy.Subscriber('/joint4_controller/state', JointState, partial(self.jointCallback, index=3),queue_size=1)
        self.sub_joint5 = rospy.Subscriber('/joint5_controller/state', JointState, partial(self.jointCallback, index=4),queue_size=1)
        self.sub_joint6 = rospy.Subscriber('/joint6_controller/state', JointState, partial(self.jointCallback, index=5),queue_size=1)
        self.sub_joint7 = rospy.Subscriber('/joint7_controller/state', JointState, partial(self.jointCallback, index=6),queue_size=1)

    def read_joints(self):
        #rospy.loginfo(self.joints_positions)
        return self.joints_positions[:5], self.joints_positions[6]

    def jointCallback(self, msg, index):
        self.joints_positions[index] = msg.current_pos
        self.read_joints()  # 在每次接收到消息后调用读取方法

    def clean_up(self):
        rospy.loginfo("Shutting down")
        return

if __name__ == '__main__':
    rospy.init_node('jointstate_reader', anonymous=True)
    joint_reader = JointState_Reader()
    rate = rospy.Rate(10)  # 10hz