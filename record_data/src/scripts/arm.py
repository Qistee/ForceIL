#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os, sys, rospy
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import Twist
from dynamixel_msgs.msg import JointState

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from arm_controller import ArmController

class Manipulation:  # 定义了一个名为Manipulation的类
    def __init__(self):
        """
        Manipulation 类的构造函数
        """
        rospy.on_shutdown(self.clean_up)  # 注册关机回调函数,当ROS节点关闭时，这个方法会被调用
        # 初始化一个包含7个false值的列表，用于跟踪7个关节是否移动
        self.joint_moving = [False] * 7

        # 定义了7个舵机的发布者，用于向ROS中的特定话题发布Float64类型的数据
        self.pub_joint1 = rospy.Publisher('/joint1_controller/command', Float64, queue_size=1)
        self.pub_joint2 = rospy.Publisher('/joint2_controller/command', Float64, queue_size=1)
        self.pub_joint3 = rospy.Publisher('/joint3_controller/command', Float64, queue_size=1)
        self.pub_joint4 = rospy.Publisher('/joint4_controller/command', Float64, queue_size=1)
        self.pub_joint5 = rospy.Publisher('/joint5_controller/command', Float64, queue_size=1)
        self.pub_joint6 = rospy.Publisher('/joint6_controller/command', Float64, queue_size=1)
        self.pub_joint7 = rospy.Publisher('/joint7_controller/command', Float64, queue_size=1)
        # 定义了7个订阅者，每个订阅者监听一个关节的话题，用于接收JointState类型的数据 

        self.sub_joint1 = rospy.Subscriber('/joint1_controller/state', JointState, self.jointCallback, 0)
        self.sub_joint2 = rospy.Subscriber('/joint2_controller/state', JointState, self.jointCallback, 1)
        self.sub_joint3 = rospy.Subscriber('/joint3_controller/state', JointState, self.jointCallback, 2)
        self.sub_joint4 = rospy.Subscriber('/joint4_controller/state', JointState, self.jointCallback, 3)
        self.sub_joint5 = rospy.Subscriber('/joint5_controller/state', JointState, self.jointCallback, 4)
        self.sub_joint6 = rospy.Subscriber('/joint6_controller/state', JointState, self.jointCallback, 5)
        self.sub_joint7 = rospy.Subscriber('/joint7_controller/state', JointState, self.jointCallback, 6)

        self.ext_controller = ArmController() 

    def jointCallback(self, data:JointState, joint_index:int):
        """
        跟踪了所有关节的移动状态
        """
        self.joint_moving[joint_index] = data.is_moving

    @property # 将函数变为属性 直接输出属性 对于无参数的函数，可以直接使用属性的形式调用
    def is_done(self):
        """
        检查是否有关节正在移动，没有关节移动，则返回true
        """
        for is_moving in self.joint_moving:
            if is_moving:
                return False
        return True
    
    def wait_for_done(self):
        """
        用于等待直到所有关节都完成移动，否则一直循环等待
        """
        rospy.sleep(0.5)
        while not self.is_done:
            rospy.sleep(0.1)

    def clean_up(self):
        """
        在节点关闭时发布执行完毕的消息
        """
        rospy.loginfo('Shutting down robot arm.')


if __name__ == '__main__':
    rospy.init_node('manipulator', anonymous=False)
    # 创建一个Manipulation类的实例
    manipulation = Manipulation()
    manipulation.arm_give()
    rospy.spin()
