#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from dynamixel_msgs.msg import JointState
from typing import List

class ArmController():
    def __init__(self) -> None:
        # 初始化舵机状态变量
        self.joint_positions = [0.0] * 7
        self.joint_moving = [False] * 7
        self.joint_targets = [-1] * 7
        self.joint_errors = [0.0] * 7
        self.joint_load = [0.0] * 7

        # 初始化舵机控制话题发布者
        self.joint_names = ['joint1_controller', 'joint2_controller', 'joint3_controller', 'joint4_controller', 'joint5_controller','joint6_controller','joint7_controller']
        
        self.pub_joint1 = rospy.Publisher(f'/{self.joint_names[0]}/command', Float64, queue_size=1)
        self.pub_joint2 = rospy.Publisher(f'/{self.joint_names[1]}/command', Float64, queue_size=1)
        self.pub_joint3 = rospy.Publisher(f'/{self.joint_names[2]}/command', Float64, queue_size=1)
        self.pub_joint4 = rospy.Publisher(f'/{self.joint_names[3]}/command', Float64, queue_size=1)
        self.pub_joint5 = rospy.Publisher(f'/{self.joint_names[4]}/command', Float64, queue_size=1)
        self.pub_joint6 = rospy.Publisher(f'/{self.joint_names[5]}/command', Float64, queue_size=1)
        self.pub_joint7 = rospy.Publisher(f'/{self.joint_names[6]}/command', Float64, queue_size=1)
        self.pub_joints = [self.pub_joint1, self.pub_joint2, self.pub_joint3, self.pub_joint4, self.pub_joint5, self.pub_joint6, self.pub_joint7]

        # 初始化舵机位置话题订阅者
        self.sub_joint1 = rospy.Subscriber(f'/{self.joint_names[0]}/state', JointState, self.jointCallback, 0)
        self.sub_joint2 = rospy.Subscriber(f'/{self.joint_names[1]}/state', JointState, self.jointCallback, 1)
        self.sub_joint3 = rospy.Subscriber(f'/{self.joint_names[2]}/state', JointState, self.jointCallback, 2)
        self.sub_joint4 = rospy.Subscriber(f'/{self.joint_names[3]}/state', JointState, self.jointCallback, 3)
        self.sub_joint5 = rospy.Subscriber(f'/{self.joint_names[4]}/state', JointState, self.jointCallback, 4)
        self.sub_joint6 = rospy.Subscriber(f'/{self.joint_names[5]}/state', JointState, self.jointCallback, 5)
        self.sub_joint7 = rospy.Subscriber(f'/{self.joint_names[6]}/state', JointState, self.jointCallback, 6)

    def arm_init(self):
        self.moveSingleJoint(1, 0.3323, 1, False)
        self.moveSingleJoint(0, 4.1775, 1, True)
        self.moveMultiJoints([None, 0.3681, 3.2878, None, None, None, None], 5)
        self.moveSingleJoint(3, 2.1798, 1, False)
        self.moveMultiJoints([None, 0.3272, 5.1695, 1.1521, None, None, None], True)

    def jointCallback(self, data: JointState, index: int):
        self.joint_positions[index] = data.current_pos
        self.joint_moving[index] = data.is_moving
        self.joint_errors[index] = data.error
        self.joint_targets[index] = data.goal_pos
        self.joint_load[index] = data.load
    
    def moveSingleJoint(self, index:int, target:float, steps=20, blocking=True):
        start = self.joint_positions[index]
        for i in range(1, steps + 1):
            self.pub_joints[index].publish(start + (target - start) * i / steps)
            rospy.sleep(0.01)
        if blocking:
            self.waitForJoint(index)
        
    def moveMultiJoints(self, targets, steps=20, blocking=True):
        indexes = [ i for i in range(len(targets)) if targets[i] is not None]
        starts = self.joint_positions.copy()
        for i in range(1, steps + 1):
            for index in indexes:
                self.pub_joints[index].publish(starts[index] + (targets[index] - starts[index]) * i / steps)
            rospy.sleep(0.01)
        if blocking:
            for index in indexes:
                self.waitForJoint(index)

    def waitForJoint(self, joint_index:int):
        self.joint_moving[joint_index] = True
        while self.joint_moving[joint_index] or abs(self.joint_errors[joint_index])>0.5 :
            rospy.sleep(0.01)

    def closeHandByLoad(self, load=0.05):
        state : JointState = rospy.wait_for_message(f'/{self.joint_names[6]}/state', JointState)

        while state.load < load:
            self.pub_joint5.publish(state.goal_pos - 0.01)
            self.waitForJoint(6)
            state : JointState = rospy.wait_for_message(f'/{self.joint_names[6]}/state', JointState)

        self.hold_target = state.goal_pos + 0.01
    
    