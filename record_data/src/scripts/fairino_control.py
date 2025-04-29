#!/home/yangchao/.conda/envs/yolo/bin/python
from fairino import Robot
import rospy
from functools import partial
import time
import numpy as np
from std_msgs.msg import Float32MultiArray, String

    

class FairinoControl:
    def __init__(self):
        self.robot = Robot.RPC('192.168.58.2')
        ret = self.robot.SetGripperConfig(6,0)
        if not ret==0:
            print("SetGripperConfig error")
            return None
        ret=self.robot.ActGripper(1,1)
        print(ret)

        #self.robot.GetAxleLuaGripperFunc(1,9)

        self.start_pub = rospy.Publisher('/start_record', String, queue_size=10)

    def reset_position(self):
        """
        复位机械臂
        """
        tool = 0 #工具坐标系编号
        user = 0 #工件坐标系编号
        reset_joints = [0,0,0,0,0,0]
        ret = self.robot.MoveJ(reset_joints, tool, user)
        if not ret==0:
            rospy.logerr("MoveJ failed, error code: %d"%ret)
        rospy.loginfo("Reset position finished")
    
    def move_gripper(self, position):
        error = self.robot.MoveGripper(1,position,48,46,30000,0,0,0,0,0)
        print("MoveGripper error code:", error)
    
    def start_record(self):
        time.sleep(3)
        self.start_pub.publish("start")
        joint_pos4 = [-83.24, -96.476, 93.688, -114.079, -62, -100]
        joint_pos5 = [-43.24, -70.476, 93.688, -114.079, -62, -80]
        joint_pos6 = [-83.24, -96.416, 43.188, -74.079, -80, -10]
        tool = 0 #工具坐标系编号
        user = 0 #工件坐标系编号
        ret = self.robot.MoveJ(joint_pos4, tool, user, vel=30)   #关节空间运动
        print("关节空间运动点4:错误码", ret)
        ret, joint_pos = self.robot.GetActualJointPosDegree(flag=1)   #获取当前关节角度
        print("当前关节角度1", joint_pos)
        ret = self.robot.MoveJ(joint_pos5, tool, user)
        print("关节空间运动点5:错误码", ret)
        ret, joint_pos = self.robot.GetActualJointPosDegree(flag=1)   #获取当前关节角度
        print("当前关节角度2", joint_pos)

        self.robot.MoveJ(joint_pos6, tool, user, offset_flag=1, offset_pos=[10,10,10,0,0,0])
        print("关节空间运动点6:错误码", ret)
        ret, joint_pos = self.robot.GetActualJointPosDegree(flag=1)   #获取当前关节角度
        print("当前关节角度3", joint_pos)

if __name__ == '__main__':
    rospy.init_node('fairino_control')
    
    fairino_control = FairinoControl()
    ret = fairino_control.robot.MoveJ([110,-90,0,-90,-90,-11.8], 0, 0, vel=30)
    
    #fairino_control.start_record()
    #rospy.spin()