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
from dynamixel_sdk import PortHandler, PacketHandler,COMM_SUCCESS

PORT_NAME = '/dev/ttyUSB0'  # Linux或Mac，Windows改为'COM3'
BAUD_RATE = 1000000           # A-12默认可能为57600，需确认
DXL_ID = 5                  # 舵机ID
PROTOCOL_VERSION = 1.0      # A-12使用Protocol 1.0
ADDR_TORQUE_ENABLE = 24  # Protocol 1.0 的扭矩使能地址
TORQUE_ENABLE = 1        # 1 启用，0 关闭
ADDR_PRESENT_POSITION = 36 
ADDR_GOAL_POSITION = 30 


class FairinoControl:
    def __init__(self):
        self.robot = Robot.RPC('192.168.58.2')
        #ret = self.robot.SetGripperConfig(6,0)
        #ret=self.robot.ActGripper(1,1)
        #print(ret)

        self.joint_pos = [110,-90,0,-90,-90,-11.8]
        self.gripper_pos = 0.0

        self.pub_ros = rospy.Publisher('/dynamixel_pos', Float32MultiArray, queue_size=10)

        self.port_handler = PortHandler(PORT_NAME)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        if self.port_handler.openPort():
            print("端口已开启")
        else:
            print("无法开启端口")
            quit()

        # 设置波特率
        if self.port_handler.setBaudRate(BAUD_RATE):
            print("波特率已设置")
        else:
            print("无法设置波特率")
            quit()


        # 启用扭矩
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE
        )

        if result != COMM_SUCCESS:
            print("启用扭矩失败！")

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
        self.gripper_pos = 600.0
        self.packet_handler.write2ByteTxRx(self.port_handler, DXL_ID, ADDR_GOAL_POSITION, int(self.gripper_pos))
        self.start_read_joint_pos()
        rospy.wait_for_message('/start_recording', Bool)
        self.robot.MoveJ(target_pos[:6], 0, 0, vel=5)
        rospy.loginfo("Move to target position finished")
        target_gripper_pos = target_pos[-1]
        init_gripper_pos = self.gripper_pos
        for i in range(100):
            self.gripper_pos = init_gripper_pos + (target_gripper_pos - init_gripper_pos) * (i+1) / 100
            self.packet_handler.write2ByteTxRx(self.port_handler, DXL_ID, ADDR_GOAL_POSITION, int(self.gripper_pos))
            time.sleep(0.01)
        rospy.wait_for_message('/stop_recording', Bool)
        rospy.loginfo("Recording finished")
        self.stop_read_joint_pos()


    
    def move_home(self):
        """
        移动到初始位置
        """
        self.robot.MoveJ([120,-90,0,-90,-90,-11.8], 0, 0, vel=30)
        rospy.loginfo("Move home finished")

    def move_init(self):
        """
        移动到初始位置
        """
        self.robot.MoveJ([120.462890625, -80.859375, 33.310546875, -87.83203125, -77.255859375, -11.8], 0, 0, vel=30)
        #self.robot.MoveGripper(1,0,100,80,2000,1,0,0,0,0)
        rospy.loginfo("Move init finished")

if __name__ == '__main__':
    rospy.init_node('fairino_control')
    
    fairino_control = FairinoControl()
    fairino_control.move_init()
    #fairino_control.move_with_record([124.765625, -52.20703125, 50.44921875, -102.421875, -92.548828125, -11.8, 290.0])

  