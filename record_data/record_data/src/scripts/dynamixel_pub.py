#!/home/yangchao/.conda/envs/yolo/bin/python
import time, pickle, zmq
from Dynamixel_arm import DynamixelArm
from multiprocessing import shared_memory
import rospy
from std_msgs.msg import Float32MultiArray


def Dynamixel2Fairino(joints_positions):
    """
    将dynamixel的关节位置转换为fairino的关节位置
    """
    fairino_joints = []
    fairino_joints.append((joints_positions[0]-1000)/4096*360+110 if (joints_positions[0]-1000)/4096*360+110>100 else 140)
    fairino_joints.append((joints_positions[1]-1000)/4096*360-90)
    fairino_joints.append((1000-joints_positions[2])/4096*360)
    fairino_joints.append((joints_positions[3])/4096*360-120)
    fairino_joints.append((-1000+joints_positions[4])/4096*360-270)
    fairino_joints.append(-11.8)   
    fairino_gripper = int(joints_positions[5]/2-500)
    fairino_gripper=max(fairino_gripper,290)
    fairino_gripper=min(fairino_gripper,600)
    #fairino_gripper%=100
    fairino_joints.append(fairino_gripper)
    return fairino_joints


rospy.init_node('dynamixel_pub')
ctx = zmq.Context()

pub = ctx.socket(zmq.PUB)
pub.bind("tcp://*:5555")

pub_ros = rospy.Publisher('/dynamixel_pos', Float32MultiArray, queue_size=10)


DYNAMIXELARM=DynamixelArm(device_name='/dev/ttyUSB1')
while True:

    pos = DYNAMIXELARM.read_position()
    pos = Dynamixel2Fairino(pos)

    pub.send(pickle.dumps(pos))      # 广播原始读数
    data = Float32MultiArray()
    data.data = pos
    pub_ros.publish(data)
    print(pos)
    time.sleep(0.05)  # 20 Hz

