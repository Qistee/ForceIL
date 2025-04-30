import time, pickle, zmq
from Dynamixel_arm import DynamixelArm
from multiprocessing import shared_memory


def Dynamixel2Fairino(joints_positions):
    """
    将dynamixel的关节位置转换为fairino的关节位置
    """
    fairino_joints = []
    fairino_joints.append((joints_positions[0]-1000)/4096*360+110)
    fairino_joints.append((joints_positions[1]-1000)/4096*360-90)
    fairino_joints.append((1000-joints_positions[2])/4096*360)
    fairino_joints.append((joints_positions[3])/4096*360-120)
    fairino_joints.append((1000-joints_positions[4])/4096*360-90)
    fairino_joints.append(-11.8)   
    fairino_gripper=(joints_positions[5]-700)*100/2100
    fairino_gripper%=100
    fairino_joints.append(fairino_gripper)
    return fairino_joints


ctx = zmq.Context()

pub = ctx.socket(zmq.PUB)
pub.bind("tcp://*:5555")

DYNAMIXELARM=DynamixelArm()
while True:

    pos = DYNAMIXELARM.read_position()
    pos = Dynamixel2Fairino(pos)

    pub.send(pickle.dumps(pos))      # 广播原始读数
    print(pos)
    time.sleep(0.05)  # 20 Hz

