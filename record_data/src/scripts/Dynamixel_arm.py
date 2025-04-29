#!/home/yangchao/.conda/envs/yolo/bin/python
from Robot import Robot
from Dynamixel import Dynamixel

class DynamixelArm:
    def __init__(self, bauderate=1_000_000, device_name='/dev/ttyUSB0'):
        self.dynamixel = Dynamixel.Config(baudrate=bauderate, device_name=device_name).instantiate()
        self.arm = Robot(self.dynamixel, servo_ids=[1,2,3,4,5,6])
    
    def torque_on(self):
        self.arm._enable_torque()
    
    def torque_off(self):
        self.arm._disable_torque()
    
    def read_position(self):
        "read degrees of each joint"
        return self.arm.read_position()
    
    def move_homepose(self):
        #[1000,1000,1000,0,1000,1000]
        self.arm.set_goal_pos([938, 1387, 542, 40, 875, 1000])
    
if __name__ == '__main__':
    arm = DynamixelArm()
    arm.torque_off()
    while True:
        print(arm.read_position())
    #arm.move_homepose()
