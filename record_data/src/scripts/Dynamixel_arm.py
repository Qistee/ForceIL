#!/home/yangchao/.conda/envs/yolo/bin/python
from Dynamixel_Robot import Robot
from Dynamixel import Dynamixel
from threading import Thread, Event
import time

class DynamixelArm:
    def __init__(self, bauderate=1_000_000, device_name='/dev/ttyUSB0'):
        self.dynamixel = Dynamixel.Config(baudrate=bauderate, device_name=device_name).instantiate()
        self.arm = Robot(self.dynamixel, servo_ids=[1,2,3,4,5,6])
        self._joint_pos=[938, 1387, 542, 40, 875, 1000]

        self._start_reading_thread()
    
    def torque_on(self):
        self.arm._enable_torque()
    
    def torque_off(self):
        self.arm._disable_torque()

    def _read_position(self):
        while True:
            time.sleep(0.001)
            self._joint_pos = self.arm.read_position()
    
    def read_position(self):
        "read degrees of each joint"
        return self._joint_pos
    
    def _start_reading_thread(self):
        self._reading_thread = Thread(target=self._read_position)
        self._reading_thread.daemon = True
        self._reading_thread.start()
    
    def move_homepose(self):
        #[1000,1000,1000,0,1000,1000]
        self.arm.set_goal_pos([938, 1387, 542, 40, 875, 1000])
    
if __name__ == '__main__':
    arm = DynamixelArm()
    #arm.torque_on()
    #arm.move_homepose()
    #time.sleep(3)
    arm.torque_off()
    while True:
        print(arm.read_position())
    #arm.move_homepose()
