from force import force_recorder
import minimalmodbus
import serial
from camera import image_reader
import time, pickle, zmq
from fairino import Robot


force_reader = force_recorder()
image_reader = image_reader()

print("Start recording...")
    for t in tqdm(range(max_timesteps)):
        action = data_grabber.get_action()
        t1=time.time()
        obs = data_grabber.get_observation()
        t2=time.time()
        if t%args.frame_skip==0:
            timesteps.append(obs)
            actions.append(action)
            actual_dt_history.append([t0,t1,t2])
        time.sleep(max(0,dt-(time.time()-t0)))
    print(f"average FPS: {max_timesteps/(time.time()-time0)}")


# ——— 0. 参数配置 ———
PORT    = '/dev/ttyUSB0'   # 串口设备
SLAVEID = 1                # Modbus 从站 ID
BAUD    = 115200           # 波特率
TIMEOUT = 0.05             # 串口超时（秒）
N_STEPS = 10               # 窗口大小 n
TOTAL_SAMPLES = 500        # 要保存的序列总条数
INTERVAL = 0.05            # 读取间隔（秒）
OUT_CSV = 'gripper_history.csv'

# ——— 1. 初始化 Modbus RTU ———
instrument = minimalmodbus.Instrument(PORT, SLAVEID)
instrument.serial.baudrate = BAUD
instrument.serial.bytesize = 8
instrument.serial.parity   = serial.PARITY_NONE
instrument.serial.stopbits = 1
instrument.serial.timeout  = TIMEOUT

REG_STATUS = 0x07D1

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://localhost:5555")
sub.setsockopt_string(zmq.SUBSCRIBE, "")

data_dict = {
        'observations/force_input':[],
        'observations/gripper_pos':[],
        'observations/image':[],
        'actions/gripper_pos':[]
    }