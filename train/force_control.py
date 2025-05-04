from force import force_recorder
import minimalmodbus
import serial
from camera import image_reader


def read_position() :
    """
    读取当前位置寄存器高字节 gPR，返回 0–255。
    """
    reg = instrument.read_registers(REG_STATUS, 1, functioncode=3)[0]
    return (reg >> 8) & 0xFF

def set_gripper(position: int, speed: int = 0xFF, force: int = 0xFF):
    """
    将夹爪移动到指定位置（0 完全开 → 255 完全闭），并设置速度/力。
    position: 0–255
    speed:    0–255
    force:    0–255
    """
    # 1) 写位置参数：写寄存器 0x03E9，高字节为 position
    value_pos = position << 8
    instrument.write_register(REG_POS, value_pos, functioncode=6)
    # 2) 写速度/力参数：寄存器 0x03EA，高字节 force，低字节 speed
    value_sf = (force << 8) | speed
    instrument.write_register(REG_SF, value_sf, functioncode=6)
    # 3) 触发动作：rACT=1, rMODE=0（动态模式）, rGTO=1 → 二进制 0000 1001 = 0x09 
    control_val = 0x09
    instrument.write_register(REG_CONTROL, control_val, functioncode=6)

force_reader = force_recorder()
image_reader = image_reader()


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
REG_CONTROL = 0x03E8
REG_SF      = 0x03EA 
REG_POS     = 0x03E9 

import time
import h5py
import numpy as np
from jodellSdk.jodellSdkDemo import ClawEpgTool

clawTool = ClawEpgTool()

com_list = clawTool.searchCom()

if not com_list:
    raise RuntimeError("未找到可用串口，请检查设备连接。")
com = com_list[0]
BAUD_RATE = 115200
flag = clawTool.serialOperation(com, BAUD_RATE, True)
if flag != 1:
    raise RuntimeError(f"串口连接失败，返回码：{flag}")

salveIdList = clawTool.scanSalveId(1, 10) 
print(salveIdList)
# 如果需要指定从站 ID，可在此处修改
SLAVE_ID = 9
flag = clawTool.clawEnable(SLAVE_ID, True)


# 3. 以 20 Hz 采样夹爪状态 :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}
interval = 1.0 / 20  # 秒
records = []  # 用于临时存储 [timestamp, status]

print("开始采集，按 Ctrl+C 停止。")
try:
    while True:
        t = time.time()
        status_list = clawTool.getClawCurrentLocation(SLAVE_ID)
        # 假设 status_list[0] 为开合状态（0/1 或其他编码）
        open_close = status_list[0]
        records.append((t, open_close))
        # 保持 20 Hz
        time.sleep(interval)
except KeyboardInterrupt:
    print("\n采集已停止，准备保存数据。")

# 4. 保存到 HDF5
data = np.array(records, dtype=np.float64)  # shape=(N, 2)
with h5py.File('claw_open_close.h5', 'w') as f:
    # 创建一个 2 列的数据集，第一列为时间戳，第二列为开合状态
    f.create_dataset('open_close', data=data, compression='gzip')
print(f"已保存 {data.shape[0]} 条记录到 claw_open_close.h5")
