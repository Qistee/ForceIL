#!/home/yangchao/.conda/envs/fairno/bin/python
import serial
import time

def parse_data_frame(frame):
    """
    解析数据帧（19字节），返回各通道数据的字典。
    """
    if len(frame) != 19:
        print("数据帧长度错误:", len(frame))
        return None

    if frame[0] != 0x24:
        print("帧头错误:", hex(frame[0]))
        return None

    if frame[-1] != 0xFF:
        print("帧尾错误:", hex(frame[-1]))
        return None

    seq = frame[1]
    channels = {}

    if seq == 0x00:
        # 通道1～8数据帧
        start = 2
        for ch in range(1, 9):
            high = frame[start]
            low = frame[start + 1]
            channels[ch] = high * 256 + low
            start += 2
    elif seq == 0x08:
        # 通道9～16数据帧
        start = 2
        for ch in range(9, 17):
            high = frame[start]
            low = frame[start + 1]
            channels[ch] = high * 256 + low
            start += 2
    else:
        print("未知序号字节:", hex(seq))
        return None

    return channels

def parse_initial_info_frame(frame):
    """
    解析初始信息帧（19字节），返回序列号。
    """
    if len(frame) != 19:
        print("初始信息帧长度错误:", len(frame))
        return None

    if frame[0] != 0x25:
        print("初始信息帧头错误:", hex(frame[0]))
        return None

    if frame[-1] != 0xFF:
        print("初始信息帧尾错误:", hex(frame[-1]))
        return None

    lot_no = frame[1:18]  
    print("收到初始信息帧，序列号：", lot_no.hex())
    return lot_no

def send_ack_frame(ser):
    """
    发送应答帧给采集板。
    """
    ack_frame = bytearray([0xAA, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF])
    info=ser.write(ack_frame)
    print("发送应答帧：", ack_frame, info)


class force_recorder:
    def __init__(self):
        self.ser = serial.Serial(
            port='/dev/ttyUSB1',           
            baudrate=115200,       # 波特率115200
            bytesize=serial.EIGHTBITS,  # 数据位8
            parity=serial.PARITY_NONE,  # 无校验
            stopbits=serial.STOPBITS_ONE,  # 停止位1
            timeout=1              # 读写超时设置
        )
        send_ack_frame(self.ser)  # 发送应答帧
        time.sleep(5)  # 等待应答帧发送完成
    
    def read_one_data(self):
        # 读取 19 字节数据（一个完整的数据帧）
        data = self.ser.read(19)
        if data:
            # 将数据转换为 bytearray 便于处理
            frame = bytearray(data)
            if frame[0] == 0x24:
                channels = parse_data_frame(frame)
                if channels:
                    data_dict = channels
            elif frame[0] == 0x25:
                lot_no = parse_initial_info_frame(frame)
                if lot_no:
                    send_ack_frame(self.ser)  # 发送应答帧
            else:
                print("收到未知帧头：", hex(frame[0]))
        return data_dict
    
    def read_data(self):
        data1 = self.read_one_data()
        while 9 in data1.keys():
            data1 = self.read_one_data()
        data2 = self.read_one_data()
        while 1 in data2.keys():
            data2 = self.read_one_data()
        data = []
        for i in range(len(data1)):
            data.append(data1[i+1])
        for i in range(len(data2)):
            data.append(data2[i+9])
        return data
    
    def close(self):
        self.ser.close()

if __name__ == '__main__':
    recorder = force_recorder()
    for i in range(10):
        data = recorder.read_data()
        print(data)
        time.sleep(0.1)
    recorder.close()