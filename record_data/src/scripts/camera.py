#!/home/yangchao/.conda/envs/yolo/bin/python
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy


class image_reader:
    def __init__(self):
        self.cap = cv2.VideoCapture(2) 
        self.bridge = CvBridge()
        self.realsense_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.realsense_callback)
        self.realsense_image = None

    def read_image_logitech(self):
        ret, frame = self.cap.read()  # ret==True/False: read successfully or not; frame: image
        if not ret:
            print("Failed to read the image.")
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert color space from BGR to RGB
        frame = cv2.resize(frame, (384, 384))  # resize image to 384x384
        return frame
    
    def realsense_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  # convert color space from BGR to RGB
        cv_image = cv2.resize(cv_image, (384, 384))  # resize image to 384x384
        self.realsense_image = cv_image
    
    def read_image_realsense(self):
        return self.realsense_image
    
    def read_image(self):
        return self.read_image_logitech(), self.read_image_realsense()

if __name__ == '__main__':
    rospy.init_node('image_reader', anonymous=True)
    image_reader = image_reader()
    img_logitech, img_realsense = image_reader.read_image()
    print(img_logitech.shape, img_realsense.shape)