#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from realsenseD415 import Camera
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0, project_root)
from Ur5e_Robot import Ur5e_Robot

"""
    像素点抓取，测试相机标定效果
"""
# --------------- Setup options ---------------
tcp_host_ip = '192.168.1.12' 
tcp_port = 50002
tool_orientation = [3.109, -0.016, -1.561] # world
# ---------------------------------------------

# Move robot to home pose
robot = Ur5e_Robot(tcp_host_ip, tcp_port)
robot.go_home()
robot.open_gripper()

# Slow down robot
robot.acc = 0.1
robot.vel = 0.25

# Callback function for clicking on OpenCV window
click_point_pix = ()
camera_color_img, camera_depth_img = robot.get_camera_data()

def mouseclick_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global camera, robot, click_point_pix
        click_point_pix = (x,y)

        # Get click point in camera coordinates
        click_z = camera_depth_img[y][x] * robot.cam_depth_scale
        click_x = np.multiply(x-robot.cam_intrinsics[0][2],click_z/robot.cam_intrinsics[0][0])
        click_y = np.multiply(y-robot.cam_intrinsics[1][2],click_z/robot.cam_intrinsics[1][1])
        if click_z == 0:
            return
        click_point = np.asarray([click_x,click_y,click_z])
        click_point.shape = (3,1)

        # Convert camera to robot coordinates
        # camera2robot = np.linalg.inv(robot.cam_pose)
        camera2robot = robot.cam_pose
        target_position = np.dot(camera2robot[0:3,0:3],click_point) + camera2robot[0:3,3:]

        target_position = target_position[0:3,0]
        print(target_position)
        # 发送的位置相对于 world坐标系
        robot.plane_grasp([target_position[0],target_position[1],target_position[2]])

# Show color and depth frames
cv2.namedWindow('color')
cv2.setMouseCallback('color', mouseclick_callback)
cv2.namedWindow('depth')

while True:
    camera_color_img, camera_depth_img = robot.get_camera_data()
    bgr_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
    if len(click_point_pix) != 0:
        bgr_data = cv2.circle(bgr_data, click_point_pix, 7, (0,0,255), 2)
    cv2.imshow('color', bgr_data)
    cv2.imshow('depth', camera_depth_img)
    
    if cv2.waitKey(1) == ord('c'):
        break

cv2.destroyAllWindows()