#coding=utf8
import time
import copy
import socket
import struct
import numpy as np
import math
from real.robotiq_gripper import RobotiqGripper
from real.realsenseD415 import Camera

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import datetime
import math
import os
import psutil
import sys


class Ur5e_Robot:
    def __init__(self, tcp_host_ip="192.168.1.12", tcp_port=50002, is_use_robotiq85=True, is_use_camera=True):
        # Parameters
        self.vel = 0.5
        self.acc = 0.5
        rtde_frequency = 500.0
        self.dt = 1.0/rtde_frequency  # 2ms
        flags = RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT

        self.lookahead_time = 0.1
        self.gain = 600

        # ur_rtde realtime priorities
        rt_receive_priority = 90
        rt_control_priority = 85

        self.rtde_r = RTDEReceive(tcp_host_ip, rtde_frequency, [], True, False, rt_receive_priority)
        self.rtde_c = RTDEControl(tcp_host_ip, rtde_frequency, flags, tcp_port, rt_control_priority)

        # Set application real-time priority
        os_used = sys.platform
        process = psutil.Process(os.getpid())
        if os_used == "win32":  # Windows (either 32-bit or 64-bit)
            process.nice(psutil.REALTIME_PRIORITY_CLASS)
        elif os_used == "linux":  # linux
            rt_app_priority = 80
            param = os.sched_param(rt_app_priority)
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO, param)
            except OSError:
                print("Failed to set real-time process scheduler to %u, priority %u" % (os.SCHED_FIFO, rt_app_priority))
            else:
                print("Process real-time priority set to: %u" % rt_app_priority)

        self.is_use_robotiq85 = is_use_robotiq85
        self.is_use_camera = is_use_camera

        # robotiq85 gripper configuration
        if(self.is_use_robotiq85):
            # reference https://gitlab.com/sdurobotics/ur_rtde
            # Gripper activate
            self.gripper = RobotiqGripper()
            self.gripper.connect(tcp_host_ip, 63352)  # don't change the 63352 port
            self.gripper._reset()
            print("Activating gripper...")
            self.gripper.activate()
            time.sleep(1.5)
            
        # realsense configuration
        if(self.is_use_camera):
            # Fetch RGB-D data from RealSense camera
            self.camera = Camera()
            # self.cam_intrinsics = self.camera.intrinsics  # get camera intrinsics
        """
            # 编码为一个3x3矩阵,格式符合相机内参的典型结构：[fx, 0, cx][0, fy, cy][0, 0, 1]
            # fx/fy为焦距,cx/cy为主点坐标。
        """
        # self.cam_intrinsics = np.array([615.284,0,309.623,0,614.557,247.967,0,0,1]).reshape(3,3) #d415 id:830112070066
        # self.cam_intrinsics = np.array([385.052, 0, 325.992, 0, 384.614, 236.406, 0, 0, 1]).reshape(3,3) #D455 id:246322301022
        self.cam_intrinsics = np.array([608.473, 0, 324.588, 0, 608.173, 248.71, 0, 0, 1]).reshape(3,3) #D435 id:233522076091
        
        # # Load camera pose (from running calibrate.py), intrinsics and depth scale
        self.cam_pose = np.loadtxt('real/cam_pose/camera_pose.txt', delimiter=' ')
        self.cam_depth_scale = np.loadtxt('real/cam_pose/camera_depth_scale.txt', delimiter=' ')

    def world_base_T_right(self):
        roll = -2.356
        pitch = -0.001 
        yaw = -3.141
        Rx = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])
        Ry = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
        Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
        # 计算最终的旋转矩阵 R = Rz * Ry * Rx(相对于固定坐标系，右乘)
        R = np.dot(Rz, np.dot(Ry, Rx))
        T = np.array([
                [R[0, 0], R[0, 1], R[0, 2], -0.380],
                [R[1, 0], R[1, 1], R[1, 2], 1.152],
                [R[2, 0], R[2, 1], R[2, 2], 0.905],
                [0, 0, 0, 1]
            ])
        return T
    
    def world_base_T_left(self):
        roll = 2.356
        pitch = 0
        yaw = 0
        Rx = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])
        Ry = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
        Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
        # 计算最终的旋转矩阵 R = Rz * Ry * Rx(相对于固定坐标系，右乘)
        R = np.dot(Rz, np.dot(Ry, Rx))
        T = np.array([
                [R[0, 0], R[0, 1], R[0, 2], 0.380],
                [R[1, 0], R[1, 1], R[1, 2], 1.152],
                [R[2, 0], R[2, 1], R[2, 2], 0.905],
                [0, 0, 0, 1]
            ])
        return T

    def tool0_world_T(self, roll, pitch, yaw, x, y, z):
        Rx = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])
        Ry = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
        Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
        # 计算最终的旋转矩阵 R = Rz * Ry * Rx(相对于固定坐标系，右乘)
        R = np.dot(Rz, np.dot(Ry, Rx))
        T = np.array([
                [R[0, 0], R[0, 1], R[0, 2], x],
                [R[1, 0], R[1, 1], R[1, 2], y],
                [R[2, 0], R[2, 1], R[2, 2], z],
                [0, 0, 0, 1]
            ])
        return T

    def get_TCP_pose_right(self, roll, pitch, yaw, x, y, z):
        """
        将基于world坐标系的(roll, pitch, yaw, x, y, z)->基座坐标系下的(x, y, z, rx, ry, rz),其中(rx, ry, rz)是旋转矢量
        """
        T06 = np.dot(self.world_base_T_right(), self.tool0_world_T(roll, pitch, yaw, x, y, z)) 
        x = T06[0,3]
        y = T06[1,3]
        z = T06[2,3]
        """计算（-pi,pi)旋转矢量"""
        R = T06[:3, :3]
        # 计算旋转角度θ（修正数值精度问题）
        trace = np.trace(R)
        theta = math.acos(max(-1.0, min(1.0, (trace - 1) / 2)))
        # 处理零旋转
        if theta < 1e-6:
            return [x, y, z, 0.0, 0.0, 0.0]
        # 计算旋转轴方向（修正反对称矩阵提取）
        axis = np.array([R[2, 1] - R[1, 2],
                        R[0, 2] - R[2, 0],
                        R[1, 0] - R[0, 1]]) / (2 * math.sin(theta))
        # 归一化旋转轴
        axis /= np.linalg.norm(axis)
        # 映射到 [0, 2π) 范围（保留原始旋转方向）
        if theta < 0:
            theta += 2 * math.pi
        elif theta > 2 * math.pi:
            theta -= 2 * math.pi
        # 组合旋转矢量：方向轴 × 角度
        rx, ry, rz = axis * theta
        TCP = [x, y, z, rx, ry, rz] #(-pi,pi)范围的旋转矢量
        return TCP
    
    def get_TCP_pose_left(self, roll, pitch, yaw, x, y, z):
        """
        将基于world坐标系的(roll, pitch, yaw, x, y, z)->基座坐标系下的(x, y, z, rx, ry, rz),其中(rx, ry, rz)是旋转矢量
        """
        T06 = np.dot(self.world_base_T_left(), self.tool0_world_T(roll, pitch, yaw, x, y, z)) # right_tool0_To_right_base_T
        x = T06[0,3]
        y = T06[1,3]
        z = T06[2,3]
        """计算（-pi,pi)旋转矢量"""
        R = T06[:3, :3]
        # 计算旋转角度θ（修正数值精度问题）
        trace = np.trace(R)
        theta = math.acos(max(-1.0, min(1.0, (trace - 1) / 2)))
        # 处理零旋转
        if theta < 1e-6:
            return [x, y, z, 0.0, 0.0, 0.0]
        # 计算旋转轴方向（修正反对称矩阵提取）
        axis = np.array([R[2, 1] - R[1, 2],
                        R[0, 2] - R[2, 0],
                        R[1, 0] - R[0, 1]]) / (2 * math.sin(theta))
        # 归一化旋转轴
        axis /= np.linalg.norm(axis)
        # 映射到 [0, 2π) 范围（保留原始旋转方向）
        if theta < 0:
            theta += 2 * math.pi
        elif theta > 2 * math.pi:
            theta -= 2 * math.pi
        # 组合旋转矢量：方向轴 × 角度
        rx, ry, rz = axis * theta
        TCP = [x, y, z, rx, ry, rz] #(-pi,pi)范围的旋转矢量
        return TCP


    def go_home(self):
        TCP_init_pose = self.get_TCP_pose_right(3.109, -0.016, -1.561, 0.257, -0.393, 1.2) #(rx, ry, rz, x, y, z)
        # print(TCP_init_pose)
        self.rtde_c.moveL(TCP_init_pose, self.vel, self.acc)
      
    ## robotiq85 gripper
    def get_current_tool_pos(self):
        """返回夹爪开度"""
        # get gripper position [0-255]  open:0 ,close:255
        return self.gripper.get_current_position()       

    def log_gripper_info(self):
        """打印夹爪开度信息"""
        print(f"Pos: {str(self.gripper.get_current_position())}")

    def close_gripper(self,speed=255,force=255):
        # position: int[0-255], speed: int[0-255], force: int[0-255] open:0 ,close:255
        self.gripper.move_and_wait_for_pos(255, speed, force)
        print("gripper had closed!")
        time.sleep(1.2)
        self.log_gripper_info()

    def open_gripper(self,speed=255,force=255):
        # position: int[0-255], speed: int[0-255], force: int[0-255] open:0 ,close:255
        self.gripper.move_and_wait_for_pos(0, speed, force)
        print("gripper had opened!")
        time.sleep(1.2)
        self.log_gripper_info()

    def get_camera_data(self):
        ## get camera data 
        color_img, depth_img = self.camera.get_data()
        return color_img, depth_img

    def check_grasp(self):
        """检查夹爪是否打开"""
        # Note: must be preceded by close_gripper()
        # if the robot grasp unsuccessfully ,then the gripper close
        return self.get_current_tool_pos()>220

    def plane_grasp(self, position, open_size=0.65, k_acc=0.5, k_vel=0.25, speed=125, force=125):
        """平面抓取"""
        rpy = [-0.6250320266265061, -0.5814812750878331, -1.506216636430596] # 相对于base的rpy

        # pre work
        self.go_home() #返回预设开始位置

        # 给夹爪一定的开度
        open_pos = int(-258*open_size +230)  # open size:0~0.85cm --> open pos:230~10
        self.gripper.move_and_wait_for_pos(open_pos, speed, force)
        print("gripper open size:")
        self.log_gripper_info()

        # # 移动到预定位置
        # # Firstly, achieve pre-grasp position
        # pre_position = copy.deepcopy(position) # 深拷贝创建一个新的复合对象，并且递归地复制对象中所有子对象。原始对象和深拷贝对象之间拥有独立的内存空间，任何修改都不会相互影响。
        # pre_position[2] = pre_position[2] + 0.2  # z axis
        # self.rtde_c.moveL(pre_position + rpy, k_acc, k_vel) # 到达比预设位置高10cm处

        # 抓取动作实现
        # Second，achieve grasp position
        self.rtde_c.moveL(position+rpy, 0.6*k_acc, 0.6*k_vel)
        self.close_gripper(speed, force)
        # self.rtde_c.moveL(pre_position+rpy, 0.6*k_acc, 0.6*k_vel)
        if(self.check_grasp()): 
            print("Check grasp fail! ")
            self.go_home()
            return False
        
        # 放置夹住的物体
        # Third,put the object into box
        box_position = self.get_TCP_pose_right(3.109, -0.016, -1.561, 0.4, -0.3, 1.2)
        self.rtde_c.moveL(box_position, k_acc, k_vel)
        self.open_gripper(speed,force)

        # 返回home位置
        self.go_home()
        print("grasp success!")
        return True

if __name__ =="__main__":
    ur_robot_right = Ur5e_Robot()
    # ur_robot_left = Ur5e_Robot(tcp_host_ip="192.168.1.11", is_use_camera=False, is_use_robotiq85=False) # 3.130, 0.071, -2.167     x0 = 0.175 y0 = 0.163+0.033*num z0 = 1.130 
    ur_robot_right.go_home()
    # home_left = ur_robot_right.get_TCP_pose_left(3.130, 0.071, -2.167, 0.175, 0.163, 1.2)
    # ur_robot_right.rtde_c.moveL(home_left, 0.5, 0.25)