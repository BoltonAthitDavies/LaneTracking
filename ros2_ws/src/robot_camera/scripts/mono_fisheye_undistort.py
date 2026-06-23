#!/usr/bin/python3

import os
import time
import yaml
import torch
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from threading import Lock

from cv2.ximgproc import createDisparityWLSFilter, createRightMatcher

class StereoFisheye2Depth(Node):
    def __init__(self):
        super().__init__('stereo_fisheye2depth_node')
        self._declare_params()
        self._init_cv()
        self._load_calibration()
        self._init_ros_io()
        # self._init_timer()
        self._init_gpu_if_available()

        # === FPS counter ===
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        self.get_logger().info("Stereo Fisheye to Depth Node initialized.")

    # === Initialization Methods ===
    def _init_cv(self):
        self.bridge = CvBridge()
        self.image_lock = Lock()
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.scale = 0.5

        # Calibration and Rectification
        self.left_K = self.left_D = self.right_K = self.right_D = None
        self.R = self.T = self.E = self.F = None
        self.img_shape = None

        # Stereo matcher and filter
        self.path_yaml = None

        self.frame = None

    def _declare_params(self):
        self.declare_parameter('show_images', False)
        self.declare_parameter('resize_image', True)
        self.declare_parameter('ori_width', 800) # 1280,800
        self.declare_parameter('ori_height', 320) # 720, 320

    def _load_calibration(self):
        try:
            file_name = 'matlab_calibration.yaml'
            pkg_path = get_package_share_directory('robot_camera')
            root_path = pkg_path.split('install')[0]
            self.path_yaml = os.path.join(root_path, 'src', 'robot_camera', 'config', file_name)
            if not os.path.exists(self.path_yaml):
                raise FileNotFoundError(f"{self.path_yaml} not found")

            with open(self.path_yaml, 'r') as f:
                calib_data = yaml.safe_load(f)

            self.left_K = np.array(calib_data['left_K'])
            self.left_D = np.array(calib_data['left_D'])
            self.right_K = np.array(calib_data['right_K'])
            self.right_D = np.array(calib_data['right_D'])
            self.R = np.array(calib_data['R'])
            self.T = np.array(calib_data['T'])
            self.img_shape = tuple(calib_data['image_shape'])
            self.get_logger().info("Loaded calibration successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration: {e}")

    def _init_ros_io(self):
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        # Publishers
        self.left_undist_pub = self.create_publisher(Image, '/cam0_undis/image_raw', qos)
        self.create_subscription(Image, '/cam0/image_raw', self.cam_callback, qos)

    def _init_timer(self):
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

    def _init_gpu_if_available(self):
        if self.cuda_available:
            self.get_logger().info(f"✅ CUDA enabled! Found {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
            self._init_gpu_resources()
        else:
            self.get_logger().warn("⚠️ CUDA not available! Using CPU fallback.")

    def _gstreamer_pipeline(self):
        pipeline = " ! ".join(["v4l2src device=/dev/video0",
                            "video/x-raw, width=1920, height=1080, framerate=30/1",
                            "videoconvert",
                            "video/x-raw, format=(string)BGR",
                            "appsink"
                            ])
        video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        # print("Backend:", video_capture.getBackendName())
        # print(cv2.getBuildInformation())
        return video_capture

    # === Timer Callback ===
    def timer_callback(self):

        # if self.frame is None:
        #     return
        # timestamp = self.get_clock().now().to_msg()
        # left_u = self.process_frame(self.frame)
        # self.publish_compressed_images(left_u, timestamp)

        # # === FPS counter ===
        # self.frame_count += 1
        # current_time = time.time()
        # if current_time - self.last_time >= 1.0:  # update every 1 second
        #     self.fps = self.frame_count / (current_time - self.last_time)
        #     self.get_logger().info(f"📸 Processing FPS Camera: {self.fps:.2f}")
        #     self.frame_count = 0
        #     self.last_time = current_time
        return

    def generate_frame(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            return
        
        # Torch processing
        tensor = torch.from_numpy(frame).cuda().float()
        tensor = torch.clamp(tensor + 20, 0, 255)
        img = tensor.byte().cpu().numpy()

        cv2.imshow('torch capture',img)
        cv2.waitKey(1)
        # # Encode as JPEG
        # ret, buffer = cv2.imencode('.jpg', img)

        # if not ret:
        #     return
        # frame_bytes = buffer.tobytes()

    # === Callback ===
    def cam_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.image_lock:
                self.latest_img = frame

            if frame is None:
                self.get_logger().warn("No image from fisheye camera")
                return
            timestamp = self.get_clock().now().to_msg()
            left_u = self.process_frame(frame)
            self.publish_compressed_images(left_u, timestamp)

            # === FPS counter ===
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:  # update every 1 second
                self.fps = self.frame_count / (current_time - self.last_time)
                self.get_logger().info(f"📸 Processing FPS Camera: {self.fps:.2f}")
                self.frame_count = 0
                self.last_time = current_time
        except Exception as e:
            self.get_logger().error(f"Failed to process camera image: {e}")

    # === GPU Processing ===
    def _init_gpu_resources(self):
        map1_l, map2_l = cv2.fisheye.initUndistortRectifyMap(self.left_K, self.left_D, np.eye(3), self.left_K, self.img_shape, cv2.CV_32FC1)
        # map1_r, map2_r = cv2.fisheye.initUndistortRectifyMap(self.right_K, self.right_D, np.eye(3), self.right_K, self.img_shape, cv2.CV_32FC1)

        self.gpu_map1_l, self.gpu_map2_l = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        # self.gpu_map1_r, self.gpu_map2_r = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        self.gpu_map1_l.upload(map1_l)
        self.gpu_map2_l.upload(map2_l)
        # self.gpu_map1_r.upload(map1_r)
        # self.gpu_map2_r.upload(map2_r)

    # === Frame Processing ===
    def process_frame(self, left_frame):
        show_images = self.get_parameter('show_images').value
        h, w = left_frame.shape[:2]
        new_size = (int(w * self.scale), int(h * self.scale))

        # Undistort using GPU
        left_u = self._undistort_frames_gpu(left_frame)

        # Download from GPU before CPU WLS
        left_u_cpu = left_u.download()

        if show_images:
            self._display_results(left_u, new_size)

        return left_u_cpu

    def _undistort_frames_gpu(self, left_frame):
        ori_width = self.get_parameter('ori_width').value
        ori_height = self.get_parameter('ori_height').value
        resize_image = self.get_parameter('resize_image').value

        d_left = cv2.cuda_GpuMat()
        d_left.upload(left_frame)

        d_left_u = cv2.cuda.remap(d_left, self.gpu_map1_l, self.gpu_map2_l, cv2.INTER_LINEAR)
        if resize_image:
            d_left_u = cv2.cuda.resize(d_left_u, (ori_width, ori_height))

        return d_left_u

    def _display_results(self, left_u, size):
        # left_img = cv2.resize(left_u.download(), size)
        left_img = left_u.download()
        cv2.imshow("Left Undistorted", left_img)
        cv2.waitKey(1)

    # === ROS Publishing ===
    def publish_compressed_images(self, left_u, timestamp):
        try:
            left_msg = self.bridge.cv2_to_imgmsg(left_u)
            left_msg.header.stamp = timestamp
            left_msg.header.frame_id = "cam0_undis"
            self.left_undist_pub.publish(left_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish images: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = StereoFisheye2Depth()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
