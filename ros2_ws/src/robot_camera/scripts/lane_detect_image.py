#!/usr/bin/python3

import os
import time
import yaml
import pickle
import numpy as np
import cv2
import rclpy
import pycuda.driver as cuda
import torch
from trt_infer import UFLDv2
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from threading import Lock
from ament_index_python import get_package_share_directory
from tf2_ros import Buffer, TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import transforms3d.quaternions as tq
from std_msgs.msg import Float32

class LaneDetectUFLD(Node):
    def __init__(self):
        super().__init__('lane_detect_ufld_node')

        self.bridge = CvBridge()
        self.image_lock = Lock()

        # Parameters
        self.declare_parameter('engine_path', 'model/tusimple_res34_bend_25_v1_fix_common_2_lanes_fix_convert_tusimple.engine')
        # self.declare_parameter('engine_path', 'model/tusimple_res34_bend_25_v1_640x480.engine')

        self.declare_parameter('config_path', 'config/tusimple_res34_bend_25_v1.py')
        # self.declare_parameter('config_path', 'config/tusimple_res34_bend_25_v1_640x480.py')
        self.declare_parameter('ori_width', 800) # 1280,800
        self.declare_parameter('ori_height', 320) # 720, 320
        self.declare_parameter('show_images', True)

        # Load UFLD model
        self._init_ufld()
        # TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Publish lane path
        # self.left_lane_pub = self.create_publisher(Path, '/left_lane_path', 10)
        # self.right_lane_pub = self.create_publisher(Path, '/right_lane_path', 10)
        # self.left_lane_cam_pub = self.create_publisher(Path, '/left_lane_cam_path', 10)
        # self.right_lane_cam_pub = self.create_publisher(Path, '/right_lane_cam_path', 10)
        # self.center_lane_pub = self.create_publisher(Path, '/center_lane_path', 10)
        # Publisher for lateral offset
        self.offset_pub = self.create_publisher(Float32, "/lane_offset", 10)

        # ROS2 subscribers
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, '/cam0_undis/image_raw', self.cam_callback, qos)
        # Timer
        # self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        # Buffers
        self.latest_img = None
        self.latest_depth = None

        # Initialize adaptive lane width (in pixels)
        self.lane_width_px_est = None  
        self.lane_width_alpha = 0.9  # smoothing factor for running average

        # === FPS counter ===
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        self.get_logger().info(f"✅ LaneDetectUFLD Node initialized. (FP16 mode: {self.fp16_mode})")

    def _init_ufld(self):
        engine_path = self.get_parameter('engine_path').value
        config_path = self.get_parameter('config_path').value
        ori_width = self.get_parameter('ori_width').value
        ori_height = self.get_parameter('ori_height').value
        ori_size = (ori_width,ori_height)
        self.fp16_mode = "_fp16" in engine_path.lower()

        self.get_logger().info(f"Engine path: {engine_path}")
        self.get_logger().info(f"ori_size: {ori_size}")

        engine_path = self.path(engine_path)
        # print("engine_path: ", engine_path)
        config_path = self.path(config_path)
        # print("config_path: ", config_path)

        self.isnet = UFLDv2(engine_path, config_path, ori_size)

    def path(self, path_name):
        """Resolve relative path inside ROS2 package"""
        directory, filename = os.path.split(path_name)
        pkg_name = 'robot_camera'
        path_pkg_share_path = get_package_share_directory(pkg_name)
        ws_path, _ = path_pkg_share_path.split('install')
        path = os.path.join(ws_path, 'src', pkg_name, directory, filename)
        # Replace ubuntu
        # path = path.replace("/home/ubuntu/", "/home/ubuntu/LaneTracking/ros2_ws/")
        return path

    # === Callbacks ===
    # def cam_callback(self, msg):
    #     try:
    #         frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
    #         # with self.image_lock:
    #         #     self.latest_img = frame
    #         if frame is None:
    #             self.get_logger().warn("No image from undistortion")
    #             return
    #         self.get_logger().info(f"frame shape: {frame.shape}") # (320, 480, 3)
    #         self.get_logger().info(f"frame type: {frame.dtype}") # uint8
    #         self.run_lane_detection(frame)
    #         # === FPS counter ===
    #         self.frame_count += 1
    #         current_time = time.time()
    #         if current_time - self.last_time >= 1.0:  # update every 1 second
    #             self.fps = self.frame_count / (current_time - self.last_time)
    #             self.get_logger().info(f"📸 Processing FPS Lane Detection Model: {self.fps:.2f}")
    #             self.frame_count = 0
    #             self.last_time = current_time
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to process camera image: {e}")
    def cam_callback(self, msg):
        try:
            # Ask cv_bridge for a standard 3-channel BGR image (uint8)
            frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

            if frame is None:
                self.get_logger().warn("No image from undistortion")
                return

            # Validate number of channels
            if frame.ndim != 3 or frame.shape[2] != 3:
                self.get_logger().error(f"Unexpected image shape/channels: {frame.shape}")
                return

            # Ensure expected dtype and contiguous memory for CUDA host->device copy
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            frame = np.ascontiguousarray(frame)

            self.run_lane_detection(frame)

            # === FPS counter ===
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.get_logger().info(f"📸 Processing FPS Lane Detection Model: {self.fps:.2f}")
                self.frame_count = 0
                self.last_time = current_time

        except Exception as e:
            self.get_logger().error(f"Failed to process camera image: {e}")


    def visualize_lane_offset(self, im0, left_lane, right_lane, lane_width):
        """
        Compute and visualize lateral distance from lane center in meters.
        Handles missing/misclassified lanes by estimating with adaptive lane width.
        """
        h, w, _ = im0.shape
        ref_y = h // 2 + h // 3  # reference row 
        # ref_y = h - 30  # reference row 
        x_car = w // 2  # camera center
        fallback = None

        # Ensure numpy arrays
        left_lane = np.array(left_lane) if left_lane is not None and len(left_lane) > 0 else None
        right_lane = np.array(right_lane) if right_lane is not None and len(right_lane) > 0 else None

        x_left, x_right = None, None

        # --- Case 1: both lanes detected ---
        if left_lane is not None and right_lane is not None:
            try:
                x_left = np.interp(ref_y, left_lane[:, 1], left_lane[:, 0])
                x_right = np.interp(ref_y, right_lane[:, 1], right_lane[:, 0])
            except Exception:
                return im0, None

            lane_width_px = x_right - x_left

            # print("lane_width_px: ",lane_width_px)

            # Sanity check: lanes must be wide enough and in correct order
            if lane_width_px < 100 or lane_width_px > w * 0.9:
                # Bad detection → treat as single lane case
                if abs(x_left - x_car) > abs(x_right - x_car):
                    # left lane is closer → reconstruct right
                    fallback == "left"
                    x_right = x_left + (self.lane_width_px_est or w * 0.45)
                    cv2.circle(im0, (int(x_right), ref_y), 12, (0, 255, 0), -1)
                    print(f"reconstruct right Case 1")
                else:
                    # right lane is closer → reconstruct left
                    fallback = "right"
                    x_left = x_right - (self.lane_width_px_est or w * 0.45)
                    cv2.circle(im0, (int(x_left), ref_y), 12, (0, 0, 255), -1)
                    print(f"reconstruct left Case 1")
                lane_width_px = x_right - x_left

            # Update adaptive lane width if valid
            if lane_width_px > 100:
                if self.lane_width_px_est is None:
                    self.lane_width_px_est = lane_width_px
                else:
                    self.lane_width_px_est = (
                        self.lane_width_alpha * self.lane_width_px_est
                        + (1 - self.lane_width_alpha) * lane_width_px
                    )

        # --- Case 2: only left lane detected ---
        elif left_lane is not None:
            fallback == "right"
            try:
                x_left = np.interp(ref_y, left_lane[:, 1], left_lane[:, 0])
            except Exception:
                return im0, None
            x_right = x_left + (self.lane_width_px_est or w * 0.45)
            cv2.circle(im0, (int(x_right), ref_y), 6, (0, 255, 0), -1)
            print(f"reconstruct right Case 2")

        # --- Case 3: only right lane detected ---
        elif right_lane is not None:
            fallback = "left"
            try:
                x_right = np.interp(ref_y, right_lane[:, 1], right_lane[:, 0])
            except Exception:
                return im0, None
            x_left = x_right - (self.lane_width_px_est or w * 0.45)
            cv2.circle(im0, (int(x_left), ref_y), 6, (0, 0, 255), -1)
            print(f"reconstruct left Case 3")

        # --- Case 4: no lanes detected ---
        else:
            print(f"No lane detecte Case 4")
            return im0, None

        # Lane center
        x_center = int((x_left + x_right) / 2.0)

        # Conversion: px → m
        lane_width_px = x_right - x_left
        if lane_width_px <= 0:
            return im0, None
        meters_per_pixel = lane_width / lane_width_px
        offset_m = (x_car - x_center) * meters_per_pixel

        # === Visualization ===
        cv2.line(im0, (x_car, 0), (x_car, h), (0, 255, 0), 2)         # camera center (green)
        cv2.line(im0, (x_center, 0), (x_center, h), (255, 0, 0), 2)   # lane center (blue)
        cv2.line(im0, (0, ref_y), (w, ref_y), (0, 255, 255), 1)       # reference row (yellow)

        cv2.putText(
            im0,
            f"Offset: {offset_m:.2f} m",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # === Debug overlay if fallback ===
        if fallback == "left":
            cv2.putText(im0,
                        "RECONSTRUCTED RIGHT",
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA)

        elif fallback == "right":
            cv2.putText(im0,
                        "RECONSTRUCTED LEFT",
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA)

        return im0, offset_m


    # === Lane Detection ===
    def run_lane_detection(self, frame):
        show_images = self.get_parameter('show_images').value

        # Double-check the frame matches the engine's expected size
        # self.isnet.ori_size should hold (width,height) if you implemented it that way.
        try:
            expected_w, expected_h = self.isnet.ori_size
        except Exception:
            expected_w, expected_h = None, None

        # If you know the engine expects a fixed resolution, verify it here:
        h, w = frame.shape[:2]
        if expected_w is not None and expected_h is not None:
            if (w, h) != (expected_w, expected_h):
                self.get_logger().warn(
                    f"Frame size {(w,h)} != engine expected {(expected_w, expected_h)}. "
                    "Resize or reconfigure ori_width/ori_height to match the engine."
                )
                # Option A: resize to expected resolution (uncomment if acceptable)
                # frame = cv2.resize(frame, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
                # h, w = frame.shape[:2]

        # last safety: ensure contiguous & correct dtype again
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # debug log right before sending to engine
        self.get_logger().debug(f"Calling isnet.forward with frame shape {frame.shape} dtype {frame.dtype}")

        # call engine (wrapped to catch CUDA copy errors with more context)
        try:
            lane_fits, im0 = self.isnet.forward(frame)
        except Exception as e:
            self.get_logger().error(f"Engine forward failed. Frame shape {frame.shape}, dtype {frame.dtype}. Error: {e}")
            return

        # Debug print for robot control
        # print("\n=== Debug: Lane data ===")
        left_lane, right_lane =None,  None
        for lane_id, data in lane_fits.items():
            if data["coeffs"] is not None:
                if lane_id == 0:
                    id_l = lane_id
                    left_lane = data['points']
                if lane_id == 1:
                    id_r = lane_id
                    right_lane = data['points']

        # if left_lane is not None and right_lane is not None:
        im0, offset_m = self.visualize_lane_offset(im0, left_lane, right_lane, lane_width=0.42)
        if offset_m is not None:
            msg = Float32()
            msg.data = float(offset_m)
            self.offset_pub.publish(msg)
            # self.get_logger().info(f"Lane offset: {offset_m:.2f} m")
    
        if show_images:
            cv2.imshow("Lane Detection", im0)
            cv2.waitKey(1)

    # def timer_callback(self):
    #     return

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectUFLD()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
