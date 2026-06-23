#!/usr/bin/python3

import os
import time
import yaml
import pickle
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from threading import Lock


class StereoFisheye2Depth(Node):
    def __init__(self):
        super().__init__('stereo_fisheye2depth_node')
        self._init_cv()
        self._declare_params()
        self._load_calibration()
        self._init_ros_io()
        self._init_camera_streams()
        self._init_timer()
        self._init_gpu_if_available()
        self.get_logger().info("Stereo Fisheye to Depth Node initialized.")

    # === Initialization Methods ===
    def _init_cv(self):
        self.bridge = CvBridge()
        self.image_lock = Lock()
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.scale = 0.5
        self.frame_skip_counter = 0
        self.left_images, self.right_images = [], []

        # Calibration and Rectification
        self.left_K = self.left_D = self.right_K = self.right_D = None
        self.R = self.T = self.E = self.F = None
        self.img_shape = None
        self.left_map1 = self.left_map2 = None
        self.right_map1 = self.right_map2 = None

        # Undistorted and depth images
        self.left_undistort_global = self.right_undistort_global = self.depth_global = None

        # GPU Buffers
        self.left_undist_buffer = self.right_undist_buffer = None

        # Stereo matcher and filter
        self.stereo_matcher_left = self.stereo_matcher_right = self.wls_filter = None
        self.path_yaml = None

    def _declare_params(self):
        self.declare_parameter('file_name_yaml', 'matlab_calibration_resize.yaml')
        self.declare_parameter('show_images', False)
        self.declare_parameter('pub_unistortion_image', False)
        self.declare_parameter('compress_depth', True)
        self.declare_parameter('resize_image', True)
        self.declare_parameter('num_disparities', 256) # The value must be 64, 128 or 256
        self.declare_parameter('block_size', 15) #25 for lane (maybe)
        self.declare_parameter('max_depth', 400.0)
        self.declare_parameter('min_depth', 0.0)

        # Bilateral filter params (GPU version)
        self.declare_parameter('bilateral_filter_enable', True) #false for lane (maybe)
        self.declare_parameter('bilateral_filter_radius', 10)
        self.declare_parameter('bilateral_filter_iters', 2)

    def _load_calibration(self):
        try:
            pkg_path = get_package_share_directory('robot_camera')
            root_path = pkg_path.split('install')[0]
            file_name = self.get_parameter('file_name_yaml').value
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
        self.pub_undist = self.get_parameter('pub_unistortion_image').value
        self.compress_depth = self.get_parameter('compress_depth').value

        if self.pub_undist:
            self.left_undist_pub = self.create_publisher(CompressedImage, '/cam0_undis/image_raw/compressed', 1)
            self.right_undist_pub = self.create_publisher(CompressedImage, '/cam1_undis/image_raw/compressed', 1)

        if self.compress_depth:
            self.depth_pub = self.create_publisher(CompressedImage, '/depth/image_raw/compressed', 1)
        else:
            self.depth_pub = self.create_publisher(Image, '/depth/image_raw', 1)

    def _init_camera_streams(self):
        self.left_cap = cv2.VideoCapture(self._gstreamer_pipeline('/dev/video0'), cv2.CAP_GSTREAMER)
        self.right_cap = cv2.VideoCapture(self._gstreamer_pipeline('/dev/video1'), cv2.CAP_GSTREAMER)

        if not self.left_cap.isOpened() or not self.right_cap.isOpened():
            self.get_logger().error("❌ Failed to open video devices.")
            return

    def _init_timer(self):
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

    def _init_gpu_if_available(self):
        if self.cuda_available:
            self.get_logger().info(f"✅ CUDA enabled! Found {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
            self._init_gpu_resources()
        else:
            self.get_logger().warn("⚠️ CUDA not available! Using CPU fallback.")

    def _gstreamer_pipeline(self, device):
        return (
            f"v4l2src device={device} ! video/x-raw, width=1920, height=1080, framerate=30/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )

    # === Timer Callback ===
    def timer_callback(self):
        ret_l, left_frame = self.left_cap.read()
        ret_r, right_frame = self.right_cap.read()
        if not ret_l or not ret_r:
            self.get_logger().warn("⚠️ Frame read failed.")
            return

        timestamp = self.get_clock().now().to_msg()
        left_u, right_u, depth = self.process_frame(left_frame, right_frame)
        self.publish_compressed_images(left_u, right_u, depth, timestamp)

    # === GPU Processing ===
    def _init_gpu_resources(self):
        map1_l, map2_l = cv2.fisheye.initUndistortRectifyMap(self.left_K, self.left_D, np.eye(3), self.left_K, self.img_shape, cv2.CV_32FC1)
        map1_r, map2_r = cv2.fisheye.initUndistortRectifyMap(self.right_K, self.right_D, np.eye(3), self.right_K, self.img_shape, cv2.CV_32FC1)

        self.gpu_map1_l, self.gpu_map2_l = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        self.gpu_map1_r, self.gpu_map2_r = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        self.gpu_map1_l.upload(map1_l)
        self.gpu_map2_l.upload(map2_l)
        self.gpu_map1_r.upload(map1_r)
        self.gpu_map2_r.upload(map2_r)

        self._init_gpu_stereo_matchers()

    def _init_gpu_stereo_matchers(self):
        num_disp = self.get_parameter('num_disparities').value
        block_size = self.get_parameter('block_size').value

        self.stereoBM = cv2.cuda.createStereoBM(numDisparities=num_disp, blockSize=block_size)
        
        radius = self.get_parameter('bilateral_filter_radius').value
        iters = self.get_parameter('bilateral_filter_iters').value
        self.dispBF = cv2.cuda.createDisparityBilateralFilter(ndisp=num_disp, radius=radius, iters=iters)

    def _colorize_disparity(self, disp):
        # disp can be float32 or int16 from GPU, normalize to 0..255
        disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
        disp_norm = np.uint8(disp_norm)
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        return disp_color
    
    def _colorize_depth(self, depth_map):

        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = np.uint8(depth_norm)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        return depth_color

    # === Frame Processing ===
    def process_frame(self, left_frame, right_frame):
        show_images = self.get_parameter('show_images').value
        h, w = left_frame.shape[:2]
        new_size = (int(w * self.scale), int(h * self.scale))

        left_u, right_u = self._undistort_frames_gpu(left_frame, right_frame)
        disp = self._compute_disparity_gpu(left_u, right_u)
        depth = self.disparity_to_depth(disp)

        if show_images:
            # disp_color = self._colorize_disparity(disp)
            # depth_color = self._colorize_depth(depth)
            # self._display_results(left_u, right_u, disp_color, depth_color, new_size)
            self._display_results(left_u, right_u, disp, depth, new_size)

        return left_u.download(), right_u.download(), depth

    def _undistort_frames_gpu(self, left_frame, right_frame):
        d_left = cv2.cuda_GpuMat()
        d_right = cv2.cuda_GpuMat()
        d_left.upload(left_frame)
        d_right.upload(right_frame)

        if self.get_parameter('resize_image').value:
            d_left = cv2.cuda.resize(d_left, self.img_shape)
            d_right = cv2.cuda.resize(d_right, self.img_shape)

        d_left_u = cv2.cuda.remap(d_left, self.gpu_map1_l, self.gpu_map2_l, cv2.INTER_LINEAR)
        d_right_u = cv2.cuda.remap(d_right, self.gpu_map1_r, self.gpu_map2_r, cv2.INTER_LINEAR)
        return d_left_u, d_right_u

    def _compute_disparity_gpu(self, d_left_u, d_right_u):
        d_left_gray = cv2.cuda.cvtColor(d_left_u, cv2.COLOR_BGR2GRAY)
        d_right_gray = cv2.cuda.cvtColor(d_right_u, cv2.COLOR_BGR2GRAY)
        num_disp = self.get_parameter('num_disparities').value
        block_size = self.get_parameter('block_size').value

        self.stereoBM = cv2.cuda.createStereoBM(numDisparities=num_disp, blockSize=block_size)
        disp_gpu = self.stereoBM.compute(d_left_gray, d_right_gray, stream=None)

        if self.get_parameter('bilateral_filter_enable').value:
            radius = self.get_parameter('bilateral_filter_radius').value
            iters = self.get_parameter('bilateral_filter_iters').value

            # Re-create filter with updated params
            self.dispBF = cv2.cuda.createDisparityBilateralFilter(
                ndisp=self.get_parameter('num_disparities').value,
                radius=radius,
                iters=iters
            )

            # Apply GPU bilateral filter
            disp_filtered_gpu = self.dispBF.apply(disp_gpu, d_left_gray)
            return disp_filtered_gpu.download()
        else:
            return disp_gpu.download()

    def disparity_to_depth(self, disparity):
        fx = self.left_K[0, 0]
        baseline = abs(self.T[0])
        with np.errstate(divide='ignore'):
            depth = (fx * baseline) / (disparity.astype(np.float32) + 1e-6)
            depth[disparity <= 0] = 0
        return depth

    # def _display_results(self, left_u, right_u, disp, depth, size):
    #     cv2.imshow("Left Undistorted", cv2.resize(left_u.download(), size))
    #     cv2.imshow("Right Undistorted", cv2.resize(right_u.download(), size))
    #     cv2.imshow("Disparity", cv2.resize(disp, size))
    #     h, w = depth.shape[:2]
    #     alpha = 1.0
    #     keypoints = {
    #         "Center": (w // 2 + 190, h // 2),
    #         "a": (w // 2 + 110, h // 3),
    #         "b": (w // 2 - 140, h // 3),
    #     }
    #     for label, (x, y) in keypoints.items():
    #         depth_value = depth[y, x]
    #         text = f"{label}: {depth_value:.2f} m"
    #         (text_width, text_height), baseline = cv2.getTextSize(
    #             text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    #         top_left = (x + 5, y + 5)
    #         bottom_right = (x + 5 + text_width, y + 5 + text_height + baseline)
    #         overlay = depth.copy()
    #         cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), thickness=-1)
    #         cv2.addWeighted(overlay, alpha, depth, 1 - alpha, 0, depth)
    #         cv2.putText(depth, text, (x + 5, y + 5 + text_height), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    #     cv2.imshow("Depth", cv2.resize(depth, size))
    #     cv2.waitKey(1)
    def _display_results(self, left_u, right_u, disp, depth, size):
        left_img = cv2.resize(left_u.download(), size)
        right_img = cv2.resize(right_u.download(), size)
        disp_img = cv2.resize(disp, size)
        depth_img = depth.copy()
        
        h, w = depth.shape[:2]
        alpha = 0.6

        # Example adjustable lines (change coordinates as needed)
        lines = {
            "Line1": ((w // 2 - 80, 120), (290, h-1)),  # horizontal middle line
            "Line2": ((w // 2 + 70, 120), (990, h-1)),  # horizontal upper line
            "Line3": ((w // 2 - 5, 120), (w // 2 - 40, h-1)),  # middle line
        }

        for label, (start, end) in lines.items():
            # Draw the line on the depth image
            cv2.line(depth_img, start, end, (0, 0, 255), 2)
            
            # Sample points along the line
            num_points = 100
            xs = np.linspace(start[0], end[0], num_points).astype(int)
            ys = np.linspace(start[1], end[1], num_points).astype(int)
            depths_along_line = depth[ys, xs]

            # Overlay depth values at each point (optional: every 10th point)
            for x, y, d in zip(xs[::10], ys[::10], depths_along_line[::10]):
                text = f"{d:.2f}m"
                cv2.putText(depth_img, text, (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        
        alpha = 1.0 
        keypoints = {
            "a": (w // 2, 230),
            "b": (w // 2, h//2)
        }
        for label, (x, y) in keypoints.items():
            depth_value = depth[y, x]
            text = f"{label}: {depth_value:.2f} m"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            top_left = (x + 5, y + 5)
            bottom_right = (x + 5 + text_width, y + 5 + text_height + baseline)
            overlay = depth.copy()
            cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), thickness=-1)
            cv2.addWeighted(overlay, alpha, depth, 1 - alpha, 0, depth)
            cv2.putText(depth_img, text, (x + 5, y + 5 + text_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.circle(depth_img, (int(x), y), 1, (0, 255, 0), -1)

        # Show images
        # cv2.imshow("Left Undistorted", left_img)
        # cv2.imshow("Right Undistorted", right_img)
        cv2.imshow("Disparity", disp_img)
        cv2.imshow("Depth", depth_img)
        cv2.waitKey(1)
    # === ROS Publishing ===
    def publish_compressed_images(self, left_u, right_u, depth, timestamp):
        try:
            if self.pub_undist:
                left_msg = self.bridge.cv2_to_compressed_imgmsg(left_u)
                right_msg = self.bridge.cv2_to_compressed_imgmsg(right_u)
                left_msg.header.stamp = right_msg.header.stamp = timestamp
                left_msg.header.frame_id = "cam0_undis"
                right_msg.header.frame_id = "cam1_undis"
                self.left_undist_pub.publish(left_msg)
                self.right_undist_pub.publish(right_msg)

            if np.isfinite(depth).any():
                if self.compress_depth:
                    norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    depth_msg = self.bridge.cv2_to_compressed_imgmsg(norm)
                else:
                    depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding='32FC1')
                depth_msg.header.stamp = timestamp
                depth_msg.header.frame_id = "cam0_depth"
                self.depth_pub.publish(depth_msg)
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
