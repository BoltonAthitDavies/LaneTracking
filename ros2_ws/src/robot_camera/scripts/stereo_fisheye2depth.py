#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import os
import pickle
import yaml
import numpy as np
import cv2
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image, CompressedImage
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
from threading import Lock
import time


class StereoFisheye2Depth(Node):
    def __init__(self):
        super().__init__('stereo_fisheye2depth_node')
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.scale = 0.5

        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Storage for calibration images
        self.left_images = []
        self.right_images = []
        self.image_lock = Lock()
        
        # Current images with timestamps
        self.current_left = None
        self.current_right = None
        self.left_timestamp = None
        self.right_timestamp = None
        
        # Calibration results
        self.left_K = None
        self.left_D = None
        self.right_K = None
        self.right_D = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        self.img_shape = None
        
        # Undistortion maps
        self.left_map1 = None
        self.left_map2 = None
        self.right_map1 = None
        self.right_map2 = None

        # Undistore Image, Depth Topics
        self.left_undistort_global = None
        self.right_undistort_global = None
        self.depth_global = None

        # Pre-allocated image buffers
        self.left_undist_buffer = None
        self.right_undist_buffer = None

        # Stereo matcher and WLS filter
        self.stereo_matcher_left = None
        self.stereo_matcher_right = None  # For WLS filter
        self.wls_filter = None
        self.baseline = None
        self.path_yaml = None
        
        # Performance tracking
        self.last_process_time = time.time()
        self.process_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Declare parameters
        self.declare_parameter('file_name_yaml', 'matlab_calibration.yaml')
        self.declare_parameter('show_images', False)
        self.declare_parameter('pub_unistortion_image', False)

        # Stereo block matching parameters
        self.declare_parameter('num_disparities', 256)
        self.declare_parameter('block_size', 15)
        
        # Depth Estimation parameters
        self.declare_parameter('max_depth', 400.0)
        self.declare_parameter('min_depth', 0.0)

        # Get paths for calibration files
        self.path_calibration()

        # Subscribers with larger queue size and best effort QoS for real-time performance
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qos_profile = QoSProfile(
            depth=1,  # Keep only latest message
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        self.left_sub = Subscriber(self, Image, '/cam0/image_raw')
        self.right_sub = Subscriber(self, Image, '/cam1/image_raw')

        self.ts = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.stereo_callback)

                
        pub_unistortion_image = self.get_parameter('pub_unistortion_image').value
        if pub_unistortion_image:
            # Publishers for undistorted images
            self.left_undist_pub = self.create_publisher(CompressedImage, '/cam0_undis/image_raw/compressed', 1)
            self.right_undist_pub = self.create_publisher(CompressedImage, '/cam1_undis/image_raw/compressed', 1)
        # Publishers for depth images
        self.depth_pub = self.create_publisher(CompressedImage, '/depth/image_raw/compressed', 1)
        
        # Timer for processing - increased frequency but with frame skipping
                
        # Display configuration
        self.display_scale = 0.5
        self.frame_skip_counter = 0

        # Load existing calibration if available
        self.load_calibration()

        # Initialize GPU resources if available
        if self.cuda_available:
            # print(f"✅ CUDA enabled! Found {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s)")
            self.get_logger().info(f"✅ CUDA enabled! Found {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s)")
            self._init_gpu_resources()
        else:
            print("⚠️  WARNING: CUDA not available! Falling back to CPU processing.")
            self.get_logger().warn("⚠️  WARNING: CUDA not available! Falling back to CPU processing.")
            return

        self.get_logger().info("Stereo Fishete to Depth Node has been start")

    def process_frame(self, left_frame, right_frame):
        """Process stereo frames based on mode"""
        # start_time = time.time()
        show_images = self.get_parameter('show_images').value
        
        h, w = left_frame.shape[:2]
        new_size = (int(w * self.scale), int(h * self.scale))
            
        left_u, right_u = self._undistort_frames_gpu(left_frame, right_frame)

        if show_images:
            left_u_cpu = left_u.download()
            right_u_cpu = right_u.download()
            left_u_cpu = cv2.resize(left_u_cpu, new_size)
            right_u_cpu = cv2.resize(right_u_cpu, new_size)
            cv2.imshow("Left Undistorted", left_u_cpu)
            cv2.imshow("Right Undistorted", right_u_cpu)

        # Compute disparity
        disp = self._compute_disparity_gpu(left_u ,right_u)

        # Visualize disparity
        if show_images:
            disp_re = cv2.resize(disp, new_size)
            cv2.imshow(f"Disparity", disp_re)
        
        # Compute depth
        depth = self.disparity_to_depth(disp)
        # self.depth_global = depth
        
        # Visualize depth
        if show_images:
            alpha = 1.0
            # Define key points: (x, y)
            keypoints = {
                "Center": (w // 2, h // 2),
                "Left": (w // 4, h // 2),
                "Right": (3 * w // 4, h // 2),
                "Top-Left": (0, 0),
                "Bottom-Right": (w - 1, h - 1),
            }

            for label, (x, y) in keypoints.items():
                depth_value = depth[y, x]
                text = f"{label}: {depth_value:.2f} m"

                # Calculate text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

                # Box coordinates
                top_left = (x + 5, y + 5)
                bottom_right = (x + 5 + text_width, y + 5 + text_height + baseline)

                # Draw filled rectangle (black box with 50% transparency)
                overlay = depth.copy()
                cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), thickness=-1)
                cv2.addWeighted(overlay, alpha, depth, 1 - alpha, 0, depth)

                # Put red text over the box
                cv2.putText(depth, text, (x + 5, y + 5 + text_height), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                depth_vis = cv2.resize(depth, new_size)
                cv2.imshow("Depth", depth_vis)
                
        if show_images:
            cv2.waitKey(1)

        # Calculate and display FPS
        # fps = 1.0 / (time.time() - start_time)
        # print(f"FPS: {fps:.1f}")
        
        return left_u.download(), right_u.download(), depth

    def _init_gpu_resources(self):
        """Initialize GPU-based stereo matchers and maps"""
        
        # Create undistortion maps
        map1_l, map2_l = cv2.fisheye.initUndistortRectifyMap(
            self.left_K, self.left_D, np.eye(3), self.left_K, self.img_shape, cv2.CV_32FC1
        )
        map1_r, map2_r = cv2.fisheye.initUndistortRectifyMap(
            self.left_K, self.left_D, np.eye(3), self.left_K, self.img_shape, cv2.CV_32FC1
        )
        
        # Upload maps to GPU
        self.gpu_map1_l, self.gpu_map2_l = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        self.gpu_map1_r, self.gpu_map2_r = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        self.gpu_map1_l.upload(map1_l)
        self.gpu_map2_l.upload(map2_l)
        self.gpu_map1_r.upload(map1_r)
        self.gpu_map2_r.upload(map2_r)
        
        # Initialize stereo matchers
        self._init_gpu_stereo_matchers()

    def _init_gpu_stereo_matchers(self):
        """Initialize GPU stereo matchers"""
        num_disparities = self.get_parameter('num_disparities').value
        block_size = self.get_parameter('block_size').value

        # BM matcher
        self.stereoBM = cv2.cuda.createStereoBM(
            numDisparities=num_disparities,
            blockSize=block_size
        )

        # Disparity Bilateral Filter
        self.dispBF = cv2.cuda.createDisparityBilateralFilter(
            ndisp=num_disparities,
            radius=1,
            iters=1
        )

    def _undistort_frames_gpu(self, left_frame, right_frame):
        """GPU-based undistortion"""
        # Upload to GPU
        d_left = cv2.cuda_GpuMat()
        d_right = cv2.cuda_GpuMat()
        d_left.upload(left_frame)
        d_right.upload(right_frame)

        # Undistort
        d_left_u = cv2.cuda.remap(d_left, self.gpu_map1_l, self.gpu_map2_l, cv2.INTER_LINEAR)
        d_right_u = cv2.cuda.remap(d_right, self.gpu_map1_r, self.gpu_map2_r, cv2.INTER_LINEAR)
        
        return d_left_u, d_right_u

    def _compute_disparity_gpu(self, d_left_u, d_right_u):
        """GPU-based disparity computation"""
        # Convert to grayscale
        d_left_gray = cv2.cuda.cvtColor(d_left_u, cv2.COLOR_BGR2GRAY)
        d_right_gray = cv2.cuda.cvtColor(d_right_u, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        d_disp = self.stereoBM.compute(d_left_gray, d_right_gray, stream=None)
        disp = d_disp.download()

        left_u = d_left_u.download()

        '''===============bilateralFilter Disparity Map============='''
        # disp = self.disparity_bilateral_filter(left_u, disp)
        '''========================================================='''
        
        return disp
    
    def disparity_bilateral_filter(self, image, disparity):

        # Upload disparity and image to GPU
        d_disp = cv2.cuda_GpuMat()
        d_img = cv2.cuda_GpuMat()

        # Convert disparity to CV_16S format (required by bilateral filter)
        disp_16s = (disparity * 16.0).astype(np.int16)
        d_disp.upload(disp_16s)
        d_img.upload(image)

        # Apply bilateral filter
        d_filtered = cv2.cuda_GpuMat()
        d_filtered.create(d_disp.size(), d_disp.type())
        self.dispBF.apply(disparity=d_disp,
                          image=d_img,
                          dst=d_filtered
                          )
        
        # Download and convert back to float
        filtered_disp = d_filtered.download()
        filtered_disp = filtered_disp.astype(np.float32) / 16.0

        return filtered_disp


    def disparity_to_depth(self, disparity):
        """Convert disparity to depth"""
        fx = self.left_K[0, 0]
        baseline = abs(self.T[0])
        
        with np.errstate(divide='ignore'):
            depth = (fx * baseline) / (disparity.astype(np.float32) + 1e-6)
            depth[disparity <= 0] = 0

        '''===============bilateralFilter Depth Map============='''
        # depth = self.depth_bilateral_filter(depth)
        '''====================================================='''
        return depth
        
    def depth_bilateral_filter(self, depth):
        # Upload to GPU
        depth_gpu = cv2.cuda_GpuMat()
        depth_gpu.upload(depth)

        filtered_gpu = cv2.cuda.bilateralFilter(src=depth_gpu,
                                           kernel_size=7,
                                           sigma_color=20.0,
                                           sigma_spatial=15.0)
        filtered_cpu = filtered_gpu.download()
        return filtered_cpu

    def path_calibration(self):
        """Get paths for calibration files"""
        pkg_name = 'robot_camera'
        path_pkg_share_path = get_package_share_directory(pkg_name)
        ws_path, _ = path_pkg_share_path.split('install')
        file_name_yaml = self.get_parameter('file_name_yaml').value
        self.path_yaml = os.path.join(ws_path, 'src', pkg_name, 'config', file_name_yaml)

    def load_calibration(self):
        """Load existing calibration parameters from YAML"""
        try:
            if os.path.exists(self.path_yaml):
                with open(self.path_yaml, 'r') as f:
                    calib_data = yaml.safe_load(f)

                self.left_K = np.array(calib_data['left_K'])
                self.left_D = np.array(calib_data['left_D'])
                self.right_K = np.array(calib_data['right_K'])
                self.right_D = np.array(calib_data['right_D'])
                self.R = np.array(calib_data['R'])
                self.T = np.array(calib_data['T'])

                # Handle tuple format in YAML
                self.img_shape = tuple(calib_data['image_shape'])

                self.get_logger().info("Loaded existing calibration and generated maps")

        except Exception as e:
            self.get_logger().error(f"Failed to load calibration: {e}")


    def stereo_callback(self, left_msg, right_msg):
        try:
            with self.image_lock:
                left_img = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
                right_img = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')
                timestamp = left_msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Error converting images: {e}")
            return

        left_u, right_u, depth = self.process_frame(left_img, right_img)
        self.publish_compressed_images(left_u, right_u, depth, timestamp)

    def publish_compressed_images(self, left_u, right_u, depth, timestamp):
        try:
            pub_unistortion_image = self.get_parameter('pub_unistortion_image').value
            if pub_unistortion_image:
                left_msg = self.bridge.cv2_to_compressed_imgmsg(left_u)
                right_msg = self.bridge.cv2_to_compressed_imgmsg(right_u)
                left_msg.header.stamp = timestamp
                right_msg.header.stamp = timestamp
                left_msg.header.frame_id = "cam0_undis"
                right_msg.header.frame_id = "cam1_undis"
                self.left_undist_pub.publish(left_msg)
                self.right_undist_pub.publish(right_msg)

            finite_depths = depth[np.isfinite(depth)]
            if finite_depths.size > 0:
                normalized_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                depth_msg = self.bridge.cv2_to_compressed_imgmsg(normalized_depth)
                depth_msg.header.stamp = timestamp
                depth_msg.header.frame_id = "cam0_depth"
                self.depth_pub.publish(depth_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish compressed images: {e}")
def main(args=None):
    rclpy.init(args=args)
    node = StereoFisheye2Depth()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()