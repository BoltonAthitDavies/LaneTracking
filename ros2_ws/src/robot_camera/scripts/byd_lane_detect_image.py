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
from byd_trt_infer import UFLDv2
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

import sys

class Undistortion():
    def __init__(self):
        self._init_cv()
        self._load_calibration()
        self._init_gpu_if_available()
        self._init_camera()
        # Removed internal FPS trackers as we will do it in the main loop
        
    def _init_cv(self):
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.scale = 0.5
        self.left_K = self.left_D = self.right_K = self.right_D = None
        self.R = self.T = self.E = self.F = None
        self.img_shape = None
        self.path_yaml = None
        self.frame = None

    def _load_calibration(self):
        try:
            # IMPORTANT: Ensure this path is correct for your system
            self.path_yaml = "/home/ubuntu/LaneTracking/ros2_ws/src/robot_camera/config/matlab_calibration.yaml"
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
            print("Loaded calibration successfully.")
        except Exception as e:
            print(f"Failed to load calibration: {e}")

    def _init_gpu_if_available(self):
        if self.cuda_available:
            print(f"✅ CUDA enabled for OpenCV! Found {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
            self._init_gpu_resources()
        else:
            print("⚠️ CUDA not available for OpenCV! Using CPU fallback.")

    def _init_gpu_resources(self):
        # Initialize GPU undistortion maps
        map1_l, map2_l = cv2.fisheye.initUndistortRectifyMap(self.left_K, self.left_D, np.eye(3), self.left_K, self.img_shape, cv2.CV_32FC1)
        # map1_r, map2_r = cv2.fisheye.initUndistortRectifyMap(self.right_K, self.right_D, np.eye(3), self.right_K, self.img_shape, cv2.CV_32FC1)

        self.gpu_map1_l, self.gpu_map2_l = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        # self.gpu_map1_r, self.gpu_map2_r = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        self.gpu_map1_l.upload(map1_l)
        self.gpu_map2_l.upload(map2_l)
        # self.gpu_map1_r.upload(map1_r)
        # self.gpu_map2_r.upload(map2_r)

    def _init_camera(self):
        pipeline = " ! ".join(["v4l2src device=/dev/video0",
                            "video/x-raw, width=1920, height=1080, framerate=30/1",
                            # "nvvidconv",  # <--- FAST (GPU/HARDWARE)
                            "videoconvert", # <--- ADD THIS LINE
                            "video/x-raw, format=(string)BGR",
                            "appsink"
                            ])
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    def undistort_frame(self, left_frame, train_width=480, train_height=320):
        """
        Undistorts a single frame using pre-computed GPU maps.
        Returns only the display-sized (640x360) undistorted image.
        """
        d_left = cv2.cuda_GpuMat()
        d_left.upload(left_frame)

        # Perform GPU-accelerated remapping
        d_left_u = cv2.cuda.remap(d_left, self.gpu_map1_l, self.gpu_map2_l, cv2.INTER_LINEAR)

        # Resize for display
        d_left_u = cv2.cuda.resize(d_left_u, (train_width, train_height))
        
        # Download the final image from GPU to CPU
        return d_left_u

class LaneDetectUFLD(Node):
    def __init__(self):
        super().__init__('lane_detect_ufld_node')

        self.bridge = CvBridge()
        self.image_lock = Lock()

        self.declare_parameter('engine_path', 'model/culane_res34_480x320.engine')
        self.declare_parameter('config_path', 'config/culane_res34_480x320.py')
        self.declare_parameter('show_images', True)
        self.show_images = self.get_parameter('show_images').value


        self._init_ufld()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.offset_pub = self.create_publisher(Float32, "/lane_offset", 10)
        self.lane_center_pub = self.create_publisher(Float32, "/lane_center", 10)
        self.image_center_pub = self.create_publisher(Float32, "/image_center", 10)
        
        self.latest_img = None
        self.latest_depth = None
        self.lane_width_px_est = None  
        self.lane_width_alpha = 0.9  

        # --- NEW: Add state for last known good lanes ---
        self.last_known_left_lane = None
        self.last_known_right_lane = None
        # --- END NEW ---

        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        # --- Timer ---
        self.create_timer(1/30, self.timer_callback)
        self.undis = Undistortion()

        self.get_logger().info(f"✅ LaneDetectUFLD Node initialized. (FP16 mode: {self.fp16_mode})")
        
        ''' --- Test with video --- 
            Before use this code, Please comment function timer_callback current before usage
        '''
        # --- Timer ---
    #     self.create_timer(1/30, self.timer_callback)
    #     self.video_path = '/home/ubuntu/LaneTracking/Deep_Lane_Detect/raw_dataset/high_bright_25_cw.mp4'
    #     self.cap = cv2.VideoCapture(self.video_path)

    # def timer_callback(self):
    #     if not self.cap.isOpened():
    #         self.get_logger().info(f"Error: Could not open video file {self.video_path}")
    #         sys.exit()
    #     success, frame = self.cap.read()
    #     if not success:
    #         self.get_logger().info("Finished processing video.")
    #         return
    #     frame_gpu = cv2.cuda_GpuMat()
    #     frame_gpu.upload(frame)
    #     frame_gpu_resize = cv2.cuda.resize(frame_gpu, (480, 320))
    #     self.run_lane_detection(frame_gpu_resize)
    ''' ----------------------- '''

    def timer_callback(self):
        ret, frame = self.undis.cap.read() 
        if not ret:
            self.get_logger().info("Camera read failed.")
            return

        # 3. Pass the "corrected" BGR frame to undistort
        frame_undis = self.undis.undistort_frame(frame, self.isnet.input_width, self.isnet.input_height)
        
        # 4. Run detection
        self.run_lane_detection(frame_undis)
    # ----------------------------------
# start_time = time.time()
# end_time = time.time()
# fps = 1 / (end_time - start_time)
# self.get_logger().info(f"FPS: {fps} fps")
    def _init_ufld(self):
        engine_path = self.get_parameter('engine_path').value
        config_path = self.get_parameter('config_path').value
        self.fp16_mode = "_fp16" in engine_path.lower()

        self.get_logger().info(f"Engine path: {engine_path}")

        engine_path = self.path(engine_path)
        config_path = self.path(config_path)

        self.isnet = UFLDv2(engine_path, config_path)

    def path(self, path_name):
        directory, filename = os.path.split(path_name)
        pkg_name = 'robot_camera'
        path_pkg_share_path = get_package_share_directory(pkg_name)
        ws_path, _ = path_pkg_share_path.split('install')
        path = os.path.join(ws_path, 'src', pkg_name, directory, filename)
        return path

    def visualize_lane_offset(self, im0, left_lane, right_lane, lane_width, pixel2meter=True):
        """
        Compute and visualize lateral distance from lane center in meters.
        
        --- MODIFIED to handle Case 4 reconstruction ---
        """
        h, w, _ = im0.shape
        ref_y = h // 2 + h // 3
        # ref_y = h - 1
        x_car = w // 2
        fallback = None # Will be 'left', 'right', or 'both'

        # --- NEW: Handle Case 4 (No Lanes) at the START ---
        if (left_lane is None or len(left_lane) == 0) and \
           (right_lane is None or len(right_lane) == 0):
            print(f"No lane detected Case 4. Attempting reconstruction...")
            if self.last_known_left_lane is not None and self.last_known_right_lane is not None:
                print(f"  > Reconstructing from last known good lanes.")
                left_lane = self.last_known_left_lane   # Use stale data
                right_lane = self.last_known_right_lane # Use stale data
                fallback = "both" # Set new fallback flag
            else:
                # No lanes this frame AND no history. Give up.
                print(f"  > No lanes detected and no history. Cannot reconstruct.")
                return im0, None
        # --- END NEW ---

        # Ensure numpy arrays
        left_lane = np.array(left_lane) if left_lane is not None and len(left_lane) > 0 else None
        right_lane = np.array(right_lane) if right_lane is not None and len(right_lane) > 0 else None

        x_left, x_right = None, None

        # --- Case 1: both lanes available (either new or from fallback) ---
        if left_lane is not None and right_lane is not None:
            try:
                x_left = np.interp(ref_y, left_lane[:, 1], left_lane[:, 0])
                x_right = np.interp(ref_y, right_lane[:, 1], right_lane[:, 0])
            except Exception as e:
                self.get_logger().warn(f"Interpolation failed: {e}")
                return im0, None
            
            if self.show_images:
                # --- NEW: Create and draw the lane overlay ---
                try:
                    # Create an overlay image, copy of im0 will be modified
                    overlay = np.zeros_like(im0)
                    
                    # Get the points for the polygon
                    # We need to reverse the right lane points to connect them correctly
                    polygon_points = np.concatenate((left_lane, np.flipud(right_lane)), axis=0)
                    
                    # Draw the polygon on the overlay
                    # Using a semi-transparent green (0, 255, 0)
                    cv2.fillPoly(overlay, [polygon_points.astype(np.int32)], (0, 255, 0))
                    
                    # Blend the overlay with the original image
                    alpha = 0.3 # Transparency factor
                    im0 = cv2.addWeighted(im0, 1, overlay, alpha, 0)
                    
                except Exception as e:
                    self.get_logger().warn(f"Failed to draw lane overlay: {e}")
                    # Don't fail the whole function, just log the warning
                # --- END NEW ---

            lane_width_px = x_right - x_left

            # Sanity check: (Only apply if NOT using 'both' fallback)
            if (lane_width_px < 100 or lane_width_px > w * 0.9) and fallback != "both":
                if abs(x_left - x_car) > abs(x_right - x_car):
                    fallback = "left"
                    x_right = x_left + (self.lane_width_px_est or w * 0.45)
                    if self.show_images:
                        cv2.circle(im0, (int(x_right), ref_y), 12, (0, 255, 0), -1)
                    print(f"reconstruct right Case 1")
                else:
                    fallback = "right"
                    x_left = x_right - (self.lane_width_px_est or w * 0.45)
                    if self.show_images:
                        cv2.circle(im0, (int(x_left), ref_y), 12, (0, 0, 255), -1)
                    print(f"reconstruct left Case 1")
                lane_width_px = x_right - x_left

            # Update adaptive lane width *only if it's a good, new, dual detection*
            if lane_width_px > 100 and fallback is None:
                if self.lane_width_px_est is None:
                    self.lane_width_px_est = lane_width_px
                else:
                    self.lane_width_px_est = (
                        self.lane_width_alpha * self.lane_width_px_est
                        + (1 - self.lane_width_alpha) * lane_width_px
                    )
        # --- Case 2: only left lane detected ---
        elif left_lane is not None:
            fallback = "right"
            try:
                x_left = np.interp(ref_y, left_lane[:, 1], left_lane[:, 0])
            except Exception as e:
                self.get_logger().warn(f"Interpolation failed (left): {e}")
                return im0, None
            x_right = x_left + (self.lane_width_px_est or w * 0.45)
            if self.show_images:
                cv2.circle(im0, (int(x_right), ref_y), 6, (0, 255, 0), -1)
            print(f"reconstruct right Case 2")

        # --- Case 3: only right lane detected ---
        elif right_lane is not None:
            fallback = "left"
            try:
                x_right = np.interp(ref_y, right_lane[:, 1], right_lane[:, 0])
            except Exception as e:
                self.get_logger().warn(f"Interpolation failed (right): {e}")
                return im0, None
            x_left = x_right - (self.lane_width_px_est or w * 0.45)
            if self.show_images:
                cv2.circle(im0, (int(x_left), ref_y), 6, (0, 0, 255), -1)
            print(f"reconstruct left Case 3")

        # --- Final check ---
        if x_left is None or x_right is None:
             print(f"Fatal error: x_left or x_right is still None. Cannot calculate offset.")
             return im0, None

        # Lane center
        x_center = int((x_left + x_right) / 2.0)
        if pixel2meter:
            lane_width_px = x_right - x_left
            if lane_width_px <= 0:
                return im0, None
            meters_per_pixel = lane_width / lane_width_px
            x_car_meters = x_car * meters_per_pixel
            x_center_meters = x_center * meters_per_pixel
            offset_m = x_car_meters - x_center_meters
            if self.show_images:
                cv2.putText(im0, f"Offset: {offset_m:.2f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            x_cars = x_car_meters
            x_centers = x_center_meters
        else:
            offset_m = x_car - x_center
            if self.show_images:
                cv2.putText(im0, f"Offset: {offset_m:.2f} pixel", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            x_cars = x_car
            x_centers = x_center

        # === Visualization ===
        if self.show_images:
            cv2.line(im0, (int(x_left), ref_y), (int(x_right), ref_y), (0, 255, 255), 2)
            cv2.line(im0, (x_car, (h//2+h//3)-10), (x_car, (h//2+h//3)+10), (0, 255, 0), 2)
            cv2.line(im0, (x_center, (h//2+h//3)-10), (x_center, (h//2+h//3)+10), (255, 0, 0), 2)
            
            # === Debug overlay if fallback ===
            if fallback == "left":
                cv2.putText(im0, "RECONSTRUCTED LEFT", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            elif fallback == "right":
                cv2.putText(im0, "RECONSTRUCTED RIGHT", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # --- NEW: Add visualization for "both" ---
            elif fallback == "both":
                cv2.putText(im0,
                            "RECONSTRUCTED BOTH (USING LAST)",
                            (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 165, 255), # Orange color
                            2,
                            cv2.LINE_AA)
        # --- END NEW ---

        return im0, offset_m, x_centers, x_cars

    def run_lane_detection(self, frame):

        try:
            coords_ori_space, im0_draw, lane_fits_params, fitted_lane_coords = self.isnet.forward(frame)
        except Exception as e:
            self.get_logger().error(f"Engine forward failed. Frame shape {frame.shape}, dtype {frame.dtype}. Error: {e}")
            return

        # Use the smooth, fitted points for offset calculation
        left_lane, right_lane = None, None
        for lane_idx, lane_pts in fitted_lane_coords:
            if lane_idx == 0: # Assuming 1 is left lane for CuLane
                left_lane = lane_pts
                # --- NEW: Update last known state ---
                self.last_known_left_lane = lane_pts
            elif lane_idx == 1: # Assuming 2 is right lane for CuLane
                right_lane = lane_pts
                # --- NEW: Update last known state ---
                self.last_known_right_lane = lane_pts
        
        # Pass the *current* frame's lanes (which could be None)
        im0_with_offset, offset_m, x_center, x_car = self.visualize_lane_offset(im0_draw, left_lane, right_lane, lane_width=0.42, pixel2meter=False)

        if offset_m is not None:
            msg = Float32()
            msg.data = float(offset_m)
            self.offset_pub.publish(msg)
        if x_center is not None:
            msg = Float32()
            msg.data = float(x_center)
            self.lane_center_pub.publish(msg)
        if x_car is not None:
            msg = Float32()
            msg.data = float(x_car)
            self.image_center_pub.publish(msg)
        if self.show_images:
            cv2.imshow("Lane Detection", im0_with_offset)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectUFLD()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

