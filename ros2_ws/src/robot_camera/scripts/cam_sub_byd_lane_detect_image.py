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

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.offset_pub = self.create_publisher(Float32, "/lane_offset", qos)

        self.create_subscription(Image, '/cam0_undis/image_raw', self.cam_callback, qos)
        
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

        self.get_logger().info(f"✅ LaneDetectUFLD Node initialized. (FP16 mode: {self.fp16_mode})")
        
    ''' --- Test with video --- 
        Before use this code, Please comment function cam_callback before usage
    '''
    #     # --- Timer ---
    #     self.create_timer(1/30, self.timer_callback)

    # def timer_callback(self):
    #     if not self.cap.isOpened():
    #         self.get_logger().info(f"Error: Could not open video file {self.video_path}")
    #         sys.exit()
    #     success, frame = self.cap.read()
    #     if not success:
    #         self.get_logger().info("Finished processing video.")
    #         return
    #     self.run_lane_detection(frame)
    ''' ----------------------- '''

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

    def cam_callback(self, msg):
        try:
            start_time = time.time()
            frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            if frame is None:
                self.get_logger().warn("No image from undistortion")
                return
            if frame.ndim != 3 or frame.shape[2] != 3:
                self.get_logger().error(f"Unexpected image shape/channels: {frame.shape}")
                return
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            frame = np.ascontiguousarray(frame)

            self.run_lane_detection(frame)
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            self.get_logger().info(f"FPS: {fps} fps")
            # self.frame_count += 1
            # current_time = time.time()
            # if current_time - self.last_time >= 1.0:
            #     self.fps = self.frame_count / (current_time - self.last_time)
            #     self.get_logger().info(f"📸 Processing FPS Lane Detection Model: {self.fps:.2f}")
            #     self.frame_count = 0
            #     self.last_time = current_time
        except Exception as e:
            self.get_logger().error(f"Failed to process camera image: {e}")

    def visualize_lane_offset(self, im0, left_lane, right_lane, lane_width, pixel2meter=True):
        """
        Compute and visualize lateral distance from lane center in meters.
        
        --- MODIFIED to handle Case 4 reconstruction ---
        """
        h, w, _ = im0.shape
        ref_y = h // 2 + h // 3
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
            offset_m = (x_car - x_center) * meters_per_pixel
            if self.show_images:
                cv2.putText(im0, f"Offset: {offset_m:.2f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            offset_m = x_car - x_center
            if self.show_images:
                cv2.putText(im0, f"Offset: {offset_m:.2f} pixel", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)


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

        return im0, offset_m

    def run_lane_detection(self, frame):

        try:
            coords_ori_space, im0_draw, lane_fits_params, fitted_lane_coords = self.isnet.forward(frame)
        except Exception as e:
            self.get_logger().error(f"Engine forward failed. Frame shape {frame.shape}, dtype {frame.dtype}. Error: {e}")
            return

        # Use the smooth, fitted points for offset calculation
        left_lane, right_lane = None, None
        for lane_idx, lane_pts in fitted_lane_coords:
            if lane_idx == 1: # Assuming 1 is left lane for CuLane
                left_lane = lane_pts
                # --- NEW: Update last known state ---
                self.last_known_left_lane = lane_pts
            elif lane_idx == 2: # Assuming 2 is right lane for CuLane
                right_lane = lane_pts
                # --- NEW: Update last known state ---
                self.last_known_right_lane = lane_pts
        
        # Pass the *current* frame's lanes (which could be None)
        im0_with_offset, offset_m = self.visualize_lane_offset(im0_draw, left_lane, right_lane, lane_width=0.42, pixel2meter=False)

        if offset_m is not None:
            msg = Float32()
            msg.data = float(offset_m)
            self.offset_pub.publish(msg)
    
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

