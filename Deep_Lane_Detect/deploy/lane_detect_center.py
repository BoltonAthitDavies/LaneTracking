import os
import time
import yaml
import pickle
import numpy as np
import cv2
import torch
from trt_infer_with_lane_detect import UFLDv2
from threading import Lock
from transforms3d import quaternions as tq
from pathlib import Path
import sys

# --- Class to emulate ROS2 parameters and logging ---
class DummyNode:
    def __init__(self):
        self._params = {
            'engine_path': '/home/nvidia/LaneTracking/Deep_Lane_Detect/weights/tusimple_res34_bend_25_v1_fix_common_2_lanes_fix_convert_tusimple.engine',
            'config_path': '/home/nvidia/LaneTracking/Deep_Lane_Detect/configs/tusimple_res34_bend_25_v1.py',
            'ori_size': (1280, 720),
            'show_images': True
        }
        self.logger = self

    def get_parameter(self, name):
        return DummyParameter(self._params.get(name))

    def declare_parameter(self, *args):
        pass

    def info(self, message):
        print(f"INFO: {message}")
    
    def error(self, message):
        print(f"ERROR: {message}")

class DummyParameter:
    def __init__(self, value):
        self.value = value

# --- Main Class ---
class LaneDetectUFLD:
    def __init__(self):
        # Emulate ROS2 node initialization
        self.node = DummyNode()

        self.image_lock = Lock()
        
        # Load UFLD model
        self._init_ufld()
        
        self.latest_img = None
        self.latest_depth = None

        # Initialize adaptive lane width (in pixels)
        self.lane_width_px_est = None  
        self.lane_width_alpha = 0.9  # smoothing factor for running average

        self.node.info("✅ LaneDetectUFLD initialized.")

    def _init_ufld(self):
        engine_path = self.node.get_parameter('engine_path').value
        config_path = self.node.get_parameter('config_path').value
        ori_size = tuple(self.node.get_parameter('ori_size').value)

        # Assuming UFLDv2 class is available
        self.isnet = UFLDv2(engine_path, config_path, ori_size)

    def visualize_lane_offset(self, im0, left_lane, right_lane, lane_width):
        """
        Compute and visualize lateral distance from lane center in meters.
        Also draw the lane center line across the image (all 4 cases).
        """
        h, w, _ = im0.shape
        ref_y = h // 2 + h // 3  # reference row
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
            if lane_width_px < 100 or lane_width_px > w * 0.9:
                if abs(x_left - x_car) > abs(x_right - x_car):
                    fallback = "left"
                    x_right = x_left + (self.lane_width_px_est or w * 0.45)
                else:
                    fallback = "right"
                    x_left = x_right - (self.lane_width_px_est or w * 0.45)
                lane_width_px = x_right - x_left

            # Update adaptive lane width
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
            fallback = "right"
            try:
                x_left = np.interp(ref_y, left_lane[:, 1], left_lane[:, 0])
            except Exception:
                return im0, None
            x_right = x_left + (self.lane_width_px_est or w * 0.45)

        # --- Case 3: only right lane detected ---
        elif right_lane is not None:
            fallback = "left"
            try:
                x_right = np.interp(ref_y, right_lane[:, 1], right_lane[:, 0])
            except Exception:
                return im0, None
            x_left = x_right - (self.lane_width_px_est or w * 0.45)

        # --- Case 4: no lanes detected ---
        else:
            return im0, None

        # === Build center line ===
        ys_min_list, ys_max_list = [], []
        if left_lane is not None and len(left_lane) > 0:
            ys_min_list.append(np.min(left_lane[:, 1]))
            ys_max_list.append(np.max(left_lane[:, 1]))
        if right_lane is not None and len(right_lane) > 0:
            ys_min_list.append(np.min(right_lane[:, 1]))
            ys_max_list.append(np.max(right_lane[:, 1]))

        if len(ys_min_list) > 0 and len(ys_max_list) > 0:
            # take max of mins → overlap start
            ys_min = int(max(ys_min_list))  + 70
            # take min of maxs → overlap end
            ys_max = int(min(h - 1, max(ys_max_list)))
        else:
            ys_min, ys_max = 0, h - 1  # fallback if no lanes
            
        # sample more points for smoother line
        ys_sample = np.linspace(ys_min, ys_max, num=72)  
        centers = []

        # keep track of last valid lane width
        last_lane_width = w * 0.45   # start with a reasonable default
        lane_width_bottom = w * 0.45     # ~45% of image width near bottom
        lane_width_top    = w * 0.1      # lanes appear closer together near horizon

        for y in ys_sample:
            try:
                if left_lane is not None and right_lane is not None:
                    xl = np.interp(y, left_lane[:, 1], left_lane[:, 0])
                    xr = np.interp(y, right_lane[:, 1], right_lane[:, 0])
                    lane_width_px_ = xr - xl


                    # compute dynamic expected width at this y
                    ratio = y / h
                    expected_width = lane_width_top + (lane_width_bottom - lane_width_top) * ratio

                    # set tolerance ±50%
                    min_width = expected_width * 0.3
                    max_width = expected_width * 2.5

                    if lane_width_px_ < min_width or lane_width_px_ > max_width:

                    # if lane_width_px_ < 100 or lane_width_px_ > w * 0.9:
                        print("============")
                        print("lane_width_px_: ",lane_width_px_)
                        print("min_width: ",min_width)
                        print("max_width: ",max_width)
                        # invalid width -> fallback using last valid or default
                        if abs(x_left - x_car) < abs(x_right - x_car):
                            # robot closer to right → use left lane
                            xr = xl + last_lane_width
                            print("Create xr")
                        else:
                            # robot closer to left → use right lane
                            xl = xr - last_lane_width
                            print("Create xl")
                    else:
                        # update last valid width
                        last_lane_width = lane_width_px_

                elif left_lane is not None:
                    xl = np.interp(y, left_lane[:, 1], left_lane[:, 0])
                    xr = xl + last_lane_width

                elif right_lane is not None:
                    xr = np.interp(y, right_lane[:, 1], right_lane[:, 0])
                    xl = xr - last_lane_width

                else:
                    continue  # no lane detected at all

                xc = int((xl + xr) / 2)
                centers.append((xc, int(y)))

            except Exception as e:
                print("interp error:", e)
                continue

        if len(centers) > 1:
            cv2.polylines(im0, [np.array(centers, dtype=np.int32)],
                        isClosed=False, color=(255, 0, 0), thickness=2)

        # Lane center at reference row
        x_center = int((x_left + x_right) / 2.0)

        # Conversion px→m
        lane_width_px = x_right - x_left
        if lane_width_px <= 0:
            return im0, None
        meters_per_pixel = lane_width / lane_width_px
        offset_m = (x_car - x_center) * meters_per_pixel

        # Visualization overlays
        cv2.line(im0, (x_car, 0), (x_car, h), (0, 255, 0), 2)       # camera center (green)
        cv2.line(im0, (x_center, 0), (x_center, h), (255, 0, 0), 2) # lane center vertical
        cv2.line(im0, (0, ref_y), (w, ref_y), (0, 255, 255), 1)     # reference row

        cv2.putText(im0,
                    f"Offset: {offset_m:.2f} m",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA)

        return im0, offset_m



    def run_lane_detection(self, frame):
        show_images = self.node.get_parameter('show_images').value
        lane_fits, im0 = self.isnet.forward(frame)

        left_lane, right_lane = None, None
        for lane_id, data in lane_fits.items():
            if data["coeffs"] is not None:
                if lane_id == 0:
                    left_lane = data['points']
                if lane_id == 1:
                    right_lane = data['points']

        im0, offset_m = self.visualize_lane_offset(im0, left_lane, right_lane, lane_width=0.42)
        # if offset_m is not None:
        #     # Emulate ROS2 publisher behavior by just printing
        #     print(f"Lane offset: {offset_m:.2f} m")
    
        if show_images:
            cv2.imshow("Lane Detection", im0)
            cv2.waitKey(1)

def main():
    lane_detector = LaneDetectUFLD()
    
    # Example usage: process a single image file or from a camera
    # Replace with your image source
    # For a video:
    cap = cv2.VideoCapture('/home/nvidia/LaneTracking/Deep_Lane_Detect/raw_dataset/high_bright_25_ccw.mp4')
    # For a webcam:
    # cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream or file.")
        sys.exit()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            lane_detector.run_lane_detection(frame)

    except KeyboardInterrupt:
        print("Exiting...")
    
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Make sure you have trt_infer.py and a model file in the correct paths.
    # The 'robot_camera' directory structure is assumed to be a parent of your current working directory.
    # E.g., if this script is at `ros2_ws/src/robot_camera/scripts/lane_detector.py`, 
    # the model and config files should be at `ros2_ws/src/robot_camera/model/` and `ros2_ws/src/robot_camera/config/`.
    main()