#!/usr/bin/python3

import os
import cv2
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory


class StereoCamKeyCapture(Node):
    def __init__(self):
        super().__init__('stereo_cam_key_capture')

        # Parameters
        self.declare_parameter('file_name_yaml', 'matlab_calibration_resize.yaml')
        self.declare_parameter('output_path', '/home/ubuntu/LaneTracking/Deep_Lane_Detect/raw_dataset')
        self.declare_parameter('mode', 'image')   # "video" or "image"

        # Params
        self.mode = self.get_parameter('mode').value
        self.output_path = self.get_parameter('output_path').value

        # Check CUDA
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if not self.cuda_available:
            self.get_logger().warn("⚠️ CUDA not available! Falling back to CPU.")

        # Load calibration
        self._load_calibration()

        # Open cameras
        self.cap_left = cv2.VideoCapture(self._gstreamer_pipeline("/dev/video0"), cv2.CAP_GSTREAMER)
        self.cap_right = cv2.VideoCapture(self._gstreamer_pipeline("/dev/video1"), cv2.CAP_GSTREAMER)

        if not self.cap_left.isOpened():
            self.get_logger().error("❌ Failed to open /dev/video0")
            return
        if not self.cap_right.isOpened():
            self.get_logger().error("❌ Failed to open /dev/video1")
            return

        # Setup outputs
        if self.mode == "video":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            os.makedirs(self.output_path, exist_ok=True)
            self.out_left = cv2.VideoWriter(os.path.join(self.output_path, "left_cam.mp4"),
                                            fourcc, 30.0, self.img_shape)
            self.out_right = cv2.VideoWriter(os.path.join(self.output_path, "right_cam.mp4"),
                                             fourcc, 30.0, self.img_shape)
            self.get_logger().info(f"🎬 Press 'c' to capture frames into videos.")
        else:
            os.makedirs(os.path.join(self.output_path, "left"), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, "right"), exist_ok=True)
            self.frame_count = 0
            self.get_logger().info(f"📸 Press 'c' to save stereo images.")

        self.capture_loop()

    def _load_calibration(self):
        try:
            pkg_path = get_package_share_directory('robot_camera')
            root_path = pkg_path.split('install')[0]
            file_name = self.get_parameter('file_name_yaml').value
            path_yaml = os.path.join(root_path, 'src', 'robot_camera', 'config', file_name)

            with open(path_yaml, 'r') as f:
                calib_data = yaml.safe_load(f)

            self.K_left = np.array(calib_data['left_K'])
            self.D_left = np.array(calib_data['left_D'])
            self.K_right = np.array(calib_data['right_K'])
            self.D_right = np.array(calib_data['right_D'])
            self.img_shape = tuple(calib_data['image_shape'])

            # self.map1_left, self.map2_left = cv2.fisheye.initUndistortRectifyMap(
            #     self.K_left, self.D_left, np.eye(3), self.K_left, self.img_shape, cv2.CV_32FC1)
            # self.map1_right, self.map2_right = cv2.fisheye.initUndistortRectifyMap(
            #     self.K_right, self.D_right, np.eye(3), self.K_right, self.img_shape, cv2.CV_32FC1)

            # Precompute rectification maps
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K_left, self.D_left, np.eye(3), self.K_left, self.img_shape, cv2.CV_32FC1
            )

            if self.cuda_available:
                self.gpu_map1 = cv2.cuda_GpuMat()
                self.gpu_map2 = cv2.cuda_GpuMat()
                self.gpu_map1.upload(map1)
                self.gpu_map2.upload(map2)
            else:
                self.map1, self.map2 = map1, map2

            self.get_logger().info("✅ Calibration loaded.")
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration: {e}")
            self.map1_left, self.map2_left = None, None
            self.map1_right, self.map2_right = None, None

    def _gstreamer_pipeline(self, device):
        return (
            f"v4l2src device={device} ! video/x-raw, width=1920, height=1080, framerate=30/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )

    def _undistort(self, frame, map1, map2):
        if map1 is not None:
            d_frame = cv2.cuda_GpuMat()
            d_frame.upload(frame)
            d_frame_re = cv2.cuda.resize(d_frame, self.img_shape)
            d_undist = cv2.cuda.remap(d_frame_re, map1, map2, interpolation=cv2.INTER_LINEAR)
            frame_undist = d_undist.download()
            return frame_undist
        return frame

    def capture_loop(self):
        state = 0
        while rclpy.ok():
            ret_l, frame_l = self.cap_left.read()
            ret_r, frame_r = self.cap_right.read()
            if not ret_l or not ret_r:
                self.get_logger().warn("⚠️ Failed to capture frame.")
                continue

            frame_l = self._undistort(frame_l, self.gpu_map1, self.gpu_map2)
            frame_r = self._undistort(frame_r, self.gpu_map1, self.gpu_map2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') or state == 1:  # capture
                if self.mode == "video" :
                    self.out_left.write(frame_l)
                    self.out_right.write(frame_r)
                    state = 1
                    print("🎬 Captured frame into video.")
                else:
                    fname_l = os.path.join(self.output_path, "left", f"frame_{self.frame_count:04d}.png")
                    fname_r = os.path.join(self.output_path, "right", f"frame_{self.frame_count:04d}.png")
                    cv2.imwrite(fname_l, frame_l)
                    cv2.imwrite(fname_r, frame_r)
                    print(f"📸 Saved {fname_l} and {fname_r}")
                    self.frame_count += 1

            if key == ord('q'):  # quit
                break

            h, w = frame_l.shape[:2]

            cv2.circle(frame_l, (w//2, h//2), 3, (0, 0, 255), -1)
            cv2.imshow("Left", frame_l)
            cv2.imshow("Right", frame_r)

        self.destroy_node()

    def destroy_node(self):
        self.cap_left.release()
        self.cap_right.release()
        if self.mode == "video":
            self.out_left.release()
            self.out_right.release()
            print("💾 Stereo videos saved successfully")
        else:
            print("💾 Stereo images saved successfully")
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StereoCamKeyCapture()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


# Record stereo videos (left_cam.mp4, right_cam.mp4):
# ros2 run robot_camera stereo_cam_recorder_node.py --ros-args -p mode:=video -p output_path:=/home/ubuntu/output_videos

# Capture stereo images (left/frame_XXXXXX.png, right/frame_XXXXXX.png):
# ros2 run robot_camera stereo_cam_recorder_node.py --ros-args -p mode:=image -p output_path:=/home/ubuntu/images
