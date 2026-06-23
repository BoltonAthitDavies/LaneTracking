#!/usr/bin/python3

import os
import cv2
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory


class LeftCamRecorder(Node):
    def __init__(self):
        super().__init__('left_cam_recorder')

        # Parameters
        self.declare_parameter('file_name_yaml', 'matlab_calibration_resize.yaml')
        self.declare_parameter('output_path', '/home/ubuntu/LaneTracking/Deep_Lane_Detect/raw_dataset/rama6.mp4')
        self.declare_parameter('fps', 30)

        # Check CUDA
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if not self.cuda_available:
            self.get_logger().warn("⚠️ CUDA not available! Falling back to CPU.")

        # Load calibration
        self._load_calibration()

        # Open left camera
        self.cap = cv2.VideoCapture(self._gstreamer_pipeline('/dev/video0'), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error("❌ Failed to open /dev/video0")
            return

        # Setup VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_name = self.get_parameter('output_path').value
        output_full_path = os.path.abspath(output_name)  # Full path
        self.out = cv2.VideoWriter(
            output_name,
            fourcc,
            float(self.get_parameter('fps').value),
            self.img_shape
        )
        self.get_logger().info(f"🎬 Saving video to: {output_name}")

        # Timer
        self.timer = self.create_timer(1.0 / self.get_parameter('fps').value, self.timer_callback)
        self.get_logger().info("🎥 Left camera recording with CUDA started.")

    def _load_calibration(self):
        try:
            pkg_path = get_package_share_directory('robot_camera')
            root_path = pkg_path.split('install')[0]
            file_name = self.get_parameter('file_name_yaml').value
            path_yaml = os.path.join(root_path, 'src', 'robot_camera', 'config', file_name)

            with open(path_yaml, 'r') as f:
                calib_data = yaml.safe_load(f)

            self.left_K = np.array(calib_data['left_K'])
            self.left_D = np.array(calib_data['left_D'])
            self.img_shape = tuple(calib_data['image_shape'])

            # Precompute rectification maps
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.left_K, self.left_D, np.eye(3), self.left_K, self.img_shape, cv2.CV_32FC1
            )

            if self.cuda_available:
                self.gpu_map1 = cv2.cuda_GpuMat()
                self.gpu_map2 = cv2.cuda_GpuMat()
                self.gpu_map1.upload(map1)
                self.gpu_map2.upload(map2)
            else:
                self.map1, self.map2 = map1, map2

            self.get_logger().info("✅ Calibration loaded for CUDA undistortion.")
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration: {e}")
            self.map1, self.map2 = None, None

    def _gstreamer_pipeline(self, device):
        return (
            f"v4l2src device={device} ! video/x-raw, width=1920, height=1080, framerate=30/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("⚠️ Failed to capture frame from left camera.")
            return

        if self.cuda_available and hasattr(self, 'gpu_map1'):
            d_frame = cv2.cuda_GpuMat()
            d_frame.upload(frame)
            d_frame_re = cv2.cuda.resize(d_frame, self.img_shape)
            d_undist = cv2.cuda.remap(d_frame_re, self.gpu_map1, self.gpu_map2, interpolation=cv2.INTER_LINEAR)
            frame_undist = d_undist.download()
        elif self.map1 is not None:
            frame_undist = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        else:
            frame_undist = frame

        self.out.write(frame_undist)

        # Optional preview
        cv2.imshow("Left Undistorted (CUDA)", frame_undist)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.destroy_node()
            rclpy.shutdown()

    def destroy_node(self):
        self.cap.release()
        self.out.release()
        print("Save video finish")
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LeftCamRecorder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
