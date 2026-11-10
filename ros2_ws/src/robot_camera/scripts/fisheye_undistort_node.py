#!/usr/bin/python3
"""Subscribes to the raw stereo fisheye topics and republishes undistorted images.

Reference: WiL/stereo_fisheye_camera_calibration/read_cam.py — the undistort
maps (cv2.fisheye.initUndistortRectifyMap) are built once per camera, from
the first frame's actual size, and reused every frame via cv2.remap(). Both
cameras share the same K/D since only one mono fisheye calibration exists for
this rig (config/wil/fisheye_calibration.json) — same assumption made for
config/wil/cam0_fisheye.yaml / cam1_fisheye.yaml in vins_fusion_ros2.
"""

import json

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class FisheyeUndistortNode(Node):
    def __init__(self):
        super().__init__('fisheye_undistort_node')

        self.declare_parameter(
            'calibration_file',
            '/home/ubuntu/LaneTracking/ros2_ws/src/vins_fusion_ros2/config/wil/fisheye_calibration.json')
        self.declare_parameter('input0_topic', '/cam0/image_raw')
        self.declare_parameter('input1_topic', '/cam1/image_raw')
        self.declare_parameter('output0_topic', '/cam0_undis/image_raw')
        self.declare_parameter('output1_topic', '/cam1_undis/image_raw')

        self.bridge = CvBridge()
        self.K, self.D = self._load_intrinsics()
        # Filled in lazily per camera, once the first frame reveals the real
        # image size (see _undistort_maps()).
        self._maps = {0: None, 1: None}

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST)

        self.pubs = {
            0: self.create_publisher(Image, self.get_parameter('output0_topic').value, qos),
            1: self.create_publisher(Image, self.get_parameter('output1_topic').value, qos),
        }

        self.create_subscription(
            Image, self.get_parameter('input0_topic').value,
            lambda msg: self._image_callback(msg, cam_id=0), qos)
        self.create_subscription(
            Image, self.get_parameter('input1_topic').value,
            lambda msg: self._image_callback(msg, cam_id=1), qos)

        self.get_logger().info(
            f"Fisheye undistort node ready "
            f"(K/D from {self.get_parameter('calibration_file').value}, shared by cam0/cam1)")

    def _load_intrinsics(self):
        path = self.get_parameter('calibration_file').value
        with open(path) as f:
            data = json.load(f)
        K = np.array(data['K'], dtype=np.float64)
        D = np.array(data['D'], dtype=np.float64).reshape(4, 1)
        return K, D

    def _undistort_maps(self, cam_id, height, width):
        maps = self._maps[cam_id]
        if maps is None:
            # Equivalent to cv::fisheye::undistortImage(src, dst, K, D, K):
            # reuse K as the output projection so scale/principal point are
            # preserved (see calib.py / deploy_stereo.py in WiL).
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.K, (width, height), cv2.CV_16SC2)
            maps = (map1, map2)
            self._maps[cam_id] = maps
            self.get_logger().info(f"Built undistort maps for cam{cam_id} at {width}x{height}")
        return maps

    def _image_callback(self, msg, cam_id):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cam{cam_id}: failed to convert image: {e}")
            return

        height, width = frame.shape[:2]
        map1, map2 = self._undistort_maps(cam_id, height, width)
        undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)

        out_msg = self.bridge.cv2_to_imgmsg(undistorted, encoding='bgr8')
        out_msg.header = msg.header
        out_msg.header.frame_id = f'cam{cam_id}_undis'
        self.pubs[cam_id].publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FisheyeUndistortNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
