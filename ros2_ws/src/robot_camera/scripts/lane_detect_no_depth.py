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

class LaneDetectUFLD(Node):
    def __init__(self):
        super().__init__('lane_detect_ufld_node')

        self.bridge = CvBridge()
        self.image_lock = Lock()

        # Parameters
        self.declare_parameter('engine_path', 'model/tusimple_res34_bend_25_v1_fix_common_2_lanes_fix_convert_tusimple.engine')
        self.declare_parameter('config_path', 'config/tusimple_res34_bend_25_v1.py')
        self.declare_parameter('ori_size', (1280, 720))
        self.declare_parameter('show_images', True)
        self.declare_parameter('file_name_yaml', 'matlab_calibration_resize.yaml')
        self.declare_parameter('resize_image', True)

        # Load UFLD model
        self._init_ufld()
        # Load camera parameter
        self._load_calibration()
        # TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Publish lane path
        self.left_lane_pub = self.create_publisher(Path, '/left_lane_path', 10)
        self.right_lane_pub = self.create_publisher(Path, '/right_lane_path', 10)
        self.left_lane_cam_pub = self.create_publisher(Path, '/left_lane_cam_path', 10)
        self.right_lane_cam_pub = self.create_publisher(Path, '/right_lane_cam_path', 10)
        self.center_lane_pub = self.create_publisher(Path, '/center_lane_path', 10)

        # ROS2 subscribers
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, "/cam0_undis/image_raw", self.cam_callback, qos)

        self.get_logger().info("✅ LaneDetectUFLD Node initialized.")

        # Buffers
        self.latest_img = None
        self.latest_depth = None

    def _load_calibration(self):
        try:
            pkg_path = get_package_share_directory('robot_camera')
            root_path = pkg_path.split('install')[0]
            resize_image = self.get_parameter('resize_image').value
            if resize_image:
                file_name = self.get_parameter('file_name_yaml').value
            else:
                file_name = 'matlab_calibration.yaml'
            self.path_yaml = os.path.join(root_path, 'src', 'robot_camera', 'config', file_name)
            # self.path_yaml = self.path_yaml.replace("/home/ubuntu/", "/home/ubuntu/LaneTracking/ros2_ws/")
            print("self.path_yaml: ", self.path_yaml)
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

    def _init_ufld(self):
        engine_path = self.get_parameter('engine_path').value
        config_path = self.get_parameter('config_path').value
        resize_image = (self.get_parameter('resize_image').value)
        if resize_image:
            ori_size = tuple(self.get_parameter('ori_size').value)
        else:
            ori_size = (1920, 1080)

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
    def cam_callback(self, msg):
        try:
            # frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            with self.image_lock:
                self.latest_img = frame
            self.run_lane_detection(frame)
        except Exception as e:
            self.get_logger().error(f"Failed to process camera image: {e}")

    # def pixels_to_camera_lane(self, lane_points, im0):
    #     """
    #     Convert a list of lane pixel coordinates [(u,v), ...] 
    #     into camera coordinates [(X,Y,Z), ...].
    #     Invalid depth values are linearly interpolated.
    #     """
    #     h, w = im0.shape[:2]
    #     fx = self.left_K[0, 0]
    #     fy = self.left_K[1, 1]
    #     cx = self.left_K[0, 2]
    #     cy = self.left_K[1, 2]

    #     # Convert to integer pixel coords and clamp
    #     us = np.clip([int(u) for (u, v) in lane_points], 0, w - 1)
    #     vs = np.clip([int(v) for (u, v) in lane_points], 0, h - 1)

    #     # Use fixed Z instead of depth
    #     Zs = np.full(len(lane_points), 0.302, dtype=np.float32)

    #     # Back-project into camera coordinates
    #     Xs = (us - cx) * Zs / fx
    #     Ys = (vs - cy) * Zs / fy

    #     return np.stack([Xs, Ys, Zs], axis=-1)

    def pixels_to_camera_lane(self, lane_points_uv, cam_frame: str = "cam0_undis", base_frame: str = "base_footprint"):
        """
        Project lane pixel points (u,v) onto the ground plane (Z=0 in base_footprint).
        Uses camera intrinsics + TF instead of fixed Z.
        Returns Nx3 points in base_footprint frame.
        """
        if len(lane_points_uv) == 0:
            return np.empty((0,3))

        # Camera intrinsics
        K = self.left_K
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]

        # Pixels → normalized rays in camera frame
        lane_pixels_uv = np.array(lane_points_uv, dtype=np.float32)
        us, vs = lane_pixels_uv[:,0], lane_pixels_uv[:,1]
        rays_cam = np.stack([(us - cx)/fx, (vs - cy)/fy, np.ones_like(us)], axis=1)

        # Lookup TF: base ← cam
        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                base_frame,
                cam_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return np.empty((0, 3))

        # Rotation + translation
        q = trans.transform.rotation
        R = tq.quat2mat([q.w, q.x, q.y, q.z])    # 3x3
        t = np.array([trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z])

        # Rays in base frame
        rays_base = (R @ rays_cam.T).T
        cam_origin_base = t

        # Intersect each ray with ground plane Z=0 in base frame
        Xs, Ys, Zs = [], [], []
        for r in rays_base:
            denom = r[2]
            if abs(denom) < 1e-6:   # parallel to ground
                continue
            s = -cam_origin_base[2] / denom
            if s <= 0:              # intersection behind camera
                continue
            p = cam_origin_base + s * r
            Xs.append(p[0])
            Ys.append(p[1])
            Zs.append(p[2])

        return np.stack([Xs, Ys, Zs], axis=-1) if len(Xs) > 0 else np.empty((0,3))


    # === transform camera coordinates → base_footprint ===
    def camera_to_base_batch(self, cam_points, cam_frame="cam0_undis", base_frame="base_footprint", force_z0=False):
        """
        cam_points: Nx3 numpy array of [X,Y,Z] in camera frame
        Returns Nx3 numpy array in base_footprint frame
        """
        if cam_points.shape[0] == 0:
            return np.empty((0, 3))

        try:
            # Lookup TF
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                base_frame,
                cam_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return np.empty((0, 3))

        # Convert quaternion → rotation matrix using transforms3d
        q = trans.transform.rotation
        # transforms3d uses [w, x, y, z]
        R = tq.quat2mat([q.w, q.x, q.y, q.z])  # 3x3 rotation matrix
        # print('=========TF=========\n')
        # print(trans)
        t = np.array([trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z]).reshape((3, 1))  # 3x1 column

        # Apply transform: p_base = R * p_cam + t
        cam_points_T = cam_points.T  # 3xN
        base_points_T = R @ cam_points_T + t  # 3xN
        base_points = base_points_T.T  # Nx3

        # Force Z = 0 if requested
        if force_z0:
            base_points[:, 2] = 0.0

        return base_points

    # === Publish lane path ===
    def publish_path(self, lane_points, frame_id, pub):
        """
        lane_points: Nx3 numpy array in base_footprint
        frame_id: usually 'base_footprint'
        pub: ROS2 publisher
        """
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id

        for X, Y, Z in lane_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(X)
            pose.pose.position.y = float(Y)
            pose.pose.position.z = float(Z)
            pose.pose.orientation.w = 1.0  # identity quaternion
            path_msg.poses.append(pose)

        pub.publish(path_msg)

    def compute_and_publish_center_lane(self, left_lane_base, right_lane_base, pub, frame_id="base_footprint", lane_width=3.5):
        """
        Compute center path between left and right lanes.
        If one lane is missing, estimate it using lane_width.
        Publish as Path.
        """
        # Case 1: Both lanes detected
        if left_lane_base.shape[0] > 0 and right_lane_base.shape[0] > 0:
            min_len = min(left_lane_base.shape[0], right_lane_base.shape[0])
            left = left_lane_base[:min_len]
            right = right_lane_base[:min_len]
            centers = (left + right) / 2.0

        # Case 2: Only left lane detected → estimate right using lane_width
        elif left_lane_base.shape[0] > 0:
            left = left_lane_base
            # Shift right by lane_width along the normal (X forward, Y left in ROS base_footprint)
            offset = np.array([0.0, -lane_width, 0.0])
            right = left + offset
            centers = (left + right) / 2.0

        # Case 3: Only right lane detected → estimate left using lane_width
        elif right_lane_base.shape[0] > 0:
            right = right_lane_base
            offset = np.array([0.0, lane_width, 0.0])
            left = right + offset
            centers = (left + right) / 2.0

        else:
            # No lanes → nothing to publish
            return

        # === Publish Path ===
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id

        for X, Y, Z in centers:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(X)
            pose.pose.position.y = float(Y)
            pose.pose.position.z = float(Z)
            pose.pose.orientation.w = 1.0  # facing forward
            path_msg.poses.append(pose)

        pub.publish(path_msg)

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

        return im0, centers

    # === Lane Detection ===
    def run_lane_detection(self, frame):
        show_images = self.get_parameter('show_images').value
        lane_fits, im0 = self.isnet.forward(frame)

        # Debug print for robot control
        # print("\n=== Debug: Lane data ===")
        for lane_id, data in lane_fits.items():
            if data["coeffs"] is not None:
                if lane_id == 0:
                    id_l = lane_id
                    left_lane = data['points']
                if lane_id == 1:
                    id_r = lane_id
                    right_lane = data['points']
        im0, centers = self.visualize_lane_offset(im0, left_lane, right_lane, lane_width=0.42)
        # Show result
        if show_images:
            cv2.imshow("result", im0)
            cv2.waitKey(1)
        # Pixel → camera coordinates
        left_lane_cam = self.pixels_to_camera_lane(left_lane, "cam0_undis", "base_footprint")
        right_lane_cam = self.pixels_to_camera_lane(right_lane, "cam0_undis", "base_footprint")
        center = self.pixels_to_camera_lane(centers, "cam0_undis", "base_footprint")
        # Camera → base_footprint with Z=0
        # left_lane_base = self.camera_to_base_batch(left_lane_cam, force_z0=False)
        # right_lane_base = self.camera_to_base_batch(right_lane_cam, force_z0=False)

        # Publish to ROS2
        # self.publish_path(left_lane_base, 'base_footprint', self.left_lane_pub)
        # self.publish_path(right_lane_base, 'base_footprint', self.right_lane_pub)
        # Publish in camera frame
        self.publish_path(left_lane_cam, 'base_footprint', self.left_lane_cam_pub)
        self.publish_path(right_lane_cam, 'base_footprint', self.right_lane_cam_pub)
        self.publish_path(center, 'base_footprint', self.center_lane_pub)

        # self.compute_and_publish_center_lane(left_lane_cam, right_lane_cam, self.center_lane_pub, frame_id="base_footprint", lane_width=0.42)

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectUFLD()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
