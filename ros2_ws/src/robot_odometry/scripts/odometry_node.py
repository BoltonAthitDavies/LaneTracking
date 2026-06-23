#!/usr/bin/python3

from robot_odometry.dummy_module import dummy_function, dummy_var
import rclpy
from rclpy.node import Node
import transforms3d as tf
import math
import tf2_ros
import numpy as np
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Path
from geometry_msgs.msg import TransformStamped, Vector3
from std_srvs.srv import Empty
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')

        # Vehicle Parameters
        self.declare_parameter('wheelbase', 0.463)   # meters
        self.declare_parameter('track_width', 0.345) # meters
        self.declare_parameter('steering_ratio', 1.0)
        self.wheelbase = self.get_parameter('wheelbase').value
        self.track_width = self.get_parameter('track_width').value
        self.steering_ratio = self.get_parameter('steering_ratio').value

        # State variables
        # self.steering = 0.0
        self.steering_deg = 0.0
        self.steering_rad = 0.0
        self.velocity = 0.0
        self.BETA = 0.0  # Assuming no lateral slip
        self.yaw_pose = 0.0
        self.orientation = [0.0, 0.0, 0.0, 1.0]

        self.x_curr = 0.0 # m
        self.y_curr = 0.0 # m
        self.v_curr = 0.0 # m/s
        self.w_curr = 0.0 # rad/s
        self.yaw_rate = 0.0 # rad/s
        self.theta_curr = 0.0 # rad

        self.x_curr_1Track = 0.0 # m
        self.y_curr_1Track = 0.0 # m
        self.v_curr_1Track = 0.0 # m/s
        self.w_curr_1Track = 0.0 # rad/s
        self.theta_curr_1Track = 0.0  # rad

        self.x_curr_2Track = 0.0 # m
        self.y_curr_2Track = 0.0 # m
        self.v_curr_2Track = 0.0 # m/s
        self.w_curr_2Track = 0.0 # rad/s
        self.theta_curr_2Track = 0.0 # rad

        self.x_gt = 0.0
        self.y_gt = 0.0
        self.quat_gt = []

        # ROS 2 subscriptions
        # self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Float64, 'feedback/velocity', self.velocity_callback, 10)
        self.create_subscription(Float64, 'feedback/steering', self.steering_callback, 10)
        self.create_subscription(Imu, '/hwt101ct_yaw_publisher', self.imu_callback, 10)

        # ROS 2 publishers
        self.yaw_rate_publisher = self.create_publisher(Odometry, '/odometry/yaw_rate', 10)
        self.single_track_publisher = self.create_publisher(Odometry, '/odometry/single_track', 10)
        self.double_track_publisher = self.create_publisher(Odometry, '/odometry/double_track', 10)
        
        # TF broadcaster
        # self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Timer to update odometry
        self.dt = 0.02 # 50 Hz
        self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info("odometry_node has been start.")

    def imu_callback(self, msg:Imu):
        """ Callback to get yaw rate """
        self.yaw_rate = msg.angular_velocity.z  # Rotational velocity around Z-axis

        # Extract quaternion from ROS message
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w
        # self.orientation = [qx, qy, qz, qw]
        # transforms3d expects [w, x, y, z]
        quat_tf3d = [qw, qx, qy, qz]

        # Convert to euler angles (roll, pitch, yaw)
        roll, pitch, yaw = tf.euler.quat2euler(quat_tf3d, axes='sxyz')
        self.yaw_pose = yaw

    def velocity_callback(self, msg: Float64):
        self.velocity = msg.data * (1.0 / 3.0)
        # self.get_logger().info(f"Received Velocity: {self.velocity:.2f}")

    def steering_callback(self, msg: Float64):
        self.steering_deg = -msg.data
        self.steering_rad = math.radians(self.steering_deg)   # ✅ convert degrees → radians
        # self.get_logger().info(f"Steering: {self.steering_deg:.2f}° = {self.steering_rad:.4f} rad")
    
    def timer_callback(self):
        # Publish Odom (odom -> base_footprint)
        self.OdoYawRate()
        # self.Odo1Track()
        # self.Odo2Track()
        # self.get_logger().info(f'x_pose: {self.x_curr}')
        # self.get_logger().info(f'y_pose: {self.y_curr}')
        # self.get_logger().info(f'x_linear_velo: {self.velocity}')
        # self.get_logger().info(f'theta: {self.theta_curr}')
        self.publish_tf(self.x_curr, self.y_curr, self.quaternion)
    
    def OdoYawRate(self):
        # Compute new pose        
        self.x_curr = self.x_curr + (self.velocity * self.dt * math.cos(self.theta_curr + self.BETA + ((self.yaw_rate * self.dt) / 2)))
        self.y_curr = self.y_curr + (self.velocity * self.dt * math.sin(self.theta_curr + self.BETA + ((self.yaw_rate * self.dt) / 2)))
        self.theta_curr = self.theta_curr + (self.yaw_rate * self.dt)
        quaternion = tf.quaternions.axangle2quat([0, 0, 1], self.theta_curr)
        self.quaternion = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]

        # Publish odometry message
        self.publish_odom("odom", "base_footprint", self.x_curr, self.y_curr, self.quaternion, self.velocity, self.yaw_rate, self.yaw_rate_publisher)

    def Odo1Track(self):
        # Compute new pose  
        self.x_curr_1Track = self.x_curr_1Track + (self.velocity * self.dt * math.cos(self.theta_curr_1Track + self.BETA + ((self.w_curr_1Track * self.dt) / 2)))
        self.y_curr_1Track = self.y_curr_1Track + (self.velocity * self.dt * math.sin(self.theta_curr_1Track + self.BETA + ((self.w_curr_1Track * self.dt) / 2)))
        self.theta_curr_1Track = self.theta_curr_1Track + (self.w_curr_1Track * self.dt)
        quaternion = tf.quaternions.axangle2quat([0, 0, 1], self.theta_curr_1Track)
        self.quaternion_1Track = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        self.w_curr_1Track = (self.velocity / self.wheelbase) * math.tan(self.steering_rad)

        # Publish odometry message
        self.publish_odom("odom", "base_footprint", self.x_curr_1Track, self.y_curr_1Track, self.quaternion_1Track, self.velocity, self.w_curr_1Track, self.single_track_publisher)

    # def Odo2Track(self):
    #     # Compute new pose  
    #     self.x_curr_2Track = self.x_curr_2Track + (self.v_curr_2Track * self.dt * math.cos(self.theta_curr_2Track + self.BETA + ((self.w_curr_2Track * self.dt) / 2)))
    #     self.y_curr_2Track = self.y_curr_2Track + (self.v_curr_2Track * self.dt * math.sin(self.theta_curr_2Track + self.BETA + ((self.w_curr_2Track * self.dt) / 2)))
    #     self.theta_curr_2Track = self.theta_curr_2Track + (self.w_curr_2Track * self.dt)
    #     self.quaternion_2Track = tf_transformations.quaternion_from_euler(0.0, 0.0, self.theta_curr_2Track)
    #     self.v_curr_2Track = self.velocity
    #     self.w_curr_2Track = (self.v_rr - self.v_rl) / self.track_width

    #     # Publish odometry message
    #     self.publish_odom("odom", "base_footprint", self.x_curr_2Track, self.y_curr_2Track, self.quaternion_2Track, self.v_curr_2Track, self.w_curr_2Track, self.double_track_publisher)

    def publish_odom(self, frame_id, child_frame_id, pose_x, pose_y, quaternion_list, v_curr, w_curr, publisher):
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = frame_id
        odom_msg.child_frame_id = child_frame_id

        # Position
        odom_msg.pose.pose.position.x = pose_x
        odom_msg.pose.pose.position.y = pose_y
        odom_msg.pose.pose.position.z = 0.0
        
        # Orientation (Quaternion from Yaw)
        odom_msg.pose.pose.orientation.x = quaternion_list[0]
        odom_msg.pose.pose.orientation.y = quaternion_list[1]
        odom_msg.pose.pose.orientation.z = quaternion_list[2]
        odom_msg.pose.pose.orientation.w = quaternion_list[3]

        # Twist
        odom_msg.twist.twist.linear = Vector3(x=v_curr, y=0.0, z=0.0)
        odom_msg.twist.twist.angular = Vector3(x=0.0, y=0.0, z=w_curr)

        # Publish odometry
        publisher.publish(odom_msg)    


    def publish_tf(self, pose_x, pose_y, quaternion_list):
        """ Publishes the transformation from 'odom' to 'base_footprint' """

        if len(quaternion_list) != 4:
            return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"

        # Position
        t.transform.translation.x = pose_x
        t.transform.translation.y = pose_y
        t.transform.translation.z = 0.0

        # Orientation
        t.transform.rotation.x = quaternion_list[0]
        t.transform.rotation.y = quaternion_list[1]
        t.transform.rotation.z = quaternion_list[2]
        t.transform.rotation.w = quaternion_list[3]

        # Publish TF transform
        # self.tf_broadcaster.sendTransform(t)
        self.tf_static_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
