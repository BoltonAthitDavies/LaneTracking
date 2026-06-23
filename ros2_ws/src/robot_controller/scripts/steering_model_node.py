#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math
from std_msgs.msg import Float64
import numpy as np

class SteeringModelNode(Node):
    def __init__(self):
        super().__init__('steering_model_node')
        self.get_logger().info("steering_model_node has been start.")
        # Sub
        self.subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        # Pub
        self.cmd_velocity = self.create_publisher(Float64, 'cmd_velocity', 10)
        self.cmd_steering = self.create_publisher(Float64, 'cmd_steering', 10)

        # Vehicle Parameters
        self.delta_steer = 0.001 # rad
        self.v = 0.0 # m/s
        self.omega = 0.0 # rad/s
        self.declare_parameter('wheelbase', 0.42)   # meters

    def cmd_vel_callback(self, msg:Twist):
        self.v = msg.linear.x
        self.omega = msg.angular.z
        wheelbase = self.get_parameter('wheelbase').value

        if self.omega == 0:
            self.delta_steer = 0.001
        else:
            self.delta_steer = math.atan(wheelbase * self.omega / self.v) if self.v != 0 else 0
            self.delta_steer = max(-0.6, min(self.delta_steer, 0.6))
            # Convert radian to degree
            self.delta_steer = math.degrees(self.delta_steer)

        # Create VelocityControllers for the robot velocity
        robot_velocity = Float64()
        robot_velocity.data = self.v
        # Publish the robot velocity
        self.cmd_velocity.publish(robot_velocity)

        # Create SteerControllers for the steer
        robot_steer_pose = Float64()
        robot_steer_pose.data = -self.delta_steer
        # Publish the robot steer
        self.cmd_steering.publish(robot_steer_pose)

def main(args=None):
    rclpy.init(args=args)
    node = SteeringModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()