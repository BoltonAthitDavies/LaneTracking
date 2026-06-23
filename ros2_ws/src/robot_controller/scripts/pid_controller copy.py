#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float64
import time

class PIDControllerNode(Node):
    def __init__(self):
        super().__init__('pid_controller_node')

        # --- Steering PID ---
        self.declare_parameter('kp_steer', 400.0)
        self.declare_parameter('ki_steer', 0.0)
        self.declare_parameter('kd_steer', 0.0)
        self.max_steer = 35.0
        self.min_steer = -35.0
        self.integral_steer = 0.0
        self.prev_error_steer = 0.0
        self.integral_limit_steer = 35.0

        # --- Speed PID ---
        self.declare_parameter('kp_speed', 3.0)
        self.declare_parameter('ki_speed', 0.0)
        self.declare_parameter('kd_speed', 0.0)
        self.declare_parameter('max_speed', 2.0)
        self.max_speed = self.get_parameter('max_speed').value
        self.min_speed = 0.0
        self.integral_speed = 0.0
        self.prev_error_speed = 0.0
        self.integral_limit_speed = 3.0

        # --- Control state ---
        self.offset = 0.0
        self.dt = 0.02  # 50 Hz
        self.last_offset_time = self.get_clock().now()

        # --- Subscribers ---
        self.offset_sub = self.create_subscription(
            Float32,
            '/lane_offset',
            self.offset_callback,
            10
        )

        # --- Publishers ---
        self.steer_pub = self.create_publisher(Float64, '/cmd_steering', 10)
        self.speed_pub = self.create_publisher(Float64, '/cmd_velocity', 10)

        # --- Timer ---
        self.create_timer(self.dt, self.timer_callback)

    # --- Callbacks ---
    def offset_callback(self, msg):
        self.offset = msg.data
        self.last_offset_time = self.get_clock().now()

    # --- PID computation ---
    def compute_steer(self):
        kp = self.get_parameter('kp_steer').value
        ki = self.get_parameter('ki_steer').value
        kd = self.get_parameter('kd_steer').value
        error = -self.offset  # want offset -> 0
        self.integral_steer += error * self.dt
        # Anti-windup
        self.integral_steer = max(min(self.integral_steer, self.integral_limit_steer),
                                  -self.integral_limit_steer)
        derivative = (error - self.prev_error_steer) / self.dt if self.dt > 0 else 0.0
        cmd = kp * error + ki * self.integral_steer + kd * derivative
        cmd = max(min(cmd, self.max_steer), self.min_steer)
        self.prev_error_steer = error
        self.steer_pub.publish(Float64(data=float(cmd)))
        return cmd

    def compute_speed(self):
        # Stop if no offset received recently (watchdog)
        # ==========================
        elapsed = (self.get_clock().now() - self.last_offset_time).nanoseconds * 1e-9
        if elapsed > 0.5:
            self.integral_speed = 0.0
            self.prev_error_speed = 0.0
            self.speed_pub.publish(Float64(data=0.0))
            return 0.0

        kp = self.get_parameter('kp_speed').value
        ki = self.get_parameter('ki_speed').value
        kd = self.get_parameter('kd_speed').value
        error = abs(self.offset)  # larger offset -> slower speed
        self.integral_speed += error * self.dt
        self.integral_speed = max(min(self.integral_speed, self.integral_limit_speed),
                                  -self.integral_limit_speed)
        derivative = (error - self.prev_error_speed) / self.dt if self.dt > 0 else 0.0
        cmd = self.max_speed - (kp * error + ki * self.integral_speed + kd * derivative)
        cmd = max(min(cmd, self.max_speed), self.min_speed)
        self.prev_error_speed = error
        # ==========================
        # cmd = 0.5
        # ==========================
        self.speed_pub.publish(Float64(data=float(cmd)))
        return cmd

    def timer_callback(self):
        steer_cmd = self.compute_steer()
        speed_cmd = self.compute_speed()
        self.get_logger().info(f"Offset: {self.offset:.3f} m | Steer: {steer_cmd:.2f} deg | Speed: {speed_cmd:.2f} m/s")

    def stop_robot(self):
        self.get_logger().info("Node stopping: setting speed to 0 for safety")
        self.speed_pub.publish(Float64(data=0.0))
        self.steer_pub.publish(Float64(data=0.0))

def main(args=None):
    rclpy.init(args=args)
    node = PIDControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
