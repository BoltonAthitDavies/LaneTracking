#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float64
import csv
import os
from datetime import datetime

class CascadePIDNode(Node):
    def __init__(self):
        super().__init__('cascade_pid_node')

        # Mode
        self.declare_parameter('tune', False)
        self.declare_parameter('target_speed', 0.864) # Best: 1.3, 1.72, 2.15, 2.6, 3.0
        self.declare_parameter('target_steer', 0.0)

        # --- Steering PID ---
        self.declare_parameter('kp_steer', 0.0) # should less than 0.01
        self.declare_parameter('ki_steer', 10.0)  # should be around 1.0 - 2.0
        self.declare_parameter('kd_steer', 0.0)  # Do not change
        self.declare_parameter('max_steer', 35.0)
        self.declare_parameter('min_steer', -35.0)
        self.integral_steer = 0.0
        self.prev_error_steer = 0.0
        self.declare_parameter('integral_limit_steer', 5.0)

        # --- Inner PID (speed) ---
        self.declare_parameter('kp_speed', 0.3) 
        self.declare_parameter('ki_speed', 1.3)  # 1.3 for constant speed  at 1.3 m/s
        self.declare_parameter('kd_speed', 0.0)  
        self.declare_parameter('max_speed', 1.3)
        self.declare_parameter('min_speed', 0.0)
        self.integral_speed = 0.0
        self.prev_error_speed = 0.0
        self.declare_parameter('integral_limit_speed', 1.05) # 1.0 for constant speed  at 1.3 m/s

        # --- Outer loop (offset -> target speed) ---
        self.declare_parameter('kp_outer_speed', 20.0)    # 3.0   # mapping offset -> speed reduction
        self.declare_parameter('kp_outer_steer', 500.0) 

        self.declare_parameter('log_feedback', False)

        # --- States ---
        self.offset = 0.0
        self.first_offset_time = None
        self.feedback_velocity = 0.0
        self.feedback_steering = 0.0
        self.dt = 0.02  # 50 Hz
        self.speed_error = 0.0
        self.steer_error  =  0.0
        self.offset_received = False
        self.target_speed= 0.0
        self.target_steer = 0.0

        # --- Subscribers ---
        self.create_subscription(Float32, '/lane_offset', self.offset_callback, 10)
        # self.create_subscription(Float64, 'feedback/velocity', self.velocity_callback, 10)
        self.create_subscription(Float64, 'feedback/velocity_filtered', self.velocity_callback, 10)
        self.create_subscription(Float64, 'feedback/steering', self.steering_callback, 10)

        # --- Publishers ---
        self.steer_pub = self.create_publisher(Float64, '/cmd_steering', 10)
        self.speed_pub = self.create_publisher(Float64, '/cmd_velocity', 10)

        # --- CSV Logging Setup ---
        # Creates a file named "pid_log_YYYYMMDD_HHMMSS.csv" in the current directory
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f'/home/ubuntu/LaneTracking/log_error_test/pid_log_{timestamp_str}.csv'
        
        # Write the header row
        with open(self.csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time_sec', 'offset', 'target_speed', 'feedback_velocity', 'speed_error', 'target_steer',  'feedback_steering', 'steer_error'])
            
        self.get_logger().info(f"CSV Logging started at: {self.csv_filename}")
        self.start_time_ns = self.get_clock().now().nanoseconds

        # --- Timer ---
        self.create_timer(self.dt, self.timer_callback)
        self.create_timer(0.1, self.log_callback)

    # ========== Callbacks ==========
    def offset_callback(self, msg):
        self.offset = msg.data
        # If this is the very first message, record the time
        if self.first_offset_time is None:
            self.first_offset_time = self.get_clock().now()
            self.get_logger().info(">> First offset received! Starting 1-second warm-up...")
            
    def velocity_callback(self, msg):
        self.feedback_velocity = msg.data

    def steering_callback(self, msg):
        self.feedback_steering = msg.data

    # ========== Outer loop ==========
    def compute_target_speed(self, offset):
        max_speed = self.get_parameter('max_speed').value
        min_speed = self.get_parameter('min_speed').value
        kp_outer_speed = self.get_parameter('kp_outer_speed').value

        error_offset = abs(offset)
        self.target_speed = max_speed - kp_outer_speed * error_offset
        self.target_speed = max(min(self.target_speed, max_speed), min_speed)
    
    def compute_target_steer(self, offset):
        kp_outer_steer = self.get_parameter('kp_outer_steer').value
        max_steer = self.get_parameter('max_steer').value
        min_steer = self.get_parameter('min_steer').value
        
        error_offset = -offset
        self.target_steer = kp_outer_steer * error_offset
        self.target_steer = max(min(self.target_steer, max_steer), min_steer)

    # ========== Inner loop ==========
    def compute_speed_pid(self):
        kp = self.get_parameter('kp_speed').value
        ki = self.get_parameter('ki_speed').value
        kd = self.get_parameter('kd_speed').value
        integral_limit_speed = self.get_parameter('integral_limit_speed').value
        max_speed = self.get_parameter('max_speed').value
        min_speed = self.get_parameter('min_speed').value

        self.speed_error = self.target_speed - self.feedback_velocity
        self.integral_speed += self.speed_error * self.dt
        self.integral_speed = max(min(self.integral_speed, integral_limit_speed), -integral_limit_speed)

        derivative = (self.speed_error - self.prev_error_speed) / self.dt if self.dt > 0 else 0.0
        cmd = kp * self.speed_error + ki * self.integral_speed + kd * derivative
        # clamp
        # cmd = max(min(cmd, max_speed), min_speed)

        self.prev_error_speed = self.speed_error

        return cmd

    def compute_steer_pid(self):
        kp = self.get_parameter('kp_steer').value
        ki = self.get_parameter('ki_steer').value
        kd = self.get_parameter('kd_steer').value
        integral_limit_steer = self.get_parameter('integral_limit_steer').value
        max_steer = self.get_parameter('max_steer').value
        min_steer = self.get_parameter('min_steer').value

        error = self.target_steer - self.feedback_steering
        self.integral_steer += error * self.dt
        self.integral_steer = max(min(self.integral_steer, integral_limit_steer), -integral_limit_steer)

        derivative = (error - self.prev_error_steer) / self.dt if self.dt > 0 else 0.0
        cmd = kp * error + ki * self.integral_steer + kd * derivative
        cmd = max(min(cmd, max_steer), min_steer)

        self.prev_error_steer = error
        return cmd

    def is_system_active(self):
        """Helper to check if we are past the offset wait and 3s delay."""
        if self.first_offset_time is None:
            return False
        
        elapsed = (self.get_clock().now() - self.first_offset_time).nanoseconds / 1e9
        return elapsed >= 0.5

    # ========== Main loop ==========
    def timer_callback(self):
        if self.first_offset_time is None:
            self.get_logger().warn("Waiting for /lane_offset to start...", throttle_duration_sec=1.0)
            return

        # elapsed = (self.get_clock().now() - self.first_offset_time).nanoseconds / 1e9
        # if elapsed < 0.5:
        #     self.get_logger().warn(f"Warm-up: {0.5 - elapsed:.1f}s remaining...", throttle_duration_sec=0.5)
        #     return
        
        tune = self.get_parameter('tune').value
        if tune:
            self.target_speed = self.get_parameter('target_speed').value
            self.target_steer = self.get_parameter('target_steer').value
        else:
            # --- Compute Target ---
            # self.compute_target_speed(self.offset)
            self.target_speed = self.get_parameter('target_speed').value
            self.compute_target_steer(self.offset)

        if self.target_steer <  0.1  and self.target_steer > -0.1:
            self.target_steer = 0.1    

        steer_cmd = self.compute_steer_pid()
        speed_cmd = self.compute_speed_pid()

        self.steer_error = self.target_steer - self.feedback_steering

        self.steer_pub.publish(Float64(data=self.target_steer))
        # self.steer_pub.publish(Float64(data=steer_cmd))
        self.speed_pub.publish(Float64(data=self.target_speed))

        if self.get_parameter('log_feedback').value:
            self.get_logger().info(f"============= Log ===============")
            self.get_logger().info(f"Offset:{self.offset:.4f}")
            self.get_logger().info(f"TgtSpd:{self.target_speed:.4f} | FbSpd:{self.feedback_velocity:.4f} | CmdSpd:{speed_cmd:.2f}")
            # self.get_logger().info(f"TgtSteer:{target_steer:.2f} | FbSteer:{self.feedback_steering} | CmdSteer:{steer_cmd:.2f}")
            self.get_logger().info(f"TgtSteer:{self.target_steer:.4f} | FbSteer:{self.feedback_steering}")
            self.get_logger().info(f"integral_speed: {self.integral_speed}")

    def log_callback(self):
        # if not self.is_system_active():
        #     return
        # Calculate elapsed time since node start
        current_time_ns = self.get_clock().now().nanoseconds
        elapsed_time_sec = (current_time_ns - self.start_time_ns) / 1e9
        
        try:
            # Open file in append mode
            with open(self.csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    f"{elapsed_time_sec:.4f}", 
                    f"{self.offset:.4f}", 
                    f"{self.target_speed:.4f}", 
                    f"{self.feedback_velocity:.4f}", 
                    f"{self.speed_error:.4f}", 
                    f"{self.target_steer:.4f}", 
                    f"{self.feedback_steering:.4f}", 
                    f"{self.steer_error:.4f}"
                ])
        except Exception as e:
            self.get_logger().warn(f"Could not write to CSV: {e}")




def main(args=None):
    rclpy.init(args=args)
    node = CascadePIDNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
