#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float64
import csv
import os
from datetime import datetime
import math

class CascadePIDNode(Node):
    def __init__(self):
        super().__init__('cascade_pid_node')

        # Mode
        self.declare_parameter('tune', False)
        self.declare_parameter('test_mode', False) # New Parameter
        self.declare_parameter('max_test_speed', 3.0) # New Parameter: Limit for ramp

        # --- Test Mode Parameters (Steer Ramp) ---
        self.declare_parameter('test_steer_min', -35.0) 
        self.declare_parameter('test_steer_max', 35.0) 
        self.declare_parameter('test_steer_step', 0.001) # Degrees change per tick (0.02s). 0.5 = 25 deg/sec
        # --- Steer Sine Wave  ---
        # self.declare_parameter('test_steer_freq', 0.5) # Frequency in Hz (0.5 means full cycle every 2 seconds)
        # self.declare_parameter('test_steer_amp', 35.0) # Amplitude in degrees (-35 to 35)

        self.declare_parameter('target_speed', 0.864) # Best: 1.3, 1.72, 2.15, 2.6, 3.0
        # self.declare_parameter('target_speed', 1.296) # Best: 1.3, 1.72, 2.15, 2.6, 3.0
        self.declare_parameter('target_steer', 0.0)
        self.declare_parameter('offset', 0.0)

        # --- Steering PID ---
        # self.declare_parameter('kp_steer', 166.7) # For test ramp
        self.declare_parameter('kp_steer', 500.0) # 550 for direction cw
        self.declare_parameter('ki_steer', 0.0)  
        self.declare_parameter('kd_steer', 0.0)  
        # FOR SPEED = 1.296
        # self.declare_parameter('kp_steer', 340.0)
        # self.declare_parameter('ki_steer', 0.0)
        # self.declare_parameter('kd_steer', 46.0)
        self.declare_parameter('max_steer', 35.0)
        self.declare_parameter('min_steer', -35.0)
        self.integral_steer = 0.0
        self.prev_error_steer = 0.0
        self.prev_derivative = 0.0
        self.alpha = 0.1
        self.declare_parameter('integral_limit_steer', 100.0)

        # --- Outer loop (offset -> target speed) ---
        # self.declare_parameter('kp_outer_steer', 500.0) 

        self.declare_parameter('log_feedback', False)
        # ros2 launch robot_bringup robot_lanedetect_control.launch.py kp_steer:=0.6 ki_steer:=0.0 kd_steer:=0.0
        # --- States ---
        self.offset = 0.0
        self.lane_center = 0.0
        self.image_center = 0.0
        self.offset_received = False
        self.feedback_velocity = 0.0
        self.feedback_steering = 0.0
        self.dt = 0.02  # 50 Hz
        # self.dt = 0.03  # 30 Hz
        self.speed_error = 0.0
        self.steer_error  =  0.0
        self.target_speed= 0.0
        self.target_steer = 0.0

        # --- Test Mode States ---
        self.current_ramp_speed = 0.0
        self.last_ramp_update_time = self.get_clock().now()

        # Initialize steer ramp at min value
        self.current_test_steer = -35.0 
        self.test_steer_dir = 1.0 # 1.0 for Up, -1.0 for Down

        # --- Subscribers ---
        self.create_subscription(Float32, '/lane_offset', self.offset_callback, 10)
        self.create_subscription(Float32, '/lane_center', self.lane_center_callback, 10)
        self.create_subscription(Float32, '/image_center', self.image_center_callback, 10)

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
            writer.writerow(['time_sec', 'offset', 'target_speed', 'feedback_velocity', 'speed_error', 'target_steer',  'feedback_steering', 'steer_error', 'lane_center', 'image_center'])
            
        self.get_logger().info(f"CSV Logging started at: {self.csv_filename}")
        self.start_time_ns = self.get_clock().now().nanoseconds

        # --- Timer ---
        self.create_timer(self.dt, self.timer_callback)
        self.create_timer(0.1, self.log_callback)

    # ========== Callbacks ==========
    def offset_callback(self, msg):
        self.offset = msg.data
        # Check first message
        self.offset_received = True

    def lane_center_callback(self, msg):
        self.lane_center = msg.data

    def image_center_callback(self, msg):
        self.image_center = msg.data
            
    def velocity_callback(self, msg):
        self.feedback_velocity = msg.data

    def steering_callback(self, msg):
        self.feedback_steering = msg.data

    # ========== Outer loop ==========
    # def compute_target_steer(self):
    #     kp_outer_steer = self.get_parameter('kp_outer_steer').value
    #     max_steer = self.get_parameter('max_steer').value
    #     min_steer = self.get_parameter('min_steer').value
        
    #     error_offset = -self.offset
    #     self.target_steer = kp_outer_steer * error_offset
    #     self.target_steer = max(min(self.target_steer, max_steer), min_steer)

    def compute_steer(self):
        kp = self.get_parameter('kp_steer').value
        ki = self.get_parameter('ki_steer').value
        kd = self.get_parameter('kd_steer').value
        integral_limit_steer = self.get_parameter('integral_limit_steer').value
        max_steer = self.get_parameter('max_steer').value
        min_steer = self.get_parameter('min_steer').value

        error = -self.offset  # want offset -> 0
        self.integral_steer += error * self.dt
        # Anti-windup
        self.integral_steer = max(min(self.integral_steer, integral_limit_steer),
                                  -integral_limit_steer)
        
        # derivative = (error - self.prev_error_steer) / self.dt if self.dt > 0 else 0.0
        # -- New  Code --
        # --- D Term (Modified with Low-Pass Filter) ---
        if self.dt > 0:
            raw_derivative = (error - self.prev_error_steer) / self.dt
        else:
            raw_derivative = 0.0

        # ใช้สูตร Exponential Moving Average (Low-Pass Filter)
        # alpha ประมาณ 0.1 - 0.3 กำลังดีสำหรับงาน Vision
        alpha = 0.2 
        filtered_derivative = alpha * raw_derivative + (1 - alpha) * self.prev_derivative
        
        # อัปเดตค่าเก่าเก็บไว้
        self.prev_derivative = filtered_derivative
        self.prev_error_steer = error

        # คำนวณ PID โดยใช้ filtered_derivative แทน
        self.target_steer = kp * error + ki * self.integral_steer + kd * filtered_derivative
        # -- End Code --

        # self.target_steer = kp * error + ki * self.integral_steer + kd * derivative
        self.target_steer = max(min(self.target_steer, max_steer), min_steer)
        self.prev_error_steer = error

    def is_system_active(self):
        """Helper to check if we are past the offset wait and 3s delay."""
        if self.first_offset_time is None:
            return False
        
        elapsed = (self.get_clock().now() - self.first_offset_time).nanoseconds / 1e9
        return elapsed >= 0.5

    # ========== Main loop ==========
    def timer_callback(self):
        tune = self.get_parameter('tune').value
        test_mode = self.get_parameter('test_mode').value

        if test_mode == False and tune == False:
            if self.offset_received == False:
                self.get_logger().warn("Waiting for /lane_offset to start...", throttle_duration_sec=1.0)
                return
        
        # Priority 1: Test Mode (Ramp Generator)
        if test_mode:
            # 1. Speed Ramp Logic
            # max_test_speed = self.get_parameter('max_test_speed').value
            # now = self.get_clock().now()
            
            # # Calculate time difference in seconds
            # dt_ramp = (now - self.last_ramp_update_time).nanoseconds / 1e9
            
            # # Every 0.1 seconds, increase speed by 0.001
            # if dt_ramp >= 0.1:
            #     if self.current_ramp_speed < max_test_speed:
            #         self.current_ramp_speed += 0.001
            #         # Clamp to max speed
            #         if self.current_ramp_speed > max_test_speed:
            #             self.current_ramp_speed = max_test_speed
                
            #     self.last_ramp_update_time = now
            
            # self.target_speed = self.current_ramp_speed
            # self.compute_steer()
            # ----------------------------------------------------------------------------
            # 2. Sine Wave Steering Logic
            # Calculate total time elapsed since start
            # current_time_ns = self.get_clock().now().nanoseconds
            # t = (current_time_ns - self.start_time_ns) / 1e9
            
            # freq = self.get_parameter('test_steer_freq').value
            # amp = self.get_parameter('test_steer_amp').value
            
            # # Formula: amp * sin(2 * pi * f * t)
            # self.target_steer = amp * math.sin(2.0 * math.pi * freq * t)
            # ----------------------------------------------------------------------------
            # 3. Steering Ramp Logic (Triangle Wave)
            s_min = self.get_parameter('test_steer_min').value
            s_max = self.get_parameter('test_steer_max').value
            s_step = self.get_parameter('test_steer_step').value

            # Increment/Decrement steer
            self.current_test_steer += (s_step * self.test_steer_dir)

            # Check bounds and flip direction
            if self.current_test_steer >= s_max:
                self.current_test_steer = s_max
                self.test_steer_dir = -1.0 # Go Down
            elif self.current_test_steer <= s_min:
                self.current_test_steer = s_min
                self.test_steer_dir = 1.0 # Go Up
            
            self.target_steer = self.current_test_steer

        # Priority 2: Tune Mode (Static params + manual offset injection)
        elif tune:
            self.target_speed = self.get_parameter('target_speed').value
            # self.target_steer = self.get_parameter('target_steer').value
            self.offset = self.get_parameter('offset').value
            self.compute_steer()
            
        # Priority 3: Normal Operation
        else:
            self.target_speed = self.get_parameter('target_speed').value
            self.compute_steer()

        # Deadzone fix
        # if self.target_steer <  0.1  and self.target_steer > -0.1:
        #     self.target_steer = 0.1    

        self.steer_error = self.target_steer - self.feedback_steering
        self.speed_error = self.target_speed - self.feedback_velocity

        self.steer_pub.publish(Float64(data=self.target_steer))
        self.speed_pub.publish(Float64(data=self.target_speed))

        if self.get_parameter('log_feedback').value:
            self.get_logger().info(f"============= Log ===============")
            self.get_logger().info(f"Offset:{self.offset:.4f}")
            self.get_logger().info(f"TgtSpd:{self.target_speed:.4f} | FbSpd:{self.feedback_velocity:.4f}")
            self.get_logger().info(f"TgtSteer:{self.target_steer:.4f} | FbSteer:{self.feedback_steering}")
            self.get_logger().info(f"integral_steer: {self.integral_steer}")

    def log_callback(self):
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
                    f"{self.steer_error:.4f}",
                    f"{self.lane_center:.4f}",
                    f"{self.image_center:.4f}"
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
