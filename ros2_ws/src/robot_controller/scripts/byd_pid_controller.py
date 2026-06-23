#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float64
import time
import socket     # <--- NEW
import threading  # <--- NEW
import subprocess # <--- NEW
import platform   # <--- NEW

class PIDControllerNode(Node):
    def __init__(self):
        super().__init__('pid_controller_node')

        # --- Steering PID ---
        self.declare_parameter('kp_steer', 400.0)
        self.declare_parameter('ki_steer', 0.0)
        self.declare_parameter('kd_steer', 0.0)
        self.declare_parameter('max_steer', 35.0)
        self.declare_parameter('min_steer', -35.0)
        self.declare_parameter('command', "EXTERNAL")
        self.declare_parameter('log_feedback', True)
        self.integral_steer = 0.0
        self.prev_error_steer = 0.0
        self.integral_limit_steer = 35.0

        # --- Speed PID ---
        self.declare_parameter('constant_speed', 1.0)

        # --- Control state ---
        self.offset = 0.0
        self.dt = 0.02  # 50 Hz
        self.last_offset_time = self.get_clock().now()
        
        # --- NEW: UDP Setup ---
        self.udp_ip = "192.168.1.200"
        self.udp_port = 12345  # Using a different port (12345 was for offset)
        self.udp_socket = None
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.get_logger().info(f"UDP Sender Initialized: Sending to {self.udp_ip}:{self.udp_port}")
        except socket.error as e:
            self.get_logger().error(f"Failed to create UDP socket: {e}")

        # --- NEW: Ping Check Setup ---
        self.host_reachable = False # State variable
        self.start_ping_thread()

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

        self.get_logger().info(f"✅ PID Controller Node has been start")

    # --- Callbacks ---
    def offset_callback(self, msg):
        self.offset = msg.data
        self.get_logger().info(f"offset: {self.offset}")
        # self.last_offset_time = self.get_clock().now()

    # --- PID computation ---
    def compute_steer(self):
        kp = self.get_parameter('kp_steer').value
        ki = self.get_parameter('ki_steer').value
        kd = self.get_parameter('kd_steer').value
        max_steer = self.get_parameter('max_steer').value
        min_steer = self.get_parameter('min_steer').value

        error = -self.offset  # want offset -> 0
        # self.get_logger().info(f"offset: {self.offset}")

        self.integral_steer += error * self.dt
        self.integral_steer = max(min(self.integral_steer, self.integral_limit_steer),
                                  -self.integral_limit_steer)
        derivative = (error - self.prev_error_steer) / self.dt if self.dt > 0 else 0.0
        cmd = kp * error + ki * self.integral_steer + kd * derivative
        cmd = max(min(cmd, max_steer), min_steer)
        self.steer_pub.publish(Float64(data=float(cmd)))
        return cmd

    def compute_speed(self):
        cmd = self.get_parameter('constant_speed').value
        self.speed_pub.publish(Float64(data=float(cmd)))
        return cmd

    # --- NEW: Ping Thread Logic ---
    def check_ping_command(self):
        """Checks if the 'ping' command is available on this system."""
        try:
            # Check for ping command, hide output
            subprocess.run(['ping', '-c', '1', '127.0.0.1'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            return False

    def start_ping_thread(self):
        if not self.check_ping_command():
            self.get_logger().warn("--- 'ping' command not found. ---")
            self.get_logger().warn("Host reachability check is DISABLED.")
            self.get_logger().warn("To enable, 'apt-get install iputils-ping' in your Docker container.")
            return

        # Start the thread as a daemon so it exits when the main node exits
        self.get_logger().info(f"Starting continuous ping thread for {self.udp_ip}")
        ping_thread = threading.Thread(target=self.ping_thread_loop, daemon=True)
        ping_thread.start()

    def ping_thread_loop(self):
        """Continuously pings the host in a separate thread."""
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        command = ['ping', param, '1', self.udp_ip]
        
        while rclpy.ok():
            try:
                response = subprocess.run(command, 
                                          stdout=subprocess.DEVNULL, 
                                          stderr=subprocess.DEVNULL)
                
                is_reachable = (response.returncode == 0)
                
                if is_reachable and not self.host_reachable:
                    self.get_logger().info(f"Host {self.udp_ip} is NOW REACHABLE.")
                elif not is_reachable and self.host_reachable:
                    self.get_logger().warn(f"Host {self.udp_ip} has BECOME UNREACHABLE.")
                    
                self.host_reachable = is_reachable
                
            except Exception as e:
                self.get_logger().error(f"Ping thread exception: {e}")
                self.host_reachable = False
            
            time.sleep(1.0) # Ping once per second

    # --- Main Timer ---
    def timer_callback(self):
        steer_cmd = self.compute_steer()
        speed_cmd = self.compute_speed()
        command = self.get_parameter('command').value
        log_feedback = self.get_parameter('log_feedback').value

        
        # --- NEW: Send commands over UDP ---
        if self.udp_socket is not None:
            try:
                # Format: "steer,speed"  e.g., "32.1234,0.5000"
                message = f"{-steer_cmd:.4f},{speed_cmd:.4f},{command}"
                self.udp_socket.sendto(message.encode('utf-8'), (self.udp_ip, self.udp_port))
            
            except socket.error as e:
                # This is the *most important* check. 
                # If this fails, the send failed.
                if self.host_reachable: # Only log if we *thought* it was reachable
                   self.get_logger().warn(f"UDP send error (host may be down): {e}")
                self.host_reachable = False # Sending failed, so it's not reachable
        # --- END NEW ---

        # Optional: Log the status
        if log_feedback:
            self.get_logger().info(f"Offset: {self.offset:.3f} | Steer: {steer_cmd:.2f} rad | Speed: {speed_cmd:.2f} m/s | Mode: {command} | Reachable: {self.host_reachable}")


    def stop_robot(self):
        self.get_logger().info("Node stopping: setting speed to 0 for safety")
        self.speed_pub.publish(Float64(data=0.0))
        self.steer_pub.publish(Float64(data=0.0))
        
        # --- NEW: Close socket ---
        if self.udp_socket:
            self.get_logger().info("Closing UDP socket.")
            self.udp_socket.close()

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