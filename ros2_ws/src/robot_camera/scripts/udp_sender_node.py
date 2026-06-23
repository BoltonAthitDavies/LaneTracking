#!/usr/bin/python3
import rclpy
from rclpy.node import Node
import socket
import subprocess  # <--- NEW: To run external commands (like ping)
import platform    # <--- NEW: To check the Operating System

class UdpSenderNode(Node):

    def __init__(self):
        super().__init__('udp_sender_node')
        
        # --- Configuration ---
        self.target_ip = '192.168.1.200'
        self.target_port = 12345
        timer_period = 1.0
        # --- End Configuration ---

        # --- NEW: Check host reachability before starting ---
        if not self.check_host_reachability():
            self.get_logger().error(f'Host {self.target_ip} is NOT REACHABLE. Shutting down.')
            # We use a timer to shutdown cleanly after init is done
            self.create_timer(0.1, self.shutdown_node)
            return
        # --- End NEW ---
            
        self.get_logger().info(f'Host {self.target_ip} is reachable. Starting UDP sender.')

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as e:
            self.get_logger().error(f'Failed to create socket: {e}')
            self.shutdown_node() # Use our shutdown helper
            return

        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.counter = 0

    def shutdown_node(self):
        """Helper function to shut down the rclpy context."""
        rclpy.shutdown()

    def check_host_reachability(self):
        """
        Pings the target IP once to see if it's reachable.
        Returns True if reachable, False otherwise.
        """
        self.get_logger().info(f'Pinging {self.target_ip}...')
        
        # Build the ping command based on the OS
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        command = ['ping', param, '1', self.target_ip]
        
        try:
            # Run the command and hide the output
            response = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Check the return code. 0 means success.
            return response.returncode == 0
            
        except FileNotFoundError:
            self.get_logger().error("Ping command not found. Cannot check host reachability.")
            return False # Assume it's not reachable if we can't even ping

    def timer_callback(self):
        message = f"Hello from ROS 2! Count: {self.counter}"
        
        try:
            self.sock.sendto(message.encode('utf-8'), (self.target_ip, self.target_port))
            self.get_logger().info(f'Sent: "{message}"')
            self.counter += 1
            
        except socket.error as e:
            # This is your *runtime* check.
            # If the host becomes unreachable, you'll get an error here.
            # e.g., [Errno 113] No route to host
            self.get_logger().warn(f'Socket error while sending: {e}')
            # You could choose to stop the timer or shut down here
            # self.timer.cancel()
            # self.shutdown_node()

    def destroy_node(self):
        # Clean up the socket
        if hasattr(self, 'sock'): # Check if socket exists before closing
            self.get_logger().info('Closing socket.')
            self.sock.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    udp_sender_node = UdpSenderNode()
    
    try:
        rclpy.spin(udp_sender_node)
    except KeyboardInterrupt:
        pass
    except rclpy.executors.ExternalShutdownException:
        # This will be raised when we call rclpy.shutdown() from inside the node
        pass
    finally:
        udp_sender_node.destroy_node()
        # Only shutdown if it's not already shutting down
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()