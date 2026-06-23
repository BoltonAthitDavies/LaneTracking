#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import subprocess, time, os, signal

class CameraWatchdog(Node):
    def __init__(self):
        super().__init__('camera_watchdog')
        self.topic = '/cam0/image_raw'
        self.process_name = 'v4l2_camera_node'
        self.timeout = 2.0  # seconds without update before restart

        self.last_msg_time = self.get_clock().now()
        self.sub = self.create_subscription(Image, self.topic, self.cb, 10)
        self.timer = self.create_timer(1.0, self.check_timeout)

        self.get_logger().info("Camera Watchog Node initialized.")

    def cb(self, msg):
        self.last_msg_time = self.get_clock().now()

    def check_timeout(self):
        elapsed = (self.get_clock().now() - self.last_msg_time).nanoseconds * 1e-9
        if elapsed > self.timeout:
            self.get_logger().warn(f"No image on {self.topic} for {elapsed:.1f}s. Restarting camera node...")
            self.restart_camera()

    def restart_camera(self):
        # Kill existing node
        subprocess.call(['pkill', '-f', self.process_name])
        time.sleep(1)
        # Relaunch node (adjust command as needed)
        subprocess.Popen([
            'ros2', 'run', 'v4l2_camera', 'v4l2_camera_node',
            '--ros-args', '-r', f'/image_raw:={self.topic}',
            '-p', 'video_device:=/dev/video0',
            '-p', 'pixel_format:=UYVY',
            '-p', 'image_size:=[1920, 1080]',
            '-p', 'frame_id:=camera0_frame',
            '-p', 'output_encoding:=yuv422',
        ])
        self.get_logger().info("Restart camera.")
        self.last_msg_time = self.get_clock().now()

def main(args=None):
    rclpy.init(args=args)
    node = CameraWatchdog()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()