#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import yaml
import os
import time
from datetime import datetime

class OdomLogger(Node):
    def __init__(self):
        super().__init__('odom_logger')

        # === CONFIG: Path ที่จะเก็บไฟล์ ===
        save_dir = "/home/ubuntu/LaneTracking/odom_test"
        os.makedirs(save_dir, exist_ok=True)

        # สร้างไฟล์ใหม่พร้อม timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.filename = os.path.join(save_dir, f'odom_log_{timestamp}.yaml')

        self.get_logger().info(f"Logging odometry to {self.filename}")

        # Subscriber
        self.subscription = self.create_subscription(
            Odometry,
            '/odometry/yaw_rate',
            self.odom_callback,
            10)

        # เวลาเริ่มนับ
        self.start_time = self.get_clock().now().nanoseconds / 1e9

        # เขียน header ของไฟล์
        with open(self.filename, 'w') as f:
            f.write("odom_data:\n")

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if x is None or y is None:
            return

        # เวลา relative
        now = self.get_clock().now().nanoseconds / 1e9
        time_sec = round(now - self.start_time, 4)

        entry = {
            'time_sec': time_sec,
            'x': x,
            'y': y
        }

        # append ลงไฟล์ทันที (ไม่ต้องเก็บใน list)
        with open(self.filename, 'a') as f:
            yaml.safe_dump([entry], f, default_flow_style=False)

def main(args=None):
    rclpy.init(args=args)
    node = OdomLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
