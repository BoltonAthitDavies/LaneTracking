#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import struct
import threading
import socket
import subprocess
import time
import os
import can

CAN_FRAME_FMT = "=IB3x8s"
CAN_FRAME_SIZE = 16

class RobotCANInterface(Node):
    def __init__(self):
        super().__init__('robot_can_interface')
        
        self.declare_parameter('can_interface', 'can0')
        self.declare_parameter('interface_type', 'canable')
        self.declare_parameter('canable_port', '/dev/ttyCANable')
        self.declare_parameter('control_period_ms', 10)
        self.declare_parameter('velocity_limit', 5.0)
        self.declare_parameter('steering_limit', 40.0)
        self.declare_parameter('log_feedback', False)
        
        self.can_interface = self.get_parameter('can_interface').value
        self.interface_type = self.get_parameter('interface_type').value
        self.canable_port = self.get_parameter('canable_port').value
        self.control_period_ms = self.get_parameter('control_period_ms').value
        self.velocity_limit = self.get_parameter('velocity_limit').value
        self.steering_limit = self.get_parameter('steering_limit').value
        self.log_feedback = self.get_parameter('log_feedback').value
        
        self.control_can_id = 0x202
        self.feedback_can_id = 0x182
        
        self.use_socketcan = (self.interface_type == 'socketcan')
        self.initialization_successful = False
        
        try:
            if self.use_socketcan:
                self.init_socketcan()
            else:
                self.init_canable()
            
            self.initialization_successful = True
            self.get_logger().info(f'CAN interface initialized: {self.interface_type}')
            self.get_logger().info(f'Control period: {self.control_period_ms}ms ({1000/self.control_period_ms:.1f}Hz)')
            self.get_logger().info(f'TX CAN ID: 0x{self.control_can_id:03X}, RX CAN ID: 0x{self.feedback_can_id:03X}')
            self.get_logger().info(f'Feedback logging: {"enabled" if self.log_feedback else "disabled"}')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize CAN interface: {e}')
            if not self.use_socketcan:
                self.get_logger().error('CANable connection failed. Node will shut down.')
                self.get_logger().error('Please check:')
                self.get_logger().error('  1. CANable is connected')
                self.get_logger().error('  2. Device permissions (sudo chmod 666 /dev/ttyACM*)')
                self.get_logger().error('  3. python-can is installed (pip install python-can)')
                return
            raise
        
        self.cmd_velocity = 0.0
        self.cmd_steering = 0.1
        self.cmd_lock = threading.Lock()

        self.running = True
        self.tx_thread = None
        self.rx_thread = None
        
        if not self.initialization_successful:
            self.get_logger().error('Initialization failed. Node will not start.')
            return
        
        self.velocity_sub = self.create_subscription(
            Float64,
            'cmd_velocity',
            self.cmd_velocity_callback,
            10
        )
        
        self.steering_sub = self.create_subscription(
            Float64,
            'cmd_steering',
            self.cmd_steering_callback,
            10
        )
        
        self.feedback_velocity_pub = self.create_publisher(Float64, 'feedback/velocity', 10)
        self.feedback_steering_pub = self.create_publisher(Float64, 'feedback/steering', 10)
        
        self.tx_thread = threading.Thread(target=self.can_transmit_thread, daemon=True)
        self.rx_thread = threading.Thread(target=self.can_receive_thread, daemon=True)
        self.tx_thread.start()
        self.rx_thread.start()
        
        self.get_logger().info('robot CAN Interface Node initialized')
        self.get_logger().info('Subscribing to: cmd_velocity (Float64) and cmd_steering (Float64)')
    
    def init_socketcan(self):
        self.setup_can_interface()
        
        self.can_tx_socket = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.can_tx_socket.bind((self.can_interface,))
        
        self.can_rx_socket = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.can_rx_socket.bind((self.can_interface,))
        self.can_rx_socket.settimeout(0.001)
        
        self.get_logger().info(f'SocketCAN interface {self.can_interface} initialized')
    
    def init_canable(self):
        device = self.canable_port
        if not os.path.exists(device):
            for port in ['/dev/ttyCANable', '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2']:
                if os.path.exists(port):
                    device = port
                    break
            else:
                raise Exception(f"CANable device not found at {self.canable_port}")
        
        self.get_logger().info(f"Connecting to CANable at {device}...")
        
        try:
            self.can_bus = can.Bus(interface='slcan', channel=device, bitrate=500000)
            self.get_logger().info(f"Connected via slcan on {device}")
        except Exception as e:
            self.get_logger().warn(f"SLCAN failed: {e}, trying serial interface...")
            try:
                self.can_bus = can.Bus(interface='serial', channel=device, bitrate=500000)
                self.get_logger().info(f"Connected via serial on {device}")
            except Exception as e:
                raise Exception(f"Failed to connect to CANable: {e}")
    
    def setup_can_interface(self):
        try:
            result = subprocess.run(['ip', 'link', 'show', self.can_interface], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                self.get_logger().error(f'CAN interface {self.can_interface} not found')
                raise Exception(f'CAN interface {self.can_interface} not found')
            
            if 'UP' not in result.stdout:
                self.get_logger().warn(f'CAN interface {self.can_interface} is down, attempting to bring it up...')
                subprocess.run(['sudo', 'ip', 'link', 'set', self.can_interface, 'type', 'can', 'bitrate', '500000'])
                subprocess.run(['sudo', 'ip', 'link', 'set', self.can_interface, 'up'])
        except Exception as e:
            self.get_logger().error(f'Failed to setup CAN interface: {e}')
            raise
    
    def cmd_velocity_callback(self, msg):
        with self.cmd_lock:
            self.cmd_velocity = max(-self.velocity_limit, min(self.velocity_limit, msg.data))
    
    def cmd_steering_callback(self, msg):
        with self.cmd_lock:
            self.cmd_steering = max(-self.steering_limit, min(self.steering_limit, msg.data))
    
    def send_can_message(self, velocity, steering):
        if self.use_socketcan:
            velocity_bytes = struct.pack('<f', velocity)
            steering_bytes = struct.pack('<f', steering)
            
            data = velocity_bytes + steering_bytes
            
            can_id = self.control_can_id
            can_dlc = 8
            can_data = data.ljust(8, b'\x00')
            frame = struct.pack(CAN_FRAME_FMT, can_id, can_dlc, can_data)
            
            self.can_tx_socket.send(frame)
        else:
            data = struct.pack('<ff', velocity, steering)
            msg = can.Message(arbitration_id=self.control_can_id, data=data, is_extended_id=False)
            self.can_bus.send(msg)
    
    def can_transmit_thread(self):
        period_s = self.control_period_ms / 1000.0
        next_send_time = time.monotonic()
        
        while self.running:
            try:
                with self.cmd_lock:
                    velocity = self.cmd_velocity
                    steering = self.cmd_steering
                
                self.send_can_message(velocity, steering)
                
                next_send_time += period_s
                sleep_time = next_send_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_send_time = time.monotonic()
                    
            except Exception as e:
                self.get_logger().error(f'CAN transmit error: {e}')
                time.sleep(0.001)
    
    def can_receive_thread(self):
        while self.running:
            try:
                if self.use_socketcan:
                    frame, _ = self.can_rx_socket.recvfrom(CAN_FRAME_SIZE)
                    
                    can_id, can_dlc, can_data = struct.unpack(CAN_FRAME_FMT, frame)
                    
                    actual_can_id = can_id & 0x1FFFFFFF
                    
                    if actual_can_id == self.feedback_can_id and can_dlc == 8:
                        self.process_feedback(can_data)
                else:
                    msg = self.can_bus.recv(timeout=0.001)
                    
                    if msg and msg.arbitration_id == self.feedback_can_id and len(msg.data) == 8:
                        self.process_feedback(msg.data)
                        
            except socket.timeout:
                pass
            except Exception as e:
                if self.running and 'Resource temporarily unavailable' not in str(e):
                    self.get_logger().error(f'CAN receive error: {e}')
    
    def process_feedback(self, data):
        feedback_velocity = struct.unpack('<f', data[0:4])[0]
        feedback_steering = struct.unpack('<f', data[4:8])[0]
        
        if self.log_feedback:
            self.get_logger().info(
                f'\n       > Feedback steering angle [deg]: {feedback_steering:.6f}\n'
                f'       > Feedback speed [m/s]: {feedback_velocity:.6f}'
            )
        
        vel_msg = Float64()
        vel_msg.data = feedback_velocity# * (1.0/3.0)
        self.feedback_velocity_pub.publish(vel_msg)
        
        steer_msg = Float64()
        steer_msg.data = feedback_steering
        self.feedback_steering_pub.publish(steer_msg)
    
    def destroy_node(self):
        self.running = False
        
        try:
            self.send_can_message(0.0, 0.0)
            self.get_logger().info('Sent stop command')
        except:
            pass
        
        if self.tx_thread:
            self.tx_thread.join(timeout=1.0)
        if self.rx_thread:
            self.rx_thread.join(timeout=1.0)
        
        if self.use_socketcan:
            if hasattr(self, 'can_tx_socket'):
                self.can_tx_socket.close()
            if hasattr(self, 'can_rx_socket'):
                self.can_rx_socket.close()
        else:
            if hasattr(self, 'can_bus'):
                self.can_bus.shutdown()
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RobotCANInterface()
        
        if hasattr(node, 'initialization_successful') and node.initialization_successful:
            rclpy.spin(node)
        else:
            node.get_logger().error('Node initialization failed. Shutting down.')
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()