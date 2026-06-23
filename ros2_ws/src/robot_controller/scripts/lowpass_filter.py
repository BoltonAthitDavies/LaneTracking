#!/usr/bin/python3
"""
Velocity Low-Pass Filter Node
Subscribes to feedback/velocity and publishes low-pass filtered velocity data
Uses scipy Butterworth filter for professional-grade filtering
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import numpy as np
from collections import deque
import time
from scipy import signal


class VelocityFilterNode(Node):
    def __init__(self):
        super().__init__('velocity_filter_node')
        
        # Declare parameters
        self.declare_parameter('cutoff_frequency', 1.0)  # Cutoff frequency (Hz)
        self.declare_parameter('input_frequency', 50.0)  # Expected input frequency (Hz)
        self.declare_parameter('filter_order', 2)  # Order of the low-pass filter
        self.declare_parameter('publish_raw', False)  # Whether to republish raw data
        self.declare_parameter('log_stats', False)  # Whether to log filter statistics
        self.declare_parameter('stats_period', 5.0)  # Period for logging stats (seconds)
        
        # Get parameters
        self.cutoff_frequency = self.get_parameter('cutoff_frequency').value
        self.input_frequency = self.get_parameter('input_frequency').value
        self.filter_order = self.get_parameter('filter_order').value
        self.publish_raw = self.get_parameter('publish_raw').value
        self.log_stats = self.get_parameter('log_stats').value
        self.stats_period = self.get_parameter('stats_period').value
        
        # Validate parameters
        if self.filter_order < 1:
            self.get_logger().warn(f"Filter order should be >= 1, got {self.filter_order}") 
            self.filter_order = max(1, self.filter_order)
        
        # Initialize low-pass filter
        self.init_filter()
        
        # Statistics tracking
        self.message_count = 0
        self.last_message_time = None
        self.frequency_samples = deque(maxlen=100)  # Track frequency
        self.last_stats_time = time.time()
        
        # Create subscriber
        self.velocity_sub = self.create_subscription(
            Float64,
            'feedback/velocity',
            self.velocity_callback,
            10
        )
        
        # Create publishers
        self.filtered_velocity_pub = self.create_publisher(
            Float64, 
            'feedback/velocity_filtered', 
            10
        )
        
        if self.publish_raw:
            self.raw_velocity_pub = self.create_publisher(
                Float64,
                'feedback/velocity_raw',
                10
            )
        
        # Log configuration
        self.get_logger().info(f'Velocity Low-Pass Filter Node initialized')
        self.get_logger().info(f'Cutoff frequency: {self.cutoff_frequency} Hz, Order: {self.filter_order}')
        self.get_logger().info(f'Expected input frequency: {self.input_frequency} Hz')
        self.get_logger().info(f'Publishing raw data: {self.publish_raw}')
        self.get_logger().info(f'Statistics logging: {"enabled" if self.log_stats else "disabled"}')
    
    def init_filter(self):
        """Initialize Butterworth low-pass filter"""
        # Design Butterworth low-pass filter using scipy
        nyquist = self.input_frequency / 2.0
        normalized_cutoff = self.cutoff_frequency / nyquist
        
        # Ensure cutoff frequency is valid
        if normalized_cutoff >= 1.0:
            self.get_logger().warn(f"Cutoff frequency {self.cutoff_frequency} Hz is too high for sampling rate {self.input_frequency} Hz")
            normalized_cutoff = 0.9
            self.cutoff_frequency = normalized_cutoff * nyquist
        
        # Design digital Butterworth filter
        self.sos = signal.butter(self.filter_order, normalized_cutoff, 
                               btype='low', analog=False, output='sos')
        
        # Initialize filter state
        self.zi = signal.sosfilt_zi(self.sos)
        self.filter_initialized = False
        
        self.get_logger().info(f'Butterworth low-pass filter designed: '
                             f'Order={self.filter_order}, '
                             f'Fc={self.cutoff_frequency:.2f} Hz, '
                             f'Normalized Fc={normalized_cutoff:.4f}')
    
    def velocity_callback(self, msg):
        """Process incoming velocity data"""
        current_time = time.time()
        raw_velocity = msg.data
        
        # Track message frequency
        if self.last_message_time is not None:
            dt = current_time - self.last_message_time
            if dt > 0:
                frequency = 1.0 / dt
                self.frequency_samples.append(frequency)
        
        self.last_message_time = current_time
        self.message_count += 1
        
        # Apply low-pass filter
        filtered_velocity = self.low_pass_filter(raw_velocity)
        
        # Publish filtered data
        filtered_msg = Float64()
        filtered_msg.data = filtered_velocity
        self.filtered_velocity_pub.publish(filtered_msg)
        
        # Publish raw data if enabled
        if self.publish_raw:
            self.raw_velocity_pub.publish(msg)
        
        # Log statistics periodically
        if self.log_stats and (current_time - self.last_stats_time) >= self.stats_period:
            self.log_statistics(raw_velocity, filtered_velocity)
            self.last_stats_time = current_time
    
    def low_pass_filter(self, raw_velocity):
        """Apply Butterworth low-pass filter using scipy"""
        if not self.filter_initialized:
            # Initialize filter state with first sample
            self.zi = self.zi * raw_velocity
            self.filter_initialized = True
        
        # Apply filter (single sample)
        filtered_sample, self.zi = signal.sosfilt(self.sos, [raw_velocity], zi=self.zi)
        return filtered_sample[0]
    
    def log_statistics(self, raw_velocity, filtered_velocity):
        """Log filter statistics"""
        avg_frequency = np.mean(self.frequency_samples) if self.frequency_samples else 0
        
        self.get_logger().info(
            f'Filter Stats - Messages: {self.message_count}, '
            f'Avg freq: {avg_frequency:.1f} Hz, '
            f'Raw: {raw_velocity:.4f}, '
            f'Filtered: {filtered_velocity:.4f}'
        )
    
    def destroy_node(self):
        """Clean up resources"""
        self.get_logger().info('Velocity Low-Pass Filter Node shutting down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VelocityFilterNode()
        rclpy.spin(node)
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