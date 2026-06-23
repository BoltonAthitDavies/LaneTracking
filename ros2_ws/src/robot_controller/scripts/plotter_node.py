#!/usr/bin/python3
"""
feedback_realtime_plot.py

Real-time plotting node for ROS2 topics.
Supports dynamic plotting selection via command-line arguments:
 - /feedback/velocity (Raw)
 - /feedback/velocity_filtered (Filtered)
 - /cmd_velocity (Command/Target)
 - /feedback/steering (Actual)
 - /cmd_steering (Command/Target)

Usage Examples:
  # Plot Command vs Actual Steering
  python3 feedback_realtime_plot.py --plot-cmd-steering --plot-steering

  # Plot Everything
  python3 feedback_realtime_plot.py --plot-cmd-vel --plot-vel --plot-filtered-vel --plot-steering --plot-cmd-steering
"""

import threading
import time
import argparse
from collections import deque
from typing import Deque, Dict, Any, List

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class FeedbackPlotNode(Node):
    """
    ROS2 Node to subscribe to feedback topics and store data in deques
    for real-time plotting.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        history_seconds: float = 10.0,
        plot_rate_hz: float = 50.0,
    ):
        super().__init__("feedback_realtime_plot_node")

        self.config = config
        self.history_seconds = history_seconds
        self.plot_rate_hz = plot_rate_hz

        # How many samples to keep
        max_samples = int(max(200, history_seconds * plot_rate_hz * 2))

        self.t0 = time.time()
        self.lock = threading.Lock()

        self.active_topics: List[str] = []

        # --- 1. Velocity Raw ---
        if self.config['velocity']['plot']:
            self.vel_t: Deque[float] = deque(maxlen=max_samples)
            self.vel_v: Deque[float] = deque(maxlen=max_samples)
            self.create_subscription(
                Float64, self.config['velocity']['topic'], self.velocity_callback, 10
            )
            self.active_topics.append(self.config['velocity']['topic'])

        # --- 2. Velocity Filtered ---
        if self.config['velocity_filtered']['plot']:
            self.vel_filtered_t: Deque[float] = deque(maxlen=max_samples)
            self.vel_filtered_v: Deque[float] = deque(maxlen=max_samples)
            self.create_subscription(
                Float64, self.config['velocity_filtered']['topic'], self.velocity_filtered_callback, 10
            )
            self.active_topics.append(self.config['velocity_filtered']['topic'])

        # --- 3. Command Velocity ---
        if self.config['cmd_velocity']['plot']:
            self.cmd_vel_t: Deque[float] = deque(maxlen=max_samples)
            self.cmd_vel_v: Deque[float] = deque(maxlen=max_samples)
            self.create_subscription(
                Float64, self.config['cmd_velocity']['topic'], self.cmd_velocity_callback, 10
            )
            self.active_topics.append(self.config['cmd_velocity']['topic'])

        # --- 4. Steering (Feedback) ---
        if self.config['steering']['plot']:
            self.st_t: Deque[float] = deque(maxlen=max_samples)
            self.st_v: Deque[float] = deque(maxlen=max_samples)
            self.create_subscription(
                Float64, self.config['steering']['topic'], self.steering_callback, 10
            )
            self.active_topics.append(self.config['steering']['topic'])

        # --- 5. Command Steering (NEW) ---
        if self.config['cmd_steering']['plot']:
            self.cmd_st_t: Deque[float] = deque(maxlen=max_samples)
            self.cmd_st_v: Deque[float] = deque(maxlen=max_samples)
            self.create_subscription(
                Float64, self.config['cmd_steering']['topic'], self.cmd_steering_callback, 10
            )
            self.active_topics.append(self.config['cmd_steering']['topic'])

        if not self.active_topics:
            self.get_logger().warn("No topics selected for plotting.")
        else:
            self.get_logger().info(f"Subscribed to: {', '.join(self.active_topics)}")

    def velocity_callback(self, msg: Float64):
        ts = time.time() - self.t0
        with self.lock:
            self.vel_t.append(ts)
            self.vel_v.append(float(msg.data))

    def velocity_filtered_callback(self, msg: Float64):
        ts = time.time() - self.t0
        with self.lock:
            self.vel_filtered_t.append(ts)
            self.vel_filtered_v.append(float(msg.data))

    def cmd_velocity_callback(self, msg: Float64):
        ts = time.time() - self.t0
        with self.lock:
            self.cmd_vel_t.append(ts)
            self.cmd_vel_v.append(float(msg.data))

    def steering_callback(self, msg: Float64):
        ts = time.time() - self.t0
        with self.lock:
            self.st_t.append(ts)
            self.st_v.append(float(msg.data))

    def cmd_steering_callback(self, msg: Float64):
        """Callback for command steering."""
        ts = time.time() - self.t0
        with self.lock:
            self.cmd_st_t.append(ts)
            self.cmd_st_v.append(float(msg.data))


def make_figure(node: FeedbackPlotNode) -> Dict[str, Any]:
    """Dynamically creates the figure, subplots, and plot lines."""
    
    # Determine active sections
    plot_vel_section = (node.config['velocity']['plot'] or 
                        node.config['velocity_filtered']['plot'] or 
                        node.config['cmd_velocity']['plot'])
                        
    plot_st_section = (node.config['steering']['plot'] or 
                       node.config['cmd_steering']['plot'])

    num_subplots = int(plot_vel_section) + int(plot_st_section)
    if num_subplots == 0:
        raise ValueError("No topics selected for plotting.")

    fig, axes = plt.subplots(num_subplots, 1, sharex=True, figsize=(14, 8))
    fig.suptitle("Realtime Feedback Plotting")

    if num_subplots == 1:
        axes = [axes]
    
    plot_elements: Dict[str, Any] = {'fig': fig}
    current_ax_index = 0

    # --- Velocity Subplot ---
    if plot_vel_section:
        ax_vel = axes[current_ax_index]
        ax_vel.set_ylabel("Velocity (m/s)")
        ax_vel.set_ylim(0.0, 4.0)
        ax_vel.grid(True)
        
        # 1. Plot Raw Velocity
        if node.config['velocity']['plot']:
            vel_line, = ax_vel.plot([], [], label="Actual Raw", linestyle="-", color='#3366ff', alpha=0.6)
            plot_elements['vel_line'] = vel_line
            
        # 2. Plot Filtered Velocity
        if node.config['velocity_filtered']['plot']:
            vel_filtered_line, = ax_vel.plot([], [], label="Actual Filtered", linestyle="-", color='#ff4444', linewidth=2)
            plot_elements['vel_filtered_line'] = vel_filtered_line

        # 3. Plot Command Velocity
        if node.config['cmd_velocity']['plot']:
            cmd_vel_line, = ax_vel.plot([], [], label="Command Target", linestyle=":", color='green', linewidth=2)
            plot_elements['cmd_vel_line'] = cmd_vel_line
        
        ax_vel.legend(loc="upper right")
        plot_elements['ax_vel'] = ax_vel
        current_ax_index += 1

    # --- Steering Subplot ---
    if plot_st_section:
        ax_st = axes[current_ax_index]
        if current_ax_index == num_subplots - 1:
            ax_st.set_xlabel("Time (s)")
            
        ax_st.set_ylabel("Steering (deg)")
        # Adjusted limits slightly to accommodate potential command spikes
        ax_st.set_ylim(-45.0, 45.0)
        ax_st.grid(True)
        
        # 1. Plot Feedback Steering
        if node.config['steering']['plot']:
            st_line, = ax_st.plot([], [], label="Steering Actual", linestyle="-", color='#009900')
            plot_elements['st_line'] = st_line

        # 2. Plot Command Steering (NEW)
        if node.config['cmd_steering']['plot']:
            cmd_st_line, = ax_st.plot([], [], label="Steering Cmd", linestyle=":", color='green', linewidth=2)
            plot_elements['cmd_st_line'] = cmd_st_line

        ax_st.legend(loc="upper right")
        plot_elements['ax_st'] = ax_st

    return plot_elements

def start_plot(node: FeedbackPlotNode):
    try:
        plot_elements = make_figure(node)
    except ValueError as e:
        node.get_logger().error(str(e))
        return

    fig = plot_elements['fig']

    active_axes = [v for k, v in plot_elements.items() if k.startswith('ax_')]
    for ax in active_axes:
        ax.set_xlim(0, node.history_seconds)

    def update(frame):
        return_lines = []
        now = time.time() - node.t0
        xmin = max(0.0, now - node.history_seconds)
        xmax = xmin + node.history_seconds

        with node.lock:
            # --- Velocity Section ---
            if 'vel_line' in plot_elements:
                t = np.array(node.vel_t) if len(node.vel_t) else np.array([])
                v = np.array(node.vel_v) if len(node.vel_v) else np.array([])
                if t.size:
                    mask = t >= xmin
                    plot_elements['vel_line'].set_data(t[mask], v[mask])
                return_lines.append(plot_elements['vel_line'])
                
            if 'vel_filtered_line' in plot_elements:
                t = np.array(node.vel_filtered_t) if len(node.vel_filtered_t) else np.array([])
                v = np.array(node.vel_filtered_v) if len(node.vel_filtered_v) else np.array([])
                if t.size:
                    mask = t >= xmin
                    plot_elements['vel_filtered_line'].set_data(t[mask], v[mask])
                return_lines.append(plot_elements['vel_filtered_line'])

            if 'cmd_vel_line' in plot_elements:
                t = np.array(node.cmd_vel_t) if len(node.cmd_vel_t) else np.array([])
                v = np.array(node.cmd_vel_v) if len(node.cmd_vel_v) else np.array([])
                if t.size:
                    mask = t >= xmin
                    plot_elements['cmd_vel_line'].set_data(t[mask], v[mask])
                return_lines.append(plot_elements['cmd_vel_line'])
            
            # Update Velocity Axis Limits
            if 'ax_vel' in plot_elements:
                plot_elements['ax_vel'].set_xlim(xmin, xmax)

            # --- Steering Section ---
            if 'st_line' in plot_elements:
                t = np.array(node.st_t) if len(node.st_t) else np.array([])
                v = np.array(node.st_v) if len(node.st_v) else np.array([])
                if t.size:
                    mask = t >= xmin
                    plot_elements['st_line'].set_data(t[mask], v[mask])
                return_lines.append(plot_elements['st_line'])

            if 'cmd_st_line' in plot_elements:
                t = np.array(node.cmd_st_t) if len(node.cmd_st_t) else np.array([])
                v = np.array(node.cmd_st_v) if len(node.cmd_st_v) else np.array([])
                if t.size:
                    mask = t >= xmin
                    plot_elements['cmd_st_line'].set_data(t[mask], v[mask])
                return_lines.append(plot_elements['cmd_st_line'])

            # Update Steering Axis Limits
            if 'ax_st' in plot_elements:
                plot_elements['ax_st'].set_xlim(xmin, xmax)

        return tuple(return_lines)

    interval_ms = int(1000.0 / node.plot_rate_hz)
    ani = animation.FuncAnimation(fig, update, interval=interval_ms, blit=True)

    def on_close(event):
        node.get_logger().info("Plot window closed by user — shutting down rclpy.")
        try:
            rclpy.shutdown()
        except Exception:
            pass

    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Real-time ROS2 feedback plotting tool.")
    parser.add_argument(
        "--plot-vel", action="store_true", help="Plot raw velocity (topic: feedback/velocity)"
    )
    parser.add_argument(
        "--plot-filtered-vel", action="store_true", help="Plot filtered velocity (topic: feedback/velocity_filtered)"
    )
    parser.add_argument(
        "--plot-cmd-vel", action="store_true", help="Plot command velocity (topic: /cmd_velocity)"
    )
    parser.add_argument(
        "--plot-steering", action="store_true", help="Plot steering angle (topic: feedback/steering)"
    )
    # --- Added new argument ---
    parser.add_argument(
        "--plot-cmd-steering", action="store_true", help="Plot command steering (topic: /cmd_steering)"
    )
    
    parser.add_argument(
        "--history-s", type=float, default=10.0, help="Duration of history to display in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--rate-hz", type=float, default=50.0, help="Plot update rate in Hz (default: 50.0)"
    )

    args = parser.parse_args()

    plot_config = {
        'velocity': {
            'topic': "feedback/velocity", 
            'plot': args.plot_vel
        },
        'velocity_filtered': {
            'topic': "feedback/velocity_filtered", 
            'plot': args.plot_filtered_vel
        },
        'cmd_velocity': {
            'topic': "/cmd_velocity",
            'plot': args.plot_cmd_vel
        },
        'steering': {
            'topic': "feedback/steering", 
            'plot': args.plot_steering
        },
        # --- Added new config ---
        'cmd_steering': {
            'topic': "/cmd_steering",
            'plot': args.plot_cmd_steering
        },
    }

    if not any(cfg['plot'] for cfg in plot_config.values()):
        print("\nError: Please select at least one topic to plot.")
        print("Example: python3 feedback_realtime_plot.py --plot-cmd-steering --plot-steering\n")
        return

    rclpy.init()
    node = FeedbackPlotNode(
        config=plot_config,
        history_seconds=args.history_s, 
        plot_rate_hz=args.rate_hz
    )

    def spin_thread():
        try:
            rclpy.spin(node)
        except Exception as e:
            node.get_logger().error(f"rclpy.spin exception: {e}")

    t = threading.Thread(target=spin_thread, daemon=True)
    t.start()

    try:
        start_plot(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt - exiting")
    finally:
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
        t.join(timeout=1.0)


if __name__ == "__main__":
    main()

# cd LaneTracking/ros2_ws/src/robot_controller/scripts/
# python plotter_node.py --plot-vel --plot-filtered-vel --plot-cmd-vel --plot-steering --plot-cmd-steering