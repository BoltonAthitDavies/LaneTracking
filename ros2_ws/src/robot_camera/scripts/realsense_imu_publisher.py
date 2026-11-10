#!/usr/bin/python3
"""ROS2 driver for an Intel RealSense motion module (accel + gyro), e.g. D455.

Publishes sensor_msgs/Imu on 'realsense_imu_publisher'. Remap this to
'/hwt101ct_yaw_publisher' in the launch file to drop in as a replacement for
hwt101ct_yaw_publisher.py -- robot_odometry/odometry_node.py only reads
angular_velocity.z from that topic, so the missing orientation fix-up below
is not a functional gap for it.

Needs a pyrealsense2 build new enough for your device/kernel combo (the
aarch64 pip wheel tops out at 2.55.1.6486, which has a device-enumeration
bug on some kernel/UVC driver combos -- see README's Docker run command for
the working source-built alternative). Mirrors imu.py's device/profile
discovery.
"""

import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


def find_motion_sensor():
    """Return (device, motion_sensor) for the first connected RealSense."""
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        raise RuntimeError(
            "No RealSense device found. If lsusb shows the L515 but this "
            "fails, the two usual causes are:\n"
            "  1. USB 2.0 connection -- the L515 needs USB 3.1 Gen 1. Check "
            "`lsusb -t`: it must show 5000M and Class=Video, not 480M and "
            "Class=Vendor Specific Class.\n"
            "  2. librealsense >= 2.55, which dropped L515 support.")

    dev = devices[0]
    for sensor in dev.query_sensors():
        profiles = sensor.get_stream_profiles()
        if any(p.stream_type() in (rs.stream.accel, rs.stream.gyro) for p in profiles):
            return dev, sensor

    name = dev.get_info(rs.camera_info.name)
    raise RuntimeError(f"{name} exposes no motion module (accel/gyro) streams.")


class RealsenseImuPublisher(Node):
    def __init__(self):
        super().__init__('realsense_imu_node')

        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('gyro_rate_hz', 0.0)   # 0 = fastest advertised
        self.declare_parameter('accel_rate_hz', 0.0)  # 0 = fastest advertised
        self.declare_parameter('reconnect_period_sec', 2.0)
        self.frame_id = self.get_parameter('frame_id').value
        self.imu_pub = self.create_publisher(Imu, 'realsense_imu_publisher', 10)

        self._accel = (0.0, 0.0, 0.0)
        self._sensor = None
        self._serial = None

        period = self.get_parameter('reconnect_period_sec').value
        self._connect()  # try immediately so a healthy startup doesn't wait a full period
        self._poll_timer = self.create_timer(period, self._poll_connection)

    def _poll_connection(self):
        """Runs every reconnect_period_sec: reconnect if down, else confirm still up.

        USB/IIO disconnects don't reliably surface as an exception in the
        frame callback, so presence is checked by re-enumerating devices
        rather than waiting for a stream error -- this is what makes
        recovery automatic instead of requiring a process/container restart.

        This is a timer callback, so any exception that escapes it propagates
        up through rclpy.spin() and kills the whole node -- exactly the
        "permanent, need to restart Docker" failure this node exists to avoid.
        A USB bus glitch (e.g. another device on the same hub being replugged)
        can make query_devices()/sensor calls throw mid-transition, so
        everything here must be defensive: catch broadly, tear down, and let
        the next tick retry instead of ever letting an exception through.
        """
        try:
            if self._sensor is None:
                self._connect()
                return

            serials = {d.get_info(rs.camera_info.serial_number)
                       for d in rs.context().query_devices()}
            if self._serial not in serials:
                self.get_logger().warn("RealSense disconnected, will retry reconnecting.")
                self._teardown()
        except Exception as e:
            self.get_logger().warn(
                f"RealSense connection check failed, will retry: {e}",
                throttle_duration_sec=10.0)
            self._teardown()

    def _connect(self):
        try:
            dev, sensor = find_motion_sensor()
        except RuntimeError as e:
            self.get_logger().warn(f"{e}", throttle_duration_sec=10.0)
            return

        # Everything below can throw mid-transition if the bus is still
        # settling (e.g. right after another USB device on the same hub was
        # replugged) -- caught broadly so a bad attempt just gets retried on
        # the next poll tick instead of taking the whole node down.
        try:
            # Query the device's own advertised profiles rather than hard-coding
            # an fps it may not support (see imu.py); default to the fastest one
            # unless gyro_rate_hz/accel_rate_hz asks for a specific supported rate.
            wanted = {
                rs.stream.gyro: self.get_parameter('gyro_rate_hz').value,
                rs.stream.accel: self.get_parameter('accel_rate_hz').value,
            }
            by_stream = {}
            for p in sensor.get_stream_profiles():
                st = p.stream_type()
                if st in (rs.stream.accel, rs.stream.gyro):
                    by_stream.setdefault(st, {})[p.fps()] = p

            if rs.stream.gyro not in by_stream:
                self.get_logger().error(
                    "RealSense motion module has no gyro stream. "
                    "Will keep retrying in case a different device connects.")
                return

            chosen_profiles = []
            chosen_rates = {}
            for st, profiles_by_fps in by_stream.items():
                want = wanted.get(st)
                if want and want in profiles_by_fps:
                    fps = want
                else:
                    if want:
                        self.get_logger().error(
                            f"{st.name}_rate_hz={want} not supported, choices: "
                            f"{sorted(profiles_by_fps)}. Using fastest instead.")
                    fps = max(profiles_by_fps)
                chosen_profiles.append(profiles_by_fps[fps])
                chosen_rates[st] = fps

            # Open the sensor directly and use its own frame callback instead
            # of rs.pipeline()/wait_for_frames(): accel and gyro run at
            # different native rates, and the pipeline's frame synchronizer
            # re-pairs a stale sample from one stream with every fresh sample
            # from the other, so a naive "publish on gyro frame" loop over
            # wait_for_frames ends up republishing stale gyro data at the
            # combined accel+gyro rate. Direct per-sensor callbacks deliver
            # each stream's frames independently, at its own real rate, with
            # no such pairing.
            sensor.open(chosen_profiles)
            sensor.start(self._on_frame)
        except Exception as e:
            self.get_logger().warn(
                f"Failed to start RealSense stream, will retry: {e}",
                throttle_duration_sec=10.0)
            try:
                sensor.stop()
                sensor.close()
            except Exception:
                pass
            return

        self._sensor = sensor
        self._serial = dev.get_info(rs.camera_info.serial_number)
        self.get_logger().info(
            f"RealSense connected (serial={self._serial}), streaming: "
            f"{', '.join(f'{s.name}={hz}Hz' for s, hz in chosen_rates.items())}")

    def _teardown(self):
        if self._sensor is not None:
            try:
                self._sensor.stop()
                self._sensor.close()
            except Exception:
                pass
        self._sensor = None
        self._serial = None

    def _on_frame(self, frame):
        # Called directly by librealsense's own C++ thread, not through
        # rclpy's executor -- an exception escaping here crosses a C++/Python
        # callback boundary with no rclpy-level safety net, so it must never
        # be allowed to propagate.
        try:
            motion = frame.as_motion_frame()
            if not motion:
                return
            d = motion.get_motion_data()
            stream_type = motion.get_profile().stream_type()
            if stream_type == rs.stream.accel:
                self._accel = (d.x, d.y, d.z)
                return
            # A gyro frame completes a sample; publish here (matches
            # hwt101ct_yaw_publisher.py's publish-on-primary-axis cadence).
            self._publish_imu(gyro=(d.x, d.y, d.z))
        except Exception as e:
            self.get_logger().warn(
                f"Error handling RealSense frame: {e}", throttle_duration_sec=10.0)

    def _publish_imu(self, gyro):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        # Motion module gives raw accel/gyro only, no fused orientation.
        # sensor_msgs/Imu convention: orientation_covariance[0] == -1 means
        # "no estimate" -- odometry_node.py only reads angular_velocity.z,
        # so this is not a functional gap for the current consumer.
        msg.orientation_covariance[0] = -1.0

        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gyro
        msg.angular_velocity_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.0005,
        ]

        ax, ay, az = self._accel
        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az
        msg.linear_acceleration_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01,
        ]

        self.imu_pub.publish(msg)

    def destroy_node(self):
        self._teardown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealsenseImuPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
