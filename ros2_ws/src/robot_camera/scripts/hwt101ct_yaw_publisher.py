#!/usr/bin/python3
"""ROS2 driver for the WitMotion HWT101CT single-axis (Z) gyro/inclinometer.

Publishes sensor_msgs/Imu on 'hwt101ct_yaw_publisher' (-> /hwt101ct_yaw_publisher),
the topic robot_odometry/odometry_node.py already subscribes to for yaw rate +
orientation. Protocol/scaling/auto-detect logic mirrors imu/linux/hwt101.py:
WitMotion 11-byte frames, 0x52 = gyro, 0x53 = angle, only the Z axis is real
data on this part.
"""

import glob
import math
import struct
import threading
import time

import rclpy
import serial
from rclpy.node import Node
from sensor_msgs.msg import Imu
from serial.tools import list_ports

HEADER = 0x55
FRAME_LEN = 11
TYPE_GYRO = 0x52
TYPE_ANGLE = 0x53
GYRO_RANGE = 2000.0   # deg/s at full scale
ANGLE_RANGE = 180.0   # deg at full scale
CANDIDATE_BAUDS = [9600, 115200, 230400, 57600, 38400, 19200, 4800]

# Register 0x03 (RRATE) -> output rate, see imu/linux/hwt101.py --rate.
RATE_CODES = {
    0.2: 0x01, 0.5: 0x02, 1: 0x03, 2: 0x04, 5: 0x05, 10: 0x06,
    20: 0x07, 50: 0x08, 100: 0x09, 125: 0x0A, 200: 0x0B,
}


def write_reg(ser, reg, value):
    """Send the 5-byte WIT write command: FF AA <reg> <valLo> <valHi>."""
    ser.write(bytes([0xFF, 0xAA, reg & 0xFF, value & 0xFF, (value >> 8) & 0xFF]))
    ser.flush()


class FrameParser:
    """Incremental WitMotion frame parser (see imu/linux/hwt101.py)."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data):
        self.buf.extend(data)
        out = []
        while len(self.buf) >= FRAME_LEN:
            if self.buf[0] != HEADER:
                del self.buf[0]
                continue
            ftype = self.buf[1]
            if ftype not in (TYPE_GYRO, TYPE_ANGLE):
                del self.buf[0]
                continue
            frame = self.buf[:FRAME_LEN]
            if sum(frame[:FRAME_LEN - 1]) & 0xFF != frame[FRAME_LEN - 1]:
                del self.buf[0]  # bad checksum: resync one byte at a time
                continue
            del self.buf[:FRAME_LEN]
            _, _, z = struct.unpack("<hhh", frame[2:8])
            scale = GYRO_RANGE if ftype == TYPE_GYRO else ANGLE_RANGE
            out.append((ftype, z / 32768.0 * scale))
        if len(self.buf) > 4 * FRAME_LEN:
            del self.buf[:-FRAME_LEN]
        return out


def find_port():
    """Return the most likely sensor port, or None."""
    for p in list_ports.comports():
        # CH340 (1a86:7523), CP210x (10c4:ea60), FTDI (0403:*), PL2303 (067b:*)
        if p.vid in (0x1A86, 0x10C4, 0x0403, 0x067B):
            return p.device
    hits = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    return hits[0] if hits else None


def autodetect_baud(port):
    """Open the port at each candidate baud and keep the first that yields valid frames."""
    for baud in CANDIDATE_BAUDS:
        try:
            with serial.Serial(port, baud, timeout=0.2) as ser:
                time.sleep(0.3)
                ser.reset_input_buffer()
                parser = FrameParser()
                deadline = time.time() + 1.2
                frames = 0
                while time.time() < deadline:
                    chunk = ser.read(256)
                    if chunk:
                        frames += len(parser.feed(chunk))
                    if frames >= 3:
                        return baud
        except serial.SerialException:
            continue
    return None


class Hwt101ctYawPublisher(Node):
    def __init__(self):
        super().__init__('hwt101ct_yaw_node')

        self.declare_parameter('serial_port', '')   # '' = auto-detect
        self.declare_parameter('baud_rate', 0)      # 0 = auto-detect
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('rate_hz', 0.0)      # 0 = leave sensor's current setting

        self.frame_id = self.get_parameter('frame_id').value
        self.imu_pub = self.create_publisher(Imu, 'hwt101ct_yaw_publisher', 10)

        self._gyro_z_dps = 0.0
        self._stop = False
        self._ser = None

        port = self.get_parameter('serial_port').value or find_port()
        if not port:
            self.get_logger().error(
                "No serial port found for HWT101CT (checked known vendor IDs "
                "and /dev/ttyUSB*/ttyACM*). Node will stay up but publish nothing.")
            return

        baud = self.get_parameter('baud_rate').value or autodetect_baud(port)
        if not baud:
            self.get_logger().error(
                f"No valid HWT101CT frames on {port} at any standard baud rate. "
                "Check wiring/power. Node will stay up but publish nothing.")
            return

        try:
            self._ser = serial.Serial(port, baud, timeout=0.5)
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to open {port}: {e}")
            return

        self.get_logger().info(f"HWT101CT connected on {port} @ {baud}")

        rate_hz = self.get_parameter('rate_hz').value
        if rate_hz:
            code = RATE_CODES.get(rate_hz)
            if code is None:
                self.get_logger().error(
                    f"Unsupported rate_hz={rate_hz}, leaving sensor as-is. "
                    f"Choose one of: {', '.join(str(k) for k in RATE_CODES)}")
            else:
                write_reg(self._ser, 0x69, 0xB588)  # unlock
                time.sleep(0.2)
                write_reg(self._ser, 0x03, code)     # RRATE
                time.sleep(0.2)
                write_reg(self._ser, 0x00, 0x00)     # save to flash
                time.sleep(0.2)
                self._ser.reset_input_buffer()
                self.get_logger().info(f"HWT101CT output rate set to {rate_hz} Hz")

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        parser = FrameParser()
        while not self._stop:
            try:
                chunk = self._ser.read(64)
            except serial.SerialException as e:
                self.get_logger().error(f"Serial read failed, stopping: {e}")
                break
            if not chunk:
                continue
            for ftype, z in parser.feed(chunk):
                if ftype == TYPE_GYRO:
                    self._gyro_z_dps = z
                    continue
                # An angle frame completes a sample; publish here (matches
                # hwt101.py's print-on-angle-frame cadence).
                self._publish_imu(angle_z_deg=z)

    def _publish_imu(self, angle_z_deg):
        yaw = math.radians(angle_z_deg)
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        # Single-axis sensor: only Z (yaw) is real data, roll/pitch are 0.
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = math.sin(yaw / 2.0)
        msg.orientation.w = math.cos(yaw / 2.0)
        msg.orientation_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.001,
        ]

        msg.angular_velocity.x = 0.0
        msg.angular_velocity.y = 0.0
        msg.angular_velocity.z = math.radians(self._gyro_z_dps)
        msg.angular_velocity_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.0005,
        ]

        # This driver doesn't parse accelerometer frames -> mark as unavailable
        # (sensor_msgs/Imu convention: covariance[0] == -1 means "no estimate").
        msg.linear_acceleration_covariance[0] = -1.0

        self.imu_pub.publish(msg)

    def destroy_node(self):
        self._stop = True
        if self._ser is not None:
            try:
                self._ser.close()
            except serial.SerialException:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Hwt101ctYawPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
