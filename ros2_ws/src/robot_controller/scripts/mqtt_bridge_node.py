#!/usr/bin/env python3
"""
MQTT <-> ROS2 bridge node.
Subscribes to MQTT topics from IOT MQTT Panel app and publishes to ROS2 topics.
Also forwards ROS2 feedback topics to MQTT for display on the phone.

MQTT topics (phone -> robot):
  robot/cmd_velocity  -> /cmd_velocity  (Float64, m/s)
  robot/cmd_steering  -> /cmd_steering  (Float64, degrees)
  robot/stop          -> publishes 0.0 to both velocity and steering

MQTT topics (robot -> phone):
  feedback/velocity   <- /feedback/velocity  (actual speed from CAN, m/s)
  feedback/steering   <- /feedback/steering  (actual steering from CAN, degrees)
  robot/status        <- connection heartbeat (1 Hz)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import paho.mqtt.client as mqtt
import threading
import json

MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_KEEPALIVE   = 60

TOPIC_CMD_VEL      = "robot/cmd_velocity"
TOPIC_CMD_STEER    = "robot/cmd_steering"
TOPIC_STOP         = "robot/stop"
TOPIC_STATUS       = "robot/status"
TOPIC_FB_VEL            = "feedback/velocity"
TOPIC_FB_VEL_FILTERED   = "feedback/velocity_filtered"
TOPIC_FB_STEER          = "feedback/steering"


class MqttBridgeNode(Node):
    def __init__(self):
        super().__init__("mqtt_bridge_node")

        self.vel_pub = self.create_publisher(Float64, "/cmd_velocity", 10)
        self.steer_pub = self.create_publisher(Float64, "/cmd_steering", 10)

        self.create_subscription(Float64, "feedback/velocity", self._fb_vel_callback, 10)
        self.create_subscription(Float64, "feedback/velocity_filtered", self._fb_vel_filtered_callback, 10)
        self.create_subscription(Float64, "feedback/steering", self._fb_steer_callback, 10)

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self._connect_mqtt()

        # heartbeat: publish connection status to phone every second
        self.create_timer(1.0, self._publish_status)

        self.get_logger().info(
            f"MQTT bridge started — broker at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}"
        )

    # ------------------------------------------------------------------
    def _connect_mqtt(self):
        try:
            self.client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEPALIVE)
            thread = threading.Thread(target=self.client.loop_forever, daemon=True)
            thread.start()
        except Exception as e:
            self.get_logger().error(f"Failed to connect to MQTT broker: {e}")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self.get_logger().info("Connected to MQTT broker")
            client.subscribe(TOPIC_CMD_VEL)
            client.subscribe(TOPIC_CMD_STEER)
            client.subscribe(TOPIC_STOP)
        else:
            self.get_logger().error(f"MQTT connect failed, reason: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.get_logger().warn("Disconnected from MQTT broker")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = msg.payload.decode("utf-8").strip()
            self.get_logger().info(f"MQTT recv [{topic}]: {payload}")

            if topic == TOPIC_STOP:
                self._publish_ros(self.vel_pub, 0.0)
                self._publish_ros(self.steer_pub, 0.0)
                return

            # Accept plain number ("1.5") or JSON ({"data": 1.5})
            try:
                value = float(json.loads(payload)["data"])
            except (KeyError, TypeError, json.JSONDecodeError):
                value = float(payload)

            if topic == TOPIC_CMD_VEL:
                self._publish_ros(self.vel_pub, value)
            elif topic == TOPIC_CMD_STEER:
                self._publish_ros(self.steer_pub, value)

        except ValueError as e:
            self.get_logger().warn(f"Bad payload on [{topic}]: {msg.payload!r} — {e}")

    def _publish_ros(self, publisher, value: float):
        msg = Float64()
        msg.data = value
        publisher.publish(msg)

    def _fb_vel_callback(self, msg: Float64):
        self.client.publish(TOPIC_FB_VEL, str(round(msg.data, 4)), qos=0, retain=False)

    def _fb_vel_filtered_callback(self, msg: Float64):
        self.client.publish(TOPIC_FB_VEL_FILTERED, str(round(msg.data, 4)), qos=0, retain=False)

    def _fb_steer_callback(self, msg: Float64):
        self.client.publish(TOPIC_FB_STEER, str(round(msg.data, 4)), qos=0, retain=False)

    def _publish_status(self):
        self.client.publish(TOPIC_STATUS, "online", qos=0, retain=False)

    def destroy_node(self):
        self.client.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MqttBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
