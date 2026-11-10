# LaneTracking

An autonomous lane-following robot system running on NVIDIA Jetson. It uses a stereo fisheye camera to see the road, a deep learning model to detect lane lines, and a PID controller to steer and drive the robot along the lane via CAN bus.

---

## System Overview

```
Stereo Fisheye Cameras
        │
        ▼
Depth Estimation (stereo disparity)
        │
        ▼
Lane Detection (TensorRT deep learning model)
        │
        ▼
PID Controller (steering + speed)
        │
        ▼
CAN Bus → Robot Motors
```

---

## Prerequisites

- NVIDIA Jetson (tested on Jetson with L4T r35.4.1)
- Docker installed on host
- Two fisheye cameras connected at `/dev/video0` and `/dev/video1`
- CANable USB-CAN adapter at `/dev/ttyCANable`

---

## 1. Start the Docker Container

This project runs inside a custom Docker image (`lanetracking/ros:humble`) that has ROS2 Humble, CUDA, PyTorch, and OpenCV pre-installed.

```bash
docker run -it --rm \
  --runtime nvidia \
  --network=host \
  --privileged \
  -e ROS_DOMAIN_ID=29 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/video0:/dev/video0 \
  -v /dev/video1:/dev/video1 \
  -v /dev/ttyCANable:/dev/ttyCANable \
  -v /dev/ttyUSB0:/dev/ttyUSB0 \
  -v /dev/bus/usb:/dev/bus/usb \
  -v /home/nvidia/torch-py38/lib/python3.8/site-packages/pyrealsense2:/usr/local/lib/python3.8/dist-packages/pyrealsense2:ro \
  -v /home/nvidia/librealsense-src/build/Release:/home/nvidia/librealsense-src/build/Release:ro \
  -v ~/LaneTracking:/home/ubuntu/LaneTracking \
  lanetracking/ros:humble
```

> To allow the container to display windows on your screen, run this **once on the host**:
> ```bash
> xhost +local:docker
> ```

> The `/dev/bus/usb`, `pyrealsense2`, and `librealsense-src/build/Release`
> mounts are for the RealSense D455 IMU (`realsense_imu_publisher.py`,
> launched by `dual_camera.launch.py`). The D455's motion module talks over
> raw USB, not `/dev/videoN`, so it needs the USB bus passed through. The
> `pyrealsense2` mount reuses a working build from the host's `torch-py38`
> venv, built from librealsense source at `v2.56.5` — the pip wheel for
> aarch64 tops out at `2.55.1.6486`, which has a device enumeration bug on
> this host's kernel/UVC driver. That built `.so` dynamically links
> `librealsense2.so` via an absolute-path RUNPATH into `librealsense-src`'s
> build output (not a system lib), so **both** mounts are required together —
> the `pyrealsense2` mount alone fails with `librealsense2.so.2.56: cannot
> open shared object file`. If you rebuild the venv or the librealsense
> source tree from scratch, redo the source build before these mounts will
> work again.

**Open a second terminal inside the same running container:**
```bash
docker exec -it <container_id> bash
```

---

## 2. Run the Full System

The easiest way — one command to launch everything (cameras + lane detection + controller + CAN interface):

```bash
ros2 launch robot_bringup robot_lanedetect_control.launch.py \
  show_images:=false \
  ori_width:=480 \
  ori_height:=320 \
  engine_path:=model/tusimple_res34_bend_25_v1_480x320.engine \
  config_path:=config/tusimple_res34_bend_25_v1_480x320.py
```

Then in a separate terminal, start the PID controller:
```bash
ros2 run robot_controller pid_controller.py \
  --ros-args \
  -p kp_steer:=400.0 \
  -p kp_speed:=12.0 \
  -p kd_speed:=0.0 \
  -p max_speed:=2.7 \
  -p ki_speed:=0.0
```

---

## 3. Run Nodes Individually

If you want to run each component separately for debugging:

### CAN Bus Setup (on host, before starting Docker)
```bash
sudo ./canbus/setup_canable.sh -d /dev/ttyACM0
```

### Camera → Depth Image
```bash
ros2 run robot_camera stereo_fisheye2depth_pipeline.py \
  --ros-args \
  -p resize_image:=true \
  -p show_images:=false \
  -p calc_depth:=true \
  -p pub_depth_image:=true \
  -p num_disparities:=128 \
  -p block_size:=5 \
  -p use_wls:=true \
  -p sigma_color:=1.5 \
  -p lamda:=8000.0 \
  -p stereo_algorithm:=sgbm
```

### Lane Detection
```bash
ros2 run robot_camera lane_detect_image.py --ros-args -p show_images:=false
```

### CAN Bus Interface (send speed/steering to motors)
```bash
ros2 run robot_can_interface robot_can_interface_node.py \
  --ros-args \
  -p log_feedback:=true \
  -p interface_type:=canable
```

### PID Controller
```bash
ros2 run robot_controller pid_controller.py \
  --ros-args \
  -p kp_steer:=400.0 \
  -p kp_speed:=10.0 \
  -p kd_speed:=10.0 \
  -p max_speed:=3.0 \
  -p ki_speed:=5.0
```

### Visualize in RViz
```bash
ros2 launch robot_description rviz.launch.py
```

### Plot velocity
```bash
ros2 run robot_controller plotter_node.py --plot-vel --plot-filtered-vel
```

---

## 4. Manual Robot Control (Testing)

Publish speed and steering directly to test the motors without the controller:

```bash
ros2 topic pub /cmd_velocity std_msgs/msg/Float64 "data: 1.0"
ros2 topic pub /cmd_steering std_msgs/msg/Float64 "data: 0.0"
```

---

## 5. Smartphone Control via MQTT (IOT MQTT Panel)

The `mqtt_bridge_node` connects your phone to the robot over Wi-Fi using MQTT.

### Requirements

- **Mosquitto** broker installed in the Docker container (`apt-get install -y mosquitto`)
- **paho-mqtt** Python library installed in the container (`pip3 install paho-mqtt`)
- **IOT MQTT Panel** app on your smartphone (Android/iOS)

### Start the MQTT Broker

```bash
# Inside the Docker container
mosquitto -c /home/ubuntu/LaneTracking/mosquitto.conf -v
```

### Start the Bridge Node

```bash
# In a second terminal inside the container
ros2 run robot_controller mqtt_bridge_node.py
```

### Phone App Setup (IOT MQTT Panel)

| Field | Value |
|-------|-------|
| Broker address | `10.157.10.156` (Jetson IP) |
| Port | `1883` |
| Client ID | anything (e.g. `phone`) |
| Username / Password | leave blank |

Add controls that publish to these MQTT topics:

| MQTT Topic | Type | Effect |
|---|---|---|
| `robot/cmd_velocity` | Slider / number | Sets speed (m/s) |
| `robot/cmd_steering` | Slider / number | Sets steering angle (degrees) |
| `robot/stop` | Button | Sets both velocity and steering to 0 |
| `robot/status` | Subscribe (text) | Shows `online` heartbeat from robot |

Payload format: plain number (e.g. `1.5`) or JSON `{"data": 1.5}`.

---

## 6. VINS-Fusion (Visual-Inertial Odometry)

`vins_fusion_ros2` (symlinked into `ros2_ws/src/`) builds as part of the workspace. It provides
`vins_fusion_ros2_node`, a generic stereo + IMU visual-inertial odometry estimator.

### Run on this robot's stereo fisheye cameras (WiL config)

First, start the cameras so images are actually publishing on `/cam0/image_raw` and
`/cam1/image_raw`:
```bash
ros2 launch robot_camera dual_camera.launch.py
```

Then, in another terminal, run the estimator against the real calibration
(`config/wil/stereo.yaml`, `config/wil/cam0_fisheye.yaml`, `config/wil/cam1_fisheye.yaml` — built
from this rig's `fisheye_calibration.json` intrinsics and `camera0/1_rot_trans.dat` stereo
extrinsics):
```bash
ros2 run vins_fusion_ros2 vins_fusion_ros2_node --ros-args -p config_file:=/home/ubuntu/LaneTracking/vins_fusion_ros2/config/wil/stereo.yaml -p use_sim_time:=false
```

> `cam1_fisheye.yaml` currently reuses cam0's intrinsics since only one mono calibration exists
> for this rig — replace it with a real cam1-specific calibration if the two cameras differ
> meaningfully.

### Other configs

```bash
ros2 launch vins_fusion_ros2 vins_fusion_ros2.launch.py
# or, with an explicit config:
ros2 run vins_fusion_ros2 vins_fusion_ros2_node \
  --ros-args -p config_file:=src/vins_fusion_ros2/config/euroc/euroc_stereo_config.yaml
```

> The remaining shipped configs (`config/euroc`, `config/mynteye`, `config/realsense_d435i`, etc.)
> are examples for other camera rigs, not this robot. A handful of CARLA-simulator and
> `global_fusion`-dependent executables from this package (`carla_vins_node`,
> `carla_vins_multi_node`, `vins_bag_reader`) are excluded from the default build since they need
> a CARLA SDK / sibling package not present here; they can be re-enabled with
> `colcon build --cmake-args -DWITH_CARLA=ON` on a machine that has both.

---

## 7. Stereo Depth Tuning Parameters

Adjust these parameters in the camera node to tune depth quality:

| Parameter | Higher value | Lower value |
|---|---|---|
| `block_size` | Better depth at close range, less detail far away | Better far detail, noisier close |
| `num_disparities` | More accurate depth, less noise | Less accurate, more noise |

---

## 8. Deep Learning Model — Lane Detection

The lane detection model (UFLD-V2, ResNet-34 backbone) must be converted for fast inference on Jetson using TensorRT.

### Convert PyTorch → ONNX → TensorRT Engine

Run inside `~/LaneTracking/Deep_Lane_Detect/`:

```bash
# Step 1: .pth → .onnx
export PYTHONPATH=$PYTHONPATH:$(pwd)
python deploy/pt2onnx.py \
  --config_path configs/tusimple_res34.py \
  --model_path logs/<run_folder>/model_best.pth

# Step 2: .onnx → .engine (TensorRT)
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/nvidia/Downloads/tusimple_res34_v3.onnx \
  --saveEngine=/home/nvidia/Downloads/tusimple_res34_v3.engine
```

---

## 9. Troubleshooting

**Serial port permission denied:**
```bash
sudo chmod 666 /dev/ttyUSB0
```

**Files owned by root after Docker (can't edit on host):**
```bash
sudo chown -R nvidia:nvidia ~/LaneTracking
```

**ROS2 build warning about catkin prefix path:**
```bash
cd /path/to/ros2_ws
rm -rf build install log
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
source /opt/ros/humble/setup.bash
colcon build
```

---

## 10. Useful Docker Commands

```bash
docker ps                          # list running containers
docker images                      # list images
docker exec -it <container_id> bash  # open terminal in running container
docker commit <container_id> <image_name:tag>  # save container as new image
docker image prune                 # remove unused images
docker system prune -a --volumes   # full cleanup (careful — removes everything)
```
