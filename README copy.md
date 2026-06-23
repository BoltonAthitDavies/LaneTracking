# LaneTracking
ณัฐภัทร

## Command in Jetson
==Full command docker run ros2 humble with --privileged==
```
docker run -p 6080:80 \
  --security-opt seccomp=unconfined \
  --shm-size=512m \
  --privileged \
  -v /dev/video0:/dev/video0 \
  -v /dev/video1:/dev/video1 \
  -v /dev/video2:/dev/video2 \
  -v /dev/video3:/dev/video3 \
  ghcr.io/tiryoh/ros2-desktop-vnc:humble
```
==Install GStreamer on Ubuntu (Jetson)==
```
https://lifestyletransfer.com/how-to-install-gstreamer-on-ubuntu/
```
-main-
```
sudo apt update
sudo apt install -y gstreamer1.0-tools
```
-additional-
```
sudo apt install -y \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav
```
==Check group==
```
groups
```
==Add user ubuntu to the video group inside the container==
```
sudo usermod -aG video ubuntu
 #Then restart the container (or just logout/login inside it if possible):
 ```
---
==My Docker (after commit)==
```
  docker run -p 6080:80   --security-opt seccomp=unconfined   --shm-size=512m   --privileged   -v /dev/video0:/dev/video0   -v /dev/video1:/dev/video1   -v /dev/video2:/dev/video2   -v /dev/video3:/dev/video3 -v ~/LaneTracking:/home/ubuntu/LaneTracking ros2_humble_image:v1
```
==Browse for container==
```
http://127.0.0.1:6080/
```
==Git docker ros2==
```
https://github.com/Tiryoh/docker-ros2-desktop-vnc/tree/master?tab=readme-ov-file
```
==Docker command==
```
docker ps # list container
docker images # list updated timestamp for image
docker image prune # remove unused dangling images
docker rmi <image_id> # remove a specific old image by ID
```
==How to commit Docker==
```
docker commit <container_id> <my_image_name:my_tag>
docker commit <container_id> ros2_humble_image:v1
```

## Command in Container (Docker)
- Start two fish eye camera
    ```bash
    ros2 launch robot_camera dual_camera.launch.py
    ```
- Run node undistortion, stereo to depth
    ```bash
    ros2 run robot_camera stereo2depth_node.py --ros-args -p show_undistort_images:=True -p skip_frames:=2
    ```

## Command Stereo Fisheye Camera to Depth (On host with cuda)

```bash
python3 optimized_test_undistort_dispariry_depth.py --mode depth --algorithm sgm
```
```bash
python3 bilateral_filter_without_crop.py --mode depth --algorithm bm --bilateral-iters 1 --bilateral-radius 1 --num-disparities 256 --block-size 15 --uniqueness-ratio 5 --sgm-p1 10 --sgm-p2 120
```
## Tune Disparity Parameter

- block_size:
  - High: Find good depth at close distance and less detail at far distance.
  - Low: Opposite high.

- num_disparity:
  - High: Good depth accuracy and not have noise of depth
  - Low: Opposite high

## Run Docker from jetson-container

```bash
docker run -it --rm   --runtime nvidia   --network=host   --privileged   -e DISPLAY=$DISPLAY   -v /tmp/.X11-unix:/tmp/.X11-unix   -v /dev/video0:/dev/video0   -v /dev/video1:/dev/video1   -v ~/LaneTracking:/home/ubuntu/LaneTracking   dustynv/ros:humble-desktop-pytorch-l4t-r35.4.1   /bin/bash
```

### Allow X server connections:

You need to tell your host's X server to accept connections from the Docker container. Use it on host once.

```bash
xhost +local:docker
```

--

## Docker Ros2

### Build image
```bash
docker buildx build --platform=linux/arm64 -t ros2-torch-opencv-cuda:foxy .
```
### Run container
```bash
docker run -p 6080:80   --security-opt seccomp=unconfined   --shm-size=512m   --privileged   -v /dev/video0:/dev/video0   -v /dev/video1:/dev/video1   -v /dev/video2:/dev/video2   -v /dev/video3:/dev/video3 -v ~/LaneTracking:/home/ubuntu/LaneTracking ros2-torch-opencv-cuda:foxy
```

### Clean Docker's Own Cache and Data
```bash
docker system prune -a --volumes
```
## Docker lanetracking/ros:humble

- **Docker Run**
```bash
docker run -it --rm --runtime nvidia --network=host --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/video0:/dev/video0 -v /dev/video1:/dev/video1 -v /dev/ttyCANable:/dev/ttyCANable -v /dev/ttyUSB0:/dev/ttyUSB0 -v ~/LaneTracking:/home/ubuntu/LaneTracking lanetracking/ros:humble
```

- **Open docker another terminal**

```bash
docker exec -it <container_id> bash
```

- **Save c++ code**

```bash
g++ docker_optimized_test_undistort_dispariry_depth.cpp -o stereo_vision     -std=c++17     `pkg-config --cflags --libs opencv4`     -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
```

- **Run stereo fisheye depth code in docker**

```bash
./stereo_vision
```

### How to fix **WARNING:colcon.colcon_ros.prefix_path.catkin:** that append when colcon build
```bash
cd /path/to/your/ros2_ws
rm -rf build install log

unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
source /opt/ros/humble/setup.bash

colcon build
```

### Command run package robot_camera
- Use cv2.VideoCapture with GStreamer to initialize fisheye camera
```bash
ros2 run robot_camera stereo_fisheye2depth_pipeline.py --ros-args -p show_images:=false -p  compress_depth:=true
```
```bash
ros2 run robot_camera stereo_fisheye2depth_lane_detect_pipeline.py
```
- Use v4l2_camera package ros2 to initialize fisheye camera
```bash
ros2 run robot_camera stereo_fisheye2depth.py --ros-args -p show_images:=False
```

### Command for Change Ownership to Your User

On Jetson host (outside Docker):
```bash
sudo chown -R nvidia:nvidia ~/LaneTracking
```

### Command run package robot_can_interface
- Send velocity and steer to robot with canbus
```bash
ros2 run robot_can_interface robot_can_interface_node.py --ros-args -p log_feedback:=true -p interface_type:=canable
```

### Command run package robot_controller
- Control steer
```bash
ros2 run robot_controller steering_model_node.py
```

### Command run launch file
```bash
ros2 launch robot_bringup robot_bringup.launch.py log_feedback:=false
```

### Command convert model
Convert on host (In ~/LaneTracking/Deep_Lane_Detect directory)
- .pth to .onnx 
```bash
python deploy/pt2onnx.py --config_path configs/tusimple_res34.py --model_path logs/20250716_155153_lr_1e-02_b_32/model_best.pth
```
- .onnx to .engine
```bash
/usr/src/tensorrt/bin/trtexec --onnx=/home/nvidia/Downloads/tusimple_res34_v3.onnx --saveEngine=/home/nvidia/Downloads/tusimple_res34_v3.engine
```

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

---
sudo ./setup_canable.sh -d /dev/ttyACM0

ros2 launch robot_bringup robot_bringup.launch.py log_feedback:=false

ros2 run robot_camera stereo_fisheye2depth_pipeline.py --ros-args -p resize_image:=true -p show_images:=false -p block_size:=25 -p compress_depth:=false -p calc_depth:=false -p pub_depth_image:=false

PUB DEPTH: ros2 run robot_camera stereo_fisheye2depth_pipeline.py --ros-args -p resize_image:=true -p show_images:=true -p block_size:=25 -p compress_depth:=false -p calc_depth:=true -p pub_depth_image:=true


ros2 run robot_camera lane_detect_image.py --ros-args -p show_images:=false

ros2 run robot_controller pid_controller.py --ros-args -p kp_steer:=400.0 -p kp_speed:=10.0 -p kd_speed:=10.0 -p max_speed:=3.0 -p ki_speed:=5.0
---

---
ros2 launch robot_description rviz.launch.py
---

ros2 run robot_camera stereo_fisheye2depth_pipeline_add_sgbm_wls.py --ros-args -p resize_image:=true -p show_images:=true -p block_size:=17 -p compress_depth:=false -p calc_depth:=true -p pub_depth_image:=true -p num_disparities:=256 -p lamda:=8000.0 -p sigma_color:=1.5

==Best==
ros2 run robot_camera stereo_fisheye2depth_pipeline.py --ros-args -p resize_image:=true -p show_images:=true -p compress_depth:=false -p calc_depth:=true -p pub_depth_image:=true -p num_disparities:=128 -p block_size:=5 -p use_wls:=true -p sigma_color:=1.5 -p lamda:=8000.0 -p stereo_algorithm:=sgbm

### Command to Check serial port permissions /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB0

ros2 topic pub /cmd_velocity std_msgs/msg/Float64 "data: 0.5"

### Launch file for full lane detection

ros2 launch robot_bringup robot_lanedetect_control.launch.py show_images:=false ori_width:=480 ori_height:=320 engine_path:=model/tusimple_res34_bend_25_v1_480x320.engine config_path:=config/tusimple_res34_bend_25_v1_480x320.py

ros2 run robot_controller pid_controller.py --ros-args -p kp_steer:=400.0 -p kp_speed:=12.0 -p kd_speed:=0.0 -p max_speed:=2.7 -p ki_speed:=0.0

ros2 run robot_controller cascade_pid_controller.py

---

python3 undistort_single_camera.py

ros2 run robot_controller steering_model_node.py

ros2 run robot_can_interface robot_can_interface_node.py

ros2 run robot_controller plotter_node.py --plot-vel --plot-filtered-vel

ros2 topic pub /cmd_velocity std_msgs/msg/Float64 "data: 1.0"

ros2 topic pub /cmd_steering std_msgs/msg/Float64 "data: 0.0"

---