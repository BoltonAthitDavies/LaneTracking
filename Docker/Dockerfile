# Dockerfile for LaneTracking with ROS2 Humble, Pytorch, OpenCV with CUDA Support
FROM dustynv/ros:humble-desktop-pytorch-l4t-r35.4.1

# Install as root
USER root

# Set environment to avoid debconf warnings
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS=yes

# Update package lists quietly
RUN apt-get update 2>&1 | grep -v "debconf: delaying package configuration" || true

RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Install ROS2 packages
RUN apt-get update && \
    apt-get install -y \
    # ros-humble-multires-image \
    # ros-humble-joy* \
    # ros-humble-teleop-twist-joy \
    # ros-humble-image-transport \
    # ros-humble-compressed-image-transport \
    # ros-humble-message-filters \
    # ros-humble-image-transport-plugins \
    # ros-humble-tf-transformations \
    # ros-humble-plotjuggler-ros \
    iputils-ping \
    curl \
    wget \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common \
    git \
    htop \
    nano \
    vim \
    gedit \
    usbutils \
    net-tools \
    iproute2 \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-setuptools \
    python3-wheel \
    python3-empy \
    python3-numpy \
    python3-yaml \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    cmake \
    build-essential \
    wget \
    file \
    qtbase5-dev \
    libqt5svg5-dev \
    libqt5websockets5-dev \
    libqt5opengl5-dev \
    libqt5x11extras5-dev \
    libprotoc-dev \
    libzmq3-dev \
    liblz4-dev \
    libzstd-dev \
    libmosquittopp-dev \
    protobuf-compiler \
    build-essential \
    python3-catkin-pkg \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-empy \
    python3-numpy \
    python3-yaml \
    python3-lark 
    # || echo "Some packages failed to install, continuing..."

# RUN git clone https://github.com/pytorch/vision.git /tmp/vision \
#     && cd /tmp/vision \
#     && git checkout tags/v0.15.1 \
#     && python3 setup.py install \
#     && rm -rf /tmp/vision 

# Clean apt cache
RUN rm -rf /var/lib/apt/lists/*

# Initialize rosdep if not already done
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
        rosdep init || echo "rosdep init failed"; \
    fi

# Make sure pip is up to date
RUN apt-get update && \
    pip3 install \
        "numpy<1.24" \
        scipy \
        matplotlib \
        pyserial \
        catkin_pkg \
        # empy==3.3.4 \
        lark \
        colcon-common-extensions \
        python-can \
        "python-can[serial]" \
        cantools \
        can-isotp \
        pandas \
        transforms3d \
        addict \
        torchvision==0.15.1 \
        pycuda 
        # onnxmltools \
        # onnxconverter-commonl

# Create /home/ubuntu directory
RUN mkdir -p /home/ubuntu

# Fix file ownership (IMPORTANT — do this before switching user)
RUN chown -R 1000:1000 /home/ubuntu

# Switch to the correct user and set HOME
USER 1000
ENV HOME=/home/ubuntu
WORKDIR /home/ubuntu

# Create comprehensive bashrc using echo commands
RUN echo '# Basic bash configuration' > /home/ubuntu/.bashrc && \
    echo 'export PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> /home/ubuntu/.bashrc && \
    echo 'export TERM=xterm-256color' >> /home/ubuntu/.bashrc && \
    echo '' >> /home/ubuntu/.bashrc && \
    echo '# ROS2 Humble setup' >> /home/ubuntu/.bashrc && \
    echo 'if [ -f /opt/ros/humble/install/setup.bash ]; then' >> /home/ubuntu/.bashrc && \
    echo '    source /opt/ros/humble/install/setup.bash' >> /home/ubuntu/.bashrc && \
    echo 'fi' >> /home/ubuntu/.bashrc && \
    echo '' >> /home/ubuntu/.bashrc && \
    echo '# Workspace navigation' >> /home/ubuntu/.bashrc && \
    echo 'cd /home/ubuntu 2>/dev/null || cd ~' >> /home/ubuntu/.bashrc && \
    echo '' >> /home/ubuntu/.bashrc && \
    echo '# Source workspace if it exists' >> /home/ubuntu/.bashrc && \
    echo 'if [ -f /home/ubuntu/LaneTracking/ros2_ws/install/setup.bash ]; then' >> /home/ubuntu/.bashrc && \
    echo '    source /home/ubuntu/LaneTracking/ros2_ws/install/setup.bash' >> /home/ubuntu/.bashrc && \
    echo 'fi' >> /home/ubuntu/.bashrc && \
    echo '' >> /home/ubuntu/.bashrc && \
    echo '# Colcon autocomplete' >> /home/ubuntu/.bashrc && \
    echo 'if [ -f /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash ]; then' >> /home/ubuntu/.bashrc && \
    echo '    source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash' >> /home/ubuntu/.bashrc && \
    echo 'fi' >> /home/ubuntu/.bashrc && \
    echo '' >> /home/ubuntu/.bashrc && \
    echo '# Useful aliases' >> /home/ubuntu/.bashrc && \
    echo "alias ll='ls -alF'" >> /home/ubuntu/.bashrc && \
    echo "alias la='ls -A'" >> /home/ubuntu/.bashrc && \
    echo "alias l='ls -CF'" >> /home/ubuntu/.bashrc && \
    echo "alias ..='cd ..'" >> /home/ubuntu/.bashrc && \
    echo "alias ...='cd ../..'" >> /home/ubuntu/.bashrc && \
    echo '' >> /home/ubuntu/.bashrc && \
    echo '# ROS2 workspace aliases' >> /home/ubuntu/.bashrc && \
    echo "alias cb='cd ~/LaneTracking/ros2_ws && colcon build && source install/setup.bash'" >> /home/ubuntu/.bashrc && \
    echo "alias cc='cd ~/LaneTracking/ros2_ws && rm -rf build install log'" >> /home/ubuntu/.bashrc && \
    echo "alias ct='cd ~/LaneTracking/ros2_ws && colcon test'" >> /home/ubuntu/.bashrc && \
    echo "alias s='source ~/.bashrc 2>/dev/null'" >> /home/ubuntu/.bashrc && \
    echo '' >> /home/ubuntu/.bashrc && \
    echo '# Quick navigation' >> /home/ubuntu/.bashrc && \
    echo "alias ws='cd ~/LaneTracking/ros2_ws 2>/dev/null'" >> /home/ubuntu/.bashrc && \
    echo "alias src='cd ~/LaneTracking/ros2_ws/src 2>/dev/null'" >> /home/ubuntu/.bashrc && \
    echo '' >> /home/ubuntu/.bashrc && \
    echo '# USB permission fix' >> /home/ubuntu/.bashrc && \
    echo 'alias fix_usb="sudo chmod 666 /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo \"No USB devices found\""' >> /home/ubuntu/.bashrc && \
    echo '' >> /home/ubuntu/.bashrc && \
    echo '# ROS2 utilities' >> /home/ubuntu/.bashrc && \
    echo "alias topics='ros2 topic list'" >> /home/ubuntu/.bashrc && \
    echo "alias nodes='ros2 node list'" >> /home/ubuntu/.bashrc && \
    echo "alias tf='ros2 run tf2_tools view_frames'" >> /home/ubuntu/.bashrc && \
    echo '' >> /home/ubuntu/.bashrc && \
    echo '# Auto-fix USB permissions on login' >> /home/ubuntu/.bashrc && \
    echo 'sudo chmod 666 /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true' >> /home/ubuntu/.bashrc && \
    echo '' >> /home/ubuntu/.bashrc && \
    echo '# Welcome message' >> /home/ubuntu/.bashrc && \
    echo 'echo "ROS2 Humble Docker Environment Ready!"' >> /home/ubuntu/.bashrc && \
    echo 'echo "Run ros2_build to build your workspace"' >> /home/ubuntu/.bashrc
    # echo 'export ROS_DOMAIN_ID=30' >> /home/ubuntu/.bashrc && \
    # echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> /home/ubuntu/.bashrc

# Install as root
USER root

# Create USB permission fix script
RUN echo '#!/bin/bash' > /usr/local/bin/fix-usb-permissions.sh && \
    echo 'chmod 666 /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true' >> /usr/local/bin/fix-usb-permissions.sh && \
    echo 'echo "USB permissions updated"' >> /usr/local/bin/fix-usb-permissions.sh && \
    chmod +x /usr/local/bin/fix-usb-permissions.sh

# Add USB permissions to sudoers
RUN echo 'ALL ALL=(ALL) NOPASSWD: /usr/local/bin/fix-usb-permissions.sh' >> /etc/sudoers && \
    echo 'ALL ALL=(ALL) NOPASSWD: /bin/chmod' >> /etc/sudoers

# Update root's bashrc
RUN echo "source /opt/ros/humble/install/setup.bash 2>/dev/null" >> /root/.bashrc
    
# Set proper permissions
RUN chown -R 1000:1000 /home/ubuntu && \
    chmod -R 755 /home/ubuntu

# Create ROS directories
RUN mkdir -p /home/ubuntu/.ros && \
    mkdir -p /home/ubuntu/.cache && \
    chown -R 1000:1000 /home/ubuntu/.ros /home/ubuntu/.cache

# Update rosdep as user
USER 1000
WORKDIR /home/ubuntu
ENV HOME=/home/ubuntu

RUN rosdep update --rosdistro humble 2>/dev/null || \
    echo "rosdep update failed, continuing..."

# Switch back to root for final setup
USER root

# Create udev rules for USB devices
RUN echo 'SUBSYSTEM=="tty", MODE="0666"' > /etc/udev/rules.d/99-usb-serial.rules

# Reset environment
ENV DEBIAN_FRONTEND=dialog

# Set working directory
WORKDIR /home/ubuntu

# The base image already handles VNC startup, so we don't need a custom entrypoint
# Just ensure USB permissions are set on container start
CMD ["/bin/bash"]

# Set the default shell to Bash and ensure .bashrc is sourced
SHELL ["/bin/bash", "-c"]