# Copyright 2020-2023 Tiryoh<tiryoh@gmail.com>
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# This Dockerfile is based on https://github.com/AtsushiSaito/docker-ubuntu-sweb
# which is released under the Apache-2.0 license.

# Define common build arguments
ARG CUDA_VERSION=11.4.3
ARG UBUNTU_VERSION=20.04
ARG CUDNN_VERSION=8

# --- Build Stage for OpenCV ---
# Using the devel image with specified CUDA, cuDNN, and Ubuntu versions
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS opencv_builder

ARG TARGETPLATFORM
ARG OPENCV_VERSION=4.10.0
ENV OPENCV_VERSION=$OPENCV_VERSION
ARG CUDA_ARCH=8.7
ENV CUDA_ARCH=$CUDA_ARCH

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
	python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libxine2-dev \
    libglew-dev \
    libtiff5-dev \
    zlib1g-dev \
    libjpeg-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libpostproc-dev \
    libswscale-dev \
    libeigen3-dev \
    libtbb-dev \
    libgtk2.0-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN cd /opt/ &&\
    # Download and unzip OpenCV and opencv_contrib and delte zip files
    wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    # Create build folder and switch to it
    mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
    # Cmake configure
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=${CUDA_ARCH} \
        -DCMAKE_BUILD_TYPE=RELEASE \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. &&\
    # Make
    make -j $(nproc --all) && \
    make install && \
    ldconfig &&\
    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}

RUN apt-get update && apt-get upgrade -y &&\
    apt-get install -y \
        libvips \
        libvips-dev \
        vim
RUN apt-get update
RUN pip install --upgrade pip

# --- Final Image Stage ---
# Use the CUDA runtime image to keep the final image smaller
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG TARGETPLATFORM
LABEL maintainer="Tiryoh<tiryoh@gmail.com>"

SHELL ["/bin/bash", "-c"]

# FIX: Add this block to fix GPG errors before first apt update
RUN apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    apt-transport-https \
    software-properties-common \
    ubuntu-keyring && \
    rm -rf /var/lib/apt/lists/*

# Upgrade OS
RUN apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    apt-get autoclean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Install Ubuntu Mate desktop
RUN apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ubuntu-mate-desktop && \
    apt-get autoclean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Add Package
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tigervnc-standalone-server tigervnc-common \
    supervisor wget curl gosu git sudo python3-pip tini \
    vim sudo lsb-release locales \
    bash-completion tzdata terminator \
    dos2unix \
    libjpeg8 libpng16-16 libtiff5 libavcodec58 libavformat58 libswscale5 libavresample4 \
    libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
    libxvidcore4 \
    libx264-dev \
    libmp3lame0 \
    libv4l-0 \
    libxine2 libtbb2 libeigen3-dev \
    python3-dev python3-numpy \
    libvips-dev \
    libopenblas-base \
    libopenblas-dev \
    # MODIFIED: Changed libgfortran-dev to gfortran
    gfortran \
    gedit \
    && apt-get autoclean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Copy built OpenCV from the builder stage
COPY --from=opencv_builder /usr/local /usr/local
COPY --from=opencv_builder /etc/ld.so.conf.d/ /etc/ld.so.conf.d/
RUN ldconfig # Re-run ldconfig to ensure copied libs are found

# noVNC and Websockify
RUN git clone https://github.com/AtsushiSaito/noVNC.git -b add_clipboard_support /usr/lib/novnc
RUN pip install git+https://github.com/novnc/websockify.git@v0.10.0
RUN ln -s /usr/lib/novnc/vnc.html /usr/lib/novnc/index.html

# Set remote resize function enabled by default
RUN sed -i "s/UI.initSetting('resize', 'off');/UI.initSetting('resize', 'remote');/g" /usr/lib/novnc/app/ui.js

# Disable auto update and crash report
RUN sed -i 's/Prompt=.*/Prompt=never/' /etc/update-manager/release-upgrades
RUN sed -i 's/enabled=1/enabled=0/g' /etc/default/apport

# Install Firefox
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:mozillateam/ppa -y && \
    echo 'Package: *' > /etc/apt/preferences.d/mozilla-firefox && \
    echo 'Pin: release o=LP-PPA-mozillateam' >> /etc/apt/preferences.d/mozilla-firefox && \
    echo 'Pin-Priority: 1001' >> /etc/apt/preferences.d/mozilla-firefox && \
    apt-get update -q && \
    apt-get install -y \
    firefox && \
    apt-get autoclean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Install VSCodium
RUN wget https://gitlab.com/paulcarroty/vscodium-deb-rpm-repo/raw/master/pub.gpg \
    -O /usr/share/keyrings/vscodium-archive-keyring.asc && \
    echo 'deb [ signed-by=/usr/share/keyrings/vscodium-archive-keyring.asc ] https://paulcarroty.gitlab.io/vscodium-deb-rpm-repo/debs vscodium main' \
    | tee /etc/apt/sources.list.d/vscodium.list > /dev/null && \
    apt-get update -q && \
    apt-get install -y codium && \
    apt-get autoclean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# --- PyTorch Installation ---
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
    echo "Installing PyTorch for linux/amd64 with CUDA 11.8 compatible wheels..." && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    echo "PyTorch installation for linux/amd64 completed."; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
    echo "Installing PyTorch for linux/arm64 with CUDA 11.4 compatible wheels (or attempting build)..." && \
    # Crucial for ARM64: Find the correct PyTorch wheel for your specific ARM64 platform (e.g., Jetson)
    # and CUDA 11.4. This link is an *example* for JetPack 5.1.1 (which has CUDA 11.4).
    # You MUST verify and replace this URL with the correct one for your ARM64 device.
    # Refer to NVIDIA's PyTorch for Jetson documentation if using a Jetson:
    # https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
    export TORCH_INSTALL="https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"; \
    if curl --head --silent --fail "$TORCH_INSTALL"; then \
        python3 -m pip install --upgrade pip && \
        python3 -m pip install numpy=='1.24.4' && \
        python3 -m pip install --no-cache "$TORCH_INSTALL"; \
    else \
        echo "Pre-built PyTorch wheel for ARM64 with CUDA 11.4 not found at specified URL. Attempting to install generic PyTorch if possible, which might be CPU-only or fail."; \
        pip install torch torchvision torchaudio; \
        echo "Check if PyTorch is CUDA-enabled after this fallback installation for ARM64."; \
    fi && \
    echo "PyTorch installation for linux/arm64 completed."; \
    else \
    echo "Unsupported platform: $TARGETPLATFORM. Skipping PyTorch installation."; \
    fi

# Install ROS
ENV ROS_DISTRO foxy
# desktop or ros-base
ARG INSTALL_PACKAGE=desktop

RUN apt-get update -q && \
    apt-get install -y curl gnupg2 lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update -q && \
    apt-get install -y ros-${ROS_DISTRO}-${INSTALL_PACKAGE} \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-rosdep python3-vcstool && \
    rosdep init && \
    rm -rf /var/lib/apt/lists/*

RUN rosdep update

RUN apt-get update -q && \
    apt-get install -y \
    ros-${ROS_DISTRO}-gazebo-ros-pkgs \
    ros-${ROS_DISTRO}-ros-ign-gazebo && \
    rm -rf /var/lib/apt/lists/*

# Enable apt-get completion after running `apt-get update` in the container
RUN rm /etc/apt/apt.conf.d/docker-clean

COPY ./entrypoint.sh /
ENTRYPOINT [ "/bin/bash", "-c", "/entrypoint.sh" ]

ENV USER ubuntu
ENV PASSWD ubuntu