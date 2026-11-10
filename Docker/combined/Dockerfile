# syntax=docker/dockerfile:1.6
# Combined image: LaneTracking (robot I/O, CAN, cameras, lane detection) and
# wil_project (VINS-Fusion / ORB-SLAM3 / RTAB-Map / Nav2) in ONE container.
#
# WHY THIS IS A PORT AND NOT A MERGE OF THE TWO DOCKERFILES
#   LaneTracking/Docker/Dockerfile  dustynv/ros:...-l4t-r35.4.1 -> Ubuntu 20.04,
#                                   ROS 2 Humble BUILT FROM SOURCE in
#                                   /opt/ros/humble/install, torch 2.0.0+nv
#                                   (L4T CUDA 11.4), OpenCV 4.5.0 with CUDA.
#   wil_project/docker/Dockerfile   osrf/ros:humble-desktop -> Ubuntu 22.04,
#                                   ROS from apt, torch 2.5.1+cu121 from PyPI.
#
#   On this Jetson (aarch64, L4T R35) the second one cannot be built at all:
#   there is no aarch64 cu121 wheel, and JetPack 5 has no CUDA torch for 22.04.
#   And every `ros-humble-*` apt package wil_project asks for is jammy-only, so
#   on focal it has to come from source. Hence: keep the L4T base, source-build
#   what wil_project needs on top.
#
# WHAT IS DELIBERATELY NOT HERE
#   Gazebo Fortress, ros_gz, the AWS warehouse world and gz_kinematic_trajectory.
#   ros_gz for Humble has no focal binaries, and the real rig does not simulate.
#   wil_project's real-rig launches (mapping_real, localize_real, navigate_real,
#   nav2_real) need none of it. Also skipped: rtabmap's Qt GUI/databaseViewer,
#   rtabmap_odom (the VO being replaced), nav2_amcl and the DWB controller
#   (nav2_real.yaml uses regulated pure pursuit + Smac hybrid).
#
# NEITHER REPOSITORY IS COPIED IN. Both are bind-mounted by docker.sh and built
# inside with `ws_build`, so a code change does not mean an image rebuild.
#
# Build:  ./docker.sh build      (hours; nav2 + RTAB-Map + ORB-SLAM3 from source)
FROM dustynv/ros:humble-desktop-pytorch-l4t-r35.4.1

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS=yes
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ---------------------------------------------------------------------------
# 1. LaneTracking's own system packages (same set as Docker/Dockerfile)
# ---------------------------------------------------------------------------
# The base image is from 2023 and its ROS 2 apt key EXPIRED on 2025-06-01.
# Without this refresh `apt-get update` fails with
#   EXPKEYSIG F42ED6FBAB17C654 Open Robotics / repository is not signed
# and exits non-zero, so the `&& apt-get install` after it never runs. The
# current key from rosdistro is valid to 2030-06-01.
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg

# This build downloads a lot over a long time; a single transient failure
# should not throw away hours of work. (It already did once: every Packages
# index timed out mid-build while the InRelease files had come down fine.)
RUN printf '%s\n' \
        'Acquire::Retries "5";' \
        'Acquire::http::Timeout "60";' \
        'Acquire::https::Timeout "60";' \
        > /etc/apt/apt.conf.d/99-retries

RUN apt-get update && apt-get install -y --no-install-recommends \
        iputils-ping curl wget ca-certificates gnupg lsb-release \
        software-properties-common git htop nano vim usbutils net-tools iproute2 \
        python3-pip python3-dev python3-venv python3-setuptools python3-wheel \
        python3-empy python3-numpy python3-yaml python3-lark \
        python3-catkin-pkg python3-colcon-common-extensions python3-rosdep \
        python3-vcstool \
        libjpeg-dev libpng-dev libtiff-dev cmake build-essential file \
        qtbase5-dev libqt5svg5-dev libqt5websockets5-dev libqt5opengl5-dev \
        libqt5x11extras5-dev libprotoc-dev protobuf-compiler \
        libzmq3-dev liblz4-dev libzstd-dev libmosquittopp-dev \
        mosquitto mosquitto-clients \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. System packages wil_project needs on top
#
#    NOT libopencv-dev: this image already carries a CUDA-enabled OpenCV 4.5.0
#    installed by the base as opencv-dev/opencv-libs into /lib with its CMake
#    config in /usr/lib/cmake/opencv4. apt's libopencv-dev is 4.5.4 and would
#    sit beside it, so RTAB-Map, ORB-SLAM3, VINS and cv_bridge could end up
#    linking two different OpenCVs -- the exact link-time conflict recorded in
#    rtabmap_ros/MEMORY_CLAUDE/README.md section 1.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Ceres 2.2 (VINS) and g2o (RTAB-Map graph optimisation)
        libgoogle-glog-dev libgflags-dev libsuitesparse-dev libmetis-dev \
        libatlas-base-dev libtbb-dev \
        # RTAB-Map core
        libsqlite3-dev libpcl-dev libproj-dev libyaml-cpp-dev zlib1g-dev \
        # Pangolin 0.9 + ORB-SLAM3
        libglew-dev libegl1-mesa-dev libgl1-mesa-dev libepoxy-dev \
        libwayland-dev libxkbcommon-dev libx11-dev libssl-dev \
        libboost-serialization-dev libboost-system-dev libboost-filesystem-dev \
        libboost-coroutine-dev libboost-thread-dev \
        # Nav2: OMPL (Smac), nlohmann json (Smac), GraphicsMagick + SDL
        # (nav2_map_server), boost program_options, ncurses (BehaviorTree.CPP)
        libompl-dev nlohmann-json3-dev \
        libgraphicsmagick++1-dev libgraphicsmagick1-dev \
        libsdl1.2-dev libsdl-image1.2-dev \
        libboost-program-options-dev libncurses-dev \
        # runtime GL for rviz2 / Pangolin windows
        libgl1-mesa-dri libglx-mesa0 mesa-utils x11-apps \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 3. Python
#
#    torch 2.0.0+nv23.05 and cv2 4.5.0 are already in the base and are the only
#    CUDA-capable builds available for JetPack 5, so nothing here may upgrade
#    them. ultralytics is installed with --no-deps for exactly that reason: its
#    dependency list would pull opencv-python and a CPU-only torch over the top.
#
#    rosbags is 0.9.23, NOT the 0.11.5 in wil_project/docker/requirements.txt:
#    this base is Python 3.8 and every rosbags >= 0.10 requires >= 3.10. 0.9.23
#    still exposes rosbags.typesys.Stores, get_typestore and highlevel.AnyReader,
#    which is all script/bag_to_mp4.py uses, so it runs unchanged.
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir \
        "numpy<1.24" scipy matplotlib pyserial catkin_pkg lark \
        colcon-common-extensions \
        python-can "python-can[serial]" cantools can-isotp \
        pandas transforms3d addict torchvision==0.15.1 pycuda \
        paho-mqtt \
    && pip3 install --no-cache-dir rosbags==0.9.23 \
    && pip3 install --no-cache-dir --no-deps ultralytics==8.4.138 \
    && pip3 install --no-cache-dir py-cpuinfo tqdm psutil seaborn requests \
                                  ultralytics-thop

# ---------------------------------------------------------------------------
# 4. /opt/ros/humble/setup.bash shim
#
#    The base built ROS from source, so its setup script is at
#    /opt/ros/humble/install/setup.bash. 15 files across wil_project source
#    the conventional path; this makes both work.
# ---------------------------------------------------------------------------
RUN if [ ! -f /opt/ros/humble/setup.bash ]; then \
        printf '%s\n' \
            '# Shim: this image builds ROS 2 from source, so the real setup' \
            '# script lives one level down. wil_project scripts expect this path.' \
            'source /opt/ros/humble/install/setup.bash' \
            > /opt/ros/humble/setup.bash ; \
    fi

# ---------------------------------------------------------------------------
# 5. Ceres 2.2.0 -- VINS-Fusion uses the ceres::Manifold API, which apt's
#    1.14 predates. SuiteSparse is on (g2o/RTAB-Map want it anyway).
# ---------------------------------------------------------------------------
RUN git clone --depth 1 --branch 2.2.0 \
        https://github.com/ceres-solver/ceres-solver.git /tmp/ceres \
    && cmake -S /tmp/ceres -B /tmp/ceres/build \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
        -DBUILD_BENCHMARKS=OFF -DUSE_CUDA=OFF \
    && cmake --build /tmp/ceres/build -j"$(nproc)" \
    && cmake --install /tmp/ceres/build \
    && ldconfig \
    && rm -rf /tmp/ceres

# ---------------------------------------------------------------------------
# 6. g2o -- NOT optional. Without a g2o or GTSAM optimiser RTAB-Map falls back
#    to TORO, which cannot carry gravity constraints, and the real-rig map
#    depends on them (Mem/UseOdomGravity + Optimizer/GravitySigma; see
#    rtabmap_ros/MEMORY_CLAUDE/README.md fix 7 -- 0 -> 164 gravity links).
#    focal has no libg2o-dev, so it is built from the release RTAB-Map 0.23
#    is known to work with.
# ---------------------------------------------------------------------------
#    The tag really is "20201223_git" -- upstream suffixes every release tag
#    with _git. This is the snapshot ros-humble-libg2o packages, i.e. the one
#    the 22.04 host's apt RTAB-Map was built against.
RUN git clone --depth 1 --branch 20201223_git \
        https://github.com/RainerKuemmerle/g2o.git /tmp/g2o \
    && cmake -S /tmp/g2o -B /tmp/g2o/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/wil/g2o \
        -DBUILD_SHARED_LIBS=ON -DG2O_BUILD_APPS=OFF -DG2O_BUILD_EXAMPLES=OFF \
        -DG2O_USE_OPENGL=OFF -DG2O_BUILD_SLAM2D_TYPES=ON \
    && cmake --build /tmp/g2o/build -j"$(nproc)" \
    && cmake --install /tmp/g2o/build \
    && rm -rf /tmp/g2o

ENV CMAKE_PREFIX_PATH=/opt/wil/g2o:/opt/wil/rtabmap
ENV LD_LIBRARY_PATH=/opt/wil/g2o/lib:/opt/wil/rtabmap/lib:${LD_LIBRARY_PATH}

# g2o's installed solvers/csparse/csparse_extension.h does `#include <cs.h>`,
# but Ubuntu ships CSparse's header as /usr/include/suitesparse/cs.h and g2o's
# exported CMake config does not carry that directory. Anything that compiles
# against g2o headers therefore fails with "cs.h: No such file or directory" --
# RTAB-Map's OptimizerG2O.cpp does. CPATH fixes it for every later layer at
# once, including the colcon builds, rather than per-cmake-invocation.
ENV CPATH=/usr/include/suitesparse${CPATH:+:$CPATH}

# ---------------------------------------------------------------------------
# 7. RTAB-Map core 0.23.7
#
#    The rtabmap_ros fork in wil_project is pinned to upstream 2eef2b3, whose
#    rtabmap_conversions/CMakeLists.txt wants RTABMap >= 0.23.5. 0.23.7 is the
#    version it was validated against (apt on the 22.04 host). GUI, tools and
#    examples are off: nothing in the stack opens them, and pose_gate.py reads
#    the .db with plain sqlite3.
# ---------------------------------------------------------------------------
RUN git clone --depth 1 --branch 0.23.7 \
        https://github.com/introlab/rtabmap.git /tmp/rtabmap \
    && cmake -S /tmp/rtabmap -B /tmp/rtabmap/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/wil/rtabmap \
        -DWITH_QT=OFF -DBUILD_APP=OFF -DBUILD_TOOLS=OFF -DBUILD_EXAMPLES=OFF \
        -DWITH_G2O=ON -DG2O_ROOT=/opt/wil/g2o \
        -DWITH_TORO=ON -DWITH_GTSAM=OFF -DWITH_CERES=OFF \
        -DWITH_PYTHON=OFF -DWITH_REALSENSE2=OFF -DWITH_OPENNI2=OFF \
        -DWITH_FREENECT=OFF -DWITH_FREENECT2=OFF -DWITH_K4A=OFF \
        -DWITH_ZED=OFF -DWITH_DEPTHAI=OFF -DWITH_LIBLAS=OFF \
    && cmake --build /tmp/rtabmap/build -j"$(nproc)" \
    && cmake --install /tmp/rtabmap/build \
    && ldconfig \
    && rm -rf /tmp/rtabmap

# ---------------------------------------------------------------------------
# 8. The ROS 2 packages wil_project needs that the base image lacks and focal
#    apt cannot supply. Own overlay prefix so it can be sourced on its own.
#
#      bond_core            nav2_util's bond heartbeat
#      diagnostics          diagnostic_updater (nav2_costmap_2d, rtabmap_sync)
#      perception_pcl       pcl_ros -- find_package(... REQUIRED) at
#                           rtabmap_util/CMakeLists.txt:25, the one hard dep
#      BehaviorTree.CPP 3.8 nav2_behavior_tree / bt_navigator
#      joint_state_publisher  robot_description bringup
#      rosbag2_storage_mcap   mcap bag playback (wil_project's bags)
# ---------------------------------------------------------------------------
RUN mkdir -p /opt/wil/rosdeps_src && cd /opt/wil/rosdeps_src \
    && git clone --depth 1 --branch humble https://github.com/ros/bond_core.git \
    && git clone --depth 1 --branch ros2-humble https://github.com/ros/diagnostics.git \
    && git clone --depth 1 --branch humble https://github.com/ros-perception/perception_pcl.git \
    && git clone --depth 1 --branch v3.8 https://github.com/BehaviorTree/BehaviorTree.CPP.git \
    && git clone --depth 1 --branch humble https://github.com/ros/joint_state_publisher.git \
    && git clone --depth 1 https://github.com/ros-tooling/rosbag2_storage_mcap.git \
    && touch perception_pcl/pcl_conversions/COLCON_IGNORE \
    && source /opt/ros/humble/install/setup.bash \
    && colcon build \
        --base-paths /opt/wil/rosdeps_src \
        --build-base /opt/wil/rosdeps_build \
        --install-base /opt/wil/rosdeps \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        --packages-up-to bondcpp diagnostic_updater pcl_ros \
                         behaviortree_cpp_v3 joint_state_publisher \
                         rosbag2_storage_mcap \
    && rm -rf /opt/wil/rosdeps_build /opt/wil/rosdeps_src /root/.ros/log log

# ---------------------------------------------------------------------------
# 9. Nav2 (humble), only the servers and plugins nav2_real.yaml actually names:
#    regulated pure pursuit, Smac hybrid, SimpleSmoother, the four behaviours,
#    static + inflation costmap layers, map_server, velocity_smoother,
#    lifecycle_manager, bt_navigator. nav2_amcl, the DWB controller,
#    nav2_collision_monitor, nav2_waypoint_follower, nav2_constrained_smoother
#    and nav2_system_tests are left out.
#
#    nav2_common supplies launch.RewrittenYaml, which nav2_real.launch.py
#    imports. Do NOT put nav2_bringup in this list: it depends on the
#    `navigation2` METAPACKAGE, which drags in every nav2 package there is --
#    nav2_mppi_controller (needs xtensor, no focal apt package), nav2_amcl,
#    nav2_collision_monitor, nav2_constrained_smoother and the whole DWB
#    stack -- silently defeating the subset above. Nothing in wil_project's
#    real-rig launches imports nav2_bringup; vslam_navigation lists it as an
#    exec_depend only, which does not affect the build.
# ---------------------------------------------------------------------------
RUN mkdir -p /opt/wil/nav2_src && cd /opt/wil/nav2_src \
    && git clone --depth 1 --branch humble \
        https://github.com/ros-navigation/navigation2.git \
    && source /opt/ros/humble/install/setup.bash \
    && source /opt/wil/rosdeps/setup.bash \
    && colcon build \
        --base-paths /opt/wil/nav2_src \
        --build-base /opt/wil/nav2_build \
        --install-base /opt/wil/nav2 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        --packages-up-to nav2_common nav2_map_server nav2_controller \
                         nav2_planner nav2_smoother nav2_behaviors \
                         nav2_bt_navigator nav2_velocity_smoother \
                         nav2_lifecycle_manager nav2_smac_planner \
                         nav2_regulated_pure_pursuit_controller \
                         nav2_rviz_plugins \
    && rm -rf /opt/wil/nav2_build /opt/wil/nav2_src /root/.ros/log log

# ---------------------------------------------------------------------------
# 10. ORB-SLAM3 (fork, pinned commit) + Pangolin 0.9.1, built by the repo's own
#     script so the pins stay in one place. It builds into <script dir>/thirdparty,
#     so running it from /opt/wil puts the result in /opt/wil/thirdparty, which
#     the entrypoint symlinks into the bind-mounted workspace.
#     docker.sh stages this file from the wil_project checkout.
# ---------------------------------------------------------------------------
#     build_orbslam3.sh's own ORB_COMMIT is STALE relative to the submodule it
#     is meant to reproduce: the script pins 5c2b00f, while wil_project pins
#     thirdparty/ORB_SLAM3 at a639fa4 ("Patch 15: YAML switches for the
#     inertial initialisation"), a strict descendant. orbslam3_ros2's
#     slam_wrapper.cpp uses APIs that exist only in the newer one --
#     ConfigureEvaluationLogging, GetEvaluationState, SaveEvaluationTrajectories,
#     SetBoxVoteMinAge, BoxVote and the extended TrackStereo overload -- so
#     building the script's pin gives "class ORB_SLAM3::System has no member
#     named ...". docker.sh reads the real pin out of the submodule and passes
#     it here, so the two cannot drift apart again.
ARG ORB_SLAM3_COMMIT=""
COPY build_orbslam3.sh /opt/wil/build_orbslam3.sh
RUN chmod +x /opt/wil/build_orbslam3.sh \
    && if [ -n "$ORB_SLAM3_COMMIT" ]; then \
           echo "[image] ORB-SLAM3 pinned to submodule commit $ORB_SLAM3_COMMIT" ; \
           sed -i "s/^ORB_COMMIT=.*/ORB_COMMIT=\"$ORB_SLAM3_COMMIT\"/" /opt/wil/build_orbslam3.sh ; \
           grep -n '^ORB_COMMIT=' /opt/wil/build_orbslam3.sh ; \
       fi \
    && cd /opt/wil && ./build_orbslam3.sh \
    && rm -rf /opt/wil/thirdparty/ORB_SLAM3/.git /opt/wil/thirdparty/Pangolin

# ---------------------------------------------------------------------------
# 11. User, paths and shell
#
#     One UID 1000 user, `ubuntu`, as in LaneTracking/Docker/Dockerfile.
#     wil_project hardcodes /home/ambushee/wil_project in 13 code and config
#     files (the VINS stereo_imu.yaml files, two VINS launch files,
#     dynamic_detector_node.py, two scripts). A symlink resolves all of them
#     with no edits, so the host checkout stays clean.
# ---------------------------------------------------------------------------
RUN mkdir -p /home/ubuntu/.ros /home/ubuntu/.cache /home/ubuntu/wil_project \
             /home/ubuntu/LaneTracking \
    && ln -sfn /home/ubuntu /home/ambushee \
    && chown -R 1000:1000 /home/ubuntu /opt/wil

# A REAL account for uid 1000, not just `USER 1000`.
#
# LaneTracking/Docker/Dockerfile switches to `USER 1000` without ever creating
# the user, which leaves the process as uid=1000 gid=0(root) with no passwd
# entry. Two things break:
#   * colcon calls getpwuid() for the build log metadata and dies with
#     KeyError: 'getpwuid(): uid not found: 1000'
#   * gid 0 means everything written into the bind-mounted repositories lands
#     as <user>:root on the host (seen on wil_project/build_docker).
# Creating group and user 1000 fixes both, and makes host files come out
# nvidia:nvidia since the host's own user is 1000:1000 too.
#
# The supplementary groups matter just as much. --privileged passes the device
# NODES through but grants no group membership, so without `video` the CSI
# cameras are unreadable: dual_camera.launch.py opens /dev/video8 and
# /dev/video9 ("vi-output, nru2mp"), which are crw-rw---- root:video. The
# RealSense devices hide this, being crw-rw-rw- world-writable with the IMU
# arriving over /dev/bus/usb -- so the IMU works while images silently do not.
# video=44 and plugdev=46 in this image match the host, so joining by name
# lands on the right gids. dialout covers /dev/ttyUSB* and /dev/ttyACM*
# (CAN adapter, HWT101CT) without relying on the chmod 666 helper.
RUN (getent group 1000 >/dev/null || groupadd -g 1000 ubuntu) \
    && (getent passwd 1000 >/dev/null \
        || useradd -u 1000 -g 1000 -d /home/ubuntu -s /bin/bash ubuntu) \
    && for g in sudo video plugdev dialout; do \
           getent group "$g" >/dev/null && usermod -aG "$g" ubuntu || true; \
       done \
    && chown -R 1000:1000 /home/ubuntu /opt/wil

RUN printf '%s\n' '#!/bin/bash' \
        'chmod 666 /dev/ttyACM* /dev/ttyUSB* /dev/ttyCANable* 2>/dev/null || true' \
        > /usr/local/bin/fix-usb-permissions.sh \
    && chmod +x /usr/local/bin/fix-usb-permissions.sh \
    && echo 'ALL ALL=(ALL) NOPASSWD: /usr/local/bin/fix-usb-permissions.sh' >> /etc/sudoers \
    && echo 'ALL ALL=(ALL) NOPASSWD: /bin/chmod' >> /etc/sudoers \
    && echo "source /opt/ros/humble/install/setup.bash 2>/dev/null" >> /root/.bashrc

# Apt-layout compatibility symlinks, the companion to the setup.bash shim above.
# This base builds ROS from source into /opt/ros/humble/install, but packages
# written against the apt layout look under /opt/ros/$ROS_DISTRO directly, and
# some do it with NO_DEFAULT_PATH so nothing else can rescue them.
# rtabmap_msgs/CMakeLists.txt:15 is the case in hand -- an aarch64-only block
# (upstream issues #1285, #1288) doing
#   find_library(rcutils_LIB NAMES rcutils
#                PATHS "/opt/ros/$ENV{ROS_DISTRO}/lib" NO_DEFAULT_PATH REQUIRED)
# which fails here with "Could not find rcutils_LIB". It never fires on an
# x86_64 host, which is why wil_project never hit it.
# Kept in this late layer on purpose: it is only needed when the workspaces are
# built, so putting it here avoids invalidating the hours of source-build layers
# above it.
RUN for d in lib include share bin; do \
        [ -e "/opt/ros/humble/$d" ] || ln -sfn "/opt/ros/humble/install/$d" "/opt/ros/humble/$d"; \
    done

# ROS 2 renamed its C++ headers from .h to .hpp and backported the change to
# Humble; current Humble ships BOTH spellings. This base image's ROS is a
# source build from Nov 2023, which predates the backport and has only .h, so
# code written against today's Humble fails on headers that do exist upstream.
# The rtabmap_ros fork needs eleven of them -- cv_bridge/cv_bridge.hpp,
# tf2_ros/{buffer,transform_listener,transform_broadcaster}.hpp, tf2/utils.hpp,
# image_geometry/{pinhole,stereo}_camera_model.hpp and four message_filters
# headers -- and rtabmap_conversions is where it first stops:
#   MsgConversion.h:32: fatal error: tf2_ros/buffer.hpp: No such file
# Rather than chase them one at a time, give every .h a sibling .hpp symlink
# where none exists. Only ever additive: an existing .hpp is left alone.
RUN cd /opt/ros/humble/install/include \
    && find . -name '*.h' | while read -r h; do \
           hpp="${h%.h}.hpp"; \
           [ -e "$hpp" ] || ln -s "$(basename "$h")" "$hpp"; \
       done \
    && echo "[image] .hpp shims created: $(find . -name '*.hpp' -type l | wc -l)"

COPY --chmod=755 entrypoint.sh /usr/local/bin/robot_entrypoint
COPY --chmod=755 ws_build      /usr/local/bin/ws_build

RUN cat > /home/ubuntu/.bashrc <<'BASHRC' && chown 1000:1000 /home/ubuntu/.bashrc
export PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "
export TERM=xterm-256color

# --- ROS 2 and the baked overlays ------------------------------------------
source /opt/ros/humble/install/setup.bash
[ -f /opt/wil/rosdeps/setup.bash ] && source /opt/wil/rosdeps/setup.bash
[ -f /opt/wil/nav2/setup.bash ]    && source /opt/wil/nav2/setup.bash

# --- the two bind-mounted workspaces, wil first then LaneTracking ----------
if [ -f ~/wil_project/install_docker/setup.bash ]; then
    source ~/wil_project/install_docker/setup.bash
else
    echo "[wil] no ~/wil_project/install_docker -- run: ws_build wil" >&2
fi
if [ -f ~/LaneTracking/ros2_ws/install_docker/setup.bash ]; then
    source ~/LaneTracking/ros2_ws/install_docker/setup.bash
elif [ -f ~/LaneTracking/ros2_ws/install/setup.bash ]; then
    source ~/LaneTracking/ros2_ws/install/setup.bash
else
    echo "[lane] no ~/LaneTracking/ros2_ws/install_docker -- run: ws_build lane" >&2
fi

[ -f /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash ] \
    && source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash

# --- aliases ----------------------------------------------------------------
alias ll='ls -alF'
alias la='ls -A'
alias ..='cd ..'
alias ws='cd ~/LaneTracking/ros2_ws'
alias src='cd ~/LaneTracking/ros2_ws/src'
alias wil='cd ~/wil_project'
alias cb='ws_build lane'
alias cbw='ws_build wil'
alias cc='cd ~/LaneTracking/ros2_ws && rm -rf build_docker install_docker log_docker'
alias s='source ~/.bashrc 2>/dev/null'
alias topics='ros2 topic list'
alias nodes='ros2 node list'
alias fix_usb='sudo /usr/local/bin/fix-usb-permissions.sh'

sudo /usr/local/bin/fix-usb-permissions.sh >/dev/null 2>&1 || true
cd ~ 2>/dev/null

echo "LaneTracking + wil_project container"
echo "  ws_build          build both workspaces (wil_project, then LaneTracking)"
echo "  ws_build wil      only wil_project    ws_build lane   only LaneTracking"
BASHRC

USER 1000
ENV HOME=/home/ubuntu
ENV ROS_DOMAIN_ID=29
ENV WIL_WS=/home/ubuntu/wil_project
ENV LANE_WS=/home/ubuntu/LaneTracking/ros2_ws
WORKDIR /home/ubuntu

RUN rosdep update --rosdistro humble 2>/dev/null || echo "rosdep update failed, continuing..."

USER root
RUN echo 'SUBSYSTEM=="tty", MODE="0666"' > /etc/udev/rules.d/99-usb-serial.rules
ENV DEBIAN_FRONTEND=dialog

USER 1000
WORKDIR /home/ubuntu
ENTRYPOINT ["/usr/local/bin/robot_entrypoint"]
CMD ["bash"]
