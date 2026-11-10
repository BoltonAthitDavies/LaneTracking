#!/usr/bin/env bash
# Entrypoint for the combined LaneTracking + wil_project container.
#
# Sources ROS 2, then whichever of the two workspace overlays already exist,
# and makes sure wil_project can find the baked ORB-SLAM3 / Pangolin tree.
# `docker exec` bypasses this file; interactive shells get the same environment
# from ~/.bashrc, which the Dockerfile writes.
set -e

# ROS setup scripts read unset variables.
set +u
source /opt/ros/humble/setup.bash
[ -f /opt/wil/rosdeps/setup.bash ] && source /opt/wil/rosdeps/setup.bash
[ -f /opt/wil/nav2/setup.bash ]    && source /opt/wil/nav2/setup.bash
set -u

WIL_WS="${WIL_WS:-$HOME/wil_project}"
LANE_WS="${LANE_WS:-$HOME/LaneTracking/ros2_ws}"

# ORB-SLAM3 and Pangolin are built into /opt/wil/thirdparty. `ws_build` points
# orbslam3_ros2 at them through its ORB_SLAM3_ROOT / THIRDPARTY_PREFIX cache
# variables, and the resulting node carries them in its DT_RPATH, so nothing
# needs to be linked or copied into the bind-mounted workspace here.
#
# The ORB-SLAM3 launch files resolve their vocabulary to
# ~/wil_project/thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt, which the submodule
# ships only as ORBvoc.txt.tar.gz. Normally it has been unpacked on the host
# (145 MB, gitignored by the submodule); warn if it is missing rather than
# writing into the mounted repository from in here.
if [ -d "$WIL_WS/thirdparty/ORB_SLAM3/Vocabulary" ] \
   && [ ! -e "$WIL_WS/thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt" ]; then
    echo "[wil] ORBvoc.txt is not unpacked in the mounted workspace." >&2
    echo "[wil]   launch with  vocabulary:=/opt/wil/thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt" >&2
    echo "[wil]   or unpack once:  cd ~/wil_project/thirdparty/ORB_SLAM3/Vocabulary && tar xzf ORBvoc.txt.tar.gz" >&2
fi

set +u
[ -f "$WIL_WS/install_docker/setup.bash" ] && source "$WIL_WS/install_docker/setup.bash"
{ [ -f "$LANE_WS/install_docker/setup.bash" ] && source "$LANE_WS/install_docker/setup.bash"; } \
    || { [ -f "$LANE_WS/install/setup.bash" ] && source "$LANE_WS/install/setup.bash"; }
set -u

# Serial / CAN adapters come up owned by root in a privileged container.
sudo /usr/local/bin/fix-usb-permissions.sh >/dev/null 2>&1 || true

mkdir -p "${XDG_RUNTIME_DIR:-/tmp/runtime-ros}" 2>/dev/null || true
exec "$@"
