#!/bin/bash
# Build / run the combined LaneTracking + wil_project container.
#
#   ./docker.sh build          # build the image (long: see README)
#   ./docker.sh run            # start it and drop into a shell
#   ./docker.sh exec [name]    # second shell in the running container
#
# Both repositories are bind-mounted, not copied into the image, so code is
# edited on the host and built inside with `ws_build`.
set -u

IMAGE_NAME="robot/ros:humble-combined"
CONTAINER_NAME="robot_combined"

LANE_HOST="${LANE_HOST:-$HOME/LaneTracking}"
WIL_HOST="${WIL_HOST:-$HOME/wil_project}"

# pyrealsense2 built from librealsense source on the host: the aarch64 pip
# wheel (<= 2.55.1.6486) mis-enumerates devices on this kernel. Both paths are
# needed together -- the .so has an absolute RUNPATH into the build tree.
# See LaneTracking/README.md section 1.
PYRS_HOST="${PYRS_HOST:-$HOME/torch-py38/lib/python3.8/site-packages/pyrealsense2}"
LIBRS_HOST="${LIBRS_HOST:-$HOME/librealsense-src/build/Release}"

build_image() {
    local here; here="$(cd "$(dirname "$0")" && pwd)"

    # The image builds ORB-SLAM3 and Pangolin with wil_project's own
    # build_orbslam3.sh, so the pinned commits stay defined in one place. That
    # script lives in the other repository, outside this build context, so it
    # is staged in here first (gitignored).
    if [ ! -f "$WIL_HOST/build_orbslam3.sh" ]; then
        echo "error: $WIL_HOST/build_orbslam3.sh not found." >&2
        echo "       Set WIL_HOST=/path/to/wil_project and retry." >&2
        return 1
    fi
    cp "$WIL_HOST/build_orbslam3.sh" "$here/build_orbslam3.sh"

    # The ORB-SLAM3 revision to build is taken from wil_project's SUBMODULE pin,
    # not from build_orbslam3.sh's own ORB_COMMIT, which is stale (it pins an
    # ancestor that lacks the APIs orbslam3_ros2/src/slam_wrapper.cpp calls).
    # `git rev-parse HEAD:<path>` returns the gitlink SHA the superproject
    # records, bare. Do NOT parse `git submodule status` and strip its leading
    # +/-/U with `tr -d '+-U'`: tr reads that as the RANGE 0x2B-0x55, which
    # includes every digit, so it silently eats the numerals out of the SHA
    # (a639fa46...20b -> afaddffddefbb).
    local orb_commit=""
    orb_commit="$(cd "$WIL_HOST" && git rev-parse "HEAD:thirdparty/ORB_SLAM3" 2>/dev/null)"
    if [ -n "$orb_commit" ]; then
        echo "ORB-SLAM3 submodule pin: $orb_commit"
    else
        echo "warning: could not read the thirdparty/ORB_SLAM3 submodule pin;" >&2
        echo "         falling back to build_orbslam3.sh's own ORB_COMMIT, which" >&2
        echo "         may be too old for orbslam3_ros2 to compile against." >&2
    fi

    echo "Building $IMAGE_NAME (expect hours: nav2, RTAB-Map and ORB-SLAM3 are source builds)"
    docker build -f "$here/Dockerfile" -t "$IMAGE_NAME" \
        --build-arg ORB_SLAM3_COMMIT="$orb_commit" "$here"
}

run_container() {
    local opt=()
    [ -d "$PYRS_HOST" ]  && opt+=(-v "$PYRS_HOST:/usr/local/lib/python3.8/dist-packages/pyrealsense2:ro")
    [ -d "$LIBRS_HOST" ] && opt+=(-v "$LIBRS_HOST:$LIBRS_HOST:ro")
    [ -d /media/nvidia ] && opt+=(-v /media/nvidia:/media/nvidia:ro)
    [ -d "$WIL_HOST" ]   || echo "warning: $WIL_HOST not found -- wil_project will be missing in the container" >&2

    echo "Running $CONTAINER_NAME"
    docker run -it --rm \
    --runtime nvidia \
    --network=host \
    --ipc=host \
    --pid=host \
    --privileged \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v /etc/enctune.conf:/etc/enctune.conf \
    -v /etc/nv_tegra_release:/etc/nv_tegra_release \
    -v /tmp/nv_ds_performance:/tmp/nv_ds_performance \
    -v /dev/bus/usb:/dev/bus/usb \
    -e ROS_DOMAIN_ID=29 \
    -e DISPLAY="${DISPLAY:-:0}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$LANE_HOST:/home/ubuntu/LaneTracking" \
    -v "$WIL_HOST:/home/ubuntu/wil_project" \
    -v /dev/video0:/dev/video0 \
    -v /dev/video1:/dev/video1 \
    -v /dev/video8:/dev/video8 \
    -v /dev/video9:/dev/video9 \
    -v /dev/ttyCANable:/dev/ttyCANable \
    -v /dev/ttyUSB0:/dev/ttyUSB0 \
    "${opt[@]}" \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME"
}

# run_container() {
#     local opt=()
#     [ -d "$PYRS_HOST" ]  && opt+=(-v "$PYRS_HOST:/usr/local/lib/python3.8/dist-packages/pyrealsense2:ro")
#     [ -d "$LIBRS_HOST" ] && opt+=(-v "$LIBRS_HOST:$LIBRS_HOST:ro")
#     [ -d /media/nvidia ] && opt+=(-v /media/nvidia:/media/nvidia:ro)
#     [ -d "$WIL_HOST" ]   || echo "warning: $WIL_HOST not found -- wil_project will be missing in the container" >&2

#     echo "Running $CONTAINER_NAME"
#     docker run -it --rm \
#     --runtime nvidia \
#     --network=host \
#     --privileged \
#     -v /tmp/argus_socket:/tmp/argus_socket \
#     -v /etc/enctune.conf:/etc/enctune.conf \
#     -v /etc/nv_tegra_release:/etc/nv_tegra_release \
#     -v /tmp/nv_ds_performance:/tmp/nv_ds_performance \
#     -v /dev/bus/usb:/dev/bus/usb \
#     -e ROS_DOMAIN_ID=29 \
#     -e DISPLAY="$DISPLAY" \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -v ~/LaneTracking:/home/ubuntu/LaneTracking \
#     -v "$WIL_HOST:/home/ubuntu/wil_project" \
#     -v /dev/video0:/dev/video0 \
#     -v /dev/video1:/dev/video1 \
#     -v /dev/video8:/dev/video8 \
#     -v /dev/video9:/dev/video9 \
#     -v /dev/ttyCANable:/dev/ttyCANable \
#     -v /dev/ttyUSB0:/dev/ttyUSB0 \
#     -v /home/nvidia/torch-py38/lib/python3.8/site-packages/pyrealsense2:/usr/local/lib/python3.8/dist-packages/pyrealsense2:ro \
#     -v /home/nvidia/librealsense-src/build/Release:/home/nvidia/librealsense-src/build/Release:ro \
#     "${opt[@]}" \
#     --name "$CONTAINER_NAME" \
#     "$IMAGE_NAME"
#     # --ipc=host \
#     # docker run -it --rm \
#     #     --runtime nvidia \
#     #     --network=host \
#     #     --ipc=host \
#     #     --gpus all \
#     #     --privileged \
#     #     --ipc=host \
#     #     -v /tmp/argus_socket:/tmp/argus_socket \
#     #     -v /etc/enctune.conf:/etc/enctune.conf \
#     #     -v /etc/nv_tegra_release:/etc/nv_tegra_release \
#     #     -v /tmp/nv_ds_performance:/tmp/nv_ds_performance \
#     #     -v /dev/bus/usb:/dev/bus/usb \
#     #     -e ROS_DOMAIN_ID=29 \
#     #     -e DISPLAY="$DISPLAY" \
#     #     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     #     -v "$LANE_HOST:/home/ubuntu/LaneTracking" \
#     #     -v /dev/video0:/dev/video0 \
#     #     -v /dev/video1:/dev/video1 \
#     #     -v /dev/video8:/dev/video8 \
#     #     -v /dev/video9:/dev/video9 \
#     #     -v "$WIL_HOST:/home/ubuntu/wil_project" \
#     #     -e LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0 \
#     #     "${opt[@]}" \
#     #     --name "$CONTAINER_NAME" \
#     #     "$IMAGE_NAME"
# }

exec_container() {
    local target=${1:-$CONTAINER_NAME}
    echo "Executing bash in container: $target"
    docker exec -it "$target" bash
}

case "${1:-}" in
    build) build_image ;;
    run)   run_container ;;
    exec)  exec_container "${2:-}" ;;
    *)
        echo "Usage: ./docker.sh [build|run|exec]"
        echo "  build : build $IMAGE_NAME"
        echo "  run   : run $CONTAINER_NAME with both repos bind-mounted"
        echo "  exec [container] : bash in a running container (default $CONTAINER_NAME)"
        ;;
esac
