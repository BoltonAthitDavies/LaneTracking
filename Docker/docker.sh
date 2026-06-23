#!/bin/bash

IMAGE_NAME="lanetracking/ros:humble"
CONTAINER_NAME="lanetracking_container"

# Build the image
build_image() {
    echo "🚀 Building Docker image: $IMAGE_NAME"
    docker build -t $IMAGE_NAME .
}

# Run the container
run_container() {
    echo "▶️ Running Docker container: $CONTAINER_NAME"
    docker run -it --rm \
        --runtime nvidia \
        --network=host \
        --gpus all \
        --privileged \
        -v /tmp/argus_socket:/tmp/argus_socket \
        -v /etc/enctune.conf:/etc/enctune.conf \
        -v /etc/nv_tegra_release:/etc/nv_tegra_release \
        -v /tmp/nv_ds_performance:/tmp/nv_ds_performance \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v ~/LaneTracking:/home/ubuntu/LaneTracking \
        -e LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0 \
        --name $CONTAINER_NAME \
        $IMAGE_NAME
        # -v /dev/video1:/dev/video1 \
        # -v /dev/ttyCANable:/dev/ttyCANable \
        # -v /dev/ttyUSB0:/dev/ttyUSB0 \
}

# Exec into a running container
exec_container() {
    # Use the argument $1 if provided, otherwise default to $CONTAINER_NAME
    local TARGET_CONTAINER=${1:-$CONTAINER_NAME}

    echo "✅ Executing bash in container: $TARGET_CONTAINER"
    docker exec -it "$TARGET_CONTAINER" bash
}

# Help message
show_help() {
    echo "Usage: ./docker.sh [build|run|exec]"
    echo "  build : Build the Docker image"
    echo "  run   : Run the Docker container"
    echo "  exec [container_name_or_id] : Execute bash in a container."
    echo "        (Defaults to '$CONTAINER_NAME' if no name/id is given)"
}

# Main
if [ "$1" == "build" ]; then
    build_image
elif [ "$1" == "run" ]; then
    run_container
elif [ "$1" == "exec" ]; then
    exec_container "$2" # Pass the second argument (optional)
else
    show_help
fi