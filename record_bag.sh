#!/usr/bin/env bash
# Record the standard topic set to a rosbag, using compressed camera streams
# instead of raw image_raw. Raw 1920x1080 frames from both cameras drive
# ~90MB/s of sustained writes, which overruns rosbag2's sqlite3 storage
# backend and causes it to drop messages ("Cache buffers lost messages per
# topic" warning) across every recorded topic at once. Compressed frames
# cut that by ~10-20x.
set -euo pipefail

OUTPUT="${1:-rosbag_$(date +%Y%m%d_%H%M%S)}"

ros2 bag record \
  -o "$OUTPUT" \
  --max-cache-size 536870912 \
  /cam0/image_raw/compressed \
  /cam1/image_raw/compressed \
  /feedback/steering \
  /feedback/velocity \
  /hwt101ct_yaw_publisher \
  /rosout
