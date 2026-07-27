#!/bin/bash
# Strip camera/event topics from in-house ROS 2 bags before release
# (LiDAR + IMU only). Requires a sourced ROS 2 environment.
set -e
OUT=${2:?usage: filter_ih_bags.sh <bag_dir> <out_dir>}
ros2 bag convert -i "$1" -o <(cat <<YAML
output_bags:
  - uri: $OUT
    include_topics: [/livox/lidar, /livox/imu]
YAML
)
