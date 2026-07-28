#!/bin/bash
# Strip non-essential topics from in-house ROS 2 bags before release
# (LiDAR + IMU only). Requires a sourced ROS 2 environment.
#
# The IH production bags (mapping_2_floor, loc_2_floor) already contain only
# the fused cloud + IMU (/driver/lidar/point_cloud/Data,
# /driver/lidar/lidar_front/imu/Data) and can be released as-is; this script
# is for bags that carry extra (e.g. camera) topics, such as the raw
# cross-device recordings.
set -e
OUT=${2:?usage: filter_ih_bags.sh <bag_dir> <out_dir> [topics...]}
BAG=$1; shift 2
TOPICS=${*:-/livox/lidar /livox/imu}
ros2 bag convert -i "$BAG" -o <(cat <<YAML
output_bags:
  - uri: $OUT
    include_topics: [$(echo $TOPICS | sed 's/ /, /g')]
YAML
)
