#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 RAW_BAG CONFIG_YAML QOS_YAML OUTPUT_ROOT" >&2
  exit 2
fi

RAW_BAG=$(realpath "$1")
CONFIG=$(realpath "$2")
QOS=$(realpath "$3")
OUTPUT_ROOT=$(realpath -m "$4")
SLAM=${UPDOWN_SC_ROOT}/OneDrive/icra2027/slam
SESSION="$OUTPUT_ROOT/query/session"
SCD="$OUTPUT_ROOT/query/scans.scd"
LOG_DIR="$OUTPUT_ROOT/logs/query_capture"

if [[ -e "$SESSION" || -e "$SCD" ]]; then
  echo "Refusing to overwrite an existing 2 m query session: $OUTPUT_ROOT" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$(dirname "$SESSION")"
source /opt/ros/jazzy/setup.bash
source "$SLAM/install/setup.bash"
set -u

NODE_PID=
stop_node() {
  [[ -n "${NODE_PID:-}" ]] || return 0
  kill -0 "$NODE_PID" 2>/dev/null || return 0
  kill -INT "$NODE_PID" 2>/dev/null || true
  for _ in {1..120}; do
    if ! kill -0 "$NODE_PID" 2>/dev/null; then
      wait "$NODE_PID" 2>/dev/null || true
      return 0
    fi
    sleep 0.25
  done
  kill -TERM "$NODE_PID" 2>/dev/null || true
  wait "$NODE_PID" 2>/dev/null || true
}
trap stop_node EXIT INT TERM

cd "$OUTPUT_ROOT"
"$SLAM/install/fast_lio/lib/fast_lio/fastlio_mapping" --ros-args \
  --params-file "$CONFIG" \
  -p use_sim_time:=true \
  -p runtime.profile:=mapping \
  -p prior_map.scan_context.enable:=true \
  -p prior_map.scan_context.database_path:="$SCD" \
  -p prior_map.scan_context.keyframe_meter_gap:=2.0 \
  -p prior_map.scan_context.keyframe_yaw_gap_deg:=360.0 \
  -p manual_loop_export.enable:=true \
  -p manual_loop_export.session_dir:="$SESSION" \
  -p manual_loop_export.overwrite:=true \
  -p publish.scan_bodyframe_pub_en:=false \
  -p publish.scan_publish_en:=false \
  -p publish.effect_map_en:=false \
  -p pcd_save.pcd_save_en:=false \
  -p debug.save_registered_pcd_en:=false \
  -p common.lidar_qos_reliability:=reliable \
  -p common.imu_qos_reliability:=reliable \
  -p common.lidar_subscribe_qos_depth:=500 \
  -p common.imu_subscribe_qos_depth:=4000 \
  >"$LOG_DIR/fast_lio.log" 2>&1 &
NODE_PID=$!

sleep 5
ros2 bag play "$RAW_BAG" \
  --topics /driver/lidar/point_cloud/Data /driver/lidar/lidar_front/imu/Data \
  --qos-profile-overrides-path "$QOS" \
  --read-ahead-queue-size 10000 \
  --rate 3.0 --clock 20 \
  >"$LOG_DIR/play.log" 2>&1

python3 \
  ${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/two_bag_pilot_20260715/scripts/wait_for_sparse_topic_idle.py \
  --topic /Odometry \
  --message-type odometry \
  --idle-s 5 \
  --max-wait-s 300 \
  >"$LOG_DIR/drain.log" 2>&1

stop_node
trap - EXIT INT TERM

python3 "$SLAM/experiments/common/select_spatial_keyframes.py" \
  --input-tum "$SESSION/optimized_poses_tum.txt" \
  --output-dir "$OUTPUT_ROOT/protocol/query" \
  --spacing-m 2.0 \
  >"$LOG_DIR/protocol_check.log"

test -s "$SCD"
test -s "$SESSION/optimized_poses_tum.txt"
test -s "$SESSION/scan_context_gravity.csv"
echo "2 m query session ready: $OUTPUT_ROOT"
