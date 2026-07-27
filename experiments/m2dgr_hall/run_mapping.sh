#!/bin/zsh
# Map one converted M2DGR hall sequence with the UpDown-SC FAST-LIO fork and
# export the manual-loop session (keyframes + poses + gravity + SCD).
# Usage: ./run_mapping.sh hall_04
set -e
SEQ=$1
source /opt/ros/jazzy/setup.zsh
cd ${UPDOWN_SC_ROOT}/OneDrive/icra2027/slam
source install/setup.zsh
ROOT=${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/m2dgr_hall_2m/$SEQ
mkdir -p $ROOT
ros2 run fast_lio fastlio_mapping --ros-args \
  --params-file install/fast_lio/share/fast_lio/config/mid360.yaml \
  --params-file fast_lio/config/m2dgr_vlp32.yaml \
  -p runtime.profile:=mapping \
  -p common.lidar_qos_reliability:=reliable \
  -p common.imu_qos_reliability:=reliable \
  -p pcd_save.pcd_save_en:=false \
  -p prior_map.scan_context.database_path:=$ROOT/scans.scd \
  -p manual_loop_export.enable:=true \
  -p manual_loop_export.session_dir:=$ROOT/session \
  > $ROOT/mapping.log 2>&1 &
sleep 8
# Bag replay must be RELIABLE on both ends: a best-effort subscriber drops
# roughly half the frames during replay.
cat > /tmp/m2dgr_qos.yaml <<'YAML'
/velodyne_points:
  reliability: reliable
  history: keep_last
  depth: 200
/handsfree/imu:
  reliability: reliable
  history: keep_last
  depth: 4000
YAML
ros2 bag play ${UPDOWN_SC_ROOT}/icra2027_runtime/datasets/m2dgr/ros2/$SEQ \
  --topics /velodyne_points /handsfree/imu --rate 1.0 \
  --qos-profile-overrides-path /tmp/m2dgr_qos.yaml
sleep 4
pkill -INT -x fastlio_mapping
for i in $(seq 1 60); do
  pgrep -x fastlio_mapping > /dev/null || break
  sleep 2
done
echo MAPPING_RUN_DONE
ls $ROOT/session
ls $ROOT/session/key_point_frame | wc -l
