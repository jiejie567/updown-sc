#!/bin/zsh
# Run all seven methods on the M2DGR hall 2 m evaluation tree produced by
# build_eval.py. Assumes the UpDown-SC fork is built (scan_context_rebuild,
# scan_context_cross_sequence_evaluator) and the baseline environments from
# baselines/python exist.
set -e
source /opt/ros/jazzy/setup.zsh
cd ${UPDOWN_SC_ROOT}/OneDrive/icra2027/slam
source install/setup.zsh
ROOT=${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/m2dgr_hall_eval_2m
mkdir -p $ROOT/results $ROOT/scd

echo "=== UpDown SCD rebuild (gravity, measured origin heights) ==="
ros2 run fast_lio scan_context_rebuild $ROOT/seq1/session $ROOT/scd/map.scd \
  gravity ${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/m2dgr_hall_2m/hall_04/session/runtime_params.yaml 0.790
ros2 run fast_lio scan_context_rebuild $ROOT/query_session $ROOT/scd/query.scd \
  gravity ${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/m2dgr_hall_2m/hall_02/session/runtime_params.yaml 0.799

echo "=== UpDown retrieval (final weights 0.3/0.7) ==="
ros2 run fast_lio scan_context_cross_sequence_evaluator \
  $ROOT/scd/map.scd $ROOT/scd/query.scd $ROOT/results/updown_candidates.csv \
  100 0.3 0.7 2 0.5 0.1

echo "=== SC / SC++ / SOLiD / M2DP (+G) ==="
cd ${UPDOWN_SC_ROOT}/OneDrive/icra2027/slam/experiments/common
python3 run_gravity_frontend_transfer.py \
  --map-dir $ROOT/seq1/session/key_point_frame \
  --map-poses $ROOT/seq1/session/optimized_poses_tum.txt \
  --map-gravity $ROOT/seq1/session/scan_context_gravity.csv \
  --query-dir $ROOT/derived/queries_seq2 \
  --query-gravity $ROOT/derived/queries_seq2/gravity.csv \
  --cache-dir $ROOT/cache_g --output-dir $ROOT/results/gravity \
  --correct-radius 2.0 --top-k 33   # top-k <= map size (33 keyframes)

echo "=== RING++ +G (CPU env with scikit-image) ==="
${UPDOWN_SC_ROOT}/icra2027_runtime/envs/ringpp-cpu/bin/python run_ringpp_gravity_transfer.py \
  --map-dir $ROOT/seq1/session/key_point_frame \
  --map-poses $ROOT/seq1/session/optimized_poses_tum.txt \
  --map-gravity $ROOT/seq1/session/scan_context_gravity.csv \
  --query-dir $ROOT/derived/queries_seq2 \
  --query-gravity $ROOT/derived/queries_seq2/gravity.csv \
  --output-dir $ROOT/results/ringpp --correct-radius 2.0 --workers 6 --top-k 33

echo "=== LiDAR Iris +G (query gravity.csv must be 5-column: query_id,stamp,up_x,up_y,up_z) ==="
IRIS_MAP_DIR=$ROOT/seq1/session/key_point_frame \
IRIS_MAP_POSES=$ROOT/seq1/session/optimized_poses_tum.txt \
IRIS_MAP_GRAVITY=$ROOT/seq1/session/scan_context_gravity.csv \
IRIS_QUERY_DIR=$ROOT/derived/queries_seq2 \
IRIS_OUTPUT=$ROOT/results/lidar_iris_per_query.csv \
IRIS_CORRECT_RADIUS=2.0 IRIS_GRAVITY_CANONICALIZE=1 \
${UPDOWN_SC_ROOT}/icra2027_runtime/baselines/LiDAR-Iris/build/benchmark | tail -1
echo M2DGR_EVAL_DONE
