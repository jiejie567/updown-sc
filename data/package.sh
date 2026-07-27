#!/bin/bash
# Assemble the evaluation data package with checksums.
# Run on the experiment machine; outputs tarballs under $OUT.
set -e
OUT=${1:-${UPDOWN_SC_ROOT}/icra2027_runtime/release/data_package}
RT=${UPDOWN_SC_ROOT}/icra2027_runtime
mkdir -p $OUT
pack () {  # pack <name> <src...>
  local name=$1; shift
  tar -czf $OUT/$name.tar.gz "$@"
  (cd $OUT && sha256sum $name.tar.gz >> SHA256SUMS)
  echo "packed $name"
}
pack ih_2m -C $RT/experiments private_two_bag_2m/map private_two_bag_2m/derived private_two_bag_2m/results
pack ch_2m -C $RT/experiments rtk_slam_construction_hall_2m/seq1 rtk_slam_construction_hall_2m/derived rtk_slam_construction_hall_2m/results
pack nc_2m -C $RT/experiments newer_college_quad_easy_2m
pack cross_device_2m -C $RT/experiments indoor_cross_device_2m
pack metrics -C $RT/experiments metrics_augment_20260725 updown_weight_ablation_real_20260721/selected updown_weight_ablation_real_20260721/selected_summary.csv
pack production_map -C $RT manual_loop/gravity
echo "done -> $OUT"
