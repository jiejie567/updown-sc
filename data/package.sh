#!/usr/bin/env bash
# Assemble the evaluation data package with checksums.
# Run on the experiment machine; outputs tarballs under $OUT.
set -euo pipefail

: "${UPDOWN_SC_ROOT:?Set UPDOWN_SC_ROOT to the data workspace root}"
OUT=${1:-${UPDOWN_SC_ROOT}/icra2027_runtime/release/data_package}
RT=${UPDOWN_SC_ROOT}/icra2027_runtime
mkdir -p "$OUT"
CHECKSUMS_TMP=$(mktemp "$OUT/.SHA256SUMS.XXXXXX")
trap 'rm -f "$CHECKSUMS_TMP"' EXIT

audit_archive () {
  local archive=$1
  local leaked_home
  leaked_home=$(
    tar -xOzf "$archive" 2>/dev/null |
      LC_ALL=C grep -aEom1 "(/home/[^/[:space:]\"']+|/Users/[^/[:space:]\"']+|[A-Za-z]:\\\\Users\\\\[^\\\\[:space:]\"']+)" || true
  )
  if [[ -n "$leaked_home" ]]; then
    echo "refusing archive with a machine-specific home path: $leaked_home" >&2
    return 1
  fi
}

pack () {  # pack <name> <src...>
  local name=$1
  local archive
  shift
  archive=$(mktemp "$OUT/.${name}.XXXXXX")
  tar -czf "$archive" "$@"
  if ! audit_archive "$archive"; then
    rm -f "$archive"
    return 1
  fi
  mv "$archive" "$OUT/$name.tar.gz"
  chmod 664 "$OUT/$name.tar.gz"
  (cd "$OUT" && sha256sum "$name.tar.gz" >> "$CHECKSUMS_TMP")
  echo "packed $name"
}
pack ih_2m -C "$RT/experiments" private_two_bag_2m/map private_two_bag_2m/query private_two_bag_2m/derived private_two_bag_2m/results
pack ch_2m -C "$RT/experiments" rtk_slam_construction_hall_2m/seq1 rtk_slam_construction_hall_2m/seq2 rtk_slam_construction_hall_2m/derived rtk_slam_construction_hall_2m/results
pack nc_2m -C "$RT/experiments" newer_college_quad_easy_2m
pack cross_device_2m -C "$RT/experiments" indoor_cross_device_2m
pack m2dgr_2m -h -C "$RT/experiments" m2dgr_hall_eval_2m/seq1 m2dgr_hall_eval_2m/query_session m2dgr_hall_eval_2m/derived m2dgr_hall_eval_2m/results m2dgr_hall_eval_2m/scd m2dgr_hall_eval_2m/split_hall_02 m2dgr_hall_eval_2m/split_hall_04 m2dgr_hall_eval_2m/build_report.json
pack metrics -C "$RT/experiments" metrics_augment_20260725 updown_weight_ablation_real_20260721/selected updown_weight_ablation_real_20260721/selected_summary.csv
pack learned_ot -C "$RT/experiments" learned_baseline_20260831
pack learned_minkloc3dv2 -C "$RT/experiments" learned_minkloc3dv2_20260831
pack production_map -C "$RT" manual_loop/gravity
mv "$CHECKSUMS_TMP" "$OUT/SHA256SUMS"
trap - EXIT
echo "done -> $OUT"
