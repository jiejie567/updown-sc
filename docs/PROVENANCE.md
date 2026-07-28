# Metrics augmentation 2026-07-25 — provenance

Paper edits in `latex/paper_en.tex` (commit of this date) added Wilson CIs,
McNemar tests, F1max/AUPR (Table III), and yaw-seed errors. Every new number
traces to a CSV in this directory or the recorded runs listed below.

## Rescored baseline runs (this directory)

`ih_native/`, `ih_gravity/`, `ch_native/`, `ch_gravity/` were regenerated with
`slam/experiments/common/rescore_native_baselines.py` (native, recorded
protocol: official 20-ring SC, existing `construction_*` caches reused) and
`slam/experiments/common/run_gravity_frontend_transfer.py` (+G, existing
`gravity_transfer_*` caches reused), plus the LiDAR-Iris `benchmark.cpp`
rebuilt with a `top1_score` column. Recall@1/5 of every rerun matches the
recorded per-query CSVs exactly (24/24 method-condition pairs); the only
addition is the `top1_score` column. Recorded runs remain untouched in their
original directories.

## Paper number → source

| Paper statement | Source |
|---|---|
| Wilson CI half-widths (±5 IH, ±8 CH, ±12 deployments) | `ci_summary.csv` |
| McNemar p=0.002 (vs SC+G, IH), p=0.053/0.15 (vs RING++) | `mcnemar_summary.csv` |
| Table III F1max/AUPR (IH+G, CH+G) | `pr_f1_summary.csv` |
| Native AUPR sentence (IH 0.477 vs 0.370; CH 0.374 vs 0.354) | `pr_f1_summary.csv` |
| Yaw seed error 1.9°/6.5° (n=222) | `yaw_error_summary.csv` |
| STD exclusion sentence (2.2/7.7% IH; 0.7/0.7% CH) | `.../private_two_bag_2m/results/std_exact/std_submap_verified_per_query.csv`, `.../rtk_slam_construction_hall_2m/results/std_exact|std_native/...` (recall_at_1/5 column means) |
| SOLiD audit sentence (3.8e-6 max abs err) | `.../baseline_20260715/README.md` |
| Latency provenance (92 queries, 2,574-keyframe production DB) | `.../baseline_20260715/README.md` |
| Descriptor memory ≈7.9 KB | 2×16×60×4 B + 2×16×60 bit masks |

## Scripts

- `slam/experiments/common/compute_retrieval_stats.py` — CIs + PR/F1 (also
  regenerates `ci_summary.csv`, `pr_f1_summary.csv`).
- `slam/experiments/common/compute_yaw_error.py` — yaw seed error
  (`yaw_error_summary.csv`); reference = full-quaternion localization odometry,
  prediction = candidate keyframe yaw − yaw_shift_rad, correct-top-1 queries.
- McNemar: inline in `compute_retrieval_stats.py`'s module functions (see
  `mcnemar_summary.csv` header for the exact pairs; exact two-sided binomial).

## Notes / traps

- The authoritative UpDown-SC row of paper Table II is
  `updown_weight_ablation_real_20260721/selected_summary.csv`; the older
  `*_2m/results/updown_*` CSVs are pre-weight-selection runs and do NOT match
  the paper.
- Native SC in the recorded protocol uses the official 20-ring descriptor
  (cache `construction_sc_r20_*`), while +G SC uses 16 rings
  (`gravity_transfer_sc_r16_*`, see gravity_transfer README). The current
  `run_single_scan_baselines.py` was later edited to 16 rings and no longer
  reproduces the recorded native CSVs; `rescore_native_baselines.py` pins the
  recorded protocol.

# M2DGR hall benchmark 2026-07-27 — provenance

Paper Table II's M2DGR column (+G, n_q = 28) traces to
`experiments/m2dgr_hall_eval_2m/results/` (packaged as the `m2dgr_2m` data
group):

| Paper statement | Source |
|---|---|
| SC 96.4/96.4, SC++ 75.0/100.0, SOLiD 92.9/100.0, M2DP 46.4/85.7 | `results/gravity/per_query.csv` (recall_at_1/5 means per algorithm) |
| LiDAR Iris 85.7/92.9 | `results/lidar_iris_per_query.csv` |
| RING++ 85.7/100.0 | `results/ringpp/results/ringpp_per_query.csv` |
| UpDown-SC 92.9/100.0 | `results/updown_candidates.csv` (rank ≤ 1/5 candidate within 2 m of `gravity/per_query.csv` truth) |
| Alignment quality (4.4/4.9 cm RMSE, 10.3 cm cross-day median) | `build_report.json` + `experiments/m2dgr_hall/build_eval.py` log |
| Origin heights 0.79/0.80 m | `build_report.json` height audit |

BTC is `--` in the paper: its causal ten-scan accumulated windows were not
constructed for M2DGR. All seven recall pairs were independently recomputed
from the per-query CSVs before the paper was frozen (2026-07-27) and match.

# Release restructure 2026-07-28 — standalone UpDown-SC

The FAST-LIO2 fork (GPL-2.0, our LiDAR-inertial front end) was removed from
the release; `updown_sc/` now contains the standalone, ROS-free UpDown-SC
implementation (Eigen + PCL + yaml-cpp) with the same sources:
`scan_context.{hpp,cpp}` and the `scan_context_rebuild` /
`scan_context_cross_sequence_evaluator` / `scan_context_convert` /
`scan_context_subset` tools, byte-for-byte identical to the fork's files.

Regression check (2026-07-28): the standalone `scan_context_rebuild` on the
released M2DGR sessions produced **byte-identical** `map.scd`/`query.scd`
(cmp vs the recorded `m2dgr_hall_eval_2m/scd/`), and the standalone
evaluator's `updown_candidates.csv` reproduces 92.9/100.0 Recall@1/5
(n = 28), matching the paper. Raw rosbags are no longer part of the data
package; reproduction starts from the released deskewed keyframe sessions.
The whole release is now MIT-licensed (`LICENSE`).
