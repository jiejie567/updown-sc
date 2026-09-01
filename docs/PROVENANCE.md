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

# OverlapTransformer learned baseline 2026-08-31

Paper Table II's learned OT row was evaluated with the official
OverlapTransformer KITTI checkpoint, without fine-tuning. The adapter is
`experiments/common/run_overlap_transformer.py`; the per-query evidence,
summaries, descriptor caches, and manifests are packaged as `learned_ot`.

- Official repository commit: `9809ac2c7bce9ebf446392f79bfe514927ab4c63`
- Checkpoint SHA-256: `5b1da14c09e990257a41b05460cc390c7c62ddb65b0bc8900d421a78a2864bae`
- Input: the common deskewed single frame and 0.3--30 m horizontal crop
- Projection: the official 64 x 900 spherical range image, +3/-25 degree
  vertical field of view, 80 m projection cap
- Retrieval: squared L2 distance between the official 256-D descriptors
- `+G`: the same gravity canonicalization supplied to the compatible
  single-frame baselines; no other parameter changes

| Table II condition | R@1/R@5 (%) | Source |
|---|---:|---|
| IH native | 15.6/41.6 | `ih_native/per_query.csv` |
| IH +G | 12.5/31.9 | `ih_gravity/per_query.csv` |
| CH native | 12.8/41.2 | `ch_native/per_query.csv` |
| CH +G | 27.0/49.3 | `ch_gravity/per_query.csv` |
| H1 to H2 +G | 8.6/20.7 | `indoor_handle2_gravity/per_query.csv` |
| H1 to V1 +G | 12.3/22.8 | `indoor_vehicle1_gravity/per_query.csv` |
| M2DGR +G | 42.9/78.6 | `m2dgr_gravity/per_query.csv` |
| Newer College +G | 68.3/95.0 | `nc_gravity/per_query.csv` |

The Table II latency, 44.6 ms median, comes from
`production_latency_native/per_query.csv`: 92 query clouds were preloaded and
ranked against 2,574 precomputed database descriptors on the same Intel Core
Ultra 5 225H. Map construction and file I/O are excluded from query timing.

# MinkLoc3Dv2 learned baseline 2026-08-31

Paper Table II's MinkLoc-v2 row uses the official MinkLoc3Dv2 Oxford-only
baseline checkpoint, without fine-tuning. The adapter is
`experiments/common/run_minkloc3dv2.py`; per-query evidence, summaries,
descriptor caches, and manifests are packaged as `learned_minkloc3dv2`.

- Official repository commit: `2413a5ddc2fca941be54d300f8bafc45043e9d62`
- Checkpoint SHA-256: `559242b5f97be756c391d1d1d4c622ea6b1ea21d809c559d12584167134d4e79`
- Input: the common deskewed single frame and 0.3--30 m horizontal crop;
  optional `+G` uses the same gravity canonicalization as the other compatible
  single-frame baselines
- Deterministic adapter preprocessing: 0.10 m voxel prefilter and an
  order-independent 4,096-point cap, followed by the released PointNetVLAD
  centroid/mean-radius normalization; no ground removal is applied
- Retrieval: squared L2 distance (rank-equivalent to L2) between the official
  256-D sparse-voxel descriptors
- Runtime: Python 3.8.20, PyTorch 1.10.1 CPU, MinkowskiEngine 0.5.4 CPU. The
  unmodified model code and checkpoint are used; building MinkowskiEngine
  0.5.4 with GCC 13 required adding the standard `<cstdint>` header to its
  `src/quantization.cpp` and `pybind/minkowski.cpp` compatibility units.

| Table II condition | R@1/R@5 (%) | Source |
|---|---:|---|
| IH native | 55.0/78.8 | `ih_native/per_query.csv` |
| IH +G | 54.7/80.9 | `ih_gravity/per_query.csv` |
| CH native | 39.9/70.3 | `ch_native/per_query.csv` |
| CH +G | 52.0/83.1 | `ch_gravity/per_query.csv` |
| H1 to H2 +G | 15.5/56.9 | `indoor_handle2_gravity/per_query.csv` |
| H1 to V1 +G | 19.3/52.6 | `indoor_vehicle1_gravity/per_query.csv` |
| M2DGR +G | 50.0/96.4 | `m2dgr_gravity/per_query.csv` |
| Newer College +G | 83.3/93.3 | `nc_gravity/per_query.csv` |

The Table II latency, 394.4 ms median, comes from
`production_latency_native/per_query.csv`: 92 preloaded query clouds were
ranked against 2,574 precomputed database descriptors on the same Intel Core
Ultra 5 225H. Map construction and file I/O are excluded from query timing.
