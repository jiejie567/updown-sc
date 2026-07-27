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
