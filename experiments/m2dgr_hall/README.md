# M2DGR hall cross-session indoor benchmark

Public indoor benchmark with surveyed per-frame truth: the M2DGR `hall`
sequences (SJTU, MIT license) are independent traversals of the same building
hall with per-frame Leica MS60 3-DOF position ground truth, a Velodyne
VLP-32C, and a Handsfree A9 IMU (`/handsfree/imu`, 150 Hz) in ROS1 bags.
This is the paper's spinning-32-beam transfer check.

## Split (as used in the paper)

- Map/database: `hall_04` (2021-08-15) — 33 experimental keyframes.
- Queries: `hall_02` (2021-08-08, different day) — 29 keyframes, 28 with a
  map keyframe inside the 2 m positive radius (n_q = 28).
- Keyframe rule: experiment-only translation-only 2 m sampling; correctness
  radius 2 m against Leica truth.

## Pipeline

1. `segmented_download.py` (dataset dir) — chunked, resumable SharePoint
   download (SJTU throttles to ~0.2–1 MB/s total).
2. `convert_bags.py` — ROS1 → ROS2 via `rosbags`, keeping
   `/velodyne_points` + `/handsfree/imu` only.
3. `run_mapping.sh <seq>` — FAST-LIO fork with
   `fast_lio/config/m2dgr_vlp32.yaml` (VELO16 handler, scan_line 32,
   timestamp_unit seconds, LiDAR-in-IMU T [0.27255, -0.00053, 0.17954]) and
   reliable QoS on both replay ends; exports the manual-loop session.
4. `build_eval.py` — builds the CH-layout evaluation tree
   (`m2dgr_hall_eval_2m/`): 2 m splits, GT alignment, cross-day
   registration, query bin/gravity/metadata export, height audit.
5. `run_eval.sh` — all seven methods on that tree.

## Ground-truth alignment (key findings)

- Per-session GT **clock offsets** exist and must be estimated: +1.65 s
  (hall_04), +0.10 s (hall_02), found by an ATE-minimizing sweep jointly
  solving a prism **lever arm** (~0.50 m along body +x). After correction the
  FAST-LIO trajectories align to Leica at 4.4/4.9 cm RMSE.
- The two days' Leica **station frames differ** (GT2 → GT4: yaw −41.79°,
  t = (26.88, 1.40)). Registered map-to-map by a global yaw-grid 2D ICP
  (2° steps over 360°), 10.3 cm median residual; a plain local ICP gets stuck.
- Measured descriptor-origin heights: 0.790 m (hall_04) / 0.799 m (hall_02).

## Results (+G protocol, n_q = 28, Recall@1/@5 %)

| Method | R@1 | R@5 |
|---|---|---|
| SC + G | 96.4 | 96.4 |
| **UpDown-SC (ours)** | 92.9 | 100.0 |
| SOLiD + G | 92.9 | 100.0 |
| LiDAR Iris + G | 85.7 | 92.9 |
| RING++ + G | 85.7 | 100.0 |
| SC++ (PC) + G | 75.0 | 100.0 |
| M2DP + G | 46.4 | 85.7 |

The single-hall database saturates most methods (one query = 3.6 points);
the paper reports this as a beam-pattern transfer check with surveyed truth,
not a discriminative ranking. BTC's causal ten-scan windows were not
constructed for M2DGR.

Per-query CSVs: `m2dgr_hall_eval_2m/results/` (`updown_candidates.csv`,
`gravity/per_query.csv`, `ringpp/results/ringpp_per_query.csv`,
`lidar_iris_per_query.csv`), packaged in the `m2dgr_2m` data group.

## Traps

- Do **not** pass `prior_map.scan_context.enable:=false` — it disables the
  manual-loop session export.
- The LiDAR-Iris benchmark requires the query `gravity.csv` to use the
  5-column schema `query_id,stamp,up_x,up_y,up_z` (it SIGSEGVs otherwise).
- `--top-k` must not exceed the map size (33) for the Python runners.
- Remaining "IMU gap" warnings during replay are in-bag jitter; LiDAR frame
  gaps are 0 with reliable QoS on both ends.
