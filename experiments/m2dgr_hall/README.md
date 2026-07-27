# M2DGR hall cross-session indoor benchmark (in progress)

Public indoor-only benchmark for UpDown-SC: the M2DGR `hall` sequences are
five independent traversals of the same building hall (SJTU), with per-frame
Leica MS60 3-DOF position ground truth (mm-level), a Velodyne VLP-32C, and a
Handsfree A9 IMU (`/handsfree/imu`, 150 Hz) in ROS1 bags (MIT license).

## Split

- Map/database session: `hall_01` (2021-08-01, 351 s random walk, 29.1 GB).
- Query sessions: `hall_02` (128 s) and `hall_04` (181 s), recorded on
  different days (08-08 / 08-15) — genuine cross-session queries.
- Keyframe rule: experiment-only translation-only 2 m sampling, as in the
  IH/CH/NC experiments. Correctness radius 2 m (Leica truth; see below).

## Pipeline (mirrors the Construction Hall recipe)

1. `segmented_download.py` (in the dataset dir) — chunked parallel SharePoint
   download; resumable at 64 MB chunk granularity.
2. `convert_bags.py` — ROS1 -> ROS2 via `rosbags`, keeping
   `/velodyne_points` + `/handsfree/imu` only.
3. FAST-LIO per sequence with
   `fast_lio/config/m2dgr_vlp32.yaml` (VELO16 handler, scan_line 32,
   timestamp_unit seconds, Handsfree extrinsic from
   `calibration_results.txt`: identity R, LiDAR-in-IMU T
   [0.27255, -0.00053, 0.17954] — validate sign via mapping quality).
4. Session export (key_point_frame + optimized_poses_tum.txt +
   scan_context_gravity.csv), as for CH seq1/seq2.
5. Truth: align each FAST-LIO trajectory to its Leica GT (position Umeyama);
   express query truth in the hall_01 map frame via cross-session
   registration (CH precedent: "Seq. 2 truth expressed in Seq. 1 frame").
   GT frames may differ between days (Leica re-setup) — do NOT assume a
   common frame.
6. Descriptor-origin height audit (ground-plane distance after gravity
   rotation) to set `origin_height_from_ground` (CH audit procedure).
7. UpDown candidates via the C++ offline relocalization exporter; baselines
   via `rescore_native_baselines.py`, `run_gravity_frontend_transfer.py`,
   the LiDAR-Iris benchmark binary, and the RING++ runner — all take
   CH-layout experiment roots.

## Status

- [x] Dataset survey (M2DGR selected; Hilti'21 Basement as backup)
- [x] GT downloaded: hall_01/02/04 (TUM, position-only, quaternions zero)
- [x] hall_02 downloaded (15.0 GB), converted (LiDAR+IMU only), mapped:
      201-keyframe session exported (key_point_frame / TUM / gravity / scd).
      Leica alignment with jointly estimated prism lever arm
      ([0.528, -0.027, -0.002] m body frame) gives ATE 8.7 cm RMSE /
      5.6 cm median / 31 cm max -- pipeline validated end to end.
      Note: session export requires prior_map.scan_context.enable (do NOT
      pass the NC-style disable flag) and shutdown SIGINT must target the
      fastlio_mapping binary (see scratchpad run_m2dgr_mapping.sh pattern).
- [ ] hall_04 downloading; then map-session run, 2 m keyframe split,
      cross-session truth registration, height audit, full evaluation
