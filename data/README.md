# Evaluation data package

Hosted separately (Zenodo record; DOI added at release). Contents and the
paths the scripts expect:

| Group | Contents | Approx. size | Expected path (as-run) |
|---|---|---|---|
| `ih_2m/` | IH map session (376 keyframes: PCDs, TUM poses, gravity CSV, SCD), 322 query bins + gravity + metadata, per-query CSVs | ~2 GB | `experiments/private_two_bag_2m/` |
| `ch_2m/` | Construction Hall seq1/seq2 sessions, query bins, per-query CSVs | ~2 GB | `experiments/rtk_slam_construction_hall_2m/` |
| `nc_2m/` | Newer College derived split + per-query CSVs | ~0.5 GB | `experiments/newer_college_quad_easy_2m/` |
| `cross_device_2m/` | H1/H2/V1 sessions, queries, per-query CSVs | ~1 GB | `experiments/indoor_cross_device_2m/` |
| `metrics/` | `metrics_augment_20260725/` CSVs (CI, PR/F1, McNemar, yaw) + weight-ablation `selected/` per-query and candidates | ~50 MB | `experiments/metrics_augment_20260725/` |
| `m2dgr_2m/` | M2DGR hall_04 map session (33 keyframes) + hall_02 query session/bins (28 eligible queries), per-query CSVs for all 7 methods, SCDs, build report | ~14 MB | `experiments/m2dgr_hall_eval_2m/` |
| `bags/ih/` | In-house rosbags, LiDAR+IMU topics only: `mapping_2_floor` (27 GB, IH map session), `loc_2_floor` (22 GB, IH localization/query session; both already fused-cloud + IMU only: `/driver/lidar/point_cloud/Data`, `/driver/lidar/lidar_front/imu/Data`), `indoor_handle1/2_ros2`, `indoor_vehicle1_ros2` (H1/H2/V1) | ~55 GB | n/a (inputs for full re-runs) |
| `production_map/` | 2,574-keyframe production prior map + SCD used by the latency benchmark | ~1 GB | `manual_loop/gravity/` |

Public datasets (RTK-SLAM, Newer College, M2DGR raw bags) are not
redistributed; download from their original sources and use
`experiments/*/prepare_*.py` to regenerate the derived splits.

`filter_ih_bags.sh` strips camera topics from the raw in-house bags before
packaging. `package.sh` assembles the groups with SHA-256 checksums.
