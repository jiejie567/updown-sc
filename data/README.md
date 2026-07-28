# Evaluation data package

Hosted separately (Zenodo record; DOI added at release). Total ~720 MB. Contents and the
paths the scripts expect:

| Group | Contents | Approx. size | Expected path (as-run) |
|---|---|---|---|
| `ih_2m/` | IH map session (376 keyframes: PCDs, TUM poses, gravity CSV, SCD) + query session (322 keyframes, same layout), query bins + gravity + metadata, per-query CSVs | ~74 MB | `experiments/private_two_bag_2m/` |
| `ch_2m/` | Construction Hall seq1 + seq2 sessions (incl. SCDs), query bins, per-query CSVs | ~43 MB | `experiments/rtk_slam_construction_hall_2m/` |
| `nc_2m/` | Newer College derived split + per-query CSVs | ~252 MB | `experiments/newer_college_quad_easy_2m/` |
| `cross_device_2m/` | H1/H2/V1 sessions, queries, per-query CSVs | ~87 MB | `experiments/indoor_cross_device_2m/` |
| `metrics/` | `metrics_augment_20260725/` CSVs (CI, PR/F1, McNemar, yaw) + weight-ablation `selected/` per-query and candidates | ~8 MB | `experiments/metrics_augment_20260725/` |
| `m2dgr_2m/` | M2DGR hall_04 map session (33 keyframes) + hall_02 query session/bins (28 eligible queries), per-query CSVs for all 7 methods, SCDs, build report | ~14 MB | `experiments/m2dgr_hall_eval_2m/` |
| `production_map/` | 2,574-keyframe production prior map + SCD used by the latency benchmark | ~245 MB | `manual_loop/gravity/` |

The release starts from deskewed, motion-compensated keyframe clouds: every
session directory ships keyframe PCDs, TUM poses, and per-keyframe gravity
directions, so all retrieval results regenerate without the LiDAR-inertial
front end (which is not released). Descriptor parameters are in
`updown_sc/config/descriptor_params.yaml`; the M2DGR sessions additionally
ship the exact `runtime_params.yaml` they were built with, and rebuilding
their SCDs from the released clouds was verified byte-identical. Raw rosbags are not distributed. Public datasets
(RTK-SLAM, Newer College, M2DGR) can be downloaded from their original
sources and re-derived with `experiments/*/prepare_*.py` / `build_eval.py`.

`package.sh` assembles the groups with SHA-256 checksums.
