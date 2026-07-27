# UpDown-SC: Gravity-Canonicalized Dual-Envelope Scan Context

Code, baselines, evaluation protocols, and data release for the ICRA 2027
submission "UpDown-SC: A Gravity-Canonicalized Dual-Envelope Scan Context for
Indoor LiDAR Place Recognition" (anonymous during review).

Every number in the paper traces to a per-query CSV in the data package; see
`docs/PROVENANCE.md`.

## Layout

| Directory | Contents |
|---|---|
| `updown_sc/fast_lio/` | Modified FAST-LIO2 (ROS 2) containing the production UpDown-SC implementation: dual-envelope SCD construction, masked non-uniform matching, keyframe database, session export, and the offline relocalization/candidate exporter. |
| `baselines/python/` | Audited formula-equivalent CPU baselines (SC, SC++ (PC), SOLiD, M2DP, RING++ CPU port) and query/gravity export tools. |
| `baselines/adapters/` | Our adapters for official C++ cores (LiDAR-Iris benchmark harness; BTC/STD run via `experiments/` scripts). Official cores are fetched from their upstream repositories; we do not redistribute them. |
| `experiments/` | Per-dataset protocol scripts and READMEs (in-house two-session, RTK-SLAM Construction Hall, Newer College, indoor cross-device, M2DGR hall). |
| `figures/` | Deterministic figure generators with their source-data files and integrity metadata. |
| `docs/` | Provenance (paper number -> CSV mapping) and protocol manifests. |
| `data/` | Download manifest and packaging scripts for the evaluation data (hosted separately; see `data/README.md`). |

## Bags and configs

FAST-LIO runs with a base parameter file plus one dataset override:

| Dataset / rosbag | LiDAR | Config |
|---|---|---|
| In-house two-session, indoor deployment (H1/H2/V1), Construction Hall | Livox MID-360 | `config/mid360.yaml` (base, production defaults) |
| Newer College quad-easy | Ouster OS0-128 | `config/mid360.yaml` + `config/newer_college_os0.yaml` |
| M2DGR hall | Velodyne VLP-32C | `config/mid360.yaml` + `config/m2dgr_vlp32.yaml` |

Example (mapping a session and exporting the keyframe database):

```bash
ros2 run fast_lio fastlio_mapping --ros-args \
  --params-file <share>/config/mid360.yaml \
  --params-file fast_lio/config/m2dgr_vlp32.yaml \
  -p runtime.profile:=mapping \
  -p prior_map.scan_context.database_path:=<out>/scans.scd \
  -p manual_loop_export.enable:=true \
  -p manual_loop_export.session_dir:=<out>/session
ros2 bag play <bag> --topics <lidar_topic> <imu_topic>
# SIGINT the fastlio_mapping binary to trigger the session export.
```

Session export requires `prior_map.scan_context.enable` (default true).

## Reproducing the paper tables

1. Download the data package (`data/README.md`): query bins, gravity CSVs,
   map sessions, SCD databases, and all recorded per-query CSVs.
2. Recompute statistics without re-running retrieval:
   `experiments/common/compute_retrieval_stats.py` regenerates the Wilson
   intervals and F1max/AUPR behind Table II's uncertainty statements and
   Fig. 6; `compute_yaw_error.py` regenerates the yaw-seed numbers.
3. Re-run retrieval from inputs: `experiments/common/rescore_native_baselines.py`
   and `run_gravity_frontend_transfer.py` (Python baselines),
   `baselines/adapters/lidar_iris/benchmark.cpp` (official Iris core), and the
   UpDown-SC C++ exporter for candidate lists.

Some experiment scripts retain the absolute paths of the original runs for
archival fidelity; the path map is documented in `data/README.md`.

## Licenses

`updown_sc/fast_lio` is a derivative of FAST-LIO2 and remains under GPL-2.0
(`LICENSES/`). Our Python baselines, adapters, experiment, and figure code are
released under the MIT license. Official baseline cores (LiDAR-Iris, BTC,
STD) keep their upstream licenses and are fetched, not redistributed.

## Citation

Citation entry will be added after the review period.
