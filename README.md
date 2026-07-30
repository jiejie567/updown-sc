# UpDown-SC: Gravity-Canonicalized Dual-Envelope Scan Context

Code, baselines, evaluation protocols, and data release for the paper
"UpDown-SC: A Gravity-Canonicalized Dual-Envelope Scan Context for Indoor
LiDAR Place Recognition" (under review).

Every number in the paper traces to a per-query CSV in the data package; see
`docs/PROVENANCE.md`.

## Layout

| Directory | Contents |
|---|---|
| `updown_sc/` | Standalone C++ implementation of UpDown-SC (no ROS dependency): dual-envelope SCD construction with gravity canonicalization and adaptive split (`scan_context_rebuild`), masked non-uniform retrieval with yaw hypotheses (`scan_context_cross_sequence_evaluator`), plus SCD conversion/subset tools. |
| `baselines/python/` | Audited formula-equivalent CPU baselines (SC, SC++ (PC), SOLiD, M2DP, RING++ CPU port) and query/gravity export tools. |
| `baselines/adapters/` | Our adapters for official C++ cores (LiDAR-Iris benchmark harness; BTC/STD run via `experiments/` scripts). Official cores are fetched from their upstream repositories; we do not redistribute them. |
| `experiments/` | Per-dataset protocol scripts and READMEs (in-house two-session, RTK-SLAM Construction Hall, Newer College, indoor cross-device, M2DGR hall). |
| `figures/` | Deterministic figure generators with their source-data files and integrity metadata. |
| `docs/` | Provenance (paper number -> CSV mapping) and protocol manifests. |
| `data/` | Download manifest and packaging scripts for the evaluation data (hosted separately; see `data/README.md`). |

## What is (and is not) released

The release starts from **deskewed, motion-compensated keyframe clouds**: the
data package ships every map session (keyframe PCDs, TUM poses, per-keyframe
gravity directions, descriptor parameters, SCD databases) and every query set
(deskewed single-frame bins + gravity + metadata), plus all recorded per-query
result CSVs. The LiDAR-inertial front end that produced these sessions (a
FAST-LIO2-based ROS 2 pipeline) is **not part of this release**; experiment
scripts that drove it are retained for protocol documentation only. All paper
retrieval results reproduce from the released clouds alone — regenerating the
descriptors from the shipped sessions was verified to give byte-identical SCD
databases and identical recall.

## Building UpDown-SC

Dependencies: CMake >= 3.16, a C++17 compiler, Eigen3, PCL (common + io),
yaml-cpp. No ROS required.

```bash
cd updown_sc && mkdir build && cd build
cmake .. && make -j
```

Tools:

```bash
# Rebuild an SCD database from a session directory
# (key_point_frame/*.pcd + optimized_poses_tum.txt + scan_context_gravity.csv
#  + runtime_params.yaml):
./scan_context_rebuild <session_dir> <out.scd> gravity <runtime_params.yaml> \
    <origin_height_m> [fixed_split_m]

# Cross-sequence retrieval (candidates CSV consumed by the evaluation scripts):
./scan_context_cross_sequence_evaluator <map.scd> <query.scd> <out.csv> \
    <shortlist_K> <w_low> <w_high> <min_joint_rings> <dist_thresh> <margin>
# Paper setting: 100 0.3 0.7 2 0.5 0.1
```

Canonical descriptor parameters (16 rings, 60 sectors, 30 m radius, 0.25 m
voxel, dual-envelope enabled) are in `updown_sc/config/descriptor_params.yaml`;
each released session ships the `runtime_params.yaml` it was built with.

## Reproducing the paper tables

1. Download the data package (`data/README.md`): query bins, gravity CSVs,
   map sessions, SCD databases, and all recorded per-query CSVs.
2. Recompute statistics without re-running retrieval:
   `experiments/common/compute_retrieval_stats.py` regenerates the Wilson
   intervals and F1max/AUPR behind Table II's uncertainty statements and
   Fig. 6; `compute_yaw_error.py` regenerates the yaw-seed numbers.
3. Re-run retrieval from inputs: the UpDown-SC tools above for our method;
   `experiments/common/rescore_native_baselines.py` and
   `run_gravity_frontend_transfer.py` (Python baselines);
   `baselines/adapters/lidar_iris/benchmark.cpp` (official Iris core).

Some experiment scripts retain the absolute paths of the original runs for
archival fidelity; the path map is documented in `data/README.md`.

## Licenses

All released code (UpDown-SC implementation, Python baselines, adapters,
experiment, and figure code) is under the MIT license (`LICENSE`). Official
baseline cores (LiDAR-Iris, BTC, STD) keep their upstream licenses and are
fetched, not redistributed.

## Citation

Citation entry will be added after the review period.
