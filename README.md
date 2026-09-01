# UpDown-SC: Gravity-Canonicalized Dual-Envelope Scan Context

Public code, baselines, evaluation protocols, and reproducibility artifacts
for the preprint "UpDown-SC: A Gravity-Canonicalized Dual-Envelope Scan
Context for Indoor LiDAR Place Recognition."

Every number in the paper traces to a per-query CSV in the data package; see
`docs/PROVENANCE.md`.

## Layout

| Directory | Contents |
|---|---|
| `updown_sc/` | Standalone C++ implementation of UpDown-SC (no ROS dependency): dual-envelope SCD construction with gravity canonicalization and adaptive split (`scan_context_rebuild`), masked non-uniform retrieval with yaw hypotheses (`scan_context_cross_sequence_evaluator`), plus SCD conversion/subset tools. |
| `baselines/python/` | Audited formula-equivalent CPU baselines (SC, SC++ (PC), SOLiD, M2DP, RING++ CPU port) and query/gravity export tools. |
| `baselines/adapters/` | Our adapters for official C++ cores (LiDAR-Iris benchmark harness; BTC/STD run via `experiments/` scripts). Official cores are fetched from their upstream repositories; we do not redistribute them. |
| `experiments/` | Per-dataset protocol scripts and READMEs (in-house two-session, RTK-SLAM Construction Hall, Newer College, indoor cross-device, M2DGR hall), plus official-checkpoint OverlapTransformer and MinkLoc3Dv2 adapters. |
| `figures/` | Deterministic figure generators with their source-data files and integrity metadata. |
| `docs/` | Provenance (paper number -> CSV mapping) and protocol manifests. |
| `data/` | Compact result bundles, checksums, and the manifest for the complete evaluation package (see `data/README.md`). |

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
cmake -S updown_sc -B build/updown-sc -DCMAKE_BUILD_TYPE=Release
cmake --build build/updown-sc -j
```

Tools:

```bash
# Rebuild an SCD database from a session directory
# (key_point_frame/*.pcd + optimized_poses_tum.txt + scan_context_gravity.csv
#  + runtime_params.yaml):
build/updown-sc/scan_context_rebuild <session_dir> <out.scd> gravity <runtime_params.yaml> \
    <origin_height_m> [fixed_split_m]

# Cross-sequence retrieval (candidates CSV consumed by the evaluation scripts):
build/updown-sc/scan_context_cross_sequence_evaluator <map.scd> <query.scd> <out.csv> \
    <shortlist_K> <w_low> <w_high> <min_joint_rings> <dist_thresh> <margin>
# Paper setting: 100 0.3 0.7 2 0.5 0.1
```

Canonical descriptor parameters (16 rings, 60 sectors, 30 m radius, 0.25 m
voxel, dual-envelope enabled) are in `updown_sc/config/descriptor_params.yaml`;
each released session ships the `runtime_params.yaml` it was built with.

## Python environment

The non-ROS evaluation and plotting utilities use Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Figure/PPT regeneration has additional optional dependencies in
`requirements-figures.txt`. The learned adapters intentionally use the exact
upstream environments recorded in `docs/PROVENANCE.md`; their model code and
weights are not vendored here.

## Reproducing the paper tables

1. Use the checked compact result bundles in `data/package/`, or download the
   complete data package described in `data/README.md`: query bins, gravity
   CSVs, map sessions, SCD databases, and all recorded per-query CSVs.
2. Recompute statistics without re-running retrieval:
   `experiments/common/compute_retrieval_stats.py` regenerates the Wilson
   intervals and F1max/AUPR behind Table II's uncertainty statements and
   Fig. 6; `compute_yaw_error.py` regenerates the yaw-seed numbers.
3. Re-run retrieval from inputs: the UpDown-SC tools above for our method;
   `experiments/common/rescore_native_baselines.py` and
   `run_gravity_frontend_transfer.py` (Python baselines);
   `baselines/adapters/lidar_iris/benchmark.cpp` (official Iris core); and
   `experiments/common/run_overlap_transformer.py` and
   `run_minkloc3dv2.py` for the two learned rows. These consume upstream
   [OverlapTransformer](https://github.com/haomo-ai/OverlapTransformer)
   and [MinkLoc3Dv2](https://github.com/jac99/MinkLoc3Dv2) checkouts with
   their official KITTI- and Oxford-trained checkpoints, respectively;
   neither upstream code nor weights are redistributed here. Each adapter
   verifies the checkpoint SHA-256 before running and records the upstream
   commit and full protocol in `run_manifest.json`.

Archived path fields use the portable `${UPDOWN_SC_ROOT}` placeholder; the
path convention is documented in `data/PATHS.md`.

The release preflight builds all four C++ tools, validates the Python entry
points and archive checksums, and replays the released M2DGR retrieval:

```bash
./scripts/release_check.sh
```

## Licenses

Project-authored code is under the MIT license (`LICENSE`). Official baseline
cores and checkpoints (LiDAR-Iris, BTC, STD, OverlapTransformer,
MinkLoc3Dv2) keep their upstream licenses and are fetched, not redistributed.
Dataset, baseline, and media attribution is recorded in
`THIRD_PARTY_NOTICES.md` and `docs/assets/ATTRIBUTION.md`.

## Citation

See `CITATION.cff`. The arXiv identifier will be added when the preprint is
posted.
