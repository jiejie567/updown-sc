<div align="center">

# UpDown-SC

**Gravity-Canonicalized Dual-Envelope Scan Context for Indoor LiDAR Place Recognition**

<a href="https://anyverse.com/"><img src="docs/assets/anyverse-dynamics-logo.png" width="280" alt="Anyverse Dynamics"></a>

[Project Page](https://jiejie567.github.io/updown-sc/) · [Results & Provenance](docs/PROVENANCE.md) · [Data](data/README.md)

</div>

English | [中文](./README_CN.md)

UpDown-SC is a training-free LiDAR place-recognition descriptor for indoor
scenes. Standard Scan Context keeps only the maximum height in each polar
cell, so a broad ceiling can hide the lower structures that distinguish nearby
rooms and corridors. UpDown-SC first aligns each scan with gravity, then keeps
two surfaces per cell: the upper envelope of lower/middle structures and the
lower envelope of overhead structures.

The repository contains the standalone C++ implementation, experiment
protocols, baseline adapters, result files, and figure code used for the
preprint.

<p align="center">
  <a href="https://jiejie567.github.io/updown-sc/"><img src="docs/assets/updown_sc_hero.png" width="380" alt="UpDown-SC indoor LiDAR descriptor overview"></a>
</p>

## See it in motion

<p align="center"><img src="docs/assets/updown_sc_method_preview.gif" width="720" alt="Measured indoor scan separating lower and overhead structure"></p>
<p align="center"><em>One measured scan: the lower and overhead structures carry complementary cues.</em></p>

<p align="center"><img src="docs/assets/updown_sc_relocalization_preview.gif" width="720" alt="Successive ICP registration attempts over a fixed prior point-cloud map"></p>
<p align="center"><em>Successive ICP registrations on a fixed prior map; failed attempts are marked.</em></p>

The [complete English video plays on the project page](https://jiejie567.github.io/updown-sc/#video).

## Quick start

Dependencies: CMake 3.16+, a C++17 compiler, Eigen3, PCL (`common` and `io`),
and yaml-cpp. ROS is not required.

```bash
git clone https://github.com/jiejie567/updown-sc.git
cd updown-sc

cmake -S updown_sc -B build/updown-sc -DCMAKE_BUILD_TYPE=Release
cmake --build build/updown-sc -j
```

Rebuild a descriptor database from a released session:

```bash
build/updown-sc/scan_context_rebuild \
    <session_dir> <out.scd> gravity <runtime_params.yaml> \
    <origin_height_m> [fixed_split_m]
```

Run cross-sequence retrieval:

```bash
build/updown-sc/scan_context_cross_sequence_evaluator \
    <map.scd> <query.scd> <out.csv> \
    <shortlist_K> <w_low> <w_high> <min_joint_rings> <dist_thresh> <margin>

# Paper setting
# 100 0.3 0.7 2 0.5 0.1
```

The default descriptor uses 16 rings, 60 sectors, a 30 m radius, and a 0.25 m
voxel size. See `updown_sc/config/descriptor_params.yaml`; each released
session also includes the exact `runtime_params.yaml` used to build it.

## Reproducing the results

Compact result bundles and checksums are under `data/package/`. The complete
evaluation package is described in [`data/README.md`](data/README.md).

- [`docs/PROVENANCE.md`](docs/PROVENANCE.md) maps each reported number to its
  per-query CSV and experiment configuration.
- `experiments/common/compute_retrieval_stats.py` recomputes Wilson intervals,
  F1max, and AUPR.
- `experiments/common/compute_yaw_error.py` recomputes the yaw-seed results.
- `experiments/common/rescore_native_baselines.py` and
  `run_gravity_frontend_transfer.py` rerun the Python baselines.
- `experiments/common/run_overlap_transformer.py` and `run_minkloc3dv2.py`
  run the learned baselines with their official checkpoints.

Run the release check before reproducing individual experiments:

```bash
./scripts/release_check.sh
```

It builds the four C++ tools, checks the Python entry points and archives, and
replays the released M2DGR retrieval.

## Repository map

| Path | Contents |
|---|---|
| `updown_sc/` | ROS-free C++ descriptor construction, retrieval, and SCD utilities |
| `baselines/python/` | CPU implementations of SC, SC++ (PC), SOLiD, M2DP, and RING++ |
| `baselines/adapters/` | Interfaces to the official LiDAR-Iris, BTC, and STD code |
| `experiments/` | Dataset protocols and learned-baseline adapters |
| `data/` | Result bundles, checksums, and full-package manifest |
| `figures/` | Figure scripts and source data |
| `docs/` | Project page, protocol manifest, and result provenance |

Official baseline repositories and learned checkpoints are fetched from their
upstream sources and are not redistributed here. The exact commits, checkpoint
hashes, and transfer settings are recorded in
[`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## Data scope

The released experiments start from deskewed, motion-compensated keyframe
clouds. The package includes map sessions, query frames, poses, gravity
directions, descriptor parameters, SCD databases, and per-query result CSVs.

The FAST-LIO2-based ROS 2 front end used to produce those sessions is not part
of this release. Descriptor construction and retrieval can be reproduced from
the released clouds alone. Archived path fields use the portable
`${UPDOWN_SC_ROOT}` placeholder documented in [`data/PATHS.md`](data/PATHS.md).

## Python tools

The evaluation and plotting utilities use Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional figure and presentation dependencies are listed in
`requirements-figures.txt`.

## Deployment guide

[`updown-sc-deployment/SKILL.md`](.agents/skills/updown-sc-deployment/SKILL.md)
provides a compact deployment checklist for coding agents and automated setup
tools. The commands above remain the canonical manual installation path.

## License

Project-authored code is released under the [MIT License](LICENSE). Dataset,
baseline, and media attribution is listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`docs/assets/ATTRIBUTION.md`](docs/assets/ATTRIBUTION.md).

## Citation

See [`CITATION.cff`](CITATION.cff). The arXiv identifier will be added after
the submitted preprint receives its public identifier.
