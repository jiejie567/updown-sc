# Third-party notices

UpDown-SC's standalone C++ implementation and project-authored adapters are
released under the repository's MIT license. The repository does not vendor
official baseline repositories or learned-model checkpoints. The adapters
load separately obtained upstream code or operate as independent
formula-equivalent implementations.

| Resource | Use in this repository | Upstream license / boundary |
|---|---|---|
| [Scan Context](https://github.com/gisbi-kim/scancontext) | Independent implementation of the published descriptor and distance | No upstream source code is redistributed. |
| [M2DP](https://github.com/LiHeUA/M2DP) | Independent NumPy implementation of the published procedure | The upstream repository does not declare an SPDX license; its MATLAB source is not redistributed. |
| [SOLiD](https://github.com/sparolab/SOLiD) | Independent formula-equivalent NumPy implementation | Upstream BSD-3-Clause; upstream source is not redistributed. |
| [RING / RING++](https://github.com/lus6-Jenny/RING) | CPU formula port used for audited comparisons | Upstream MIT; the official CUDA repository is not vendored. |
| [OverlapTransformer](https://github.com/haomo-ai/OverlapTransformer) | Adapter for the official KITTI checkpoint | Upstream GPL-3.0; neither source nor checkpoint is redistributed. Users obtain and run it separately under its upstream terms. |
| [MinkLoc3Dv2](https://github.com/jac99/MinkLoc3Dv2) | Adapter for the official Oxford checkpoint | Upstream MIT; neither source nor checkpoint is redistributed. |
| LiDAR-Iris, BTC, and STD | Dataset/protocol adapters for official C++ cores | Official cores are fetched separately and retain their upstream terms. |
| [M2DGR](https://github.com/SJTU-ViSYS/M2DGR) | Processed hall-session evaluation subset | Upstream MIT; cite the M2DGR paper when using the subset. |

Per-run manifests in the learned result bundles record the exact upstream
commit and checkpoint SHA-256. Academic method citations are listed in the
paper; this file documents software and data provenance rather than replacing
those citations.
