# Private two-bag 2 m retrieval experiment

This directory regenerates the former 220-query visualization/regression
experiment with the common experiment-only spatial protocol.

## Protocol

- Retain the first frame, then retain the next frame whose 3-D Euclidean
  translation from the last retained frame is at least 2 m.
- Do not use a time trigger or yaw trigger.
- Do not change FAST-LIO's production mapping/localization keyframe policy
  (`0.5 m OR 10 deg`). The capture script applies temporary ROS parameter
  overrides only to export the experiment session.
- Use the loop-corrected mapping session as the database: 376 keyframes.
- Use the complete localization bag as the query traversal: 322 selected
  keyframes, of which 320 have a database keyframe within the 2 m correctness
  radius.
- Query positions are associated with a dense prior-map localization replay.
  They are pseudo-reference positions, not independent ground truth, so this
  split is used only for private regression and video evidence.
- Single-scan methods use one common deskewed `base_link` cloud per query.
  BTC uses the current raw LiDAR scan and its nine immediately preceding
  scans, registered into the current scan's `base_link` frame. Its submap
  centers are then sampled at the same experiment-only 2 m locations.
  The older 10-spatial-keyframe BTC/STD runs are retained only as diagnostics
  and are not paper results.

## Current measured recall

| Method | Input | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|
| **UpDown-SC** | 1 keyframe | **70.6** | 86.2 | 92.8 |
| SC | 1 keyframe | 59.1 | 74.1 | 82.5 |
| SC++ (PC) | 1 keyframe | 53.4 | 76.9 | 83.8 |
| SOLiD | 1 keyframe | 35.6 | 57.5 | 65.6 |
| M2DP | 1 keyframe | 47.8 | 63.7 | 68.4 |
| LiDAR-Iris | 1 keyframe | 58.1 | 72.2 | 78.1 |
| RING++ | 1 keyframe | 62.2 | **88.8** | **94.4** |
| BTC (native) | 10 consecutive scans | 48.3 | 66.1 | 70.8 |
| BTC (+G) | 10 consecutive scans | 52.0 | 64.3 | 68.3 |

The complete generated data and result CSVs live outside the synchronized
workspace at:

`${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/private_two_bag_2m`

The auditable protocol is recorded in
`derived/protocol_manifest.json`, and the merged measurements are in
`results/recall_summary.csv`.

The audited causal BTC rerun is stored under
`${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/btc_completion_20260720`.
It excludes the first incomplete center, leaving 375 map submaps and 319
eligible queries. The retired 13.1/30.1 result accumulated ten 2 m-spaced
experimental keyframes rather than ten consecutive LiDAR scans.
