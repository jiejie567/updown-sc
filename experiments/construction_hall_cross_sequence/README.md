# RTK-SLAM Construction Hall cross-sequence experiment

This experiment uses the larger Construction Hall Seq. 1 traversal as the
map/database and the reverse Seq. 2 traversal as the query/localization run.

## Protocol

- Sensor: Livox MID-360 with its built-in IMU.
- Map/database: Construction Hall Seq. 1, 189 experimental keyframes.
- Query/localization: Construction Hall Seq. 2, 148 experimental keyframes.
- Keyframe rule: retain the first frame, then retain a frame when its 3-D
  translation from the last retained frame is `>= 2 m`.
- The 2 m rule is experiment-only. The production FAST-LIO mapping and
  localization keyframe policy remains `0.5 m OR 10 deg`.
- Fixed-time sampling: disabled.
- Yaw-triggered experimental sampling: disabled.
- Ground truth: valid GNSS segments independently align each FAST-LIO
  trajectory to a metric local frame; Seq. 2 truth is then expressed in the
  Seq. 1 map frame.
- Correct retrieval: at least one returned database keyframe lies within
  `5 m` of the aligned query truth.
- Query coverage: all 148 single-scan queries have a database place within
  5 m, so none are excluded.
- Input fairness: single-scan baselines use the same native, deskewed
  `base_link` cloud. The paper reports both native and explicitly labelled
  `+G` columns; `+G` applies the same gravity front end to every compatible
  single-scan method.
- BTC requires accumulated geometry and is reported separately from the
  single-scan methods. It follows its released protocol: each experimental
  center uses the current raw LiDAR scan and nine immediately preceding causal
  scans, registered with the FAST-LIO trajectory into the current scan frame.
  The same 0.3--30 m crop is applied. The same 139 complete one-to-one query
  centers are evaluated. The diagnostic `+G` variant removes the accumulated
  submap reference roll/pitch after registration while preserving yaw.

## Results

| Method | Input | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|
| UpDown-SC (native) | 1 keyframe | 42.6 | 70.9 | 79.1 |
| **UpDown-SC (+G)** | 1 keyframe | **63.5** | **85.8** | **91.9** |
| SC | 1 keyframe | 51.4 | 66.9 | 79.7 |
| SC++ (PC) | 1 keyframe | 39.9 | 68.9 | 79.1 |
| SOLiD | 1 keyframe | 35.8 | 75.0 | 82.4 |
| M2DP | 1 keyframe | 48.6 | 68.2 | 75.7 |
| LiDAR-Iris | 1 keyframe | 42.6 | 67.6 | 81.8 |
| RING++ | 1 keyframe | 34.5 | 56.8 | 69.6 |
| BTC (native) | 10 consecutive scans | 11.5 | 23.0 | 25.9 |
| BTC (+G) | 10 consecutive scans | 15.8 | 30.9 | 37.4 |

RING++ is an audited faithful CPU formula port of the released CUDA method;
it is suitable for recall comparison here but not for runtime claims. SC,
SC++ (PC), SOLiD, and M2DP use audited formula-equivalent CPU implementations.
LiDAR-Iris and BTC use their official C++ cores with path/dataset adapters.
UpDown-SC calls the current production C++ database/query code.

The retired BTC value 42.4/61.9 was generated in each traversal's odometry
world axes and without the declared 30 m common crop. That representation can
carry a favorable trajectory-heading prior and is not the body-frame
cross-session input used by the other methods; it is therefore not a paper
result.

## Descriptor-origin height audit

The exported `base_link` origin in this experiment coincides with the
MID-360 built-in IMU; it is not a ground-referenced robot base. After rotating
each selected keyframe by its recorded gravity vector, the local ground
distribution gives median origin heights of `1.294 m` for Seq. 1 and `1.290 m`
for Seq. 2. Both descriptor sets therefore use the physically measured
`origin_height_from_ground: 1.3`.

A clean V7 rebuild from the 189/148 original selected keyframes reproduces
42.6/70.9 R@1/R@5 without gravity and 63.5/85.8 with gravity. The independent
audit is stored in:

```text
${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/ch_height_calibration_v7_20260720/
```

For diagnosis only, changing the calibrated height to 1.2/1.4 m gives
64.9/88.5 and 62.8/85.1 under `+G`. Setting `h=0` gives 65.5/91.9, but is
physically invalid: it moves the nominal 2.5 m partition to approximately
3.8 m above the actual ground. This diagnostic must not be reported as a
height-calibrated result. The superseded 66.2/92.6 snapshot came from the
pre-ground-relative descriptor semantics and is retired.

The online Seq. 2 replay localized once in 371 ms (ICP fitness 0.0146,
overlap 1.0) and produced 5872 trusted poses without a health warning. Against
the independent GNSS-aligned trajectory, its position error is 0.155 m median,
0.308 m at the 95th percentile, and 0.349 m maximum.

## Outputs

All generated data live under:

```text
${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/rtk_slam_construction_hall_2m/
```

Important files:

- `results/recall_summary.csv`: merged method table.
- `results/*_per_query.csv`: per-query results.
- `derived/protocol_manifest.json`: split, keyframe, truth, and overlap audit.
- `localization/seq2_trusted_pose.csv`: trusted online localization output.
- `localization/trajectory_evaluation.json`: independent trajectory check.
- `figures/construction_hall_cross_sequence.{svg,pdf,png,tiff}`: paper figure.
- `video/construction_hall_experiment.mp4`: data-driven experiment video.

The scripts in this directory prepare truth, build spatial-keyframe submaps,
run/merge methods, draw the paper figure, record/evaluate localization, and
render the Manim video. They use the experiment-only translation-only 2 m
rule and never use fixed-time or yaw-triggered experimental sampling.
