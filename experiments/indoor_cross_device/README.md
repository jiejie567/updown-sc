# Indoor cross-device/platform replay

This experiment reuses the `indoor_handle1_ros2` mapping session as one prior
map and independently replays `indoor_handle2_ros2` and
`indoor_vehicle1_ros2` as localization queries.

All three recordings use the same MID-360 model. The experiment therefore
tests acquisition-device, platform, and mounting-height transfer; it does not
claim transfer across different LiDAR beam patterns.

## Protocol

- The online prior contains all 477 production keyframes from Handle1.
- Online relocalization uses one 0.1 s deskewed scan and then runs each complete
  bag against the same prior map.
- The analysis-only subset uses translation-only 2 m keyframes. This does not
  change FAST-LIO's production keyframe policy.
- A query is eligible when its localization-derived pseudo-reference lies
  within 2 m of a selected map pose.
- Place and yaw are selected first. The vertical estimate is then evaluated on
  the highest-ranked spatially correct hypothesis, so the estimated `delta_z`
  cannot alter retrieval rank.

## Result snapshot

| Replay | Eligible / Z-evaluable | Online init. | Fitness / overlap | Estimated / reference median `delta_z` | Z MAE |
|---|---:|---:|---:|---:|---:|
| Handle1 to Handle2 | 58 / 14 | 656 ms | 0.011 / 1.000 | -0.019 / -0.025 m | 1.2 cm |
| Handle1 to Vehicle1 | 57 / 17 | 689 ms | 0.016 / 0.999 | -0.807 / -0.827 m | 2.7 cm |

Both complete online replays produced trusted localization trajectories without
a localization-health warning. The reference height is derived from the
continuous-localization trajectory and is therefore a pseudo-reference, not
survey-grade ground truth. The two completed bags are deployment case studies,
not a statistical success-rate estimate.

The corresponding Table-III retrieval columns use the common gravity front end
and a 2 m correctness radius. Ground-relative descriptor heights use
point-cloud-verified origin heights of 1.6 m for Handle and 0.8 m for Vehicle,
with a common 2.5 m physical split and retrieval-only offset of 0.1 m.
UpDown-SC reaches 44.8/70.7 and 52.6/82.5 R@1/R@5 on Handle2 and Vehicle1,
with ranked candidates for all 58/58 and 57/57 eligible queries. LiDAR Iris
reaches 44.8/69.0 and 61.4/86.0, respectively. A deliberately incorrect
Vehicle origin height of 1.1 m lowers UpDown-SC to 33.3/68.4, so the platform
height is treated as a calibration quantity rather than a retrieval
hyperparameter.

The accumulated-input BTC completion uses the same 2 m centers and the current
scan plus nine causal predecessors. Its gravity-canonicalized registered
submaps reach 25.9/46.6 on Handle2 and 19.3/47.4 on Vehicle1. Native registered
submaps reach 19.0/32.8 and 19.3/31.6, respectively.

## Reproducible artifacts

The runtime root is:

```text
${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/indoor_cross_device_2m
```

Important files:

- `results/experiment_summary.json`: values consumed by the paper and video.
- `results/deployment_summary.csv`: online and vertical-estimation summary.
- `results/table3_retrieval_columns.csv`: fair +G retrieval columns for Table III.
- `results/{handle2,vehicle1}/updown_candidates.csv`: candidate-level evidence.
- `results/{handle2,vehicle1}/updown_summary.csv`: per-query aggregates.
- `localization/{handle2,vehicle1}/trusted_pose.csv`: full trusted trajectories.
- `localization/{handle2,vehicle1}/fastlio.log`: complete replay logs.
- `video/indoor_cross_device_experiment.mp4`: standalone experiment chapter.

Render the standalone chapter with:

```bash
cd ${UPDOWN_SC_ROOT}/updown-sc/experiments/indoor_cross_device
INDOOR_CROSS_DEVICE_EXPERIMENT=${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/indoor_cross_device_2m \
  ${UPDOWN_SC_ROOT}/icra2027_runtime/envs/construction-hall-video/bin/manim \
  -qh --fps 30 --resolution 1920,1080 \
  indoor_cross_device_video.py IndoorCrossDeviceExperiment \
  --media_dir ${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/indoor_cross_device_2m/video/media_final
```

The experiment scripts are data-driven and do not contain hard-coded result
values. See `video_plan.md` for the evidence and claim contract.

Rebuild the Table-III source CSV with:

```bash
python3 build_table3_columns.py \
  --experiment-root ${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/indoor_cross_device_2m
```

The 64-frame map is smaller than the official LiDAR-Iris adapter's fixed
Top-100 shortlist. Apply `patches/lidar_iris_dynamic_topk.patch` so the adapter
uses `min(100, database_size)`; this only prevents an out-of-range shortlist
and does not change ranking when the database has at least 100 entries.
