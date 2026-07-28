# Newer College outdoor trial

This trial uses the public Newer College `quad_easy` sequence. The sensor is an
Ouster OS0-128 spinning LiDAR with its built-in IMU. To avoid downloading the
camera streams, the runtime dataset uses the LiDAR/IMU-only ROS 2 repack
published with the IROS 2024 dynamic-LiDAR annotations.

Runtime data live under:

```text
${UPDOWN_SC_ROOT}/icra2027_runtime/datasets/newer_college/quad_easy_ros2/
```

The downloaded bag contains only:

```text
/os_cloud_node/points  sensor_msgs/msg/PointCloud2  10 Hz
/os_cloud_node/imu     sensor_msgs/msg/Imu          100 Hz
```

The session was produced with our FAST-LIO2-based front end (dataset override
`newer_college_os0.yaml`; front end not part of this release — see the root
README). The released `nc_2m` data group contains the resulting deskewed
keyframe session and query set, from which everything below reproduces.

`replay_raw_os1.py` is a low-storage compatibility utility for the original
2020 raw PCD/CSV release. It streams scans directly from a ZIP and does not
extract a second copy.

## BTC accumulated-input result

BTC uses the official C++ core with the outdoor configuration. Each input is
the current deskewed LiDAR scan plus its nine immediately preceding scans,
registered by FAST-LIO into the current `base_link` frame. The two temporal
halves are replayed independently so the first query window cannot reuse scans
from the map half. Dense ten-scan windows are then sampled at the same
experiment-only 2 m centers used by the single-scan methods.

The incomplete causal prefix is excluded, leaving 58 map submaps and 59 query
submaps. With the common 5 m correctness radius, official BTC verification
gives:

| Variant | R@1 | R@5 | R@10 | Empty candidates |
|---|---:|---:|---:|---:|
| Native | 52.5 | 61.0 | 64.4 | 8/59 |
| +G | 45.8 | 59.3 | 61.0 | 8/59 |

The exact-self audit returns Rank-1 for both variants. Generated submaps,
per-query CSVs, and logs are under:

```text
${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/newer_college_btc_2m
```
