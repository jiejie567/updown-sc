#!/usr/bin/env python3
"""Convert M2DGR ROS1 hall bags to ROS2, keeping LiDAR + IMU topics only."""

from __future__ import annotations

import sys
from pathlib import Path

from rosbags.convert import convert

SRC = Path("${UPDOWN_SC_ROOT}/icra2027_runtime/datasets/m2dgr")
DST = SRC / "ros2"
TOPICS = ["/velodyne_points", "/handsfree/imu"]


def main() -> None:
    DST.mkdir(exist_ok=True)
    names = sys.argv[1:] or ["hall_02", "hall_04", "hall_01"]
    for name in names:
        src = SRC / f"{name}.bag"
        dst = DST / name
        if not src.exists():
            print(f"skip {name}: bag not downloaded yet")
            continue
        if dst.exists():
            print(f"skip {name}: already converted")
            continue
        print(f"converting {name} ...", flush=True)
        convert(srcs=[src], dst=dst, include_topics=TOPICS)
        print(f"done {name}", flush=True)


if __name__ == "__main__":
    main()
