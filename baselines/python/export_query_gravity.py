#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import rosbag2_py
from geometry_msgs.msg import Vector3Stamped
from rclpy.serialization import deserialize_message


parser = argparse.ArgumentParser()
parser.add_argument("--bag", required=True)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--topic", default="/scan_context_gravity_up")
args = parser.parse_args()


reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id="mcap"),
            rosbag2_py.ConverterOptions("cdr", "cdr"))
rows = []
while reader.has_next():
    topic, data, _ = reader.read_next()
    if topic != args.topic:
        continue
    message = deserialize_message(data, Vector3Stamped)
    rows.append({
        "query_id": len(rows),
        "stamp": message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
        "up_x": message.vector.x,
        "up_y": message.vector.y,
        "up_z": message.vector.z,
    })

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} gravity samples to {args.output}")
