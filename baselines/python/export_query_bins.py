#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--result-csv", required=True, type=Path)
    parser.add_argument("--truth-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--topic", default="/cloud_registered_body")
    args = parser.parse_args()

    with args.result_csv.open(newline="") as handle:
        result_rows = list(csv.DictReader(handle))
    with args.truth_csv.open(newline="") as handle:
        truth = {int(row["window"]): row for row in csv.DictReader(handle)}

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=args.bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"))
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if args.topic not in topic_types:
        raise RuntimeError(f"Topic not found: {args.topic}")
    message_type = get_message(topic_types[args.topic])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    query_index = 0
    while reader.has_next():
        topic, data, storage_stamp = reader.read_next()
        if topic != args.topic:
            continue
        if query_index >= len(result_rows):
            raise RuntimeError("Bag contains more query clouds than the reference result CSV")
        message = deserialize_message(data, message_type)
        field_names = {field.name for field in message.fields}
        requested = ["x", "y", "z"]
        if "intensity" in field_names:
            requested.append("intensity")
        points = point_cloud2.read_points_numpy(
            message, field_names=requested, skip_nans=True).astype(np.float32, copy=False)
        if points.shape[1] == 3:
            points = np.column_stack((points, np.zeros(points.shape[0], dtype=np.float32)))
        points = points[np.isfinite(points[:, :3]).all(axis=1)]

        result = result_rows[query_index]
        expected_points = int(result["raw_points"])
        if len(points) != expected_points:
            raise RuntimeError(
                f"Query {query_index}: point count {len(points)} != reference {expected_points}")
        filename = f"{query_index:06d}.bin"
        points.tofile(args.output_dir / filename)

        truth_index = round(float(result["start_s"]) / 10.0)
        reference = truth.get(truth_index)
        metadata.append({
            "query_id": query_index,
            "window": result["window"],
            "start_s": result["start_s"],
            "header_stamp": message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
            "storage_stamp": storage_stamp * 1e-9,
            "points": len(points),
            "file": filename,
            "truth_index": truth_index,
            "truth_valid": reference is not None,
            "truth_x": reference["truth_x"] if reference else "",
            "truth_y": reference["truth_y"] if reference else "",
        })
        query_index += 1

    if query_index != len(result_rows):
        raise RuntimeError(
            f"Bag contains {query_index} query clouds, reference CSV contains {len(result_rows)}")
    metadata_path = args.output_dir / "metadata.csv"
    with metadata_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    print(f"Exported {query_index} query clouds to {args.output_dir}")
    print(metadata_path)


if __name__ == "__main__":
    main()
