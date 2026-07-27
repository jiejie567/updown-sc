#!/usr/bin/env python3
"""Build causal 10-scan input at the 2 m query locations.

STD's ``sub_frame_num`` denotes consecutive LiDAR scans, not experimental
spatial keyframes.  This exporter reads ``/livox/points`` from the original
bag, registers the current scan and its nine predecessors with the FAST-LIO
trajectory, and retains the odometry/gravity axes used by the official Livox
demo.  The output centers still follow the experiment's 2 m keyframe rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import PointCloud2


def read_tum(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.shape[1] != 8 or np.any(np.diff(poses[:, 0]) <= 0.0):
        raise RuntimeError(f"Invalid TUM trajectory: {path}")
    return poses


class PoseInterpolator:
    def __init__(self, poses: np.ndarray) -> None:
        self.stamps = poses[:, 0]
        self.translation = poses[:, 1:4]
        self.slerp = Slerp(
            self.stamps,
            Rotation.from_quat(poses[:, 4:8]),
        )

    def at(self, stamp: float) -> tuple[np.ndarray, np.ndarray]:
        clipped = float(np.clip(stamp, self.stamps[0], self.stamps[-1]))
        translation = np.asarray(
            [np.interp(clipped, self.stamps, self.translation[:, axis])
             for axis in range(3)]
        )
        return translation, self.slerp(clipped).as_matrix()


def pointcloud_xyzi(message: PointCloud2) -> np.ndarray:
    offsets = {field.name: field.offset for field in message.fields}
    required = ("x", "y", "z", "intensity")
    if any(name not in offsets for name in required):
        raise RuntimeError(f"PointCloud2 lacks XYZI fields: {sorted(offsets)}")
    dtype = np.dtype({
        "names": list(required),
        "formats": ["<f4"] * 4,
        "offsets": [offsets[name] for name in required],
        "itemsize": message.point_step,
    })
    raw = np.frombuffer(message.data, dtype=dtype, count=message.width * message.height)
    points = np.column_stack([raw[name] for name in required]).astype(np.float32)
    return points[np.isfinite(points[:, :3]).all(axis=1)]


def voxel_centroids(points: np.ndarray, leaf: float) -> np.ndarray:
    cells = np.floor(points[:, :3] / leaf).astype(np.int32)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, 4), dtype=np.float64)
    np.add.at(sums, inverse, points)
    counts = np.bincount(inverse)
    return (sums / counts[:, None]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--source-poses", required=True, type=Path)
    parser.add_argument("--selected-poses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--topic", default="/livox/points")
    parser.add_argument("--scans-per-submap", type=int, default=10)
    parser.add_argument("--voxel", type=float, default=0.1)
    parser.add_argument("--minimum-radius", type=float, default=0.3)
    parser.add_argument("--maximum-radius", type=float, default=30.0)
    parser.add_argument(
        "--reference-orientation",
        choices=("world", "native"),
        default="world",
        help=(
            "world retains the historical STD gravity-aligned odometry frame; "
            "native expresses each submap in the current scan body frame for BTC"
        ),
    )
    args = parser.parse_args()
    if args.minimum_radius < 0.0:
        parser.error("--minimum-radius must be nonnegative")
    if 0.0 < args.maximum_radius <= args.minimum_radius:
        parser.error("--maximum-radius must exceed --minimum-radius or be <= 0")

    source_poses = read_tum(args.source_poses)
    selected_poses = read_tum(args.selected_poses)
    interpolator = PoseInterpolator(source_poses)
    target_index = 0
    # Keep the first frame before the target when the newly read frame is
    # already just after it.
    history: deque[tuple[float, np.ndarray]] = deque(
        maxlen=args.scans_per_submap + 1
    )
    metadata: list[dict[str, object]] = []
    submap_dir = args.output / "submaps"
    submap_dir.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    while reader.has_next() and target_index < len(selected_poses):
        topic, data, _ = reader.read_next()
        if topic != args.topic:
            continue
        message = deserialize_message(data, PointCloud2)
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        history.append((stamp, pointcloud_xyzi(message)))

        while (
            target_index < len(selected_poses)
            and stamp >= selected_poses[target_index, 0]
        ):
            target_stamp = selected_poses[target_index, 0]
            causal = [item for item in history if item[0] <= target_stamp]
            if len(causal) < args.scans_per_submap:
                target_index += 1
                continue
            reference_translation, reference_rotation = interpolator.at(target_stamp)
            registered = []
            raw_points = 0
            for scan_stamp, points in causal[-args.scans_per_submap:]:
                translation, rotation = interpolator.at(scan_stamp)
                xyz = points[:, :3] @ rotation.T + translation
                xyz -= reference_translation
                if args.reference_orientation == "native":
                    xyz = xyz @ reference_rotation
                registered.append(np.column_stack((xyz, points[:, 3])))
                raw_points += len(points)
            cloud = np.concatenate(registered)
            radial_squared = np.square(cloud[:, 0], dtype=np.float64) + np.square(
                cloud[:, 1], dtype=np.float64
            )
            keep = radial_squared >= args.minimum_radius * args.minimum_radius
            if args.maximum_radius > 0.0:
                keep &= radial_squared <= args.maximum_radius * args.maximum_radius
            cloud = voxel_centroids(cloud[keep], args.voxel)
            filename = f"submaps/{target_index:06d}.bin"
            cloud.tofile(args.output / filename)
            metadata.append({
                "submap_id": target_index,
                "reference_stamp": f"{target_stamp:.9f}",
                "start_stamp": f"{causal[-args.scans_per_submap][0]:.9f}",
                "end_stamp": f"{causal[-1][0]:.9f}",
                "scans": args.scans_per_submap,
                "raw_points": raw_points,
                "output_points": len(cloud),
                "file": filename,
                "world_frame": (
                    "current_scan_native_body"
                    if args.reference_orientation == "native"
                    else "gravity_aligned_odometry"
                ),
                "body_frame": message.header.frame_id,
                "x": selected_poses[target_index, 1],
                "y": selected_poses[target_index, 2],
                "z": selected_poses[target_index, 3],
                "qx": selected_poses[target_index, 4],
                "qy": selected_poses[target_index, 5],
                "qz": selected_poses[target_index, 6],
                "qw": selected_poses[target_index, 7],
                "keyframe_start_index": target_index,
                "keyframe_end_index": target_index,
            })
            target_index += 1
            if len(metadata) % 25 == 0:
                print(f"Causal raw-scan submaps {len(metadata)}/{len(selected_poses)}",
                      flush=True)

    if not metadata:
        raise RuntimeError("No complete causal submaps were produced")
    with (args.output / "metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    manifest = {
        "format": "causal_consecutive_lidar_submaps_v2",
        "topic": args.topic,
        "center_rule": "experiment-only translation-only 2 m keyframes",
        "scans_per_submap": args.scans_per_submap,
        "causal": True,
        "reference_orientation": args.reference_orientation,
        "voxel_size_m": args.voxel,
        "minimum_radius_m": args.minimum_radius,
        "maximum_radius_m": args.maximum_radius,
        "selected_centers": len(selected_poses),
        "output_submaps": len(metadata),
    }
    with (args.output / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
