#!/usr/bin/env python3
"""Prepare a reproducible 2 m cross-traversal Newer College experiment.

The public ``quad_easy`` sequence starts and ends at the same location.  Its
first and second temporal halves are treated as database and query traversals.
Only experiment inputs are subsampled; FAST-LIO's production keyframe policy
is not changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import struct
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2


POINT_FIELD_FORMATS = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def select_spatial_keyframes(poses: np.ndarray, spacing_m: float) -> np.ndarray:
    selected = [0]
    last = poses[0, 1:4]
    for index in range(1, len(poses)):
        if np.linalg.norm(poses[index, 1:4] - last) >= spacing_m:
            selected.append(index)
            last = poses[index, 1:4]
    return np.asarray(selected, dtype=np.int64)


def quaternion_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("Invalid ground-truth quaternion")
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def gravity_up_in_body(pose: np.ndarray) -> np.ndarray:
    rotation_world_body = quaternion_rotation(*pose[4:8])
    up = rotation_world_body.T @ np.asarray([0.0, 0.0, 1.0])
    return up / np.linalg.norm(up)


def pointcloud_xyzi(message: PointCloud2) -> np.ndarray:
    field_by_name = {field.name: field for field in message.fields}
    required = ("x", "y", "z", "intensity")
    if any(name not in field_by_name for name in required):
        raise RuntimeError(f"PointCloud2 is missing one of {required}")
    names, formats, offsets = [], [], []
    for name in required:
        field = field_by_name[name]
        if field.datatype not in POINT_FIELD_FORMATS or field.count != 1:
            raise RuntimeError(f"Unsupported PointCloud2 field: {name}")
        names.append(name)
        formats.append(POINT_FIELD_FORMATS[field.datatype])
        offsets.append(field.offset)
    dtype = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": message.point_step,
    })
    rows = []
    for row in range(message.height):
        points = np.ndarray(
            shape=(message.width,),
            dtype=dtype,
            buffer=message.data,
            offset=row * message.row_step,
            strides=(message.point_step,),
        )
        rows.append(np.column_stack([points[name] for name in required]))
    result = np.concatenate(rows).astype(np.float32, copy=False)
    finite = np.isfinite(result[:, :3]).all(axis=1)
    radius = np.linalg.norm(result[:, :3], axis=1)
    return result[finite & (radius >= 0.3) & (radius <= 80.0)]


def write_binary_pcd(path: Path, points: np.ndarray) -> None:
    cloud = np.zeros((len(points), 8), dtype="<f4")
    cloud[:, :3] = points[:, :3]
    cloud[:, 6] = points[:, 3]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z normal_x normal_y normal_z intensity curvature\n"
        "SIZE 4 4 4 4 4 4 4 4\n"
        "TYPE F F F F F F F F\n"
        "COUNT 1 1 1 1 1 1 1 1\n"
        f"WIDTH {len(cloud)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(cloud)}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        cloud.tofile(handle)


def locate_database(bag_dir: Path) -> Path:
    databases = sorted(bag_dir.glob("*.db3"))
    if len(databases) != 1:
        raise RuntimeError(f"Expected exactly one db3 file in {bag_dir}")
    return databases[0]


def lidar_message_index(connection: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray]:
    topic = connection.execute(
        "SELECT id FROM topics WHERE name = ?", ("/os_cloud_node/points",)
    ).fetchone()
    if topic is None:
        raise RuntimeError("LiDAR topic /os_cloud_node/points is absent")
    rows = connection.execute(
        "SELECT id, timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (int(topic[0]),),
    ).fetchall()
    return (
        np.asarray([row[0] for row in rows], dtype=np.int64),
        np.asarray([row[1] for row in rows], dtype=np.int64),
    )


def nearest_message_indices(
    target_seconds: np.ndarray, message_timestamps_ns: np.ndarray
) -> np.ndarray:
    target_ns = np.rint(target_seconds * 1e9).astype(np.int64)
    right = np.searchsorted(message_timestamps_ns, target_ns)
    right = np.clip(right, 0, len(message_timestamps_ns) - 1)
    left = np.clip(right - 1, 0, len(message_timestamps_ns) - 1)
    choose_left = (
        np.abs(message_timestamps_ns[left] - target_ns)
        <= np.abs(message_timestamps_ns[right] - target_ns)
    )
    return np.where(choose_left, left, right)


def export_traversal(
    connection: sqlite3.Connection,
    message_ids: np.ndarray,
    message_timestamps_ns: np.ndarray,
    poses: np.ndarray,
    selected: np.ndarray,
    output_session: Path,
    query_dir: Path | None,
    map_positions: np.ndarray | None,
) -> list[dict]:
    frame_dir = output_session / "key_point_frame"
    frame_dir.mkdir(parents=True, exist_ok=True)
    if query_dir is not None:
        query_dir.mkdir(parents=True, exist_ok=True)
    chosen = nearest_message_indices(poses[selected, 0], message_timestamps_ns)
    association_errors = []
    metadata = []
    pose_rows = []
    gravity_rows = []
    for output_index, (pose_index, message_index) in enumerate(zip(selected, chosen)):
        pose = poses[pose_index]
        message_id = int(message_ids[message_index])
        serialized = connection.execute(
            "SELECT data FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if serialized is None:
            raise RuntimeError(f"Missing message id {message_id}")
        message = deserialize_message(bytes(serialized[0]), PointCloud2)
        header_stamp = (
            int(message.header.stamp.sec)
            + int(message.header.stamp.nanosec) * 1e-9
        )
        points = pointcloud_xyzi(message)
        write_binary_pcd(frame_dir / f"{output_index}.pcd", points)
        if query_dir is not None:
            points.astype("<f4", copy=False).tofile(
                query_dir / f"{output_index:06d}.bin"
            )
        pose_rows.append(pose)
        up = gravity_up_in_body(pose)
        gravity_rows.append((output_index, pose[0], *up))
        db_stamp = message_timestamps_ns[message_index] * 1e-9
        association_errors.append(abs(db_stamp - pose[0]))
        if map_positions is not None:
            errors = np.linalg.norm(map_positions[:, :2] - pose[None, 1:3], axis=1)
            nearest = int(np.argmin(errors))
            nearest_distance = float(errors[nearest])
        else:
            nearest = -1
            nearest_distance = float("nan")
        metadata.append({
            "query_id": output_index,
            "window": output_index,
            "start_s": pose[0] - poses[0, 0],
            "header_stamp": header_stamp,
            "points": len(points),
            "file": f"{output_index:06d}.bin",
            "truth_valid": map_positions is None or nearest_distance <= 5.0,
            "truth_x": pose[1],
            "truth_y": pose[2],
            "nearest_map_index": nearest,
            "nearest_map_distance_m": nearest_distance,
            "keyframe_rule": "translation_only_2m",
        })

    np.savetxt(output_session / "optimized_poses_tum.txt", pose_rows, fmt="%.9f")
    with (output_session / "scan_context_gravity.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "stamp", "up_x", "up_y", "up_z"))
        writer.writerows(gravity_rows)
    with (output_session / "association_audit.json").open("w") as handle:
        json.dump({
            "selected_count": len(selected),
            "max_pose_to_bag_stamp_error_s": max(association_errors),
            "median_pose_to_bag_stamp_error_s": float(np.median(association_errors)),
        }, handle, indent=2)
        handle.write("\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-dir", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--spacing-m", type=float, default=2.0)
    args = parser.parse_args()

    poses = np.loadtxt(args.ground_truth, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 8:
        raise RuntimeError("Ground truth must be an 8-column TUM trajectory")
    if np.any(np.diff(poses[:, 0]) <= 0.0):
        raise RuntimeError("Ground-truth timestamps are not strictly increasing")
    split = len(poses) // 2
    map_poses = poses[:split]
    query_poses = poses[split:]
    map_selected = select_spatial_keyframes(map_poses, args.spacing_m)
    query_selected = select_spatial_keyframes(query_poses, args.spacing_m)

    root = args.output_root
    map_session = root / "map/session"
    query_session = root / "query/session"
    query_bins = root / "derived/queries"
    for directory in (map_session, query_session, query_bins, root / "protocol"):
        directory.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(locate_database(args.bag_dir))
    message_ids, message_timestamps_ns = lidar_message_index(connection)
    map_metadata = export_traversal(
        connection, message_ids, message_timestamps_ns,
        map_poses, map_selected, map_session, None, None,
    )
    selected_map_poses = map_poses[map_selected]
    query_metadata = export_traversal(
        connection, message_ids, message_timestamps_ns,
        query_poses, query_selected, query_session, query_bins,
        selected_map_poses[:, 1:4],
    )
    connection.close()

    with (query_bins / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(query_metadata[0]))
        writer.writeheader()
        writer.writerows(query_metadata)
    query_gravity = list(csv.DictReader(
        (query_session / "scan_context_gravity.csv").open(newline="")
    ))
    with (query_bins / "gravity.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("query_id", "stamp", "up_x", "up_y", "up_z"))
        for row in query_gravity:
            writer.writerow((
                row["index"], row["stamp"], row["up_x"], row["up_y"], row["up_z"]
            ))

    np.savetxt(
        root / "protocol/map_selected_poses_tum.txt",
        selected_map_poses, fmt="%.9f",
    )
    np.savetxt(
        root / "protocol/query_selected_poses_tum.txt",
        query_poses[query_selected], fmt="%.9f",
    )
    manifest = {
        "format": "newer_college_quad_easy_cross_traversal_v1",
        "dataset": "Newer College quad_easy LiDAR/IMU-only ROS 2",
        "split_rule": "first temporal half is map; second temporal half is query",
        "split_source_index": split,
        "split_stamp": float(poses[split, 0]),
        "keyframe_rule": (
            "retain first pose; retain a later pose when 3-D translation from "
            "the last retained pose is at least spacing_m"
        ),
        "scope": "experiment_only",
        "runtime_fast_lio_keyframe_policy_changed": False,
        "spacing_m": args.spacing_m,
        "map_source_poses": len(map_poses),
        "query_source_poses": len(query_poses),
        "map_keyframes": len(map_selected),
        "query_keyframes": len(query_selected),
        "eligible_queries_within_5m": int(sum(
            row["truth_valid"] for row in query_metadata
        )),
        "all_points_retained_after_finite_0p3_to_80m_crop": True,
        "dynamic_labels_used": False,
        "source_bag": str(args.bag_dir.resolve()),
        "source_ground_truth": str(args.ground_truth.resolve()),
    }
    with (root / "protocol/manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
