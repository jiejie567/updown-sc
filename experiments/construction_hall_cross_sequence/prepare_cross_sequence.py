#!/usr/bin/env python3
"""Prepare RTK-SLAM Construction Hall for cross-sequence retrieval.

Seq.1 is the map traversal and Seq.2 is the query traversal.  The experiment
inputs are translation-only spatial subsets of FAST-LIO's source keyframes;
the production mapping/localization keyframe policy is not changed.

Both independent odometry trajectories are aligned to the valid outdoor GNSS
segments with a robust metric SE(2) fit.  Query truth is then expressed in the
Seq.1 map frame.  A query is eligible only when the map contains a keyframe
within the configured correctness radius.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from scipy.optimize import least_squares
from sensor_msgs.msg import NavSatFix


EARTH_RADIUS_M = 6378137.0


def read_tum(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.shape[1] != 8 or len(poses) == 0:
        raise RuntimeError(f"Invalid TUM trajectory: {path}")
    if np.any(np.diff(poses[:, 0]) <= 0.0):
        raise RuntimeError(f"Trajectory timestamps are not increasing: {path}")
    return poses


def read_gravity(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for expected, row in enumerate(rows):
        if int(row["index"]) != expected:
            raise RuntimeError(f"Non-contiguous gravity index in {path}")
    return rows


def read_binary_pcd(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = {}
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Missing DATA header in {path}")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break
        if header["DATA"][0].lower() != "binary":
            raise RuntimeError(f"Only binary PCD is supported: {path}")
        fields = header["FIELDS"]
        sizes = [int(value) for value in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        if any(size != 4 or kind != "F" or count != 1
               for size, kind, count in zip(sizes, types, counts)):
            raise RuntimeError(f"Unsupported PCD layout: {path}")
        dtype = np.dtype([(field, "<f4") for field in fields])
        raw = np.fromfile(handle, dtype=dtype, count=int(header["POINTS"][0]))
    intensity = raw["intensity"] if "intensity" in fields else np.zeros(len(raw), np.float32)
    points = np.column_stack((raw["x"], raw["y"], raw["z"], intensity))
    return points[np.isfinite(points[:, :3]).all(axis=1)].astype(np.float32, copy=False)


def read_gnss(bag: Path) -> np.ndarray:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    rows = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != "/gnss/fix":
            continue
        message = deserialize_message(data, NavSatFix)
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        rows.append((
            stamp,
            message.status.status,
            message.latitude,
            message.longitude,
            message.altitude,
            message.position_covariance[0],
            message.position_covariance[4],
            message.position_covariance[8],
        ))
    result = np.asarray(rows, dtype=np.float64)
    if len(result) == 0:
        raise RuntimeError(f"No /gnss/fix messages in {bag}")
    return result


def geodetic_to_local(gnss: np.ndarray, lat0: float, lon0: float, alt0: float) -> np.ndarray:
    latitude = np.radians(gnss[:, 2])
    longitude = np.radians(gnss[:, 3])
    east = EARTH_RADIUS_M * (longitude - math.radians(lon0)) * math.cos(math.radians(lat0))
    north = EARTH_RADIUS_M * (latitude - math.radians(lat0))
    up = gnss[:, 4] - alt0
    return np.column_stack((east, north, up))


def associate_valid_gnss(poses: np.ndarray, gnss: np.ndarray, local_gnss: np.ndarray,
                         max_time_error_s: float = 0.12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_indices = np.flatnonzero(gnss[:, 1] >= 0.0)
    valid_times = gnss[valid_indices, 0]
    selected_pose = []
    selected_gnss = []
    selected_time_error = []
    for pose_index, stamp in enumerate(poses[:, 0]):
        insertion = int(np.searchsorted(valid_times, stamp))
        choices = []
        if insertion < len(valid_times):
            choices.append(insertion)
        if insertion > 0:
            choices.append(insertion - 1)
        if not choices:
            continue
        nearest = min(choices, key=lambda index: abs(valid_times[index] - stamp))
        time_error = abs(valid_times[nearest] - stamp)
        if time_error > max_time_error_s:
            continue
        selected_pose.append(pose_index)
        selected_gnss.append(local_gnss[valid_indices[nearest], :2])
        selected_time_error.append(time_error)
    if len(selected_pose) < 20:
        raise RuntimeError("Too few valid GNSS/keyframe associations")
    return (
        np.asarray(selected_pose, dtype=np.int64),
        np.asarray(selected_gnss, dtype=np.float64),
        np.asarray(selected_time_error, dtype=np.float64),
    )


def apply_se2(xy: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    yaw, tx, ty = parameters
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return xy @ rotation.T + np.asarray([tx, ty])


def initial_se2(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    translation = target_center - source_center @ rotation.T
    return np.asarray([yaw, translation[0], translation[1]])


def robust_se2(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    initial = initial_se2(source, target)
    result = least_squares(
        lambda parameters: (apply_se2(source, parameters) - target).ravel(),
        initial,
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=1000,
    )
    residual = np.linalg.norm(apply_se2(source, result.x) - target, axis=1)
    return result.x, residual


def inverse_se2(xy: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    yaw, tx, ty = parameters
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return (xy - np.asarray([tx, ty])) @ rotation


def nearest_map_distance(query_xy: np.ndarray, map_xy: np.ndarray,
                         chunk_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    distances = np.empty(len(query_xy), dtype=np.float64)
    indices = np.empty(len(query_xy), dtype=np.int64)
    for start in range(0, len(query_xy), chunk_size):
        chunk = query_xy[start:start + chunk_size]
        squared = np.sum((chunk[:, None, :] - map_xy[None, :, :]) ** 2, axis=2)
        indices[start:start + len(chunk)] = np.argmin(squared, axis=1)
        distances[start:start + len(chunk)] = np.sqrt(np.min(squared, axis=1))
    return distances, indices


def export_query_clouds(query_session: Path, output_dir: Path, poses: np.ndarray,
                        truth_xy: np.ndarray, nearest_distance: np.ndarray,
                        nearest_index: np.ndarray, correct_radius_m: float) -> None:
    source_dir = query_session / "key_point_frame"
    pcd_paths = sorted(source_dir.glob("*.pcd"), key=lambda path: int(path.stem))
    if len(pcd_paths) != len(poses):
        raise RuntimeError(
            f"Query pose/PCD mismatch: poses={len(poses)} PCDs={len(pcd_paths)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    first_stamp = poses[0, 0]
    for index, path in enumerate(pcd_paths):
        if int(path.stem) != index:
            raise RuntimeError(f"Non-contiguous query PCD index: {path}")
        points = read_binary_pcd(path)
        filename = f"{index:06d}.bin"
        points.tofile(output_dir / filename)
        metadata.append({
            "query_id": index,
            "window": index,
            "start_s": f"{poses[index, 0] - first_stamp:.9f}",
            "header_stamp": f"{poses[index, 0]:.9f}",
            "points": len(points),
            "file": filename,
            "truth_valid": bool(nearest_distance[index] <= correct_radius_m),
            "truth_x": f"{truth_xy[index, 0]:.9f}",
            "truth_y": f"{truth_xy[index, 1]:.9f}",
            "nearest_map_index": int(nearest_index[index]),
            "nearest_map_distance_m": f"{nearest_distance[index]:.9f}",
            "keyframe_rule": "translation_only_2m",
        })
    with (output_dir / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)

    gravity = read_gravity(query_session / "scan_context_gravity.csv")
    if len(gravity) != len(poses):
        raise RuntimeError("Query pose/gravity count mismatch")
    with (output_dir / "gravity.csv").open("w", newline="") as handle:
        fields = ["query_id", "stamp", "up_x", "up_y", "up_z"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(gravity):
            writer.writerow({
                "query_id": index,
                "stamp": row["stamp"],
                "up_x": row["up_x"],
                "up_y": row["up_y"],
                "up_z": row["up_z"],
            })


def write_aligned_trajectory(path: Path, poses: np.ndarray, xy: np.ndarray) -> None:
    output = poses.copy()
    output[:, 1:3] = xy
    np.savetxt(path, output, fmt="%.9f")


def residual_summary(residual: np.ndarray) -> dict[str, float]:
    return {
        "count": int(len(residual)),
        "median_m": float(np.median(residual)),
        "p95_m": float(np.percentile(residual, 95)),
        "max_m": float(np.max(residual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--correct-radius", type=float, default=5.0)
    parser.add_argument("--experimental-spacing", type=float, default=2.0)
    args = parser.parse_args()

    map_session = args.experiment_root / "seq1/session"
    query_session = args.experiment_root / "seq2/session"
    map_poses = read_tum(map_session / "optimized_poses_tum.txt")
    query_poses = read_tum(query_session / "optimized_poses_tum.txt")

    map_gnss = read_gnss(args.dataset_root / "construction_seq1")
    query_gnss = read_gnss(args.dataset_root / "construction_seq2")
    valid_geodetic = np.vstack((
        map_gnss[map_gnss[:, 1] >= 0.0, 2:5],
        query_gnss[query_gnss[:, 1] >= 0.0, 2:5],
    ))
    lat0, lon0, alt0 = np.mean(valid_geodetic, axis=0)
    map_enu = geodetic_to_local(map_gnss, lat0, lon0, alt0)
    query_enu = geodetic_to_local(query_gnss, lat0, lon0, alt0)

    map_indices, map_targets, map_dt = associate_valid_gnss(map_poses, map_gnss, map_enu)
    query_indices, query_targets, query_dt = associate_valid_gnss(
        query_poses, query_gnss, query_enu)
    map_transform, map_residual = robust_se2(map_poses[map_indices, 1:3], map_targets)
    query_transform, query_residual = robust_se2(
        query_poses[query_indices, 1:3], query_targets)

    map_global_xy = apply_se2(map_poses[:, 1:3], map_transform)
    query_global_xy = apply_se2(query_poses[:, 1:3], query_transform)
    query_in_map_xy = inverse_se2(query_global_xy, map_transform)
    nearest_distance, nearest_index = nearest_map_distance(
        query_in_map_xy, map_poses[:, 1:3])

    derived = args.experiment_root / "derived"
    query_output = derived / "queries_seq2"
    derived.mkdir(parents=True, exist_ok=True)
    export_query_clouds(
        query_session,
        query_output,
        query_poses,
        query_in_map_xy,
        nearest_distance,
        nearest_index,
        args.correct_radius,
    )
    write_aligned_trajectory(
        derived / "seq1_poses_enu_tum.txt", map_poses, map_global_xy)
    write_aligned_trajectory(
        derived / "seq2_poses_enu_tum.txt", query_poses, query_global_xy)
    write_aligned_trajectory(
        derived / "seq2_truth_in_seq1_map_tum.txt", query_poses, query_in_map_xy)

    manifest = {
        "protocol": {
            "map_sequence": "Construction Hall Seq.1",
            "query_sequence": "Construction Hall Seq.2",
            "keyframe_rule": (
                f"first frame, then 3-D translation >= "
                f"{args.experimental_spacing:g} m from the last retained frame"
            ),
            "experimental_spacing_m": args.experimental_spacing,
            "time_based_sampling": False,
            "yaw_based_sampling": False,
            "scope": "experiment only; production FAST-LIO keyframe policy unchanged",
            "correct_radius_m": args.correct_radius,
            "query_count_total": int(len(query_poses)),
            "query_count_with_map_place": int(np.sum(nearest_distance <= args.correct_radius)),
            "exclusion_rule": (
                "Exclude a query only when Seq.1 has no keyframe within the "
                "predeclared correctness radius."
            ),
        },
        "alignment": {
            "reference": "valid outdoor GNSS fixes (NavSatFix status >= 0)",
            "model": "independent robust metric SE(2) fit for each traversal",
            "map_transform_yaw_tx_ty": map_transform.tolist(),
            "query_transform_yaw_tx_ty": query_transform.tolist(),
            "map_residual": residual_summary(map_residual),
            "query_residual": residual_summary(query_residual),
            "map_time_association_p95_s": float(np.percentile(map_dt, 95)),
            "query_time_association_p95_s": float(np.percentile(query_dt, 95)),
            "enu_origin_lat_lon_alt": [float(lat0), float(lon0), float(alt0)],
        },
        "overlap": {
            "nearest_map_distance_median": float(np.median(nearest_distance)),
            "nearest_map_distance_p95": float(np.percentile(nearest_distance, 95)),
            "nearest_map_distance_max": float(np.max(nearest_distance)),
        },
    }
    with (derived / "protocol_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
