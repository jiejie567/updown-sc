#!/usr/bin/env python3
"""Prepare the private two-bag 2 m retrieval split.

The query clouds/descriptors are produced by a mapping-mode replay with the
experiment-only 2 m keyframe override.  Map-frame query positions come from a
continuous prior-map localization replay and are therefore explicitly marked
as pseudo-reference rather than independent ground truth.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_tum(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.shape[1] != 8 or len(poses) == 0:
        raise RuntimeError(f"Invalid TUM trajectory: {path}")
    return poses


def load_odometry(path: Path) -> np.ndarray:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = np.asarray(
        [[float(row[name]) for name in ("stamp", "x", "y", "z")] for row in rows],
        dtype=np.float64,
    )
    if len(output) == 0 or np.any(np.diff(output[:, 0]) <= 0.0):
        raise RuntimeError(f"Invalid odometry CSV: {path}")
    return output


def nearest_rows(stamps: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(reference[:, 0], stamps)
    right = np.clip(right, 0, len(reference) - 1)
    left = np.clip(right - 1, 0, len(reference) - 1)
    choose_right = np.abs(reference[right, 0] - stamps) < np.abs(
        reference[left, 0] - stamps)
    indices = np.where(choose_right, right, left)
    return reference[indices], np.abs(reference[indices, 0] - stamps)


def nearest_map(query_xyz: np.ndarray, map_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.empty(len(query_xyz), dtype=np.float64)
    index = np.empty(len(query_xyz), dtype=np.int64)
    for start in range(0, len(query_xyz), 128):
        chunk = query_xyz[start:start + 128]
        squared = np.sum((chunk[:, None, :] - map_xyz[None, :, :]) ** 2, axis=2)
        index[start:start + len(chunk)] = np.argmin(squared, axis=1)
        distance[start:start + len(chunk)] = np.sqrt(np.min(squared, axis=1))
    return distance, index


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
        dtype = np.dtype([(field, "<f4") for field in header["FIELDS"]])
        raw = np.fromfile(handle, dtype=dtype, count=int(header["POINTS"][0]))
    intensity = (
        raw["intensity"]
        if "intensity" in header["FIELDS"]
        else np.zeros(len(raw), dtype=np.float32)
    )
    points = np.column_stack((raw["x"], raw["y"], raw["z"], intensity))
    return points[np.isfinite(points[:, :3]).all(axis=1)].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--pseudo-reference", required=True, type=Path)
    parser.add_argument("--correct-radius", type=float, default=2.0)
    parser.add_argument("--max-time-error", type=float, default=0.06)
    args = parser.parse_args()

    map_session = args.experiment_root / "map/session"
    query_session = args.experiment_root / "query/session"
    map_poses = load_tum(map_session / "optimized_poses_tum.txt")
    query_poses = load_tum(query_session / "optimized_poses_tum.txt")
    reference = load_odometry(args.pseudo_reference)
    truth, time_error = nearest_rows(query_poses[:, 0], reference)
    if np.max(time_error) > args.max_time_error:
        raise RuntimeError(
            f"Pseudo-reference association exceeds tolerance: {np.max(time_error):.6f} s")

    nearest_distance, nearest_index = nearest_map(truth[:, 1:4], map_poses[:, 1:4])
    output = args.experiment_root / "derived/queries"
    output.mkdir(parents=True, exist_ok=True)
    source_clouds = sorted(
        (query_session / "key_point_frame").glob("*.pcd"),
        key=lambda path: int(path.stem),
    )
    if len(source_clouds) != len(query_poses):
        raise RuntimeError(
            f"Query pose/PCD mismatch: {len(query_poses)} != {len(source_clouds)}")

    metadata = []
    for query_id, source in enumerate(source_clouds):
        points = read_binary_pcd(source)
        filename = f"{query_id:06d}.bin"
        points.tofile(output / filename)
        metadata.append({
            "query_id": query_id,
            "window": query_id,
            "start_s": f"{query_poses[query_id, 0] - query_poses[0, 0]:.9f}",
            "header_stamp": f"{query_poses[query_id, 0]:.9f}",
            "points": len(points),
            "file": filename,
            "truth_valid": bool(nearest_distance[query_id] <= args.correct_radius),
            "truth_x": f"{truth[query_id, 1]:.9f}",
            "truth_y": f"{truth[query_id, 2]:.9f}",
            "truth_z": f"{truth[query_id, 3]:.9f}",
            "truth_source": "continuous_prior_map_localization_pseudo_reference",
            "truth_time_error_s": f"{time_error[query_id]:.9f}",
            "nearest_map_index": int(nearest_index[query_id]),
            "nearest_map_distance_m": f"{nearest_distance[query_id]:.9f}",
            "keyframe_rule": "translation_only_2m",
        })
    with (output / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)

    with (query_session / "scan_context_gravity.csv").open(newline="") as handle:
        gravity = list(csv.DictReader(handle))
    if len(gravity) != len(query_poses):
        raise RuntimeError("Query gravity/pose mismatch")
    with (output / "gravity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("query_id", "stamp", "up_x", "up_y", "up_z"))
        writer.writeheader()
        for query_id, row in enumerate(gravity):
            writer.writerow({
                "query_id": query_id,
                "stamp": row["stamp"],
                "up_x": row["up_x"],
                "up_y": row["up_y"],
                "up_z": row["up_z"],
            })

    manifest = {
        "format": "private_two_bag_retrieval_v2",
        "keyframe_protocol": {
            "scope": "experiment_only",
            "spacing_m": 2.0,
            "translation_dimension": "3-D Euclidean",
            "time_trigger": False,
            "yaw_trigger": False,
            "map_count": int(len(map_poses)),
            "query_count": int(len(query_poses)),
            "production_fast_lio_policy_changed": False,
        },
        "truth": {
            "type": "pseudo-reference",
            "source": str(args.pseudo_reference.resolve()),
            "warning": (
                "Continuous prior-map localization is not independent ground truth; "
                "this split is a private regression/video experiment only."
            ),
            "association_time_error_max_s": float(np.max(time_error)),
            "correct_radius_m": args.correct_radius,
            "eligible_queries": int(np.sum(nearest_distance <= args.correct_radius)),
        },
        "overlap": {
            "nearest_map_distance_median": float(np.median(nearest_distance)),
            "nearest_map_distance_p95": float(np.percentile(nearest_distance, 95)),
            "nearest_map_distance_max": float(np.max(nearest_distance)),
        },
    }
    with (args.experiment_root / "derived/protocol_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
