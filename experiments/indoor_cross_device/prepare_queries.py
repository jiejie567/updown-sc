#!/usr/bin/env python3
"""Prepare one 2 m indoor query track against trusted online localization.

The source mapping sessions retain FAST-LIO's production keyframes.  This
script consumes an already materialized translation-only 2 m subset and
associates each query timestamp with the trusted pose produced by a full
prior-map localization replay.  Those poses are explicitly pseudo-reference,
not independent surveyed ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_tum(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != 8 or len(values) == 0:
        raise RuntimeError(f"Invalid TUM trajectory: {path}")
    if np.any(np.diff(values[:, 0]) <= 0.0):
        raise RuntimeError(f"Non-increasing TUM timestamps: {path}")
    return values


def read_trusted_pose(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    values = np.asarray(
        [
            [
                int(row["stamp_ns"]) * 1e-9,
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    if len(values) == 0:
        raise RuntimeError(f"No trusted poses in {path}")
    order = np.argsort(values[:, 0])
    values = values[order]
    unique = np.r_[True, np.diff(values[:, 0]) > 0.0]
    return values[unique]


def nearest_time(query_stamp: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(reference[:, 0], query_stamp)
    right = np.clip(right, 0, len(reference) - 1)
    left = np.clip(right - 1, 0, len(reference) - 1)
    use_right = (
        np.abs(reference[right, 0] - query_stamp)
        < np.abs(reference[left, 0] - query_stamp)
    )
    index = np.where(use_right, right, left)
    return reference[index], np.abs(reference[index, 0] - query_stamp)


def nearest_map(query_xyz: np.ndarray, map_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    squared = np.sum(
        (query_xyz[:, None, :2] - map_xyz[None, :, :2]) ** 2,
        axis=2,
    )
    index = np.argmin(squared, axis=1)
    distance = np.sqrt(squared[np.arange(len(query_xyz)), index])
    return distance, index


def read_binary_pcd(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header: dict[str, list[str]] = {}
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError(f"Missing DATA line in {path}")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break
        if header["DATA"][0].lower() != "binary":
            raise RuntimeError(f"Expected binary PCD: {path}")
        fields = header["FIELDS"]
        sizes = [int(value) for value in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        dtype_fields = []
        for field, size, kind, count in zip(fields, sizes, types, counts):
            if size != 4 or kind != "F" or count != 1:
                raise RuntimeError(f"Unsupported PCD field layout in {path}")
            dtype_fields.append((field, "<f4"))
        raw = np.fromfile(
            stream,
            dtype=np.dtype(dtype_fields),
            count=int(header["POINTS"][0]),
        )
    intensity = (
        raw["intensity"]
        if "intensity" in fields
        else np.zeros(len(raw), dtype=np.float32)
    )
    points = np.column_stack((raw["x"], raw["y"], raw["z"], intensity))
    return points[np.isfinite(points[:, :3]).all(axis=1)].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-session", required=True, type=Path)
    parser.add_argument("--query-session", required=True, type=Path)
    parser.add_argument("--trusted-pose", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--track-name", required=True)
    parser.add_argument("--correct-radius", type=float, default=2.0)
    parser.add_argument("--max-time-error", type=float, default=0.06)
    args = parser.parse_args()

    map_poses = read_tum(args.map_session / "optimized_poses_tum.txt")
    query_poses = read_tum(args.query_session / "optimized_poses_tum.txt")
    trusted = read_trusted_pose(args.trusted_pose)
    truth, time_error = nearest_time(query_poses[:, 0], trusted)
    associated = time_error <= args.max_time_error

    nearest_distance = np.full(len(query_poses), np.inf, dtype=np.float64)
    nearest_index = np.full(len(query_poses), -1, dtype=np.int64)
    distance, index = nearest_map(truth[:, 1:4], map_poses[:, 1:4])
    nearest_distance[associated] = distance[associated]
    nearest_index[associated] = index[associated]
    eligible = associated & (nearest_distance <= args.correct_radius)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cloud_paths = sorted(
        (args.query_session / "key_point_frame").glob("*.pcd"),
        key=lambda path: int(path.stem),
    )
    if len(cloud_paths) != len(query_poses):
        raise RuntimeError(
            f"Query pose/PCD mismatch: {len(query_poses)} != {len(cloud_paths)}"
        )

    metadata: list[dict[str, object]] = []
    first_stamp = query_poses[0, 0]
    for query_id, cloud_path in enumerate(cloud_paths):
        points = read_binary_pcd(cloud_path)
        filename = f"{query_id:06d}.bin"
        points.tofile(args.output_dir / filename)
        metadata.append(
            {
                "query_id": query_id,
                "window": query_id,
                "track": args.track_name,
                "start_s": f"{query_poses[query_id, 0] - first_stamp:.9f}",
                "header_stamp": f"{query_poses[query_id, 0]:.9f}",
                "points": len(points),
                "file": filename,
                "truth_valid": bool(eligible[query_id]),
                "truth_x": f"{truth[query_id, 1]:.9f}",
                "truth_y": f"{truth[query_id, 2]:.9f}",
                "truth_z": f"{truth[query_id, 3]:.9f}",
                "truth_source": "continuous_prior_map_localization_pseudo_reference",
                "truth_time_error_s": f"{time_error[query_id]:.9f}",
                "nearest_map_index": int(nearest_index[query_id]),
                "nearest_map_distance_m": f"{nearest_distance[query_id]:.9f}",
                "keyframe_rule": "translation_only_2m",
            }
        )
    with (args.output_dir / "metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)

    with (args.query_session / "scan_context_gravity.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        gravity = list(csv.DictReader(stream))
    if len(gravity) != len(query_poses):
        raise RuntimeError("Query pose/gravity count mismatch")
    with (args.output_dir / "gravity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("query_id", "stamp", "up_x", "up_y", "up_z"),
        )
        writer.writeheader()
        for query_id, row in enumerate(gravity):
            writer.writerow(
                {
                    "query_id": query_id,
                    "stamp": row["stamp"],
                    "up_x": row["up_x"],
                    "up_y": row["up_y"],
                    "up_z": row["up_z"],
                }
            )

    manifest = {
        "format": "indoor_cross_device_track_v1",
        "track": args.track_name,
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
            "warning": (
                "Continuous prior-map localization is not independent ground truth."
            ),
            "trusted_pose_count": int(len(trusted)),
            "associated_queries": int(np.sum(associated)),
            "eligible_queries": int(np.sum(eligible)),
            "correct_radius_m": args.correct_radius,
            "max_time_error_s": args.max_time_error,
            "association_time_error_max_s": float(np.max(time_error[associated])),
        },
        "overlap": {
            "nearest_map_distance_median": float(np.median(nearest_distance[eligible])),
            "nearest_map_distance_p95": float(
                np.percentile(nearest_distance[eligible], 95)
            ),
            "nearest_map_distance_max": float(np.max(nearest_distance[eligible])),
        },
    }
    with (args.output_dir / "protocol_manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
