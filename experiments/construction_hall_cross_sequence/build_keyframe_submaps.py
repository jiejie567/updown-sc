#!/usr/bin/env python3
"""Build causal 10-keyframe submaps for accumulated-keyframe baselines.

The input frames are the experiment's translation-only 2 m keyframes, not
fixed-time scans.  Each output is registered into the newest keyframe frame
and contains that frame plus the preceding nine keyframes.  STD instead uses
``build_std_raw_submaps.py`` because its released ``sub_frame_num`` denotes
consecutive LiDAR scans.
"""

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


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
        dtype = np.dtype([(field, "<f4") for field in fields])
        raw = np.fromfile(handle, dtype=dtype, count=int(header["POINTS"][0]))
    intensity = raw["intensity"] if "intensity" in fields else np.zeros(len(raw), np.float32)
    points = np.column_stack((raw["x"], raw["y"], raw["z"], intensity))
    return points[np.isfinite(points[:, :3]).all(axis=1)].astype(np.float32, copy=False)


def voxel_centroids(points: np.ndarray, leaf: float) -> np.ndarray:
    if not len(points):
        return points
    cells = np.floor(points[:, :3] / leaf).astype(np.int32)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, 4), dtype=np.float64)
    np.add.at(sums, inverse, points)
    counts = np.bincount(inverse)
    return (sums / counts[:, None]).astype(np.float32)


def pose_matrices(tum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = tum[:, 1:4]
    rotation = Rotation.from_quat(tum[:, 4:8]).as_matrix()
    return translation, rotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--keyframes-per-submap", type=int, default=10)
    parser.add_argument("--voxel", type=float, default=0.1)
    parser.add_argument("--min-radius", type=float, default=0.3)
    parser.add_argument("--max-radius", type=float, default=30.0)
    parser.add_argument(
        "--reference-orientation",
        choices=("body", "gravity"),
        default="body",
        help=(
            "Use the newest keyframe body axes, or retain the gravity-aligned "
            "odometry axes. STD requires gravity because its plane/corner "
            "extraction runs before the rigid-invariant triangle matching."
        ),
    )
    args = parser.parse_args()

    poses = np.loadtxt(args.session / "optimized_poses_tum.txt", dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    paths = sorted(
        (args.session / "key_point_frame").glob("*.pcd"),
        key=lambda path: int(path.stem),
    )
    if len(paths) != len(poses):
        raise RuntimeError(f"Pose/PCD mismatch: {len(poses)} != {len(paths)}")
    translation, rotation = pose_matrices(poses)

    submap_dir = args.output / "submaps"
    submap_dir.mkdir(parents=True, exist_ok=True)
    history = deque(maxlen=args.keyframes_per_submap)
    metadata = []
    for index, path in enumerate(paths):
        history.append((index, read_binary_pcd(path)))
        if len(history) < args.keyframes_per_submap:
            continue
        reference_translation = translation[index]
        reference_rotation = rotation[index]
        registered = []
        raw_points = 0
        for source_index, points in history:
            raw_points += len(points)
            world = points[:, :3] @ rotation[source_index].T + translation[source_index]
            xyz = world - reference_translation
            if args.reference_orientation == "body":
                xyz = xyz @ reference_rotation
            registered.append(np.column_stack((xyz, points[:, 3])))
        cloud = np.concatenate(registered).astype(np.float32, copy=False)
        radius = np.hypot(cloud[:, 0], cloud[:, 1])
        cloud = cloud[
            np.isfinite(cloud[:, :3]).all(axis=1) &
            (radius >= args.min_radius) &
            (radius <= args.max_radius)
        ]
        cloud = voxel_centroids(cloud, args.voxel)
        filename = f"submaps/{index:06d}.bin"
        cloud.tofile(args.output / filename)
        start_index = index - args.keyframes_per_submap + 1
        metadata.append({
            "submap_id": index,
            "reference_stamp": f"{poses[index, 0]:.9f}",
            "start_stamp": f"{poses[start_index, 0]:.9f}",
            "end_stamp": f"{poses[index, 0]:.9f}",
            "scans": args.keyframes_per_submap,
            "raw_points": raw_points,
            "output_points": len(cloud),
            "file": filename,
            "world_frame": "map",
            "body_frame": "base_link",
            "x": poses[index, 1],
            "y": poses[index, 2],
            "z": poses[index, 3],
            "qx": poses[index, 4],
            "qy": poses[index, 5],
            "qz": poses[index, 6],
            "qw": poses[index, 7],
            "keyframe_start_index": start_index,
            "keyframe_end_index": index,
        })
        if len(metadata) % 100 == 0 or index + 1 == len(paths):
            print(f"Submaps {len(metadata)}/{len(paths) - args.keyframes_per_submap + 1}",
                  flush=True)

    with (args.output / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    manifest = {
        "format": "registered_spatial_keyframe_submaps_v1",
        "keyframe_source": "experiment-only translation-only 2 m keyframes",
        "experimental_spacing_m": 2.0,
        "time_based_sampling": False,
        "keyframes_per_submap": args.keyframes_per_submap,
        "submap_stride_keyframes": 1,
        "causal": True,
        "reference": "newest keyframe",
        "reference_orientation": args.reference_orientation,
        "voxel_size_m": args.voxel,
        "minimum_radius_m": args.min_radius,
        "maximum_radius_m": args.max_radius,
        "input_keyframes": len(paths),
        "output_submaps": len(metadata),
        "excluded_prefix_keyframes": args.keyframes_per_submap - 1,
    }
    with (args.output / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
