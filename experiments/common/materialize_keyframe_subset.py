#!/usr/bin/env python3
"""Materialize a selected keyframe session without duplicating PCD payloads."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def load_indices(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64)
    values = np.atleast_1d(values)
    if len(values) == 0 or values[0] < 0 or np.any(np.diff(values) <= 0):
        raise RuntimeError(f"Indices must be non-empty and strictly increasing: {path}")
    return values


def load_tum(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.shape[1] != 8:
        raise RuntimeError(f"Expected 8-column TUM trajectory: {path}")
    return poses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-session", required=True, type=Path)
    parser.add_argument("--indices", required=True, type=Path)
    parser.add_argument("--output-session", required=True, type=Path)
    args = parser.parse_args()

    indices = load_indices(args.indices)
    poses = load_tum(args.source_session / "optimized_poses_tum.txt")
    if indices[-1] >= len(poses):
        raise RuntimeError(f"Selected index {indices[-1]} exceeds {len(poses)} poses")

    source_cloud_dir = args.source_session / "key_point_frame"
    output_cloud_dir = args.output_session / "key_point_frame"
    output_cloud_dir.mkdir(parents=True, exist_ok=True)
    for output_index, source_index in enumerate(indices):
        source = (source_cloud_dir / f"{int(source_index)}.pcd").resolve()
        if not source.is_file():
            raise RuntimeError(f"Missing source cloud: {source}")
        destination = output_cloud_dir / f"{output_index}.pcd"
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        destination.symlink_to(os.path.relpath(source, destination.parent))

    np.savetxt(
        args.output_session / "optimized_poses_tum.txt",
        poses[indices],
        fmt="%.9f",
    )

    gravity_path = args.source_session / "scan_context_gravity.csv"
    with gravity_path.open(newline="") as handle:
        gravity = list(csv.DictReader(handle))
    if len(gravity) != len(poses):
        raise RuntimeError(f"Gravity/pose mismatch: {len(gravity)} != {len(poses)}")
    with (args.output_session / "scan_context_gravity.csv").open("w", newline="") as handle:
        fields = list(gravity[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for output_index, source_index in enumerate(indices):
            row = dict(gravity[int(source_index)])
            row["index"] = output_index
            writer.writerow(row)

    manifest = {
        "format": "keyframe_session_subset_v1",
        "source_session": str(args.source_session.resolve()),
        "selected_indices": str(args.indices.resolve()),
        "source_count": int(len(poses)),
        "selected_count": int(len(indices)),
        "pcd_storage": "relative_symlinks_to_source",
    }
    with (args.output_session / "subset_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
