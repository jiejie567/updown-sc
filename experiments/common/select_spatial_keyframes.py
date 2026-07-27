#!/usr/bin/env python3
"""Select experiment keyframes with a translation-only spatial rule.

The first pose is retained.  A later pose is retained when its 3-D Euclidean
distance from the last retained pose is at least ``spacing_m``.  There is no
time or yaw trigger.  This script is intentionally separate from FAST-LIO's
runtime keyframe policy: it defines only the paper/video evaluation protocol.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_tum(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.shape[1] != 8 or len(poses) == 0:
        raise RuntimeError(f"Expected a non-empty 8-column TUM trajectory: {path}")
    if np.any(np.diff(poses[:, 0]) <= 0.0):
        raise RuntimeError(f"Timestamps are not strictly increasing: {path}")
    return poses


def select_indices(poses: np.ndarray, spacing_m: float) -> np.ndarray:
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    selected = [0]
    last_xyz = poses[0, 1:4]
    for index in range(1, len(poses)):
        if np.linalg.norm(poses[index, 1:4] - last_xyz) >= spacing_m:
            selected.append(index)
            last_xyz = poses[index, 1:4]
    return np.asarray(selected, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tum", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--spacing-m", type=float, default=2.0)
    args = parser.parse_args()

    poses = read_tum(args.input_tum)
    indices = select_indices(poses, args.spacing_m)
    selected = poses[indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(args.output_dir / "selected_indices.txt", indices, fmt="%d")
    np.savetxt(args.output_dir / "selected_poses_tum.txt", selected, fmt="%.9f")
    with (args.output_dir / "selected_keyframes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("selected_index", "source_index", "stamp", "x", "y", "z"))
        for selected_index, (source_index, pose) in enumerate(zip(indices, selected)):
            writer.writerow((
                selected_index,
                int(source_index),
                f"{pose[0]:.9f}",
                f"{pose[1]:.9f}",
                f"{pose[2]:.9f}",
                f"{pose[3]:.9f}",
            ))

    manifest = {
        "format": "translation_only_spatial_keyframes_v1",
        "scope": "experiment_only",
        "runtime_fast_lio_keyframe_policy_changed": False,
        "rule": (
            "retain first pose; then retain a pose when its 3-D Euclidean "
            "translation from the last retained pose is >= spacing_m"
        ),
        "time_trigger": False,
        "yaw_trigger": False,
        "spacing_m": args.spacing_m,
        "source_pose_count": int(len(poses)),
        "selected_pose_count": int(len(indices)),
        "source_tum": str(args.input_tum.resolve()),
    }
    with (args.output_dir / "keyframe_protocol.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
