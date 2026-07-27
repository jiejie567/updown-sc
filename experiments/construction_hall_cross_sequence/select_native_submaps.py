#!/usr/bin/env python3
"""Select native accumulated submaps at predeclared experiment centers.

The source submaps retain a method's native consecutive-scan construction.
Only their database/query centers are subsampled to the same spatial-keyframe
timestamps used by the single-scan retrieval experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--centers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-time-error", type=float, default=0.75)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="hard-link selected point files into output/submaps",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="skip experiment centers without a native submap inside the time tolerance",
    )
    args = parser.parse_args()

    source = read_rows(args.source / "metadata.csv")
    poses = np.loadtxt(args.centers, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    source_stamps = np.asarray(
        [float(row["reference_stamp"]) for row in source], dtype=np.float64
    )
    args.output.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, str]] = []
    used: set[int] = set()
    for center_id, stamp in enumerate(poses[:, 0]):
        source_index = int(np.argmin(np.abs(source_stamps - stamp)))
        error = abs(source_stamps[source_index] - stamp)
        if error > args.max_time_error:
            if args.skip_missing:
                continue
            raise RuntimeError(
                f"Center {center_id} has no native submap within "
                f"{args.max_time_error:g} s (nearest={error:.6f} s)"
            )
        if source_index in used:
            raise RuntimeError(f"Native submap {source_index} selected twice")
        used.add(source_index)
        row = dict(source[source_index])
        row["submap_id"] = str(center_id)
        source_file = (args.source / row["file"]).resolve()
        if args.materialize:
            relative = Path("submaps") / f"{center_id:06d}.bin"
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination.unlink()
            os.link(source_file, destination)
            row["file"] = str(relative)
        else:
            row["file"] = str(source_file)
        selected.append(row)

    with (args.output / "metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    manifest = {
        "format": "native_submaps_at_experiment_centers_v1",
        "source": str(args.source.resolve()),
        "center_poses": str(args.centers.resolve()),
        "selection": "nearest native submap reference timestamp",
        "maximum_time_error_s": args.max_time_error,
        "requested_centers": len(poses),
        "selected_submaps": len(selected),
        "skipped_centers": len(poses) - len(selected),
        "point_files": (
            "hard-linked into output/submaps"
            if args.materialize
            else "referenced in place; no cloud duplication"
        ),
    }
    with (args.output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
