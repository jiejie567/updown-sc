#!/usr/bin/env python3
"""Gravity-canonicalize causal BTC submaps while preserving heading.

Input clouds are expressed in the current scan's native body frame.  The
reference pose quaternion supplies roll/pitch.  Each output cloud is rotated
to a level frame whose yaw matches the reference body yaw, so this operation
does not globally align headings between sessions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def quaternion_to_rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    q = np.asarray([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata_path = args.input / "metadata.csv"
    with metadata_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No submaps in {metadata_path}")
    required = {"file", "qx", "qy", "qz", "qw"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Metadata lacks reference pose fields: {sorted(missing)}")

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output is not empty: {args.output}; pass --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    submap_dir = args.output / "submaps"
    submap_dir.mkdir(exist_ok=True)

    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        source = args.input / row["file"]
        points = np.fromfile(source, dtype="<f4")
        if points.size % 4:
            raise RuntimeError(f"Invalid XYZI file: {source}")
        points = points.reshape((-1, 4))
        body = quaternion_to_rotation(
            float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])
        )
        yaw = math.atan2(body[1, 0], body[0, 0])
        cy, sy = math.cos(yaw), math.sin(yaw)
        yaw_rotation = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        native_to_gravity = yaw_rotation.T @ body
        output_points = points.copy()
        output_points[:, :3] = (
            points[:, :3].astype(np.float64) @ native_to_gravity.T
        ).astype(np.float32)
        relative = Path("submaps") / f"{index:06d}.bin"
        output_points.astype("<f4", copy=False).tofile(args.output / relative)
        output_row = dict(row)
        output_row["file"] = str(relative)
        output_row["descriptor_frame"] = "gravity_canonical_heading_preserved"
        output_rows.append(output_row)

    with (args.output / "metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    input_manifest_path = args.input / "manifest.json"
    input_manifest = (
        json.loads(input_manifest_path.read_text(encoding="utf-8"))
        if input_manifest_path.exists() else {}
    )
    manifest = {
        "format": "btc_gravity_canonical_submaps_v1",
        "source": str(args.input.resolve()),
        "source_manifest": input_manifest,
        "submaps": len(output_rows),
        "gravity_canonicalized": True,
        "yaw_preserved": True,
        "operation": "remove reference roll/pitch after causal scan registration",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_manifest"}, indent=2))


if __name__ == "__main__":
    main()
