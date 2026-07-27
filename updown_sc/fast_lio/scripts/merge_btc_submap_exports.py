#!/usr/bin/env python3
"""Merge time-overlapping BTC export parts without duplicating point data."""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


GENERATED_NAMES = ("submaps", "metadata.csv", "poses_tum.txt", "manifest.json")
COMPATIBLE_KEYS = (
    "format",
    "scans_per_submap",
    "submap_stride_scans",
    "minimum_radius_m",
    "maximum_radius_m",
    "voxel_size_m",
)


def prepare_output(output: Path, overwrite: bool) -> None:
    existing = [output / name for name in GENERATED_NAMES if (output / name).exists()]
    if existing and not overwrite:
        raise RuntimeError(f"output exists under {output}; use --overwrite")
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    (output / "submaps").mkdir(parents=True, exist_ok=True)


def load_part(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"incomplete BTC export: {root}")
    with (root / "metadata.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    if not rows or fields is None:
        raise RuntimeError(f"empty BTC export: {root}")
    for row in rows:
        row["_root"] = root
        row["_start"] = float(row["start_stamp"])
        row["_end"] = float(row["end_stamp"])
    return manifest, fields, rows


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("input", nargs="+", type=Path)
    arguments = parser.parse_args()

    parts = [load_part(path.expanduser().resolve()) for path in arguments.input]
    reference = parts[0][0]
    for manifest, _, _ in parts[1:]:
        for key in COMPATIBLE_KEYS:
            if manifest.get(key) != reference.get(key):
                raise RuntimeError(f"incompatible export parameter {key}")
    fields = parts[0][1]
    if any(part_fields != fields for _, part_fields, _ in parts[1:]):
        raise RuntimeError("metadata columns differ between exports")

    candidates = sorted(
        (row for _, _, rows in parts for row in rows), key=lambda row: row["_start"]
    )
    selected = []
    last_end = float("-inf")
    for row in candidates:
        # Parts intentionally overlap. Keep the first complete non-overlapping
        # submap at each time so no scan is represented twice.
        if row["_start"] <= last_end + 1.0e-6:
            continue
        selected.append(row)
        last_end = row["_end"]

    output = arguments.output_dir.expanduser().resolve()
    prepare_output(output, arguments.overwrite)
    with (output / "metadata.csv").open(
        "w", newline="", encoding="utf-8", buffering=1
    ) as metadata_stream, (output / "poses_tum.txt").open(
        "w", encoding="utf-8", buffering=1
    ) as pose_stream:
        writer = csv.DictWriter(metadata_stream, fieldnames=fields)
        writer.writeheader()
        for new_id, source_row in enumerate(selected):
            relative = Path("submaps") / f"{new_id:06d}.bin"
            source = source_row["_root"] / source_row["file"]
            link_or_copy(source, output / relative)
            row = {field: source_row[field] for field in fields}
            row["submap_id"] = str(new_id)
            row["file"] = str(relative)
            writer.writerow(row)
            pose_stream.write(
                "{reference_stamp} {x} {y} {z} {qx} {qy} {qz} {qw}\n".format(**row)
            )

    gaps = [
        float(current["start_stamp"]) - float(previous["end_stamp"])
        for previous, current in zip(selected, selected[1:])
    ]
    excluded = {
        "submaps", "received_odometry", "received_clouds", "synchronized_clouds",
        "dropped_unsynchronized", "dropped_gap_frames", "status",
    }
    manifest = {key: value for key, value in reference.items() if key not in excluded}
    manifest.update({
        "status": "complete",
        "submaps": len(selected),
        "source_exports": [str(path.expanduser().resolve()) for path in arguments.input],
        "source_submaps": len(candidates),
        "overlapping_submaps_removed": len(candidates) - len(selected),
        "maximum_inter_submap_gap_s": max(gaps, default=0.0),
    })
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Merged {len(selected)} BTC submaps into {output}; "
        f"removed {len(candidates) - len(selected)} overlapping submaps"
    )


if __name__ == "__main__":
    main()
