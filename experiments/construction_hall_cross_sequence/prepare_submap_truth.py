#!/usr/bin/env python3
"""Create one-to-one truth rows for spatial-keyframe submap benchmarks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-metadata", type=Path, required=True)
    parser.add_argument("--submap-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fields, queries = read_rows(args.query_metadata)
    _, submaps = read_rows(args.submap_metadata)
    query_by_id = {int(row["query_id"]): row for row in queries}

    selected: list[dict[str, str]] = []
    for submap in submaps:
        keyframe_id = int(submap["keyframe_end_index"])
        query = query_by_id.get(keyframe_id)
        if query is None:
            raise RuntimeError(f"Missing truth for submap keyframe {keyframe_id}")
        stamp_error = abs(
            float(query["header_stamp"]) - float(submap["reference_stamp"])
        )
        if stamp_error > 1e-6:
            raise RuntimeError(
                f"Submap {keyframe_id} stamp mismatch: {stamp_error:.9f} s"
            )
        selected.append(query)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    print(
        f"Wrote {len(selected)} one-to-one spatial-keyframe submap truth rows "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
