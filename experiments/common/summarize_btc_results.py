#!/usr/bin/env python3
"""Summarize audited BTC per-query CSVs into one traceable table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(path: Path, experiment: str, variant: str) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    times = np.asarray([float(row["retrieval_ms"]) for row in rows])
    output: dict[str, object] = {
        "experiment": experiment,
        "variant": variant,
        "queries": len(rows),
        "empty_candidates": sum(int(row["candidates_returned"]) == 0 for row in rows),
    }
    for rank in (1, 5, 10, 100):
        output[f"recall_at_{rank}"] = float(
            np.mean([int(row[f"recall_at_{rank}"]) for row in rows])
        )
    output["retrieval_ms_median"] = float(np.median(times))
    output["retrieval_ms_p95"] = float(np.percentile(times, 95))
    output["source_file"] = str(path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--newer-college-root",
        type=Path,
        help="Optional Newer College BTC experiment root.",
    )
    args = parser.parse_args()
    specifications: list[tuple[str, str, Path]] = [
        ("IH pilot", "Native", args.root / "ih/results/native"),
        ("IH pilot", "+G", args.root / "ih/results/gravity"),
        ("CH public", "Native", args.root / "ch/results/native_cropped"),
        ("CH public", "+G", args.root / "ch/results/gravity_cropped"),
        ("H1->H2", "Native", args.root / "indoor/results/handle2_native"),
        ("H1->H2", "+G", args.root / "indoor/results/handle2_gravity"),
        ("H1->V1", "Native", args.root / "indoor/results/vehicle1_native"),
        ("H1->V1", "+G", args.root / "indoor/results/vehicle1_gravity"),
    ]
    if args.newer_college_root is not None:
        specifications.extend([
            (
                "NC outdoor",
                "Native",
                args.newer_college_root / "results/native",
            ),
            (
                "NC outdoor",
                "+G",
                args.newer_college_root / "results/gravity",
            ),
        ])
    rows = []
    for experiment, variant, result_root in specifications:
        source = result_root / "btc_submap_verified_per_query.csv"
        if source.exists():
            rows.append(summarize(source, experiment, variant))
    if not rows:
        raise RuntimeError(f"No BTC result files found below {args.root}")

    result_dir = args.root / "results"
    result_dir.mkdir(exist_ok=True)
    fields = list(rows[0])
    with (result_dir / "btc_completion_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (result_dir / "btc_completion_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(
            f"{row['experiment']:<10} {row['variant']:<6} "
            f"n={row['queries']:>3} empty={row['empty_candidates']:>3} "
            f"R@1/5={100*row['recall_at_1']:.1f}/{100*row['recall_at_5']:.1f} "
            f"R@10={100*row['recall_at_10']:.1f}"
        )


if __name__ == "__main__":
    main()
