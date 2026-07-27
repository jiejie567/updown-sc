#!/usr/bin/env python3
"""Build the two indoor Table-III columns from auditable result files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


METHODS = (
    "SC",
    "SC++ (PC)",
    "SOLiD",
    "M2DP",
    "LiDAR Iris",
    "RING++",
    "BTC (10 scans)",
    "UpDown-SC",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def one_row(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {path}, got {len(rows)}")
    return rows[0]


def summarize_per_query(path: Path) -> dict[str, float]:
    rows = read_rows(path)
    if not rows:
        raise RuntimeError(f"No per-query rows in {path}")
    return {
        "truth_queries": len(rows),
        "recall_at_1": float(
            np.mean(
                [
                    row["recall_at_1"].strip().lower()
                    in {"1", "true", "yes"}
                    for row in rows
                ]
            )
        ),
        "recall_at_5": float(
            np.mean(
                [
                    row["recall_at_5"].strip().lower()
                    in {"1", "true", "yes"}
                    for row in rows
                ]
            )
        ),
    }


def collect_track(root: Path, track: str) -> list[dict[str, object]]:
    source = root / "table3_2m" / track
    gravity_rows = {
        row["algorithm"].removesuffix(" + G"): row
        for row in read_rows(source / "gravity/summary.csv")
    }
    ring = one_row(source / "ringpp/results/ringpp_summary.csv")
    updown = one_row(root / f"results/{track}/updown_summary.csv")
    iris = summarize_per_query(source / "iris/per_query.csv")
    btc = summarize_per_query(
        root
        / f"results/btc_causal/{track}_gravity/btc_submap_verified_per_query.csv"
    )

    results: dict[str, dict[str, object]] = {
        **gravity_rows,
        "LiDAR Iris": iris,
        "RING++": ring,
        "BTC (10 scans)": btc,
        "UpDown-SC": updown,
    }
    output: list[dict[str, object]] = []
    for method in METHODS:
        row = results[method]
        output.append(
            {
                "track": track,
                "protocol": (
                    "gravity_canonicalized_causal_10_scan_submap"
                    if method == "BTC (10 scans)"
                    else "gravity_canonicalized_single_scan"
                ),
                "correct_radius_m": 2.0,
                "algorithm": method,
                "truth_queries": int(row["truth_queries"]),
                "recall_at_1": float(row["recall_at_1"]),
                "recall_at_5": float(row["recall_at_5"]),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    args = parser.parse_args()

    rows = [
        *collect_track(args.experiment_root, "handle2"),
        *collect_track(args.experiment_root, "vehicle1"),
    ]
    output = args.experiment_root / "results/table3_retrieval_columns.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['track']:8s} {row['algorithm']:12s} "
            f"{100.0 * row['recall_at_1']:.1f}/"
            f"{100.0 * row['recall_at_5']:.1f}"
        )


if __name__ == "__main__":
    main()
