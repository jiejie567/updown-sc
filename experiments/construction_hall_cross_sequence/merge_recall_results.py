#!/usr/bin/env python3
"""Merge verified Construction Hall retrieval results into one traceable table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


ORDER = [
    "SC",
    "SC++ (PC)",
    "SOLiD",
    "M2DP",
    "LiDAR-Iris",
    "RING++",
    "BTC (10 scans, native)",
    "BTC (10 scans, +G)",
    "UpDown-SC",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize_per_query(
    path: Path,
    algorithm: str,
    protocol: str,
    implementation: str,
) -> dict[str, object]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    errors = np.asarray(
        [
            float(row["top1_error_m"])
            for row in rows
            if math.isfinite(float(row["top1_error_m"]))
        ],
        dtype=np.float64,
    )
    times = np.asarray(
        [
            float(row["retrieval_ms"])
            for row in rows
            if math.isfinite(float(row["retrieval_ms"]))
        ],
        dtype=np.float64,
    )
    return {
        "algorithm": algorithm,
        "protocol": protocol,
        "truth_queries": len(rows),
        **{
            f"recall_at_{rank}": float(
                np.mean([int(row[f"recall_at_{rank}"]) for row in rows])
            )
            for rank in (1, 5, 10, 100)
        },
        "top1_error_median": float(np.median(errors)) if len(errors) else math.inf,
        "top1_error_p95": float(np.percentile(errors, 95)) if len(errors) else math.inf,
        "retrieval_ms_median": float(np.median(times)) if len(times) else math.nan,
        "retrieval_ms_p95": float(np.percentile(times, 95)) if len(times) else math.nan,
        "implementation": implementation,
        "source_file": str(path),
    }


def normalize_summary(
    row: dict[str, str],
    source: Path,
    implementation: str,
) -> dict[str, object]:
    output: dict[str, object] = dict(row)
    if str(output["algorithm"]).startswith("RING++"):
        output["algorithm"] = "RING++"
    output["implementation"] = implementation
    output["source_file"] = str(source)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    results = args.experiment_root / "results"

    merged: list[dict[str, object]] = []
    summary_sources = [
        (
            results / "updown_summary.csv",
            "current FAST-LIO C++ Scan Context database/query implementation",
        ),
        (
            results / "single_scan_summary.csv",
            "audited formula-equivalent CPU baseline implementation",
        ),
        (
            results / "m2dp_summary.csv",
            "audited formula-equivalent CPU baseline implementation",
        ),
        (
            results / "ringpp_summary.csv",
            "audited faithful CPU formula port of the released CUDA method",
        ),
    ]
    for path, implementation in summary_sources:
        if not path.exists():
            print(f"Skipping unavailable summary: {path}")
            continue
        for row in read_csv(path):
            merged.append(normalize_summary(row, path, implementation))

    per_query_sources = [
        (
            results / "lidar_iris_per_query.csv",
            "LiDAR-Iris",
            "native_base_link_single_scan",
            "official C++ core with compatibility/path adapter",
        ),
        (
            results / "btc_causal_native/btc_submap_verified_per_query.csv",
            "BTC (10 scans, native)",
            "native_body_causal_10_consecutive_lidar_scans",
            "official C++ core with dataset/result-path adapter",
        ),
        (
            results / "btc_causal_gravity/btc_submap_verified_per_query.csv",
            "BTC (10 scans, +G)",
            "gravity_canonical_causal_10_consecutive_lidar_scans",
            "official C++ core with dataset/result-path adapter",
        ),
    ]
    for path, algorithm, protocol, implementation in per_query_sources:
        if not path.exists():
            print(f"Skipping unavailable per-query result: {path}")
            continue
        merged.append(
            summarize_per_query(path, algorithm, protocol, implementation)
        )

    order = {name: index for index, name in enumerate(ORDER)}
    merged.sort(key=lambda row: order.get(str(row["algorithm"]), len(order)))
    if not merged:
        raise RuntimeError("No result files were found")

    fields = [
        "algorithm",
        "protocol",
        "truth_queries",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_100",
        "top1_error_median",
        "top1_error_p95",
        "retrieval_ms_median",
        "retrieval_ms_p95",
        "implementation",
        "source_file",
    ]
    output = results / "recall_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)

    for row in merged:
        print(
            f"{row['algorithm']:<22} n={int(row['truth_queries']):>3} "
            f"R@1={100 * float(row['recall_at_1']):5.1f}% "
            f"R@5={100 * float(row['recall_at_5']):5.1f}% "
            f"R@10={100 * float(row['recall_at_10']):5.1f}% "
            f"R@100={100 * float(row['recall_at_100']):5.1f}%"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
