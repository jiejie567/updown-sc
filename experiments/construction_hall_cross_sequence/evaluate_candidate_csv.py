#!/usr/bin/env python3
"""Score a Top-K candidate CSV against prepared cross-sequence truth."""

import argparse
import csv
from pathlib import Path

import numpy as np


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-per-query", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--protocol", default="proposed_gravity_canonicalized_single_scan")
    parser.add_argument("--correct-radius", type=float, default=5.0)
    args = parser.parse_args()

    with args.metadata.open(newline="") as handle:
        metadata = {
            int(row["query_id"]): row
            for row in csv.DictReader(handle)
            if truthy(row["truth_valid"])
        }
    grouped = {}
    with args.candidates.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(int(row["query_index"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["candidate_rank"]))

    output = []
    for query_index, truth in metadata.items():
        candidates = grouped.get(query_index, [])
        if candidates:
            xy = np.asarray([
                [float(row["candidate_x"]), float(row["candidate_y"])]
                for row in candidates
            ])
            errors = np.linalg.norm(
                xy - np.asarray([float(truth["truth_x"]), float(truth["truth_y"])]),
                axis=1,
            )
            matches = np.flatnonzero(errors <= args.correct_radius)
            rank = int(matches[0] + 1) if len(matches) else None
            first = candidates[0]
            top1_index = int(first["candidate_index"])
            top1_x, top1_y = xy[0]
            top1_error = float(errors[0])
            retrieval_ms = float(first["retrieval_ms"])
        else:
            rank = None
            top1_index = -1
            top1_x = top1_y = float("nan")
            top1_error = float("inf")
            retrieval_ms = float("nan")
        output.append({
            "algorithm": args.algorithm,
            "query_id": query_index,
            "window": truth["window"],
            "start_s": truth["start_s"],
            "truth_x": truth["truth_x"],
            "truth_y": truth["truth_y"],
            "top1_index": top1_index,
            "top1_x": top1_x,
            "top1_y": top1_y,
            "top1_error_m": top1_error,
            "first_correct_rank": rank if rank is not None else "",
            "recall_at_1": rank is not None and rank <= 1,
            "recall_at_5": rank is not None and rank <= 5,
            "recall_at_10": rank is not None and rank <= 10,
            "recall_at_100": rank is not None and rank <= 100,
            "retrieval_ms": retrieval_ms,
        })

    finite_times = np.asarray([
        row["retrieval_ms"] for row in output if np.isfinite(row["retrieval_ms"])
    ])
    finite_errors = np.asarray([
        row["top1_error_m"] for row in output if np.isfinite(row["top1_error_m"])
    ])
    summary = {
        "algorithm": args.algorithm,
        "protocol": args.protocol,
        "truth_queries": len(output),
        **{
            key: float(np.mean([row[key] for row in output]))
            for key in ("recall_at_1", "recall_at_5", "recall_at_10", "recall_at_100")
        },
        "top1_error_median": float(np.median(finite_errors)),
        "top1_error_p95": float(np.percentile(finite_errors, 95)),
        "retrieval_ms_median": float(np.median(finite_times)),
        "retrieval_ms_p95": float(np.percentile(finite_times, 95)),
    }
    args.output_per_query.parent.mkdir(parents=True, exist_ok=True)
    with args.output_per_query.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    with args.output_summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(summary)


if __name__ == "__main__":
    main()
