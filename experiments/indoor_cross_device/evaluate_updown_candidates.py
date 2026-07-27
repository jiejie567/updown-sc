#!/usr/bin/env python3
"""Score UpDown-SC retrieval and its conditioned vertical estimate."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    parser.add_argument("--correct-radius", type=float, default=2.0)
    parser.add_argument(
        "--vertical-estimate-offset",
        type=float,
        default=0.0,
        help="Known platform-frame offset added to the descriptor estimate before scoring.",
    )
    args = parser.parse_args()

    with args.metadata.open(newline="", encoding="utf-8") as stream:
        metadata = [
            row for row in csv.DictReader(stream) if truthy(row["truth_valid"])
        ]
    grouped: dict[int, list[dict[str, str]]] = {}
    with args.candidates.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault(int(row["query_index"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["candidate_rank"]))

    output: list[dict[str, object]] = []
    vertical_errors: list[float] = []
    vertical_estimates: list[float] = []
    vertical_references: list[float] = []
    for truth in metadata:
        query_id = int(truth["query_id"])
        candidates = grouped.get(query_id, [])
        if not candidates:
            output.append(
                {
                    "algorithm": "UpDown-SC",
                    "query_id": query_id,
                    "track": truth["track"],
                    "start_s": truth["start_s"],
                    "truth_x": truth["truth_x"],
                    "truth_y": truth["truth_y"],
                    "truth_z": truth["truth_z"],
                    "top1_index": -1,
                    "top1_x": math.nan,
                    "top1_y": math.nan,
                    "top1_error_m": math.inf,
                    "first_correct_rank": "",
                    "recall_at_1": False,
                    "recall_at_5": False,
                    "conditioned_candidate_index": "",
                    "conditioned_candidate_rank": "",
                    "vertical_shift_estimate_m": math.nan,
                    "vertical_shift_reference_m": math.nan,
                    "vertical_shift_error_m": math.nan,
                    "retrieval_ms": math.nan,
                }
            )
            continue
        truth_xyz = np.asarray(
            [
                float(truth["truth_x"]),
                float(truth["truth_y"]),
                float(truth["truth_z"]),
            ]
        )
        candidate_xyz = np.asarray(
            [
                [
                    float(row["candidate_x"]),
                    float(row["candidate_y"]),
                    float(row["candidate_z"]),
                ]
                for row in candidates
            ]
        )
        xy_error = np.linalg.norm(candidate_xyz[:, :2] - truth_xyz[:2], axis=1)
        correct = np.flatnonzero(xy_error <= args.correct_radius)
        first_correct_rank = int(correct[0] + 1) if len(correct) else None

        # Vertical correction is meaningful only after a place/yaw hypothesis
        # exists.  Use the highest-ranked spatially correct hypothesis, matching
        # the production order (retrieval first, Delta-z second).  A query with
        # no correct descriptor hypothesis is retained for recall but excluded
        # from the conditional vertical-error aggregate.
        conditioned_index = int(correct[0]) if len(correct) else None
        if conditioned_index is not None:
            conditioned = candidates[conditioned_index]
            conditioned_candidate_index: int | str = int(
                conditioned["candidate_index"]
            )
            estimate = (
                float(conditioned["vertical_shift"])
                + args.vertical_estimate_offset
            )
            reference = truth_xyz[2] - float(conditioned["candidate_z"])
            error = estimate - reference
            vertical_estimates.append(estimate)
            vertical_references.append(reference)
            vertical_errors.append(error)
        else:
            conditioned_candidate_index = ""
            estimate = reference = error = math.nan

        first = candidates[0]
        output.append(
            {
                "algorithm": "UpDown-SC",
                "query_id": query_id,
                "track": truth["track"],
                "start_s": truth["start_s"],
                "truth_x": truth["truth_x"],
                "truth_y": truth["truth_y"],
                "truth_z": truth["truth_z"],
                "top1_index": int(first["candidate_index"]),
                "top1_x": first["candidate_x"],
                "top1_y": first["candidate_y"],
                "top1_error_m": float(xy_error[0]),
                "first_correct_rank": (
                    first_correct_rank if first_correct_rank is not None else ""
                ),
                "recall_at_1": first_correct_rank is not None
                and first_correct_rank <= 1,
                "recall_at_5": first_correct_rank is not None
                and first_correct_rank <= 5,
                "conditioned_candidate_index": conditioned_candidate_index,
                "conditioned_candidate_rank": (
                    conditioned_index + 1 if conditioned_index is not None else ""
                ),
                "vertical_shift_estimate_m": estimate,
                "vertical_shift_reference_m": reference,
                "vertical_shift_error_m": error,
                "retrieval_ms": float(first["retrieval_ms"]),
            }
        )

    errors = np.asarray(vertical_errors, dtype=np.float64)
    estimates = np.asarray(vertical_estimates, dtype=np.float64)
    references = np.asarray(vertical_references, dtype=np.float64)
    if len(errors) == 0:
        raise RuntimeError("No spatially correct hypothesis supports vertical evaluation")
    timing_path = Path(str(args.candidates) + ".timing.csv")
    eligible_query_ids = {int(row["query_id"]) for row in metadata}
    candidate_covered_queries = sum(
        query_id in eligible_query_ids for query_id in grouped
    )
    if timing_path.exists():
        with timing_path.open(newline="", encoding="utf-8") as stream:
            timing_rows = [
                row
                for row in csv.DictReader(stream)
                if int(row["query_index"]) in eligible_query_ids
            ]
        times = np.asarray(
            [float(row["retrieval_ms"]) for row in timing_rows],
            dtype=np.float64,
        )
        candidate_covered_queries = sum(
            int(row["candidate_count"]) > 0 for row in timing_rows
        )
    else:
        times = np.asarray(
            [
                float(row["retrieval_ms"])
                for row in output
                if math.isfinite(float(row["retrieval_ms"]))
            ],
            dtype=np.float64,
        )
    summary = {
        "algorithm": "UpDown-SC",
        "track": metadata[0]["track"],
        "truth_queries": len(output),
        "recall_at_1": float(np.mean([bool(row["recall_at_1"]) for row in output])),
        "recall_at_5": float(np.mean([bool(row["recall_at_5"]) for row in output])),
        "candidate_covered_queries": int(candidate_covered_queries),
        "candidate_coverage_fraction": float(
            candidate_covered_queries / len(output)
        ),
        "vertical_evaluable_queries": int(len(errors)),
        "vertical_evaluable_fraction": float(len(errors) / len(output)),
        "vertical_reference_median_m": float(np.median(references)),
        "vertical_estimate_median_m": float(np.median(estimates)),
        "vertical_estimate_iqr_m": float(
            np.percentile(estimates, 75) - np.percentile(estimates, 25)
        ),
        "vertical_mae_m": float(np.mean(np.abs(errors))),
        "vertical_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "vertical_error_median_m": float(np.median(errors)),
        "vertical_error_p95_abs_m": float(np.percentile(np.abs(errors), 95)),
        "vertical_estimate_offset_m": args.vertical_estimate_offset,
        "retrieval_ms_median": float(np.median(times)),
        "retrieval_ms_p95": float(np.percentile(times, 95)),
        "vertical_conditioning": (
            "highest-ranked spatially correct hypothesis; retrieval rank unchanged"
        ),
    }

    args.output_per_query.parent.mkdir(parents=True, exist_ok=True)
    with args.output_per_query.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    with args.output_summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
