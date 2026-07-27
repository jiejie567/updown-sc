#!/usr/bin/env python3
"""Sweep one global UpDown-SC channel weight on the real Table-II datasets.

Descriptor files are reused verbatim.  Only the runtime lower/upper weights are
overridden, so descriptor construction, ring-key shortlisting, yaw search, and
truth protocols remain fixed.  The selected setting maximizes the unweighted
macro-average Recall@1 over the seven paper columns; macro Recall@5 breaks ties.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def datasets(root: Path) -> list[dict[str, object]]:
    adaptive = root / "adaptive_split_conditional_20260720"
    return [
        {
            "key": "ih_native",
            "label": "IH Native",
            "map": adaptive / "native/ih/map.scd",
            "query": adaptive / "native/ih/query.scd",
            "metadata": root / "private_two_bag_2m/derived/queries/metadata.csv",
            "radius": 2.0,
        },
        {
            "key": "ih_gravity",
            "label": "IH +G",
            "map": adaptive / "private/map.scd",
            "query": adaptive / "private/query.scd",
            "metadata": root / "private_two_bag_2m/derived/queries/metadata.csv",
            "radius": 2.0,
        },
        {
            "key": "ch_native",
            "label": "CH Native",
            "map": adaptive / "native/ch/map.scd",
            "query": adaptive / "native/ch/query.scd",
            "metadata": root / "rtk_slam_construction_hall_2m/derived/queries_seq2/metadata.csv",
            "radius": 5.0,
        },
        {
            "key": "ch_gravity",
            "label": "CH +G",
            "map": adaptive / "ch/map.scd",
            "query": adaptive / "ch/query.scd",
            "metadata": root / "rtk_slam_construction_hall_2m/derived/queries_seq2/metadata.csv",
            "radius": 5.0,
        },
        {
            "key": "h1_h2",
            "label": "H1->H2 +G",
            "map": adaptive / "ih/map.scd",
            "query": adaptive / "ih/handle2.scd",
            "metadata": root / "indoor_cross_device_2m/derived/handle2/metadata.csv",
            "radius": 2.0,
        },
        {
            "key": "h1_v1",
            "label": "H1->V1 +G",
            "map": adaptive / "ih/map.scd",
            "query": adaptive / "ih/vehicle1.scd",
            "metadata": root / "indoor_cross_device_2m/derived/vehicle1/metadata.csv",
            "radius": 2.0,
        },
        {
            "key": "nc_gravity",
            "label": "NC +G",
            "map": root / "newer_college_quad_easy_2m/map/scans.scd",
            "query": root / "newer_college_quad_easy_2m/query/scans.scd",
            "metadata": root / "newer_college_quad_easy_2m/derived/queries/metadata.csv",
            "radius": 5.0,
        },
    ]


def load_truth(path: Path) -> dict[int, tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            int(row["query_id"]): (float(row["truth_x"]), float(row["truth_y"]))
            for row in csv.DictReader(stream)
            if truthy(row["truth_valid"])
        }


def evaluate(candidates: Path, metadata: Path, radius: float) -> tuple[dict[str, float], list[dict[str, object]]]:
    truth = load_truth(metadata)
    grouped: dict[int, list[dict[str, str]]] = {}
    with candidates.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault(int(row["query_index"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["candidate_rank"]))

    per_query: list[dict[str, object]] = []
    for query_id, (truth_x, truth_y) in truth.items():
        rows = grouped.get(query_id, [])
        first_correct_rank: int | None = None
        top1_error = math.inf
        retrieval_ms = math.nan
        top1_index = -1
        if rows:
            retrieval_ms = float(rows[0]["retrieval_ms"])
            top1_index = int(rows[0]["candidate_index"])
            for rank, row in enumerate(rows, start=1):
                error = math.hypot(
                    float(row["candidate_x"]) - truth_x,
                    float(row["candidate_y"]) - truth_y,
                )
                if rank == 1:
                    top1_error = error
                if first_correct_rank is None and error <= radius:
                    first_correct_rank = rank
        per_query.append(
            {
                "algorithm": "UpDown-SC pure adaptive",
                "query_id": query_id,
                "truth_x": truth_x,
                "truth_y": truth_y,
                "top1_index": top1_index,
                "top1_error_m": top1_error,
                "first_correct_rank": first_correct_rank or "",
                "recall_at_1": first_correct_rank is not None and first_correct_rank <= 1,
                "recall_at_5": first_correct_rank is not None and first_correct_rank <= 5,
                "retrieval_ms": retrieval_ms,
            }
        )
    summary = {
        "queries": len(per_query),
        "recall_at_1": sum(bool(row["recall_at_1"]) for row in per_query) / len(per_query),
        "recall_at_5": sum(bool(row["recall_at_5"]) for row in per_query) / len(per_query),
    }
    return summary, per_query


def run_retrieval(dataset: dict[str, object], candidates: Path, low: float, high: float) -> None:
    command = [
        "ros2", "run", "fast_lio", "scan_context_cross_sequence_evaluator",
        str(dataset["map"]), str(dataset["query"]), str(candidates),
        "100", f"{low:.1f}", f"{high:.1f}", "2", "0.5", "0.1",
    ]
    subprocess.run(command, check=True)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("${UPDOWN_SC_ROOT}/icra2027_runtime/experiments"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = datasets(args.runtime_root.resolve())
    for dataset in data:
        for field in ("map", "query", "metadata"):
            if not Path(dataset[field]).exists():
                raise FileNotFoundError(dataset[field])

    args.output.mkdir(parents=True, exist_ok=True)
    sweep_rows: list[dict[str, object]] = []
    macro_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="updown_weight_") as temp_dir:
        temp = Path(temp_dir)
        for low_tenths in range(11):
            low = low_tenths / 10.0
            high = 1.0 - low
            weight_rows = []
            for dataset in data:
                candidates = temp / f"{dataset['key']}.csv"
                run_retrieval(dataset, candidates, low, high)
                summary, _ = evaluate(
                    candidates, Path(dataset["metadata"]), float(dataset["radius"])
                )
                row = {
                    "lower_weight": low,
                    "upper_weight": high,
                    "dataset": dataset["key"],
                    "label": dataset["label"],
                    **summary,
                }
                sweep_rows.append(row)
                weight_rows.append(row)
                candidates.unlink(missing_ok=True)
                Path(str(candidates) + ".timing.csv").unlink(missing_ok=True)
            macro_rows.append(
                {
                    "lower_weight": low,
                    "upper_weight": high,
                    "macro_recall_at_1": sum(float(row["recall_at_1"]) for row in weight_rows) / len(weight_rows),
                    "macro_recall_at_5": sum(float(row["recall_at_5"]) for row in weight_rows) / len(weight_rows),
                }
            )

    selected = max(
        macro_rows,
        key=lambda row: (float(row["macro_recall_at_1"]), float(row["macro_recall_at_5"])),
    )
    write_rows(args.output / "weight_sweep_by_dataset.csv", sweep_rows)
    write_rows(args.output / "weight_sweep_macro.csv", macro_rows)

    low = float(selected["lower_weight"])
    high = float(selected["upper_weight"])
    selected_rows: list[dict[str, object]] = []
    for dataset in data:
        output_dir = args.output / "selected" / str(dataset["key"])
        output_dir.mkdir(parents=True, exist_ok=True)
        candidates = output_dir / "candidates.csv"
        run_retrieval(dataset, candidates, low, high)
        summary, per_query = evaluate(
            candidates, Path(dataset["metadata"]), float(dataset["radius"])
        )
        summary_row = {
            "dataset": dataset["key"],
            "label": dataset["label"],
            "lower_weight": low,
            "upper_weight": high,
            **summary,
        }
        selected_rows.append(summary_row)
        write_rows(output_dir / "per_query.csv", per_query)
        write_rows(output_dir / "summary.csv", [summary_row])

    write_rows(args.output / "selected_summary.csv", selected_rows)
    manifest = {
        "selection_rule": "maximize seven-column macro Recall@1; break ties with macro Recall@5",
        "weights_tested": [[i / 10.0, 1.0 - i / 10.0] for i in range(11)],
        "selected": selected,
        "fixed_retrieval": {
            "top_k": 100,
            "min_joint_rings": 2,
            "sector_support_exponent": 0.5,
            "retrieval_height_offset_m": 0.1,
        },
        "datasets": data,
    }
    with (args.output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, default=str)
    print(json.dumps({"selected": selected, "results": selected_rows}, indent=2))


if __name__ == "__main__":
    main()
