#!/usr/bin/env python3
"""Merge the two indoor tracks into auditable paper/video source data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


RELOCALIZATION = re.compile(
    r"success time_ms=(?P<time>[0-9.]+).*?"
    r"fitness=(?P<fitness>[0-9.]+) overlap=(?P<overlap>[0-9.]+)"
)


def read_one(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {path}")
    return rows[0]


def read_many(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def localization_result(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    match = RELOCALIZATION.search(text)
    if match is None:
        raise RuntimeError(f"No successful relocalization record in {path}")
    if "Localization health warning" in text:
        raise RuntimeError(f"Localization health warning found in {path}")
    return {name: float(value) for name, value in match.groupdict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.experiment_root

    tracks = [
        ("handle2", "Handle1 -> Handle2", "same acquisition class"),
        ("vehicle1", "Handle1 -> Vehicle1", "cross-platform/device"),
    ]
    method_rows: list[dict[str, object]] = []
    deployment_rows: list[dict[str, object]] = []
    for key, label, condition in tracks:
        updown = read_one(root / f"results/{key}/updown_summary.csv")
        core = read_many(root / f"tracks/{key}/results/single_scan_summary.csv")
        m2dp = read_one(root / f"tracks/{key}/results/m2dp_summary.csv")
        for row in [*core, m2dp]:
            method_rows.append(
                {
                    "track": key,
                    "condition": condition,
                    "algorithm": row["algorithm"],
                    "truth_queries": int(row["truth_queries"]),
                    "recall_at_1": float(row["recall_at_1"]),
                    "recall_at_5": float(row["recall_at_5"]),
                }
            )
        method_rows.append(
            {
                "track": key,
                "condition": condition,
                "algorithm": "UpDown-SC",
                "truth_queries": int(updown["truth_queries"]),
                "recall_at_1": float(updown["recall_at_1"]),
                "recall_at_5": float(updown["recall_at_5"]),
            }
        )

        protocol = json.loads(
            (root / f"derived/{key}/protocol_manifest.json").read_text()
        )
        online = localization_result(root / f"localization/{key}/fastlio.log")
        deployment_rows.append(
            {
                "track": key,
                "label": label,
                "condition": condition,
                "map_keyframes": protocol["keyframe_protocol"]["map_count"],
                "eligible_queries": protocol["truth"]["eligible_queries"],
                "online_relocalization_success": True,
                "online_relocalization_ms": online["time"],
                "icp_fitness": online["fitness"],
                "icp_overlap": online["overlap"],
                "trusted_pose_count": protocol["truth"]["trusted_pose_count"],
                "vertical_evaluable_queries": int(
                    updown["vertical_evaluable_queries"]
                ),
                "vertical_reference_median_m": float(
                    updown["vertical_reference_median_m"]
                ),
                "vertical_estimate_median_m": float(
                    updown["vertical_estimate_median_m"]
                ),
                "vertical_mae_m": float(updown["vertical_mae_m"]),
                "vertical_error_p95_abs_m": float(
                    updown["vertical_error_p95_abs_m"]
                ),
            }
        )

    output = root / "results"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "retrieval_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(method_rows[0]))
        writer.writeheader()
        writer.writerows(method_rows)
    with (output / "deployment_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(deployment_rows[0]))
        writer.writeheader()
        writer.writerows(deployment_rows)
    manifest = {
        "format": "indoor_cross_device_summary_v1",
        "map_bag": "indoor_handle1_ros2",
        "query_bags": ["indoor_handle2_ros2", "indoor_vehicle1_ros2"],
        "keyframe_rule": "translation-only 2 m, experiment-only",
        "truth": "continuous prior-map localization pseudo-reference",
        "deployment": deployment_rows,
    }
    (output / "experiment_summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
