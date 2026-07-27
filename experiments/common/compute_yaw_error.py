#!/usr/bin/env python3
"""Top-1 yaw-seed error of UpDown-SC on the IH gravity condition.

The predicted map-frame query heading applies the estimated relative yaw to
the retrieved keyframe's stored map-frame heading:
predicted = candidate_yaw - yaw_shift_rad (roll and pitch are within a few
hundredths of a radian here, so the body yaw and the canonical heading agree
far below the 6 deg sector resolution). The reference heading comes from the
continuous prior-map localization odometry (full quaternion), associated by
nearest timestamp (max association error < 0.5 ms). Errors are reported over
queries whose top-1 place is correct, since a yaw seed is only consumed when
the place hypothesis is right.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

RT = Path("${UPDOWN_SC_ROOT}/icra2027_runtime")
CANDIDATES = (RT / "experiments/updown_weight_ablation_real_20260721/"
              "selected/ih_gravity/candidates.csv")
PER_QUERY = (RT / "experiments/updown_weight_ablation_real_20260721/"
             "selected/ih_gravity/per_query.csv")
ODOMETRY = (RT / "video/relocalization_experiment/current_map_localization/"
            "odometry.csv")
OUT = RT / "experiments/metrics_augment_20260725/yaw_error_summary.csv"


def quaternion_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def main() -> None:
    stamps, yaws = [], []
    with ODOMETRY.open(newline="") as handle:
        for row in csv.DictReader(handle):
            stamps.append(float(row["stamp"]))
            yaws.append(quaternion_yaw(
                float(row["qx"]), float(row["qy"]),
                float(row["qz"]), float(row["qw"])))
    stamps = np.asarray(stamps)
    yaws = np.asarray(yaws)
    order = np.argsort(stamps)
    stamps, yaws = stamps[order], yaws[order]

    with PER_QUERY.open(newline="") as handle:
        correct = {row["query_id"]: row["recall_at_1"] == "True"
                   for row in csv.DictReader(handle)}

    predicted, reference, kept = [], [], 0
    max_dt = 0.0
    with CANDIDATES.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["candidate_rank"]) != 1:
                continue
            if not correct.get(row["query_index"], False):
                continue
            stamp = float(row["query_stamp"])
            index = int(np.searchsorted(stamps, stamp))
            index = min(max(index, 1), len(stamps) - 1)
            if abs(stamps[index - 1] - stamp) < abs(stamps[index] - stamp):
                index -= 1
            max_dt = max(max_dt, abs(stamps[index] - stamp))
            predicted.append(
                float(row["candidate_yaw"]) - float(row["yaw_shift_rad"]))
            reference.append(yaws[index])
            kept += 1

    errors = np.degrees(np.abs(wrap(np.asarray(predicted) - np.asarray(reference))))
    summary = {
        "condition": "ih_gravity",
        "queries_with_correct_top1": kept,
        "association_dt_max_s": round(max_dt, 6),
        "yaw_error_median_deg": round(float(np.median(errors)), 3),
        "yaw_error_p95_deg": round(float(np.percentile(errors, 95)), 3),
        "yaw_error_max_deg": round(float(np.max(errors)), 3),
        "sector_resolution_deg": 6.0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(summary)


if __name__ == "__main__":
    main()
