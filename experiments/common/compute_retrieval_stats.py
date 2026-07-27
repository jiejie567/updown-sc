#!/usr/bin/env python3
"""Wilson confidence intervals and PR/F1max/AUPR for the 2 m retrieval runs.

Recall CIs use the recorded per-query CSVs of every paper condition. PR-style
metrics follow the common LPR protocol: each query's top-1 candidate carries a
confidence score; sweeping an acceptance threshold yields precision (correct
accepted / accepted) and recall (correct accepted / all queries). F1max is the
maximum F1 over thresholds and AUPR the trapezoidal area under the PR curve.
Score direction is per method: distances (SC, SC++, M2DP, Iris, UpDown-SC)
rank ascending, similarities/correlations (SOLiD, RING++) descending.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

RT = Path("${UPDOWN_SC_ROOT}/icra2027_runtime/experiments")
SELECTED = RT / "updown_weight_ablation_real_20260721/selected"
AUG = RT / "metrics_augment_20260725"
GT = RT / "gravity_transfer_2m_20260718"
T3 = RT / "indoor_cross_device_2m/table3_2m"
OUT = AUG

# score direction: True when smaller scores mean higher confidence
DISTANCE_LIKE = {
    "SC": True, "SC++ (PC)": True, "SOLiD": False, "M2DP": True,
    "LiDAR-Iris": True, "RING++": False, "UpDown-SC": True,
}


def read_rows(path: Path, algorithm: str | None = None) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if algorithm is not None:
        rows = [row for row in rows if row["algorithm"] == algorithm]
    if not rows:
        raise RuntimeError(f"No rows for {algorithm} in {path}")
    return rows


def as_bool(value: str) -> bool:
    return value in ("True", "true", "1")


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return center - margin, center + margin


# ---------------------------------------------------------------------------
# Condition -> {method: (csv path, algorithm label)} for the recorded runs.
# ---------------------------------------------------------------------------

def cond_files() -> dict:
    ih = RT / "private_two_bag_2m/results"
    ch = RT / "rtk_slam_construction_hall_2m/results"
    nc = RT / "newer_college_quad_easy_2m/results"
    return {
        "ih_native": {
            "SC": (ih / "single_scan_per_query.csv", "SC"),
            "SC++ (PC)": (ih / "single_scan_per_query.csv", "SC++ (PC)"),
            "SOLiD": (ih / "single_scan_per_query.csv", "SOLiD"),
            "M2DP": (ih / "m2dp_per_query.csv", "M2DP"),
            "LiDAR-Iris": (ih / "lidar_iris_per_query.csv", None),
            "RING++": (ih / "ringpp_per_query.csv", None),
            "UpDown-SC": (SELECTED / "ih_native/per_query.csv", None),
        },
        "ih_gravity": {
            "SC": (GT / "ih/results/per_query.csv", "SC + G"),
            "SC++ (PC)": (GT / "ih/results/per_query.csv", "SC++ (PC) + G"),
            "SOLiD": (GT / "ih/results/per_query.csv", "SOLiD + G"),
            "M2DP": (GT / "ih/results/per_query.csv", "M2DP + G"),
            "LiDAR-Iris": (GT / "ih/results/lidar_iris_per_query.csv", None),
            "RING++": (GT / "ih/ringpp/results/ringpp_per_query.csv", None),
            "UpDown-SC": (SELECTED / "ih_gravity/per_query.csv", None),
        },
        "ch_native": {
            "SC": (ch / "single_scan_per_query.csv", "SC"),
            "SC++ (PC)": (ch / "single_scan_per_query.csv", "SC++ (PC)"),
            "SOLiD": (ch / "single_scan_per_query.csv", "SOLiD"),
            "M2DP": (ch / "m2dp_per_query.csv", "M2DP"),
            "LiDAR-Iris": (ch / "lidar_iris_per_query.csv", None),
            "RING++": (ch / "ringpp_per_query.csv", None),
            "UpDown-SC": (SELECTED / "ch_native/per_query.csv", None),
        },
        "ch_gravity": {
            "SC": (GT / "ch/results/per_query.csv", "SC + G"),
            "SC++ (PC)": (GT / "ch/results/per_query.csv", "SC++ (PC) + G"),
            "SOLiD": (GT / "ch/results/per_query.csv", "SOLiD + G"),
            "M2DP": (GT / "ch/results/per_query.csv", "M2DP + G"),
            "LiDAR-Iris": (GT / "ch/results/lidar_iris_per_query.csv", None),
            "RING++": (GT / "ch/ringpp/results/ringpp_per_query.csv", None),
            "UpDown-SC": (SELECTED / "ch_gravity/per_query.csv", None),
        },
        "h1_h2": {
            "SC": (T3 / "handle2/gravity/per_query.csv", "SC + G"),
            "SC++ (PC)": (T3 / "handle2/gravity/per_query.csv", "SC++ (PC) + G"),
            "SOLiD": (T3 / "handle2/gravity/per_query.csv", "SOLiD + G"),
            "M2DP": (T3 / "handle2/gravity/per_query.csv", "M2DP + G"),
            "LiDAR-Iris": (T3 / "handle2/iris/per_query.csv", None),
            "RING++": (T3 / "handle2/ringpp/results/ringpp_per_query.csv", None),
            "UpDown-SC": (SELECTED / "h1_h2/per_query.csv", None),
        },
        "h1_v1": {
            "SC": (T3 / "vehicle1/gravity/per_query.csv", "SC + G"),
            "SC++ (PC)": (T3 / "vehicle1/gravity/per_query.csv", "SC++ (PC) + G"),
            "SOLiD": (T3 / "vehicle1/gravity/per_query.csv", "SOLiD + G"),
            "M2DP": (T3 / "vehicle1/gravity/per_query.csv", "M2DP + G"),
            "LiDAR-Iris": (T3 / "vehicle1/iris/per_query.csv", None),
            "RING++": (T3 / "vehicle1/ringpp/results/ringpp_per_query.csv", None),
            "UpDown-SC": (SELECTED / "h1_v1/per_query.csv", None),
        },
        "nc_gravity": {
            "SC": (nc / "gravity/per_query.csv", "SC + G"),
            "SC++ (PC)": (nc / "gravity/per_query.csv", "SC++ (PC) + G"),
            "SOLiD": (nc / "gravity/per_query.csv", "SOLiD + G"),
            "M2DP": (nc / "gravity/per_query.csv", "M2DP + G"),
            "LiDAR-Iris": (nc / "lidar_iris_gravity_per_query.csv", None),
            "RING++": (nc / "ringpp_per_query.csv", None),
            "UpDown-SC": (SELECTED / "nc_gravity/per_query.csv", None),
        },
    }


# PR conditions use the rescored CSVs that carry top1_score.
def pr_files() -> dict:
    result = {}
    for cond, native_dir in (
            ("ih_native", AUG / "ih_native"), ("ih_gravity", AUG / "ih_gravity"),
            ("ch_native", AUG / "ch_native"), ("ch_gravity", AUG / "ch_gravity")):
        gravity = cond.endswith("gravity")
        suffix = " + G" if gravity else ""
        recorded = cond_files()[cond]
        result[cond] = {
            "SC": (native_dir / ("per_query.csv" if gravity else "single_scan_per_query.csv"), "SC" + suffix),
            "SC++ (PC)": (native_dir / ("per_query.csv" if gravity else "single_scan_per_query.csv"), "SC++ (PC)" + suffix),
            "SOLiD": (native_dir / ("per_query.csv" if gravity else "single_scan_per_query.csv"), "SOLiD" + suffix),
            "M2DP": (native_dir / ("per_query.csv" if gravity else "m2dp_per_query.csv"), "M2DP" + suffix),
            "LiDAR-Iris": (native_dir / "lidar_iris_per_query.csv", None),
            "RING++": recorded["RING++"],
            "UpDown-SC": (SELECTED / f"{cond}/candidates.csv", "CANDIDATES"),
        }
    return result


def updown_scores(cond: str):
    """Top-1 scores and correctness for UpDown-SC from candidates + per_query."""
    per_query = read_rows(SELECTED / f"{cond}/per_query.csv")
    correct = {row["query_id"]: as_bool(row["recall_at_1"]) for row in per_query}
    top1 = {}
    with (SELECTED / f"{cond}/candidates.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["candidate_rank"]) == 1:
                top1[row["query_index"]] = float(row["distance"])
    scores, flags = [], []
    for query_id, is_correct in correct.items():
        if query_id not in top1:
            continue  # failed query with no candidates: kept separately below
        scores.append(top1[query_id])
        flags.append(is_correct)
    missing = len(correct) - len(scores)
    return np.asarray(scores), np.asarray(flags, dtype=bool), missing, len(correct)


def pr_metrics(scores: np.ndarray, correct: np.ndarray, ascending: bool,
               total_queries: int) -> tuple[float, float]:
    order = np.argsort(scores if ascending else -scores, kind="stable")
    flags = correct[order]
    tp = np.cumsum(flags)
    accepted = np.arange(1, len(flags) + 1)
    precision = tp / accepted
    recall = tp / total_queries
    f1 = np.where(tp > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    # AUPR by trapezoid over recall, prepending the first point at recall 0.
    r = np.concatenate(([0.0], recall))
    p = np.concatenate(([precision[0] if len(precision) else 1.0], precision))
    aupr = float(np.trapz(p, r))
    return float(np.max(f1)), aupr


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    ci_rows = []
    for cond, methods in cond_files().items():
        for method, (path, algo) in methods.items():
            rows = read_rows(path, algo)
            n = len(rows)
            for k in ("recall_at_1", "recall_at_5"):
                successes = sum(as_bool(row[k]) for row in rows)
                low, high = wilson_interval(successes, n)
                point = successes / n
                ci_rows.append({
                    "condition": cond, "method": method, "metric": k,
                    "n": n, "successes": successes,
                    "value_pct": round(100 * point, 2),
                    "ci_low_pct": round(100 * low, 2),
                    "ci_high_pct": round(100 * high, 2),
                    "half_width_pts": round(100 * (high - low) / 2, 2),
                })
    with (OUT / "ci_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ci_rows[0]))
        writer.writeheader()
        writer.writerows(ci_rows)
    print(f"ci_summary.csv: {len(ci_rows)} rows")

    pr_rows = []
    for cond, methods in pr_files().items():
        for method, (path, algo) in methods.items():
            if algo == "CANDIDATES":
                scores, flags, missing, total = updown_scores(cond)
            else:
                rows = read_rows(path, algo)
                total = len(rows)
                scores = np.asarray([float(row["top1_score"]) for row in rows])
                flags = np.asarray([as_bool(row["recall_at_1"]) for row in rows])
                missing = 0
            f1max, aupr = pr_metrics(
                scores, flags, DISTANCE_LIKE[method], total)
            r1 = 100.0 * flags.sum() / total
            pr_rows.append({
                "condition": cond, "method": method, "n": total,
                "queries_without_candidates": missing,
                "recall_at_1_pct": round(r1, 2),
                "f1max": round(f1max, 4), "aupr": round(aupr, 4),
            })
            print(f"{cond:10s} {method:11s} R@1={r1:6.2f} "
                  f"F1max={f1max:.3f} AUPR={aupr:.3f}")
    with (OUT / "pr_f1_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pr_rows[0]))
        writer.writeheader()
        writer.writerows(pr_rows)
    print(f"pr_f1_summary.csv: {len(pr_rows)} rows")


if __name__ == "__main__":
    main()
