#!/usr/bin/env python3
import csv
from pathlib import Path

import numpy as np


ROOT = Path("${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/baseline_20260715")
RESULTS = ROOT / "results"
BTC_RESULTS = Path(
    "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/btc_submaps/results/"
    "btc_submap_verified_per_query.csv")
STD_RESULTS = Path(
    "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/std_submaps/results/"
    "std_submap_verified_per_query.csv")


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def summarize_per_query(path, algorithm, implementation, protocol="native_base_link",
                        map_build_ms="", finite_errors_only=False):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors = np.asarray([float(row["top1_error_m"]) for row in rows])
    finite_errors = errors[np.isfinite(errors)]
    selected_errors = finite_errors if finite_errors_only else errors
    times = np.asarray([float(row["retrieval_ms"]) for row in rows])
    return {
        "algorithm": algorithm,
        "protocol": protocol,
        "implementation": implementation,
        "truth_queries": len(rows),
        **{
            name: float(np.mean([truthy(row[name]) for row in rows]))
            for name in ("recall_at_1", "recall_at_5", "recall_at_10", "recall_at_100")
        },
        "top1_error_median": float(np.median(selected_errors)),
        "top1_error_p95": float(
            np.percentile(selected_errors, 95) if finite_errors_only
            else np.quantile(selected_errors, 0.95, method="nearest")),
        "retrieval_ms_median": float(np.median(times)),
        "retrieval_ms_p95": float(np.percentile(times, 95)),
        "map_build_ms": map_build_ms,
    }


def main():
    with (RESULTS / "retrieval_summary.csv").open(newline="") as handle:
        core = list(csv.DictReader(handle))
    rows = []
    for row in core:
        if row["algorithm"] == "SC":
            rows.append(summarize_per_query(
                RESULTS / "sc_cpp_official_per_query.csv", "SC",
                "official C++ core; common 30 m, 20x60, Top-100 protocol with matched 0.25 m sampling",
                map_build_ms="807.414"))
            continue
        if row["algorithm"] == "SC++ (PC)":
            rows.append(summarize_per_query(
                RESULTS / "scpp_cpp_formula_per_query.csv", "SC++ (PC)",
                "C++ adapter of the released T-RO Polar Context formula with 3 virtual roots",
                map_build_ms="982.251"))
            continue
        if row["algorithm"] == "UpDown-SC":
            rows.append(summarize_per_query(
                RESULTS / "updown_sector_local_w04_w06_per_query.csv", "UpDown-SC",
                "FAST-LIO C++ pipeline with SC-style sector-key local yaw search",
                "proposed_gravity_canonicalized"))
            continue
        row["protocol"] = (
            "proposed_gravity_canonicalized"
            if row["algorithm"] == "UpDown-SC"
            else "native_base_link"
        )
        implementations = {
            "SC (16x60 matched)": "SC formula on the proposed 16x60 grid with exhaustive yaw",
            "SC (16x60 + gravity)": "matched SC plus the proposed gravity canonicalization diagnostic",
            "SC (32x60 capacity)": "SC formula with 1920 cells and exhaustive yaw",
            "SOLiD": "validated official-formula NumPy adapter",
        }
        row["implementation"] = implementations[row["algorithm"]]
        rows.append(row)
    rows.append(summarize_per_query(
        RESULTS / "m2dp_voxel025_per_query.csv", "M2DP",
        "direct NumPy translation of official MATLAB formula with matched 0.25 m voxel sampling",
        map_build_ms="48825.13981399825"))
    rows.append(summarize_per_query(
        RESULTS / "lidar_iris_native_per_query.csv", "LiDAR-Iris",
        "official C++ core with OpenCV 4 compatibility fixes"))
    rows.append(summarize_per_query(
        RESULTS / "ringpp_cpu_per_query.csv", "RING++",
        "faithful CPU port of the released six-channel feature, BEV, Radon, and circular-correlation formulas",
        map_build_ms="308787.679275003"))
    rows.append(summarize_per_query(
        RESULTS / "lidar_iris_gravity_aligned_per_query.csv",
        "LiDAR-Iris + gravity",
        "official C++ core plus proposed gravity canonicalization",
        "gravity_canonicalized_diagnostic"))
    rows.append(summarize_per_query(
        BTC_RESULTS,
        "BTC (10 scans)",
        "official C++ core with registered non-overlapping 10-scan submaps",
        "native_base_link_10_scan_submap"))
    rows.append(summarize_per_query(
        STD_RESULTS,
        "STD (10 scans)",
        "official C++ core with registered non-overlapping 10-scan submaps and geometric verification",
        "native_base_link_10_scan_submap", map_build_ms="12371.1",
        finite_errors_only=True))
    order = {name: index for index, name in enumerate(
        ("SC", "SC (16x60 matched)", "SC (16x60 + gravity)",
         "SC (32x60 capacity)",
         "SC++ (PC)", "SOLiD", "M2DP", "LiDAR-Iris", "RING++", "BTC (10 scans)",
         "STD (10 scans)",
         "LiDAR-Iris + gravity", "UpDown-SC"))}
    rows.sort(key=lambda row: order[row["algorithm"]])
    fields = [
        "algorithm", "protocol", "implementation", "truth_queries",
        "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_100",
        "top1_error_median", "top1_error_p95", "retrieval_ms_median",
        "retrieval_ms_p95", "map_build_ms",
    ]
    with (RESULTS / "open_source_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['algorithm']:12s} "
            f"R@1={100 * float(row['recall_at_1']):5.1f}% "
            f"R@5={100 * float(row['recall_at_5']):5.1f}% "
            f"R@10={100 * float(row['recall_at_10']):5.1f}% "
            f"R@100={100 * float(row['recall_at_100']):5.1f}%")


if __name__ == "__main__":
    main()
