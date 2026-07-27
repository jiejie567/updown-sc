#!/usr/bin/env python3
"""Merge native C++ timings into the in-memory retrieval latency report."""

import csv
from pathlib import Path

import numpy as np


ROOT = Path("${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/baseline_20260715")
RESULTS = ROOT / "results"
PER_QUERY = RESULTS / "retrieval_latency_in_memory_per_query.csv"
SUMMARY = RESULTS / "retrieval_latency_in_memory_summary.csv"
QUERY_METADATA = ROOT / "queries/loc_2_floor/metadata.csv"
UPDOWN_WINDOWS = Path(
    "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/two_bag_pilot_20260715/"
    "results/loc_2_floor_sector_local_w04_w06/relocalization_windows.csv")
NATIVE = {
    "SC": (
        RESULTS / "sc_cpp_official_per_query.csv", "single_frame"),
    "SC++ (PC)": (
        RESULTS / "scpp_cpp_formula_per_query.csv", "single_frame"),
    "LiDAR-Iris": (
        RESULTS / "lidar_iris_native_per_query.csv", "single_frame"),
    "BTC (10 scans)": (
        Path("${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/btc_submaps/results/"
             "btc_submap_verified_per_query.csv"), "10_scan_submap"),
    "STD (10 scans)": (
        Path("${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/std_submaps/results/"
             "std_submap_verified_per_query.csv"), "10_scan_submap"),
}


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    result = []
    for algorithm in dict.fromkeys(row["algorithm"] for row in rows):
        group = [row for row in rows if row["algorithm"] == algorithm]
        total = np.asarray([float(row["total_ms"]) for row in group])
        descriptor = [float(row["descriptor_ms"]) for row in group
                      if row["descriptor_ms"]]
        search = [float(row["search_ms"]) for row in group if row["search_ms"]]
        result.append({
            "algorithm": algorithm,
            "protocol": group[0]["protocol"],
            "queries": len(group),
            "descriptor_median_ms": np.median(descriptor) if descriptor else "",
            "search_median_ms": np.median(search) if search else "",
            "total_median_ms": np.median(total),
            "total_p95_ms": np.percentile(total, 95),
            "mean_ms": np.mean(total),
            "under_100ms_percent": 100.0 * np.mean(total <= 100.0),
        })
    order = {name: index for index, name in enumerate((
        "SC", "SC++ (PC)", "SOLiD", "M2DP", "LiDAR-Iris",
        "RING++ (faithful CPU port)", "UpDown-SC",
        "BTC (10 scans)", "STD (10 scans)"))}
    result.sort(key=lambda row: order.get(row["algorithm"], len(order)))
    return result


def main():
    valid_windows = {
        int(row["window"]) for row in read(QUERY_METADATA)
        if row["truth_valid"] == "True"
    }
    native_names = set(NATIVE) | {"UpDown-SC"}
    rows = [
        row for row in read(PER_QUERY)
        if row["algorithm"] not in native_names
    ]
    for algorithm, (path, protocol) in NATIVE.items():
        for row in read(path):
            rows.append({
                "algorithm": algorithm,
                "protocol": protocol,
                "query_id": row["query_id"],
                "descriptor_ms": row.get("descriptor_ms", ""),
                "search_ms": row.get("search_ms", ""),
                "total_ms": row["retrieval_ms"],
            })
    for row in read(UPDOWN_WINDOWS):
        if int(row["window"]) not in valid_windows:
            continue
        rows.append({
            "algorithm": "UpDown-SC",
            "protocol": "single_frame",
            "query_id": row["window"],
            "descriptor_ms": "",
            "search_ms": "",
            "total_ms": row["scan_context_ms"],
        })
    write(PER_QUERY, rows)
    summary = summarize(rows)
    write(SUMMARY, summary)
    for row in summary:
        print(f"{row['algorithm']:<30} median={float(row['total_median_ms']):8.2f} "
              f"p95={float(row['total_p95_ms']):8.2f} ms "
              f"<=100ms={float(row['under_100ms_percent']):5.1f}%")


if __name__ == "__main__":
    main()
