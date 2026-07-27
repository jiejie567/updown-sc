#!/usr/bin/env python3
"""Benchmark descriptor construction plus database retrieval without disk I/O.

The database descriptors and all query point clouds are loaded before timing.
Map construction, file I/O, truth evaluation, and geometric registration are
therefore outside the reported interval.  Each truth-valid query is measured
once after one untimed warm-up query.
"""

from __future__ import annotations

import csv
import platform
import time
from pathlib import Path

import numpy as np

import run_m2dp_baseline as m2dp_impl
import run_retrieval_baselines as common
import run_ringpp_cpu_baseline as ringpp_impl


OUT_ROWS = common.RESULT_DIR / "retrieval_latency_in_memory_per_query.csv"
OUT_SUMMARY = common.RESULT_DIR / "retrieval_latency_in_memory_summary.csv"
OUT_README = common.RESULT_DIR / "retrieval_latency_protocol.md"


def timed(callable_):
    start = time.perf_counter_ns()
    value = callable_()
    return value, (time.perf_counter_ns() - start) / 1.0e6


def add_row(rows, algorithm, query_id, descriptor_ms, search_ms,
            protocol="single_frame"):
    rows.append({
        "algorithm": algorithm,
        "protocol": protocol,
        "query_id": query_id,
        "descriptor_ms": descriptor_ms,
        "search_ms": search_ms,
        "total_ms": descriptor_ms + search_ms,
    })


def benchmark_sc(rows, query_rows, clouds):
    cached = np.load(common.CACHE_DIR / "sc_r20_s60_r30_v025_native_exact2574_map.npz")
    maps = cached["descriptors"]
    ring_keys = np.mean(maps, axis=2)

    def run(points):
        descriptor, descriptor_ms = timed(lambda: common.scan_context(points))

        def search():
            ring_key = np.mean(descriptor, axis=1)
            distances = np.linalg.norm(ring_keys - ring_key[None, :], axis=1)
            selected = np.argpartition(distances, common.TOP_K - 1)[:common.TOP_K]
            exact = [common.official_sc_distance(descriptor, maps[index])[0]
                     for index in selected]
            return selected[np.argsort(exact)]

        _, search_ms = timed(search)
        return descriptor_ms, search_ms

    run(clouds[0])
    for query, points in zip(query_rows, clouds):
        descriptor_ms, search_ms = run(points)
        add_row(rows, "SC", query["query_id"], descriptor_ms, search_ms)
        print(f"latency SC {len([r for r in rows if r['algorithm'] == 'SC'])}/92", flush=True)


def benchmark_scpp(rows, query_rows, clouds):
    cached = np.load(common.CACHE_DIR / "scpp_pc_r20_s60_r30_v05_rawz_aug3_exact2574_map.npz")
    maps = cached["descriptors"]
    ring_keys = np.mean(maps, axis=3)

    def run(points):
        descriptor, descriptor_ms = timed(lambda: common.scan_context_plus_single(points))

        def search():
            ring_key = np.mean(descriptor, axis=1)
            root_distances = np.linalg.norm(ring_keys - ring_key[None, None, :], axis=2)
            frame_distances = np.min(root_distances, axis=1)
            selected = np.argpartition(frame_distances, common.TOP_K - 1)[:common.TOP_K]
            exact = [min(common.official_sc_distance(descriptor, root)[0]
                         for root in maps[index]) for index in selected]
            return selected[np.argsort(exact)]

        _, search_ms = timed(search)
        return descriptor_ms, search_ms

    run(clouds[0])
    for query, points in zip(query_rows, clouds):
        descriptor_ms, search_ms = run(points)
        add_row(rows, "SC++ (PC)", query["query_id"], descriptor_ms, search_ms)
        print(f"latency SC++ {len([r for r in rows if r['algorithm'] == 'SC++ (PC)'])}/92", flush=True)


def benchmark_solid(rows, query_rows, clouds):
    cached = np.load(common.CACHE_DIR / "solid_r40_s60_h64_r30_native_exact2574_map.npz")
    maps = cached["descriptors"]
    norms = np.linalg.norm(maps, axis=1)

    def run(points):
        descriptor, descriptor_ms = timed(lambda: common.solid(points)[0])

        def search():
            denominator = np.maximum(norms * np.linalg.norm(descriptor), 1e-12)
            similarity = maps @ descriptor / denominator
            return np.argpartition(-similarity, common.TOP_K - 1)[:common.TOP_K]

        _, search_ms = timed(search)
        return descriptor_ms, search_ms

    run(clouds[0])
    for query, points in zip(query_rows, clouds):
        descriptor_ms, search_ms = run(points)
        add_row(rows, "SOLiD", query["query_id"], descriptor_ms, search_ms)
        print(f"latency SOLiD {len([r for r in rows if r['algorithm'] == 'SOLiD'])}/92", flush=True)


def benchmark_m2dp(rows, query_rows, clouds):
    cached = np.load(common.CACHE_DIR / "m2dp_voxel025_map2574.npz")
    maps = cached["descriptors"]

    def run(points):
        descriptor, descriptor_ms = timed(
            lambda: m2dp_impl.m2dp(m2dp_impl.prepare_points(points[:, :3], True)))
        _, search_ms = timed(
            lambda: np.argpartition(
                m2dp_impl.descriptor_distances(maps, descriptor, False),
                common.TOP_K - 1)[:common.TOP_K])
        return descriptor_ms, search_ms

    run(clouds[0])
    for query, points in zip(query_rows, clouds):
        descriptor_ms, search_ms = run(points)
        add_row(rows, "M2DP", query["query_id"], descriptor_ms, search_ms)
        print(f"latency M2DP {len([r for r in rows if r['algorithm'] == 'M2DP'])}/92", flush=True)


def benchmark_ringpp(rows, query_rows, clouds):
    map_path, _, _ = ringpp_impl.cache_paths()
    maps = np.lib.format.open_memmap(map_path, mode="r")

    def run(points):
        descriptor, descriptor_ms = timed(lambda: ringpp_impl.tiring_descriptor(points))
        _, search_ms = timed(lambda: np.argpartition(
            -ringpp_impl.circular_scores(descriptor, maps),
            common.TOP_K - 1)[:common.TOP_K])
        return descriptor_ms, search_ms

    run(clouds[0])
    for index, (query, points) in enumerate(zip(query_rows, clouds), 1):
        descriptor_ms, search_ms = run(points)
        add_row(rows, "RING++ (faithful CPU port)", query["query_id"],
                descriptor_ms, search_ms)
        print(f"latency RING++ {index}/92 total={descriptor_ms + search_ms:.1f}ms", flush=True)


def add_updown(rows, query_rows):
    valid_windows = {int(row["window"]) for row in query_rows}
    with (common.UPDOWN_RESULTS / "relocalization_windows.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["window"]) not in valid_windows or not row.get("scan_context_ms"):
                continue
            rows.append({
                "algorithm": "UpDown-SC",
                "protocol": "single_frame",
                "query_id": row["window"],
                "descriptor_ms": "",
                "search_ms": "",
                "total_ms": float(row["scan_context_ms"]),
            })


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    result = []
    algorithms = list(dict.fromkeys(row["algorithm"] for row in rows))
    for algorithm in algorithms:
        group = [row for row in rows if row["algorithm"] == algorithm]
        values = np.asarray([float(row["total_ms"]) for row in group])
        descriptor = [float(row["descriptor_ms"]) for row in group
                      if row["descriptor_ms"] != ""]
        search = [float(row["search_ms"]) for row in group
                  if row["search_ms"] != ""]
        result.append({
            "algorithm": algorithm,
            "protocol": group[0]["protocol"],
            "queries": len(group),
            "descriptor_median_ms": np.median(descriptor) if descriptor else "",
            "search_median_ms": np.median(search) if search else "",
            "total_median_ms": np.median(values),
            "total_p95_ms": np.percentile(values, 95),
            "mean_ms": np.mean(values),
            "under_100ms_percent": 100.0 * np.mean(values <= 100.0),
        })
    return result


def main():
    query_rows = [row for row in common.load_queries() if row["truth_valid"] == "True"]
    print(f"preloading {len(query_rows)} query clouds (outside timing)", flush=True)
    clouds = [common.load_query_points(row) for row in query_rows]
    rows = []
    benchmark_solid(rows, query_rows, clouds)
    benchmark_m2dp(rows, query_rows, clouds)
    benchmark_ringpp(rows, query_rows, clouds)
    add_updown(rows, query_rows)
    write_csv(OUT_ROWS, rows, [
        "algorithm", "protocol", "query_id", "descriptor_ms", "search_ms", "total_ms"])
    summary = summarize(rows)
    write_csv(OUT_SUMMARY, summary, list(summary[0]))
    OUT_README.write_text(
        "# In-memory retrieval latency protocol\n\n"
        f"- Host: {platform.node()}\n"
        "- CPU: Intel Core Ultra 5 225H, 14 logical CPUs (one thread per core reported by `lscpu`)\n"
        f"- Python: {platform.python_version()}\n"
        f"- Database: {len(np.load(common.CACHE_DIR / 'sc_r20_s60_r30_v025_native_exact2574_map.npz')['descriptors'])} keyframes\n"
        f"- Queries: {len(query_rows)} truth-valid single frames\n"
        "- Timing: query descriptor construction + database matching/ranking.\n"
        "- Excluded: file I/O, database construction, truth evaluation, and ICP/geometric registration.\n"
        "- Measurement: one untimed warm-up query, then one timed pass over every query.\n"
        "- UpDown-SC uses the C++ `scan_context_ms` interval, from its in-memory source cloud through candidate/seed construction.\n"
        "- RING++ is the labelled faithful CPU formula port, not the released CUDA extension.\n"
        "- LiDAR-Iris uses its official C++ core; BTC and STD use their official C++ cores and registered non-overlapping 10-scan submaps.\n"
        "- BTC/STD native candidate verification is included, but final localization ICP is excluded.\n"
        "- The final SC and SC++ timings are supplied by the C++ benchmark during the merge step.\n"
        "- Other formula-adapter results characterize the tested implementations, not language-independent lower bounds.\n")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
