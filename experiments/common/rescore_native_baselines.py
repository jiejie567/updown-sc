#!/usr/bin/env python3
"""Re-run the recorded native single-scan baselines with top-1 score export.

This driver reproduces the exact recorded native protocol of the 2 m
retrieval experiments (SC with the official 20-ring descriptor, SC++ (PC),
SOLiD, and M2DP), reusing the existing map descriptor caches, and writes the
per-query CSVs -- now including ``top1_score`` -- to a separate output
directory so the recorded result CSVs are never overwritten. Recall values
must match the recorded runs exactly; the score column is the only addition.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import time
from pathlib import Path

import numpy as np


AUDITED_ROOT = Path(
    "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/baseline_20260715/scripts")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--correct-radius", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    common = load_module(
        "rescore_common", AUDITED_ROOT / "run_retrieval_baselines.py")
    common.ROOT = args.experiment_root
    common.MAP_DIR = args.experiment_root / "seq1/session/key_point_frame"
    common.MAP_POSES = args.experiment_root / "seq1/session/optimized_poses_tum.txt"
    common.QUERY_DIR = args.experiment_root / "derived/queries_seq2"
    common.QUERY_GRAVITY = common.QUERY_DIR / "gravity.csv"
    common.CACHE_DIR = args.experiment_root / "cache"
    common.RESULT_DIR = args.output_dir
    common.CORRECT_RADIUS = args.correct_radius
    common.TOP_K = args.top_k
    args.output_dir.mkdir(parents=True, exist_ok=True)

    map_poses = common.load_map_poses()
    queries = common.load_queries()

    # Recorded native protocol: official 20-ring SC (existing cache names).
    sc_map, _, _ = common.build_or_load_map_cache(
        "construction_sc_r20_s60_r30_native", common.scan_context)
    sc_plus_map, _, _ = common.build_or_load_map_cache(
        "construction_scpp_pc_r20_s60_r30_aug3", common.scan_context_plus_map)
    solid_map, _, _ = common.build_or_load_map_cache(
        "construction_solid_r40_s60_h64_r30_native",
        lambda points: common.solid(points)[0])

    rows: list[dict] = []
    rows.extend(common.evaluate_sc(sc_map, map_poses, queries))
    rows.extend(common.evaluate_sc_plus(sc_plus_map, map_poses, queries))
    rows.extend(common.evaluate_solid(solid_map, map_poses, queries))
    write_csv(args.output_dir / "single_scan_per_query.csv", rows)

    m2dp = load_module("rescore_m2dp", AUDITED_ROOT / "run_m2dp_baseline.py")
    m2dp.ROOT = args.experiment_root
    m2dp.MAP_DIR = common.MAP_DIR
    m2dp.MAP_POSES = common.MAP_POSES
    m2dp.QUERY_DIR = common.QUERY_DIR
    m2dp.RESULT_DIR = args.output_dir
    m2dp.CORRECT_RADIUS = args.correct_radius
    m2dp.TOP_K = args.top_k
    m2dp.VOXEL025_CACHE = args.experiment_root / "cache/construction_m2dp_voxel025.npz"
    m2dp.VOXEL025_OUTPUT = args.output_dir / "m2dp_per_query.csv"

    descriptors, _ = m2dp.build_map(pca4=False, voxel025=True)
    positions = m2dp.map_positions()
    m2dp_rows: list[dict] = []
    query_rows = [row for row in m2dp.queries() if row["truth_valid"] == "True"]
    for ordinal, query in enumerate(query_rows, 1):
        started = time.perf_counter()
        points = m2dp.prepare_points(m2dp.query_points(query), voxel025=True)
        descriptor = m2dp.m2dp(points)
        distances = m2dp.descriptor_distances(descriptors, descriptor, pca4=False)
        candidates = np.argsort(distances)[:args.top_k]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        truth = np.asarray([float(query["truth_x"]), float(query["truth_y"])])
        errors = np.linalg.norm(positions[candidates] - truth, axis=1)
        matches = np.flatnonzero(errors <= args.correct_radius)
        rank = int(matches[0] + 1) if len(matches) else None
        top = int(candidates[0])
        m2dp_rows.append({
            "algorithm": "M2DP",
            "query_id": query["query_id"],
            "window": query["window"],
            "start_s": query["start_s"],
            "truth_x": query["truth_x"],
            "truth_y": query["truth_y"],
            "top1_index": top,
            "top1_x": positions[top, 0],
            "top1_y": positions[top, 1],
            "top1_error_m": errors[0],
            "first_correct_rank": rank if rank else "",
            "recall_at_1": rank is not None and rank <= 1,
            "recall_at_5": rank is not None and rank <= 5,
            "recall_at_10": rank is not None and rank <= 10,
            "recall_at_100": rank is not None and rank <= 100,
            "retrieval_ms": elapsed_ms,
            "top1_score": float(distances[top]),
        })
        if ordinal % 100 == 0 or ordinal == len(query_rows):
            print(f"M2DP query {ordinal}/{len(query_rows)}", flush=True)
    write_csv(m2dp.VOXEL025_OUTPUT, m2dp_rows)

    for algorithm in ("SC", "SC++ (PC)", "SOLiD"):
        group = [row for row in rows if row["algorithm"] == algorithm]
        r1 = float(np.mean([row["recall_at_1"] for row in group]))
        r5 = float(np.mean([row["recall_at_5"] for row in group]))
        print(f"{algorithm:10s} n={len(group)} R@1={100 * r1:.2f} R@5={100 * r5:.2f}")
    r1 = float(np.mean([row["recall_at_1"] for row in m2dp_rows]))
    r5 = float(np.mean([row["recall_at_5"] for row in m2dp_rows]))
    print(f"M2DP       n={len(m2dp_rows)} R@1={100 * r1:.2f} R@5={100 * r5:.2f}")


if __name__ == "__main__":
    main()
