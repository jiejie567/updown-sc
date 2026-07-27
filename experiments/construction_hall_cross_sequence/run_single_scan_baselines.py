#!/usr/bin/env python3
"""Run single-scan retrieval baselines on the prepared cross-sequence split.

This reuses the already-audited formula implementations from the main
baseline experiment while replacing all dataset paths and caches.  Baselines
receive the native deskewed base-link clouds; only UpDown-SC uses its proposed
gravity canonicalization, evaluated separately by the C++ database tool.
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np


AUDITED_ROOT = Path("${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/baseline_20260715")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], build_ms: dict[str, float]) -> list[dict]:
    output = []
    for algorithm in ("SC", "SC++ (PC)", "SOLiD"):
        group = [row for row in rows if row["algorithm"] == algorithm]
        times = np.asarray([float(row["retrieval_ms"]) for row in group])
        errors = np.asarray([float(row["top1_error_m"]) for row in group])
        output.append({
            "algorithm": algorithm,
            "protocol": "native_base_link_single_scan",
            "truth_queries": len(group),
            "recall_at_1": float(np.mean([row["recall_at_1"] for row in group])),
            "recall_at_5": float(np.mean([row["recall_at_5"] for row in group])),
            "recall_at_10": float(np.mean([row["recall_at_10"] for row in group])),
            "recall_at_100": float(np.mean([row["recall_at_100"] for row in group])),
            "top1_error_median": float(np.median(errors)),
            "top1_error_p95": float(np.percentile(errors, 95)),
            "retrieval_ms_median": float(np.median(times)),
            "retrieval_ms_p95": float(np.percentile(times, 95)),
            "map_build_ms": build_ms[algorithm],
        })
    return output


def run_core(args) -> None:
    common = load_module(
        "construction_common",
        AUDITED_ROOT / "scripts/run_retrieval_baselines.py",
    )
    common.ROOT = args.experiment_root
    common.MAP_DIR = args.experiment_root / "seq1/session/key_point_frame"
    common.MAP_POSES = args.experiment_root / "seq1/session/optimized_poses_tum.txt"
    common.MAP_GRAVITY = args.experiment_root / "seq1/session/scan_context_gravity.csv"
    common.QUERY_DIR = args.experiment_root / "derived/queries_seq2"
    common.QUERY_GRAVITY = common.QUERY_DIR / "gravity.csv"
    common.CACHE_DIR = args.experiment_root / "cache"
    common.RESULT_DIR = args.experiment_root / "results"
    common.CORRECT_RADIUS = args.correct_radius
    common.TOP_K = args.top_k
    common.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    common.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    map_poses = common.load_map_poses()
    queries = common.load_queries()
    map_count = len(list(common.MAP_DIR.glob("*.pcd")))
    if len(map_poses) != map_count:
        raise RuntimeError(f"Map pose/PCD mismatch: {len(map_poses)} != {map_count}")

    sc_map, sc_build_ms, _ = common.build_or_load_map_cache(
        "construction_sc_r16_s60_r30_native",
        lambda points: common.scan_context(points, rings=16),
    )
    sc_plus_map, sc_plus_build_ms, _ = common.build_or_load_map_cache(
        "construction_scpp_pc_r20_s60_r30_aug3",
        common.scan_context_plus_map,
    )
    solid_map, solid_build_ms, _ = common.build_or_load_map_cache(
        "construction_solid_r40_s60_h64_r30_native",
        lambda points: common.solid(points)[0],
    )

    rows = []
    rows.extend(
        common.evaluate_sc(
            sc_map,
            map_poses,
            queries,
            descriptor_fn=lambda points: common.scan_context(points, rings=16),
        )
    )
    rows.extend(common.evaluate_sc_plus(sc_plus_map, map_poses, queries))
    rows.extend(common.evaluate_solid(solid_map, map_poses, queries))
    summaries = summarize(rows, {
        "SC": sc_build_ms,
        "SC++ (PC)": sc_plus_build_ms,
        "SOLiD": solid_build_ms,
    })
    write_csv(common.RESULT_DIR / "single_scan_per_query.csv", rows)
    write_csv(common.RESULT_DIR / "single_scan_summary.csv", summaries)
    for row in summaries:
        print(
            f"{row['algorithm']:10s} "
            f"R@1={100.0 * row['recall_at_1']:.2f}% "
            f"R@5={100.0 * row['recall_at_5']:.2f}% "
            f"R@10={100.0 * row['recall_at_10']:.2f}%")


def run_m2dp(args) -> None:
    module = load_module(
        "construction_m2dp",
        AUDITED_ROOT / "scripts/run_m2dp_baseline.py",
    )
    module.ROOT = args.experiment_root
    module.MAP_DIR = args.experiment_root / "seq1/session/key_point_frame"
    module.MAP_POSES = args.experiment_root / "seq1/session/optimized_poses_tum.txt"
    module.QUERY_DIR = args.experiment_root / "derived/queries_seq2"
    module.RESULT_DIR = args.experiment_root / "results"
    module.CORRECT_RADIUS = args.correct_radius
    module.TOP_K = args.top_k
    module.VOXEL025_CACHE = args.experiment_root / "cache/construction_m2dp_voxel025.npz"
    module.VOXEL025_OUTPUT = args.experiment_root / "results/m2dp_per_query.csv"

    descriptors, build_ms = module.build_map(pca4=False, voxel025=True)
    positions = module.map_positions()
    query_rows = [row for row in module.queries() if row["truth_valid"] == "True"]
    rows = []
    for ordinal, query in enumerate(query_rows, 1):
        import time
        started = time.perf_counter()
        points = module.prepare_points(module.query_points(query), voxel025=True)
        descriptor = module.m2dp(points)
        distances = module.descriptor_distances(descriptors, descriptor, pca4=False)
        candidates = np.argsort(distances)[:args.top_k]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        truth = np.asarray([float(query["truth_x"]), float(query["truth_y"])])
        errors = np.linalg.norm(positions[candidates] - truth, axis=1)
        matches = np.flatnonzero(errors <= args.correct_radius)
        rank = int(matches[0] + 1) if len(matches) else None
        top = int(candidates[0])
        rows.append({
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
        })
        if ordinal % 100 == 0 or ordinal == len(query_rows):
            print(f"M2DP query {ordinal}/{len(query_rows)}", flush=True)
    write_csv(module.VOXEL025_OUTPUT, rows)
    summary = {
        "algorithm": "M2DP",
        "protocol": "native_base_link_single_scan",
        "truth_queries": len(rows),
        **{
            key: float(np.mean([bool(row[key]) for row in rows]))
            for key in ("recall_at_1", "recall_at_5", "recall_at_10", "recall_at_100")
        },
        "top1_error_median": float(np.median([row["top1_error_m"] for row in rows])),
        "top1_error_p95": float(np.percentile([row["top1_error_m"] for row in rows], 95)),
        "retrieval_ms_median": float(np.median([row["retrieval_ms"] for row in rows])),
        "retrieval_ms_p95": float(np.percentile([row["retrieval_ms"] for row in rows], 95)),
        "map_build_ms": build_ms,
    }
    write_csv(args.experiment_root / "results/m2dp_summary.csv", [summary])
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--correct-radius", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--methods",
        choices=("core", "m2dp", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.methods in ("core", "all"):
        run_core(args)
    if args.methods in ("m2dp", "all"):
        run_m2dp(args)


if __name__ == "__main__":
    main()
