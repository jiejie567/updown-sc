#!/usr/bin/env python3
"""Evaluate single-scan retrieval baselines after gravity canonicalization.

The map and query clouds are independently rotated from their native body
frames so that their measured up vectors map to +Z.  Descriptor definitions,
sampling, search, candidate depth, and the localization correctness rule are
otherwise identical to the native baseline runs.
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


def summarize(rows: list[dict], build_ms: dict[str, float]) -> list[dict]:
    output = []
    for algorithm in ("SC + G", "SC++ (PC) + G", "SOLiD + G", "M2DP + G"):
        group = [row for row in rows if row["algorithm"] == algorithm]
        ranks = [
            int(row["first_correct_rank"])
            if row["first_correct_rank"] != "" else None
            for row in group
        ]
        times = np.asarray([float(row["retrieval_ms"]) for row in group])
        errors = np.asarray([float(row["top1_error_m"]) for row in group])
        output.append({
            "algorithm": algorithm,
            "protocol": "gravity_canonicalized_native_single_scan",
            "truth_queries": len(group),
            "recall_at_1": float(np.mean(
                [rank is not None and rank <= 1 for rank in ranks])),
            "recall_at_5": float(np.mean(
                [rank is not None and rank <= 5 for rank in ranks])),
            "recall_at_10": float(np.mean(
                [rank is not None and rank <= 10 for rank in ranks])),
            "recall_at_100": float(np.mean(
                [rank is not None and rank <= 100 for rank in ranks])),
            "top1_error_median": float(np.median(errors)),
            "top1_error_p95": float(np.percentile(errors, 95)),
            "retrieval_ms_median": float(np.median(times)),
            "retrieval_ms_p95": float(np.percentile(times, 95)),
            "map_build_ms": build_ms[algorithm],
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-dir", required=True, type=Path)
    parser.add_argument("--map-poses", required=True, type=Path)
    parser.add_argument("--map-gravity", required=True, type=Path)
    parser.add_argument("--query-dir", required=True, type=Path)
    parser.add_argument("--query-gravity", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--correct-radius", required=True, type=float)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    common = load_module(
        "gravity_transfer_common", AUDITED_ROOT / "run_retrieval_baselines.py")
    m2dp = load_module(
        "gravity_transfer_m2dp", AUDITED_ROOT / "run_m2dp_baseline.py")

    common.MAP_DIR = args.map_dir
    common.MAP_POSES = args.map_poses
    common.MAP_GRAVITY = args.map_gravity
    common.QUERY_DIR = args.query_dir
    common.QUERY_GRAVITY = args.query_gravity
    common.CACHE_DIR = args.cache_dir
    common.RESULT_DIR = args.output_dir
    common.CORRECT_RADIUS = args.correct_radius
    common.TOP_K = args.top_k
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    map_poses = common.load_map_poses()
    queries = common.load_queries()
    map_gravity = common.load_gravity(args.map_gravity, "index")
    query_gravity = common.load_gravity(args.query_gravity, "query_id")
    map_paths = sorted(args.map_dir.glob("*.pcd"), key=lambda path: int(path.stem))
    if len(map_paths) != len(map_poses) or len(map_gravity) != len(map_poses):
        raise RuntimeError(
            "Map PCD/pose/gravity mismatch: "
            f"{len(map_paths)}/{len(map_poses)}/{len(map_gravity)}")
    if len(query_gravity) != len(queries):
        raise RuntimeError(
            f"Query metadata/gravity mismatch: {len(queries)}/{len(query_gravity)}")

    def canonical_map(points: np.ndarray, index: int) -> np.ndarray:
        return common.gravity_canonicalize(points, map_gravity[index])

    def canonical_query(row: dict) -> np.ndarray:
        points = common.load_query_points(row)
        return common.gravity_canonicalize(
            points, query_gravity[int(row["query_id"])])

    sc_map, sc_build_ms, _ = common.build_or_load_map_cache(
        "gravity_transfer_sc_r16_s60",
        lambda points, index: common.scan_context(
            canonical_map(points, index), rings=16),
        indexed=True,
    )
    scpp_map, scpp_build_ms, _ = common.build_or_load_map_cache(
        "gravity_transfer_scpp_pc_r20_s60_aug3",
        lambda points, index: common.scan_context_plus_map(
            canonical_map(points, index)),
        indexed=True,
    )
    solid_map, solid_build_ms, _ = common.build_or_load_map_cache(
        "gravity_transfer_solid_r40_s60_h64",
        lambda points, index: common.solid(canonical_map(points, index))[0],
        indexed=True,
    )

    rows: list[dict] = []
    rows.extend(common.evaluate_sc(
        sc_map,
        map_poses,
        queries,
        algorithm="SC + G",
        query_descriptor_fn=lambda _points, row: common.scan_context(
            canonical_query(row), rings=16),
    ))

    scpp_ring_keys = np.mean(scpp_map, axis=3)
    for query in queries:
        if query["truth_valid"] != "True":
            continue
        started = time.perf_counter()
        descriptor = common.scan_context_plus_single(canonical_query(query))
        ring_key = np.mean(descriptor, axis=1)
        root_distances = np.linalg.norm(
            scpp_ring_keys - ring_key[None, None, :], axis=2)
        frame_distances = np.min(root_distances, axis=1)
        preselected = np.argpartition(
            frame_distances, args.top_k - 1)[:args.top_k]
        exact = []
        for index in preselected:
            distance = min(
                common.official_sc_distance(descriptor, root)[0]
                for root in scpp_map[index])
            exact.append((distance, int(index)))
        exact.sort()
        candidates = np.asarray([index for _, index in exact], dtype=np.int32)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rank, top1_error = common.first_correct_rank(
            candidates, map_poses,
            float(query["truth_x"]), float(query["truth_y"]))
        rows.append(common.make_row(
            "SC++ (PC) + G", query, int(candidates[0]), rank,
            top1_error, elapsed_ms, map_poses,
            top1_score=float(exact[0][0])))

    solid_norms = np.linalg.norm(solid_map, axis=1)
    for query in queries:
        if query["truth_valid"] != "True":
            continue
        started = time.perf_counter()
        descriptor, _ = common.solid(canonical_query(query))
        descriptor_norm = np.linalg.norm(descriptor)
        similarity = solid_map @ descriptor / np.maximum(
            solid_norms * descriptor_norm, 1e-12)
        candidates = np.argsort(-similarity)[:args.top_k]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rank, top1_error = common.first_correct_rank(
            candidates, map_poses,
            float(query["truth_x"]), float(query["truth_y"]))
        rows.append(common.make_row(
            "SOLiD + G", query, int(candidates[0]), rank,
            top1_error, elapsed_ms, map_poses,
            top1_score=float(similarity[candidates[0]])))

    m2dp_started = time.perf_counter()
    m2dp_map = []
    for index, path in enumerate(map_paths):
        points = common.read_binary_pcd(path)
        points = canonical_map(points, index)[:, :3]
        points = m2dp.prepare_points(points, voxel025=True)
        m2dp_map.append(m2dp.m2dp(points))
    m2dp_map = np.asarray(m2dp_map)
    m2dp_build_ms = (time.perf_counter() - m2dp_started) * 1000.0
    map_xy = map_poses[:, 2:4]
    for query in queries:
        if query["truth_valid"] != "True":
            continue
        started = time.perf_counter()
        points = m2dp.prepare_points(canonical_query(query)[:, :3], voxel025=True)
        descriptor = m2dp.m2dp(points)
        distances = m2dp.descriptor_distances(
            m2dp_map, descriptor, pca4=False)
        candidates = np.argsort(distances)[:args.top_k]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        truth = np.asarray(
            [float(query["truth_x"]), float(query["truth_y"])])
        errors = np.linalg.norm(map_xy[candidates] - truth, axis=1)
        matches = np.flatnonzero(errors <= args.correct_radius)
        rank = int(matches[0] + 1) if len(matches) else None
        rows.append(common.make_row(
            "M2DP + G", query, int(candidates[0]), rank,
            float(errors[0]), elapsed_ms, map_poses,
            top1_score=float(distances[candidates[0]])))

    build_times = {
        "SC + G": sc_build_ms,
        "SC++ (PC) + G": scpp_build_ms,
        "SOLiD + G": solid_build_ms,
        "M2DP + G": m2dp_build_ms,
    }
    summaries = summarize(rows, build_times)
    write_csv(args.output_dir / "per_query.csv", rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    for row in summaries:
        print(
            f"{row['algorithm']:16s} "
            f"R@1={100.0 * row['recall_at_1']:.2f}% "
            f"R@5={100.0 * row['recall_at_5']:.2f}% "
            f"R@10={100.0 * row['recall_at_10']:.2f}% "
            f"median={row['retrieval_ms_median']:.2f} ms")


if __name__ == "__main__":
    main()
