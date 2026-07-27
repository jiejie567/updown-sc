#!/usr/bin/env python3
"""Run the audited RING++ CPU adapter on gravity-canonicalized scans."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path


AUDITED_SCRIPTS = Path(
    "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/baseline_20260715/scripts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-dir", required=True, type=Path)
    parser.add_argument("--map-poses", required=True, type=Path)
    parser.add_argument("--map-gravity", required=True, type=Path)
    parser.add_argument("--query-dir", required=True, type=Path)
    parser.add_argument("--query-gravity", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--correct-radius", required=True, type=float)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, str(AUDITED_SCRIPTS))
    module_path = AUDITED_SCRIPTS / "run_ringpp_cpu_baseline.py"
    spec = importlib.util.spec_from_file_location(
        "ringpp_gravity_transfer_impl", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    ring = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ring
    spec.loader.exec_module(ring)

    ring.common.MAP_DIR = args.map_dir
    ring.common.MAP_POSES = args.map_poses
    ring.common.MAP_GRAVITY = args.map_gravity
    ring.common.QUERY_DIR = args.query_dir
    ring.common.QUERY_GRAVITY = args.query_gravity
    ring.common.CORRECT_RADIUS = args.correct_radius
    map_count = len(list(args.map_dir.glob("*.pcd")))
    ring.common.TOP_K = min(args.top_k, map_count)
    ring.CACHE_ROOT = args.output_dir / "cache"
    ring.RESULT_PATH = args.output_dir / "results/ringpp_per_query.csv"
    ring.SUMMARY_PATH = args.output_dir / "results/ringpp_summary.csv"
    ring.RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    map_gravity = ring.common.load_gravity(args.map_gravity, "index")
    query_gravity = ring.common.load_gravity(args.query_gravity, "query_id")
    native_read_pcd = ring.common.read_binary_pcd
    native_load_query = ring.common.load_query_points

    def canonical_read_pcd(path: Path):
        points = native_read_pcd(path)
        return ring.common.gravity_canonicalize(
            points, map_gravity[int(path.stem)])

    def canonical_load_query(row: dict):
        points = native_load_query(row)
        return ring.common.gravity_canonicalize(
            points, query_gravity[int(row["query_id"])])

    ring.common.read_binary_pcd = canonical_read_pcd
    ring.common.load_query_points = canonical_load_query

    started = time.perf_counter()
    maps, completed = ring.build_map_cache()
    if completed != len(maps):
        raise RuntimeError(f"Incomplete RING++ cache: {completed}/{len(maps)}")
    rows = ring.evaluate(maps, max(1, args.workers))
    for row in rows:
        row["algorithm"] = "RING++ + G"
    ring.common.write_csv(ring.RESULT_PATH, rows)
    summary = ring.summarize(rows, (time.perf_counter() - started) * 1000.0)
    summary["algorithm"] = "RING++ + G"
    summary["protocol"] = (
        "gravity_canonicalized_single_scan_faithful_cpu_port")
    ring.common.write_csv(ring.SUMMARY_PATH, [summary])
    print(summary)


if __name__ == "__main__":
    main()
