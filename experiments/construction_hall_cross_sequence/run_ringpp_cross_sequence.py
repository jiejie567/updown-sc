#!/usr/bin/env python3
"""Run the previously audited RING++ CPU formula port on Construction Hall."""

import argparse
import importlib.util
import sys
import time
from pathlib import Path


AUDITED_SCRIPTS = Path(
    "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/baseline_20260715/scripts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--correct-radius", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, str(AUDITED_SCRIPTS))
    module_path = AUDITED_SCRIPTS / "run_ringpp_cpu_baseline.py"
    spec = importlib.util.spec_from_file_location("construction_ringpp", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    ring = importlib.util.module_from_spec(spec)
    # Multiprocessing pickles worker functions by their module name.  Register
    # the dynamically loaded audited implementation before executing it so
    # spawned/forked workers can resolve that name.
    sys.modules[spec.name] = ring
    spec.loader.exec_module(ring)

    ring.common.ROOT = args.experiment_root
    ring.common.MAP_DIR = args.experiment_root / "seq1/session/key_point_frame"
    ring.common.MAP_POSES = args.experiment_root / "seq1/session/optimized_poses_tum.txt"
    ring.common.QUERY_DIR = args.experiment_root / "derived/queries_seq2"
    ring.common.QUERY_GRAVITY = ring.common.QUERY_DIR / "gravity.csv"
    ring.common.CACHE_DIR = args.experiment_root / "cache"
    ring.common.RESULT_DIR = args.experiment_root / "results"
    ring.common.CORRECT_RADIUS = args.correct_radius
    map_count = len(list(ring.common.MAP_DIR.glob("*.pcd")))
    ring.common.TOP_K = min(100, map_count)
    ring.CACHE_ROOT = args.experiment_root / "cache/ringpp_cpu"
    ring.RESULT_PATH = args.experiment_root / "results/ringpp_per_query.csv"
    ring.SUMMARY_PATH = args.experiment_root / "results/ringpp_summary.csv"

    started = time.perf_counter()
    maps, completed = ring.build_map_cache()
    if completed != len(maps):
        raise RuntimeError(f"Incomplete RING++ map cache: {completed}/{len(maps)}")
    rows = ring.evaluate(maps, max(1, args.workers))
    summary = ring.summarize(rows, (time.perf_counter() - started) * 1000.0)
    summary["protocol"] = "native_base_link_single_scan_faithful_cpu_port"
    ring.common.write_csv(ring.SUMMARY_PATH, [summary])


if __name__ == "__main__":
    main()
