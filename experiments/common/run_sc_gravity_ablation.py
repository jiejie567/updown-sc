#!/usr/bin/env python3
"""Compare conventional Scan Context with and without gravity canonicalization.

Only the point-frame rotation changes. Both variants use the same native
single-frame clouds, 20x60 max-height descriptor, voxel/crop preprocessing,
ring-key shortlist, and released SC sector-key local yaw search.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np


AUDITED_IMPLEMENTATION = Path(
    "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/"
    "baseline_20260715/scripts/run_retrieval_baselines.py"
)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("sc_gravity_common", path)
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


def summarize(rows: list[dict], protocol: str) -> dict:
    times = np.asarray([float(row["retrieval_ms"]) for row in rows])
    errors = np.asarray([float(row["top1_error_m"]) for row in rows])
    return {
        "algorithm": rows[0]["algorithm"],
        "protocol": protocol,
        "truth_queries": len(rows),
        **{
            key: float(np.mean([bool(row[key]) for row in rows]))
            for key in (
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "recall_at_100",
            )
        },
        "top1_error_median": float(np.median(errors)),
        "top1_error_p95": float(np.percentile(errors, 95)),
        "retrieval_ms_median": float(np.median(times)),
        "retrieval_ms_p95": float(np.percentile(times, 95)),
    }


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

    common = load_module(AUDITED_IMPLEMENTATION)
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
    map_count = len(list(args.map_dir.glob("*.pcd")))
    if len(map_poses) != map_count or len(map_gravity) != map_count:
        raise RuntimeError(
            "Map pose/cloud/gravity mismatch: "
            f"{len(map_poses)}/{map_count}/{len(map_gravity)}"
        )
    if len(query_gravity) != len(queries):
        raise RuntimeError(
            f"Query metadata/gravity mismatch: {len(queries)}/{len(query_gravity)}"
        )

    native_map, _, _ = common.build_or_load_map_cache(
        "sc20_native",
        common.scan_context,
    )
    gravity_map, _, _ = common.build_or_load_map_cache(
        "sc20_gravity",
        lambda points, index: common.scan_context(
            common.gravity_canonicalize(points, map_gravity[index])
        ),
        indexed=True,
    )

    native_rows = common.evaluate_sc(
        native_map,
        map_poses,
        queries,
        algorithm="SC",
    )
    gravity_rows = common.evaluate_sc(
        gravity_map,
        map_poses,
        queries,
        algorithm="SC + gravity",
        query_descriptor_fn=lambda points, row: common.scan_context(
            common.gravity_canonicalize(
                points,
                query_gravity[int(row["query_id"])],
            )
        ),
    )
    all_rows = native_rows + gravity_rows
    summaries = [
        summarize(native_rows, "native_base_link_single_scan"),
        summarize(
            gravity_rows,
            "gravity_canonicalized_single_scan_max_height_sc",
        ),
    ]
    write_csv(args.output_dir / "per_query.csv", all_rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    for row in summaries:
        print(
            f"{row['algorithm']}: "
            f"R@1={100.0 * row['recall_at_1']:.2f}% "
            f"R@5={100.0 * row['recall_at_5']:.2f}% "
            f"median={row['retrieval_ms_median']:.3f} ms"
        )


if __name__ == "__main__":
    main()
