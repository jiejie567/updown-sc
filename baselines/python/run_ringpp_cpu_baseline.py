#!/usr/bin/env python3
"""Faithful CPU port of the released RING++ feature-retrieval path.

The released implementation computes six hand-crafted local point features,
max-pools them into a 120x120 BEV, applies a per-channel Radon transform and a
row FFT, and ranks candidates using circular correlation.  Its public feature
and Radon extensions are CUDA-only.  This adapter translates those formulas to
NumPy/SciPy/scikit-image without changing the descriptor definition.

This is intentionally kept separate from the official repository.  Results
must be labelled "RING++ (faithful CPU port)" until its descriptors are checked
against the released CUDA path on the same point clouds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from skimage.transform import radon


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_retrieval_baselines as common  # noqa: E402


ROOT = common.ROOT
CACHE_ROOT = common.CACHE_DIR / "ringpp_cpu_feat6_r120_s120_z1_20_v025"
RESULT_PATH = common.RESULT_DIR / "ringpp_cpu_per_query.csv"
SUMMARY_PATH = common.RESULT_DIR / "ringpp_cpu_summary.csv"

GRID = 120
CHANNELS = 6
K_NEIGHBORS = 30
XY_BOUND = 70.0
Z_MIN = 1.0
Z_MAX = 20.0
VOXEL_LEAF = 0.25
EPS = 1e-12


def prepare_points(points: np.ndarray) -> np.ndarray:
    """Apply the common sampling/crop, then released RING++ bounds/normalization."""
    points = common.voxel_centroids(common.crop(points), VOXEL_LEAF)[:, :3]
    keep = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] > -XY_BOUND)
        & (points[:, 0] < XY_BOUND)
        & (points[:, 1] > -XY_BOUND)
        & (points[:, 1] < XY_BOUND)
        & (points[:, 2] > Z_MIN)
        & (points[:, 2] < Z_MAX)
    )
    points = points[keep].astype(np.float32, copy=True)
    if len(points):
        points[:, 0] /= XY_BOUND
        points[:, 1] /= XY_BOUND
        points[:, 2] = (points[:, 2] - Z_MIN) / (Z_MAX - Z_MIN)
    return points


def point_features(points: np.ndarray) -> np.ndarray:
    """Vectorized translation of the released 13-feature CUDA kernel, selecting its six channels."""
    n = len(points)
    if n < K_NEIGHBORS:
        return np.zeros((n, CHANNELS), dtype=np.float32)
    _, neighbors = cKDTree(points).query(points, k=K_NEIGHBORS, workers=-1)
    neighborhoods = points[neighbors]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True) / (K_NEIGHBORS - 1)

    eig3 = np.linalg.eigvalsh(covariance)[:, ::-1]
    eig2 = np.linalg.eigvalsh(covariance[:, :2, :2])[:, ::-1]
    eig_sum = eig3.sum(axis=1)
    eig_prod = eig3.prod(axis=1)

    curvature = np.divide(eig3[:, 2], eig_sum, out=np.zeros(n), where=eig_sum > EPS)
    omnivariance = np.cbrt(
        np.divide(eig_prod, eig_sum**3, out=np.zeros(n), where=eig_sum > EPS)
    )
    probabilities = np.divide(
        eig3,
        eig_sum[:, None],
        out=np.zeros_like(eig3),
        where=eig_sum[:, None] > EPS,
    )
    entropy = -np.sum(
        np.where(probabilities > 0.0, probabilities * np.log(np.maximum(probabilities, EPS)), 0.0),
        axis=1,
    )
    linearity_2d = np.divide(eig2[:, 1], eig2[:, 0], out=np.zeros(n), where=eig2[:, 0] > EPS)
    local_z = neighborhoods[:, :, 2]
    delta_z = local_z.max(axis=1) - local_z.min(axis=1)
    variance_z = local_z.var(axis=1)

    features = np.column_stack(
        (curvature, omnivariance, entropy, linearity_2d, delta_z, variance_z)
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def feature_bev(points: np.ndarray) -> np.ndarray:
    """Translate the released CUDA max-pooling grid into deterministic NumPy operations."""
    features = point_features(points)
    bev = np.zeros((CHANNELS, GRID, GRID), dtype=np.float32)
    if not len(points):
        return bev
    clipped = np.clip(points, -0.9999, 0.9999)
    ix = np.floor((clipped[:, 0] + 1.0) * GRID / 2.0).astype(np.int32)
    iy = np.floor((clipped[:, 1] + 1.0) * GRID / 2.0).astype(np.int32)
    for channel in range(CHANNELS):
        np.maximum.at(bev[channel], (ix, iy), features[:, channel])
    return bev


def tiring_descriptor(points: np.ndarray) -> np.ndarray:
    """Generate the released feature RING followed by its translation-invariant row FFT."""
    normalized = prepare_points(points)
    bev = feature_bev(normalized)
    theta = np.linspace(0.0, 360.0, GRID, endpoint=False, dtype=np.float32)
    ring = np.stack(
        [radon(bev[channel], theta=theta, circle=True).T for channel in range(CHANNELS)],
        axis=0,
    ).astype(np.float32)
    tiring = np.abs(np.fft.fft(ring, axis=-1, norm="ortho"))
    return tiring.astype(np.float32)


def instance_normalize(batch: np.ndarray) -> np.ndarray:
    """Match affine-free InstanceNorm2d used by the released circular matcher."""
    mean = batch.mean(axis=(-2, -1), keepdims=True)
    variance = batch.var(axis=(-2, -1), keepdims=True)
    return (batch - mean) / np.sqrt(variance + 1e-5)


def circular_scores(query: np.ndarray, maps: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Return the released max circular-correlation score for every map descriptor."""
    query_fft = np.fft.fft(instance_normalize(query[None])[0], axis=1, norm="ortho")
    scores = np.empty(len(maps), dtype=np.float32)
    denominator = 0.15 * CHANNELS * GRID * GRID
    for start in range(0, len(maps), batch_size):
        stop = min(start + batch_size, len(maps))
        candidate = instance_normalize(np.asarray(maps[start:stop], dtype=np.float32))
        candidate_fft = np.fft.fft(candidate, axis=2, norm="ortho")
        corr = np.fft.ifft(
            query_fft[None] * np.conj(candidate_fft), axis=2, norm="ortho"
        )
        curve = np.abs(corr).sum(axis=(1, 3))
        scores[start:stop] = curve.max(axis=1) / denominator
    return scores


def cache_paths() -> tuple[Path, Path, Path]:
    return (
        CACHE_ROOT / "map_tiring.npy",
        CACHE_ROOT / "map_progress.json",
        CACHE_ROOT / "metadata.json",
    )


def build_map_cache(limit: int | None = None) -> tuple[np.ndarray, int]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    map_path, progress_path, metadata_path = cache_paths()
    paths = sorted(common.MAP_DIR.glob("*.pcd"), key=lambda path: int(path.stem))
    total = len(paths)
    requested = total if limit is None else min(limit, total)

    if map_path.exists():
        descriptors = np.lib.format.open_memmap(map_path, mode="r+")
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"completed": 0}
        completed = int(progress["completed"])
    else:
        descriptors = np.lib.format.open_memmap(
            map_path, mode="w+", dtype=np.float32, shape=(total, CHANNELS, GRID, GRID)
        )
        completed = 0
        progress_path.write_text(json.dumps({"completed": 0, "total": total}, indent=2) + "\n")

    started = time.perf_counter()
    for index in range(completed, requested):
        if int(paths[index].stem) != index:
            raise RuntimeError(f"Non-contiguous map frame index at {paths[index]}")
        frame_started = time.perf_counter()
        descriptors[index] = tiring_descriptor(common.read_binary_pcd(paths[index]))
        descriptors.flush()
        elapsed = time.perf_counter() - frame_started
        progress_path.write_text(
            json.dumps(
                {
                    "completed": index + 1,
                    "total": total,
                    "last_frame_s": elapsed,
                    "updated_unix_s": time.time(),
                },
                indent=2,
            )
            + "\n"
        )
        if index < 3 or (index + 1) % 25 == 0:
            print(f"map {index + 1}/{total}: {elapsed:.3f} s", flush=True)

    metadata_path.write_text(
        json.dumps(
            {
                "algorithm": "RING++ (faithful CPU port)",
                "source_repository": "lus6-Jenny/RING",
                "channels": ["C", "O", "E", "L2", "dZ", "vZ"],
                "grid": [GRID, GRID],
                "k_neighbors": K_NEIGHBORS,
                "common_radius_m": [common.MIN_RADIUS, common.MAX_RADIUS],
                "voxel_leaf_m": VOXEL_LEAF,
                "released_bounds": {"xy_m": [-XY_BOUND, XY_BOUND], "z_m": [Z_MIN, Z_MAX]},
                "completed": requested,
                "total": total,
                "last_invocation_s": time.perf_counter() - started,
                "validation_status": "CPU formula port; pending same-cloud CUDA descriptor comparison",
            },
            indent=2,
        )
        + "\n"
    )
    return descriptors, requested


def summarize(rows: list[dict], map_build_ms: float | str = "") -> dict:
    ranks = [int(row["first_correct_rank"]) if row["first_correct_rank"] != "" else None for row in rows]
    times = np.asarray([float(row["retrieval_ms"]) for row in rows])
    errors = np.asarray([float(row["top1_error_m"]) for row in rows])
    return {
        "algorithm": "RING++ (faithful CPU port)",
        "truth_queries": len(rows),
        "recall_at_1": np.mean([rank is not None and rank <= 1 for rank in ranks]),
        "recall_at_5": np.mean([rank is not None and rank <= 5 for rank in ranks]),
        "recall_at_10": np.mean([rank is not None and rank <= 10 for rank in ranks]),
        "recall_at_100": np.mean([rank is not None and rank <= 100 for rank in ranks]),
        "top1_error_median": np.median(errors),
        "top1_error_p95": np.percentile(errors, 95),
        "retrieval_ms_median": np.median(times),
        "retrieval_ms_p95": np.percentile(times, 95),
        "map_build_ms": map_build_ms,
        "validation_status": "pending same-cloud CUDA descriptor comparison",
    }


_WORKER_MAPS = None
_WORKER_POSES = None


def _worker_init(map_path: str) -> None:
    global _WORKER_MAPS, _WORKER_POSES
    _WORKER_MAPS = np.lib.format.open_memmap(map_path, mode="r")
    _WORKER_POSES = common.load_map_poses()


def _evaluate_query(row: dict, maps: np.ndarray, poses: np.ndarray) -> dict:
    started = time.perf_counter()
    descriptor = tiring_descriptor(common.load_query_points(row))
    scores = circular_scores(descriptor, maps)
    candidates = np.argsort(-scores)[: common.TOP_K]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    rank, top1_error = common.first_correct_rank(
        candidates, poses, float(row["truth_x"]), float(row["truth_y"])
    )
    result = common.make_row(
        "RING++ (faithful CPU port)",
        row,
        int(candidates[0]),
        rank,
        top1_error,
        elapsed_ms,
        poses,
    )
    result["top1_score"] = float(scores[candidates[0]])
    return result


def _evaluate_query_worker(row: dict) -> dict:
    return _evaluate_query(row, _WORKER_MAPS, _WORKER_POSES)


def evaluate(maps: np.ndarray, workers: int = 1) -> list[dict]:
    poses = common.load_map_poses()
    queries = [row for row in common.load_queries() if row["truth_valid"] == "True"]
    rows = []
    if workers > 1:
        map_path = str(cache_paths()[0])
        context = mp.get_context("fork")
        pool = context.Pool(workers, initializer=_worker_init, initargs=(map_path,))
        iterator = pool.imap(_evaluate_query_worker, queries, chunksize=1)
    else:
        pool = None
        iterator = (_evaluate_query(row, maps, poses) for row in queries)
    try:
        for result in iterator:
            rows.append(result)
            print(
                f"query {result['query_id']}: top1={result['top1_index']} "
                f"rank={result['first_correct_rank'] or None} "
                f"error={float(result['top1_error_m']):.2f}m "
                f"time={float(result['retrieval_ms']):.1f}ms",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    common.write_csv(RESULT_PATH, rows)
    summary = summarize(rows)
    common.write_csv(SUMMARY_PATH, [summary])
    print(summary)
    return rows


def self_test() -> None:
    source = np.load(Path("${UPDOWN_SC_ROOT}/icra2027_runtime/baselines/RING/test.npy"))[:, :3]
    angle = math.radians(37.0)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    transformed = source @ rotation.T + np.asarray([1.5, -0.8, 0.0], dtype=np.float32)
    started = time.perf_counter()
    first = tiring_descriptor(np.column_stack((source, np.zeros(len(source), dtype=np.float32))))
    second = tiring_descriptor(np.column_stack((transformed, np.zeros(len(transformed), dtype=np.float32))))
    scores = circular_scores(first, np.stack((first, second)))
    print(
        json.dumps(
            {
                "source_points": int(len(source)),
                "descriptor_shape": list(first.shape),
                "finite": bool(np.isfinite(first).all() and np.isfinite(second).all()),
                "self_score": float(scores[0]),
                "rigid_transform_score": float(scores[1]),
                "elapsed_s": time.perf_counter() - started,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--max-map-frames", type=int)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    started = time.perf_counter()
    maps, completed = build_map_cache(args.max_map_frames)
    if args.build_only or completed < len(maps):
        print(f"cache complete through frame {completed}; evaluation skipped")
        return
    build_ms = (time.perf_counter() - started) * 1000.0
    rows = evaluate(maps, max(1, args.workers))
    summary = summarize(rows, build_ms)
    common.write_csv(SUMMARY_PATH, [summary])


if __name__ == "__main__":
    main()
