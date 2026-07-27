#!/usr/bin/env python3
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np


ROOT = Path("${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/baseline_20260715")
MAP_DIR = Path("${UPDOWN_SC_ROOT}/icra2027_runtime/manual_loop/gravity/key_point_frame")
MAP_POSES = Path("${UPDOWN_SC_ROOT}/icra2027_runtime/manual_loop/gravity/optimized_poses_tum.txt")
QUERY_DIR = Path(os.environ.get(
    "BASELINE_QUERY_DIR", str(ROOT / "queries/loc_2_floor")))
RESULT_DIR = Path(os.environ.get("BASELINE_RESULT_DIR", str(ROOT / "results")))
CACHE = ROOT / "cache/m2dp_official_formula_map2574.npz"
OUTPUT = RESULT_DIR / "m2dp_per_query.csv"
PCA4_CACHE = ROOT / "cache/m2dp_pca4_map2574.npz"
PCA4_OUTPUT = RESULT_DIR / "m2dp_pca4_per_query.csv"
VOXEL025_CACHE = ROOT / "cache/m2dp_voxel025_map2574.npz"
VOXEL025_OUTPUT = RESULT_DIR / "m2dp_voxel025_per_query.csv"
PCA4_VOXEL025_CACHE = ROOT / "cache/m2dp_pca4_voxel025_map2574.npz"
PCA4_VOXEL025_OUTPUT = RESULT_DIR / "m2dp_pca4_voxel025_per_query.csv"
MIN_RADIUS = 0.3
MAX_RADIUS = 30.0
CORRECT_RADIUS = 2.0
TOP_K = 100


def read_binary_pcd(path):
    with path.open("rb") as handle:
        header = {}
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"missing DATA in {path}")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break
        fields = header["FIELDS"]
        dtype = np.dtype([(field, "<f4") for field in fields])
        points = np.fromfile(handle, dtype=dtype, count=int(header["POINTS"][0]))
    return np.column_stack((points["x"], points["y"], points["z"])).astype(np.float64)


def crop(points):
    finite = np.isfinite(points).all(axis=1)
    radius = np.hypot(points[:, 0], points[:, 1])
    return points[finite & (radius >= MIN_RADIUS) & (radius <= MAX_RADIUS)]


def voxel_centroids(points, leaf=0.25):
    if not len(points):
        return points
    cells = np.floor(points[:, :3] / leaf).astype(np.int32)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, 3), dtype=np.float64)
    np.add.at(sums, inverse, points[:, :3])
    return sums / np.bincount(inverse)[:, None]


def prepare_points(points, voxel025=False):
    points = crop(points)
    return voxel_centroids(points) if voxel025 else points


def deterministic_axes(matrix):
    result = matrix.copy()
    for column in range(result.shape[1]):
        pivot = np.argmax(np.abs(result[:, column]))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def descriptor_from_aligned(data):
    azimuths = np.linspace(-np.pi / 2.0, np.pi / 2.0, 4)
    elevations = np.linspace(0.0, np.pi / 2.0, 16)
    theta_edges = np.linspace(-np.pi, np.pi, 17)
    max_rho = np.sqrt(np.max(np.sum(data * data, axis=1)))
    rho_edges = np.linspace(0.0, np.sqrt(max_rho), 9) ** 2
    rho_edges[-1] += 0.001
    signature = np.zeros((64, 128), dtype=np.float64)
    plane = 0
    for azimuth in azimuths:
        for elevation in elevations:
            normal = np.asarray([
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ])
            px = np.asarray([1.0, 0.0, 0.0]) - normal[0] * normal
            py = np.cross(normal, px)
            projected = np.column_stack((data @ px, data @ py))
            theta = np.arctan2(projected[:, 1], projected[:, 0])
            rho = np.hypot(projected[:, 0], projected[:, 1])
            histogram, _, _ = np.histogram2d(theta, rho, bins=(theta_edges, rho_edges))
            signature[plane] = histogram.reshape(-1, order="F") / len(data)
            plane += 1
    u, _, vh = np.linalg.svd(signature, full_matrices=False)
    # Fix the coupled SVD sign ambiguity without changing the descriptor.
    pivot = np.argmax(np.abs(u[:, 0]))
    sign = -1.0 if u[pivot, 0] < 0.0 else 1.0
    return np.concatenate((u[:, 0] * sign, vh[0] * sign)).astype(np.float32)


def m2dp(points):
    """Direct translation of the official LiHeUA/M2DP MATLAB formula."""
    points = crop(points)
    if len(points) < 3:
        return np.zeros(192, dtype=np.float32)
    centered = points - points.mean(axis=0)
    _, _, axes_t = np.linalg.svd(centered, full_matrices=False)
    axes = deterministic_axes(axes_t.T)
    return descriptor_from_aligned(centered @ axes)


def m2dp_pca4(points):
    """M2DP over all four right-handed PCA sign frames.

    PCA eigenvectors are axes, not oriented vectors. Enumerating the four
    orientation-preserving sign choices prevents a rigid rotation from choosing
    a different intrinsic frame. This is reported separately from the literal
    one-frame official MATLAB translation.
    """
    points = crop(points)
    if len(points) < 3:
        return np.zeros((4, 192), dtype=np.float32)
    centered = points - points.mean(axis=0)
    _, _, axes_t = np.linalg.svd(centered, full_matrices=False)
    axes = axes_t.T
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    signs = ((1.0, 1.0, 1.0), (1.0, -1.0, -1.0),
             (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0))
    return np.asarray([
        descriptor_from_aligned(centered @ (axes * np.asarray(sign)))
        for sign in signs
    ])


def map_positions():
    result = []
    for line in MAP_POSES.read_text().splitlines():
        values = [float(value) for value in line.split()]
        result.append(values[1:3])
    return np.asarray(result)


def queries():
    with (QUERY_DIR / "metadata.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def query_points(row):
    return np.fromfile(QUERY_DIR / row["file"], dtype=np.float32).reshape(-1, 4)[:, :3]


def build_map(pca4=False, voxel025=False):
    cache_path = (PCA4_VOXEL025_CACHE if pca4 and voxel025 else
                  PCA4_CACHE if pca4 else
                  VOXEL025_CACHE if voxel025 else CACHE)
    descriptor_function = m2dp_pca4 if pca4 else m2dp
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["descriptors"], float(cached["build_ms"])
    started = time.perf_counter()
    descriptors = []
    paths = sorted(MAP_DIR.glob("*.pcd"), key=lambda path: int(path.stem))
    for index, path in enumerate(paths):
        points = prepare_points(read_binary_pcd(path), voxel025)
        descriptors.append(descriptor_function(points))
        if (index + 1) % 250 == 0:
            print(f"M2DP map {index + 1}/{len(paths)}", flush=True)
    descriptors = np.asarray(descriptors)
    build_ms = (time.perf_counter() - started) * 1000.0
    np.savez(cache_path, descriptors=descriptors, build_ms=build_ms)
    return descriptors, build_ms


def descriptor_distances(descriptors, descriptor, pca4=False):
    if not pca4:
        direct = np.linalg.norm(descriptors - descriptor[None, :], axis=1)
        flipped = np.linalg.norm(descriptors + descriptor[None, :], axis=1)
        return np.minimum(direct, flipped)
    distances = np.full(len(descriptors), np.inf, dtype=np.float32)
    for query_variant in descriptor:
        direct = np.linalg.norm(
            descriptors - query_variant[None, None, :], axis=2)
        flipped = np.linalg.norm(
            descriptors + query_variant[None, None, :], axis=2)
        distances = np.minimum(
            distances, np.minimum(direct, flipped).min(axis=1))
    return distances


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pca4", action="store_true",
        help="evaluate all four right-handed PCA sign frames")
    parser.add_argument(
        "--voxel025", action="store_true",
        help="apply the mapping database's 0.25 m voxel sampling to both sessions")
    args = parser.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    descriptors, build_ms = build_map(args.pca4, args.voxel025)
    positions = map_positions()
    rows = []
    valid_queries = [query for query in queries() if query["truth_valid"] == "True"]
    for query in valid_queries:
        started = time.perf_counter()
        points = prepare_points(query_points(query), args.voxel025)
        descriptor = (m2dp_pca4 if args.pca4 else m2dp)(points)
        # The official descriptor is compared in Euclidean space. The minimum
        # handles only the unavoidable coupled sign ambiguity of an SVD pair.
        distances = descriptor_distances(descriptors, descriptor, args.pca4)
        candidates = np.argsort(distances)[:TOP_K]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        truth = np.asarray([float(query["truth_x"]), float(query["truth_y"])])
        errors = np.linalg.norm(positions[candidates] - truth, axis=1)
        matches = np.flatnonzero(errors <= CORRECT_RADIUS)
        rank = int(matches[0] + 1) if len(matches) else None
        top = int(candidates[0])
        rows.append({
            "algorithm": ("M2DP-PCA4-voxel0.25" if args.pca4 and args.voxel025 else
                          "M2DP-PCA4" if args.pca4 else
                          "M2DP-voxel0.25" if args.voxel025 else "M2DP"),
            "query_id": query["query_id"], "window": query["window"],
            "start_s": query["start_s"], "truth_x": query["truth_x"], "truth_y": query["truth_y"],
            "top1_index": top, "top1_x": positions[top, 0], "top1_y": positions[top, 1],
            "top1_error_m": errors[0], "first_correct_rank": rank if rank else "",
            "recall_at_1": rank is not None and rank <= 1,
            "recall_at_5": rank is not None and rank <= 5,
            "recall_at_10": rank is not None and rank <= 10,
            "recall_at_100": rank is not None and rank <= 100,
            "retrieval_ms": elapsed_ms,
        })
        print(f"M2DP query {len(rows)}/{len(valid_queries)}", flush=True)
    output_path = (PCA4_VOXEL025_OUTPUT if args.pca4 and args.voxel025 else
                   PCA4_OUTPUT if args.pca4 else
                   VOXEL025_OUTPUT if args.voxel025 else OUTPUT)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    times = [row["retrieval_ms"] for row in rows]
    errors = [row["top1_error_m"] for row in rows]
    summary = {
        "algorithm": ("M2DP-PCA4-voxel0.25" if args.pca4 and args.voxel025 else
                      "M2DP-PCA4" if args.pca4 else
                      "M2DP-voxel0.25" if args.voxel025 else "M2DP"),
        "recall_at_1": np.mean([row["recall_at_1"] for row in rows]),
        "recall_at_5": np.mean([row["recall_at_5"] for row in rows]),
        "recall_at_10": np.mean([row["recall_at_10"] for row in rows]),
        "recall_at_100": np.mean([row["recall_at_100"] for row in rows]),
        "top1_error_median": percentile(errors, 50), "top1_error_p95": percentile(errors, 95),
        "retrieval_ms_median": percentile(times, 50), "retrieval_ms_p95": percentile(times, 95),
        "map_build_ms": build_ms,
    }
    print(summary)


if __name__ == "__main__":
    main()
