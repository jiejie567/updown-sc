#!/usr/bin/env python3
import csv
import math
import os
import time
from pathlib import Path

import numpy as np


ROOT = Path("${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/baseline_20260715")
MAP_DIR = Path("${UPDOWN_SC_ROOT}/icra2027_runtime/manual_loop/gravity/key_point_frame")
MAP_POSES = Path("${UPDOWN_SC_ROOT}/icra2027_runtime/manual_loop/gravity/optimized_poses_tum.txt")
MAP_GRAVITY = Path("${UPDOWN_SC_ROOT}/icra2027_runtime/manual_loop/gravity/scan_context_gravity.csv")
QUERY_DIR = Path(os.environ.get(
    "BASELINE_QUERY_DIR", str(ROOT / "queries/loc_2_floor")))
QUERY_GRAVITY = QUERY_DIR / "gravity.csv"
CACHE_DIR = ROOT / "cache"
RESULT_DIR = Path(os.environ.get("BASELINE_RESULT_DIR", str(ROOT / "results")))
UPDOWN_RESULTS = Path(os.environ.get(
    "UPDOWN_RESULTS",
    "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/two_bag_pilot_20260715/"
    "results/loc_2_floor_sector_local_w04_w06"))

MAX_RADIUS = 30.0
MIN_RADIUS = 0.3
CORRECT_RADIUS = 2.0
TOP_K = 100


def read_binary_pcd(path):
    with path.open("rb") as handle:
        header = {}
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Missing DATA header in {path}")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break
        if header["DATA"][0].lower() != "binary":
            raise RuntimeError(f"Only binary PCD is supported: {path}")
        fields = header["FIELDS"]
        sizes = [int(value) for value in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        if any(size != 4 or kind != "F" or count != 1
               for size, kind, count in zip(sizes, types, counts)):
            raise RuntimeError(f"Unsupported PCD layout: {path}")
        dtype = np.dtype([(field, "<f4") for field in fields])
        points = np.fromfile(handle, dtype=dtype, count=int(header["POINTS"][0]))
    intensity = points["intensity"] if "intensity" in fields else np.zeros(len(points), np.float32)
    return np.column_stack((points["x"], points["y"], points["z"], intensity)).astype(
        np.float32, copy=False)


def crop(points):
    finite = np.isfinite(points[:, :3]).all(axis=1)
    radius_sq = np.square(points[:, 0]) + np.square(points[:, 1])
    keep = finite & (radius_sq >= MIN_RADIUS**2) & (radius_sq <= MAX_RADIUS**2)
    return points[keep]


def voxel_centroids(points, leaf=1.0):
    if not len(points):
        return points
    cells = np.floor(points[:, :3] / leaf).astype(np.int32)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, points.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, points)
    counts = np.bincount(inverse)
    return (sums / counts[:, None]).astype(np.float32)


def load_gravity(path, index_column):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for expected, row in enumerate(rows):
        if int(row[index_column]) != expected:
            raise RuntimeError(f"Non-contiguous gravity index in {path}: {row[index_column]}")
        up = np.asarray(
            [float(row["up_x"]), float(row["up_y"]), float(row["up_z"])],
            dtype=np.float64)
        norm = np.linalg.norm(up)
        if not np.isfinite(up).all() or norm < 1e-12:
            raise RuntimeError(f"Invalid gravity vector {expected} in {path}")
        result.append(up / norm)
    return np.asarray(result)


def gravity_canonical_rotation(up):
    source = np.asarray(up, dtype=np.float64)
    source /= np.linalg.norm(source)
    target = np.asarray([0.0, 0.0, 1.0])
    cross = np.cross(source, target)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-12:
        return np.eye(3) if dot > 0.0 else np.diag([1.0, -1.0, -1.0])
    axis = cross / cross_norm
    skew = np.asarray([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    angle = math.atan2(cross_norm, dot)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def gravity_canonicalize(points, up):
    result = points.copy()
    result[:, :3] = points[:, :3] @ gravity_canonical_rotation(up).T
    return result


def scan_context(points, rings=20, sectors=60, lidar_height=2.0,
                 voxel_leaf=0.25):
    # The saved mapping keyframes already use the FAST-LIO Scan Context
    # 0.25 m voxel. Apply the same sampling to queries (and idempotently to the
    # map frames) so density differs only because of scene visibility.
    points = voxel_centroids(crop(points), voxel_leaf)
    result = np.full((rings, sectors), -1000.0, dtype=np.float32)
    if len(points):
        radius = np.hypot(points[:, 0], points[:, 1])
        theta = np.mod(np.degrees(np.arctan2(points[:, 1], points[:, 0])), 360.0)
        ring = np.clip(np.ceil(radius / MAX_RADIUS * rings).astype(np.int32), 1, rings) - 1
        sector = np.clip(np.ceil(theta / 360.0 * sectors).astype(np.int32), 1, sectors) - 1
        np.maximum.at(result, (ring, sector), points[:, 2] + lidar_height)
    result[result == -1000.0] = 0.0
    return result


def scan_context_plus_single(points, y_shift=0.0, rings=20, sectors=60):
    """Official SC++ Polar Context preprocessing for one virtual root.

    The released T-RO code uses 0.5 m grid-average downsampling, rejects polar
    cells with fewer than five downsampled points, and augments each database
    scan with virtual roots translated by +/-2 m in the lateral direction.
    """
    points = voxel_centroids(crop(points), 0.5).copy()
    result = np.zeros((rings, sectors), dtype=np.float32)
    if not len(points):
        return result
    points[:, 1] -= y_shift
    radius = np.hypot(points[:, 0], points[:, 1])
    theta = np.mod(np.degrees(np.arctan2(points[:, 1], points[:, 0])), 360.0)
    ring = np.minimum(np.floor(radius / (MAX_RADIUS / rings)).astype(np.int32), rings - 1)
    sector = np.maximum(np.ceil(theta / (360.0 / sectors)).astype(np.int32), 1) - 1
    sector = np.minimum(sector, sectors - 1)
    # The released T-RO Polar Context code stores raw maximum z. Unlike the
    # original IROS Scan Context implementation, it does not add lidar height.
    values = points[:, 2]
    counts = np.zeros((rings, sectors), dtype=np.int32)
    maxima = np.full((rings, sectors), -np.inf, dtype=np.float32)
    np.add.at(counts, (ring, sector), 1)
    np.maximum.at(maxima, (ring, sector), values)
    valid = counts >= 5
    result[valid] = maxima[valid]
    return result


def scan_context_plus_map(points):
    # First three virtual roots in the official tree-merge evaluation: the
    # sensor origin followed by lateral roots at -2 m and +2 m.
    return np.asarray([
        scan_context_plus_single(points, y_shift)
        for y_shift in (0.0, -2.0, 2.0)
    ], dtype=np.float32)


def circular_shift(matrix, shift):
    return np.roll(matrix, shift, axis=1)


def direct_sc_distance(query, candidate):
    query_norm = np.linalg.norm(query, axis=0)
    candidate_norm = np.linalg.norm(candidate, axis=0)
    valid = (query_norm > 0.0) & (candidate_norm > 0.0)
    if not np.any(valid):
        return 1.0
    similarities = np.sum(query[:, valid] * candidate[:, valid], axis=0)
    similarities /= query_norm[valid] * candidate_norm[valid]
    return 1.0 - float(np.mean(similarities))


def official_sc_distance(query, candidate):
    query_sector_key = np.mean(query, axis=0)
    candidate_sector_key = np.mean(candidate, axis=0)
    coarse = min(
        range(query.shape[1]),
        key=lambda shift: np.linalg.norm(query_sector_key - np.roll(candidate_sector_key, shift)))
    search_radius = round(0.5 * 0.1 * query.shape[1])
    shifts = {coarse}
    for offset in range(1, search_radius + 1):
        shifts.add((coarse + offset) % query.shape[1])
        shifts.add((coarse - offset) % query.shape[1])
    scored = [(direct_sc_distance(query, circular_shift(candidate, shift)), shift)
              for shift in sorted(shifts)]
    return min(scored)


def exhaustive_sc_distance(query, candidate):
    return min(
        (direct_sc_distance(query, circular_shift(candidate, shift)), shift)
        for shift in range(query.shape[1]))


def solid(points, rings=40, sectors=60, height_bins=64, fov_down=-90.0, fov_up=90.0):
    points = voxel_centroids(crop(points), 1.0)
    ring_height = np.zeros((rings, height_bins), dtype=np.float32)
    sector_height = np.zeros((sectors, height_bins), dtype=np.float32)
    if len(points):
        radius = np.hypot(points[:, 0], points[:, 1])
        theta = np.mod(np.degrees(np.arctan2(points[:, 1], points[:, 0])), 360.0)
        elevation = np.degrees(np.arctan2(points[:, 2], np.maximum(radius, 1e-9)))
        ring = np.clip(np.floor(radius / (MAX_RADIUS / rings)).astype(np.int32), 0, rings - 1)
        sector = np.clip(np.floor(theta / (360.0 / sectors)).astype(np.int32), 0, sectors - 1)
        height = np.floor((elevation - fov_down) / ((fov_up - fov_down) / height_bins)).astype(np.int32)
        valid = (height >= 0) & (height < height_bins)
        np.add.at(ring_height, (ring[valid], height[valid]), 1.0)
        np.add.at(sector_height, (sector[valid], height[valid]), 1.0)
    number = np.sum(ring_height, axis=0)
    span = float(number.max() - number.min())
    weights = (number - number.min()) / span if span > 0.0 else np.zeros_like(number)
    return ring_height @ weights, sector_height @ weights


def load_map_poses():
    poses = []
    with MAP_POSES.open() as handle:
        for index, line in enumerate(handle):
            values = [float(value) for value in line.split()]
            poses.append((index, values[0], values[1], values[2], values[3]))
    return np.asarray(poses, dtype=np.float64)


def load_queries():
    with (QUERY_DIR / "metadata.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_query_points(row):
    return np.fromfile(QUERY_DIR / row["file"], dtype=np.float32).reshape(-1, 4)


def build_or_load_map_cache(name, descriptor_fn, indexed=False):
    cache_path = CACHE_DIR / f"{name}_map.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["descriptors"], float(cached["build_ms"]), cache_path.stat().st_size
    started = time.perf_counter()
    descriptors = []
    paths = sorted(MAP_DIR.glob("*.pcd"), key=lambda path: int(path.stem))
    for index, path in enumerate(paths):
        if int(path.stem) != index:
            raise RuntimeError(f"Non-contiguous map frame index at {path}")
        points = read_binary_pcd(path)
        descriptors.append(
            descriptor_fn(points, index) if indexed else descriptor_fn(points))
    descriptors = np.asarray(descriptors, dtype=np.float32)
    build_ms = (time.perf_counter() - started) * 1000.0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, descriptors=descriptors, build_ms=np.asarray(build_ms))
    return descriptors, build_ms, cache_path.stat().st_size


def first_correct_rank(candidate_indices, map_poses, truth_x, truth_y):
    xy = map_poses[candidate_indices, 2:4]
    errors = np.linalg.norm(xy - np.asarray([truth_x, truth_y]), axis=1)
    matches = np.flatnonzero(errors <= CORRECT_RADIUS)
    return int(matches[0] + 1) if len(matches) else None, float(errors[0])


def evaluate_sc(map_descriptors, map_poses, queries, algorithm="SC",
                descriptor_fn=scan_context, exhaustive_yaw=False,
                query_descriptor_fn=None):
    ring_keys = np.mean(map_descriptors, axis=2)
    rows = []
    for query in queries:
        if query["truth_valid"] != "True":
            continue
        started = time.perf_counter()
        points = load_query_points(query)
        descriptor = (query_descriptor_fn(points, query)
                      if query_descriptor_fn is not None else descriptor_fn(points))
        ring_key = np.mean(descriptor, axis=1)
        ring_distances = np.linalg.norm(ring_keys - ring_key[None, :], axis=1)
        preselected = np.argpartition(ring_distances, TOP_K - 1)[:TOP_K]
        distance_fn = exhaustive_sc_distance if exhaustive_yaw else official_sc_distance
        exact = [(distance_fn(descriptor, map_descriptors[index])[0], index)
                 for index in preselected]
        exact.sort()
        candidates = np.asarray([index for _, index in exact], dtype=np.int32)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rank, top1_error = first_correct_rank(
            candidates, map_poses, float(query["truth_x"]), float(query["truth_y"]))
        rows.append(make_row(
            algorithm, query, candidates[0], rank, top1_error, elapsed_ms, map_poses,
            top1_score=float(exact[0][0])))
    return rows


def evaluate_sc_plus(map_descriptors, map_poses, queries):
    # map_descriptors: [map frame, virtual root, ring, sector]
    ring_keys = np.mean(map_descriptors, axis=3)
    rows = []
    for query in queries:
        if query["truth_valid"] != "True":
            continue
        started = time.perf_counter()
        descriptor = scan_context_plus_single(load_query_points(query))
        ring_key = np.mean(descriptor, axis=1)
        root_distances = np.linalg.norm(ring_keys - ring_key[None, None, :], axis=2)
        frame_distances = np.min(root_distances, axis=1)
        preselected = np.argpartition(frame_distances, TOP_K - 1)[:TOP_K]
        exact = []
        for index in preselected:
            distance = min(
                official_sc_distance(descriptor, root_descriptor)[0]
                for root_descriptor in map_descriptors[index])
            exact.append((distance, index))
        exact.sort()
        candidates = np.asarray([index for _, index in exact], dtype=np.int32)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rank, top1_error = first_correct_rank(
            candidates, map_poses, float(query["truth_x"]), float(query["truth_y"]))
        rows.append(make_row(
            "SC++ (PC)", query, candidates[0], rank, top1_error, elapsed_ms, map_poses,
            top1_score=float(exact[0][0])))
    return rows


def evaluate_solid(map_descriptors, map_poses, queries):
    norms = np.linalg.norm(map_descriptors, axis=1)
    rows = []
    for query in queries:
        if query["truth_valid"] != "True":
            continue
        started = time.perf_counter()
        descriptor, _ = solid(load_query_points(query))
        descriptor_norm = np.linalg.norm(descriptor)
        similarity = map_descriptors @ descriptor / np.maximum(norms * descriptor_norm, 1e-12)
        candidates = np.argsort(-similarity)[:TOP_K]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rank, top1_error = first_correct_rank(
            candidates, map_poses, float(query["truth_x"]), float(query["truth_y"]))
        rows.append(make_row(
            "SOLiD", query, candidates[0], rank, top1_error, elapsed_ms, map_poses,
            top1_score=float(similarity[candidates[0]])))
    return rows


def make_row(algorithm, query, top1_index, rank, top1_error, elapsed_ms, map_poses,
             top1_score=None):
    # top1_score keeps each method's native convention: SC/SC++/M2DP/Iris store a
    # distance (smaller is better), SOLiD stores a cosine similarity (larger is
    # better). Empty when the caller predates score export.
    return {
        "algorithm": algorithm,
        "query_id": query["query_id"],
        "window": query["window"],
        "start_s": query["start_s"],
        "truth_x": query["truth_x"],
        "truth_y": query["truth_y"],
        "top1_index": int(top1_index),
        "top1_x": map_poses[top1_index, 2],
        "top1_y": map_poses[top1_index, 3],
        "top1_error_m": top1_error,
        "first_correct_rank": rank if rank is not None else "",
        "recall_at_1": rank is not None and rank <= 1,
        "recall_at_5": rank is not None and rank <= 5,
        "recall_at_10": rank is not None and rank <= 10,
        "recall_at_100": rank is not None and rank <= 100,
        "retrieval_ms": elapsed_ms,
        "top1_score": top1_score if top1_score is not None else "",
    }


def evaluate_updown(map_poses, queries):
    query_by_window = {int(row["window"]): row for row in queries if row["truth_valid"] == "True"}
    grouped = {}
    with (UPDOWN_RESULTS / "scan_context_candidate_hypotheses.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            window = int(row["window"])
            rank = int(row["candidate_rank"])
            grouped.setdefault(window, {}).setdefault(rank, row)
    timings = {}
    with (UPDOWN_RESULTS / "relocalization_windows.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            timings[int(row["window"])] = float(row["scan_context_ms"])

    rows = []
    for window, query in query_by_window.items():
        ranked = grouped.get(window, {})
        if not ranked:
            rows.append({
                "algorithm": "UpDown-SC",
                "query_id": query["query_id"],
                "window": query["window"],
                "start_s": query["start_s"],
                "truth_x": query["truth_x"],
                "truth_y": query["truth_y"],
                "top1_index": -1,
                "top1_x": float("nan"),
                "top1_y": float("nan"),
                "top1_error_m": float("inf"),
                "first_correct_rank": "",
                "recall_at_1": False,
                "recall_at_5": False,
                "recall_at_10": False,
                "recall_at_100": False,
                "retrieval_ms": timings.get(window, 0.0),
            })
            continue
        candidate_rows = [ranked[rank] for rank in sorted(ranked)[:TOP_K]]
        candidate_xy = np.asarray(
            [[float(row["candidate_x"]), float(row["candidate_y"])] for row in candidate_rows])
        errors = np.linalg.norm(
            candidate_xy - np.asarray([float(query["truth_x"]), float(query["truth_y"])]), axis=1)
        matches = np.flatnonzero(errors <= CORRECT_RADIUS)
        rank = int(matches[0] + 1) if len(matches) else None
        top1_index = int(candidate_rows[0]["candidate_index"])
        row = make_row(
            "UpDown-SC", query, top1_index, rank, float(errors[0]), timings[window], map_poses)
        row["top1_x"] = candidate_xy[0, 0]
        row["top1_y"] = candidate_xy[0, 1]
        rows.append(row)
    return rows


def summarize(rows, map_build):
    summaries = []
    for algorithm in (
            "SC", "SC (16x60 matched)", "SC (16x60 + gravity)",
            "SC (32x60 capacity)",
            "SC++ (PC)", "SOLiD", "UpDown-SC"):
        group = [row for row in rows if row["algorithm"] == algorithm]
        times = np.asarray([float(row["retrieval_ms"]) for row in group])
        summaries.append({
            "algorithm": algorithm,
            "truth_queries": len(group),
            "recall_at_1": np.mean([row["recall_at_1"] for row in group]),
            "recall_at_5": np.mean([row["recall_at_5"] for row in group]),
            "recall_at_10": np.mean([row["recall_at_10"] for row in group]),
            "recall_at_100": np.mean([row["recall_at_100"] for row in group]),
            "top1_error_median": np.median([row["top1_error_m"] for row in group]),
            "top1_error_p95": np.percentile([row["top1_error_m"] for row in group], 95),
            "retrieval_ms_median": np.median(times),
            "retrieval_ms_p95": np.percentile(times, 95),
            "map_build_ms": map_build.get(algorithm, ""),
        })
    return summaries


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    map_poses = load_map_poses()
    queries = load_queries()
    map_gravity = load_gravity(MAP_GRAVITY, "index")
    query_gravity = load_gravity(QUERY_GRAVITY, "query_id")
    if len(map_poses) != len(list(MAP_DIR.glob("*.pcd"))):
        raise RuntimeError("Map pose/keyframe count mismatch")

    sc_map, sc_build_ms, _ = build_or_load_map_cache(
        "sc_r20_s60_r30_v025_native_exact2574", scan_context)
    sc_matched_map, sc_matched_build_ms, _ = build_or_load_map_cache(
        "sc_r16_s60_r30_v025_matched_exact2574",
        lambda points: scan_context(points, rings=16))
    sc_capacity_map, sc_capacity_build_ms, _ = build_or_load_map_cache(
        "sc_r32_s60_r30_v025_capacity_exact2574",
        lambda points: scan_context(points, rings=32))
    sc_gravity_map, sc_gravity_build_ms, _ = build_or_load_map_cache(
        "sc_r16_s60_r30_v025_gravity_exact2574",
        lambda points, index: scan_context(
            gravity_canonicalize(points, map_gravity[index]), rings=16),
        indexed=True)
    sc_plus_map, sc_plus_build_ms, _ = build_or_load_map_cache(
        "scpp_pc_r20_s60_r30_v05_rawz_aug3_exact2574", scan_context_plus_map)
    solid_map, solid_build_ms, _ = build_or_load_map_cache(
        "solid_r40_s60_h64_r30_native_exact2574", lambda p: solid(p)[0])
    rows = []
    rows.extend(evaluate_sc(sc_map, map_poses, queries))
    rows.extend(evaluate_sc(
        sc_matched_map, map_poses, queries, "SC (16x60 matched)",
        lambda points: scan_context(points, rings=16), exhaustive_yaw=True))
    rows.extend(evaluate_sc(
        sc_gravity_map, map_poses, queries, "SC (16x60 + gravity)",
        lambda points: scan_context(points, rings=16), exhaustive_yaw=True,
        query_descriptor_fn=lambda points, row: scan_context(
            gravity_canonicalize(points, query_gravity[int(row["query_id"])]),
            rings=16)))
    rows.extend(evaluate_sc(
        sc_capacity_map, map_poses, queries, "SC (32x60 capacity)",
        lambda points: scan_context(points, rings=32), exhaustive_yaw=True))
    rows.extend(evaluate_sc_plus(sc_plus_map, map_poses, queries))
    rows.extend(evaluate_solid(solid_map, map_poses, queries))
    rows.extend(evaluate_updown(map_poses, queries))
    summaries = summarize(rows, {
        "SC": sc_build_ms,
        "SC (16x60 matched)": sc_matched_build_ms,
        "SC (16x60 + gravity)": sc_gravity_build_ms,
        "SC (32x60 capacity)": sc_capacity_build_ms,
        "SC++ (PC)": sc_plus_build_ms,
        "SOLiD": solid_build_ms,
    })
    write_csv(RESULT_DIR / "retrieval_per_query.csv", rows)
    write_csv(RESULT_DIR / "retrieval_summary.csv", summaries)
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
