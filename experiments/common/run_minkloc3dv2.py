#!/usr/bin/env python3
"""Evaluate official pretrained MinkLoc3Dv2 without fine-tuning.

The adapter preserves the paper's database/query split, horizontal crop, and
position-radius correctness rule. It imports the model and quantizer from an
explicit checkout of the official repository and verifies the official
Oxford-baseline checkpoint by SHA-256. Native and gravity-canonicalized inputs
are selected with ``--protocol``.

MinkLoc3Dv2 was trained on PointNetVLAD-format point clouds: ground-removed,
fixed-size local submaps scaled to roughly [-1, 1]. For deterministic single-
scan transfer, this adapter uses the common 0.3--30 m crop, a 0.10 m voxel
prefilter, a deterministic 4096-point cap, and the official PointNetVLAD
centroid/mean-radius normalization.
No test-set labels, fine-tuning, or per-cloud scale fitting are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


EXPECTED_WEIGHTS_SHA256 = (
    "559242b5f97be756c391d1d1d4c622ea6b1ea21d809c559d12584167134d4e79"
)


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_binary_pcd(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header: dict[str, list[str]] = {}
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
        if any(
            size != 4 or kind != "F" or count != 1
            for size, kind, count in zip(sizes, types, counts)
        ):
            raise RuntimeError(f"Unsupported PCD layout: {path}")
        dtype = np.dtype([(field, "<f4") for field in fields])
        points = np.fromfile(handle, dtype=dtype, count=int(header["POINTS"][0]))
    intensity = (
        points["intensity"]
        if "intensity" in fields
        else np.zeros(len(points), dtype=np.float32)
    )
    return np.column_stack(
        (points["x"], points["y"], points["z"], intensity)
    ).astype(np.float32, copy=False)


def read_cloud(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".pcd":
        return read_binary_pcd(path)
    if path.suffix.lower() == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4:
            raise RuntimeError(f"Expected Nx4 float32 cloud: {path}")
        return raw.reshape(-1, 4)
    raise RuntimeError(f"Unsupported cloud format: {path}")


def indexed_clouds(directory: Path) -> list[Path]:
    paths = [*directory.glob("*.pcd"), *directory.glob("*.bin")]
    paths.sort(key=lambda path: int(path.stem))
    for expected, path in enumerate(paths):
        if int(path.stem) != expected:
            raise RuntimeError(f"Non-contiguous cloud index at {path}")
    if not paths:
        raise RuntimeError(f"No PCD or BIN clouds found in {directory}")
    return paths


def load_map_xy(path: Path) -> np.ndarray:
    rows = []
    with path.open() as handle:
        for line in handle:
            values = [float(value) for value in line.split()]
            if len(values) != 8:
                raise RuntimeError(f"Expected TUM pose row in {path}: {line.rstrip()}")
            rows.append(values[1:3])
    return np.asarray(rows, dtype=np.float64)


def load_queries(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"query_id", "file", "truth_valid", "truth_x", "truth_y"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise RuntimeError(f"Missing query columns in {path}: {sorted(missing)}")
    return rows


def load_gravity(path: Path) -> dict[int, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty gravity file: {path}")
    index_column = next(
        (name for name in ("index", "query_id", "keyframe_id") if name in rows[0]),
        None,
    )
    if index_column is None:
        raise RuntimeError(f"No recognized index column in {path}")
    result = {}
    for row in rows:
        index = int(row[index_column])
        up = np.asarray(
            [float(row["up_x"]), float(row["up_y"]), float(row["up_z"])],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(up))
        if not np.isfinite(up).all() or norm < 1e-12:
            raise RuntimeError(f"Invalid gravity vector {index} in {path}")
        result[index] = up / norm
    return result


def gravity_canonical_rotation(up: np.ndarray) -> np.ndarray:
    source = np.asarray(up, dtype=np.float64)
    source /= np.linalg.norm(source)
    target = np.asarray([0.0, 0.0, 1.0])
    cross = np.cross(source, target)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-12:
        return np.eye(3) if dot > 0.0 else np.diag([1.0, -1.0, -1.0])
    axis = cross / cross_norm
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    angle = math.atan2(cross_norm, dot)
    return (
        np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def stable_point_cap(points: np.ndarray, limit: int) -> np.ndarray:
    """Select a deterministic, order-independent subset using coordinate hashes."""
    if len(points) <= limit:
        return points
    millimetres = np.rint(points * 1000.0).astype(np.int64, copy=False)
    keys = (
        millimetres[:, 0].astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)
        ^ millimetres[:, 1].astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
        ^ millimetres[:, 2].astype(np.uint64) * np.uint64(0x165667B19E3779F9)
    )
    selected = np.argpartition(keys, limit - 1)[:limit]
    return points[np.sort(selected)]


def prepare_cloud(
    points: np.ndarray,
    up: np.ndarray | None,
    min_radius: float,
    max_radius: float,
    prevoxel_m: float,
    sample_points: int,
) -> np.ndarray:
    result = points[:, :3].astype(np.float32, copy=True)
    if up is not None:
        result = (result @ gravity_canonical_rotation(up).T).astype(
            np.float32, copy=False
        )
    finite = np.isfinite(result).all(axis=1)
    radius_sq = np.square(result[:, 0]) + np.square(result[:, 1])
    keep = finite & (radius_sq >= min_radius**2) & (radius_sq <= max_radius**2)
    result = result[keep]
    if not len(result):
        raise RuntimeError("Point cloud is empty after the common crop")

    voxel = np.floor(result / prevoxel_m).astype(np.int64)
    _, first = np.unique(voxel, axis=0, return_index=True)
    result = result[np.sort(first)]
    result = stable_point_cap(result, sample_points)

    centroid = np.mean(result, axis=0, dtype=np.float64).astype(np.float32)
    result = result - centroid
    mean_radius = float(np.mean(np.linalg.norm(result, axis=1)))
    if not np.isfinite(mean_radius) or mean_radius < 1e-6:
        raise RuntimeError("Degenerate point cloud for mean-radius normalization")
    result = result * np.float32(0.5 / mean_radius)
    result = result[np.all(np.abs(result) <= 1.0, axis=1)]
    if not len(result):
        raise RuntimeError("Point cloud is empty after official normalization")
    return np.ascontiguousarray(result, dtype=np.float32)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_official_model(args: argparse.Namespace):
    sys.path.insert(0, str(args.minkloc_root))
    import torch
    import MinkowskiEngine as ME
    from misc.utils import ModelParams
    from models.model_factory import model_factory

    torch.set_num_threads(args.threads)
    model_params = ModelParams(str(args.model_config))
    model = model_factory(model_params)
    try:
        state_dict = torch.load(args.weights, map_location="cpu", weights_only=False)
    except TypeError:
        state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return torch, ME, model, model_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minkloc-root", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--map-dir", required=True, type=Path)
    parser.add_argument("--map-poses", required=True, type=Path)
    parser.add_argument("--query-dir", required=True, type=Path)
    parser.add_argument("--query-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", choices=("native", "gravity"), default="native")
    parser.add_argument("--map-gravity", type=Path)
    parser.add_argument("--query-gravity", type=Path)
    parser.add_argument("--correct-radius", required=True, type=float)
    parser.add_argument("--min-radius", type=float, default=0.3)
    parser.add_argument("--max-radius", type=float, default=30.0)
    parser.add_argument("--prevoxel-m", type=float, default=0.10)
    parser.add_argument("--sample-points", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--allow-weight-mismatch", action="store_true")
    args = parser.parse_args()

    if args.protocol == "gravity" and (not args.map_gravity or not args.query_gravity):
        parser.error("gravity protocol requires --map-gravity and --query-gravity")
    if args.prevoxel_m <= 0 or args.sample_points <= 0:
        parser.error("prevoxel and point count must be positive")
    weights_hash = sha256(args.weights)
    if weights_hash != EXPECTED_WEIGHTS_SHA256 and not args.allow_weight_mismatch:
        raise RuntimeError(
            "Unexpected MinkLoc3Dv2 weights SHA-256: "
            f"{weights_hash}; expected {EXPECTED_WEIGHTS_SHA256}"
        )

    map_paths = indexed_clouds(args.map_dir)
    map_xy = load_map_xy(args.map_poses)
    queries = load_queries(args.query_metadata)
    if len(map_paths) != len(map_xy):
        raise RuntimeError(f"Map cloud/pose mismatch: {len(map_paths)}/{len(map_xy)}")
    map_gravity = load_gravity(args.map_gravity) if args.map_gravity else {}
    query_gravity = load_gravity(args.query_gravity) if args.query_gravity else {}
    if args.protocol == "gravity":
        if set(map_gravity) != set(range(len(map_paths))):
            raise RuntimeError("Map gravity indices do not match map clouds")
        query_ids = {int(row["query_id"]) for row in queries}
        if set(query_gravity) != query_ids:
            raise RuntimeError("Query gravity indices do not match query metadata")

    torch, ME, model, model_params = load_official_model(args)

    def descriptor(points: np.ndarray, up: np.ndarray | None) -> np.ndarray:
        cloud = prepare_cloud(
            points,
            up,
            args.min_radius,
            args.max_radius,
            args.prevoxel_m,
            args.sample_points,
        )
        coordinates, _ = model_params.quantizer(torch.from_numpy(cloud))
        batched = ME.utils.batched_coordinates([coordinates])
        features = torch.ones((batched.shape[0], 1), dtype=torch.float32)
        batch = {"coords": batched, "features": features}
        with torch.no_grad():
            embedding = model(batch)["global"]
        return embedding[0].detach().cpu().numpy().astype(np.float32, copy=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    map_started = time.perf_counter()
    map_descriptors = np.asarray(
        [
            descriptor(read_cloud(path), map_gravity.get(index))
            for index, path in enumerate(map_paths)
        ],
        dtype=np.float32,
    )
    map_build_ms = (time.perf_counter() - map_started) * 1000.0
    np.savez_compressed(
        args.output_dir / "map_descriptors.npz", descriptors=map_descriptors
    )

    top_k = min(args.top_k, len(map_paths))
    eligible_queries = [row for row in queries if truthy(row["truth_valid"])]
    query_clouds = {
        int(row["query_id"]): read_cloud(args.query_dir / row["file"])
        for row in eligible_queries
    }
    output = []
    for ordinal, query in enumerate(queries, 1):
        if not truthy(query["truth_valid"]):
            continue
        query_id = int(query["query_id"])
        started = time.perf_counter()
        query_descriptor = descriptor(query_clouds[query_id], query_gravity.get(query_id))
        distances = np.sum(
            np.square(map_descriptors - query_descriptor[None, :]), axis=1
        )
        candidates = np.argsort(distances)[:top_k]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        truth = np.asarray(
            [float(query["truth_x"]), float(query["truth_y"])], dtype=np.float64
        )
        errors = np.linalg.norm(map_xy[candidates] - truth, axis=1)
        matches = np.flatnonzero(errors <= args.correct_radius)
        rank = int(matches[0] + 1) if len(matches) else None
        top = int(candidates[0])
        output.append(
            {
                "algorithm": "MinkLoc3Dv2" + (
                    " + G" if args.protocol == "gravity" else ""
                ),
                "query_id": query_id,
                "truth_x": query["truth_x"],
                "truth_y": query["truth_y"],
                "top1_index": top,
                "top1_x": map_xy[top, 0],
                "top1_y": map_xy[top, 1],
                "top1_error_m": float(errors[0]),
                "top1_distance": float(distances[top]),
                "first_correct_rank": rank if rank is not None else "",
                "recall_at_1": rank is not None and rank <= 1,
                "recall_at_5": rank is not None and rank <= 5,
                "recall_at_10": rank is not None and rank <= 10,
                "retrieval_ms": elapsed_ms,
            }
        )
        if ordinal % 100 == 0 or ordinal == len(queries):
            print(f"queries {ordinal}/{len(queries)}", flush=True)

    ranks = [
        int(row["first_correct_rank"]) if row["first_correct_rank"] != "" else None
        for row in output
    ]
    times = np.asarray([float(row["retrieval_ms"]) for row in output])
    summary = {
        "algorithm": output[0]["algorithm"],
        "protocol": args.protocol + "_single_scan_oxford_pretrained_zero_shot",
        "truth_queries": len(output),
        "recall_at_1": float(np.mean([rank is not None and rank <= 1 for rank in ranks])),
        "recall_at_5": float(np.mean([rank is not None and rank <= 5 for rank in ranks])),
        "recall_at_10": float(np.mean([rank is not None and rank <= 10 for rank in ranks])),
        "retrieval_ms_median": float(np.median(times)),
        "retrieval_ms_p95": float(np.percentile(times, 95)),
        "map_build_ms": map_build_ms,
    }
    write_csv(args.output_dir / "per_query.csv", output)
    write_csv(args.output_dir / "summary.csv", [summary])

    try:
        commit = subprocess.run(
            ["git", "-C", str(args.minkloc_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        commit = "unavailable"
    manifest = {
        "official_repository": "https://github.com/jac99/MinkLoc3Dv2",
        "official_commit": commit,
        "weights_sha256": weights_hash,
        "checkpoint": "minkloc3dv2_baseline.pth",
        "pretraining": "Oxford RobotCar baseline training split",
        "fine_tuning": False,
        "protocol": args.protocol,
        "map_clouds": len(map_paths),
        "queries_total": len(queries),
        "queries_eligible": len(output),
        "correct_radius_m": args.correct_radius,
        "horizontal_crop_m": [args.min_radius, args.max_radius],
        "input_preprocessing": {
            "prevoxel_m": args.prevoxel_m,
            "deterministic_point_cap": args.sample_points,
            "normalization": "PointNetVLAD centroid and mean-radius (s=0.5/d)",
            "per_cloud_centering": True,
            "ground_removal": False,
        },
        "official_model_quantization_step_normalized": model_params.quantization_step,
        "descriptor_dimensions": int(map_descriptors.shape[1]),
        "distance": "squared L2 (rank-equivalent to official L2)",
        "runtime_compatibility": (
            "MinkowskiEngine v0.5.4 CPU build; standard <cstdint> include "
            "added for GCC 13, with model code and weights unchanged"
        ),
        "threads": args.threads,
        "query_input_preloaded_before_timing": True,
        "summary": summary,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
