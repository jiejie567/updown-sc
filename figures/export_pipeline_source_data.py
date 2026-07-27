#!/usr/bin/env python3
"""Export measured source data for the Fig. 2 pipeline schematic.

Selects one running example from the paper's authoritative IH+G 2 m retrieval
evaluation (final channel weights): among queries whose top-1 retrieval is
correct, pick the one whose dual-envelope descriptor has the most mixed cells
(the published Fig. 3 selection rule applied to the 2 m set). Exports the
gravity-canonicalized query cloud, its Up/Down envelopes, the ranked top-100
candidate distances, the matched map keyframe cloud, and the experimental map
trajectory into ``pipeline_source_data.npz`` so the figure regenerates without
the runtime tree. The envelope computation was validated bit-exact against
``descriptor_detail_source_data.csv`` on its published query.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

RT = Path.home() / "icra2027_runtime"
EXP = RT / "experiments/private_two_bag_2m"
SELECTED = RT / "experiments/updown_weight_ablation_real_20260721/selected/ih_gravity"
OUT = Path(__file__).resolve().parent

TAU = 2.1
RINGS, SECTORS, RADIUS, VOXEL = 16, 60, 30.0, 0.25
CORRECT_RADIUS = 2.0


def canonical_rotation(up: np.ndarray) -> np.ndarray:
    up = up / np.linalg.norm(up)
    target = np.array([0.0, 0.0, 1.0])
    cross = np.cross(up, target)
    dot = float(np.clip(up @ target, -1.0, 1.0))
    norm = float(np.linalg.norm(cross))
    if norm < 1e-12:
        return np.eye(3) if dot > 0 else np.diag([1.0, -1.0, -1.0])
    axis = cross / norm
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    angle = np.arctan2(norm, dot)
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def crop(xyz: np.ndarray) -> np.ndarray:
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    keep = np.isfinite(xyz).all(axis=1) & (radius >= 0.3) & (radius <= RADIUS)
    return xyz[keep]


def voxel_centroids(xyz: np.ndarray, leaf: float) -> np.ndarray:
    cells = np.floor(xyz / leaf).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, 3))
    np.add.at(sums, inverse, xyz)
    return (sums / np.bincount(inverse)[:, None]).astype(np.float32)


def envelopes(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = voxel_centroids(crop(xyz), VOXEL)
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    theta = np.mod(np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])), 360.0)
    ring = np.clip(np.ceil(radius / RADIUS * RINGS).astype(int), 1, RINGS) - 1
    sector = np.clip(np.ceil(theta / 360.0 * SECTORS).astype(int), 1, SECTORS) - 1
    up = np.full((RINGS, SECTORS), np.nan)
    down = np.full((RINGS, SECTORS), np.nan)
    for (i, j, z) in zip(ring, sector, xyz[:, 2]):
        if z <= TAU:
            if np.isnan(down[i, j]) or z > down[i, j]:
                down[i, j] = z
        else:
            if np.isnan(up[i, j]) or z < up[i, j]:
                up[i, j] = z
    return up, down


def read_binary_pcd(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = {}
        while True:
            text = handle.readline().decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break
        fields = header["FIELDS"]
        dtype = np.dtype([(f, "<f4") for f in fields])
        points = np.fromfile(handle, dtype=dtype, count=int(header["POINTS"][0]))
    return np.column_stack([points["x"], points["y"], points["z"]]).astype(np.float32)


def load_gravity(path: Path, key: str) -> list[np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    out = [None] * len(rows)
    for row in rows:
        up = np.array([float(row["up_x"]), float(row["up_y"]), float(row["up_z"])])
        out[int(row[key])] = up / np.linalg.norm(up)
    return out


def main() -> None:
    queries = list(csv.DictReader((EXP / "derived/queries/metadata.csv").open()))
    query_gravity = load_gravity(EXP / "derived/queries/gravity.csv", "query_id")
    candidates: dict[int, list[dict]] = {}
    with (SELECTED / "candidates.csv").open() as handle:
        for row in csv.DictReader(handle):
            candidates.setdefault(int(row["query_index"]), []).append(row)
    for rows in candidates.values():
        rows.sort(key=lambda r: int(r["candidate_rank"]))

    best = None
    for row in queries:
        if row["truth_valid"] != "True":
            continue
        qid = int(row["query_id"])
        ranked = candidates.get(qid)
        if not ranked:
            continue
        truth = np.array([float(row["truth_x"]), float(row["truth_y"])])
        top = ranked[0]
        top_xy = np.array([float(top["candidate_x"]), float(top["candidate_y"])])
        if np.linalg.norm(top_xy - truth) > CORRECT_RADIUS:
            continue
        xyz = np.fromfile(EXP / "derived/queries" / row["file"], dtype=np.float32).reshape(-1, 4)[:, :3]
        xyz = xyz @ canonical_rotation(query_gravity[qid]).T
        up, down = envelopes(xyz)
        mixed = int((~np.isnan(up) & ~np.isnan(down)).sum())
        if best is None or mixed > best["mixed"]:
            best = {"row": row, "qid": qid, "xyz": xyz, "up": up, "down": down,
                    "mixed": mixed, "ranked": ranked}
    assert best is not None

    row, ranked = best["row"], best["ranked"]
    top = ranked[0]
    kf_index = int(top["candidate_index"])
    map_gravity = load_gravity(EXP / "map/session/scan_context_gravity.csv", "index")
    kf_xyz = read_binary_pcd(EXP / f"map/session/key_point_frame/{kf_index}.pcd")
    kf_xyz = crop(kf_xyz @ canonical_rotation(map_gravity[kf_index]).T)
    tum = np.loadtxt(EXP / "map/session/optimized_poses_tum.txt")

    np.savez_compressed(
        OUT / "pipeline_source_data.npz",
        query_xyz=voxel_centroids(crop(best["xyz"]), 0.10),
        map_xyz=voxel_centroids(kf_xyz, 0.10),
        up=best["up"], down=best["down"], tau=TAU,
        distances=np.array([float(r["distance"]) for r in ranked], dtype=np.float64),
        top1_distance=float(top["distance"]),
        sector_shift=int(top["sector_shift"]),
        yaw_shift_rad=float(top["yaw_shift_rad"]),
        candidate_index=kf_index,
        candidate_xy=np.array([float(top["candidate_x"]), float(top["candidate_y"])]),
        truth_xy=np.array([float(row["truth_x"]), float(row["truth_y"])]),
        trajectory_xy=tum[:, 1:3],
        query_id=best["qid"],
    )
    meta = {
        "selection_rule": "IH+G 2 m evaluation, final channel weights: among queries "
                          "with a correct top-1 retrieval, maximize dual-envelope mixed cells",
        "query_id": best["qid"], "mixed_cells": best["mixed"],
        "top1": {"candidate_index": kf_index, "distance": float(top["distance"]),
                 "sector_shift": int(top["sector_shift"])},
        "descriptor": {"rings": RINGS, "sectors": SECTORS, "radius_m": RADIUS,
                       "voxel_m": VOXEL, "split_height_m": TAU},
        "display_voxel_m": 0.10,
        "integrity": "All panels derive from this one measured query, its recorded "
                     "candidate list, the matched map keyframe, and the experimental "
                     "map trajectory. No cell or point is edited.",
    }
    (OUT / "pipeline_source_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"query {best['qid']}: mixed={best['mixed']}, top1 kf={kf_index}, "
          f"d={top['distance']}, shift={top['sector_shift']}")


if __name__ == "__main__":
    main()
