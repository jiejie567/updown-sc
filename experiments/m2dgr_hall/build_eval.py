#!/usr/bin/env python3
"""Build the M2DGR hall cross-session evaluation tree (CH layout).

Steps:
1. 2 m translation-only splits of the hall_04 (map) and hall_02 (query)
   sessions via the shared experiment tools.
2. Per-session truth alignment: joint estimate of GT clock offset, world
   transform, and prism lever arm against the Leica track.
3. Cross-day frame check: register the two sessions' wall points in their
   GT frames (the Leica station may move between days).
4. Query truth expressed in the hall_04 map (FAST-LIO) frame; 2 m coverage
   audit; metadata/gravity/bin export for the Python baselines.
5. Descriptor-origin height audit per session (ground-plane distance after
   gravity rotation).
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

RT = Path("${UPDOWN_SC_ROOT}/icra2027_runtime")
SRC = RT / "experiments/m2dgr_hall_2m"
ROOT = RT / "experiments/m2dgr_hall_eval_2m"
COMMON = Path("${UPDOWN_SC_ROOT}/updown-sc/experiments/common")
GT_DIR = RT / "datasets/m2dgr"
SPACING = 2.0
RADIUS = 2.0


def read_pcd(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii").strip()
            key, *values = line.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break
        fields = header["FIELDS"]
        n = int(header["POINTS"][0])
        dtype = np.dtype([(fld, "<f4") for fld in fields])
        data = np.frombuffer(f.read(), dtype=dtype, count=n)
    return data


def quat_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def align_to_gt(tum: np.ndarray, gt: np.ndarray):
    """Joint (offset, world R/t, lever) estimate; returns dict."""
    def solve(dt_off):
        g = gt.copy()
        g[:, 0] += dt_off
        gi = np.searchsorted(g[:, 0], tum[:, 0]).clip(1, len(g) - 1)
        prev = np.abs(g[gi - 1, 0] - tum[:, 0]) < np.abs(g[gi, 0] - tum[:, 0])
        gi[prev] -= 1
        keep = np.abs(g[gi, 0] - tum[:, 0]) < 0.10
        if keep.sum() < 50:
            return None
        P = tum[keep, 1:4]
        Q = g[gi[keep], 1:4]
        Rb = np.stack([quat_R(q) for q in tum[keep, 4:8]])
        lever = np.zeros(3)
        for _ in range(12):
            Pc = P + np.einsum("nij,j->ni", Rb, lever)
            mp, mq = Pc.mean(0), Q.mean(0)
            U, S, Vt = np.linalg.svd((Q - mq).T @ (Pc - mp))
            D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
            Rw = U @ D @ Vt
            tw = mq - Rw @ mp
            A = np.einsum("ij,njk->nik", Rw, Rb).reshape(-1, 3)
            b = (Q - (P @ Rw.T + tw)).reshape(-1)
            lever, *_ = np.linalg.lstsq(A, b, rcond=None)
        E = np.linalg.norm(
            (P + np.einsum("nij,j->ni", Rb, lever)) @ Rw.T + tw - Q, axis=1)
        return {"rmse": float(np.sqrt((E ** 2).mean())), "R": Rw, "t": tw,
                "lever": lever, "pairs": int(keep.sum()), "offset": dt_off}
    best = None
    for off in np.arange(-3.0, 3.01, 0.25):
        r = solve(off)
        if r and (best is None or r["rmse"] < best["rmse"]):
            best = r
    for off in np.arange(best["offset"] - 0.25, best["offset"] + 0.26, 0.05):
        r = solve(off)
        if r and r["rmse"] < best["rmse"]:
            best = r
    return best


def session_wall_points(session: Path, indices: np.ndarray, R: np.ndarray,
                        t: np.ndarray) -> np.ndarray:
    tum = np.loadtxt(session / "optimized_poses_tum.txt")
    points = []
    for idx in indices:
        data = read_pcd(session / "key_point_frame" / f"{idx}.pcd")
        xyz = np.column_stack([data["x"], data["y"], data["z"]])
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        rr = np.hypot(xyz[:, 0], xyz[:, 1])
        xyz = xyz[(rr < 25) & (rr > 0.5)]
        pose_R = quat_R(tum[idx, 4:8])
        world = xyz @ pose_R.T + tum[idx, 1:4]
        keep = (world[:, 2] > 0.5) & (world[:, 2] < 2.0)
        points.append(world[keep])
    pts = np.vstack(points)
    gt = pts @ R.T + t
    cells = np.round(gt[:, :2] / 0.15).astype(np.int64)
    _, uniq = np.unique(cells, axis=0, return_index=True)
    return gt[uniq, :2]


def icp_refine(source, tree, target, R, t, iters, gate):
    for _ in range(iters):
        moved = source @ R.T + t
        dist, idx = tree.query(moved, k=1)
        keep = dist < gate
        if keep.sum() < 50:
            return R, t, np.inf
        S, T = source[keep], target[idx[keep]]
        ms, mt = S.mean(0), T.mean(0)
        U, _, Vt = np.linalg.svd((T - mt).T @ (S - ms))
        D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
        R = U @ D @ Vt
        t = mt - R @ ms
    moved = source @ R.T + t
    dist, _ = tree.query(moved, k=1)
    return R, t, float(np.median(dist))


def icp2d(source: np.ndarray, target: np.ndarray, iters: int = 30):
    """Global SE(2) registration: yaw grid + centroid init, ICP refinement."""
    tree = cKDTree(target)
    mt = target.mean(0)
    ms = source.mean(0)
    best = (np.eye(2), np.zeros(2), np.inf)
    for yaw_deg in np.arange(0.0, 360.0, 2.0):
        a = np.radians(yaw_deg)
        R0 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        t0 = mt - R0 @ ms
        R1, t1, med = icp_refine(source, tree, target, R0, t0, 8, 1.5)
        if med < best[2]:
            best = (R1, t1, med)
    R, t, _ = best
    R, t, med = icp_refine(source, tree, target, R, t, 25, 0.6)
    return R, t, med


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    report = {}

    # -- step 1: 2 m splits --------------------------------------------------
    splits = {}
    for name, seq in (("map", "hall_04"), ("query", "hall_02")):
        out = ROOT / f"split_{seq}"
        subprocess.run([
            "python3", str(COMMON / "select_spatial_keyframes.py"),
            "--input-tum", str(SRC / seq / "session/optimized_poses_tum.txt"),
            "--output-dir", str(out), "--spacing-m", str(SPACING)],
            check=True)
        idx_file = next(out.glob("*indices*"))
        indices = np.loadtxt(idx_file, dtype=int).reshape(-1)
        splits[name] = indices
        print(f"{seq}: {len(indices)} keyframes at {SPACING} m")
    subprocess.run([
        "python3", str(COMMON / "materialize_keyframe_subset.py"),
        "--source-session", str(SRC / "hall_04/session"),
        "--indices", str(next((ROOT / "split_hall_04").glob("*indices*"))),
        "--output-session", str(ROOT / "seq1/session")], check=True)
    subprocess.run([
        "python3", str(COMMON / "materialize_keyframe_subset.py"),
        "--source-session", str(SRC / "hall_02/session"),
        "--indices", str(next((ROOT / "split_hall_02").glob("*indices*"))),
        "--output-session", str(ROOT / "query_session")], check=True)

    # -- step 2: per-session truth alignment ---------------------------------
    aligns = {}
    for seq in ("hall_04", "hall_02"):
        tum = np.loadtxt(SRC / seq / "session/optimized_poses_tum.txt")
        gt = np.loadtxt(GT_DIR / f"{seq}_gt.txt")
        aligns[seq] = align_to_gt(tum, gt)
        a = aligns[seq]
        report[f"align_{seq}"] = {
            "offset_s": round(a["offset"], 2), "rmse_m": round(a["rmse"], 3),
            "lever_m": [round(v, 3) for v in a["lever"]], "pairs": a["pairs"]}
        print(f"{seq}: offset {a['offset']:+.2f}s rmse {a['rmse']*100:.1f}cm "
              f"lever {np.round(a['lever'],3)}")

    # -- step 3: cross-day frame registration --------------------------------
    w4 = session_wall_points(SRC / "hall_04/session", splits["map"],
                             aligns["hall_04"]["R"], aligns["hall_04"]["t"])
    w2 = session_wall_points(SRC / "hall_02/session", splits["query"],
                             aligns["hall_02"]["R"], aligns["hall_02"]["t"])
    R42, t42, med = icp2d(w2, w4)
    yaw = float(np.degrees(np.arctan2(R42[1, 0], R42[0, 0])))
    report["gt2_to_gt4"] = {
        "yaw_deg": round(yaw, 2), "t_m": [round(v, 3) for v in t42],
        "post_icp_median_m": round(med, 3),
        "wall_points": [len(w2), len(w4)]}
    print(f"GT2->GT4: yaw {yaw:.2f} deg, t {np.round(t42,3)}, "
          f"post-ICP median {med*100:.1f} cm")

    # -- step 4: query truth in map (hall_04 FAST-LIO) frame ------------------
    a4, a2 = aligns["hall_04"], aligns["hall_02"]
    tum2 = np.loadtxt(SRC / "hall_02/session/optimized_poses_tum.txt")
    q_idx = splits["query"]
    p2 = tum2[q_idx, 1:4]
    gt2 = p2 @ a2["R"].T + a2["t"]
    gt4 = np.column_stack([gt2[:, :2] @ R42.T + t42, gt2[:, 2]])
    map_frame = (gt4 - a4["t"]) @ a4["R"]
    map_tum = np.loadtxt(ROOT / "seq1/session/optimized_poses_tum.txt")
    map_xy = map_tum[:, 1:3]
    tree = cKDTree(map_xy)
    dist, _ = tree.query(map_frame[:, :2], k=1)
    coverage = dist <= RADIUS
    report["coverage"] = {
        "queries": len(q_idx), "eligible": int(coverage.sum()),
        "nearest_median_m": round(float(np.median(dist)), 3),
        "nearest_max_m": round(float(dist.max()), 3)}
    print(f"coverage: {coverage.sum()}/{len(q_idx)} within {RADIUS} m "
          f"(median nearest {np.median(dist):.2f} m)")

    # -- step 5: derived queries for the Python baselines ---------------------
    qdir = ROOT / "derived/queries_seq2"
    qdir.mkdir(parents=True, exist_ok=True)
    src_grav = list(csv.DictReader(
        (SRC / "hall_02/session/scan_context_gravity.csv").open()))
    grav_rows, meta_rows = [], []
    for qid, src_idx in enumerate(q_idx):
        data = read_pcd(SRC / "hall_02/session/key_point_frame" / f"{src_idx}.pcd")
        arr = np.column_stack([
            data["x"], data["y"], data["z"],
            data["intensity"] if "intensity" in data.dtype.names
            else np.zeros(len(data), np.float32)]).astype(np.float32)
        fname = f"{qid:06d}.bin"
        arr.tofile(qdir / fname)
        g = src_grav[src_idx]
        grav_rows.append({"query_id": qid,
                          "stamp": f"{tum2[src_idx, 0]:.7f}",
                          "up_x": g["up_x"], "up_y": g["up_y"],
                          "up_z": g["up_z"]})
        meta_rows.append({
            "query_id": qid, "window": qid,
            "start_s": round(tum2[src_idx, 0] - tum2[0, 0], 3),
            "points": len(arr), "file": fname,
            "truth_valid": bool(coverage[qid]) and "True" or "False",
            "truth_x": round(float(map_frame[qid, 0]), 6),
            "truth_y": round(float(map_frame[qid, 1]), 6),
            "source_keyframe": int(src_idx)})
    with (qdir / "gravity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grav_rows[0]))
        w.writeheader(); w.writerows(grav_rows)
    with (qdir / "metadata.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta_rows[0]))
        w.writeheader(); w.writerows(meta_rows)
    print(f"wrote {len(meta_rows)} query bins to {qdir}")

    # -- step 6: descriptor-origin height audit -------------------------------
    for seq, name in (("hall_04", "map"), ("hall_02", "query")):
        grav = list(csv.DictReader(
            (SRC / seq / "session/scan_context_gravity.csv").open()))
        idx = splits[name]
        floors = []
        for i in idx[:: max(1, len(idx) // 40)]:
            data = read_pcd(SRC / seq / "session/key_point_frame" / f"{i}.pcd")
            xyz = np.column_stack([data["x"], data["y"], data["z"]])
            up = np.array([float(grav[i]["up_x"]), float(grav[i]["up_y"]),
                           float(grav[i]["up_z"])])
            up /= np.linalg.norm(up)
            target = np.array([0, 0, 1.0])
            cr = np.cross(up, target)
            d = float(np.clip(up @ target, -1, 1))
            n = np.linalg.norm(cr)
            if n < 1e-9:
                R = np.eye(3)
            else:
                a = cr / n
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]],
                              [-a[1], a[0], 0]])
                ang = np.arctan2(n, d)
                R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
            z = (xyz @ R.T)[:, 2]
            rr = np.hypot(xyz[:, 0], xyz[:, 1])
            near = z[(rr > 1.0) & (rr < 8.0)]
            if len(near) > 200:
                floors.append(np.percentile(near, 2))
        h = -float(np.median(floors))
        report[f"origin_height_{seq}"] = round(h, 3)
        print(f"{seq}: descriptor-origin height above ground = {h:.3f} m")

    (ROOT / "build_report.json").write_text(json.dumps(report, indent=2))
    print("report ->", ROOT / "build_report.json")


if __name__ == "__main__":
    main()
