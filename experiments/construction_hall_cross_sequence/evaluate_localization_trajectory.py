#!/usr/bin/env python3
"""Check trusted online localization poses against GNSS-aligned query truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--truth-tum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-stamp-error", type=float, default=0.15)
    args = parser.parse_args()

    with args.poses.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    stamps = np.asarray([int(row["stamp_ns"]) * 1e-9 for row in rows])
    xy = np.asarray([[float(row["x"]), float(row["y"])] for row in rows])
    valid = np.isfinite(stamps) & np.isfinite(xy).all(axis=1)
    stamps, xy = stamps[valid], xy[valid]
    order = np.argsort(stamps)
    stamps, xy = stamps[order], xy[order]

    truth = np.loadtxt(args.truth_tum, dtype=np.float64)
    if truth.ndim == 1:
        truth = truth[None, :]
    insertion = np.searchsorted(stamps, truth[:, 0])
    insertion = np.clip(insertion, 1, len(stamps) - 1)
    left = insertion - 1
    right = insertion
    choose_right = (
        np.abs(stamps[right] - truth[:, 0])
        < np.abs(stamps[left] - truth[:, 0])
    )
    matched = np.where(choose_right, right, left)
    stamp_error = np.abs(stamps[matched] - truth[:, 0])
    accepted = stamp_error <= args.max_stamp_error
    error = np.linalg.norm(xy[matched] - truth[:, 1:3], axis=1)
    accepted_error = error[accepted]

    summary = {
        "trusted_pose_count": int(len(stamps)),
        "truth_keyframes": int(len(truth)),
        "associated_keyframes": int(np.sum(accepted)),
        "association_fraction": float(np.mean(accepted)),
        "max_stamp_error_s": args.max_stamp_error,
        "position_error_median": float(np.median(accepted_error)),
        "position_error_p95": float(np.percentile(accepted_error, 95)),
        "position_error_max": float(np.max(accepted_error)),
        "trusted_xy_min": xy.min(axis=0).tolist(),
        "trusted_xy_max": xy.max(axis=0).tolist(),
        "truth_xy_min": truth[:, 1:3].min(axis=0).tolist(),
        "truth_xy_max": truth[:, 1:3].max(axis=0).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
