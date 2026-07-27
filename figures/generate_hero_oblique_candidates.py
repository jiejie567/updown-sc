#!/usr/bin/env python3
"""Render labeled oblique-view candidates for the Fig. 1 hero panel.

Candidate A reproduces the current hero (query 3, azimuth -45 deg,
elevation 16 deg). B--I show alternative measured scenes at the same view,
chosen by descending dual-envelope mixed-cell count with at least 8 m
trajectory spacing. J--L show the current scene under alternative views.
Every panel uses the manuscript hero pipeline (same descriptor construction,
documented display crop, orthographic oblique projection, height colormap),
so the chosen candidate renders identically in the final figure.
"""

from __future__ import annotations

import csv
import json
import string
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import generate_roof_contamination_figure as roof
import generate_scancontext_style_figures as gsf

OUT = Path(__file__).resolve().parent
SPACING_M = 8.0
ALT_SCENES = 8
VIEW_VARIANTS = ((-20.0, 16.0), (-70.0, 16.0), (-45.0, 28.0))


def load_truth() -> dict[int, np.ndarray]:
    truth = {}
    with (gsf.HERO_QUERY_DIR / "metadata.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["truth_valid"] == "True":
                truth[int(row["query_id"])] = np.array(
                    [float(row["truth_x"]), float(row["truth_y"])])
    return truth


def render(ax, xyz: np.ndarray, azim: float, elev: float) -> dict:
    gsf.HERO_AZIM_DEG, gsf.HERO_ELEV_DEG = azim, elev
    shown, _, cutoff, _, _, stats = gsf.prepare_hero_display(xyz)
    horizontal, vertical, depth = gsf.orthographic_project(shown)
    order = np.argsort(depth)
    norm = mpl.colors.Normalize(vmin=-0.5, vmax=3.2, clip=True)
    ax.scatter(horizontal[order], vertical[order], s=0.5,
               c=shown[order, 2], cmap=gsf.HEIGHT_CMAP, norm=norm,
               linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return stats


def main() -> None:
    roof.SPLIT_HEIGHT_M = gsf.HERO_SPLIT_HEIGHT_M
    records = {r.query_id: r for r in roof.load_queries(gsf.HERO_QUERY_DIR)}
    truth = load_truth()

    mixed: dict[int, tuple[int, np.ndarray]] = {}
    for qid, record in records.items():
        if qid not in truth:
            continue
        descriptor = roof.make_descriptor(record)
        mixed[qid] = (int(descriptor.hidden_mask.sum()), descriptor.render_xyz)
    print(f"evaluated {len(mixed)} truth-valid queries")

    current_id = gsf.HERO_QUERY_ID
    picked: list[int] = []
    for qid in sorted(mixed, key=lambda q: -mixed[q][0]):
        if qid == current_id:
            continue
        pos = truth[qid]
        anchors = [truth[current_id]] + [truth[p] for p in picked]
        if min(np.linalg.norm(pos - a) for a in anchors) < SPACING_M:
            continue
        picked.append(qid)
        if len(picked) == ALT_SCENES:
            break

    plan = [(current_id, gsf.HERO_AZIM_DEG, gsf.HERO_ELEV_DEG, "current hero")]
    plan += [(qid, -45.0, 16.0, f"mixed={mixed[qid][0]}") for qid in picked]
    plan += [(current_id, a, e, "view variant") for a, e in VIEW_VARIANTS]

    fig, axes = plt.subplots(2, 6, figsize=(15.0, 5.8), facecolor="white")
    metadata = []
    for label, (qid, azim, elev, note), ax in zip(
            string.ascii_uppercase, plan, axes.flat):
        stats = render(ax, mixed[qid][1], azim, elev)
        ax.set_title(
            f"{label} \u00b7 q{qid} \u00b7 {note}\n"
            f"az {azim:.0f}\u00b0, el {elev:.0f}\u00b0",
            loc="center", fontsize=8.5, fontweight="bold", color="#24313D",
            pad=2, linespacing=1.15)
        metadata.append({
            "label": label, "query_id": qid, "azimuth_deg": azim,
            "elevation_deg": elev, "note": note,
            "mixed_cells": mixed[qid][0],
            "points_displayed": stats["points_displayed"],
        })

    fig.suptitle(
        "Fig. 1 hero oblique-view candidates (A = current; B–I scenes by "
        "mixed-cell count, ≥ 8 m apart; J–L view variants)",
        fontsize=13, fontweight="bold", color="#24313D", y=0.99)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.86,
                        wspace=0.06, hspace=0.30)
    fig.savefig(OUT / "hero_oblique_candidates.png", dpi=300,
                facecolor="white")
    plt.close(fig)
    (OUT / "hero_oblique_candidates.json").write_text(json.dumps({
        "selection_purpose": "Fig. 1 hero visual candidate review only",
        "display_rule": "manuscript hero pipeline: dominant-ceiling cutoff, "
                        "8 m radial crop, 2-98% XY percentiles",
        "candidates": metadata}, indent=2))
    for entry in metadata:
        print(entry)


if __name__ == "__main__":
    main()
