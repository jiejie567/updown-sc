#!/usr/bin/env python3
"""Render near-field oblique candidates that expose indoor furniture."""

from __future__ import annotations

import json
import string
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import generate_roof_contamination_figure as roof


OUT = Path(__file__).resolve().parent
QUERY_DIR = Path(
    "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/"
    "baseline_20260715/queries/loc_2_floor_continuous"
)
CANDIDATE_IDS = [0, 1, 2, 3, 4, 5, 6, 8]
RANGE_M = 8.0
ELEV_DEG = 16.0
AZIM_DEG = -45.0
SPLIT_M = 2.1
HEIGHT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "height",
    ["#263B73", "#3775BA", "#D9EFEA", "#F4C66A", "#B64342"],
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def dominant_ceiling(xyz: np.ndarray) -> float:
    upper = xyz[:, 2][(xyz[:, 2] >= 2.3) & (xyz[:, 2] <= 4.5)]
    edges = np.arange(2.3, 4.52, 0.02)
    counts, _ = np.histogram(upper, bins=edges)
    peak = int(np.argmax(counts))
    return float(0.5 * (edges[peak] + edges[peak + 1]))


def main() -> None:
    roof.SPLIT_HEIGHT_M = SPLIT_M
    records = {record.query_id: record for record in roof.load_queries(QUERY_DIR)}
    norm = mpl.colors.Normalize(vmin=-0.5, vmax=3.2, clip=True)
    fig = plt.figure(figsize=(14.0, 6.7), facecolor="white")
    metadata = []

    for index, (label, query_id) in enumerate(zip(string.ascii_uppercase, CANDIDATE_IDS)):
        descriptor = roof.make_descriptor(records[query_id])
        xyz = descriptor.render_xyz
        ceiling = dominant_ceiling(xyz)
        radius = np.hypot(xyz[:, 0], xyz[:, 1])
        keep = (radius <= RANGE_M) & (xyz[:, 2] <= ceiling + 0.12)
        retained = xyz[keep]
        low = np.percentile(retained[:, :2], 2.0, axis=0)
        high = np.percentile(retained[:, :2], 98.0, axis=0)
        ordered = retained[np.argsort(retained[:, 2])]

        ax = fig.add_subplot(2, 4, index + 1, projection="3d", computed_zorder=False)
        ax.scatter(
            ordered[:, 0],
            ordered[:, 1],
            ordered[:, 2],
            s=0.72,
            c=ordered[:, 2],
            cmap=HEIGHT_CMAP,
            norm=norm,
            linewidths=0,
            depthshade=False,
        )
        ax.view_init(elev=ELEV_DEG, azim=AZIM_DEG)
        ax.set_proj_type("ortho")
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_zlim(-0.25, 3.25)
        ax.set_box_aspect((high[0] - low[0], high[1] - low[1], 3.5))
        ax.set_axis_off()
        ax.set_title(
            f"{label}  ·  query {query_id}",
            loc="left",
            fontsize=11,
            fontweight="bold",
            color="#24313D",
            pad=1,
        )
        metadata.append(
            {
                "label": label,
                "query_id": query_id,
                "single_frame_duration_s": 0.1,
                "range_crop_m": RANGE_M,
                "elevation_deg": ELEV_DEG,
                "azimuth_deg": AZIM_DEG,
                "dominant_ceiling_height_m": ceiling,
                "points_total": int(len(xyz)),
                "points_displayed_before_axis_crop": int(len(retained)),
            }
        )

    fig.suptitle(
        "Near-field indoor candidates with visible lower structure",
        fontsize=15,
        fontweight="bold",
        color="#24313D",
        y=0.995,
    )
    fig.text(
        0.5,
        0.962,
        "0.1 s single frames · orthographic oblique view · returns within 8 m",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#6B7783",
    )
    color_ax = fig.add_axes([0.945, 0.19, 0.011, 0.62])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=HEIGHT_CMAP), cax=color_ax
    )
    colorbar.set_ticks([0.0, SPLIT_M, 3.2])
    colorbar.set_ticklabels(["0", "2.1", "3.2"])
    colorbar.set_label(r"height $z_g$ (m)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.01, right=0.93, bottom=0.015, top=0.91, wspace=0.01, hspace=0.02)
    fig.savefig(OUT / "hero_furniture_view_candidates.png", dpi=360, facecolor="white")
    plt.close(fig)

    (OUT / "hero_furniture_view_candidates.json").write_text(
        json.dumps(
            {
                "selection_purpose": "Fig. 1 near-field furniture visibility review only",
                "image_integrity": (
                    "Measured points are not moved or interpolated. The review view uses an 8 m radial crop, "
                    "a 2--98% horizontal display crop, and removes returns above the dominant ceiling mode "
                    "plus 0.12 m. Descriptor construction and experiments are unchanged."
                ),
                "candidates": metadata,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
