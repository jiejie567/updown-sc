#!/usr/bin/env python3
"""Render consistently styled measured side views for Fig. 1 selection."""

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
CANDIDATE_IDS = [0, 2, 4, 6, 8, 12, 190, 192, 194, 196, 198, 200]
VIEW_DEG = 60.0
SPLIT_M = 2.1
CEILING_MARGIN_M = 0.12
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


def ceiling_cutoff(xyz: np.ndarray) -> tuple[float, float]:
    candidates = xyz[:, 2][(xyz[:, 2] >= 2.3) & (xyz[:, 2] <= 4.5)]
    edges = np.arange(2.3, 4.52, 0.02)
    counts, _ = np.histogram(candidates, bins=edges)
    peak = int(np.argmax(counts))
    ceiling = 0.5 * (edges[peak] + edges[peak + 1])
    return float(ceiling), float(ceiling + CEILING_MARGIN_M)


def main() -> None:
    roof.SPLIT_HEIGHT_M = SPLIT_M
    records = {record.query_id: record for record in roof.load_queries(QUERY_DIR)}
    norm = mpl.colors.Normalize(vmin=-0.5, vmax=3.2, clip=True)
    angle = np.deg2rad(VIEW_DEG)

    fig, axes = plt.subplots(2, 6, figsize=(15.0, 5.2), facecolor="white")
    metadata: list[dict[str, float | int | str]] = []
    for label, query_id, ax in zip(string.ascii_uppercase, CANDIDATE_IDS, axes.flat):
        descriptor = roof.make_descriptor(records[query_id])
        xyz = descriptor.render_xyz
        ceiling, cutoff = ceiling_cutoff(xyz)
        retained = xyz[xyz[:, 2] <= cutoff]
        horizontal = retained[:, 0] * np.cos(angle) + retained[:, 1] * np.sin(angle)
        low, high = np.percentile(horizontal, [5.0, 95.0])

        ax.scatter(
            horizontal,
            retained[:, 2],
            s=0.60,
            c=retained[:, 2],
            cmap=HEIGHT_CMAP,
            norm=norm,
            linewidths=0,
        )
        ax.axhline(SPLIT_M, color="#D97706", linewidth=0.9, linestyle=(0, (4, 3)))
        ax.set_xlim(low, high)
        ax.set_ylim(-0.25, 3.24)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(
            f"{label}  ·  {'START' if query_id < 100 else 'END'}  ·  query {query_id}",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            color="#24313D",
            pad=3,
        )
        metadata.append(
            {
                "label": label,
                "query_id": query_id,
                "view_deg": VIEW_DEG,
                "dominant_ceiling_height_m": ceiling,
                "ceiling_display_cutoff_m": cutoff,
                "points_total": int(len(xyz)),
                "points_displayed_before_axis_crop": int(len(retained)),
            }
        )

    fig.suptitle(
        "Indoor start/end side-view candidates for Fig. 1",
        fontsize=15,
        fontweight="bold",
        color="#24313D",
        y=0.995,
    )
    fig.text(
        0.5,
        0.965,
        "Common 60° projection · blue-to-red gravity-aligned height · orange dashed split at 2.1 m",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#6B7783",
    )
    color_ax = fig.add_axes([0.945, 0.18, 0.010, 0.66])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=HEIGHT_CMAP), cax=color_ax
    )
    colorbar.set_ticks([0.0, SPLIT_M, 3.2])
    colorbar.set_ticklabels(["0", "2.1", "3.2"])
    colorbar.set_label(r"height $z_g$ (m)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.018, right=0.93, bottom=0.035, top=0.90, wspace=0.04, hspace=0.20)
    fig.savefig(OUT / "hero_endpoint_sideview_candidates.png", dpi=360, facecolor="white")
    plt.close(fig)
    (OUT / "hero_endpoint_sideview_candidates.json").write_text(
        json.dumps(
            {
                "selection_purpose": "Fig. 1 indoor start/end visual candidate review only",
                "candidate_count": len(metadata),
                "common_display_rule": (
                    "Gravity-canonicalized measured points; 5--95% horizontal crop; "
                    "returns above the dominant ceiling mode plus 0.12 m omitted for display."
                ),
                "candidates": metadata,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
