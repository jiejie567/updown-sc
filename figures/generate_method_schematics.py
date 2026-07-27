#!/usr/bin/env python3
"""Generate deterministic conceptual schematics for the UpDown-SC paper.

These figures explain the implemented method.  They contain no experimental
measurements, simulated performance values, or inferred quantitative results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT_DIR = Path(__file__).resolve().parent

BLUE = "#0F4D92"
BLUE_2 = "#3775BA"
BLUE_LIGHT = "#DCE8F5"
TEAL = "#2A8C8C"
TEAL_LIGHT = "#D9EFEA"
ORANGE = "#D97706"
ORANGE_LIGHT = "#FBE8C9"
PURPLE = "#7A5AA6"
PURPLE_LIGHT = "#E9E1F2"
RED = "#B64342"
GREEN = "#2E8B57"
UPPER = RED
INK = "#25313C"
MID = "#65717C"
LIGHT = "#EEF2F5"
BORDER = "#C9D1D8"
WHITE = "#FFFFFF"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def clean_axis(ax: mpl.axes.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.axis("off")


def panel_frame(ax: mpl.axes.Axes, label: str, title: str, tint: str = WHITE) -> None:
    clean_axis(ax)
    ax.add_patch(
        Rectangle(
            (0.01, 0.01),
            0.98,
            0.98,
            facecolor=tint,
            edgecolor=BORDER,
            linewidth=0.8,
            zorder=-10,
        )
    )
    ax.text(0.035, 0.955, label, ha="left", va="top", weight="bold", fontsize=8, color=INK)
    ax.text(0.14, 0.952, title, ha="left", va="top", weight="bold", fontsize=6.2, color=INK)


def arrow(
    ax: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = MID,
    width: float = 1.1,
    scale: float = 8,
    style: str = "-|>",
    zorder: int = 5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=scale,
            linewidth=width,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def figure_arrow(
    fig: mpl.figure.Figure,
    x0: float,
    x1: float,
    y: float = 0.50,
    color: str = BLUE,
    linestyle: str = "solid",
) -> None:
    fig.add_artist(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.2,
            color=color,
            linestyle=linestyle,
            zorder=20,
        )
    )


def tag(ax: mpl.axes.Axes, x: float, y: float, text: str, face: str, color: str = INK) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=5.8,
        color=color,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": face, "edgecolor": "none"},
    )


def draw_fixed_cloud(ax: mpl.axes.Axes, center: tuple[float, float], scale: float, tilt_deg: float = 0) -> None:
    """Draw a deterministic room-like point set as a conceptual scan icon."""
    wall_x = np.linspace(-0.44, 0.44, 12)
    floor = np.column_stack([wall_x, -0.31 + 0.035 * np.cos(8 * wall_x)])
    ceiling = np.column_stack([wall_x, 0.32 + 0.025 * np.sin(7 * wall_x)])
    shelf_y = np.linspace(-0.22, 0.08, 7)
    shelf = np.column_stack([np.full_like(shelf_y, 0.25), shelf_y])
    doorway_y = np.linspace(-0.30, 0.22, 9)
    doorway = np.column_stack([np.full_like(doorway_y, -0.33), doorway_y])
    points = np.vstack([floor, ceiling, shelf, doorway])
    angle = np.deg2rad(tilt_deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = points @ rot.T
    points[:, 0] = center[0] + scale * points[:, 0]
    points[:, 1] = center[1] + scale * points[:, 1]
    ax.scatter(points[:, 0], points[:, 1], s=4.5, color=BLUE_2, alpha=0.88, linewidths=0)
    ax.add_patch(Circle(center, 0.018 * scale, facecolor=ORANGE, edgecolor=WHITE, linewidth=0.5, zorder=5))


def draw_axes(ax: mpl.axes.Axes, origin: tuple[float, float], angle_deg: float, scale: float = 0.18) -> None:
    a = np.deg2rad(angle_deg)
    x_end = (origin[0] + scale * np.cos(a), origin[1] + scale * np.sin(a))
    z_end = (origin[0] - scale * np.sin(a), origin[1] + scale * np.cos(a))
    arrow(ax, origin, x_end, color=MID, width=1.0, scale=7)
    arrow(ax, origin, z_end, color=ORANGE, width=1.2, scale=7)
    ax.text(x_end[0] + 0.015, x_end[1], "x", fontsize=5.5, color=MID, va="center")
    ax.text(z_end[0], z_end[1] + 0.022, "z", fontsize=5.5, color=ORANGE, ha="center")


def draw_polar_grid(ax: mpl.axes.Axes, center: tuple[float, float], radius: float) -> None:
    for frac in (0.34, 0.67, 1.0):
        ax.add_patch(Circle(center, radius * frac, fill=False, edgecolor=BLUE_2, linewidth=0.65, alpha=0.8))
    for angle_deg in range(0, 360, 45):
        a = np.deg2rad(angle_deg)
        ax.plot(
            [center[0], center[0] + radius * np.cos(a)],
            [center[1], center[1] + radius * np.sin(a)],
            color=BLUE_2,
            linewidth=0.55,
            alpha=0.8,
        )
    ax.add_patch(Circle(center, 0.018, facecolor=ORANGE, edgecolor="none", zorder=5))


def draw_descriptor_strip(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    pattern: tuple[int, ...],
) -> None:
    cell_w = width / len(pattern)
    for idx, level in enumerate(pattern):
        alpha = (0.12, 0.33, 0.58, 0.86)[level]
        ax.add_patch(
            Rectangle(
                (x + idx * cell_w, y),
                cell_w - 0.002,
                height,
                facecolor=color,
                edgecolor=WHITE,
                linewidth=0.3,
                alpha=alpha,
            )
        )


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_DIR / f"{stem}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def make_pipeline() -> None:
    """Draw a hierarchical four-stage pipeline led by the descriptor."""
    fig = plt.figure(figsize=(7.16, 1.92), facecolor=WHITE)
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.25, 1.92, 1.28, 1.20],
        left=0.008,
        right=0.992,
        bottom=0.035,
        top=0.985,
        wspace=0.15,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    # a) Common single-frame front end.
    ax = axes[0]
    panel_frame(ax, "a", "Gravity-canonicalized scan", "#FAFBFC")
    draw_fixed_cloud(ax, (0.26, 0.60), 0.38, tilt_deg=-17)
    draw_axes(ax, (0.26, 0.55), -17, 0.14)
    arrow(ax, (0.43, 0.59), (0.60, 0.59), color=BLUE, width=1.3, scale=9)
    ax.text(
        0.515,
        0.68,
        r"deskew + $R_g$",
        ha="center",
        fontsize=5.9,
        color=BLUE,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.08", "facecolor": WHITE, "edgecolor": "none", "alpha": 0.9},
    )
    draw_fixed_cloud(ax, (0.76, 0.60), 0.38, tilt_deg=0)
    draw_axes(ax, (0.76, 0.55), 0, 0.14)
    arrow(ax, (0.11, 0.79), (0.20, 0.69), color=ORANGE, width=1.3, scale=8)
    ax.text(0.08, 0.83, "IMU up", fontsize=5.8, color=ORANGE, ha="left")
    tag(ax, 0.25, 0.27, "blind 0.3 m", LIGHT)
    tag(ax, 0.74, 0.27, "body frame", BLUE_LIGHT, BLUE)
    ax.text(0.50, 0.14, "one 0.1 s scan", ha="center", fontsize=5.9, color=INK)
    ax.text(0.50, 0.075, "roll/pitch removed · yaw free", ha="center", fontsize=5.45, color=MID)

    # b) Map-adaptive dual-envelope descriptor (hero panel).
    ax = axes[1]
    panel_frame(ax, "b", "Adaptive dual-envelope descriptor", "#F5F9FD")
    draw_polar_grid(ax, (0.22, 0.57), 0.18)
    ax.text(0.22, 0.35, "polar cells", ha="center", fontsize=5.7, color=MID)

    # Cell cross-section: lower geometry and overhead surface remain separate.
    ax.plot([0.48, 0.94], [0.61, 0.61], linestyle=(0, (3, 2)), color=ORANGE, linewidth=1.0)
    ax.text(
        0.71,
        0.635,
        r"cell-balanced $\tau_g$",
        color=ORANGE,
        fontsize=5.6,
        ha="center",
        va="bottom",
    )
    ax.add_patch(Rectangle((0.53, 0.34), 0.055, 0.21, facecolor=BLUE_2, edgecolor="none", alpha=0.78))
    ax.add_patch(Rectangle((0.63, 0.34), 0.055, 0.14, facecolor=BLUE_2, edgecolor="none", alpha=0.53))
    ax.add_patch(Rectangle((0.73, 0.34), 0.055, 0.18, facecolor=BLUE_2, edgecolor="none", alpha=0.67))
    ax.plot([0.49, 0.84], [0.55, 0.55], color=BLUE, linewidth=2.5)
    ax.text(0.88, 0.50, r"$E^\downarrow$: max below", fontsize=5.7, color=BLUE, va="center", ha="right", weight="bold")
    ax.plot([0.49, 0.84], [0.77, 0.77], color=UPPER, linewidth=2.5)
    ax.add_patch(Rectangle((0.53, 0.77), 0.055, 0.10, facecolor=UPPER, edgecolor="none", alpha=0.58))
    ax.add_patch(Rectangle((0.63, 0.77), 0.055, 0.07, facecolor=UPPER, edgecolor="none", alpha=0.78))
    ax.add_patch(Rectangle((0.73, 0.77), 0.055, 0.12, facecolor=UPPER, edgecolor="none", alpha=0.66))
    ax.text(0.88, 0.715, r"$E^\uparrow$: min above", fontsize=5.7, color=UPPER, va="center", ha="right", weight="bold")
    # Keep the descriptor rows in physical vertical order: overhead/red
    # above and lower-middle/blue below.
    draw_descriptor_strip(ax, 0.20, 0.20, 0.66, 0.060, UPPER, (3, 2, 0, 1, 3, 2, 2, 1))
    draw_descriptor_strip(ax, 0.20, 0.125, 0.66, 0.060, BLUE, (1, 3, 2, 0, 2, 3, 1, 2))
    ax.text(0.16, 0.229, r"$E^\uparrow/M^\uparrow$", fontsize=5.2, color=UPPER, ha="right")
    ax.text(0.16, 0.154, r"$E^\downarrow/M^\downarrow$", fontsize=5.2, color=BLUE, ha="right")
    ax.text(0.52, 0.055, "overhead cannot overwrite lower geometry", ha="center", fontsize=5.55, color=INK, weight="bold")

    # c) Masked SC-style retrieval.
    ax = axes[2]
    panel_frame(ax, "c", "Masked place–yaw retrieval", "#FAFBFC")
    card_y = (0.70, 0.57, 0.44)
    for idx, yy in enumerate(card_y):
        face = BLUE_LIGHT if idx == 1 else WHITE
        edge = BLUE if idx == 1 else BORDER
        ax.add_patch(Rectangle((0.12, yy - 0.055), 0.34, 0.10, facecolor=face, edgecolor=edge, linewidth=0.8))
        draw_descriptor_strip(ax, 0.16, yy - 0.018, 0.26, 0.035, BLUE_2, (1 + idx % 2, 3, 0, 2, 1, 2))
    ax.text(0.29, 0.80, "ring-key Top-K", ha="center", fontsize=5.7, color=MID)
    arrow(ax, (0.49, 0.57), (0.62, 0.57), color=BLUE, width=1.2, scale=8)
    ax.add_patch(Circle((0.77, 0.57), 0.12, fill=False, edgecolor=PURPLE, linewidth=1.2))
    ax.add_patch(Arc((0.77, 0.57), 0.18, 0.18, theta1=30, theta2=300, color=PURPLE, linewidth=1.4))
    arrow(ax, (0.81, 0.65), (0.86, 0.60), color=PURPLE, width=1.2, scale=7)
    ax.text(0.77, 0.57, r"$\pm3$", ha="center", va="center", fontsize=6.1, color=PURPLE, weight="bold")
    ax.text(0.50, 0.31, "joint-valid rings", ha="center", fontsize=5.9, color=INK, weight="bold")
    ax.text(0.50, 0.225, "mask support × height cosine", ha="center", fontsize=5.5, color=MID)
    tag(ax, 0.50, 0.115, "weighted two-channel distance", PURPLE_LIGHT, PURPLE)
    ax.text(0.50, 0.045, "ranked place + yaw hypotheses", ha="center", fontsize=5.2, color=MID)

    # d) Optional downstream geometric verification. Retrieval recall is
    # evaluated at the output of panel c, before this stage.
    ax = axes[3]
    panel_frame(ax, "d", "Optional 3-D verification", "#FAFBFC")
    ax.text(0.50, 0.80, "place + yaw seed", ha="center", fontsize=5.9, color=PURPLE, weight="bold")
    ax.add_patch(Rectangle((0.18, 0.49), 0.34, 0.24, facecolor=BLUE_LIGHT, edgecolor=BLUE, linewidth=0.9))
    ax.add_patch(Rectangle((0.48, 0.43), 0.34, 0.24, facecolor="none", edgecolor=ORANGE, linewidth=1.2))
    arrow(ax, (0.37, 0.75), (0.53, 0.69), color=PURPLE, width=1.1, scale=7)
    ax.text(0.50, 0.35, "point-to-map ICP", ha="center", fontsize=5.9, color=INK, weight="bold")
    tag(ax, 0.50, 0.20, "accept / reject", TEAL_LIGHT, TEAL)
    ax.text(0.50, 0.095, "loop closure · relocalization", ha="center", fontsize=5.1, color=MID)

    positions = [axis.get_position() for axis in axes]
    for index, (left, right) in enumerate(zip(positions[:-1], positions[1:])):
        is_optional = index == 2
        figure_arrow(
            fig,
            left.x1 + 0.002,
            right.x0 - 0.002,
            y=(left.y0 + left.y1) / 2,
            color=MID if is_optional else BLUE,
            linestyle="dashed" if is_optional else "solid",
        )

    save_figure(fig, "updown_sc_pipeline")


def make_place_recognition_pipeline() -> None:
    """Draw the pipeline with measured data in every panel.

    All content derives from one measured IH query (pipeline_source_data.npz,
    exported by export_pipeline_source_data.py from the paper's final IH+G 2 m
    evaluation): the canonicalized cloud, its dual-envelope SCD, the recorded
    top-100 candidate distances and yaw shift, the matched map keyframe cloud,
    and the experimental map trajectory. Structural styling stays flat ink;
    channel colors match Figs. 1, 3, and 4.
    """
    data = np.load(OUT_DIR / "pipeline_source_data.npz")
    q_xyz = data["query_xyz"]
    m_xyz = data["map_xyz"]
    up = data["up"]
    down = data["down"]
    tau = float(data["tau"])
    distances = data["distances"]
    shift_deg = int(data["sector_shift"]) * 6
    yaw_shift = float(data["yaw_shift_rad"])
    cand_xy = data["candidate_xy"]
    truth_xy = data["truth_xy"]
    traj = data["trajectory_xy"]

    fig = plt.figure(figsize=(7.16, 1.66), facecolor=WHITE)
    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[1.30, 1.58, 0.95, 1.00, 1.12],
        left=0.006, right=0.994, bottom=0.158, top=0.818, wspace=0.38,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    titles = (
        "Canonicalized query scan",
        "Dual-envelope SCD",
        "Masked retrieval",
        "6-DoF ICP verification",
        "Place association",
    )
    captions = (
        "side view of one measured scan",
        "ring \u00d7 sector (16 \u00d7 60), white = masked",
        "candidate rank (top-100)",
        "yaw-seeded overlay, before ICP",
        "loop closure / relocalization",
    )
    for index, (ax, title, caption) in enumerate(zip(axes, titles, captions)):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color(INK)
            spine.set_linestyle((0, (4, 2.5)))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate(chr(ord("a") + index), (0.0, 1.0), xycoords="axes fraction",
                    xytext=(0, 3.2), textcoords="offset points", ha="left",
                    va="bottom", fontsize=7.0, color=INK, weight="bold")
        ax.annotate(title, (0.0, 1.0), xycoords="axes fraction",
                    xytext=(9.5, 3.8), textcoords="offset points", ha="left",
                    va="bottom", fontsize=5.7, color=INK, weight="bold")
        ax.text(0.5, -0.165, caption, transform=ax.transAxes, ha="center",
                va="top", fontsize=5.8, color=MID)

    def box_ratio(ax: mpl.axes.Axes) -> float:
        bbox = ax.get_position()
        return (bbox.height * 1.66) / (bbox.width * 7.16)

    # 1) Side elevation of the measured query cloud, colored by height.
    ax = axes[0]
    keep = np.abs(q_xyz[:, 1]) <= 14.0
    side = q_xyz[keep]
    side = side[np.argsort(side[:, 1])[::-1]]
    norm = mpl.colors.Normalize(vmin=-0.5, vmax=3.2)
    ax.scatter(side[:, 0], side[:, 2], c=side[:, 2], cmap="coolwarm", norm=norm,
               s=0.7, linewidths=0, rasterized=True)
    ax.axhline(tau, color=ORANGE, linewidth=0.9, linestyle=(0, (4, 2)))
    ax.set_xlim(-11.0, 11.0)
    ax.set_ylim(-1.0, 3.8)
    ax.set_aspect("auto")
    ax.text(10.4, tau + 0.16, r"$\tau_g$", fontsize=6.0, color=ORANGE,
            ha="right", va="bottom")
    arrow(ax, (-9.9, 3.45), (-9.9, 2.45), color=ORANGE, width=1.1, scale=7)
    ax.text(-9.0, 2.90, "g", fontsize=6.0, color=ORANGE, style="italic",
            bbox={"boxstyle": "square,pad=0.10", "facecolor": WHITE,
                  "edgecolor": "none", "alpha": 0.85})

    # 2) Measured Up/Down envelopes; white cells are masked (no observation).
    ax = axes[1]
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.5, 36.5)
    ax.set_aspect("auto")
    # The Up map is drawn with near rings at the top so that its dense
    # overhead content sits high and both channels' masked far rings meet at
    # the central gap, echoing the ceiling-up/floor-down side view of panel 1.
    for img, cmap, y0, origin, vmin, vmax in (
        (np.ma.masked_invalid(up), "Reds", 20.0, "upper", tau, 4.0),
        (np.ma.masked_invalid(down), "Blues", 0.0, "lower", 0.0, tau),
    ):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(WHITE)
        ax.imshow(img, cmap=cm, vmin=vmin, vmax=vmax, origin=origin,
                  extent=(0, 60, y0, y0 + 16), interpolation="nearest",
                  aspect="auto")
    label_box = {"boxstyle": "square,pad=0.14", "facecolor": WHITE,
                 "edgecolor": "none", "alpha": 0.85}
    ax.text(1.2, 33.6, "Up (overhead): min height", fontsize=5.5, color=UPPER,
            weight="bold", va="top", bbox=label_box)
    ax.text(1.2, 13.6, "Down (lower): max height", fontsize=5.5, color=BLUE,
            weight="bold", va="top", bbox=label_box)

    # 3) Recorded top-100 candidate distances of this query.
    ax = axes[2]
    ranks = np.arange(1, len(distances) + 1)
    ax.plot(ranks, distances, color=INK, linewidth=0.9)
    ax.scatter([1], [distances[0]], s=12, color=ORANGE, zorder=5)
    span = float(distances.max() - distances.min())
    ax.set_xlim(-3, len(distances) + 4)
    ax.set_ylim(distances.min() - 0.26 * span, distances.max() + 0.16 * span)
    ax.set_xticks([1, 50, 100])
    ax.tick_params(axis="x", length=1.6, width=0.5, pad=1.6,
                   labelsize=5.2, colors=MID)
    ax.text(0.055, 0.62, "SCD distance", transform=ax.transAxes, rotation=90,
            ha="left", va="center", fontsize=5.4, color=MID)
    ax.annotate("top-1 =\ncorrect match\n$d_1$ = " + f"{distances[0]:.2f}",
                (1, float(distances[0])),
                textcoords="offset points", xytext=(18, 3), fontsize=5.4,
                color=ORANGE, ha="left", va="bottom", linespacing=1.2,
                arrowprops={"arrowstyle": "-", "color": ORANGE,
                            "linewidth": 0.5, "shrinkB": 2})
    ax.text(0.95, 0.075,
            "est. yaw $\\widehat{{\\Delta\\psi}}$ = " + f"{shift_deg}\u00b0",
            transform=ax.transAxes, ha="right", fontsize=5.8, color=INK)

    # 4) Seed alignment: query cloud rotated by the estimated yaw over the
    #    matched keyframe cloud, before ICP refinement (top view).
    ax = axes[3]
    rot = np.array([[np.cos(yaw_shift), -np.sin(yaw_shift)],
                    [np.sin(yaw_shift), np.cos(yaw_shift)]])
    q_seed = q_xyz[:, :2] @ rot.T
    ax.scatter(m_xyz[:, 0], m_xyz[:, 1], s=0.4, color=MID, alpha=0.5,
               linewidths=0, rasterized=True)
    ax.scatter(q_seed[:, 0], q_seed[:, 1], s=0.4, color=BLUE_2, alpha=0.5,
               linewidths=0, rasterized=True)
    half_w = 12.5
    half_h = half_w * box_ratio(ax)
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(-half_h, half_h)
    ax.set_aspect("auto")
    box = {"boxstyle": "square,pad=0.14", "facecolor": WHITE,
           "edgecolor": "none", "alpha": 0.85}
    ax.text(0.05, 0.90, "map", transform=ax.transAxes, fontsize=5.5,
            color=MID, weight="bold", bbox=box)
    ax.text(0.05, 0.78, "query", transform=ax.transAxes, fontsize=5.5,
            color=BLUE_2, weight="bold", bbox=box)

    # 5) Matched pair on the experimental map trajectory.
    ax = axes[4]
    ax.plot(traj[:, 0], traj[:, 1], color="#B4BCC4", linewidth=0.9)
    ax.scatter([cand_xy[0]], [cand_xy[1]], s=15, facecolor=WHITE,
               edgecolor=INK, linewidth=0.8, zorder=5)
    ax.scatter([truth_xy[0]], [truth_xy[1]], s=9, marker="^", color=ORANGE,
               zorder=6)
    cx = 0.5 * (traj[:, 0].min() + traj[:, 0].max())
    cy = 0.5 * (traj[:, 1].min() + traj[:, 1].max())
    half_w = 0.56 * (traj[:, 0].max() - traj[:, 0].min())
    half_h = half_w * box_ratio(ax)
    ax.set_xlim(cx - half_w, cx + half_w)
    ax.set_ylim(cy - half_h, cy + half_h)
    ax.set_aspect("auto")
    ax.annotate(
        "matched pair",
        (float(cand_xy[0]), float(cand_xy[1])),
        textcoords="offset points", xytext=(6, 8), fontsize=5.0, color=INK,
        ha="left",
        arrowprops={"arrowstyle": "-", "color": MID, "linewidth": 0.5},
    )

    # Stage arrows with the data product transferred between panels.
    labels = ("", "top-$K$", "seed", "accept")
    positions = [axis.get_position() for axis in axes]
    for (left, right), label in zip(zip(positions[:-1], positions[1:]), labels):
        y = (left.y0 + left.y1) / 2
        figure_arrow(fig, left.x1 + 0.005, right.x0 - 0.005, y=y, color=INK)
        if label:
            fig.text((left.x1 + right.x0) / 2, y + 0.050, label, ha="center",
                     va="bottom", fontsize=5.5, color=MID)
    fig.text(0.006, 0.985,
             "LIO deskew + inertial gravity \u2192 gravity canonicalization "
             "(roll/pitch removed, yaw free); shared map/query front end",
             ha="left", va="top", fontsize=5.5, color=MID)

    save_figure(fig, "updown_sc_pipeline")


def draw_room_cross_section(ax: mpl.axes.Axes, dual: bool) -> None:
    ax.plot([0.12, 0.90], [0.78, 0.78], color=INK, linewidth=2.0)
    ax.text(0.90, 0.81, "ceiling", ha="right", fontsize=5.2, color=INK)
    ax.add_patch(Rectangle((0.19, 0.20), 0.22, 0.28, facecolor=BLUE_LIGHT, edgecolor=BLUE_2, linewidth=0.8))
    ax.text(0.30, 0.17, "shelf", ha="center", fontsize=5.0, color=BLUE)
    ax.add_patch(Rectangle((0.56, 0.20), 0.08, 0.38, facecolor="#E6E8EB", edgecolor=MID, linewidth=0.7))
    ax.add_patch(Circle((0.60, 0.63), 0.045, facecolor="#E6E8EB", edgecolor=MID, linewidth=0.7))
    ax.text(0.60, 0.14, "person", ha="center", fontsize=5.0, color=MID)
    ax.plot([0.10, 0.92], [0.20, 0.20], color=MID, linewidth=0.8)
    ax.scatter([0.22, 0.29, 0.37, 0.58, 0.62, 0.71, 0.79], [0.48, 0.47, 0.48, 0.58, 0.56, 0.78, 0.78], s=8, color=INK)

    if not dual:
        ax.plot([0.13, 0.88], [0.78, 0.78], color=RED, linewidth=3.0, alpha=0.75)
        arrow(ax, (0.78, 0.68), (0.78, 0.76), color=RED, scale=7)
        ax.text(0.49, 0.69, "selected max z", ha="center", fontsize=5.4, color=RED, weight="bold")
        ax.plot([0.21, 0.40], [0.42, 0.24], color=RED, linewidth=1.0)
        ax.plot([0.40, 0.21], [0.42, 0.24], color=RED, linewidth=1.0)
        ax.text(0.31, 0.075, "lower structure\nnot represented", ha="center", fontsize=5.0, color=RED, linespacing=0.95)
        return

    ax.plot([0.12, 0.90], [0.59, 0.59], color=ORANGE, linewidth=0.9, linestyle=(0, (3, 2)))
    ax.text(0.91, 0.59, "τ", ha="left", va="center", fontsize=6, color=ORANGE)
    ax.plot([0.18, 0.42], [0.48, 0.48], color=BLUE, linewidth=3.0)
    ax.plot([0.66, 0.87], [0.78, 0.78], color=UPPER, linewidth=3.0)
    ax.text(0.30, 0.54, "Down: max below τ", ha="center", fontsize=5.0, color=BLUE, weight="bold")
    ax.text(0.76, 0.70, "Up: min above τ", ha="center", fontsize=5.0, color=UPPER, weight="bold")


def draw_binary_grid(
    ax: mpl.axes.Axes,
    origin: tuple[float, float],
    values: np.ndarray,
    cell: float,
    color: str,
    label: str,
) -> None:
    rows, cols = values.shape
    for row in range(rows):
        for col in range(cols):
            value = int(values[row, col])
            ax.add_patch(
                Rectangle(
                    (origin[0] + col * cell, origin[1] + (rows - 1 - row) * cell),
                    cell * 0.92,
                    cell * 0.92,
                    facecolor=color if value else WHITE,
                    edgecolor=BORDER,
                    linewidth=0.4,
                    alpha=0.85 if value else 1.0,
                )
            )
    ax.text(origin[0] + cols * cell / 2, origin[1] + rows * cell + 0.025, label, ha="center", fontsize=5.3, color=INK)


def make_principle() -> None:
    fig, axs = plt.subplots(2, 2, figsize=(3.50, 3.78), facecolor=WHITE)
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.050, top=0.925, hspace=0.11, wspace=0.10)
    fig.suptitle("Dual-envelope encoding and matching", fontsize=8.2, weight="bold", color=INK, y=0.985)

    ax = axs[0, 0]
    panel_frame(ax, "a", "Conventional SC", "#FCF8F8")
    draw_room_cross_section(ax, dual=False)

    ax = axs[0, 1]
    panel_frame(ax, "b", "Strict dual envelope", "#F7FAFD")
    draw_room_cross_section(ax, dual=True)

    ax = axs[1, 0]
    panel_frame(ax, "c", "Joint-valid masks", "#FAFBFC")
    query = np.array([[1, 1, 0, 1, 0], [1, 0, 1, 1, 1], [0, 1, 1, 0, 1]], dtype=int)
    mapped = np.array([[1, 0, 1, 1, 0], [1, 1, 1, 0, 1], [0, 1, 0, 1, 1]], dtype=int)
    joint = query & mapped
    draw_binary_grid(ax, (0.08, 0.54), query, 0.055, BLUE_2, "query M")
    ax.text(0.39, 0.62, "∧", ha="center", va="center", fontsize=9, color=MID)
    draw_binary_grid(ax, (0.46, 0.54), mapped, 0.055, TEAL, "map M")
    arrow(ax, (0.75, 0.62), (0.83, 0.62), color=MID, scale=6)
    draw_binary_grid(ax, (0.69, 0.29), joint, 0.055, PURPLE, "joint m")
    ax.text(0.27, 0.39, "cosine uses only\njoint-valid rings", ha="center", va="center", fontsize=5.2, color=INK)
    ax.text(0.27, 0.25, "joint rings ≥ 2", ha="center", fontsize=5.1, color=PURPLE)
    tag(ax, 0.50, 0.10, "mask similarity × sector support", PURPLE_LIGHT, PURPLE)

    ax = axs[1, 1]
    panel_frame(ax, "d", "Conditional vertical seed", "#FAFBFC")
    ax.text(0.50, 0.83, "place + yaw fixed", ha="center", fontsize=5.2, color=PURPLE, weight="bold")
    ax.plot([0.14, 0.86], [0.61, 0.61], color=ORANGE, linewidth=0.9, linestyle=(0, (3, 2)))
    ax.text(0.87, 0.61, "τ", ha="left", va="center", fontsize=6, color=ORANGE)
    q_lower = (0.34, 0.42, 0.50, 0.55)
    m_lower = (0.38, 0.47, 0.53, 0.57)
    q_upper = (0.68, 0.73, 0.78)
    m_upper = (0.72, 0.76, 0.82)
    for q, m in zip(q_lower, m_lower):
        ax.plot([0.18, 0.40], [q, m], color=BLUE_2, linewidth=0.8)
        ax.scatter([0.18, 0.40], [q, m], s=7, color=BLUE_2)
    for q, m in zip(q_upper, m_upper):
        ax.plot([0.18, 0.40], [q, m], color=UPPER, linewidth=0.8)
        ax.scatter([0.18, 0.40], [q, m], s=7, color=UPPER)
    ax.text(0.18, 0.24, "query", ha="center", fontsize=5.0, color=MID)
    ax.text(0.40, 0.24, "map", ha="center", fontsize=5.0, color=MID)
    arrow(ax, (0.47, 0.51), (0.59, 0.51), color=ORANGE, width=1.2, scale=7)
    ax.text(0.53, 0.56, "r = zm − zq", ha="center", fontsize=5.0, color=ORANGE)
    ax.text(0.72, 0.70, "rank by distance\nfrom split τ", ha="center", fontsize=5.0, color=INK)
    ax.text(0.72, 0.43, "keep stable 50%", ha="center", fontsize=5.2, color=PURPLE, weight="bold")
    arrow(ax, (0.72, 0.36), (0.72, 0.27), color=ORANGE, width=1.2, scale=7)
    tag(ax, 0.72, 0.17, "weighted median → Δz", ORANGE_LIGHT, ORANGE)

    fig.text(
        0.985,
        0.012,
        "Conceptual schematic; τ and 50% are configurable in the implementation",
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=MID,
    )
    save_figure(fig, "updown_sc_principle")


def main() -> None:
    make_place_recognition_pipeline()
    print(f"Wrote retrieval pipeline schematic to {OUT_DIR}")


if __name__ == "__main__":
    main()
