#!/usr/bin/env python3
"""Generate a data-traceable roof-contamination motivation figure.

The representative query is selected by a fixed rule: maximize the number of
conventional Scan Context cells whose maximum height exceeds the 2.1 m split
selected by the final IH map.
This selects the scan with the broadest overhead-dominated conventional SC,
while the dual envelopes retain the complementary lower/middle observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


NUM_RINGS = 16
NUM_SECTORS = 60
MAX_RADIUS_M = 30.0
SPLIT_HEIGHT_M = 2.1
VOXEL_M = 0.25

BLUE = "#3775BA"
ORANGE = "#D97706"
RED = "#B64342"
INK = "#25313C"
MID = "#65717C"
LIGHT = "#EEF2F5"
BORDER = "#C9D1D8"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


@dataclass
class QueryRecord:
    query_id: int
    window: int
    truth_valid: bool
    cloud_path: Path
    up: np.ndarray


@dataclass
class DescriptorRecord:
    points_raw: int
    points_after_filter: int
    points_voxelized: int
    xyz: np.ndarray
    render_xyz: np.ndarray
    sc_max: np.ndarray
    sc_valid: np.ndarray
    lower_max: np.ndarray
    lower_valid: np.ndarray
    upper_min: np.ndarray
    upper_valid: np.ndarray

    @property
    def hidden_mask(self) -> np.ndarray:
        return self.lower_valid & self.upper_valid

    @property
    def occupied_mask(self) -> np.ndarray:
        return self.lower_valid | self.upper_valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--query-id", type=int)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(__file__).resolve().parent / "roof_contamination",
    )
    return parser.parse_args()


def load_queries(query_dir: Path) -> list[QueryRecord]:
    gravity: dict[int, np.ndarray] = {}
    with (query_dir / "gravity.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            gravity[int(row["query_id"])] = np.array(
                [float(row["up_x"]), float(row["up_y"]), float(row["up_z"])],
                dtype=float,
            )

    records: list[QueryRecord] = []
    with (query_dir / "metadata.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            query_id = int(row["query_id"])
            if query_id not in gravity:
                continue
            records.append(
                QueryRecord(
                    query_id=query_id,
                    window=int(row["window"]),
                    truth_valid=row["truth_valid"].lower() == "true",
                    cloud_path=query_dir / row["file"],
                    up=gravity[query_id],
                )
            )
    return records


def gravity_rotation(up: np.ndarray) -> np.ndarray:
    source = up / np.linalg.norm(up)
    target = np.array([0.0, 0.0, 1.0])
    cross = np.cross(source, target)
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    sine = float(np.linalg.norm(cross))
    if sine < 1e-12:
        return np.eye(3) if cosine > 0.0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - cosine) / (sine * sine))


def voxel_centroids(xyz: np.ndarray, leaf: float) -> np.ndarray:
    keys = np.floor(xyz / leaf).astype(np.int32)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    np.add.at(sums, inverse, xyz)
    counts = np.bincount(inverse)
    return sums / counts[:, None]


def bin_indices(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    theta = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2.0 * np.pi)
    ring = np.ceil(radius / MAX_RADIUS_M * NUM_RINGS).astype(int) - 1
    sector = np.ceil(theta / (2.0 * np.pi) * NUM_SECTORS).astype(int) - 1
    return (
        np.clip(ring, 0, NUM_RINGS - 1),
        np.clip(sector, 0, NUM_SECTORS - 1),
        radius,
    )


def make_descriptor(record: QueryRecord) -> DescriptorRecord:
    points = np.fromfile(record.cloud_path, dtype=np.float32).reshape(-1, 4)
    points_raw = len(points)
    points = points[np.isfinite(points[:, :3]).all(axis=1)]
    xy_sq = points[:, 0] ** 2 + points[:, 1] ** 2
    in_blind = (
        (xy_sq <= 0.3**2)
        & (points[:, 2] >= -0.5)
        & (points[:, 2] <= 2.0)
    )
    points = points[~in_blind]
    canonical = (gravity_rotation(record.up) @ points[:, :3].T).T
    radius = np.hypot(canonical[:, 0], canonical[:, 1])
    in_range = (radius > 1e-6) & (radius <= MAX_RADIUS_M)
    canonical = canonical[in_range]
    points_after_filter = len(canonical)
    xyz = voxel_centroids(canonical, VOXEL_M)
    ring, sector, radius = bin_indices(xyz)
    keep = (radius > 1e-6) & (radius <= MAX_RADIUS_M)
    xyz, ring, sector = xyz[keep], ring[keep], sector[keep]
    flat = ring * NUM_SECTORS + sector
    cells = NUM_RINGS * NUM_SECTORS

    sc_flat = np.full(cells, -np.inf)
    np.maximum.at(sc_flat, flat, xyz[:, 2])
    sc_valid = np.isfinite(sc_flat)

    lower = xyz[:, 2] <= SPLIT_HEIGHT_M
    lower_flat = np.full(cells, -np.inf)
    np.maximum.at(lower_flat, flat[lower], xyz[lower, 2])
    lower_valid = np.isfinite(lower_flat)

    upper_flat = np.full(cells, np.inf)
    np.minimum.at(upper_flat, flat[~lower], xyz[~lower, 2])
    upper_valid = np.isfinite(upper_flat)

    return DescriptorRecord(
        points_raw=points_raw,
        points_after_filter=points_after_filter,
        points_voxelized=len(xyz),
        xyz=xyz,
        render_xyz=canonical,
        sc_max=sc_flat.reshape(NUM_RINGS, NUM_SECTORS),
        sc_valid=sc_valid.reshape(NUM_RINGS, NUM_SECTORS),
        lower_max=lower_flat.reshape(NUM_RINGS, NUM_SECTORS),
        lower_valid=lower_valid.reshape(NUM_RINGS, NUM_SECTORS),
        upper_min=upper_flat.reshape(NUM_RINGS, NUM_SECTORS),
        upper_valid=upper_valid.reshape(NUM_RINGS, NUM_SECTORS),
    )


def select_query(
    records: list[QueryRecord], requested_id: int | None
) -> tuple[QueryRecord, DescriptorRecord, list[dict[str, float | int]]]:
    summaries: list[dict[str, float | int]] = []
    cache: dict[int, DescriptorRecord] = {}
    for record in records:
        if not record.truth_valid:
            continue
        descriptor = make_descriptor(record)
        cache[record.query_id] = descriptor
        hidden = int(descriptor.hidden_mask.sum())
        occupied = int(descriptor.occupied_mask.sum())
        overhead_dominated = int(
            (descriptor.sc_valid & (descriptor.sc_max > SPLIT_HEIGHT_M)).sum()
        )
        display_overrange = int(
            (descriptor.sc_valid & (descriptor.sc_max >= 4.0)).sum()
        )
        summaries.append(
            {
                "query_id": record.query_id,
                "window": record.window,
                "hidden_cells": hidden,
                "occupied_cells": occupied,
                "hidden_fraction": hidden / occupied if occupied else 0.0,
                "overhead_dominated_cells": overhead_dominated,
                "overhead_dominated_fraction": (
                    overhead_dominated / occupied if occupied else 0.0
                ),
                "display_overrange_cells": display_overrange,
                "lower_cells": int(descriptor.lower_valid.sum()),
                "upper_cells": int(descriptor.upper_valid.sum()),
            }
        )
    summaries.sort(
        key=lambda row: (
            int(row["overhead_dominated_cells"]),
            int(row["display_overrange_cells"]),
            int(row["hidden_cells"]),
        ),
        reverse=True,
    )
    if requested_id is None:
        chosen_id = int(summaries[0]["query_id"])
    else:
        chosen_id = requested_id
        if chosen_id not in cache:
            raise ValueError(f"query {chosen_id} is unavailable or lacks valid truth")
    chosen = next(record for record in records if record.query_id == chosen_id)
    return chosen, cache[chosen_id], summaries


def masked(values: np.ndarray, valid: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(~valid, values)


def draw_polar_descriptor(
    ax: mpl.axes.Axes,
    values: np.ndarray,
    valid: np.ndarray,
    title: str,
    subtitle: str,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    hidden: np.ndarray | None = None,
    subtitle_color: str = MID,
) -> mpl.collections.QuadMesh:
    theta_edges = np.linspace(0.0, 2.0 * np.pi, NUM_SECTORS + 1)
    radius_edges = np.linspace(0.0, MAX_RADIUS_M, NUM_RINGS + 1)
    ax.grid(False)
    image = ax.pcolormesh(
        theta_edges,
        radius_edges,
        masked(values, valid),
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    if hidden is not None:
        ring_idx, sector_idx = np.nonzero(hidden)
        theta = (sector_idx + 0.5) * 2.0 * np.pi / NUM_SECTORS
        radius = (ring_idx + 0.5) * MAX_RADIUS_M / NUM_RINGS
        ax.scatter(theta, radius, s=2.2, marker=".", color=RED, alpha=0.9, zorder=5)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0.0, MAX_RADIUS_M)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels([])
    ax.set_yticks([10, 20, 30])
    ax.set_yticklabels([])
    ax.grid(color="white", linewidth=0.45, alpha=0.75)
    ax.spines["polar"].set_color(BORDER)
    ax.spines["polar"].set_linewidth(0.7)
    ax.set_title(title, fontsize=7.2, fontweight="bold", color=INK, pad=8)
    if subtitle:
        ax.text(
            0.5,
            -0.11,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=5.7,
            color=subtitle_color,
        )
    return image


def add_height_colorbar(
    fig: mpl.figure.Figure,
    image: mpl.collections.QuadMesh,
    axes: list[mpl.axes.Axes],
) -> None:
    colorbar = fig.colorbar(
        image,
        ax=axes,
        orientation="vertical",
        fraction=0.025,
        pad=0.025,
        aspect=25,
        extend="max",
    )
    colorbar.set_label(r"height $z$ (m)", fontsize=5.7, labelpad=2)
    colorbar.ax.tick_params(labelsize=5.2, length=2, width=0.5)
    colorbar.outline.set_linewidth(0.5)


def save_figure(
    fig: mpl.figure.Figure,
    output_prefix: Path,
    *,
    tight: bool = True,
) -> None:
    bbox = "tight" if tight else None
    pad = 0.025 if tight else 0.0
    fig.savefig(
        output_prefix.with_suffix(".svg"),
        dpi=600,
        bbox_inches=bbox,
        pad_inches=pad,
    )
    fig.savefig(
        output_prefix.with_suffix(".pdf"),
        dpi=600,
        bbox_inches=bbox,
        pad_inches=pad,
    )
    fig.savefig(
        output_prefix.with_suffix(".png"),
        dpi=600,
        bbox_inches=bbox,
        pad_inches=pad,
        facecolor=fig.get_facecolor(),
        transparent=False,
    )
    plt.close(fig)


def draw_topdown_cloud(
    descriptor: DescriptorRecord,
    output_prefix: Path,
) -> None:
    """Render the selected scan from a white-background top view."""
    xyz = descriptor.render_xyz
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]

    # A compact landscape canvas keeps the scan visibly comparable with the
    # adjacent polar descriptors without making panel (a) as tall as (b)/(c).
    fig, ax = plt.subplots(figsize=(1.78, 1.62), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        s=0.80,
        color="#234E70",
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter([0.0], [0.0], s=7.0, color=ORANGE, edgecolor="white", linewidth=0.35, zorder=4)
    # Sparse returns near the 30 m range boundary otherwise make the structural
    # scan footprint illegibly small. Crop only the display to robust XY limits;
    # descriptor construction and every reported count still use all points.
    low = np.percentile(xyz[:, :2], 0.1, axis=0)
    high = np.percentile(xyz[:, :2], 99.9, axis=0)
    center = 0.5 * (low + high)
    side = 1.06 * float(np.max(high - low))
    x_min, x_max = center[0] - 0.5 * side, center[0] + 0.5 * side
    y_min, y_max = center[1] - 0.5 * side, center[1] + 0.5 * side
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    # This source font is intentionally smaller than the polar-panel source
    # font: after the different LaTeX width scalings both render identically.
    ax.set_title("Measured scan", fontsize=5.70, fontweight="bold", color=INK, pad=1)
    scale_x0 = x_min + 0.07 * side
    scale_y = y_min + 0.08 * side
    ax.plot([scale_x0, scale_x0 + 10.0], [scale_y, scale_y], color=INK, linewidth=1.2)
    ax.text(scale_x0 + 5.0, scale_y - 0.025 * side, "10 m", ha="center", va="top", fontsize=5.0, color=MID)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.92)
    save_figure(fig, output_prefix, tight=False)
    # Some PDF viewers render an RGBA image's transparency mask against black.
    # Store the manuscript raster as opaque RGB so the white background is
    # invariant across viewers.
    png_path = output_prefix.with_suffix(".png")
    with Image.open(png_path) as image:
        image.convert("RGB").save(png_path, dpi=(600, 600))


def draw_figure(descriptor: DescriptorRecord, output_prefix: Path) -> None:
    hidden = descriptor.hidden_mask

    # Keep ordinary indoor ceiling heights visually discriminative. Values
    # above 4 m remain unchanged in source data and use the colorbar's over color.
    norm = mpl.colors.Normalize(vmin=-0.5, vmax=4.0, clip=False)
    cmap = LinearSegmentedColormap.from_list(
        "height",
        ["#263B73", "#3775BA", "#D9EFEA", "#F4C66A", "#B64342"],
    ).copy()
    cmap.set_bad(LIGHT)

    fig = plt.figure(figsize=(7.16, 2.35), facecolor="white")
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        left=0.04,
        right=0.94,
        bottom=0.07,
        top=0.82,
        wspace=0.16,
    )

    axes = [fig.add_subplot(grid[0, index], projection="polar") for index in range(3)]
    image = draw_polar_descriptor(
        axes[0],
        descriptor.sc_max,
        descriptor.sc_valid,
        "Conventional SC",
        "",
        cmap,
        norm,
        hidden=hidden,
        subtitle_color=RED,
    )
    draw_polar_descriptor(
        axes[1],
        descriptor.upper_min,
        descriptor.upper_valid,
        "Up: min above split",
        "",
        cmap,
        norm,
    )
    draw_polar_descriptor(
        axes[2],
        descriptor.lower_max,
        descriptor.lower_valid,
        "Down: max below split",
        "",
        cmap,
        norm,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    draw_topdown_cloud(
        descriptor,
        output_prefix.with_name(output_prefix.name + "_cloud"),
    )
    add_height_colorbar(fig, image, axes)
    save_figure(fig, output_prefix)

    sc_prefix = output_prefix.with_name(output_prefix.name + "_sc")
    sc_fig = plt.figure(figsize=(2.15, 2.15), facecolor="white")
    sc_ax = sc_fig.add_subplot(111, projection="polar")
    sc_fig.subplots_adjust(left=0.05, right=0.95, bottom=0.04, top=0.82)
    draw_polar_descriptor(
        sc_ax,
        descriptor.sc_max,
        descriptor.sc_valid,
        "Conventional SC",
        "",
        cmap,
        norm,
        hidden=hidden,
        subtitle_color=RED,
    )
    save_figure(sc_fig, sc_prefix)

    updown_prefix = output_prefix.with_name(output_prefix.name + "_updown")
    updown_fig = plt.figure(figsize=(4.65, 2.15), facecolor="white")
    updown_grid = updown_fig.add_gridspec(
        1,
        2,
        left=0.025,
        right=0.91,
        bottom=0.04,
        top=0.82,
        wspace=0.14,
    )
    updown_axes = [
        updown_fig.add_subplot(updown_grid[0, index], projection="polar")
        for index in range(2)
    ]
    updown_image = draw_polar_descriptor(
        updown_axes[0],
        descriptor.upper_min,
        descriptor.upper_valid,
        "Up: min above split",
        "",
        cmap,
        norm,
    )
    draw_polar_descriptor(
        updown_axes[1],
        descriptor.lower_max,
        descriptor.lower_valid,
        "Down: max below split",
        "",
        cmap,
        norm,
    )
    add_height_colorbar(updown_fig, updown_image, updown_axes)
    save_figure(updown_fig, updown_prefix)


def write_source_data(
    record: QueryRecord,
    descriptor: DescriptorRecord,
    summaries: list[dict[str, float | int]],
    output_prefix: Path,
) -> None:
    selection_path = output_prefix.with_name(output_prefix.name + "_selection.csv")
    with selection_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    source_path = output_prefix.with_name(output_prefix.name + "_source_data.csv")
    fields = [
        "query_id",
        "window",
        "ring",
        "sector",
        "r_inner_m",
        "r_outer_m",
        "theta_start_deg",
        "theta_end_deg",
        "sc_max_z_m",
        "sc_valid",
        "lower_max_z_m",
        "lower_valid",
        "upper_min_z_m",
        "upper_valid",
        "roof_hides_lower",
    ]
    with source_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ring in range(NUM_RINGS):
            for sector in range(NUM_SECTORS):
                sc_valid = bool(descriptor.sc_valid[ring, sector])
                lower_valid = bool(descriptor.lower_valid[ring, sector])
                upper_valid = bool(descriptor.upper_valid[ring, sector])
                writer.writerow(
                    {
                        "query_id": record.query_id,
                        "window": record.window,
                        "ring": ring,
                        "sector": sector,
                        "r_inner_m": ring * MAX_RADIUS_M / NUM_RINGS,
                        "r_outer_m": (ring + 1) * MAX_RADIUS_M / NUM_RINGS,
                        "theta_start_deg": sector * 360.0 / NUM_SECTORS,
                        "theta_end_deg": (sector + 1) * 360.0 / NUM_SECTORS,
                        "sc_max_z_m": descriptor.sc_max[ring, sector] if sc_valid else "",
                        "sc_valid": int(sc_valid),
                        "lower_max_z_m": descriptor.lower_max[ring, sector] if lower_valid else "",
                        "lower_valid": int(lower_valid),
                        "upper_min_z_m": descriptor.upper_min[ring, sector] if upper_valid else "",
                        "upper_valid": int(upper_valid),
                        "roof_hides_lower": int(lower_valid and upper_valid),
                    }
                )

    chosen_summary = next(row for row in summaries if int(row["query_id"]) == record.query_id)
    metadata = {
        "selection_rule": (
            "maximize the number of conventional Scan Context cells whose maximum height exceeds the "
            "2.1 m map-adaptive IH split; tie-break by the number at or above the 4.0 m display over-range threshold, "
            "then by the number jointly occupied by lower/middle and overhead points"
        ),
        "evaluated_truth_valid_queries": len(summaries),
        "chosen": chosen_summary,
        "descriptor": {
            "num_rings": NUM_RINGS,
            "num_sectors": NUM_SECTORS,
            "max_radius_m": MAX_RADIUS_M,
            "split_height_m": SPLIT_HEIGHT_M,
            "voxel_m": VOXEL_M,
            "sc_visualization": "maximum signed z on the same 16x60 grid",
            "lower_envelope": "maximum signed z for z <= split height",
            "upper_envelope": "minimum signed z for z > split height",
        },
        "point_counts": {
            "raw": descriptor.points_raw,
            "after_blind_and_range_filter": descriptor.points_after_filter,
            "voxel_centroids_aggregated": descriptor.points_voxelized,
        },
        "image_integrity": (
            "The top-view panel and all three descriptor panels come from the same query frame. "
            "The cloud panel shows the scan as a single-color XY projection on a white background; "
            "reflectance/intensity is not used and no point is spatially edited. Its display-only XY limits "
            "use the 0.1--99.9 coordinate percentiles so sparse range-boundary returns do not shrink the visible "
            "structure; descriptor construction and reported counts still use every valid in-range point. "
            "Descriptor panels are computed from the same retained voxel centroids. Polar cells use direct "
            "max/min aggregation, invalid cells are masked gray, and no spatial interpolation or manual "
            "cell editing is applied. Gravity canonicalization is common preprocessing and is intentionally "
            "not presented as a separate panel. "
            "The shared display scale is capped at 4 m with an explicit over-range color; uncapped values "
            "remain in the source-data CSV."
        ),
        "export_contract": (
            "Editable SVG and embedded-font PDF are the submission assets; the dense 3-D point layer is "
            "rasterized inside those vector containers while labels remain editable. The PNG is a 600 dpi preview. "
            "TIFF is intentionally omitted because the target IEEE workflow accepts vector PDF and the "
            "project minimizes redundant raster storage."
        ),
    }
    metadata_path = output_prefix.with_name(output_prefix.name + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    records = load_queries(args.query_dir)
    chosen, descriptor, summaries = select_query(records, args.query_id)
    write_source_data(chosen, descriptor, summaries, args.output_prefix)
    draw_figure(descriptor, args.output_prefix)
    summary = next(row for row in summaries if int(row["query_id"]) == chosen.query_id)
    print(json.dumps(summary, indent=2))
    print(args.output_prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
