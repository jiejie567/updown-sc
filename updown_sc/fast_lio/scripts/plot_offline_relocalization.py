#!/usr/bin/env python3
"""Generate top-down PNG summaries for an offline relocalization run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D


def load_binary_pcd_xy(path: Path) -> np.ndarray:
    fields: list[str] = []
    sizes: list[int] = []
    types: list[str] = []
    counts: list[int] = []
    points = 0
    data_offset = 0
    data_kind = ""
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"incomplete PCD header: {path}")
            text = line.decode("ascii", errors="strict").strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            key, values = parts[0].upper(), parts[1:]
            if key == "FIELDS":
                fields = values
            elif key == "SIZE":
                sizes = [int(value) for value in values]
            elif key == "TYPE":
                types = values
            elif key == "COUNT":
                counts = [int(value) for value in values]
            elif key == "POINTS":
                points = int(values[0])
            elif key == "DATA":
                data_kind = values[0].lower()
                data_offset = stream.tell()
                break
    if data_kind != "binary":
        raise ValueError(f"only binary PCD is supported, got DATA {data_kind}: {path}")
    if not counts:
        counts = [1] * len(fields)
    numpy_types = {
        ("F", 4): "<f4", ("F", 8): "<f8", ("I", 1): "i1", ("I", 2): "<i2",
        ("I", 4): "<i4", ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4",
    }
    dtype_fields = []
    for name, size, type_code, count in zip(fields, sizes, types, counts):
        numpy_type = numpy_types.get((type_code.upper(), size))
        if numpy_type is None:
            raise ValueError(f"unsupported PCD field: {name} {type_code}{size}")
        dtype_fields.append((name, numpy_type) if count == 1 else (name, numpy_type, (count,)))
    cloud = np.fromfile(path, dtype=np.dtype(dtype_fields), count=points, offset=data_offset)
    if cloud.dtype.names is None or "x" not in cloud.dtype.names or "y" not in cloud.dtype.names:
        raise ValueError(f"PCD has no x/y fields: {path}")
    xy = np.column_stack((cloud["x"], cloud["y"])).astype(np.float64)
    return xy[np.isfinite(xy).all(axis=1)]


def load_trajectory(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as stream:
        first_line = stream.readline().strip()
    if "," in first_line and "x" in first_line.lower():
        points = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                points.append((float(row["x"]), float(row["y"])))
        return np.asarray(points, dtype=np.float64).reshape((-1, 2))
    values = np.loadtxt(path, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    values = np.atleast_2d(values)
    if values.shape[1] < 3:
        raise ValueError(f"trajectory needs at least timestamp,x,y columns: {path}")
    return values[:, 1:3]


def load_results(path: Path) -> list[dict[str, float | int | bool]]:
    results: list[dict[str, float | int | bool]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            x, y = float(row["tx"]), float(row["ty"])
            if not np.isfinite((x, y)).all():
                continue
            results.append(
                {
                    "window": int(row["window"]),
                    "success": row["success"].strip().lower() == "true",
                    "fitness": float(row["fitness"]),
                    "x": x,
                    "y": y,
                }
            )
    return results


def load_truth(path: Path | None) -> dict[int, tuple[float, float]]:
    if path is None:
        return {}
    truth: dict[int, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            x_key = "map_x" if "map_x" in row else "x"
            y_key = "map_y" if "map_y" in row else "y"
            truth[int(row["window"])] = (float(row[x_key]), float(row[y_key]))
    return truth


def density_image(ax: plt.Axes, map_xy: np.ndarray, bounds: tuple[float, float, float, float]) -> None:
    x_min, x_max, y_min, y_max = bounds
    bins_x = 1500
    aspect = max((y_max - y_min) / max(x_max - x_min, 1e-6), 0.2)
    bins_y = max(350, int(bins_x * aspect))
    density, _, _ = np.histogram2d(
        map_xy[:, 0], map_xy[:, 1], bins=(bins_x, bins_y),
        range=((x_min, x_max), (y_min, y_max)),
    )
    positive = density[density > 0]
    vmax = max(float(np.percentile(positive, 99.7)) if positive.size else 1.0, 2.0)
    ax.imshow(
        density.T, origin="lower", extent=bounds, cmap="Greys_r",
        norm=LogNorm(vmin=1.0, vmax=vmax), interpolation="nearest", zorder=0,
    )


def padded_bounds(points: np.ndarray, padding: float, minimum_span: float = 0.0) -> tuple[float, float, float, float]:
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    center = (low + high) * 0.5
    span = np.maximum(high - low, minimum_span)
    low = center - span * 0.5 - padding
    high = center + span * 0.5 + padding
    return float(low[0]), float(high[0]), float(low[1]), float(high[1])


def plot_report(
    path: Path,
    map_xy: np.ndarray,
    trajectory: np.ndarray,
    results: list[dict[str, float | int | bool]],
    truth: dict[int, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    title: str,
) -> None:
    figure, ax = plt.subplots(figsize=(12.5, 10), dpi=170, facecolor="#081018")
    ax.set_facecolor("#081018")
    density_image(ax, map_xy, bounds)
    if trajectory.size:
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="#25c6da", linewidth=1.05, alpha=0.82, zorder=3)
        ax.scatter(trajectory[:, 0], trajectory[:, 1], s=2.2, color="#80deea", alpha=0.65, linewidths=0, zorder=4)

    passed = [row for row in results if bool(row["success"])]
    failed = [row for row in results if not bool(row["success"])]
    if passed:
        xy = np.array([[row["x"], row["y"]] for row in passed], dtype=float)
        ax.scatter(xy[:, 0], xy[:, 1], s=28, marker="o", color="#43d17a", edgecolors="white", linewidths=0.45, zorder=7)
    if failed:
        xy = np.array([[row["x"], row["y"]] for row in failed], dtype=float)
        ax.scatter(xy[:, 0], xy[:, 1], s=52, marker="X", color="#ff4d4f", edgecolors="white", linewidths=0.65, zorder=8)
        labeled = failed if len(failed) <= 20 else sorted(failed, key=lambda row: float(row["fitness"]), reverse=True)[:20]
        for row in labeled:
            ax.annotate(
                f"W{row['window']}", (float(row["x"]), float(row["y"])), xytext=(5, 5),
                textcoords="offset points", color="#ffb3b4", fontsize=7.5, zorder=9,
            )

    result_by_window = {int(row["window"]): row for row in results}
    for window, true_xy in truth.items():
        ax.scatter([true_xy[0]], [true_xy[1]], s=95, marker="*", color="#ffd54f", edgecolors="white", linewidths=0.8, zorder=10)
        ax.annotate(
            f"W{window} true", true_xy, xytext=(6, 6), textcoords="offset points",
            color="white", fontsize=8, fontweight="bold", zorder=11,
        )
        result = result_by_window.get(window)
        if result is not None:
            ax.plot(
                [true_xy[0], float(result["x"])], [true_xy[1], float(result["y"])],
                color="#ffd54f", linewidth=0.8, linestyle=(0, (2, 3)), alpha=0.65, zorder=6,
            )

    handles = [
        Line2D([0], [0], color="#25c6da", lw=1.5, marker=".", label="SCD mapping trajectory"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#43d17a", markeredgecolor="white", markersize=8, label="Evaluator pass"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#ff4d4f", markeredgecolor="white", markersize=8, label="Evaluator reject"),
    ]
    if truth:
        handles.append(Line2D([0], [0], marker="*", color="none", markerfacecolor="#ffd54f", markeredgecolor="white", markersize=10, label="Provided true position"))
    legend = ax.legend(handles=handles, loc="upper right", fontsize=8.5, facecolor="#111820", edgecolor="#65727e")
    for text in legend.get_texts():
        text.set_color("white")

    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.set_xlabel("Map X [m]", color="white")
    ax.set_ylabel("Map Y [m]", color="white")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="white", alpha=0.12, linewidth=0.5)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#777777")
    figure.tight_layout()
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    map_xy = load_binary_pcd_xy(args.map)
    trajectory = load_trajectory(args.trajectory)
    results = load_results(args.results)
    evaluated_windows = {int(row["window"]) for row in results}
    truth = {
        window: xy for window, xy in load_truth(args.truth_csv).items()
        if window in evaluated_windows
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overview_points = [map_xy]
    if trajectory.size:
        overview_points.append(trajectory)
    overview_bounds = padded_bounds(np.vstack(overview_points), 2.0)
    diagnostic_points = [np.array([[row["x"], row["y"]] for row in results], dtype=float)] if results else []
    if truth:
        diagnostic_points.append(np.asarray(list(truth.values()), dtype=float))
    focus_bounds = padded_bounds(np.vstack(diagnostic_points), 4.0, 20.0) if diagnostic_points else overview_bounds
    pass_count = sum(bool(row["success"]) for row in results)
    title = f"Offline relocalization: {pass_count} pass / {len(results) - pass_count} rejected"

    plot_report(
        args.output_dir / "relocalization_top_view.png",
        map_xy, trajectory, results, truth, overview_bounds, title,
    )
    plot_report(
        args.output_dir / "relocalization_top_view_focus.png",
        map_xy, trajectory, results, truth, focus_bounds, title + " (diagnostic zoom)",
    )
    print(args.output_dir / "relocalization_top_view.png")
    print(args.output_dir / "relocalization_top_view_focus.png")


if __name__ == "__main__":
    main()
