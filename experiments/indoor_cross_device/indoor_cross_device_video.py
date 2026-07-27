#!/usr/bin/env python3
"""Data-driven Manim chapter for the indoor cross-device/platform replay."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image
from manim import (
    CapStyleType,
    Create,
    DOWN,
    Dot,
    FadeIn,
    FadeOut,
    GREEN,
    ImageMobject,
    LEFT,
    Line,
    LineJointType,
    Rectangle,
    RIGHT,
    Scene,
    Text,
    UP,
    ValueTracker,
    VGroup,
    VMobject,
    WHITE,
    always_redraw,
    config,
    linear,
)


ROOT = Path(
    os.environ.get(
        "INDOOR_CROSS_DEVICE_EXPERIMENT",
        "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/indoor_cross_device_2m",
    )
)
VIDEO_DIR = ROOT / "video"
ASSET_DIR = VIDEO_DIR / "assets"

config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30
config.background_color = "#08111F"
config.frame_width = 16.0
config.frame_height = 9.0

BG = "#08111F"
PANEL = "#102035"
MUTED = "#91A4B8"
MAP_POINTS = "#657B8C"
MAP_TRACK = "#438BDE"
HANDLE = "#F39A45"
VEHICLE = "#37C68A"
REFERENCE = "#E6C65A"
GRID = "#42566C"
GOOD = "#35D07F"
BAD = "#F05B67"
MONO = "DejaVu Sans Mono"


def label(
    text: str,
    size: int = 30,
    color: str = WHITE,
    weight: str = "NORMAL",
) -> Text:
    return Text(text, font=MONO, font_size=size, color=color, weight=weight)


def read_binary_pcd_xy(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header: dict[str, list[str]] = {}
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError(f"Missing DATA line in {path}")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break
        fields = header["FIELDS"]
        sizes = [int(value) for value in header["SIZE"]]
        types = header["TYPE"]
        dtype = []
        for field, size, kind in zip(fields, sizes, types):
            if size != 4 or kind != "F":
                raise RuntimeError(f"Unsupported PCD field in {path}")
            dtype.append((field, "<f4"))
        raw = np.fromfile(
            stream,
            dtype=np.dtype(dtype),
            count=int(header["POINTS"][0]),
        )
    xy = np.column_stack((raw["x"], raw["y"])).astype(np.float64)
    return xy[np.isfinite(xy).all(axis=1)]


def read_tum_xy(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    return values[:, 1:3]


def read_trusted_xy(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return np.asarray(
        [[float(row["x"]), float(row["y"])] for row in rows],
        dtype=np.float64,
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def make_map_raster(
    map_xy: np.ndarray,
    trajectories: list[np.ndarray],
    path: Path,
) -> tuple[float, float, float, float]:
    combined = np.vstack([map_xy, *trajectories])
    low = np.nanmin(combined, axis=0) - 3.0
    high = np.nanmax(combined, axis=0) + 3.0
    width, height = 1400, 760
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[..., :3] = np.array([8, 17, 31], dtype=np.uint8)
    image[..., 3] = 255
    normalized = (map_xy - low) / np.maximum(high - low, 1e-9)
    x = np.clip(np.round(normalized[:, 0] * (width - 1)).astype(int), 0, width - 1)
    y = np.clip(
        np.round((1.0 - normalized[:, 1]) * (height - 1)).astype(int),
        0,
        height - 1,
    )
    image[y, x, :3] = np.array([101, 123, 140], dtype=np.uint8)
    image[y, x, 3] = 150
    Image.fromarray(image, "RGBA").save(path)
    return float(low[0]), float(high[0]), float(low[1]), float(high[1])


def path_mobject(points: np.ndarray, mapper, color: str, width: float) -> VMobject:
    path = VMobject(
        stroke_color=color,
        stroke_width=width,
        joint_type=LineJointType.ROUND,
        cap_style=CapStyleType.ROUND,
    )
    path.set_points_as_corners([mapper(point) for point in points])
    return path


class IndoorCrossDeviceExperiment(Scene):
    def construct(self) -> None:
        ASSET_DIR.mkdir(parents=True, exist_ok=True)

        map_xy = read_binary_pcd_xy(ROOT / "map/full/scans.pcd")
        map_track = read_tum_xy(
            Path(
                "${UPDOWN_SC_ROOT}/icra2027_runtime/"
                "indoor_four_bag_selection_20260718/"
                "indoor_handle1_ros2/session/optimized_poses_tum.txt"
            )
        )
        handle_track = read_trusted_xy(
            ROOT / "localization/handle2/trusted_pose.csv"
        )
        vehicle_track = read_trusted_xy(
            ROOT / "localization/vehicle1/trusted_pose.csv"
        )
        handle_results = read_csv_rows(
            ROOT / "results/handle2/updown_per_query.csv"
        )
        vehicle_results = read_csv_rows(
            ROOT / "results/vehicle1/updown_per_query.csv"
        )
        handle_sc_results = [
            row
            for row in read_csv_rows(
                ROOT / "table3_2m/handle2/gravity/per_query.csv"
            )
            if row["algorithm"] == "SC + G"
        ]
        vehicle_sc_results = [
            row
            for row in read_csv_rows(
                ROOT / "table3_2m/vehicle1/gravity/per_query.csv"
            )
            if row["algorithm"] == "SC + G"
        ]
        map_asset = ASSET_DIR / "map_raster.png"
        extent = make_map_raster(
            map_xy,
            [map_track, handle_track, vehicle_track],
            map_asset,
        )

        title = label("One indoor map · two query devices", 46, WHITE, "BOLD")
        subtitle = label(
            "same MID-360 model · hand-carried map → independent query platforms",
            25,
            MUTED,
        ).next_to(title, DOWN, buff=0.28)
        self.add_subcaption(
            "One indoor map is reused by two independently recorded query platforms.",
            duration=3.0,
        )
        self.play(FadeIn(title, shift=UP * 0.2), FadeIn(subtitle), run_time=1.2)
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=0.5)
        self.wait(0.3)

        # Four synchronized panels compare methods as well as query setups.
        # Every displayed line joins a measured query position to its actual
        # Top-1 map candidate.
        base_panel = Rectangle(
            width=7.35,
            height=2.90,
            fill_color=PANEL,
            fill_opacity=1,
            stroke_color="#344860",
        )
        handle_sc_panel = base_panel.copy().move_to(
            np.array([-3.85, 2.05, 0.0])
        )
        handle_ours_panel = base_panel.copy().move_to(
            np.array([3.85, 2.05, 0.0])
        )
        vehicle_sc_panel = base_panel.copy().move_to(
            np.array([-3.85, -2.00, 0.0])
        )
        vehicle_ours_panel = base_panel.copy().move_to(
            np.array([3.85, -2.00, 0.0])
        )
        panels = (
            handle_sc_panel,
            handle_ours_panel,
            vehicle_sc_panel,
            vehicle_ours_panel,
        )
        maps = tuple(
            ImageMobject(str(map_asset)).set_width(5.25).move_to(panel)
            for panel in panels
        )
        x0, x1, y0, y1 = extent

        def panel_mapper(image: ImageMobject, point: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    image.get_left()[0]
                    + (point[0] - x0) / (x1 - x0) * image.width,
                    image.get_bottom()[1]
                    + (point[1] - y0) / (y1 - y0) * image.height,
                    0.0,
                ]
            )

        map_tracks = tuple(
            path_mobject(
                map_track,
                lambda point, image=image: panel_mapper(image, point),
                MAP_TRACK,
                2.4,
            )
            for image in maps
        )
        titles = (
            label("Handle2 · Scan Context + G", 20, MUTED, "BOLD").next_to(
                handle_sc_panel,
                UP,
                buff=0.08,
            ),
            label("Handle2 · UpDown-SC", 20, HANDLE, "BOLD").next_to(
                handle_ours_panel,
                UP,
                buff=0.08,
            ),
            label("Vehicle1 · Scan Context + G", 20, MUTED, "BOLD").next_to(
                vehicle_sc_panel,
                UP,
                buff=0.08,
            ),
            label("Vehicle1 · UpDown-SC", 20, VEHICLE, "BOLD").next_to(
                vehicle_ours_panel,
                UP,
                buff=0.08,
            ),
        )
        retrieval_progress = ValueTracker(0.0)

        def result_index(rows: list[dict[str, str]]) -> int:
            return min(
                int(retrieval_progress.get_value() * len(rows)),
                len(rows) - 1,
            )

        def dynamic_match(
            image: ImageMobject,
            rows: list[dict[str, str]],
        ) -> VGroup:
            row = rows[result_index(rows)]
            query = np.array([float(row["truth_x"]), float(row["truth_y"])])
            top1 = np.array([float(row["top1_x"]), float(row["top1_y"])])
            success = row["recall_at_1"].lower() == "true"
            color = GOOD if success else BAD
            return VGroup(
                Line(
                    panel_mapper(image, query),
                    panel_mapper(image, top1),
                    color=color,
                    stroke_width=2.4,
                ),
                Dot(panel_mapper(image, query), radius=0.050, color=WHITE),
                Dot(panel_mapper(image, top1), radius=0.060, color=color),
            )

        def running_result(
            rows: list[dict[str, str]],
            color: str,
            panel: Rectangle,
        ) -> Text:
            index = result_index(rows)
            correct = sum(
                row["recall_at_1"].lower() == "true"
                for row in rows[: index + 1]
            )
            current_ok = rows[index]["recall_at_1"].lower() == "true"
            status = "TOP-1 OK" if current_ok else "TOP-1 WRONG"
            return label(
                (
                    f"query {index + 1:02d}/{len(rows):02d} · {status} · "
                    f"running R@1 {100.0 * correct / (index + 1):4.1f}%"
                ),
                13,
                color,
                "BOLD",
            ).next_to(panel, DOWN, buff=0.05)

        panel_rows = (
            handle_sc_results,
            handle_results,
            vehicle_sc_results,
            vehicle_results,
        )
        panel_colors = (MUTED, HANDLE, MUTED, VEHICLE)
        dynamic_matches = tuple(
            always_redraw(
                lambda image=image, rows=rows: dynamic_match(image, rows)
            )
            for image, rows in zip(maps, panel_rows)
        )
        result_texts = tuple(
            always_redraw(
                lambda rows=rows, color=color, panel=panel: running_result(
                    rows,
                    color,
                    panel,
                )
            )
            for rows, color, panel in zip(
                panel_rows,
                panel_colors,
                panels,
            )
        )
        def play_retrieval_comparison() -> None:
            self.add_subcaption(
                "Rows share the same query setup; columns compare Scan Context and UpDown-SC on every measured Top-1 result.",
                duration=12.5,
            )
            self.play(
                *[FadeIn(item) for item in (*panels, *maps, *titles)],
                run_time=0.8,
            )
            self.play(
                *[Create(track) for track in map_tracks],
                run_time=1.2,
                rate_func=linear,
            )
            self.add(*dynamic_matches, *result_texts)
            self.play(
                retrieval_progress.animate.set_value(0.999999),
                run_time=11.0,
                rate_func=linear,
            )
            # The integrated video ends on this measured result.  Keep the
            # completed four-panel comparison visible instead of replacing it
            # with a last-dataset-specific closing card.
            self.wait(2.0)

        image = ImageMobject(str(map_asset)).set_width(14.5).shift(DOWN * 0.15)
        left, right = image.get_left()[0], image.get_right()[0]
        bottom, top = image.get_bottom()[1], image.get_top()[1]

        def mapper(point: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    left + (point[0] - x0) / (x1 - x0) * (right - left),
                    bottom + (point[1] - y0) / (y1 - y0) * (top - bottom),
                    0.0,
                ]
            )

        map_line = path_mobject(map_track, mapper, MAP_TRACK, 4.0)
        map_label = label("Handle1 map · 477 descriptors", 21, MAP_TRACK, "BOLD")
        map_label.to_edge(UP, buff=0.38).to_edge(LEFT, buff=0.58)
        self.add_subcaption(
            "The blue mapping trajectory establishes the shared prior map.",
            duration=3.0,
        )
        self.play(FadeIn(image), FadeIn(map_label), run_time=0.7)
        self.play(Create(map_line), run_time=1.8, rate_func=linear)

        def downsample(track: np.ndarray, count: int = 360) -> np.ndarray:
            indices = np.linspace(
                0,
                len(track) - 1,
                min(count, len(track)),
                dtype=int,
            )
            return track[indices]

        def replay(
            track: np.ndarray,
            color: str,
            name: str,
            caption: str,
        ) -> VMobject:
            samples = downsample(track)
            progress = ValueTracker(1.0)

            def sample_index() -> int:
                return int(
                    np.clip(
                        round(progress.get_value()),
                        1,
                        len(samples) - 1,
                    )
                )

            trail = always_redraw(
                lambda: path_mobject(
                    samples[: sample_index() + 1],
                    mapper,
                    color,
                    5.2,
                )
            )
            moving_dot = always_redraw(
                lambda: Dot(
                    mapper(samples[sample_index()]),
                    radius=0.105,
                    color=WHITE,
                    stroke_color=color,
                    stroke_width=5,
                )
            )
            progress_text = always_redraw(
                lambda: label(
                    (
                        f"{name} · "
                        f"{1 + int(sample_index() / (len(samples) - 1) * (len(track) - 1)):,}"
                        f" / {len(track):,} trusted poses"
                    ),
                    18,
                    color,
                    "BOLD",
                ).to_edge(UP, buff=0.38).to_edge(RIGHT, buff=0.58)
            )
            self.add_subcaption(caption, duration=5.0)
            self.add(trail, moving_dot, progress_text)
            self.play(
                progress.animate.set_value(len(samples) - 1),
                run_time=5.0,
                rate_func=linear,
            )
            final_path = path_mobject(track, mapper, color, 4.5)
            self.remove(trail, moving_dot, progress_text)
            self.add(final_path)
            return final_path

        handle_line = replay(
            handle_track,
            HANDLE,
            "Handle2 live replay",
            "The hand-carried query is replayed continuously on the real prior map.",
        )
        vehicle_line = replay(
            vehicle_track,
            VEHICLE,
            "Vehicle1 live replay",
            "The vehicle-mounted query is then replayed continuously on the same prior map.",
        )
        completed = label(
            "two complete bags · continuous trusted localization",
            22,
            WHITE,
            "BOLD",
        ).to_edge(DOWN, buff=0.44)
        self.play(FadeIn(completed, shift=UP * 0.12), run_time=0.45)
        self.wait(0.8)
        self.play(
            *[
                FadeOut(item)
                for item in (
                    image,
                    map_line,
                    handle_line,
                    vehicle_line,
                    map_label,
                    completed,
                )
            ],
            run_time=0.6,
        )
        self.wait(0.3)

        play_retrieval_comparison()
