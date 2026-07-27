#!/usr/bin/env python3
"""Manim video of the real public factory-hall cross-sequence experiment."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from manim import (
    AnimationGroup,
    BLACK,
    BLUE,
    CapStyleType,
    Create,
    DashedVMobject,
    DOWN,
    Dot,
    FadeIn,
    FadeOut,
    GREEN,
    ImageMobject,
    LEFT,
    LineJointType,
    Line,
    ORANGE,
    RED,
    RIGHT,
    Rectangle,
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
        "CONSTRUCTION_HALL_EXPERIMENT",
        "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/rtk_slam_construction_hall_2m",
    )
)
ADAPTIVE_ROOT = Path(
    os.environ.get(
        "CONSTRUCTION_HALL_ADAPTIVE_EXPERIMENT",
        "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/"
        "adaptive_split_conditional_20260720/ch",
    )
)
GRAVITY_ROOT = Path(
    os.environ.get(
        "CONSTRUCTION_HALL_GRAVITY_EXPERIMENT",
        "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/"
        "gravity_transfer_2m_20260718/ch",
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
MAP_COLOR = "#6F8798"
MAP_TRACK = "#3C8DDB"
QUERY_TRACK = "#F39A45"
OURS = "#438BDE"
SC_COLOR = "#E28A2D"
GOOD = "#37C68A"
BAD = "#DF5656"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


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
        dtype = np.dtype([(field, "<f4") for field in fields])
        points = np.fromfile(stream, dtype=dtype, count=int(header["POINTS"][0]))
    xy = np.column_stack((points["x"], points["y"])).astype(np.float64)
    return xy[np.isfinite(xy).all(axis=1)]


def read_cloud_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if len(values) % 4:
        raise RuntimeError(f"Invalid XYZI binary: {path}")
    return values.reshape(-1, 4)[:, :3].astype(np.float64)


def shortest_rotation_to_up(up: np.ndarray) -> np.ndarray:
    source = up / np.linalg.norm(up)
    target = np.array([0.0, 0.0, 1.0])
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(source, target))
    if sine < 1e-12:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    axis = cross / sine
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    angle = math.atan2(sine, cosine)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def descriptor(
    points: np.ndarray,
    dual: bool,
    split_height: float = 2.5,
) -> tuple[np.ndarray, ...]:
    rings, sectors, max_radius = 16, 60, 30.0
    radius = np.hypot(points[:, 0], points[:, 1])
    valid = np.isfinite(points).all(axis=1) & (radius > 1e-6) & (radius <= max_radius)
    points = points[valid]
    radius = radius[valid]
    theta = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    ring = np.clip(np.ceil(radius / max_radius * rings).astype(int) - 1, 0, rings - 1)
    sector = np.clip(np.ceil(theta / (2.0 * np.pi) * sectors).astype(int) - 1, 0, sectors - 1)
    if not dual:
        values = np.full((rings, sectors), np.nan)
        for r, s, z in zip(ring, sector, points[:, 2]):
            values[r, s] = z if not np.isfinite(values[r, s]) else max(values[r, s], z)
        return (values,)
    lower = np.full((rings, sectors), np.nan)
    upper = np.full((rings, sectors), np.nan)
    for r, s, z in zip(ring, sector, points[:, 2]):
        if z <= split_height:
            lower[r, s] = z if not np.isfinite(lower[r, s]) else max(lower[r, s], z)
        else:
            upper[r, s] = z if not np.isfinite(upper[r, s]) else min(upper[r, s], z)
    return lower, upper


def value_color(value: float, low: float = -1.5, high: float = 6.0) -> tuple[int, int, int]:
    if not np.isfinite(value):
        return 17, 32, 53
    t = float(np.clip((value - low) / (high - low), 0.0, 1.0))
    stops = np.asarray(
        [
            [40, 87, 140],
            [55, 154, 176],
            [239, 186, 82],
            [223, 86, 76],
        ],
        dtype=float,
    )
    position = t * (len(stops) - 1)
    index = min(int(position), len(stops) - 2)
    fraction = position - index
    return tuple(np.round(stops[index] * (1 - fraction) + stops[index + 1] * fraction).astype(int))


def render_polar(values: np.ndarray, path: Path, size: int = 520) -> None:
    image = Image.new("RGB", (size, size), (8, 17, 31))
    draw = ImageDraw.Draw(image)
    center = size / 2
    inner, outer = 28.0, size * 0.46
    rings, sectors = values.shape
    for ring in range(rings):
        r0 = inner + (outer - inner) * ring / rings + 0.6
        r1 = inner + (outer - inner) * (ring + 1) / rings - 0.6
        for sector in range(sectors):
            a0 = 2.0 * np.pi * sector / sectors - np.pi / 2 + np.deg2rad(0.10)
            a1 = 2.0 * np.pi * (sector + 1) / sectors - np.pi / 2 - np.deg2rad(0.10)
            angles = np.linspace(a0, a1, 5)
            polygon = [
                (center + r1 * np.cos(angle), center + r1 * np.sin(angle))
                for angle in angles
            ]
            polygon.extend(
                [
                    (center + r0 * np.cos(angle), center + r0 * np.sin(angle))
                    for angle in angles[::-1]
                ]
            )
            draw.polygon(polygon, fill=value_color(values[ring, sector]))
    image.save(path)


def make_map_raster(map_xy: np.ndarray, all_xy: np.ndarray, path: Path) -> tuple[float, ...]:
    combined = np.vstack((map_xy, all_xy))
    low = np.nanmin(combined, axis=0) - 4.0
    high = np.nanmax(combined, axis=0) + 4.0
    width, height = 1200, 900
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[..., :3] = np.array([8, 17, 31], dtype=np.uint8)
    image[..., 3] = 255
    normalized = (map_xy - low) / (high - low)
    pixel_x = np.clip(np.round(normalized[:, 0] * (width - 1)).astype(int), 0, width - 1)
    pixel_y = np.clip(np.round((1.0 - normalized[:, 1]) * (height - 1)).astype(int), 0, height - 1)
    image[pixel_y, pixel_x, :3] = np.array([97, 121, 137], dtype=np.uint8)
    image[pixel_y, pixel_x, 3] = 145
    Image.fromarray(image, "RGBA").save(path)
    return low[0], high[0], low[1], high[1]


def path_mobject(points: np.ndarray, mapper, color: str, width: float = 3.0) -> VMobject:
    output = VMobject(
        stroke_color=color,
        stroke_width=width,
        joint_type=LineJointType.ROUND,
        cap_style=CapStyleType.ROUND,
    )
    output.set_points_as_corners([mapper(point) for point in points])
    return output


def label(text: str, size: int = 32, color: str = WHITE, weight: str = "NORMAL") -> Text:
    return Text(text, font="DejaVu Sans", font_size=size, color=color, weight=weight)


class ConstructionHallExperiment(Scene):
    def construct(self) -> None:
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        map_xy = read_binary_pcd_xy(ROOT / "seq1/scans.pcd")
        map_track = np.loadtxt(ROOT / "seq1/session/optimized_poses_tum.txt")[:, 1:3]
        query_track = np.loadtxt(ROOT / "derived/seq2_truth_in_seq1_map_tum.txt")[:, 1:3]
        metadata = read_csv(ROOT / "derived/queries_seq2/metadata.csv")
        sc_rows = {
            int(row["query_id"]): row
            for row in read_csv(GRAVITY_ROOT / "results/per_query.csv")
            if row["algorithm"] == "SC + G"
        }
        legacy_ours_rows = {
            int(row["query_id"]): row
            for row in read_csv(ROOT / "results/updown_per_query.csv")
        }
        summary = {
            row["algorithm"]: row
            for row in read_csv(GRAVITY_ROOT / "results/summary.csv")
        }
        iris_rows = read_csv(GRAVITY_ROOT / "results/lidar_iris_per_query.csv")
        summary["LiDAR Iris + G"] = {
            "algorithm": "LiDAR Iris + G",
            "protocol": "gravity_canonicalized_single_scan",
            "recall_at_1": str(
                np.mean([truthy(row["recall_at_1"]) for row in iris_rows])
            ),
        }
        summary["RING++ + G"] = read_csv(
            GRAVITY_ROOT / "ringpp/results/ringpp_summary.csv"
        )[0]
        adaptive_per_query = ADAPTIVE_ROOT / "per_query.csv"
        adaptive_summary = ADAPTIVE_ROOT / "summary.csv"
        if adaptive_per_query.exists() and adaptive_summary.exists():
            ours_rows = {
                int(row["query_id"]): row
                for row in read_csv(adaptive_per_query)
            }
            summary["UpDown-SC"] = read_csv(adaptive_summary)[0]
        else:
            ours_rows = legacy_ours_rows
        gravity = {
            int(row["query_id"]): np.asarray(
                [float(row["up_x"]), float(row["up_y"]), float(row["up_z"])]
            )
            for row in read_csv(ROOT / "derived/queries_seq2/gravity.csv")
        }

        map_asset = ASSET_DIR / "map_raster.png"
        extent = make_map_raster(map_xy, np.vstack((map_track, query_track)), map_asset)

        # Opening.
        title = label("Public factory-hall validation", 58, WHITE, "BOLD")
        subtitle = label(
            "RTK-SLAM · repeated traversals · cross-sequence place recognition",
            30,
            MUTED,
        )
        subtitle.next_to(title, DOWN, buff=0.28)
        self.play(FadeIn(title, shift=UP * 0.25), FadeIn(subtitle), run_time=1.2)
        self.wait(1.0)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=0.6)

        # Map panel and exact keyframe policy.
        map_image = ImageMobject(str(map_asset)).set_width(10.5).shift(LEFT * 2.2)
        x0, x1, y0, y1 = extent
        left, right = map_image.get_left()[0], map_image.get_right()[0]
        bottom, top = map_image.get_bottom()[1], map_image.get_top()[1]

        def mapper(point: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    left + (point[0] - x0) / (x1 - x0) * (right - left),
                    bottom + (point[1] - y0) / (y1 - y0) * (top - bottom),
                    0.0,
                ]
            )

        map_line = path_mobject(map_track, mapper, MAP_TRACK, 4.0)
        query_line = DashedVMobject(
            path_mobject(query_track, mapper, QUERY_TRACK, 4.0),
            num_dashes=90,
            dashed_ratio=0.65,
        )
        side_title = label("Experiment protocol", 34, WHITE, "BOLD").to_edge(RIGHT, buff=0.62).shift(UP * 2.6)
        rules = VGroup(
            label("Factory run 1  →  map/database", 22, MAP_TRACK),
            label("Factory run 2  →  query/localization", 22, QUERY_TRACK),
            label("Experiment keyframe:", 25, WHITE, "BOLD"),
            label("3-D translation ≥ 2 m", 24, WHITE),
            label("No time / yaw trigger", 24, GOOD, "BOLD"),
            label("Production FAST-LIO unchanged", 21, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.20)
        rules.next_to(side_title, DOWN, aligned_edge=LEFT, buff=0.38)
        self.play(FadeIn(map_image), FadeIn(side_title), FadeIn(rules[:2]), run_time=0.8)
        self.play(Create(map_line), run_time=2.2, rate_func=linear)
        self.play(Create(query_line), FadeIn(rules[2:]), run_time=2.0, rate_func=linear)
        self.wait(0.8)
        self.play(*[FadeOut(item) for item in (map_image, map_line, query_line, side_title, rules)], run_time=0.7)

        # Keep the same representative query used by the original dynamic
        # video. The live comparison below still uses the latest adaptive
        # result for every query.
        candidates = [
            qid
            for qid in sorted(ours_rows)
            if float(sc_rows[qid]["top1_error_m"]) > 20.0
            and float(ours_rows[qid]["top1_error_m"]) <= 5.0
        ]
        chosen = candidates[len(candidates) // 2]
        query_row = metadata[chosen]
        cloud = read_cloud_bin(ROOT / "derived/queries_seq2" / query_row["file"])
        canonical = cloud @ shortest_rotation_to_up(gravity[chosen]).T
        sc_desc = descriptor(canonical, dual=False)[0]
        # The stored adaptive split is 4.0 m above ground. With the measured
        # 1.3 m descriptor-origin height, the equivalent canonical sensor-frame
        # boundary used only for this visualization is 2.7 m.
        lower, upper = descriptor(
            canonical,
            dual=True,
            split_height=2.7,
        )
        render_polar(sc_desc, ASSET_DIR / "sc_circle.png")
        render_polar(lower, ASSET_DIR / "lower_circle.png")
        render_polar(upper, ASSET_DIR / "upper_circle.png")

        # Real query cloud (top view) as a raster.
        cloud_image = np.zeros((720, 720, 3), dtype=np.uint8)
        cloud_image[:] = np.array([8, 17, 31], dtype=np.uint8)
        xy = canonical[:, :2]
        keep = np.isfinite(xy).all(axis=1) & (np.hypot(xy[:, 0], xy[:, 1]) <= 30.0)
        xy = xy[keep]
        px = np.clip(np.round((xy[:, 0] + 30.0) / 60.0 * 719).astype(int), 0, 719)
        py = np.clip(np.round((30.0 - xy[:, 1]) / 60.0 * 719).astype(int), 0, 719)
        cloud_image[py, px] = np.array([115, 181, 238], dtype=np.uint8)
        Image.fromarray(cloud_image).save(ASSET_DIR / "query_top.png")

        section_title = label(f"One real query · keyframe {chosen}", 38, WHITE, "BOLD").to_edge(UP, buff=0.35)
        query_image = ImageMobject(str(ASSET_DIR / "query_top.png")).set_width(4.2).move_to(
            np.array([-5.25, -0.20, 0.0])
        )
        query_label = label("gravity-canonicalized query cloud", 22, MUTED).next_to(query_image, DOWN, buff=0.12)
        divider = Line(
            np.array([-2.65, -3.0, 0.0]),
            np.array([-2.65, 2.55, 0.0]),
            color="#526276",
            stroke_width=2,
        )
        descriptor_y = -0.15
        descriptor_width = 2.4
        sc_circle = ImageMobject(str(ASSET_DIR / "sc_circle.png")).set_width(
            descriptor_width
        ).move_to(np.array([-0.95, descriptor_y, 0.0]))
        lower_circle = ImageMobject(str(ASSET_DIR / "lower_circle.png")).set_width(
            descriptor_width
        ).move_to(np.array([2.15, descriptor_y, 0.0]))
        upper_circle = ImageMobject(str(ASSET_DIR / "upper_circle.png")).set_width(
            descriptor_width
        ).move_to(np.array([5.25, descriptor_y, 0.0]))
        sc_label = label("SC + gravity", 27, SC_COLOR, "BOLD").move_to(
            np.array([-0.95, 2.25, 0.0])
        )
        ours_label = label("UpDown-SC", 30, OURS, "BOLD").move_to(
            np.array([3.70, 2.25, 0.0])
        )
        lower_label = label("Lower envelope", 20, MUTED).next_to(
            lower_circle, UP, buff=0.15
        )
        upper_label = label("Upper envelope", 20, MUTED).next_to(
            upper_circle, UP, buff=0.15
        )
        self.play(FadeIn(section_title), FadeIn(query_image), FadeIn(query_label), run_time=0.8)
        self.play(
            FadeIn(divider),
            FadeIn(sc_circle),
            FadeIn(sc_label),
            FadeIn(lower_circle),
            FadeIn(upper_circle),
            FadeIn(ours_label),
            FadeIn(lower_label),
            FadeIn(upper_label),
            run_time=1.2,
        )
        self.wait(2.0)
        self.play(*[FadeOut(item) for item in list(self.mobjects)], run_time=0.7)

        # Live, data-driven retrieval comparison.
        panel_width = 7.45
        left_panel = Rectangle(width=panel_width, height=6.45, fill_color=PANEL, fill_opacity=1, stroke_color="#344860").shift(LEFT * 3.9 + DOWN * 0.25)
        right_panel = left_panel.copy().shift(RIGHT * 7.8)
        left_map = ImageMobject(str(map_asset)).set_width(6.95).move_to(left_panel)
        right_map = ImageMobject(str(map_asset)).set_width(6.95).move_to(right_panel)
        # Rebuild coordinate mappers against the two displayed map images.
        def panel_mapper(image, point):
            return np.array(
                [
                    image.get_left()[0] + (point[0] - x0) / (x1 - x0) * image.width,
                    image.get_bottom()[1] + (point[1] - y0) / (y1 - y0) * image.height,
                    0.0,
                ]
            )

        sc_title = label("SC + gravity", 31, SC_COLOR, "BOLD").next_to(left_panel, UP, buff=0.18)
        our_title = label("UpDown-SC", 31, OURS, "BOLD").next_to(right_panel, UP, buff=0.18)
        left_map_track = path_mobject(
            map_track,
            lambda point: panel_mapper(left_map, point),
            MAP_TRACK,
            3.0,
        )
        right_map_track = path_mobject(
            map_track,
            lambda point: panel_mapper(right_map, point),
            MAP_TRACK,
            3.0,
        )
        left_map_track.set_stroke(opacity=0.82)
        right_map_track.set_stroke(opacity=0.82)
        tracker = ValueTracker(0)
        sample_ids = np.linspace(0, len(query_track) - 1, 110).astype(int)

        def state(qid, rows):
            truth = query_track[qid]
            top1 = np.array([float(rows[qid]["top1_x"]), float(rows[qid]["top1_y"])])
            return truth, top1, truthy(rows[qid]["recall_at_1"])

        def dynamic_match(image, rows):
            qid = int(sample_ids[min(int(tracker.get_value()), len(sample_ids) - 1)])
            truth, top1, success = state(qid, rows)
            color = GOOD if success else BAD
            line = Line(panel_mapper(image, truth), panel_mapper(image, top1), color=color, stroke_width=2.5)
            return VGroup(
                line,
                Dot(panel_mapper(image, truth), radius=0.055, color=WHITE),
                Dot(panel_mapper(image, top1), radius=0.065, color=color),
            )

        sc_dynamic = always_redraw(lambda: dynamic_match(left_map, sc_rows))
        our_dynamic = always_redraw(lambda: dynamic_match(right_map, ours_rows))

        def running_text(rows, color):
            count = min(int(tracker.get_value()) + 1, len(sample_ids))
            ids = sample_ids[:count]
            recall = 100.0 * np.mean([truthy(rows[int(qid)]["recall_at_1"]) for qid in ids])
            return label(f"running R@1  {recall:5.1f}%", 23, color, "BOLD")

        sc_counter = always_redraw(lambda: running_text(sc_rows, SC_COLOR).next_to(left_panel, DOWN, buff=0.12))
        our_counter = always_redraw(lambda: running_text(ours_rows, OURS).next_to(right_panel, DOWN, buff=0.12))
        self.play(FadeIn(left_panel), FadeIn(right_panel), FadeIn(left_map), FadeIn(right_map), FadeIn(sc_title), FadeIn(our_title), run_time=0.8)
        self.play(
            Create(left_map_track),
            Create(right_map_track),
            run_time=1.2,
            rate_func=linear,
        )
        self.add(sc_dynamic, our_dynamic, sc_counter, our_counter)
        self.play(tracker.animate.set_value(len(sample_ids) - 1), run_time=11.0, rate_func=linear)
        self.wait(0.5)
        self.play(*[FadeOut(item) for item in list(self.mobjects)], run_time=0.7)

        # Final deterministic recall results.
        displayed_methods = [
            name
            for name, row in summary.items()
            if "single_scan" in row["protocol"]
        ]
        displayed_methods.sort(
            key=lambda name: float(summary[name]["recall_at_1"]),
            reverse=True,
        )
        heading = label("Cross-sequence Recall@1", 46, WHITE, "BOLD").to_edge(UP, buff=0.35)
        note = label(
            f"same {len(metadata)} translation-only 2 m queries · "
            "5 m correctness radius",
            24,
            MUTED,
        ).next_to(heading, DOWN, buff=0.18)
        bars = VGroup()
        max_width = 8.5
        row_spacing = 0.59
        for index, method in enumerate(displayed_methods):
            value = 100.0 * float(summary[method]["recall_at_1"])
            color = OURS if method == "UpDown-SC" else "#6F8798"
            display_name = method
            name = label(
                display_name,
                22,
                WHITE,
                "BOLD" if method == "UpDown-SC" else "NORMAL",
            )
            row_y = 2.05 - index * row_spacing
            name.move_to(np.array([-5.35, row_y, 0.0])).align_to(
                np.array([-5.35, 0.0, 0.0]),
                LEFT,
            )
            bar = Rectangle(
                width=max_width * value / 100.0,
                height=0.34,
                fill_color=color,
                fill_opacity=1,
                stroke_width=0,
            )
            bar.align_to(np.array([-2.1, 0.0, 0.0]), LEFT).move_to(
                np.array([-2.1 + bar.width / 2, row_y, 0.0])
            )
            value_label = label(f"{value:.1f}%", 22, color, "BOLD").next_to(
                bar,
                RIGHT,
                buff=0.12,
            )
            bars.add(VGroup(name, bar, value_label))
        delta = 100.0 * (
            float(summary["UpDown-SC"]["recall_at_1"])
            - float(summary["SC + G"]["recall_at_1"])
        )
        takeaway = label(
            f"+{delta:.1f} percentage points over SC + gravity",
            31,
            GOOD,
            "BOLD",
        ).to_edge(DOWN, buff=0.42)
        protocol_note = label(
            "all methods use the same gravity-canonicalized single scans",
            19,
            MUTED,
        ).next_to(takeaway, UP, buff=0.16)
        self.play(FadeIn(heading), FadeIn(note), run_time=0.7)
        for group in bars:
            self.play(FadeIn(group[0]), FadeIn(group[1], shift=RIGHT * 0.15), FadeIn(group[2]), run_time=0.33)
        self.play(
            FadeIn(protocol_note),
            FadeIn(takeaway, shift=UP * 0.15),
            run_time=0.7,
        )
        self.wait(2.0)
