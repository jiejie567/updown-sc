#!/usr/bin/env python3
"""Export odometry-registered multi-scan submaps for BTC-style baselines.

The node consumes FAST-LIO's deskewed body-frame cloud and matching odometry.
For every group of N scans it transforms the current scan and its N-1 causal
predecessors into the current (last) scan's body frame, applies the common
radial crop and an optional voxel filter, then writes compact XYZI float32
``.bin`` files plus pose/metadata sidecars.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def message_stamp(message: PointCloud2 | Odometry) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def quaternion_to_rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("invalid zero/non-finite odometry quaternion")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class Pose:
    stamp: float
    translation: np.ndarray
    rotation: np.ndarray
    quaternion_xyzw: tuple[float, float, float, float]
    frame_id: str
    child_frame_id: str


@dataclass(frozen=True)
class Frame:
    stamp: float
    points: np.ndarray
    pose: Pose
    cloud_frame_id: str


def pose_from_odometry(message: Odometry) -> Pose:
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    return Pose(
        stamp=message_stamp(message),
        translation=np.asarray([position.x, position.y, position.z], dtype=np.float64),
        rotation=quaternion_to_rotation(
            orientation.x, orientation.y, orientation.z, orientation.w
        ),
        quaternion_xyzw=(
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        ),
        frame_id=message.header.frame_id,
        child_frame_id=message.child_frame_id,
    )


def cloud_to_xyzi(message: PointCloud2) -> np.ndarray:
    available = {field.name for field in message.fields}
    required = {"x", "y", "z"}
    if not required.issubset(available):
        raise ValueError(f"PointCloud2 is missing fields: {sorted(required - available)}")
    names = ("x", "y", "z", "intensity") if "intensity" in available else ("x", "y", "z")
    try:
        values = np.asarray(
            point_cloud2.read_points_numpy(message, field_names=names, skip_nans=True),
            dtype=np.float32,
        )
    except (AssertionError, ValueError):
        values = np.asarray(
            list(point_cloud2.read_points(message, field_names=names, skip_nans=True)),
            dtype=np.float32,
        )
    values = values.reshape((-1, len(names)))
    if len(names) == 3:
        values = np.column_stack((values, np.zeros(len(values), dtype=np.float32)))
    finite = np.isfinite(values[:, :3]).all(axis=1)
    return np.ascontiguousarray(values[finite], dtype=np.float32)


def transform_to_reference(points: np.ndarray, pose: Pose, reference: Pose) -> np.ndarray:
    """Transform XYZI points from ``pose`` body coordinates to ``reference`` body coordinates."""
    relative_rotation = reference.rotation.T @ pose.rotation
    relative_translation = reference.rotation.T @ (pose.translation - reference.translation)
    transformed = np.empty_like(points, dtype=np.float32)
    transformed[:, :3] = (
        points[:, :3].astype(np.float64) @ relative_rotation.T + relative_translation
    ).astype(np.float32)
    transformed[:, 3] = points[:, 3]
    return transformed


def radial_crop(points: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    radii_squared = np.square(points[:, 0], dtype=np.float64) + np.square(
        points[:, 1], dtype=np.float64
    )
    mask = radii_squared >= minimum * minimum
    if maximum > 0.0:
        mask &= radii_squared <= maximum * maximum
    return np.ascontiguousarray(points[mask], dtype=np.float32)


def voxel_centroids(points: np.ndarray, leaf_size: float) -> np.ndarray:
    if leaf_size <= 0.0 or len(points) == 0:
        return np.ascontiguousarray(points, dtype=np.float32)
    voxel = np.floor(points[:, :3].astype(np.float64) / leaf_size).astype(np.int64)
    order = np.lexsort((voxel[:, 2], voxel[:, 1], voxel[:, 0]))
    sorted_voxel = voxel[order]
    sorted_points = points[order].astype(np.float64)
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_voxel, axis=0), axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(points)])
    sums = np.add.reduceat(sorted_points, starts, axis=0)
    return np.ascontiguousarray((sums / counts[:, None]).astype(np.float32))


def combine_frames(
    frames: Iterable[Frame], minimum_radius: float, maximum_radius: float, voxel_size: float
) -> tuple[np.ndarray, int]:
    frame_list = list(frames)
    if not frame_list:
        return np.empty((0, 4), dtype=np.float32), 0
    # A complete window is causal: the newest scan is the experiment center
    # and the preceding N-1 scans provide accumulated geometry.
    reference = frame_list[-1].pose
    transformed = [transform_to_reference(frame.points, frame.pose, reference) for frame in frame_list]
    raw_points = sum(len(points) for points in transformed)
    combined = np.concatenate(transformed, axis=0)
    combined = radial_crop(combined, minimum_radius, maximum_radius)
    combined = voxel_centroids(combined, voxel_size)
    return combined, raw_points


class BtcSubmapExporter(Node):
    def __init__(self, arguments: argparse.Namespace) -> None:
        super().__init__("btc_submap_exporter")
        self.arguments = arguments
        self.output_dir = arguments.output_dir.expanduser().resolve()
        self.submap_dir = self.output_dir / "submaps"
        self._prepare_output()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=200,
            # The registered cloud publisher is reliable.  Export must also be
            # reliable: voxelizing every tenth frame briefly occupies this
            # callback, and BEST_EFFORT silently dropped clouds during replay.
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.odometry: Deque[Pose] = deque(maxlen=400)
        self.pending_clouds: Deque[PointCloud2] = deque(maxlen=20)
        self.frames: Deque[Frame] = deque()
        self.last_frame_stamp: float | None = None
        self.submap_count = 0
        self.received_odometry = 0
        self.received_clouds = 0
        self.synchronized_clouds = 0
        self.dropped_unsynchronized = 0
        self.dropped_gap_frames = 0
        self.closed = False

        self.metadata_stream = (self.output_dir / "metadata.csv").open(
            "w", encoding="utf-8", newline="", buffering=1
        )
        self.metadata_writer = csv.writer(self.metadata_stream)
        self.metadata_writer.writerow(
            [
                "submap_id", "reference_stamp", "start_stamp", "end_stamp", "scans",
                "raw_points", "output_points", "file", "world_frame", "body_frame",
                "x", "y", "z", "qx", "qy", "qz", "qw",
            ]
        )
        self.pose_stream = (self.output_dir / "poses_tum.txt").open(
            "w", encoding="utf-8", buffering=1
        )
        self._write_manifest("running")

        self.odometry_subscription = self.create_subscription(
            Odometry, arguments.odometry_topic, self._on_odometry, qos
        )
        self.cloud_subscription = self.create_subscription(
            PointCloud2, arguments.cloud_topic, self._on_cloud, qos
        )
        self.get_logger().info(
            f"BTC exporter ready: clouds={arguments.cloud_topic} "
            f"odometry={arguments.odometry_topic} scans={arguments.scans_per_submap} "
            f"stride={arguments.submap_stride_scans} "
            f"crop=[{arguments.minimum_radius:.2f}, {arguments.maximum_radius:.2f}] "
            f"voxel={arguments.voxel_size:.3f} output={self.output_dir}"
        )

    def _prepare_output(self) -> None:
        generated = [
            self.output_dir / "submaps",
            self.output_dir / "metadata.csv",
            self.output_dir / "poses_tum.txt",
            self.output_dir / "manifest.json",
        ]
        existing = [path for path in generated if path.exists()]
        if existing and not self.arguments.overwrite:
            raise RuntimeError(
                f"BTC output already exists under {self.output_dir}; use --overwrite"
            )
        if self.arguments.overwrite:
            for path in existing:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        self.submap_dir.mkdir(parents=True, exist_ok=True)

    def _write_manifest(self, status: str) -> None:
        manifest = {
            "format": "btc_registered_submaps_v2",
            "status": status,
            "cloud_topic": self.arguments.cloud_topic,
            "odometry_topic": self.arguments.odometry_topic,
            "scans_per_submap": self.arguments.scans_per_submap,
            "submap_stride_scans": self.arguments.submap_stride_scans,
            "causal": True,
            "reference_scan": "current_last",
            "reference_orientation": "native_body",
            "maximum_sync_error_s": self.arguments.maximum_sync_error,
            "maximum_scan_gap_s": self.arguments.maximum_scan_gap,
            "minimum_radius_m": self.arguments.minimum_radius,
            "maximum_radius_m": self.arguments.maximum_radius,
            "voxel_size_m": self.arguments.voxel_size,
            "submaps": self.submap_count,
            "received_odometry": self.received_odometry,
            "received_clouds": self.received_clouds,
            "synchronized_clouds": self.synchronized_clouds,
            "dropped_unsynchronized": self.dropped_unsynchronized,
            "dropped_gap_frames": self.dropped_gap_frames,
        }
        temporary = self.output_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.output_dir / "manifest.json")

    def _nearest_pose(self, stamp: float) -> Pose | None:
        if not self.odometry:
            return None
        pose = min(self.odometry, key=lambda candidate: abs(candidate.stamp - stamp))
        if abs(pose.stamp - stamp) > self.arguments.maximum_sync_error:
            return None
        return pose

    def _on_odometry(self, message: Odometry) -> None:
        self.received_odometry += 1
        try:
            self.odometry.append(pose_from_odometry(message))
        except ValueError as error:
            self.get_logger().warning(f"Dropping invalid odometry: {error}")
            return
        self._drain_pending()

    def _on_cloud(self, message: PointCloud2) -> None:
        self.received_clouds += 1
        pose = self._nearest_pose(message_stamp(message))
        if pose is None:
            if self.received_clouds == 10:
                newest = self.odometry[-1].stamp if self.odometry else float("nan")
                self.get_logger().warning(
                    f"No synchronized pose after 10 clouds: cloud_stamp={message_stamp(message):.6f} "
                    f"odometry_received={self.received_odometry} newest_pose={newest:.6f}"
                )
            if len(self.pending_clouds) == self.pending_clouds.maxlen:
                self.pending_clouds.popleft()
                self.dropped_unsynchronized += 1
            self.pending_clouds.append(message)
            return
        self._accept_cloud(message, pose)

    def _drain_pending(self) -> None:
        if not self.pending_clouds or not self.odometry:
            return
        newest_pose_stamp = self.odometry[-1].stamp
        retained: Deque[PointCloud2] = deque(maxlen=self.pending_clouds.maxlen)
        while self.pending_clouds:
            cloud = self.pending_clouds.popleft()
            stamp = message_stamp(cloud)
            pose = self._nearest_pose(stamp)
            if pose is not None:
                self._accept_cloud(cloud, pose)
            elif newest_pose_stamp > stamp + self.arguments.maximum_sync_error:
                self.dropped_unsynchronized += 1
            else:
                retained.append(cloud)
        self.pending_clouds = retained

    def _accept_cloud(self, message: PointCloud2, pose: Pose) -> None:
        if self.arguments.maximum_submaps > 0 and self.submap_count >= self.arguments.maximum_submaps:
            return
        stamp = message_stamp(message)
        try:
            points = cloud_to_xyzi(message)
        except ValueError as error:
            self.get_logger().warning(f"Dropping cloud at {stamp:.6f}: {error}")
            return
        if len(points) == 0:
            self.get_logger().warning(f"Dropping empty cloud at {stamp:.6f}")
            return

        if self.last_frame_stamp is not None:
            gap = stamp - self.last_frame_stamp
            # LiDAR stamps are floating point seconds.  Allow a tiny numerical
            # tolerance so a nominal 0.300 s gap is not treated as larger than
            # the configured 0.300 s boundary.
            if gap <= 0.0 or gap > self.arguments.maximum_scan_gap + 1.0e-6:
                self.dropped_gap_frames += len(self.frames)
                self.frames.clear()
                self.get_logger().warning(
                    f"Resetting partial BTC submap after scan timestamp gap {gap:.3f} s"
                )
        self.last_frame_stamp = stamp
        self.frames.append(Frame(stamp, points, pose, message.header.frame_id))
        self.synchronized_clouds += 1
        if len(self.frames) >= self.arguments.scans_per_submap:
            self._emit_submap(list(self.frames)[: self.arguments.scans_per_submap])
            for _ in range(self.arguments.submap_stride_scans):
                if self.frames:
                    self.frames.popleft()

    def _emit_submap(self, frames: list[Frame]) -> None:
        points, raw_points = combine_frames(
            frames,
            self.arguments.minimum_radius,
            self.arguments.maximum_radius,
            self.arguments.voxel_size,
        )
        if len(points) < self.arguments.minimum_output_points:
            self.get_logger().warning(
                f"Skipping BTC submap at {frames[0].stamp:.6f}: "
                f"only {len(points)} output points"
            )
            return
        submap_id = self.submap_count
        relative_path = Path("submaps") / f"{submap_id:06d}.bin"
        destination = self.output_dir / relative_path
        temporary = destination.with_suffix(".bin.tmp")
        points.astype("<f4", copy=False).tofile(temporary)
        os.replace(temporary, destination)

        reference = frames[-1].pose
        qx, qy, qz, qw = reference.quaternion_xyzw
        x, y, z = reference.translation
        self.metadata_writer.writerow(
            [
                submap_id, f"{reference.stamp:.9f}", f"{frames[0].stamp:.9f}",
                f"{frames[-1].stamp:.9f}", len(frames), raw_points, len(points),
                str(relative_path), reference.frame_id, reference.child_frame_id,
                f"{x:.9f}", f"{y:.9f}", f"{z:.9f}", f"{qx:.9f}",
                f"{qy:.9f}", f"{qz:.9f}", f"{qw:.9f}",
            ]
        )
        self.pose_stream.write(
            f"{reference.stamp:.9f} {x:.9f} {y:.9f} {z:.9f} "
            f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
        )
        self.submap_count += 1
        self._write_manifest("running")
        if self.submap_count == 1 or self.submap_count % 25 == 0:
            self.get_logger().info(
                f"Exported BTC submap {submap_id}: scans={len(frames)} "
                f"raw={raw_points} output={len(points)}"
            )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.dropped_unsynchronized += len(self.pending_clouds)
        self.pending_clouds.clear()
        self.metadata_stream.close()
        self.pose_stream.close()
        self._write_manifest("complete")
        summary = (
            f"BTC export complete: submaps={self.submap_count} "
            f"odometry={self.received_odometry} synchronized={self.synchronized_clouds} "
            f"dropped_sync={self.dropped_unsynchronized} output={self.output_dir}"
        )
        # Launch shutdown may invalidate the ROS context before the node's
        # finally block runs.  Avoid a misleading rosout error in that case.
        if rclpy.ok():
            self.get_logger().info(summary)
        else:
            print(summary, flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cloud-topic", default="/cloud_registered_body")
    parser.add_argument("--odometry-topic", default="/Odometry")
    parser.add_argument("--scans-per-submap", type=int, default=10)
    parser.add_argument("--submap-stride-scans", type=int, default=10)
    parser.add_argument("--maximum-sync-error", type=float, default=0.03)
    parser.add_argument("--maximum-scan-gap", type=float, default=0.3)
    parser.add_argument("--minimum-radius", type=float, default=0.3)
    parser.add_argument("--maximum-radius", type=float, default=30.0)
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--minimum-output-points", type=int, default=1000)
    parser.add_argument("--maximum-submaps", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    if arguments.scans_per_submap <= 0:
        parser.error("--scans-per-submap must be positive")
    if not 1 <= arguments.submap_stride_scans <= arguments.scans_per_submap:
        parser.error("--submap-stride-scans must be in [1, scans-per-submap]")
    if arguments.maximum_sync_error <= 0.0 or arguments.maximum_scan_gap <= 0.0:
        parser.error("sync error and scan gap limits must be positive")
    if arguments.minimum_radius < 0.0:
        parser.error("--minimum-radius must be nonnegative")
    if 0.0 < arguments.maximum_radius <= arguments.minimum_radius:
        parser.error("--maximum-radius must exceed --minimum-radius, or be <=0 to disable")
    if arguments.voxel_size < 0.0:
        parser.error("--voxel-size must be nonnegative")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    rclpy.init()
    node: BtcSubmapExporter | None = None
    try:
        node = BtcSubmapExporter(arguments)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
