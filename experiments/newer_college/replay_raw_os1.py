#!/usr/bin/env python3
"""Replay Newer College raw Ouster PCD scans and the built-in IMU to ROS 2."""

from __future__ import annotations

import argparse
import csv
import math
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from pypcd4 import PointCloud
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Header


STAMP_RE = re.compile(r"(?P<sec>\d{10})[._-](?P<nsec>\d{9})(?=\.pcd$)")
POINT_DTYPE = np.dtype(
    {
        "names": (
            "x",
            "y",
            "z",
            "intensity",
            "t",
            "reflectivity",
            "ring",
            "ambient",
            "range",
        ),
        "formats": ("<f4", "<f4", "<f4", "<f4", "<u4", "<u2", "u1", "<u2", "<u4"),
        "offsets": (0, 4, 8, 12, 16, 20, 22, 24, 28),
        "itemsize": 32,
    }
)
POINT_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="t", offset=16, datatype=PointField.UINT32, count=1),
    PointField(name="reflectivity", offset=20, datatype=PointField.UINT16, count=1),
    PointField(name="ring", offset=22, datatype=PointField.UINT8, count=1),
    PointField(name="ambient", offset=24, datatype=PointField.UINT16, count=1),
    PointField(name="range", offset=28, datatype=PointField.UINT32, count=1),
]


def ros_time(timestamp: float) -> Time:
    sec = math.floor(timestamp)
    nanosec = int(round((timestamp - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


def member_timestamp(name: str) -> float | None:
    match = STAMP_RE.search(Path(name).name)
    if match is None:
        return None
    return int(match.group("sec")) + int(match.group("nsec")) * 1e-9


def load_imu(path: Path) -> list[tuple[float, tuple[float, ...]]]:
    samples: list[tuple[float, tuple[float, ...]]] = []
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 9:
                continue
            timestamp = int(row[1]) + int(row[2]) * 1e-9
            values = tuple(float(value) for value in row[3:9])
            samples.append((timestamp, values))
    return samples


def normalized_point_time(cloud: PointCloud, point_count: int) -> np.ndarray:
    field_names = set(cloud.fields)
    source_name = "t" if "t" in field_names else "time" if "time" in field_names else None
    if source_name is not None:
        raw = np.asarray(cloud.pc_data[source_name]).reshape(-1).astype(np.float64)
        raw -= np.nanmin(raw)
        maximum = float(np.nanmax(raw)) if raw.size else 0.0
        if maximum <= 0.2:
            raw *= 1e9
        elif maximum <= 200.0:
            raw *= 1e6
        elif maximum <= 200_000.0:
            raw *= 1e3
        if np.isfinite(raw).all() and 1.0 < float(np.nanmax(raw)) <= 200_000_000.0:
            return np.clip(raw, 0.0, 200_000_000.0).astype(np.uint32)

    # The raw files preserve acquisition order. This fallback gives FAST-LIO a
    # monotonic 0.1 s offset when an exported PCD omitted the Ouster t field.
    if point_count <= 1:
        return np.zeros(point_count, dtype=np.uint32)
    return np.linspace(0, 99_999_999, point_count, dtype=np.uint32)


def cloud_message(archive: zipfile.ZipFile, member: str, timestamp: float) -> PointCloud2:
    with archive.open(member) as stream:
        cloud = PointCloud.from_fileobj(stream)

    names = set(cloud.fields)
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise RuntimeError(f"{member} has fields {cloud.fields}, expected at least x/y/z")

    xyz = cloud.numpy(("x", "y", "z")).astype(np.float32, copy=False)
    count = xyz.shape[0]
    packed = np.zeros(count, dtype=POINT_DTYPE)
    packed["x"], packed["y"], packed["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    intensity_name = "intensity" if "intensity" in names else "reflectivity" if "reflectivity" in names else None
    if intensity_name is not None:
        packed["intensity"] = np.asarray(cloud.pc_data[intensity_name]).reshape(-1)
    if "reflectivity" in names:
        packed["reflectivity"] = np.asarray(cloud.pc_data["reflectivity"]).reshape(-1)
    if "ring" in names:
        packed["ring"] = np.asarray(cloud.pc_data["ring"]).reshape(-1)
    if "ambient" in names:
        packed["ambient"] = np.asarray(cloud.pc_data["ambient"]).reshape(-1)

    packed["t"] = normalized_point_time(cloud, count)
    packed["range"] = np.clip(np.linalg.norm(xyz, axis=1) * 1000.0, 0, 2**32 - 1).astype(np.uint32)
    data = packed.tobytes()
    return PointCloud2(
        header=Header(stamp=ros_time(timestamp), frame_id="os1_lidar"),
        height=1,
        width=count,
        fields=POINT_FIELDS,
        is_bigendian=False,
        point_step=POINT_DTYPE.itemsize,
        row_step=POINT_DTYPE.itemsize * count,
        data=data,
        is_dense=bool(np.isfinite(xyz).all()),
    )


def imu_message(timestamp: float, values: tuple[float, ...]) -> Imu:
    wx, wy, wz, ax, ay, az = values
    message = Imu(header=Header(stamp=ros_time(timestamp), frame_id="os1_imu"))
    message.orientation_covariance[0] = -1.0
    message.angular_velocity.x = wx
    message.angular_velocity.y = wy
    message.angular_velocity.z = wz
    message.linear_acceleration.x = ax
    message.linear_acceleration.y = ay
    message.linear_acceleration.z = az
    return message


def wait_until(node: Node, target: float) -> None:
    while rclpy.ok():
        remaining = target - time.monotonic()
        if remaining <= 0.0:
            return
        rclpy.spin_once(node, timeout_sec=min(remaining, 0.01))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-zip", type=Path, required=True)
    parser.add_argument("--imu-csv", type=Path, required=True)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--lidar-topic", default="/os1_cloud_node/points")
    parser.add_argument("--imu-topic", default="/os1_cloud_node/imu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rate <= 0.0 or args.duration <= 0.0 or args.start_offset < 0.0:
        raise SystemExit("rate and duration must be positive; start-offset must be non-negative")

    with zipfile.ZipFile(args.scan_zip) as archive:
        scans = sorted(
            (timestamp, name)
            for name in archive.namelist()
            if (timestamp := member_timestamp(name)) is not None
        )
        if not scans:
            raise RuntimeError(f"No timestamped PCD members found in {args.scan_zip}")
        sequence_start = scans[0][0]
        begin = sequence_start + args.start_offset
        end = begin + args.duration
        scans = [(stamp, name) for stamp, name in scans if begin <= stamp <= end]
        if not scans:
            raise RuntimeError("Requested interval contains no LiDAR scans")

        imu = [(stamp, values) for stamp, values in load_imu(args.imu_csv) if begin - 0.5 <= stamp <= end + 0.2]
        events = [(stamp, 0, values) for stamp, values in imu]
        events.extend((stamp, 1, name) for stamp, name in scans)
        events.sort(key=lambda item: (item[0], item[1]))

        rclpy.init()
        node = Node("newer_college_raw_replay")
        lidar_pub = node.create_publisher(PointCloud2, args.lidar_topic, qos_profile_sensor_data)
        imu_pub = node.create_publisher(Imu, args.imu_topic, qos_profile_sensor_data)
        node.get_logger().info(
            f"Prepared {len(scans)} OS1 scans and {len(imu)} IMU samples "
            f"from offset {args.start_offset:.1f}s for {args.duration:.1f}s"
        )

        connection_deadline = time.monotonic() + 5.0
        while (
            lidar_pub.get_subscription_count() == 0 or imu_pub.get_subscription_count() == 0
        ) and time.monotonic() < connection_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)

        first_stamp = events[0][0]
        wall_start = time.monotonic()
        lidar_count = 0
        try:
            for timestamp, event_type, payload in events:
                wait_until(node, wall_start + (timestamp - first_stamp) / args.rate)
                if event_type == 0:
                    imu_pub.publish(imu_message(timestamp, payload))
                else:
                    lidar_pub.publish(cloud_message(archive, payload, timestamp))
                    lidar_count += 1
                    if lidar_count == 1 or lidar_count % 50 == 0:
                        node.get_logger().info(f"Published LiDAR scan {lidar_count}/{len(scans)}")
            for _ in range(20):
                rclpy.spin_once(node, timeout_sec=0.01)
        finally:
            node.get_logger().info(f"Replay complete: {lidar_count} LiDAR scans published")
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
