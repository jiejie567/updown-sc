#!/usr/bin/env python3
"""Record trusted FAST-LIO localization poses to a compact CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class PoseRecorder(Node):
    def __init__(self, topic: str, output: Path) -> None:
        super().__init__("construction_hall_pose_recorder")
        output.parent.mkdir(parents=True, exist_ok=True)
        self._stream = output.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._stream)
        self._writer.writerow(
            ["stamp_ns", "x", "y", "z", "qx", "qy", "qz", "qw", "frame_id"]
        )
        self._count = 0
        self.create_subscription(PoseStamped, topic, self._on_pose, 100)
        self.get_logger().info(f"Recording trusted poses from {topic} to {output}")

    def _on_pose(self, message: PoseStamped) -> None:
        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        pose = message.pose
        self._writer.writerow(
            [
                stamp_ns,
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
                message.header.frame_id,
            ]
        )
        self._count += 1
        if self._count % 100 == 0:
            self._stream.flush()
            self.get_logger().info(f"Recorded {self._count} trusted poses")

    def close(self) -> None:
        self._stream.flush()
        self._stream.close()
        if rclpy.ok():
            self.get_logger().info(f"Saved {self._count} trusted poses")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/localization_pose")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("localization_pose.csv"),
    )
    args = parser.parse_args()

    rclpy.init()
    recorder = PoseRecorder(args.topic, args.output.resolve())
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.close()
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
