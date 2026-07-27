"""Replay a raw LiDAR/IMU bag and export registered 10-scan BTC submaps."""

import math
import os.path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_parameters(path):
    with open(path, "r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    wildcard = document.get("/**", {})
    parameters = wildcard.get("ros__parameters", document.get("ros__parameters", {}))
    if not isinstance(parameters, dict):
        raise RuntimeError(f"Invalid ROS parameter YAML: {path}")
    return parameters


def _positive_float(value, name, allow_zero=False):
    result = float(str(value).strip())
    if not math.isfinite(result) or result < 0.0 or (result == 0.0 and not allow_zero):
        relation = "nonnegative" if allow_zero else "positive"
        raise RuntimeError(f"{name} must be {relation}: {value}")
    return result


def _positive_int(value, name, allow_zero=False):
    result = int(str(value).strip())
    if result < 0 or (result == 0 and not allow_zero):
        relation = "nonnegative" if allow_zero else "positive"
        raise RuntimeError(f"{name} must be {relation}: {value}")
    return result


def _as_bool(value, name):
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false: {value}")


def _setup(context):
    package = get_package_share_directory("fast_lio")
    config_path = LaunchConfiguration("config_path").perform(context)
    config_file = LaunchConfiguration("config_file").perform(context)
    config = config_file if os.path.isabs(config_file) else os.path.join(config_path, config_file)
    if not os.path.exists(config):
        raise RuntimeError(f"Config file does not exist: {config}")
    parameters = _load_parameters(config)
    common = parameters.get("common", {})
    configured_lidar = str(common.get("lid_topic", "/driver/lidar/point_cloud/Data"))
    configured_imu = str(common.get("imu_topic", "/driver/lidar/lidar_front/imu/Data"))

    bag_path = LaunchConfiguration("bag_path").perform(context).strip()
    output_dir = LaunchConfiguration("output_dir").perform(context).strip()
    if not os.path.exists(bag_path):
        raise RuntimeError(f"Bag does not exist: {bag_path}")
    if not output_dir:
        raise RuntimeError("output_dir must not be empty")

    bag_lidar = LaunchConfiguration("bag_lid_topic").perform(context).strip() or configured_lidar
    bag_imu = LaunchConfiguration("bag_imu_topic").perform(context).strip() or configured_imu
    rate_text = LaunchConfiguration("rate").perform(context)
    delay_text = LaunchConfiguration("play_delay").perform(context)
    start_text = LaunchConfiguration("start_offset").perform(context)
    duration_text = LaunchConfiguration("playback_duration").perform(context)
    rate = _positive_float(rate_text, "rate")
    delay = _positive_float(delay_text, "play_delay", allow_zero=True)
    start = _positive_float(start_text, "start_offset", allow_zero=True)
    duration = float(duration_text)
    drain_delay = _positive_float(
        LaunchConfiguration("drain_delay").perform(context),
        "drain_delay",
        allow_zero=True,
    )
    if not math.isfinite(duration):
        raise RuntimeError(f"playback_duration must be finite: {duration_text}")

    scans_text = LaunchConfiguration("scans_per_submap").perform(context)
    stride_text = LaunchConfiguration("submap_stride_scans").perform(context)
    voxel_text = LaunchConfiguration("voxel_size").perform(context)
    max_submaps_text = LaunchConfiguration("maximum_submaps").perform(context)
    scans = _positive_int(scans_text, "scans_per_submap")
    stride = _positive_int(stride_text, "submap_stride_scans")
    _positive_float(voxel_text, "voxel_size", allow_zero=True)
    _positive_int(max_submaps_text, "maximum_submaps", allow_zero=True)
    if stride > scans:
        raise RuntimeError("submap_stride_scans must not exceed scans_per_submap")

    fast_lio = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        parameters=[
            config,
            {
                "use_sim_time": True,
                "runtime.profile": "mapping",
                "runtime_pos_log_enable": False,
                "publish_tf": False,
                "publish_pose_topic": False,
                "common.lidar_qos_reliability": "reliable",
                "common.imu_qos_reliability": "reliable",
                "common.lidar_subscribe_qos_depth": 500,
                "common.imu_subscribe_qos_depth": 4000,
                "prior_map.mapping_use_prior_map": False,
                "prior_map.scan_context.enable": False,
                "publish.path_en": False,
                "publish.effect_map_en": False,
                # The first world-cloud publication initializes FAST-LIO's
                # gravity-aligned local map frame; odometry is published from
                # the following scan onward.
                "publish.scan_publish_en": True,
                "publish.scan_bodyframe_pub_en": True,
                "publish.scan_bodyframe_stride_s": 0.0,
                "pcd_save.pcd_save_en": False,
                "manual_loop_export.enable": False,
            },
        ],
        output="screen",
    )

    exporter_arguments = [
        "--output-dir", output_dir,
        "--scans-per-submap", scans_text,
        "--submap-stride-scans", stride_text,
        "--voxel-size", voxel_text,
        "--maximum-submaps", max_submaps_text,
    ]
    if _as_bool(LaunchConfiguration("overwrite").perform(context), "overwrite"):
        exporter_arguments.append("--overwrite")
    exporter = Node(
        package="fast_lio",
        executable="export_btc_submaps.py",
        arguments=exporter_arguments,
        output="screen",
    )

    qos_path = LaunchConfiguration("qos_path").perform(context).strip()
    if not qos_path:
        qos_path = os.path.join(package, "config", "rosbag_qos_reliable.yaml")
    play_command = [
        "ros2", "bag", "play", bag_path,
        "--topics", bag_lidar, bag_imu,
        "--qos-profile-overrides-path", qos_path,
        "--read-ahead-queue-size", "10000",
        "--rate", str(rate),
        "--clock", "20",
    ]
    remaps = []
    if bag_lidar != configured_lidar:
        remaps.append(f"{bag_lidar}:={configured_lidar}")
    if bag_imu != configured_imu:
        remaps.append(f"{bag_imu}:={configured_imu}")
    if remaps:
        play_command += ["--remap"] + remaps
    if start > 0.0:
        play_command += ["--start-offset", str(start)]
    if duration > 0.0:
        play_command += ["--playback-duration", str(duration)]

    player = ExecuteProcess(cmd=play_command, output="screen")
    delayed_player = TimerAction(period=delay, actions=[player])
    stop_after_bag = RegisterEventHandler(
        OnProcessExit(
            target_action=player,
            # rosbag can reach EOF while reliable LiDAR/IMU messages are still
            # queued in FAST-LIO.  Let the pipeline drain before closing the
            # exporter, especially for accelerated offline replay.
            on_exit=[
                TimerAction(
                    period=drain_delay,
                    actions=[EmitEvent(event=Shutdown(reason="BTC source bag drained"))],
                )
            ],
        )
    )
    return [fast_lio, exporter, stop_after_bag, delayed_player]


def generate_launch_description():
    package = get_package_share_directory("fast_lio")
    declarations = [
        DeclareLaunchArgument("config_path", default_value=os.path.join(package, "config")),
        DeclareLaunchArgument("config_file", default_value="mid360.yaml"),
        DeclareLaunchArgument("bag_path", default_value="${UPDOWN_SC_ROOT}/rosbag/loc_2_floor"),
        DeclareLaunchArgument("bag_lid_topic", default_value=""),
        DeclareLaunchArgument("bag_imu_topic", default_value=""),
        DeclareLaunchArgument("qos_path", default_value=""),
        DeclareLaunchArgument(
            "output_dir", default_value="${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/btc_submaps/current"
        ),
        DeclareLaunchArgument("scans_per_submap", default_value="10"),
        DeclareLaunchArgument("submap_stride_scans", default_value="10"),
        DeclareLaunchArgument("voxel_size", default_value="0.1"),
        DeclareLaunchArgument("maximum_submaps", default_value="0"),
        DeclareLaunchArgument("overwrite", default_value="false"),
        # This host sustains the mapping plus reliable submap export at 2x.
        # At 3x rosbag reaches EOF before FAST-LIO processes the final scans.
        DeclareLaunchArgument("rate", default_value="2.0"),
        DeclareLaunchArgument("play_delay", default_value="2.0"),
        DeclareLaunchArgument("drain_delay", default_value="5.0"),
        DeclareLaunchArgument("start_offset", default_value="0.0"),
        DeclareLaunchArgument("playback_duration", default_value="-1.0"),
    ]
    return LaunchDescription(declarations + [OpaqueFunction(function=_setup)])
