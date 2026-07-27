import os.path
import math
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def _as_bool(value, default=False, name='value'):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    value = str(value).strip()
    if not value:
        return default
    lowered = value.lower()
    if lowered in ('1', 'true', 'yes', 'on'):
        return True
    if lowered in ('0', 'false', 'no', 'off'):
        return False
    raise RuntimeError(f'Invalid boolean launch argument {name}: {value}')


def _as_float(value, default=0.0, name='value'):
    text = '' if value is None else str(value).strip()
    if not text:
        return default
    try:
        result = float(text)
    except (TypeError, ValueError):
        raise RuntimeError(f'Invalid numeric launch argument {name}: {text}')
    if not math.isfinite(result):
        raise RuntimeError(f'Invalid numeric launch argument {name}: {text}')
    return result


def _as_int(value, default=0, name='value'):
    text = '' if value is None else str(value).strip()
    if not text:
        return default
    try:
        return int(text, 10)
    except (TypeError, ValueError):
        raise RuntimeError(f'Invalid integer launch argument {name}: {text}')


def _load_ros_parameters(config_file):
    try:
        with open(config_file, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file) or {}
    except OSError as exc:
        raise RuntimeError(f'Failed to open config file: {config_file}') from exc

    if not isinstance(config, dict):
        raise RuntimeError(f'Config file must be a YAML mapping: {config_file}')
    wildcard = config.get('/**', {})
    if not isinstance(wildcard, dict):
        raise RuntimeError(f'Config /** entry must be a YAML mapping: {config_file}')
    ros_parameters = wildcard.get('ros__parameters', {})
    if not ros_parameters and 'ros__parameters' in config:
        ros_parameters = config.get('ros__parameters', {})
    if not isinstance(ros_parameters, dict):
        raise RuntimeError(f'Config ros__parameters must be a YAML mapping: {config_file}')
    return ros_parameters


def _launch_setup(context, *args, **kwargs):
    package_path = get_package_share_directory('fast_lio')
    default_rviz_config_path = os.path.join(package_path, 'rviz', 'fastlio_prior.rviz')
    default_qos_path = os.path.join(package_path, 'config', 'rosbag_qos_reliable.yaml')
    default_map_pcd_path = os.path.join(package_path, 'prior_map', 'scans.pcd')

    use_sim_time = LaunchConfiguration('use_sim_time')
    config_path = LaunchConfiguration('config_path')
    config_file = LaunchConfiguration('config_file')
    rviz_use = LaunchConfiguration('rviz')
    use_rviz = LaunchConfiguration('use_rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    map_pcd = LaunchConfiguration('map_pcd')
    bag_path = LaunchConfiguration('bag_path')
    bag_lid_topic = LaunchConfiguration('bag_lid_topic')
    bag_imu_topic = LaunchConfiguration('bag_imu_topic')
    qos_path = LaunchConfiguration('qos_path')
    play = LaunchConfiguration('play')
    play_delay = LaunchConfiguration('play_delay')
    start_offset = LaunchConfiguration('start_offset')
    rate = LaunchConfiguration('rate')
    clock_hz = LaunchConfiguration('clock_hz')
    read_ahead_queue_size = LaunchConfiguration('read_ahead_queue_size')

    config_path_value = config_path.perform(context)
    config_file_value = config_file.perform(context)
    if os.path.isabs(config_file_value):
        resolved_config_file = config_file_value
    else:
        resolved_config_file = os.path.join(config_path_value, config_file_value)

    ros_parameters = _load_ros_parameters(resolved_config_file)
    common = ros_parameters.get('common', {})
    if not isinstance(common, dict):
        raise RuntimeError(f'Config common must be a YAML mapping: {resolved_config_file}')
    lid_topic = str(common.get('lid_topic', '/driver/lidar/point_cloud/Data')).strip()
    imu_topic = str(common.get('imu_topic', '/driver/lidar/lidar_front/imu/Data')).strip()
    if not lid_topic:
        raise RuntimeError('common.lid_topic must not be empty')
    if not imu_topic:
        raise RuntimeError('common.imu_topic must not be empty')

    visualization = ros_parameters.get('visualization', {})
    if not isinstance(visualization, dict):
        raise RuntimeError(f'Config visualization must be a YAML mapping: {resolved_config_file}')
    rviz_enabled = _as_bool(visualization.get('rviz'), True, 'visualization.rviz')
    rviz_arg_value = rviz_use.perform(context).strip()
    use_rviz_arg_value = use_rviz.perform(context).strip()
    if rviz_arg_value:
        rviz_enabled = _as_bool(rviz_arg_value, rviz_enabled, 'rviz')
    if use_rviz_arg_value:
        rviz_enabled = _as_bool(use_rviz_arg_value, rviz_enabled, 'use_rviz')

    rviz_cfg_value = rviz_cfg.perform(context).strip()
    if not rviz_cfg_value:
        rviz_cfg_value = str(visualization.get('rviz_cfg', '')).strip()
    if not rviz_cfg_value:
        rviz_cfg_value = default_rviz_config_path
    if rviz_enabled and not os.path.exists(rviz_cfg_value):
        raise RuntimeError(f'rviz_cfg does not exist: {rviz_cfg_value}')

    map_pcd_value = map_pcd.perform(context).strip()
    if not map_pcd_value and os.path.exists(default_map_pcd_path):
        map_pcd_value = default_map_pcd_path
    if map_pcd_value and not os.path.exists(map_pcd_value):
        raise RuntimeError(f'map_pcd does not exist: {map_pcd_value}')
    extra_params = {
        'use_sim_time': use_sim_time,
        'runtime.profile': 'localization',
        'pcd_save.pcd_save_en': False,
        'publish_tf': False,
        'common.lidar_qos_reliability': 'reliable',
        'common.imu_qos_reliability': 'reliable',
        'common.lidar_subscribe_qos_depth': 500,
        'common.imu_subscribe_qos_depth': 4000,
    }
    if map_pcd_value:
        extra_params['map_file_path'] = map_pcd_value

    actions = [
        Node(
            package='fast_lio',
            executable='fastlio_mapping',
            parameters=[resolved_config_file, extra_params],
            output='screen'
        )
    ]

    if rviz_enabled:
        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_cfg_value],
            output='screen'
        ))

    read_ahead_value = read_ahead_queue_size.perform(context).strip()
    if not read_ahead_value:
        read_ahead_value = '10000'
    if _as_int(read_ahead_value, 10000, 'read_ahead_queue_size') <= 0:
        raise RuntimeError(f'read_ahead_queue_size must be > 0: {read_ahead_value}')

    rate_value = rate.perform(context).strip()
    if rate_value and _as_float(rate_value, 1.0, 'rate') <= 0.0:
        raise RuntimeError(f'rate must be > 0: {rate_value}')

    start_offset_value = start_offset.perform(context).strip()
    start_offset_sec = _as_float(start_offset_value, 0.0, 'start_offset')
    if start_offset_sec < 0.0:
        raise RuntimeError(f'start_offset must be >= 0: {start_offset_value}')

    clock_hz_value = clock_hz.perform(context).strip()
    clock_hz_sec = _as_float(clock_hz_value, 0.0, 'clock_hz')
    if clock_hz_sec < 0.0:
        raise RuntimeError(f'clock_hz must be >= 0: {clock_hz_value}')

    play_delay_value = play_delay.perform(context)
    play_delay_sec = _as_float(play_delay_value, 2.0, 'play_delay')
    if play_delay_sec < 0.0:
        raise RuntimeError(f'play_delay must be >= 0: {play_delay_value}')

    if _as_bool(play.perform(context), True, 'play'):
        bag_path_value = bag_path.perform(context).strip()
        if not bag_path_value:
            raise RuntimeError('bag_path must not be empty when play=true')
        if not os.path.exists(bag_path_value):
            raise RuntimeError(f'bag_path does not exist: {bag_path_value}')
        qos_path_value = qos_path.perform(context).strip() or default_qos_path
        if not os.path.exists(qos_path_value):
            raise RuntimeError(f'qos_path does not exist: {qos_path_value}')
        play_lid_topic = bag_lid_topic.perform(context).strip() or lid_topic
        play_imu_topic = bag_imu_topic.perform(context).strip() or imu_topic
        if not play_lid_topic:
            raise RuntimeError('bag_lid_topic must not be empty')
        if not play_imu_topic:
            raise RuntimeError('bag_imu_topic must not be empty')
        play_cmd = [
            'ros2', 'bag', 'play', bag_path_value,
            '--topics', play_lid_topic, play_imu_topic,
            '--qos-profile-overrides-path', qos_path_value,
            '--read-ahead-queue-size', read_ahead_value,
        ]
        remaps = []
        if play_lid_topic != lid_topic:
            remaps.append(f'{play_lid_topic}:={lid_topic}')
        if play_imu_topic != imu_topic:
            remaps.append(f'{play_imu_topic}:={imu_topic}')
        if remaps:
            play_cmd += ['--remap'] + remaps

        if rate_value:
            play_cmd += ['--rate', rate_value]

        if start_offset_sec > 0.0:
            play_cmd += ['--start-offset', start_offset_value]

        if clock_hz_sec > 0.0:
            play_cmd += ['--clock', clock_hz_value]

        actions.append(TimerAction(
            period=play_delay_sec,
            actions=[ExecuteProcess(cmd=play_cmd, output='screen')]
        ))

    return actions


def generate_launch_description():
    package_path = get_package_share_directory('fast_lio')
    default_config_path = os.path.join(package_path, 'config')
    default_bag_path = '${UPDOWN_SC_ROOT}/rosbag/nav_debug_bag_2'

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation clock if true'
    ))
    ld.add_action(DeclareLaunchArgument(
        'config_path', default_value=default_config_path,
        description='Yaml config file path'
    ))
    ld.add_action(DeclareLaunchArgument(
        'config_file', default_value='mid360.yaml',
        description='Config file'
    ))
    ld.add_action(DeclareLaunchArgument(
        'map_pcd', default_value='',
        description='Path to prior map PCD file (overrides map_file_path in config)'
    ))
    ld.add_action(DeclareLaunchArgument(
        'rviz', default_value='',
        description='Override visualization.rviz from yaml when set'
    ))
    ld.add_action(DeclareLaunchArgument(
        'use_rviz', default_value='',
        description='Alias override for visualization.rviz from yaml when set'
    ))
    ld.add_action(DeclareLaunchArgument(
        'rviz_cfg', default_value='',
        description='RViz config file path override'
    ))
    ld.add_action(DeclareLaunchArgument(
        'play', default_value='true',
        description='Start ros2 bag play together with localization'
    ))
    ld.add_action(DeclareLaunchArgument(
        'bag_path', default_value=default_bag_path,
        description='Rosbag2 directory to replay'
    ))
    ld.add_action(DeclareLaunchArgument(
        'bag_lid_topic', default_value='',
        description='LiDAR topic inside the bag; defaults to common.lid_topic and remaps to it when different'
    ))
    ld.add_action(DeclareLaunchArgument(
        'bag_imu_topic', default_value='',
        description='IMU topic inside the bag; defaults to common.imu_topic and remaps to it when different'
    ))
    ld.add_action(DeclareLaunchArgument(
        'qos_path', default_value='',
        description='QoS override file for rosbag replay; defaults to rosbag_qos_reliable.yaml'
    ))
    ld.add_action(DeclareLaunchArgument(
        'play_delay', default_value='2.0',
        description='Seconds to wait before starting rosbag playback'
    ))
    ld.add_action(DeclareLaunchArgument(
        'start_offset', default_value='0.0',
        description='Start playback this many seconds after the bag start'
    ))
    ld.add_action(DeclareLaunchArgument(
        'rate', default_value='3.0',
        description='Rosbag playback rate (default: 3x)'
    ))
    ld.add_action(DeclareLaunchArgument(
        'clock_hz', default_value='20',
        description='Publish /clock at this frequency; set 0 to disable'
    ))
    ld.add_action(DeclareLaunchArgument(
        'read_ahead_queue_size', default_value='10000',
        description='ros2 bag play read-ahead queue size'
    ))
    ld.add_action(OpaqueFunction(function=_launch_setup))
    return ld
