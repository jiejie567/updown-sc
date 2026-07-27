import os.path
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def _as_bool(value, default=False, name='value'):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    lowered = text.lower()
    if lowered in ('1', 'true', 'yes', 'on'):
        return True
    if lowered in ('0', 'false', 'no', 'off'):
        return False
    raise RuntimeError(f'Invalid boolean launch argument {name}: {text}')


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
    loc_rviz_config_path = os.path.join(
        package_path, 'rviz', 'fastlio_prior.rviz')
    default_map_pcd_path = os.path.join(package_path, 'prior_map', 'scans.pcd')

    use_sim_time = LaunchConfiguration('use_sim_time')
    config_path = LaunchConfiguration('config_path')
    config_file = LaunchConfiguration('config_file')
    rviz_use = LaunchConfiguration('rviz')
    use_rviz = LaunchConfiguration('use_rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    map_pcd = LaunchConfiguration('map_pcd')

    config_path_value = config_path.perform(context)
    config_file_value = config_file.perform(context)
    if os.path.isabs(config_file_value):
        resolved_config_file = config_file_value
    else:
        resolved_config_file = os.path.join(config_path_value, config_file_value)

    ros_parameters = _load_ros_parameters(resolved_config_file)
    visualization = ros_parameters.get('visualization', {})
    if not isinstance(visualization, dict):
        raise RuntimeError(f'Config visualization must be a YAML mapping: {resolved_config_file}')
    rviz_enabled = _as_bool(visualization.get('rviz'), False, 'visualization.rviz')
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
        rviz_cfg_value = loc_rviz_config_path
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
    }
    if map_pcd_value:
        extra_params['map_file_path'] = map_pcd_value

    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        parameters=[resolved_config_file, extra_params],
        output='screen'
    )

    actions = [fast_lio_node]
    if rviz_enabled:
        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_cfg_value],
            output='screen'
        ))

    return actions


def generate_launch_description():
    package_path = get_package_share_directory('fast_lio')
    default_config_path = os.path.join(package_path, 'config')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    declare_config_path_cmd = DeclareLaunchArgument(
        'config_path', default_value=default_config_path,
        description='Yaml config file path'
    )
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='mid360.yaml',
        description='Config file'
    )
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='',
        description='Override visualization.rviz from yaml when set'
    )
    declare_use_rviz_cmd = DeclareLaunchArgument(
        'use_rviz', default_value='',
        description='Alias override for visualization.rviz from yaml when set'
    )
    declare_rviz_config_path_cmd = DeclareLaunchArgument(
        'rviz_cfg', default_value='',
        description='RViz config file path override'
    )
    declare_map_pcd_cmd = DeclareLaunchArgument(
        'map_pcd', default_value='',
        description='Path to prior map PCD file (overrides map_file_path in config)'
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_path_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_rviz_config_path_cmd)
    ld.add_action(declare_map_pcd_cmd)
    ld.add_action(OpaqueFunction(function=_launch_setup))

    return ld
