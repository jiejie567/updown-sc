#include "prior_icp.hpp"
#include "preprocess.h"
#include "scan_context.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <Eigen/Geometry>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <rosbag2_cpp/converter_options.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace prior_icp = fast_lio::prior_icp;
namespace sc = fast_lio::scan_context;

namespace
{

constexpr std::size_t kMaxWindowSlots = 200000;

struct Options
{
    std::string bag_path = "${UPDOWN_SC_ROOT}/anyverse/rosbag/nav_debug_bag_2";
    std::string config_path = std::string(ROOT_DIR) + "config/mid360.yaml";
    std::string bag_topic;
    std::string map_path;
    std::string scan_context_database_path;
    std::string output_dir = std::string(ROOT_DIR) + "Log/relocalization_windows";
    double stride_s = 1.0;
    double sample_s = 0.2;
    bool input_is_undistorted = false;
    bool save_window_pcds = true;
    bool one_frame_per_window = true;
    bool save_summary_png = true;
    std::string truth_csv;
    std::string gravity_csv;
    std::string gravity_topic = "/scan_context_gravity_up";
    std::vector<int> selected_windows;
};

struct RelocConfig
{
    std::string map_file_path = std::string(ROOT_DIR) + "prior_map/scans.pcd";
    std::string scan_context_database_path;
    std::string topic = "/driver/lidar/point_cloud/Data";
    bool scan_context_enable = true;
    bool multi_seed_enable = true;
    double scan_context_voxel_leaf = 0.25;
    double scan_context_seed_xy_offset = 0.5;
    double initial_guess_x = 0.0;
    double initial_guess_y = 0.0;
    double initial_guess_yaw_deg = 0.0;
    double seed_xy_range = 1.0;
    double seed_xy_step = 0.5;
    double seed_yaw_range_deg = 30.0;
    double seed_yaw_step_deg = 10.0;
    double icp_max_corr_dist = 1.0;
    double icp_fitness_thresh = 0.1;
    double icp_min_overlap_ratio = 0.5;
    double voxel_leaf = 0.5;
    double voxel_leaf_fine = 0.25;
    int icp_max_iterations = 50;
    int icp_min_points = 1000;
    int icp_refine_top_k = 3;
    bool feature_extract_enable = false;
    int lidar_type = MID360;
    int scan_line = 4;
    int timestamp_unit = US;
    int scan_rate = 10;
    int point_filter_num = 4;
    double mapping_blind = 2.0;
    double localization_blind = 0.3;
    double blind = 0.3;
    std::string blind_filter_shape = "sphere";
    double blind_z_min = -1.0e9;
    double blind_z_max = 1.0e9;
    double max_range = -1.0;
    double det_range = 100.0;
    double max_height = 100.0;
    std::string tag_filter_mode = "low_confidence";
    sc::Config scan_context;
};

using IcpResult = prior_icp::Result;

struct WindowResult
{
    int window_index = -1;
    double start_s = 0.0;
    double end_s = 0.0;
    std::size_t raw_points = 0;
    std::size_t source_coarse_points = 0;
    std::size_t source_fine_points = 0;
    std::size_t seeds = 0;
    int coarse_valid = 0;
    int fine_valid = 0;
    double total_ms = 0.0;
    double downsample_ms = 0.0;
    double scan_context_ms = 0.0;
    double coarse_icp_ms = 0.0;
    double fine_icp_ms = 0.0;
    bool success = false;
    std::string reason;
    IcpResult best;
    int best_candidate_rank = -1;
    int best_candidate_index = -1;
    double best_candidate_distance = std::numeric_limits<double>::infinity();
    double best_coarse_vertical_shift = 0.0;
    double best_vertical_shift = 0.0;
    std::vector<sc::Candidate> scan_context_candidates;
    PointCloudXYZI::Ptr source_raw{new PointCloudXYZI()};
    PointCloudXYZI::Ptr registered_source{new PointCloudXYZI()};
};

template <typename T>
T yaml_value(const YAML::Node &node, const std::string &key, const T &fallback)
{
    const YAML::Node child = node[key];
    if (!child)
        return fallback;
    try
    {
        return child.as<T>();
    }
    catch (const YAML::Exception &e)
    {
        throw std::runtime_error("invalid config value for " + key + ": " + e.what());
    }
}

YAML::Node ros_params_node(const YAML::Node &root)
{
    if (root["/**"] && root["/**"]["ros__parameters"])
        return root["/**"]["ros__parameters"];
    if (root["ros__parameters"])
        return root["ros__parameters"];
    return root;
}

void require_yaml_map(const YAML::Node &node, const std::string &name)
{
    if (node && !node.IsMap())
        throw std::runtime_error("config " + name + " must be a YAML mapping");
}

std::string resolve_path(const std::string &path)
{
    if (path.empty())
        return path;
    fs::path p(path);
    if (p.is_absolute())
        return p.string();
    return (fs::path(ROOT_DIR) / p).lexically_normal().string();
}

int parse_tag_filter_mode(std::string value)
{
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "off" || value == "none" || value == "disable" || value == "disabled")
        return Preprocess::TAG_FILTER_OFF;
    if (value == "other" || value == "fast_lio")
        return Preprocess::TAG_FILTER_OTHER;
    if (value == "low_confidence" || value == "low-confidence" || value == "low")
        return Preprocess::TAG_FILTER_LOW_CONFIDENCE;
    if (value == "strict" || value == "high_confidence" || value == "high")
        return Preprocess::TAG_FILTER_STRICT;
    return Preprocess::TAG_FILTER_LOW_CONFIDENCE;
}

int parse_blind_filter_shape(std::string value)
{
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "sphere" || value == "spherical")
        return Preprocess::BLIND_FILTER_SPHERE;
    if (value == "cylinder" || value == "cylindrical")
        return Preprocess::BLIND_FILTER_CYLINDER;
    return Preprocess::BLIND_FILTER_SPHERE;
}

std::string detect_storage_id(const std::string &bag_path)
{
    const fs::path metadata_path = fs::path(bag_path) / "metadata.yaml";
    if (!fs::exists(metadata_path))
        return "";

    try
    {
        const YAML::Node root = YAML::LoadFile(metadata_path.string());
        const YAML::Node info = root["rosbag2_bagfile_information"];
        const YAML::Node storage =
            info ? info["storage_identifier"] : root["storage_identifier"];
        if (storage)
            return storage.as<std::string>();
    }
    catch (const std::exception &)
    {
        return "";
    }
    return "";
}

std::vector<std::string> read_pointcloud_topics(const std::string &bag_path)
{
    std::vector<std::string> topics;
    const fs::path metadata_path = fs::path(bag_path) / "metadata.yaml";
    if (!fs::exists(metadata_path))
        return topics;

    try
    {
        const YAML::Node root = YAML::LoadFile(metadata_path.string());
        const YAML::Node info = root["rosbag2_bagfile_information"];
        const YAML::Node topics_with_counts = info ? info["topics_with_message_count"] : YAML::Node();
        if (!topics_with_counts || !topics_with_counts.IsSequence())
            return topics;
        for (const auto &entry : topics_with_counts)
        {
            const YAML::Node metadata = entry["topic_metadata"];
            if (!metadata)
                continue;
            const std::string type = yaml_value<std::string>(metadata, "type", "");
            if (type != "sensor_msgs/msg/PointCloud2")
                continue;
            const std::string name = yaml_value<std::string>(metadata, "name", "");
            if (!name.empty())
                topics.push_back(name);
        }
    }
    catch (const std::exception &)
    {
        topics.clear();
    }
    return topics;
}

std::string format_topic_list(const std::vector<std::string> &topics)
{
    if (topics.empty())
        return "(none found in metadata.yaml)";
    std::ostringstream oss;
    for (std::size_t i = 0; i < topics.size(); ++i)
    {
        if (i > 0)
            oss << ", ";
        oss << topics[i];
    }
    return oss.str();
}

RelocConfig load_config(const std::string &config_path, const std::string &bag_topic_override)
{
    RelocConfig cfg;
    const YAML::Node params = ros_params_node(YAML::LoadFile(config_path));
    require_yaml_map(params, "ros__parameters");

    cfg.map_file_path = resolve_path(yaml_value<std::string>(params, "map_file_path", cfg.map_file_path));

    const YAML::Node common = params["common"];
    require_yaml_map(common, "common");
    if (common)
        cfg.topic = yaml_value<std::string>(common, "lid_topic", cfg.topic);
    if (!bag_topic_override.empty())
        cfg.topic = bag_topic_override;

    cfg.point_filter_num = yaml_value<int>(params, "point_filter_num", cfg.point_filter_num);
    cfg.feature_extract_enable =
        yaml_value<bool>(params, "feature_extract_enable", cfg.feature_extract_enable);

    const YAML::Node mapping = params["mapping"];
    require_yaml_map(mapping, "mapping");
    if (mapping)
    {
        cfg.det_range = yaml_value<double>(mapping, "det_range", cfg.det_range);
        cfg.max_height = yaml_value<double>(mapping, "max_height", cfg.max_height);
    }

    const YAML::Node preprocess = params["preprocess"];
    require_yaml_map(preprocess, "preprocess");
    if (preprocess)
    {
        cfg.lidar_type = yaml_value<int>(preprocess, "lidar_type", cfg.lidar_type);
        cfg.scan_line = yaml_value<int>(preprocess, "scan_line", cfg.scan_line);
        cfg.timestamp_unit = yaml_value<int>(preprocess, "timestamp_unit", cfg.timestamp_unit);
        cfg.scan_rate = yaml_value<int>(preprocess, "scan_rate", cfg.scan_rate);
        const double legacy_blind = yaml_value<double>(preprocess, "blind", cfg.localization_blind);
        cfg.mapping_blind = yaml_value<double>(preprocess, "mapping_blind", legacy_blind);
        cfg.localization_blind = yaml_value<double>(preprocess, "localization_blind", legacy_blind);
        cfg.blind_filter_shape = yaml_value<std::string>(preprocess, "blind_filter_shape", cfg.blind_filter_shape);
        cfg.blind_z_min = yaml_value<double>(preprocess, "blind_z_min", cfg.blind_z_min);
        cfg.blind_z_max = yaml_value<double>(preprocess, "blind_z_max", cfg.blind_z_max);
        cfg.max_range = yaml_value<double>(preprocess, "max_range", cfg.max_range);
        cfg.tag_filter_mode = yaml_value<std::string>(preprocess, "tag_filter_mode", cfg.tag_filter_mode);
    }

    const YAML::Node prior = params["prior_map"];
    require_yaml_map(prior, "prior_map");
    if (prior)
    {
        cfg.icp_max_iterations = yaml_value<int>(prior, "icp_max_iterations", cfg.icp_max_iterations);
        cfg.icp_max_corr_dist = yaml_value<double>(prior, "icp_max_corr_dist", cfg.icp_max_corr_dist);
        cfg.icp_fitness_thresh = yaml_value<double>(prior, "icp_fitness_thresh", cfg.icp_fitness_thresh);
        cfg.icp_min_overlap_ratio = yaml_value<double>(prior, "icp_min_overlap_ratio", cfg.icp_min_overlap_ratio);
        cfg.icp_min_points = yaml_value<int>(prior, "icp_min_points", cfg.icp_min_points);
        cfg.icp_refine_top_k = yaml_value<int>(prior, "icp_refine_top_k", cfg.icp_refine_top_k);
        cfg.voxel_leaf = yaml_value<double>(prior, "voxel_leaf", cfg.voxel_leaf);
        cfg.voxel_leaf_fine = yaml_value<double>(prior, "voxel_leaf_fine", cfg.voxel_leaf_fine);
        cfg.multi_seed_enable = yaml_value<bool>(prior, "multi_seed_enable", cfg.multi_seed_enable);
        cfg.seed_xy_range = yaml_value<double>(prior, "seed_xy_range", cfg.seed_xy_range);
        cfg.seed_xy_step = yaml_value<double>(prior, "seed_xy_step", cfg.seed_xy_step);
        cfg.seed_yaw_range_deg = yaml_value<double>(prior, "seed_yaw_range_deg", cfg.seed_yaw_range_deg);
        cfg.seed_yaw_step_deg = yaml_value<double>(prior, "seed_yaw_step_deg", cfg.seed_yaw_step_deg);
        cfg.initial_guess_yaw_deg = yaml_value<double>(prior, "initial_guess_yaw_deg", cfg.initial_guess_yaw_deg);
        if (prior["initial_guess_xy"] && prior["initial_guess_xy"].IsSequence() && prior["initial_guess_xy"].size() >= 2)
        {
            try
            {
                cfg.initial_guess_x = prior["initial_guess_xy"][0].as<double>();
                cfg.initial_guess_y = prior["initial_guess_xy"][1].as<double>();
            }
            catch (const YAML::Exception &e)
            {
                throw std::runtime_error(
                    std::string("invalid config value for prior_map.initial_guess_xy: ") + e.what());
            }
        }

        const YAML::Node sc_node = prior["scan_context"];
        require_yaml_map(sc_node, "prior_map.scan_context");
        if (sc_node)
        {
            cfg.scan_context_enable = yaml_value<bool>(sc_node, "enable", cfg.scan_context_enable);
            cfg.scan_context_database_path =
                resolve_path(yaml_value<std::string>(sc_node, "database_path", cfg.scan_context_database_path));
            cfg.scan_context_voxel_leaf = yaml_value<double>(sc_node, "voxel_leaf", cfg.scan_context_voxel_leaf);
            cfg.scan_context_seed_xy_offset =
                yaml_value<double>(sc_node, "seed_xy_offset", cfg.scan_context_seed_xy_offset);
            cfg.scan_context.num_rings = yaml_value<int>(sc_node, "num_rings", cfg.scan_context.num_rings);
            cfg.scan_context.num_sectors = yaml_value<int>(sc_node, "num_sectors", cfg.scan_context.num_sectors);
            cfg.scan_context.max_radius = yaml_value<double>(sc_node, "max_radius", cfg.scan_context.max_radius);
            cfg.scan_context.dual_z_layer_enable =
                yaml_value<bool>(sc_node, "dual_z_layer_enable", cfg.scan_context.dual_z_layer_enable);
            cfg.scan_context.dual_z_split_height =
                yaml_value<double>(sc_node, "dual_z_split_height", cfg.scan_context.dual_z_split_height);
            cfg.scan_context.origin_height_from_ground =
                yaml_value<double>(
                    sc_node, "origin_height_from_ground",
                    cfg.scan_context.origin_height_from_ground);
            cfg.scan_context.dual_z_low_weight =
                yaml_value<double>(sc_node, "dual_z_low_weight", cfg.scan_context.dual_z_low_weight);
            cfg.scan_context.dual_z_high_weight =
                yaml_value<double>(sc_node, "dual_z_high_weight", cfg.scan_context.dual_z_high_weight);
            cfg.scan_context.min_joint_rings =
                yaml_value<int>(sc_node, "min_joint_rings", cfg.scan_context.min_joint_rings);
            cfg.scan_context.retrieval_height_offset =
                yaml_value<double>(
                    sc_node, "retrieval_height_offset",
                    cfg.scan_context.retrieval_height_offset);
            cfg.scan_context.sector_support_exponent =
                yaml_value<double>(
                    sc_node, "sector_support_exponent",
                    cfg.scan_context.sector_support_exponent);
            cfg.scan_context.vertical_boundary_margin =
                yaml_value<double>(sc_node, "vertical_boundary_margin",
                                   cfg.scan_context.vertical_boundary_margin);
            cfg.scan_context.gravity_canonicalized =
                yaml_value<bool>(sc_node, "gravity_canonicalization_enable",
                                 cfg.scan_context.gravity_canonicalized);
            cfg.scan_context.vertical_estimation_enable =
                yaml_value<bool>(sc_node, "vertical_estimation_enable",
                                 cfg.scan_context.vertical_estimation_enable);
            cfg.scan_context.vertical_correction_min =
                yaml_value<double>(sc_node, "vertical_correction_min",
                                   cfg.scan_context.vertical_correction_min);
            cfg.scan_context.vertical_correction_max =
                yaml_value<double>(sc_node, "vertical_correction_max",
                                   cfg.scan_context.vertical_correction_max);
            cfg.scan_context.vertical_stable_fraction =
                yaml_value<double>(sc_node, "vertical_stable_fraction",
                                   cfg.scan_context.vertical_stable_fraction);
            cfg.scan_context.candidate_top_k =
                yaml_value<int>(sc_node, "candidate_top_k", cfg.scan_context.candidate_top_k);
            cfg.scan_context.yaw_top_k = yaml_value<int>(sc_node, "yaw_top_k", cfg.scan_context.yaw_top_k);
            cfg.scan_context.distance_thresh =
                yaml_value<double>(sc_node, "distance_thresh", cfg.scan_context.distance_thresh);
        }
    }

    if (cfg.scan_context_database_path.empty())
    {
        fs::path scd = cfg.map_file_path;
        scd.replace_extension(".scd");
        cfg.scan_context_database_path = scd.string();
    }

    auto finite_or = [](double value, double fallback) {
        return std::isfinite(value) ? value : fallback;
    };

    cfg.initial_guess_x = finite_or(cfg.initial_guess_x, 0.0);
    cfg.initial_guess_y = finite_or(cfg.initial_guess_y, 0.0);
    cfg.initial_guess_yaw_deg = finite_or(cfg.initial_guess_yaw_deg, 0.0);
    cfg.icp_max_iterations = std::max(1, cfg.icp_max_iterations);
    cfg.icp_max_corr_dist = std::max(1e-3, finite_or(cfg.icp_max_corr_dist, 1.0));
    cfg.icp_fitness_thresh = std::max(0.0, finite_or(cfg.icp_fitness_thresh, 0.1));
    cfg.icp_min_points = std::max(10, cfg.icp_min_points);
    cfg.icp_refine_top_k = std::max(1, cfg.icp_refine_top_k);
    cfg.icp_min_overlap_ratio = std::clamp(finite_or(cfg.icp_min_overlap_ratio, 0.5), 0.0, 1.0);
    cfg.voxel_leaf = std::max(0.0, finite_or(cfg.voxel_leaf, 0.5));
    cfg.voxel_leaf_fine = std::max(0.0, finite_or(cfg.voxel_leaf_fine, 0.25));
    cfg.seed_xy_range = std::max(0.0, finite_or(cfg.seed_xy_range, 1.0));
    cfg.seed_xy_step = std::max(0.1, finite_or(cfg.seed_xy_step, 0.5));
    cfg.seed_yaw_range_deg = std::max(0.0, finite_or(cfg.seed_yaw_range_deg, 30.0));
    cfg.seed_yaw_step_deg = std::max(1.0, finite_or(cfg.seed_yaw_step_deg, 10.0));
    cfg.seed_xy_range = std::min(cfg.seed_xy_range, 20.0);
    cfg.seed_yaw_range_deg = std::min(cfg.seed_yaw_range_deg, 180.0);
    cfg.lidar_type = std::clamp(cfg.lidar_type, static_cast<int>(AVIA), static_cast<int>(RSAIRY));
    cfg.scan_line = std::clamp(cfg.scan_line, 1, 128);
    if (cfg.timestamp_unit < SEC || cfg.timestamp_unit > NS)
        cfg.timestamp_unit = US;
    cfg.scan_rate = std::max(1, cfg.scan_rate);
    cfg.point_filter_num = std::max(1, cfg.point_filter_num);
    cfg.mapping_blind = std::max(0.0, finite_or(cfg.mapping_blind, 2.0));
    cfg.localization_blind = std::max(0.0, finite_or(cfg.localization_blind, 0.3));
    cfg.blind = cfg.localization_blind;
    cfg.blind_z_min = finite_or(cfg.blind_z_min, -1.0e9);
    cfg.blind_z_max = finite_or(cfg.blind_z_max, 1.0e9);
    if (cfg.blind_z_min > cfg.blind_z_max)
        std::swap(cfg.blind_z_min, cfg.blind_z_max);
    cfg.det_range = std::max(1.0, finite_or(cfg.det_range, 100.0));
    if (std::isfinite(cfg.max_range) && cfg.max_range > 0.0)
        cfg.det_range = std::max(1.0, cfg.max_range);
    cfg.max_height = std::max(0.0, finite_or(cfg.max_height, 100.0));
    cfg.scan_context_voxel_leaf = std::max(0.0, finite_or(cfg.scan_context_voxel_leaf, 0.25));
    cfg.scan_context_seed_xy_offset = std::max(0.0, finite_or(cfg.scan_context_seed_xy_offset, 0.5));
    cfg.scan_context.num_rings = std::max(1, cfg.scan_context.num_rings);
    cfg.scan_context.num_sectors = std::max(4, cfg.scan_context.num_sectors);
    cfg.scan_context.max_radius = std::max(1.0, finite_or(cfg.scan_context.max_radius, 80.0));
    cfg.scan_context.dual_z_split_height = finite_or(cfg.scan_context.dual_z_split_height, 2.5);
    cfg.scan_context.origin_height_from_ground = std::max(
        0.0, finite_or(cfg.scan_context.origin_height_from_ground, 0.0));
    cfg.scan_context.dual_z_low_weight = std::max(0.0, finite_or(cfg.scan_context.dual_z_low_weight, 0.3));
    cfg.scan_context.dual_z_high_weight = std::max(0.0, finite_or(cfg.scan_context.dual_z_high_weight, 0.7));
    if (cfg.scan_context.dual_z_low_weight + cfg.scan_context.dual_z_high_weight <= 1e-12)
    {
        cfg.scan_context.dual_z_low_weight = 0.3;
        cfg.scan_context.dual_z_high_weight = 0.7;
    }
    cfg.scan_context.min_joint_rings = std::clamp(
        cfg.scan_context.min_joint_rings, 1, cfg.scan_context.num_rings);
    cfg.scan_context.retrieval_height_offset = std::max(
        0.0, finite_or(cfg.scan_context.retrieval_height_offset, 0.1));
    cfg.scan_context.sector_support_exponent = std::max(
        0.0, finite_or(cfg.scan_context.sector_support_exponent, 0.5));
    cfg.scan_context.vertical_boundary_margin =
        std::max(0.0, finite_or(cfg.scan_context.vertical_boundary_margin, 0.1));
    cfg.scan_context.vertical_correction_min =
        finite_or(cfg.scan_context.vertical_correction_min, -1.5);
    cfg.scan_context.vertical_correction_max =
        finite_or(cfg.scan_context.vertical_correction_max, 1.5);
    if (cfg.scan_context.vertical_correction_min > cfg.scan_context.vertical_correction_max)
        std::swap(cfg.scan_context.vertical_correction_min, cfg.scan_context.vertical_correction_max);
    cfg.scan_context.vertical_stable_fraction = std::clamp(
        finite_or(cfg.scan_context.vertical_stable_fraction, 0.5), 1e-3, 1.0);
    cfg.scan_context.candidate_top_k = std::max(1, cfg.scan_context.candidate_top_k);
    cfg.scan_context.yaw_top_k = std::max(1, cfg.scan_context.yaw_top_k);
    cfg.scan_context.distance_thresh = std::max(1e-6, finite_or(cfg.scan_context.distance_thresh, 0.5));
    return cfg;
}

Preprocess make_preprocessor(const RelocConfig &cfg)
{
    Preprocess preprocess;
    preprocess.time_unit = cfg.timestamp_unit;
    preprocess.N_SCANS = cfg.scan_line;
    preprocess.SCAN_RATE = cfg.scan_rate;
    preprocess.set(
        cfg.feature_extract_enable,
        cfg.lidar_type,
        cfg.blind,
        cfg.det_range,
        cfg.max_height,
        cfg.point_filter_num);
    preprocess.set_tag_filter_mode(parse_tag_filter_mode(cfg.tag_filter_mode));
    preprocess.set_blind_filter(
        parse_blind_filter_shape(cfg.blind_filter_shape),
        cfg.blind_z_min,
        cfg.blind_z_max);
    return preprocess;
}

void print_usage()
{
    std::cout
        << "Usage: ros2 run fast_lio offline_relocalization_exporter [options]\n"
        << "  --bag PATH             default: ${UPDOWN_SC_ROOT}/anyverse/rosbag/nav_debug_bag_2\n"
        << "  --config PATH          default: " << ROOT_DIR << "config/mid360.yaml\n"
        << "  --bag-topic TOPIC      PointCloud2 topic inside the bag; default: common.lid_topic from config\n"
        << "  --topic TOPIC          Alias for --bag-topic\n"
        << "  --map PATH             Override map_file_path from config\n"
        << "  --scan-context-db PATH Override Scan Context database path from config\n"
        << "  --input-is-undistorted Read XYZ[I] body-frame clouds directly; skip raw LiDAR preprocessing\n"
        << "  --output-dir DIR       default: " << ROOT_DIR << "Log/relocalization_windows\n"
        << "  --stride SEC           default: 1.0\n"
        << "  --sample SEC           default: 0.2\n"
        << "  --one-frame-per-window Use only the first PointCloud2 scan in each stride window (default)\n"
        << "  --accumulate-window    Accumulate PointCloud2 scans for --sample seconds\n"
        << "  --window INDEX         Evaluate only this window; repeat to select multiple windows\n"
        << "  --no-window-pcd        Save CSV, summary PNG, and best PCD only\n"
        << "  --truth-csv PATH       Optional window,x,y or window,map_x,map_y positions for PNG overlay\n"
        << "  --gravity-topic TOPIC  Vector3Stamped physical up direction recorded with the body cloud\n"
        << "  --gravity-csv PATH     Optional stamp,up_x,up_y,up_z CSV alternative to the bag topic\n"
        << "  --no-summary-png       Do not generate the default top-down PNG report\n";
}

double parse_double_arg(const char *name, const std::string &value)
{
    std::size_t parsed = 0;
    double result = 0.0;
    try
    {
        result = std::stod(value, &parsed);
    }
    catch (const std::exception &)
    {
        throw std::runtime_error(std::string("invalid numeric value for ") + name + ": " + value);
    }

    if (parsed != value.size() || !std::isfinite(result))
        throw std::runtime_error(std::string("invalid numeric value for ") + name + ": " + value);
    return result;
}

int parse_nonnegative_int_arg(const char *name, const std::string &value)
{
    std::size_t parsed = 0;
    long long result = 0;
    try
    {
        result = std::stoll(value, &parsed);
    }
    catch (const std::exception &)
    {
        throw std::runtime_error(std::string("invalid integer value for ") + name + ": " + value);
    }

    if (parsed != value.size() || result < 0 || result >= static_cast<long long>(kMaxWindowSlots))
        throw std::runtime_error(std::string("invalid window index for ") + name + ": " + value);
    return static_cast<int>(result);
}

bool parse_args(int argc, char **argv, Options &options)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto need_value = [&](const char *name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };

        if (arg == "--bag")
            options.bag_path = need_value("--bag");
        else if (arg == "--config")
            options.config_path = need_value("--config");
        else if (arg == "--topic" || arg == "--bag-topic")
            options.bag_topic = need_value(arg.c_str());
        else if (arg == "--map")
            options.map_path = need_value("--map");
        else if (arg == "--scan-context-db")
            options.scan_context_database_path = need_value("--scan-context-db");
        else if (arg == "--input-is-undistorted")
            options.input_is_undistorted = true;
        else if (arg == "--output-dir")
            options.output_dir = need_value("--output-dir");
        else if (arg == "--stride")
            options.stride_s = parse_double_arg("--stride", need_value("--stride"));
        else if (arg == "--sample")
            options.sample_s = parse_double_arg("--sample", need_value("--sample"));
        else if (arg == "--one-frame-per-window")
            options.one_frame_per_window = true;
        else if (arg == "--accumulate-window")
            options.one_frame_per_window = false;
        else if (arg == "--window")
            options.selected_windows.push_back(
                parse_nonnegative_int_arg("--window", need_value("--window")));
        else if (arg == "--no-window-pcd")
            options.save_window_pcds = false;
        else if (arg == "--truth-csv")
            options.truth_csv = need_value("--truth-csv");
        else if (arg == "--gravity-topic")
            options.gravity_topic = need_value("--gravity-topic");
        else if (arg == "--gravity-csv")
            options.gravity_csv = need_value("--gravity-csv");
        else if (arg == "--gravity-odometry-csv")
            throw std::runtime_error(
                "--gravity-odometry-csv is no longer valid: /Odometry includes map alignment. "
                "Record /scan_context_gravity_up or use --gravity-csv instead.");
        else if (arg == "--no-summary-png")
            options.save_summary_png = false;
        else if (arg == "--help" || arg == "-h")
        {
            print_usage();
            return false;
        }
        else
        {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.stride_s <= 0.0 || options.sample_s <= 0.0 || options.sample_s > options.stride_s)
        throw std::runtime_error("--sample must be > 0 and <= --stride");
    std::sort(options.selected_windows.begin(), options.selected_windows.end());
    options.selected_windows.erase(
        std::unique(options.selected_windows.begin(), options.selected_windows.end()),
        options.selected_windows.end());
    return true;
}

PointCloudXYZI::Ptr downsample_cloud(const PointCloudXYZI::Ptr &cloud, double leaf)
{
    PointCloudXYZI::Ptr finite_cloud(new PointCloudXYZI());
    if (cloud)
    {
        finite_cloud->reserve(cloud->size());
        for (const auto &point : cloud->points)
        {
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
                finite_cloud->push_back(point);
        }
    }
    finite_cloud->width = finite_cloud->size();
    finite_cloud->height = 1;
    finite_cloud->is_dense = true;

    if (finite_cloud->empty() || !std::isfinite(leaf) || leaf <= 1e-3)
        return finite_cloud;

    PointCloudXYZI::Ptr filtered(new PointCloudXYZI());
    pcl::VoxelGrid<PointType> voxel_filter;
    voxel_filter.setLeafSize(static_cast<float>(leaf), static_cast<float>(leaf), static_cast<float>(leaf));
    voxel_filter.setInputCloud(finite_cloud);
    voxel_filter.filter(*filtered);
    return filtered;
}

bool valid_scan_context_input_point(const PointType &point, const RelocConfig &cfg)
{
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        return false;

    const double blind = std::max(0.0, cfg.mapping_blind);
    if (blind <= 1e-12)
        return true;

    const double x = static_cast<double>(point.x);
    const double y = static_cast<double>(point.y);
    const double z = static_cast<double>(point.z);
    const double xy_sq = x * x + y * y;
    const double blind_sq = blind * blind;
    if (parse_blind_filter_shape(cfg.blind_filter_shape) == Preprocess::BLIND_FILTER_CYLINDER)
    {
        const bool inside_xy = xy_sq <= blind_sq;
        const bool inside_z = z >= cfg.blind_z_min && z <= cfg.blind_z_max;
        return !(inside_xy && inside_z);
    }

    return xy_sq + z * z > blind_sq;
}

PointCloudXYZI::Ptr filter_scan_context_input_cloud(
    const PointCloudXYZI::Ptr &cloud,
    const RelocConfig &cfg)
{
    PointCloudXYZI::Ptr filtered(new PointCloudXYZI());
    if (!cloud)
        return filtered;

    filtered->reserve(cloud->size());
    for (const auto &point : cloud->points)
    {
        if (!valid_scan_context_input_point(point, cfg))
            continue;
        filtered->push_back(point);
    }
    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = true;
    return filtered;
}

std::vector<Eigen::Matrix4f> build_seeds(
    const RelocConfig &cfg,
    const sc::Database &scan_context_db,
    const PointCloudXYZI::Ptr &source,
    const Eigen::Matrix3d &R_G_B,
    std::vector<sc::Candidate> *candidates_out,
    std::vector<int> *seed_candidate_ranks = nullptr,
    std::vector<int> *seed_candidate_indices = nullptr,
    std::vector<double> *seed_candidate_distances = nullptr,
    std::vector<double> *seed_coarse_vertical_shifts = nullptr,
    std::vector<double> *seed_vertical_shifts = nullptr)
{
    std::vector<Eigen::Matrix4f> seeds;
    if (candidates_out)
        candidates_out->clear();
    if (seed_candidate_ranks)
        seed_candidate_ranks->clear();
    if (seed_candidate_indices)
        seed_candidate_indices->clear();
    if (seed_candidate_distances)
        seed_candidate_distances->clear();
    if (seed_coarse_vertical_shifts)
        seed_coarse_vertical_shifts->clear();
    if (seed_vertical_shifts)
        seed_vertical_shifts->clear();

    if (cfg.scan_context_enable && !scan_context_db.empty())
    {
        PointCloudXYZI::Ptr scan_context_source = filter_scan_context_input_cloud(source, cfg);
        if (scan_context_db.config().gravity_canonicalized)
        {
            scan_context_source.reset(new PointCloudXYZI(
                sc::gravityCanonicalize(*scan_context_source, R_G_B)));
        }
        const PointCloudXYZI::Ptr query_cloud = downsample_cloud(scan_context_source, cfg.scan_context_voxel_leaf);
        if (!query_cloud || query_cloud->empty())
            return seeds;

        sc::Database query_builder(cfg.scan_context);
        const sc::Descriptor query_descriptor =
            query_builder.makeDescriptor(*query_cloud);
        const auto candidates = scan_context_db.queryWithVerticalEstimation(
            query_descriptor, true);
        if (candidates_out)
            *candidates_out = candidates;
        if (candidates.empty())
            return seeds;

        const double xy_offset = cfg.scan_context_seed_xy_offset;
        const std::vector<double> xy_offsets =
            (xy_offset > 1e-3) ? std::vector<double>{-xy_offset, 0.0, xy_offset} : std::vector<double>{0.0};
        const sc::Config &loaded_scan_context_config = scan_context_db.config();
        seeds.reserve(
            candidates.size() *
            static_cast<std::size_t>(std::max(1, loaded_scan_context_config.yaw_top_k)) *
            xy_offsets.size() * xy_offsets.size());
        for (std::size_t candidate_rank = 0; candidate_rank < candidates.size(); ++candidate_rank)
        {
            const auto &candidate = candidates[candidate_rank];
            std::vector<sc::YawMatch> yaw_matches = candidate.yaw_matches;
            if (yaw_matches.empty())
            {
                sc::YawMatch yaw_match;
                yaw_match.distance = candidate.distance;
                yaw_match.sector_shift = candidate.sector_shift;
                yaw_match.yaw_shift_rad = candidate.yaw_shift_rad;
                yaw_match.coarse_vertical_shift = candidate.coarse_vertical_shift;
                yaw_match.vertical_shift = candidate.vertical_shift;
                yaw_matches.push_back(yaw_match);
            }

            for (const auto &yaw_match : yaw_matches)
            {
                const double seed_yaw = sc::makeCandidateSeedYaw(
                    candidate.pose.canonical_yaw, yaw_match.yaw_shift_rad);
                const Eigen::Matrix3d seed_rotation =
                    Eigen::AngleAxisd(seed_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R_G_B;
                for (const double dx : xy_offsets)
                {
                    for (const double dy : xy_offsets)
                    {
                        seeds.push_back(prior_icp::makeSeedTransform(
                            candidate.pose.x + dx, candidate.pose.y + dy,
                            candidate.pose.z + yaw_match.vertical_shift,
                            seed_rotation));
                        if (seed_candidate_ranks)
                            seed_candidate_ranks->push_back(static_cast<int>(candidate_rank) + 1);
                        if (seed_candidate_indices)
                            seed_candidate_indices->push_back(candidate.index);
                        if (seed_candidate_distances)
                            seed_candidate_distances->push_back(yaw_match.distance);
                        if (seed_coarse_vertical_shifts)
                            seed_coarse_vertical_shifts->push_back(yaw_match.coarse_vertical_shift);
                        if (seed_vertical_shifts)
                            seed_vertical_shifts->push_back(yaw_match.vertical_shift);
                    }
                }
            }
        }
        return seeds;
    }

    auto make_seed = [&](double dx, double dy, double yaw_offset_deg) {
        const double yaw = (cfg.initial_guess_yaw_deg + yaw_offset_deg) * M_PI / 180.0;
        const Eigen::Matrix3d seed_rotation =
            Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R_G_B;
        return prior_icp::makeSeedTransform(
            cfg.initial_guess_x + dx, cfg.initial_guess_y + dy, 0.0, seed_rotation);
    };

    if (!cfg.multi_seed_enable)
    {
        seeds.push_back(make_seed(0.0, 0.0, 0.0));
        if (seed_candidate_ranks)
            seed_candidate_ranks->push_back(-1);
        if (seed_candidate_indices)
            seed_candidate_indices->push_back(-1);
        if (seed_candidate_distances)
            seed_candidate_distances->push_back(std::numeric_limits<double>::infinity());
        if (seed_coarse_vertical_shifts)
            seed_coarse_vertical_shifts->push_back(0.0);
        if (seed_vertical_shifts)
            seed_vertical_shifts->push_back(0.0);
        return seeds;
    }

    const double xy_step = std::max(0.1, cfg.seed_xy_step);
    const double yaw_step = std::max(1.0, cfg.seed_yaw_step_deg);
    for (double yaw = -cfg.seed_yaw_range_deg; yaw <= cfg.seed_yaw_range_deg + 1e-6; yaw += yaw_step)
    {
        for (double dx = -cfg.seed_xy_range; dx <= cfg.seed_xy_range + 1e-6; dx += xy_step)
        {
            for (double dy = -cfg.seed_xy_range; dy <= cfg.seed_xy_range + 1e-6; dy += xy_step)
            {
                seeds.push_back(make_seed(dx, dy, yaw));
                if (seed_candidate_ranks)
                    seed_candidate_ranks->push_back(-1);
                if (seed_candidate_indices)
                    seed_candidate_indices->push_back(-1);
                if (seed_candidate_distances)
                    seed_candidate_distances->push_back(std::numeric_limits<double>::infinity());
                if (seed_coarse_vertical_shifts)
                    seed_coarse_vertical_shifts->push_back(0.0);
                if (seed_vertical_shifts)
                    seed_vertical_shifts->push_back(0.0);
            }
        }
    }
    seeds.push_back(make_seed(0.0, 0.0, 0.0));
    if (seed_candidate_ranks)
        seed_candidate_ranks->push_back(-1);
    if (seed_candidate_indices)
        seed_candidate_indices->push_back(-1);
    if (seed_candidate_distances)
        seed_candidate_distances->push_back(std::numeric_limits<double>::infinity());
    if (seed_coarse_vertical_shifts)
        seed_coarse_vertical_shifts->push_back(0.0);
    if (seed_vertical_shifts)
        seed_vertical_shifts->push_back(0.0);
    return seeds;
}

WindowResult relocalize_window(
    const RelocConfig &cfg,
    const sc::Database &scan_context_db,
    const PointCloudXYZI::Ptr &map_coarse,
    const PointCloudXYZI::Ptr &map_fine,
    const PointCloudXYZI::Ptr &source,
    const Eigen::Matrix3d &R_G_B,
    int window_index,
    double start_s,
    double end_s)
{
    WindowResult result;
    result.window_index = window_index;
    result.start_s = start_s;
    result.end_s = end_s;
    result.source_raw = source;
    result.raw_points = source ? source->size() : 0;
    const auto total_start = std::chrono::steady_clock::now();
    auto elapsed_ms_since = [](const auto &start) {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
            .count();
    };
    auto finish = [&]() {
        result.total_ms = elapsed_ms_since(total_start);
        return result;
    };

    if (!source || static_cast<int>(source->size()) < cfg.icp_min_points)
    {
        result.reason = "too_few_raw_points";
        return finish();
    }

    const auto downsample_start = std::chrono::steady_clock::now();
    const PointCloudXYZI::Ptr source_coarse = downsample_cloud(source, cfg.voxel_leaf);
    const PointCloudXYZI::Ptr source_fine = downsample_cloud(source, cfg.voxel_leaf_fine);
    result.downsample_ms = elapsed_ms_since(downsample_start);
    result.source_coarse_points = source_coarse ? source_coarse->size() : 0;
    result.source_fine_points = source_fine ? source_fine->size() : 0;

    if (result.source_coarse_points < 50 || result.source_fine_points < 50)
    {
        result.reason = "too_few_downsampled_points";
        return finish();
    }

    std::vector<sc::Candidate> candidates;
    std::vector<int> seed_candidate_ranks;
    std::vector<int> seed_candidate_indices;
    std::vector<double> seed_candidate_distances;
    std::vector<double> seed_coarse_vertical_shifts;
    std::vector<double> seed_vertical_shifts;
    const auto scan_context_start = std::chrono::steady_clock::now();
    const std::vector<Eigen::Matrix4f> seeds = build_seeds(
        cfg, scan_context_db, source, R_G_B, &candidates,
        &seed_candidate_ranks, &seed_candidate_indices, &seed_candidate_distances,
        &seed_coarse_vertical_shifts, &seed_vertical_shifts);
    result.scan_context_candidates = candidates;
    result.scan_context_ms = elapsed_ms_since(scan_context_start);
    result.seeds = seeds.size();
    if (seeds.empty())
    {
        result.reason = "no_initial_seeds";
        return finish();
    }

    std::vector<int> all_seed_indices;
    all_seed_indices.reserve(seeds.size());
    for (int i = 0; i < static_cast<int>(seeds.size()); ++i)
        all_seed_indices.push_back(i);

    prior_icp::Config icp_config;
    icp_config.max_iterations = cfg.icp_max_iterations;
    icp_config.max_corr_dist = cfg.icp_max_corr_dist;
    icp_config.min_overlap_ratio = cfg.icp_min_overlap_ratio;

    int coarse_converged = 0;
    const auto coarse_start = std::chrono::steady_clock::now();
    const auto coarse_results = prior_icp::runStage(
        icp_config, source_coarse, map_coarse, seeds, all_seed_indices, coarse_converged, result.coarse_valid);
    result.coarse_icp_ms = elapsed_ms_since(coarse_start);
    if (coarse_results.empty())
    {
        result.reason = coarse_converged > 0 ? "coarse_overlap_rejected" : "coarse_not_converged";
        return finish();
    }

    const int refine_count = std::min(cfg.icp_refine_top_k, static_cast<int>(coarse_results.size()));
    std::vector<int> refine_seed_indices;
    refine_seed_indices.reserve(refine_count);
    for (int i = 0; i < refine_count; ++i)
        refine_seed_indices.push_back(coarse_results[i].seed_index);

    int fine_converged = 0;
    const auto fine_start = std::chrono::steady_clock::now();
    const auto fine_results = prior_icp::runStage(
        icp_config, source_fine, map_fine, seeds, refine_seed_indices, fine_converged, result.fine_valid);
    result.fine_icp_ms = elapsed_ms_since(fine_start);
    if (fine_results.empty())
    {
        result.reason = fine_converged > 0 ? "fine_overlap_rejected" : "fine_not_converged";
        return finish();
    }

    result.best = fine_results.front();
    if (result.best.seed_index >= 0 &&
        result.best.seed_index < static_cast<int>(seed_candidate_ranks.size()))
    {
        result.best_candidate_rank = seed_candidate_ranks[result.best.seed_index];
        result.best_candidate_index = seed_candidate_indices[result.best.seed_index];
        result.best_candidate_distance = seed_candidate_distances[result.best.seed_index];
        result.best_coarse_vertical_shift =
            seed_coarse_vertical_shifts[result.best.seed_index];
        result.best_vertical_shift = seed_vertical_shifts[result.best.seed_index];
    }
    pcl::transformPointCloud(*source_fine, *result.registered_source, result.best.transform);
    if (result.best.fitness > cfg.icp_fitness_thresh)
    {
        result.reason = "fitness_too_high";
        return finish();
    }

    result.success = true;
    result.reason = "success";
    return finish();
}

bool pointcloud2_to_pcl(
    const sensor_msgs::msg::PointCloud2 &msg,
    Preprocess &preprocessor,
    bool input_is_undistorted,
    PointCloudXYZI &out)
{
    if (input_is_undistorted)
    {
        PointCloudXYZI converted;
        pcl::fromROSMsg(msg, converted);
        out.clear();
        out.reserve(converted.size());
        for (const auto &point : converted.points)
        {
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
                out.push_back(point);
        }
        out.width = out.size();
        out.height = 1;
        out.is_dense = true;
        return !out.empty();
    }

    PointCloudXYZI::Ptr processed(new PointCloudXYZI());
    auto msg_copy = std::make_unique<sensor_msgs::msg::PointCloud2>(msg);
    preprocessor.process(msg_copy, processed);
    out = processed ? *processed : PointCloudXYZI();
    return !out.empty();
}

int64_t pointcloud_time_ns(const sensor_msgs::msg::PointCloud2 &msg, int64_t recv_timestamp_ns)
{
    const int64_t header_stamp_ns = rclcpp::Time(msg.header.stamp).nanoseconds();
    if (header_stamp_ns > 0)
        return header_stamp_ns;
    return recv_timestamp_ns;
}

struct TimedUpDirection
{
    double stamp = 0.0;
    Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
};

std::vector<std::string> split_csv_row(const std::string &line)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ','))
        fields.push_back(field);
    return fields;
}

std::vector<TimedUpDirection> load_gravity_csv(const std::string &path)
{
    std::vector<TimedUpDirection> samples;
    if (path.empty())
        return samples;

    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("failed to open gravity CSV: " + path);

    std::string line;
    if (!std::getline(input, line))
        throw std::runtime_error("gravity CSV is empty: " + path);
    const auto header = split_csv_row(line);
    auto column = [&header](const std::string &name) {
        const auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end())
            throw std::runtime_error("gravity CSV missing column: " + name);
        return static_cast<std::size_t>(std::distance(header.begin(), it));
    };
    const std::size_t stamp_col = column("stamp");
    const std::size_t up_x_col = column("up_x");
    const std::size_t up_y_col = column("up_y");
    const std::size_t up_z_col = column("up_z");
    const std::size_t max_col = std::max({stamp_col, up_x_col, up_y_col, up_z_col});

    while (std::getline(input, line))
    {
        if (line.empty())
            continue;
        const auto fields = split_csv_row(line);
        if (fields.size() <= max_col)
            continue;
        try
        {
            TimedUpDirection sample;
            sample.stamp = std::stod(fields[stamp_col]);
            sample.up << std::stod(fields[up_x_col]),
                         std::stod(fields[up_y_col]),
                         std::stod(fields[up_z_col]);
            if (!std::isfinite(sample.stamp) || !sample.up.allFinite() || sample.up.squaredNorm() < 1e-12)
                continue;
            sample.up.normalize();
            samples.push_back(sample);
        }
        catch (const std::exception &)
        {
            continue;
        }
    }
    std::sort(samples.begin(), samples.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.stamp < rhs.stamp;
    });
    if (samples.empty())
        throw std::runtime_error("gravity CSV contains no valid vectors: " + path);
    return samples;
}

std::vector<TimedUpDirection> load_gravity_bag_topic(const Options &options)
{
    rosbag2_cpp::Reader reader;
    rosbag2_storage::StorageOptions storage_options;
    storage_options.uri = options.bag_path;
    storage_options.storage_id = detect_storage_id(options.bag_path);
    rosbag2_cpp::ConverterOptions converter_options{"cdr", "cdr"};
    reader.open(storage_options, converter_options);

    rclcpp::Serialization<geometry_msgs::msg::Vector3Stamped> serialization;
    std::vector<TimedUpDirection> samples;
    while (reader.has_next())
    {
        const auto bag_message = reader.read_next();
        if (!bag_message || bag_message->topic_name != options.gravity_topic ||
            !bag_message->serialized_data)
            continue;
        geometry_msgs::msg::Vector3Stamped msg;
        try
        {
            rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
            serialization.deserialize_message(&serialized_msg, &msg);
        }
        catch (const std::exception &)
        {
            continue;
        }
        TimedUpDirection sample;
        const int64_t header_ns = rclcpp::Time(msg.header.stamp).nanoseconds();
        sample.stamp = static_cast<double>(
            header_ns > 0 ? header_ns : bag_message->recv_timestamp) * 1e-9;
        sample.up << msg.vector.x, msg.vector.y, msg.vector.z;
        if (!std::isfinite(sample.stamp) || !sample.up.allFinite() ||
            sample.up.squaredNorm() < 1e-12)
            continue;
        sample.up.normalize();
        samples.push_back(sample);
    }
    std::sort(samples.begin(), samples.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.stamp < rhs.stamp;
    });
    if (samples.empty())
        throw std::runtime_error(
            "bag contains no valid physical gravity messages on topic: " +
            options.gravity_topic);
    return samples;
}

bool nearest_up_direction(const std::vector<TimedUpDirection> &samples,
                          double stamp, double tolerance_s,
                          Eigen::Vector3d &up)
{
    if (samples.empty() || !std::isfinite(stamp))
        return false;
    auto it = std::lower_bound(
        samples.begin(), samples.end(), stamp,
        [](const TimedUpDirection &sample, double value) { return sample.stamp < value; });
    const TimedUpDirection *best = nullptr;
    if (it != samples.end())
        best = &*it;
    if (it != samples.begin())
    {
        const auto &previous = *std::prev(it);
        if (!best || std::abs(previous.stamp - stamp) < std::abs(best->stamp - stamp))
            best = &previous;
    }
    if (!best || std::abs(best->stamp - stamp) > tolerance_s)
        return false;
    up = best->up;
    return true;
}

std::vector<PointCloudXYZI::Ptr> read_sampled_windows(
    const Options &options,
    const RelocConfig &cfg,
    std::vector<double> &window_starts,
    std::vector<Eigen::Matrix3d> &window_gravity_rotations)
{
    rosbag2_cpp::Reader reader;
    rosbag2_storage::StorageOptions storage_options;
    storage_options.uri = options.bag_path;
    storage_options.storage_id = detect_storage_id(options.bag_path);
    rosbag2_cpp::ConverterOptions converter_options{"cdr", "cdr"};
    reader.open(storage_options, converter_options);

    rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serialization;
    Preprocess preprocessor = make_preprocessor(cfg);
    const auto gravity_samples = cfg.scan_context.gravity_canonicalized
        ? (options.gravity_csv.empty()
               ? load_gravity_bag_topic(options)
               : load_gravity_csv(options.gravity_csv))
        : std::vector<TimedUpDirection>{};
    std::vector<PointCloudXYZI::Ptr> windows;
    int64_t first_stamp_ns = -1;
    int bad_message_count = 0;

    while (reader.has_next())
    {
        const auto bag_message = reader.read_next();
        if (!bag_message)
        {
            if (bad_message_count < 5)
                std::cerr << "warning: skipped empty bag message while reading topic " << cfg.topic << '\n';
            ++bad_message_count;
            continue;
        }
        if (bag_message->topic_name != cfg.topic)
            continue;
        if (!bag_message->serialized_data)
        {
            if (bad_message_count < 5)
                std::cerr << "warning: skipped empty serialized bag message on topic " << cfg.topic << '\n';
            ++bad_message_count;
            continue;
        }

        sensor_msgs::msg::PointCloud2 msg;
        try
        {
            rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
            serialization.deserialize_message(&serialized_msg, &msg);
        }
        catch (const std::exception &e)
        {
            if (bad_message_count < 5)
            {
                std::cerr << "warning: failed to deserialize PointCloud2 from topic "
                          << cfg.topic << ": " << e.what() << '\n';
            }
            ++bad_message_count;
            continue;
        }

        const int64_t stamp_ns = pointcloud_time_ns(msg, bag_message->recv_timestamp);
        if (first_stamp_ns < 0)
            first_stamp_ns = stamp_ns;
        const double rel_s = static_cast<double>(stamp_ns - first_stamp_ns) * 1e-9;
        if (rel_s < 0.0)
            continue;

        const double window_index_raw = std::floor(rel_s / options.stride_s);
        if (window_index_raw < 0.0 ||
            window_index_raw >= static_cast<double>(kMaxWindowSlots))
        {
            std::ostringstream oss;
            oss << "too many offline relocalization windows (index=" << window_index_raw
                << ", max=" << kMaxWindowSlots
                << "); increase --stride or split the bag";
            throw std::runtime_error(oss.str());
        }
        const std::size_t window_index = static_cast<std::size_t>(window_index_raw);
        if (!options.selected_windows.empty() &&
            !std::binary_search(
                options.selected_windows.begin(), options.selected_windows.end(),
                static_cast<int>(window_index)))
        {
            continue;
        }
        const double window_start = static_cast<double>(window_index) * options.stride_s;
        if (!options.one_frame_per_window &&
            (rel_s < window_start - 1e-9 ||
             rel_s > window_start + options.sample_s + 1e-6))
            continue;

        if (window_index >= windows.size())
        {
            windows.resize(window_index + 1);
            window_starts.resize(window_index + 1);
            window_gravity_rotations.resize(window_index + 1, Eigen::Matrix3d::Identity());
        }
        if (!windows[window_index])
        {
            windows[window_index].reset(new PointCloudXYZI());
            window_starts[window_index] = window_start;
        }
        else if (options.one_frame_per_window && !windows[window_index]->empty())
        {
            continue;
        }

        PointCloudXYZI cloud;
        if (!pointcloud2_to_pcl(msg, preprocessor, options.input_is_undistorted, cloud))
            continue;
        if (cfg.scan_context.gravity_canonicalized)
        {
            Eigen::Vector3d up;
            const double stamp_s = static_cast<double>(stamp_ns) * 1e-9;
            const double tolerance_s =
                std::max(0.02, 0.75 / static_cast<double>(std::max(1, cfg.scan_rate)));
            if (!nearest_up_direction(gravity_samples, stamp_s, tolerance_s, up))
            {
                if (bad_message_count < 5)
                    std::cerr << "warning: no synchronized gravity pose for cloud stamp " << stamp_s << '\n';
                ++bad_message_count;
                continue;
            }
            Eigen::Matrix3d R_G_B;
            if (!sc::makeGravityCanonicalRotation(up, R_G_B))
                continue;
            if (windows[window_index]->empty())
                window_gravity_rotations[window_index] = R_G_B;
        }
        *windows[window_index] += cloud;
    }

    return windows;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr make_overlay_cloud(
    const PointCloudXYZI::Ptr &map,
    const PointCloudXYZI::Ptr &registered_source)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr overlay(new pcl::PointCloud<pcl::PointXYZRGB>());
    overlay->reserve((map ? map->size() : 0) + (registered_source ? registered_source->size() : 0));

    if (map)
    {
        for (const auto &point : map->points)
        {
            pcl::PointXYZRGB out;
            out.x = point.x;
            out.y = point.y;
            out.z = point.z;
            out.r = 145;
            out.g = 145;
            out.b = 145;
            overlay->push_back(out);
        }
    }

    if (registered_source)
    {
        for (const auto &point : registered_source->points)
        {
            pcl::PointXYZRGB out;
            out.x = point.x;
            out.y = point.y;
            out.z = point.z;
            out.r = 0;
            out.g = 255;
            out.b = 70;
            overlay->push_back(out);
        }
    }

    overlay->width = overlay->size();
    overlay->height = 1;
    overlay->is_dense = true;
    return overlay;
}

void replace_file_with_tmp(const fs::path &tmp_path, const fs::path &output_path)
{
    std::error_code rename_ec;
    fs::rename(tmp_path, output_path, rename_ec);
    if (rename_ec)
    {
        std::error_code remove_ec;
        fs::remove(output_path, remove_ec);
        rename_ec.clear();
        fs::rename(tmp_path, output_path, rename_ec);
    }
    if (rename_ec)
        throw std::runtime_error(
            "failed to move temporary file into place: " + output_path.string() + " (" + rename_ec.message() + ")");
}

template <typename CloudT>
void save_pcd_binary_atomic(const fs::path &path, const CloudT &cloud)
{
    if (!path.parent_path().empty())
    {
        std::error_code dir_ec;
        fs::create_directories(path.parent_path(), dir_ec);
        if (dir_ec)
            throw std::runtime_error("failed to create PCD directory: " + path.parent_path().string() + " (" + dir_ec.message() + ")");
    }

    const fs::path tmp_path = path.string() + ".tmp";
    auto cleanup_tmp = [&tmp_path]() {
        std::error_code cleanup_ec;
        fs::remove(tmp_path, cleanup_ec);
    };

    try
    {
        pcl::PCDWriter writer;
        const int ret = writer.writeBinary(tmp_path.string(), cloud);
        if (ret != 0)
        {
            cleanup_tmp();
            throw std::runtime_error("PCDWriter::writeBinary returned " + std::to_string(ret));
        }
        replace_file_with_tmp(tmp_path, path);
    }
    catch (...)
    {
        cleanup_tmp();
        throw;
    }
}

void write_csv(const fs::path &path, const std::vector<WindowResult> &results)
{
    if (!path.parent_path().empty())
    {
        std::error_code dir_ec;
        fs::create_directories(path.parent_path(), dir_ec);
        if (dir_ec)
            throw std::runtime_error("failed to create CSV directory: " + path.parent_path().string() + " (" + dir_ec.message() + ")");
    }

    const fs::path tmp_path = path.string() + ".tmp";
    auto cleanup_tmp = [&tmp_path]() {
        std::error_code cleanup_ec;
        fs::remove(tmp_path, cleanup_ec);
    };

    {
        std::ofstream out(tmp_path);
        if (!out)
            throw std::runtime_error("failed to open temporary CSV for writing: " + tmp_path.string());

        out << "window,start_s,end_s,success,reason,total_ms,downsample_ms,scan_context_ms,coarse_icp_ms,fine_icp_ms,"
               "fitness,overlap,seeds,coarse_valid,fine_valid,"
               "raw_points,source_coarse_points,source_fine_points,"
               "best_seed,best_candidate_rank,best_candidate_index,best_candidate_distance,"
               "best_coarse_vertical_shift,best_vertical_shift,tx,ty,tz\n";
        out << std::fixed << std::setprecision(6);
        for (const auto &r : results)
        {
            out << r.window_index << ','
                << r.start_s << ','
                << r.end_s << ','
                << (r.success ? "true" : "false") << ','
                << r.reason << ','
                << r.total_ms << ','
                << r.downsample_ms << ','
                << r.scan_context_ms << ','
                << r.coarse_icp_ms << ','
                << r.fine_icp_ms << ','
                << r.best.fitness << ','
                << r.best.overlap << ','
                << r.seeds << ','
                << r.coarse_valid << ','
                << r.fine_valid << ','
                << r.raw_points << ','
                << r.source_coarse_points << ','
                << r.source_fine_points << ','
                << r.best.seed_index << ','
                << r.best_candidate_rank << ','
                << r.best_candidate_index << ','
                << r.best_candidate_distance << ','
                << r.best_coarse_vertical_shift << ','
                << r.best_vertical_shift << ','
                << r.best.transform(0, 3) << ','
                << r.best.transform(1, 3) << ','
                << r.best.transform(2, 3) << '\n';
        }

        out.flush();
        if (!out)
        {
            cleanup_tmp();
            throw std::runtime_error("failed to write CSV: " + tmp_path.string());
        }
    }

    try
    {
        replace_file_with_tmp(tmp_path, path);
    }
    catch (...)
    {
        cleanup_tmp();
        throw;
    }
}

void write_candidate_hypotheses_csv(
    const fs::path &path,
    const std::vector<WindowResult> &results)
{
    if (!path.parent_path().empty())
        fs::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("failed to open candidate CSV for writing: " + path.string());
    out << "window,candidate_rank,candidate_index,candidate_x,candidate_y,candidate_z,"
           "hypothesis_rank,distance,sector_shift,yaw_shift_rad,"
           "coarse_vertical_shift,vertical_shift,seed_z\n";
    out << std::fixed << std::setprecision(9);
    for (const auto &result : results)
    {
        for (std::size_t candidate_rank = 0;
             candidate_rank < result.scan_context_candidates.size();
             ++candidate_rank)
        {
            const auto &candidate = result.scan_context_candidates[candidate_rank];
            for (std::size_t hypothesis_rank = 0;
                 hypothesis_rank < candidate.yaw_matches.size();
                 ++hypothesis_rank)
            {
                const auto &match = candidate.yaw_matches[hypothesis_rank];
                out << result.window_index << ','
                    << candidate_rank + 1 << ','
                    << candidate.index << ','
                    << candidate.pose.x << ','
                    << candidate.pose.y << ','
                    << candidate.pose.z << ','
                    << hypothesis_rank + 1 << ','
                    << match.distance << ','
                    << match.sector_shift << ','
                    << match.yaw_shift_rad << ','
                    << match.coarse_vertical_shift << ','
                    << match.vertical_shift << ','
                    << candidate.pose.z + match.vertical_shift << '\n';
            }
        }
    }
}

void write_scan_context_trajectory_csv(const fs::path &path, const sc::Database &database)
{
    const fs::path tmp_path = path.string() + ".tmp";
    {
        std::ofstream out(tmp_path);
        if (!out)
            throw std::runtime_error("failed to open Scan Context trajectory CSV: " + tmp_path.string());
        out << "index,stamp,x,y,z,roll,pitch,yaw,canonical_yaw\n"
            << std::fixed << std::setprecision(9);
        const auto &entries = database.entries();
        for (std::size_t index = 0; index < entries.size(); ++index)
        {
            const auto &entry = entries[index];
            out << index << ',' << entry.stamp << ','
                << entry.pose.x << ',' << entry.pose.y << ',' << entry.pose.z << ','
                << entry.pose.roll << ',' << entry.pose.pitch << ',' << entry.pose.yaw << ','
                << entry.pose.canonical_yaw << '\n';
        }
        out.flush();
        if (!out)
            throw std::runtime_error("failed to write Scan Context trajectory CSV: " + tmp_path.string());
    }
    try
    {
        replace_file_with_tmp(tmp_path, path);
    }
    catch (...)
    {
        std::error_code cleanup_ec;
        fs::remove(tmp_path, cleanup_ec);
        throw;
    }
}

std::string shell_quote(const std::string &value)
{
    std::string quoted("'");
    for (const char ch : value)
    {
        if (ch == '\'')
            quoted += "'\\''";
        else
            quoted += ch;
    }
    quoted += '\'';
    return quoted;
}

bool generate_summary_png(
    const fs::path &map_path,
    const fs::path &trajectory_path,
    const fs::path &results_path,
    const fs::path &output_dir,
    const std::string &truth_csv)
{
    fs::path script_path;
    try
    {
        script_path = fs::path(ament_index_cpp::get_package_share_directory("fast_lio")) /
                      "scripts" / "plot_offline_relocalization.py";
    }
    catch (const std::exception &)
    {
        script_path = fs::path(ROOT_DIR) / "scripts" / "plot_offline_relocalization.py";
    }
    if (!fs::exists(script_path))
    {
        std::cerr << "Warning: summary PNG script not found: " << script_path << '\n';
        return false;
    }

    std::ostringstream command;
    command << "python3 " << shell_quote(script_path.string())
            << " --map " << shell_quote(map_path.string())
            << " --trajectory " << shell_quote(trajectory_path.string())
            << " --results " << shell_quote(results_path.string())
            << " --output-dir " << shell_quote(output_dir.string());
    if (!truth_csv.empty())
        command << " --truth-csv " << shell_quote(truth_csv);
    const int status = std::system(command.str().c_str());
    if (status != 0)
    {
        std::cerr << "Warning: failed to generate relocalization summary PNGs (status="
                  << status << "). CSV/PCD results remain valid.\n";
        return false;
    }
    return true;
}

std::string window_file_stem(const WindowResult &result)
{
    std::ostringstream oss;
    oss << "window_" << std::setw(4) << std::setfill('0') << result.window_index
        << "_t" << std::setw(6) << std::setfill('0') << static_cast<int>(std::round(result.start_s * 100.0));
    return oss.str();
}

std::size_t save_window_pcds(
    const fs::path &output_dir,
    const PointCloudXYZI::Ptr &map_fine,
    const std::vector<WindowResult> &results)
{
    const fs::path registered_dir = output_dir / "registered_windows";
    const fs::path overlay_dir = output_dir / "overlay_windows";
    std::error_code ec;
    fs::remove_all(registered_dir, ec);
    if (ec)
        throw std::runtime_error("failed to clear " + registered_dir.string() + ": " + ec.message());
    fs::remove_all(overlay_dir, ec);
    if (ec)
        throw std::runtime_error("failed to clear " + overlay_dir.string() + ": " + ec.message());
    fs::create_directories(registered_dir, ec);
    if (ec)
        throw std::runtime_error("failed to create " + registered_dir.string() + ": " + ec.message());
    fs::create_directories(overlay_dir, ec);
    if (ec)
        throw std::runtime_error("failed to create " + overlay_dir.string() + ": " + ec.message());

    std::size_t saved_count = 0;
    for (const auto &result : results)
    {
        if (!result.registered_source || result.registered_source->empty())
            continue;

        const std::string pass_tag = result.success ? "pass" : "fail_" + result.reason;
        const std::string stem = window_file_stem(result) + "_" + pass_tag;
        const fs::path registered_pcd = registered_dir / (stem + "_registered.pcd");
        const fs::path overlay_pcd = overlay_dir / (stem + "_overlay.pcd");

        save_pcd_binary_atomic(registered_pcd, *result.registered_source);
        save_pcd_binary_atomic(overlay_pcd, *make_overlay_cloud(map_fine, result.registered_source));
        ++saved_count;
    }
    return saved_count;
}

}  // namespace

int main(int argc, char **argv)
{
    Options options;
    bool rclcpp_initialized = false;
    try
    {
        if (!parse_args(argc, argv, options))
            return 0;

        rclcpp::init(argc, argv);
        rclcpp_initialized = true;

        if (options.config_path.empty() || !fs::exists(options.config_path))
            throw std::runtime_error("config file does not exist: " + options.config_path);
        if (options.bag_path.empty() || !fs::exists(options.bag_path))
            throw std::runtime_error("bag path does not exist: " + options.bag_path);
        if (options.output_dir.empty())
            throw std::runtime_error("output directory must not be empty");
        if (!options.truth_csv.empty())
        {
            options.truth_csv = fs::absolute(options.truth_csv).lexically_normal().string();
            if (!fs::exists(options.truth_csv))
                throw std::runtime_error("truth CSV does not exist: " + options.truth_csv);
        }
        if (!options.gravity_csv.empty())
        {
            options.gravity_csv =
                fs::absolute(options.gravity_csv).lexically_normal().string();
            if (!fs::exists(options.gravity_csv))
                throw std::runtime_error(
                    "gravity CSV does not exist: " + options.gravity_csv);
        }

        RelocConfig cfg = load_config(options.config_path, options.bag_topic);
        if (!options.map_path.empty())
            cfg.map_file_path = fs::absolute(options.map_path).lexically_normal().string();
        if (!options.scan_context_database_path.empty())
            cfg.scan_context_database_path =
                fs::absolute(options.scan_context_database_path).lexically_normal().string();
        if (cfg.topic.empty())
            throw std::runtime_error("LiDAR topic is empty; set common.lid_topic or pass --bag-topic");
        if (cfg.map_file_path.empty())
            throw std::runtime_error("prior map path is empty; set map_file_path in config");
        if (cfg.scan_context.gravity_canonicalized)
        {
            if (!options.input_is_undistorted)
            {
                throw std::runtime_error(
                    "gravity-canonicalized offline evaluation requires --input-is-undistorted; "
                    "raw PointCloud2 preprocessing does not reproduce FAST-LIO deskew/body coordinates");
            }
            if (!options.one_frame_per_window)
            {
                throw std::runtime_error(
                    "gravity-canonicalized offline evaluation currently requires "
                    "--one-frame-per-window so every source has one reference attitude");
            }
        }

        std::error_code output_dir_ec;
        fs::create_directories(options.output_dir, output_dir_ec);
        if (output_dir_ec)
            throw std::runtime_error("failed to create output directory: " + output_dir_ec.message());

        PointCloudXYZI::Ptr map_raw(new PointCloudXYZI());
        if (pcl::io::loadPCDFile<PointType>(cfg.map_file_path, *map_raw) < 0 || map_raw->empty())
            throw std::runtime_error("failed to load prior map: " + cfg.map_file_path);
        const PointCloudXYZI::Ptr map_coarse = downsample_cloud(map_raw, cfg.voxel_leaf);
        const PointCloudXYZI::Ptr map_fine = downsample_cloud(map_raw, cfg.voxel_leaf_fine);
        if (!map_coarse || map_coarse->empty() || !map_fine || map_fine->empty())
            throw std::runtime_error("prior map became empty after downsampling");

        sc::Database scan_context_db(cfg.scan_context);
        if (cfg.scan_context_enable)
        {
            std::string error;
            if (!scan_context_db.load(cfg.scan_context_database_path, &error))
            {
                throw std::runtime_error(
                    "failed to load Scan Context DB: " + cfg.scan_context_database_path + " (" + error + ")");
            }
            if (scan_context_db.empty())
            {
                throw std::runtime_error(
                    "Scan Context DB is empty: " + cfg.scan_context_database_path);
            }
            if (scan_context_db.config().gravity_canonicalized !=
                cfg.scan_context.gravity_canonicalized)
            {
                throw std::runtime_error(
                    "Scan Context gravity-canonicalization mismatch; regenerate scans.scd "
                    "with the current configuration");
            }
            if (scan_context_db.legacyMasksInferred())
            {
                std::cerr
                    << "Warning: loaded legacy Scan Context V1 database without explicit validity masks; "
                    << "masks were inferred from nonzero descriptor values. Rebuild the map database for exact V2 masks.\n";
            }
            const sc::Config &loaded_scan_context_config = scan_context_db.config();
            std::cout << "Loaded Scan Context DB: path=" << cfg.scan_context_database_path
                      << " entries=" << scan_context_db.size()
                      << " rings=" << loaded_scan_context_config.num_rings
                      << " sectors=" << loaded_scan_context_config.num_sectors
                      << " max_radius=" << loaded_scan_context_config.max_radius
                      << " dual_z=" << (loaded_scan_context_config.dual_z_layer_enable ? "true" : "false")
                      << " map_split="
                      << sc::effectiveDualZSplitHeight(loaded_scan_context_config)
                      << " query_split="
                      << sc::effectiveDualZSplitHeight(cfg.scan_context)
                      << " retrieval_height_offset="
                      << loaded_scan_context_config.retrieval_height_offset
                      << " weights=[" << loaded_scan_context_config.dual_z_low_weight
                      << ',' << loaded_scan_context_config.dual_z_high_weight << ']'
                      << " min_joint_rings=" << loaded_scan_context_config.min_joint_rings
                      << " vertical_boundary_margin="
                      << loaded_scan_context_config.vertical_boundary_margin
                      << " vertical_stable_fraction="
                      << loaded_scan_context_config.vertical_stable_fraction
                      << " gravity_canonicalized="
                      << (loaded_scan_context_config.gravity_canonicalized ? "true" : "false") << "\n";
        }

        std::cout << "Reading bag windows: bag=" << options.bag_path
                  << " topic=" << cfg.topic
                  << " input=" << (options.input_is_undistorted ? "fast_lio_undistorted_body" : "raw_lidar")
                  << " stride=" << options.stride_s
                  << " sample=" << options.sample_s
                  << " one_frame_per_window=" << (options.one_frame_per_window ? "true" : "false");
        if (!options.selected_windows.empty())
        {
            std::cout << " selected_windows=";
            for (std::size_t i = 0; i < options.selected_windows.size(); ++i)
            {
                if (i > 0)
                    std::cout << ',';
                std::cout << options.selected_windows[i];
            }
        }
        std::cout << '\n';
        std::vector<double> window_starts;
        std::vector<Eigen::Matrix3d> window_gravity_rotations;
        const auto windows = read_sampled_windows(
            options, cfg, window_starts, window_gravity_rotations);
        std::cout << "Loaded " << windows.size() << " window slots from bag.\n";
        const std::size_t non_empty_window_count = std::count_if(
            windows.begin(), windows.end(),
            [](const PointCloudXYZI::Ptr &window) { return window && !window->empty(); });
        if (non_empty_window_count == 0)
        {
            throw std::runtime_error(
                "no point cloud windows loaded from bag topic: " + cfg.topic +
                "; PointCloud2 topics in bag: " +
                format_topic_list(read_pointcloud_topics(options.bag_path)));
        }

        std::vector<WindowResult> results;
        results.reserve(windows.size());
        const double evaluated_window_duration =
            options.one_frame_per_window ? 1.0 / static_cast<double>(std::max(1, cfg.scan_rate))
                                         : options.sample_s;
        for (std::size_t i = 0; i < windows.size(); ++i)
        {
            if (!windows[i] || windows[i]->empty())
                continue;
            WindowResult result = relocalize_window(
                cfg, scan_context_db, map_coarse, map_fine, windows[i],
                window_gravity_rotations[i],
                static_cast<int>(i), window_starts[i], window_starts[i] + evaluated_window_duration);
            std::cout << "window=" << result.window_index
                      << " t=[" << std::fixed << std::setprecision(2) << result.start_s
                      << "," << result.end_s << "]"
                      << " success=" << (result.success ? "true" : "false")
                      << " reason=" << result.reason
                      << " total_ms=" << std::setprecision(1) << result.total_ms
                      << " sc_ms=" << result.scan_context_ms
                      << " coarse_ms=" << result.coarse_icp_ms
                      << " fine_ms=" << result.fine_icp_ms
                      << " fitness=" << std::setprecision(6) << result.best.fitness
                      << " overlap=" << std::setprecision(3) << result.best.overlap
                      << " points=" << result.raw_points << '\n';
            results.push_back(std::move(result));
        }

        const fs::path output_dir(options.output_dir);
        const fs::path results_csv = output_dir / "relocalization_windows.csv";
        const fs::path candidate_csv = output_dir / "scan_context_candidate_hypotheses.csv";
        const fs::path trajectory_csv = output_dir / "scan_context_trajectory.csv";
        write_csv(results_csv, results);
        write_candidate_hypotheses_csv(candidate_csv, results);
        write_scan_context_trajectory_csv(trajectory_csv, scan_context_db);
        const std::size_t saved_window_count =
            options.save_window_pcds ? save_window_pcds(output_dir, map_fine, results) : 0;
        const bool summary_png_saved = options.save_summary_png && generate_summary_png(
            cfg.map_file_path, trajectory_csv, results_csv, output_dir, options.truth_csv);

        auto best_it = std::min_element(results.begin(), results.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.success != rhs.success)
                return lhs.success;
            return lhs.best.fitness < rhs.best.fitness;
        });

        if (best_it == results.end() || !best_it->success)
            throw std::runtime_error("no successful relocalization window; see relocalization_windows.csv");
        if (!best_it->registered_source || best_it->registered_source->empty())
            throw std::runtime_error("best relocalization window has no registered source cloud");

        const fs::path best_pcd = output_dir / "relocalization_best_registered.pcd";
        const fs::path overlay_pcd = output_dir / "relocalization_best_overlay.pcd";
        save_pcd_binary_atomic(best_pcd, *best_it->registered_source);
        save_pcd_binary_atomic(overlay_pcd, *make_overlay_cloud(map_fine, best_it->registered_source));

        std::cout << "\nBest relocalization window: index=" << best_it->window_index
                  << " t=[" << std::fixed << std::setprecision(2) << best_it->start_s
                  << "," << best_it->end_s << "]"
                  << " fitness=" << std::setprecision(6) << best_it->best.fitness
                  << " overlap=" << std::setprecision(3) << best_it->best.overlap
                  << " seed=" << best_it->best.seed_index << '\n'
                  << "Saved registered source: " << best_pcd << '\n'
                  << "Saved overlay: " << overlay_pcd << '\n'
                  << "Saved per-window registered/overlay PCD pairs: " << saved_window_count << '\n'
                  << "  " << (output_dir / "registered_windows") << '\n'
                  << "  " << (output_dir / "overlay_windows") << '\n'
                  << "Saved CSV: " << results_csv << '\n'
                  << "Saved Scan Context trajectory: " << trajectory_csv << '\n';
        if (summary_png_saved)
        {
            std::cout << "Saved top-down summaries:\n"
                      << "  " << (output_dir / "relocalization_top_view.png") << '\n'
                      << "  " << (output_dir / "relocalization_top_view_focus.png") << '\n';
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "offline_relocalization_exporter error: " << e.what() << '\n';
        if (rclcpp_initialized)
            rclcpp::shutdown();
        return 1;
    }

    if (rclcpp_initialized)
        rclcpp::shutdown();
    return 0;
}
