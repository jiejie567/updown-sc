#include "scan_context.hpp"

#include <Eigen/Geometry>
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace sc = fast_lio::scan_context;

namespace
{

struct TimedPose
{
    double stamp = 0.0;
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
};

std::vector<TimedPose> loadPoses(const fs::path &path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("failed to open poses: " + path.string());
    std::vector<TimedPose> poses;
    TimedPose pose;
    double qx = 0.0, qy = 0.0, qz = 0.0, qw = 1.0;
    while (input >> pose.stamp
                 >> pose.translation.x() >> pose.translation.y() >> pose.translation.z()
                 >> qx >> qy >> qz >> qw)
    {
        pose.rotation = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
        poses.push_back(pose);
    }
    if (!input.eof() || poses.empty())
        throw std::runtime_error("invalid or empty pose file: " + path.string());
    return poses;
}

std::vector<Eigen::Vector3d> loadGravity(const fs::path &path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("failed to open gravity sidecar: " + path.string());
    std::string line;
    std::getline(input, line);
    std::vector<Eigen::Vector3d> gravity;
    while (std::getline(input, line))
    {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        int index = -1;
        double stamp = 0.0;
        Eigen::Vector3d up;
        if (!(row >> index >> stamp >> up.x() >> up.y() >> up.z()) ||
            index != static_cast<int>(gravity.size()) ||
            !up.allFinite() || up.norm() < 1e-12)
        {
            throw std::runtime_error("invalid gravity row: " + line);
        }
        gravity.push_back(up.normalized());
    }
    return gravity;
}

sc::Config loadConfig(const fs::path &path, bool canonicalize)
{
    const YAML::Node root = YAML::LoadFile(path.string());
    const auto find_scan_context_node = [&root]() -> YAML::Node {
        const YAML::Node direct_node = root["scan_context"];
        if (direct_node)
            return direct_node;
        const YAML::Node wildcard_node = root["/**"];
        if (!wildcard_node)
            return {};
        const YAML::Node ros_parameters = wildcard_node["ros__parameters"];
        if (!ros_parameters)
            return {};
        return ros_parameters["prior_map"]["scan_context"];
    };
    const YAML::Node node = find_scan_context_node();
    if (!node)
        throw std::runtime_error(
            "configuration has no scan_context or prior_map.scan_context section");
    sc::Config config;
    config.num_rings = node["num_rings"].as<int>(config.num_rings);
    config.num_sectors = node["num_sectors"].as<int>(config.num_sectors);
    config.max_radius = node["max_radius"].as<double>(config.max_radius);
    config.dual_z_layer_enable =
        node["dual_z_layer_enable"].as<bool>(config.dual_z_layer_enable);
    config.dual_z_split_height =
        node["dual_z_split_height"].as<double>(config.dual_z_split_height);
    config.dual_z_split_auto =
        node["dual_z_split_auto"].as<bool>(config.dual_z_split_auto);
    config.dual_z_split_auto_min =
        node["dual_z_split_auto_min"].as<double>(
            config.dual_z_split_auto_min);
    config.dual_z_split_auto_max =
        node["dual_z_split_auto_max"].as<double>(
            config.dual_z_split_auto_max);
    config.dual_z_split_auto_bin_size =
        node["dual_z_split_auto_bin_size"].as<double>(
            config.dual_z_split_auto_bin_size);
    config.dual_z_split_auto_histogram_max =
        node["dual_z_split_auto_histogram_max"].as<double>(
            config.dual_z_split_auto_histogram_max);
    config.dual_z_split_auto_min_layer_fraction =
        node["dual_z_split_auto_min_layer_fraction"].as<double>(
            config.dual_z_split_auto_min_layer_fraction);
    config.dual_z_split_auto_min_keyframes =
        node["dual_z_split_auto_min_keyframes"].as<int>(
            config.dual_z_split_auto_min_keyframes);
    config.origin_height_from_ground =
        node["origin_height_from_ground"].as<double>(
            config.origin_height_from_ground);
    config.dual_z_low_weight =
        node["dual_z_low_weight"].as<double>(config.dual_z_low_weight);
    config.dual_z_high_weight =
        node["dual_z_high_weight"].as<double>(config.dual_z_high_weight);
    config.min_joint_rings =
        node["min_joint_rings"].as<int>(config.min_joint_rings);
    config.absent_upper_fallback_max_local_fraction =
        node["absent_upper_fallback_max_local_fraction"].as<double>(
            config.absent_upper_fallback_max_local_fraction);
    config.absent_upper_fallback_radius =
        node["absent_upper_fallback_radius"].as<double>(
            config.absent_upper_fallback_radius);
    config.absent_upper_fallback_min_keyframes =
        node["absent_upper_fallback_min_keyframes"].as<int>(
            config.absent_upper_fallback_min_keyframes);
    config.retrieval_height_offset =
        node["retrieval_height_offset"].as<double>(
            config.retrieval_height_offset);
    config.sector_support_exponent =
        node["sector_support_exponent"].as<double>(
            config.sector_support_exponent);
    config.vertical_boundary_margin =
        node["vertical_boundary_margin"].as<double>(config.vertical_boundary_margin);
    config.vertical_estimation_enable =
        node["vertical_estimation_enable"].as<bool>(config.vertical_estimation_enable);
    config.vertical_correction_min =
        node["vertical_correction_min"].as<double>(config.vertical_correction_min);
    config.vertical_correction_max =
        node["vertical_correction_max"].as<double>(config.vertical_correction_max);
    config.vertical_stable_fraction =
        node["vertical_stable_fraction"].as<double>(config.vertical_stable_fraction);
    config.candidate_top_k =
        node["candidate_top_k"].as<int>(config.candidate_top_k);
    config.yaw_top_k = node["yaw_top_k"].as<int>(config.yaw_top_k);
    config.distance_thresh =
        node["distance_thresh"].as<double>(config.distance_thresh);
    config.gravity_canonicalized = canonicalize;
    return config;
}

sc::PointCloud loadCloud(const fs::path &path)
{
    pcl::PointCloud<pcl::PointXYZI> input;
    if (pcl::io::loadPCDFile(path.string(), input) < 0)
        throw std::runtime_error("failed to load PCD: " + path.string());
    sc::PointCloud output;
    output.reserve(input.size());
    for (const auto &point : input)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.z))
        {
            continue;
        }
        sc::PointType converted;
        converted.x = point.x;
        converted.y = point.y;
        converted.z = point.z;
        converted.intensity = point.intensity;
        output.push_back(converted);
    }
    return output;
}

sc::Pose makePose(const TimedPose &input, const Eigen::Matrix3d &gravity_rotation)
{
    const Eigen::Matrix3d rotation = input.rotation.toRotationMatrix();
    sc::Pose pose;
    pose.x = input.translation.x();
    pose.y = input.translation.y();
    pose.z = input.translation.z();
    pose.roll = std::atan2(rotation(2, 1), rotation(2, 2));
    pose.pitch = std::asin(std::clamp(-rotation(2, 0), -1.0, 1.0));
    pose.yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    const Eigen::Matrix3d map_from_descriptor =
        rotation * gravity_rotation.transpose();
    pose.canonical_yaw = std::atan2(
        map_from_descriptor(1, 0), map_from_descriptor(0, 0));
    return pose;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 4 || argc > 7)
    {
        std::cerr << "Usage: scan_context_rebuild SESSION_DIR OUTPUT.scd "
                     "native|gravity [RUNTIME_PARAMS.yaml [Z_OFFSET_M "
                     "[FIXED_SPLIT_HEIGHT_M]]]\n";
        return 2;
    }
    try
    {
        const fs::path session = fs::absolute(argv[1]);
        const fs::path output = fs::absolute(argv[2]);
        const std::string mode = argv[3];
        if (mode != "native" && mode != "gravity")
            throw std::runtime_error("mode must be native or gravity");
        const bool canonicalize = mode == "gravity";
        const auto poses = loadPoses(session / "optimized_poses_tum.txt");
        const auto gravity = canonicalize
            ? loadGravity(session / "scan_context_gravity.csv")
            : std::vector<Eigen::Vector3d>();
        if (canonicalize && gravity.size() != poses.size())
            throw std::runtime_error("gravity/pose count mismatch");

        const fs::path config_path =
            argc >= 5 ? fs::absolute(argv[4]) : session / "runtime_params.yaml";
        double z_offset = 0.0;
        if (argc >= 6)
        {
            std::size_t consumed = 0;
            z_offset = std::stod(argv[5], &consumed);
            if (consumed != std::string(argv[5]).size() ||
                !std::isfinite(z_offset))
            {
                throw std::runtime_error("Z_OFFSET_M must be finite");
            }
        }
        sc::Config config = loadConfig(config_path, canonicalize);
        if (argc >= 7)
        {
            std::size_t consumed = 0;
            config.dual_z_split_height = std::stod(argv[6], &consumed);
            if (consumed != std::string(argv[6]).size() ||
                !std::isfinite(config.dual_z_split_height))
            {
                throw std::runtime_error(
                    "FIXED_SPLIT_HEIGHT_M must be finite");
            }
            config.dual_z_split_auto = false;
        }
        sc::AdaptiveSplitEstimator split_estimator(config);
        std::vector<sc::PointCloud> descriptor_clouds;
        descriptor_clouds.reserve(poses.size());
        for (std::size_t index = 0; index < poses.size(); ++index)
        {
            sc::PointCloud cloud = loadCloud(
                session / "key_point_frame" / (std::to_string(index) + ".pcd"));
            Eigen::Matrix3d gravity_rotation = Eigen::Matrix3d::Identity();
            if (canonicalize &&
                !sc::makeGravityCanonicalRotation(gravity[index], gravity_rotation))
            {
                throw std::runtime_error(
                    "failed to construct gravity rotation for frame " +
                    std::to_string(index));
            }
            if (canonicalize)
                cloud = sc::gravityCanonicalize(cloud, gravity_rotation);
            if (z_offset != 0.0)
            {
                for (auto &point : cloud.points)
                    point.z += static_cast<float>(z_offset);
            }
            split_estimator.addScan(cloud);
            descriptor_clouds.push_back(std::move(cloud));
            if ((index + 1) % 100 == 0 || index + 1 == poses.size())
                std::cout << "cloud " << (index + 1) << '/' << poses.size() << '\n';
        }
        const sc::AdaptiveSplitResult split = split_estimator.estimate();
        config.dual_z_layer_enable = split.dual_layer_enabled;
        config.dual_z_split_height = split.split_height;
        sc::Database database(config);
        for (std::size_t index = 0; index < poses.size(); ++index)
        {
            Eigen::Matrix3d gravity_rotation = Eigen::Matrix3d::Identity();
            if (canonicalize &&
                !sc::makeGravityCanonicalRotation(
                    gravity[index], gravity_rotation))
            {
                throw std::runtime_error(
                    "failed to reconstruct gravity rotation for frame " +
                    std::to_string(index));
            }
            database.addEntry(
                poses[index].stamp,
                makePose(poses[index], gravity_rotation),
                database.makeDescriptor(descriptor_clouds[index]));
        }
        std::string error;
        if (!database.save(output.string(), &error))
            throw std::runtime_error("failed to save SCD: " + error);
        std::cout << "Wrote " << database.size() << " descriptors to "
                  << output << " mode=" << mode
                  << " z_offset_m=" << z_offset
                  << " split_height_m=" << split.split_height
                  << " dual_layer="
                  << (split.dual_layer_enabled ? "true" : "false")
                  << " adaptive=" << (split.adapted ? "true" : "false")
                  << " lower_fraction=" << split.lower_fraction
                  << " upper_fraction=" << split.upper_fraction
                  << " otsu_score=" << split.between_class_variance << '\n';
    }
    catch (const std::exception &error)
    {
        std::cerr << "scan_context_rebuild failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
