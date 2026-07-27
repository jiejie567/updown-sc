#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace fast_lio::source_ray_export
{

using PointType = pcl::PointXYZINormal;
using PointCloudXYZI = pcl::PointCloud<PointType>;

struct Config
{
    bool enable = false;
    bool overwrite = false;
    bool save_sensor_pcd = true;
    std::string manifest_csv;
    std::string pgo_tum;
    std::string output_dir;
    // Deliberately has no dataset-specific default.  An enabled export must
    // state its expected count explicitly so a truncated replay cannot look
    // successful.
    int expected_frame_count = 0;
    double timestamp_tolerance_us = 2.0;
    double blind_radius_m = 0.3;
    double blind_z_min_m = -0.5;
    double blind_z_max_m = 2.0;
    double maximum_range_m = 30.0;
    double endpoint_voxel_m = 0.1;
    int scan_line = 4;
    int deskew_validation_frames = 3;
    Eigen::Matrix4d T_input_front = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_input_back = Eigen::Matrix4d::Identity();
    std::vector<int64_t> selected_pose_indices;
};

struct PendingFrame
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int pose_index = -1;
    int64_t pose_timestamp_ns = 0;
    int64_t message_header_timestamp_ns = 0;
    int64_t frame_reference_timestamp_ns = 0;
    int64_t frame_end_timestamp_ns = 0;
    int front_block_index = -1;
    std::size_t fused_point_count = 0;
    std::size_t eligible_count = 0;
    std::array<std::size_t, 2> eligible_by_sensor{{0, 0}};
    Eigen::Matrix4d T_world_base = Eigen::Matrix4d::Identity();
    PointCloudXYZI hits_input;
    PointCloudXYZI origins_input;
    std::vector<float> original_ray_lengths_m;
    std::vector<uint8_t> sensor_ids;
    std::vector<uint32_t> original_indices;
};

class Exporter
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Exporter() = default;
    ~Exporter();

    bool configure(const Config &config, std::string &error);
    bool enabled() const { return configured_ && config_.enable; }
    int deskewValidationFrames() const { return config_.deskew_validation_frames; }

    std::unique_ptr<PendingFrame> prepare(
        const sensor_msgs::msg::PointCloud2 &message,
        std::string &error) const;

    bool writeDeskewed(
        PendingFrame &frame,
        const Eigen::Matrix3d &R_base_input,
        const Eigen::Vector3d &t_base_input,
        const Eigen::Vector3d &gravity_up_base,
        std::string &error);

    bool finalize(std::string &error);

private:
    struct SelectedFrame
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        int pose_index = -1;
        int64_t pose_timestamp_ns = 0;
        int64_t frame_reference_timestamp_ns = 0;
        int front_block_index = -1;
        int back_block_index = -1;
        std::size_t boundary_index = 0;
        std::size_t front_point_count = 0;
        std::size_t back_point_count = 0;
        Eigen::Matrix4d T_world_base = Eigen::Matrix4d::Identity();
    };

    const SelectedFrame *findSelectedFrame(int64_t reference_ns) const;
    bool writeSummary(std::string &error) const;

    Config config_;
    std::vector<SelectedFrame, Eigen::aligned_allocator<SelectedFrame>> frames_;
    std::unordered_set<int> active_pose_indices_;
    std::unordered_set<int> exported_pose_indices_;
    std::ofstream manifest_stream_;
    std::ofstream local_cloud_manifest_stream_;
    std::size_t exported_count_ = 0;
    std::size_t total_eligible_count_ = 0;
    std::size_t total_written_count_ = 0;
    double maximum_ray_length_error_m_ = 0.0;
    double maximum_p99_ray_length_error_m_ = 0.0;
    bool configured_ = false;
    bool finalized_ = false;
};

}  // namespace fast_lio::source_ray_export
