// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <iomanip>
#include <vector>
#include <unistd.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include "pcd_save_utils.hpp"
#include "prior_icp.hpp"
#include "scan_context.hpp"
#include "source_ray_export.hpp"

namespace pcd_save = fast_lio::pcd_save;
namespace prior_icp = fast_lio::prior_icp;
namespace sc = fast_lio::scan_context;
namespace source_ray_export = fast_lio::source_ray_export;

double INIT_TIME = 0.1;
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
bool   publish_tf = false, publish_pose_topic = true, pose_output_use_node_clock = false;
bool   debug_save_registered_pcd_en = false;
int    debug_save_registered_pcd_frame_interval = 10;
string debug_registered_pcd_path = "";
string runtime_profile = "mapping";
std::unique_ptr<source_ray_export::Exporter> source_ray_exporter;
sensor_msgs::msg::PointCloud2::SharedPtr source_ray_current_raw_message;
int source_ray_deskew_validation_count = 0;
bool   use_prior_map = false, prior_map_loaded = false, prior_map_aligned = false, prior_map_build_done = false;
bool   prior_map_pub_once_done = false, prior_map_ready_for_publish = false;
bool   localization_health_reset_requested = false;
bool   localization_output_trusted = true;
bool   localization_auto_relocalize_enable = true;
bool   localization_restart_on_timestamp_rollback = true;
int    localization_unhealthy_consecutive_frames = 10;
int    localization_min_effective_points = 100;
int    prior_icp_max_iterations = 60, prior_icp_min_points = 2000;
double prior_icp_max_corr_dist = 3.0, prior_icp_fitness_thresh = 0.8, prior_map_voxel_leaf = 0.5;
double prior_icp_min_overlap_ratio = 0.5;
double prior_map_voxel_leaf_fine = 0.25;
double prior_relocalization_accum_time_s = 0.0;
int    prior_icp_refine_top_k = 3;
int    prior_icp_fail_count = 0;
bool   prior_multi_seed_enable = true;
vector<double> prior_initial_guess_xy(2, 0.0);
double prior_initial_guess_yaw_deg = 0.0;
double prior_seed_xy_range = 2.0, prior_seed_xy_step = 1.0;
double prior_seed_yaw_range_deg = 20.0, prior_seed_yaw_step_deg = 10.0;
bool   scan_context_enable = true, scan_context_loaded = false, scan_context_dirty = false;
string scan_context_database_path = "";
double scan_context_keyframe_meter_gap = 1.0, scan_context_keyframe_yaw_gap_deg = 10.0;
double scan_context_keyframe_yaw_gap_rad = 10.0 * M_PI / 180.0;
double scan_context_voxel_leaf = 0.4;
double scan_context_seed_xy_offset = 0.5;
double preprocess_mapping_blind = 2.0, preprocess_localization_blind = 0.3;
int    scan_context_blind_filter_shape = Preprocess::BLIND_FILTER_SPHERE;
double scan_context_blind_z_min = -std::numeric_limits<double>::infinity();
double scan_context_blind_z_max = std::numeric_limits<double>::infinity();
sc::Config scan_context_config;
sc::Database scan_context_db(scan_context_config);
sc::Database scan_context_query_builder(scan_context_config);
sc::AdaptiveSplitEstimator scan_context_split_estimator(scan_context_config);
struct PendingScanContextKeyframe
{
    double stamp = 0.0;
    sc::Pose pose;
    PointCloudXYZI::Ptr cloud;
};
vector<PendingScanContextKeyframe> scan_context_pending_keyframes;
bool   scan_context_has_last_keyframe = false;
sc::Pose scan_context_last_keyframe_pose;
int    scan_context_keyframe_count = 0;
bool   scan_context_candidate_cloud_active = false;
double scan_context_candidate_cloud_expire_time = 0.0;
vector<sc::Candidate> scan_context_last_icp_candidates;
bool   manual_loop_export_enable = true, manual_loop_export_overwrite = true;
bool   manual_loop_export_initialized = false, manual_loop_has_last_pose = false;
string manual_loop_session_dir = "";
string manual_loop_keyframe_dir = "";
string manual_loop_g2o_path = "";
string manual_loop_tum_path = "";
string manual_loop_gravity_path = "";
sc::Pose manual_loop_last_pose;
Eigen::Matrix4d manual_loop_last_T = Eigen::Matrix4d::Identity();
bool   output_base_link_origin_odom = false;
std::atomic<uint64_t> lidar_timestamp_rollback_events{0};
std::atomic<uint64_t> imu_timestamp_rollback_events{0};
std::atomic<int> localization_timestamp_rollback_streak{0};
std::atomic<bool> localization_restart_pending{false};
/**************************/

inline void append_scan_context_candidate_log(
    const sc::Candidate &candidate,
    const std::size_t yaw_hypotheses,
    const double xy_offset)
{
    std::error_code ec;
    const std::string log_dir = std::string(ROOT_DIR) + "Log";
    std::filesystem::create_directories(log_dir, ec);

    std::ofstream ofs(log_dir + "/scan_context_candidates.log", std::ios::app);
    if (!ofs)
        return;

    ofs << std::fixed << std::setprecision(6)
        << "candidate idx=" << candidate.index
        << " dist=" << candidate.distance
        << " yaw_hypotheses=" << yaw_hypotheses
        << " best_sector_shift=" << candidate.sector_shift
        << std::setprecision(2)
        << " best_yaw_shift_deg=" << candidate.yaw_shift_rad * 180.0 / M_PI
        << std::setprecision(3)
        << " coarse_dz=" << candidate.coarse_vertical_shift
        << " refined_dz=" << candidate.vertical_shift
        << " pose=[" << candidate.pose.x << ' ' << candidate.pose.y << ' ' << candidate.pose.z
        << " yaw_deg=" << candidate.pose.yaw * 180.0 / M_PI << ']'
        << " xy_offset=" << xy_offset
        << '\n';
}

std::vector<float> res_last;
float DET_RANGE = 300.0f;
double MAX_HEIGHT = 5.0;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer, mtx_preprocess;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;
string localization_pose_topic = "/localization_pose";
string output_body_frame_id = "base_link";
string lidar_qos_reliability = "best_effort", imu_qos_reliability = "best_effort";

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0, last_used_imu_time = -1.0;
double lidar_frame_period_sec = 0.1, imu_frame_period_sec = 0.005;
constexpr double SENSOR_SEVERE_OUT_OF_ORDER_DROP_SEC = 0.5;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double pcd_save_voxel_leaf = 0.0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
double lidar_mean_scantime = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
int    scan_num = 0;
uint64_t current_lidar_frame_index = 0, current_lidar_rx_index = 0;
size_t current_raw_lidar_points = 0, current_undistorted_points = 0;
double current_lidar_beg_time = 0.0, current_lidar_end_time = 0.0;
bool current_measurement_no_effective_points = false;
std::vector<uint8_t> point_selected_surf;
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
double scan_body_pub_stride_s = 0.0, scan_body_pub_sample_s = 0.2;
double scan_body_pub_first_stamp = -1.0;
int64_t scan_body_pub_last_slot = -1;
bool   deskew_en = true;
bool   msg_is_XYZI = true, msg_is_XYZIRT = false;
bool   airy_imu_flip_yz = false;
bool    is_first_lidar = true;

PointCloudXYZI::Ptr init_feats_buffer(new PointCloudXYZI());
PointCloudXYZI::Ptr init_feats_buffer_local(new PointCloudXYZI());
PointCloudXYZI::Ptr init_feats_buffer_seed(new PointCloudXYZI());
bool ikdtree_built = false;
std::atomic_bool debug_registered_pcd_write_in_progress(false);

void save_waiting_pcd_on_exit();

inline double rad2deg(double rad)
{
    return rad * 180.0 / M_PI;
}

inline double wrap_angle_rad(double rad)
{
    if (!std::isfinite(rad))
        return 0.0;
    rad = std::remainder(rad, 2.0 * M_PI);
    if (rad <= -M_PI)
        rad += 2.0 * M_PI;
    return rad;
}

inline double wrap_angle_deg(double deg)
{
    if (!std::isfinite(deg))
        return 0.0;
    deg = std::remainder(deg, 360.0);
    if (deg <= -180.0)
        deg += 360.0;
    return deg;
}

inline bool point_xyz_finite(const PointType &point)
{
    return std::isfinite(point.x) &&
           std::isfinite(point.y) &&
           std::isfinite(point.z);
}

inline bool valid_scan_context_input_point(const PointType &point)
{
    if (!point_xyz_finite(point))
        return false;

    const double blind = std::max(0.0, preprocess_mapping_blind);
    if (blind <= 1e-12)
        return true;

    const double x = static_cast<double>(point.x);
    const double y = static_cast<double>(point.y);
    const double z = static_cast<double>(point.z);
    const double xy_sq = x * x + y * y;
    const double blind_sq = blind * blind;
    if (scan_context_blind_filter_shape == Preprocess::BLIND_FILTER_CYLINDER)
    {
        const bool inside_xy = xy_sq <= blind_sq;
        const bool inside_z = z >= scan_context_blind_z_min && z <= scan_context_blind_z_max;
        return !(inside_xy && inside_z);
    }

    return xy_sq + z * z > blind_sq;
}

inline string resolve_output_path(const string &path)
{
    if (path.empty())
        return "";
    if (path.front() == '/')
        return path;
    if (path.rfind("./", 0) == 0)
        return string(ROOT_DIR) + path.substr(2);
    return string(ROOT_DIR) + path;
}

inline void save_debug_registered_pcd_async(const PointCloudXYZI::Ptr &cloud, double stamp)
{
    if (!cloud || cloud->empty() || debug_registered_pcd_path.empty())
    {
        return;
    }
    if (debug_registered_pcd_write_in_progress.exchange(true))
    {
        return;
    }

    const string output_path_string = debug_registered_pcd_path;
    PointCloudXYZI::Ptr cloud_snapshot(new PointCloudXYZI(*cloud));
    std::thread([cloud_snapshot, stamp, output_path_string]() {
        struct ClearInProgress
        {
            ~ClearInProgress()
            {
                debug_registered_pcd_write_in_progress.store(false);
            }
        };
        const ClearInProgress clear_in_progress;

        try
        {
            const std::filesystem::path output_path(output_path_string);
            if (!output_path.parent_path().empty())
            {
                std::error_code dir_ec;
                std::filesystem::create_directories(output_path.parent_path(), dir_ec);
                if (dir_ec)
                {
                    RCLCPP_WARN(
                        rclcpp::get_logger("laser_mapping"),
                        "Failed to create debug PCD directory: %s", dir_ec.message().c_str());
                    return;
                }
            }

            const std::filesystem::path tmp_path = output_path.string() + ".tmp";
            auto cleanup_tmp = [&tmp_path]() {
                std::error_code cleanup_ec;
                std::filesystem::remove(tmp_path, cleanup_ec);
            };

            if (pcl::io::savePCDFileBinary(tmp_path.string(), *cloud_snapshot) < 0)
            {
                RCLCPP_WARN(
                    rclcpp::get_logger("laser_mapping"),
                    "Failed to save debug registered PCD: %s", tmp_path.string().c_str());
                cleanup_tmp();
                return;
            }

            std::error_code rename_ec;
            std::filesystem::rename(tmp_path, output_path, rename_ec);
            if (rename_ec)
            {
                std::error_code remove_ec;
                std::filesystem::remove(output_path, remove_ec);
                rename_ec.clear();
                std::filesystem::rename(tmp_path, output_path, rename_ec);
            }
            if (rename_ec)
            {
                RCLCPP_WARN(
                    rclcpp::get_logger("laser_mapping"),
                    "Failed to update debug registered PCD: %s", rename_ec.message().c_str());
                cleanup_tmp();
                return;
            }

            RCLCPP_DEBUG(
                rclcpp::get_logger("laser_mapping"),
                "Saved debug registered PCD: %s points=%zu stamp=%.6f",
                output_path_string.c_str(), cloud_snapshot->size(), stamp);
        }
        catch (const std::exception &e)
        {
            std::error_code cleanup_ec;
            std::filesystem::remove(output_path_string + ".tmp", cleanup_ec);
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "Failed to save debug registered PCD: %s", e.what());
        }
        catch (...)
        {
            std::error_code cleanup_ec;
            std::filesystem::remove(output_path_string + ".tmp", cleanup_ec);
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "Failed to save debug registered PCD: unknown exception");
        }
    }).detach();
}

static rclcpp::QoS make_sensor_qos(int depth, const std::string & reliability)
{
    rclcpp::QoS qos(rclcpp::KeepLast(std::max(1, depth)));
    if (reliability == "reliable")
    {
        qos.reliable();
    }
    else
    {
        qos.best_effort();
    }
    qos.durability_volatile();
    return qos;
}

static double time_unit_to_sec_scale(int time_unit)
{
    switch (time_unit)
    {
    case SEC:
        return 1.0;
    case MS:
        return 1.0e-3;
    case US:
        return 1.0e-6;
    case NS:
        return 1.0e-9;
    default:
        return 1.0e-6;
    }
}

static double point_timestamp_to_absolute_sec(double stamp, double header_time, int relative_time_unit)
{
    if (stamp > 1.0e17) return stamp * 1.0e-9;  // ns epoch
    if (stamp > 1.0e14) return stamp * 1.0e-6;  // us epoch
    if (stamp > 1.0e11) return stamp * 1.0e-3;  // ms epoch
    if (stamp > 1.0e9) return stamp;            // sec epoch
    return header_time + stamp * time_unit_to_sec_scale(relative_time_unit);
}

inline void record_preprocess_time_sample(int sample_index, double duration_s)
{
    if (sample_index >= 0 && sample_index < MAXN)
    {
        s_plot11[sample_index] = duration_s;
        return;
    }

    static bool warned = false;
    if (!warned)
    {
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "Runtime preprocess time log reached capacity (%d samples); further samples will not be recorded.",
            MAXN);
        warned = true;
    }
}

static bool get_pointcloud_timestamp_bounds(
    const sensor_msgs::msg::PointCloud2 &msg,
    int relative_time_unit,
    double &min_time,
    double &max_time)
{
    auto field_it = std::find_if(
        msg.fields.begin(), msg.fields.end(),
        [](const sensor_msgs::msg::PointField &field) { return field.name == "timestamp"; });
    if (field_it == msg.fields.end())
        return false;

    const auto width = static_cast<size_t>(msg.width);
    const auto height = static_cast<size_t>(msg.height);
    const auto point_step = static_cast<size_t>(msg.point_step);
    const auto row_step =
        msg.row_step > 0 ? static_cast<size_t>(msg.row_step) : width * point_step;
    const auto point_count = width * height;
    if (point_count == 0 || point_step == 0 || width == 0 || height == 0)
        return false;
    const size_t field_size =
        field_it->datatype == sensor_msgs::msg::PointField::FLOAT64 ? sizeof(double) :
        field_it->datatype == sensor_msgs::msg::PointField::FLOAT32 ? sizeof(float) : 0U;
    if (field_size == 0)
        return false;
    if (field_it->offset + field_size > point_step)
        return false;
    if (row_step < width * point_step)
        return false;
    if (height > 0 && row_step > msg.data.size() / height)
        return false;

    min_time = std::numeric_limits<double>::infinity();
    max_time = -std::numeric_limits<double>::infinity();
    const double header_time = get_time_sec(msg.header.stamp);
    bool has_time = false;
    for (size_t row = 0; row < height; ++row)
    {
        for (size_t col = 0; col < width; ++col)
        {
            const size_t offset = row * row_step + col * point_step + field_it->offset;

            double raw_time = 0.0;
            if (field_it->datatype == sensor_msgs::msg::PointField::FLOAT64)
            {
                std::memcpy(&raw_time, msg.data.data() + offset, sizeof(double));
            }
            else
            {
                float raw_time_float = 0.0f;
                std::memcpy(&raw_time_float, msg.data.data() + offset, sizeof(float));
                raw_time = raw_time_float;
            }

            if (!std::isfinite(raw_time) || std::fabs(raw_time) <= 1.0e-12)
                continue;
            const double point_time =
                point_timestamp_to_absolute_sec(raw_time, header_time, relative_time_unit);
            if (!std::isfinite(point_time))
                continue;
            min_time = std::min(min_time, point_time);
            max_time = std::max(max_time, point_time);
            has_time = true;
        }
    }

    return has_time;
}

static void warn_lidar_frame_gap_if_needed(
    const char *source,
    uint64_t rx_index,
    double previous_begin_time,
    double previous_end_time,
    double current_begin_time,
    double current_end_time,
    std::size_t point_count,
    double expected_period_sec)
{
    constexpr int kMaxReports = 50;
    const double warn_gap_sec =
        (expected_period_sec > 1e-6) ? (3.0 * expected_period_sec) : 0.30;

    if (previous_begin_time < 0.0 || current_begin_time <= previous_begin_time)
        return;

    const double previous_reference_time =
        (previous_end_time > previous_begin_time) ? previous_end_time : previous_begin_time;
    const double empty_gap = current_begin_time - previous_reference_time;
    const double start_dt = current_begin_time - previous_begin_time;
    if (empty_gap <= warn_gap_sec)
        return;

    static int report_count = 0;
    if (report_count < kMaxReports)
    {
        const double current_span =
            (current_end_time > current_begin_time) ? (current_end_time - current_begin_time) : 0.0;
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "LiDAR frame gap too large: source=%s rx=%" PRIu64
            " empty_gap=%.6fs start_dt=%.6fs warn_threshold=%.6fs expected_period=%.6fs"
            " prev=[%.6f %.6f] curr=[%.6f %.6f] curr_span=%.3fs points=%zu",
            source, rx_index, empty_gap, start_dt, warn_gap_sec, expected_period_sec,
            previous_begin_time, previous_reference_time,
            current_begin_time,
            (current_end_time > current_begin_time) ? current_end_time : current_begin_time,
            current_span, point_count);
    }
    else if (report_count == kMaxReports)
    {
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "LiDAR frame gap warning suppressed after %d reports.", kMaxReports);
    }
    ++report_count;
}

static void warn_imu_frame_gap_if_needed(double previous_time, double current_time)
{
    constexpr int kMaxReports = 50;
    const double warn_gap_sec =
        (imu_frame_period_sec > 1e-6) ? (3.0 * imu_frame_period_sec) : 0.015;

    if (previous_time < 0.0 || current_time <= previous_time)
        return;

    const double dt = current_time - previous_time;
    if (dt <= warn_gap_sec)
        return;

    static int report_count = 0;
    if (report_count < kMaxReports)
    {
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "IMU frame gap too large: dt=%.6fs warn_threshold=%.6fs expected_period=%.6fs prev=%.9f curr=%.9f",
            dt, warn_gap_sec, imu_frame_period_sec, previous_time, current_time);
    }
    else if (report_count == kMaxReports)
    {
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "IMU frame gap warning suppressed after %d reports.", kMaxReports);
    }
    ++report_count;
}

vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
vector<double>       extrinT_imu_to_base_link(3, 0.0);
vector<double>       extrinR_imu_to_base_link(9, 0.0);
bool                 transform_to_base_link = false;
V3D                  BaseLink_T_wrt_LidarIMU(Zero3d);
M3D                  BaseLink_R_wrt_LidarIMU(Eye3d);
bool                 map_world_initialized = false;
string               base_link_world_frame_id = "map";
V3D                  p_C_M0(Zero3d);
M3D                  R_C_M0(Eye3d);
M3D                  R_P_M(Eye3d);
V3D                  p_P_M(Zero3d);

inline bool use_base_link_output_frame()
{
    return output_base_link_origin_odom || transform_to_base_link || use_prior_map;
}
deque<double>                     time_buffer;
deque<double>                     lidar_end_time_buffer;
deque<uint64_t>                   lidar_rx_index_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::PointCloud2::SharedPtr> raw_pcl_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;
uint64_t last_synced_lidar_rx_index = 0;
double prior_relocalization_sensor_restart_time = -1.0;

inline void clear_lidar_buffers_locked()
{
    raw_pcl_buffer.clear();
    lidar_buffer.clear();
    time_buffer.clear();
    lidar_end_time_buffer.clear();
    lidar_rx_index_buffer.clear();
    lidar_pushed = false;
}

inline bool front_lidar_metadata_ready_locked(bool use_raw_pointcloud, const char *context)
{
    const bool lidar_available =
        use_raw_pointcloud ? !raw_pcl_buffer.empty() : !lidar_buffer.empty();
    if (!lidar_available)
        return false;

    if (time_buffer.empty() || lidar_end_time_buffer.empty() || lidar_rx_index_buffer.empty())
    {
        static int mismatch_report_count = 0;
        if (mismatch_report_count < 20)
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "LiDAR buffer metadata mismatch in %s: raw=%zu livox=%zu time=%zu end=%zu rx=%zu. Clearing LiDAR buffers.",
                context ? context : "sync",
                raw_pcl_buffer.size(),
                lidar_buffer.size(),
                time_buffer.size(),
                lidar_end_time_buffer.size(),
                lidar_rx_index_buffer.size());
            ++mismatch_report_count;
        }
        clear_lidar_buffers_locked();
        return false;
    }

    return true;
}

inline void pop_lidar_front_locked(bool use_raw_pointcloud)
{
    if (use_raw_pointcloud)
    {
        if (!raw_pcl_buffer.empty())
            raw_pcl_buffer.pop_front();
    }
    else
    {
        if (!lidar_buffer.empty())
            lidar_buffer.pop_front();
    }
    if (!time_buffer.empty())
        time_buffer.pop_front();
    if (!lidar_end_time_buffer.empty())
        lidar_end_time_buffer.pop_front();
    if (!lidar_rx_index_buffer.empty())
        lidar_rx_index_buffer.pop_front();
    lidar_pushed = false;
}

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI());
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI());
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI());
PointCloudXYZI::Ptr _featsArray;
PointCloudXYZI::Ptr prior_map_cloud(new PointCloudXYZI());
PointCloudXYZI::Ptr prior_map_cloud_coarse(new PointCloudXYZI());
PointCloudXYZI::Ptr prior_map_cloud_fine(new PointCloudXYZI());
PointCloudXYZI::Ptr prior_icp_source_cloud(new PointCloudXYZI());
PointCloudXYZI::Ptr prior_icp_seed_cloud(new PointCloudXYZI());
bool                prior_icp_source_frozen = false;
double              prior_icp_accum_start_time = -1.0;
double              prior_icp_source_start_time = -1.0;
double              prior_icp_source_end_time = -1.0;
uint64_t            prior_icp_source_first_frame = 0;
uint64_t            prior_icp_source_last_frame = 0;
uint64_t            prior_icp_source_first_rx = 0;
uint64_t            prior_icp_source_last_rx = 0;
bool                prior_icp_source_ref_pose_valid = false;
M3D                 prior_icp_source_ref_R_M_B(Eye3d);
V3D                 prior_icp_source_ref_p_M_B(Zero3d);
uint64_t            prior_icp_source_ref_frame = 0;
bool                prior_icp_source_gravity_rotation_valid = false;
M3D                 prior_icp_source_R_G_B(Eye3d);

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po);
void RGBpointBodyLidarToBaseLink(PointType const * const pi, PointType * const po);
bool current_prior_icp_body_pose_in_local_map(M3D &R_M_B, V3D &p_M_B);
bool current_scan_context_gravity_up(V3D &up_B);
bool current_scan_context_gravity_rotation(M3D &R_G_B);
void capture_prior_replay_pending_snapshot(uint64_t frame_index, double lidar_beg_time);
bool begin_prior_replay_candidate_from_pending(uint64_t frame_index, double lidar_beg_time);
void cache_prior_replay_frame(const MeasureGroup &meas, uint64_t frame_index, uint64_t rx_index);
void clear_prior_replay_cache();
void drop_sensor_backlog_for_prior_relocalization_restart(const char *reason, const char *source, bool reset_sensor_time_gate = false);
void drop_stale_sensor_backlog_after_prior_icp_failure(const char *reason);
bool reset_prior_relocalization_local_state_after_failure(const char *reason);
bool restart_prior_relocalization_from_health(
    const char *reason,
    int unhealthy_count,
    bool insufficient_effective_points,
    bool timestamp_rollback,
    int downsampled_points,
    int effective_points,
    double lidar_beg_time,
    uint64_t frame_index,
    uint64_t rx_index);

inline void clear_prior_icp_active_source()
{
    prior_icp_source_frozen = false;
    prior_icp_source_cloud->clear();
    prior_icp_seed_cloud->clear();
    prior_icp_source_start_time = -1.0;
    prior_icp_source_end_time = -1.0;
    prior_icp_source_first_frame = 0;
    prior_icp_source_last_frame = 0;
    prior_icp_source_first_rx = 0;
    prior_icp_source_last_rx = 0;
    prior_icp_source_ref_pose_valid = false;
    prior_icp_source_ref_R_M_B = Eye3d;
    prior_icp_source_ref_p_M_B = Zero3d;
    prior_icp_source_ref_frame = 0;
    prior_icp_source_gravity_rotation_valid = false;
    prior_icp_source_R_G_B = Eye3d;
}

inline void clear_prior_icp_accum_window()
{
    prior_icp_accum_start_time = -1.0;
    init_feats_buffer_local->clear();
    init_feats_buffer_seed->clear();
}

inline void note_timestamp_rollback_for_localization_restart()
{
    const int streak = localization_timestamp_rollback_streak.fetch_add(1, std::memory_order_relaxed) + 1;
    if (streak >= localization_unhealthy_consecutive_frames)
    {
        if (streak == localization_unhealthy_consecutive_frames)
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "Timestamp rollback streak reached restart threshold: consecutive=%d/%d tracking_active=%s use_prior=%s loaded=%s aligned=%s build_done=%s",
                streak,
                localization_unhealthy_consecutive_frames,
                (use_prior_map && prior_map_loaded && prior_map_aligned && prior_map_build_done) ? "true" : "false",
                use_prior_map ? "true" : "false",
                prior_map_loaded ? "true" : "false",
                prior_map_aligned ? "true" : "false",
                prior_map_build_done ? "true" : "false");
        }
        localization_restart_pending.store(true, std::memory_order_release);
    }
}

inline void clear_timestamp_rollback_streak()
{
    localization_timestamp_rollback_streak.store(0, std::memory_order_relaxed);
}

inline void reset_prior_icp_accumulation(const char *reason)
{
    clear_prior_icp_active_source();
    clear_prior_icp_accum_window();
    clear_prior_replay_cache();
    bool local_origin_reset = false;
    if (reason && std::strcmp(reason, "icp_failed") == 0)
    {
        drop_stale_sensor_backlog_after_prior_icp_failure(reason);
        local_origin_reset = reset_prior_relocalization_local_state_after_failure(reason);
    }
    scan_context_last_icp_candidates.clear();
    scan_context_candidate_cloud_active = false;
    RCLCPP_WARN(
        rclcpp::get_logger("laser_mapping"),
        "Reset prior-map relocalization source accumulation: reason=%s. Re-accumulating %.2f s of LiDAR data. local_origin_reset=%s",
        reason ? reason : "unknown", prior_relocalization_accum_time_s,
        local_origin_reset ? "true" : "false");
}

inline bool append_prior_icp_source_frame(
    const PointCloudXYZI::Ptr &scan_body,
    double lidar_beg_time,
    double lidar_end_time,
    uint64_t frame_index,
    uint64_t rx_index)
{
    if (!scan_body || scan_body->empty())
        return false;

    M3D R_M_B;
    V3D p_M_B;
    if (!current_prior_icp_body_pose_in_local_map(R_M_B, p_M_B))
    {
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "\033[1;32mRelocalization source discarded\033[0m reason=no_local_map_pose frame=%" PRIu64,
            frame_index);
        reset_prior_icp_accumulation("no_local_map_pose");
        return false;
    }

    M3D R_G_B(Eye3d);
    if (!current_scan_context_gravity_rotation(R_G_B))
    {
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "Relocalization source discarded reason=invalid_gravity frame=%" PRIu64,
            frame_index);
        reset_prior_icp_accumulation("invalid_gravity");
        return false;
    }

    const bool starting_new_source = prior_icp_accum_start_time < 0.0;
    if (starting_new_source && !begin_prior_replay_candidate_from_pending(frame_index, lidar_beg_time))
    {
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "\033[1;32mRelocalization source discarded\033[0m reason=no_replay_snapshot frame=%" PRIu64,
            frame_index);
        reset_prior_icp_accumulation("no_replay_snapshot");
        return false;
    }

    if (prior_icp_accum_start_time < 0.0)
    {
        prior_icp_accum_start_time = lidar_beg_time;
        prior_icp_source_start_time = lidar_beg_time;
        prior_icp_source_first_frame = frame_index;
        prior_icp_source_first_rx = rx_index;
    }

    prior_icp_source_end_time = lidar_end_time > lidar_beg_time ? lidar_end_time : lidar_beg_time;
    prior_icp_source_last_frame = frame_index;
    prior_icp_source_last_rx = rx_index;
    prior_icp_source_ref_pose_valid = true;
    prior_icp_source_ref_R_M_B = R_M_B;
    prior_icp_source_ref_p_M_B = p_M_B;
    prior_icp_source_ref_frame = frame_index;
    prior_icp_source_gravity_rotation_valid = true;
    prior_icp_source_R_G_B = R_G_B;

    init_feats_buffer_local->reserve(init_feats_buffer_local->size() + scan_body->size());
    init_feats_buffer_seed->reserve(init_feats_buffer_seed->size() + scan_body->size());
    for (const auto &pt_body : scan_body->points)
    {
        PointType body_point;
        if (use_base_link_output_frame())
            RGBpointBodyLidarToBaseLink(&pt_body, &body_point);
        else
            RGBpointBodyLidarToIMU(&pt_body, &body_point);
        if (!point_xyz_finite(body_point))
            continue;
        if (valid_scan_context_input_point(pt_body))
            init_feats_buffer_seed->push_back(body_point);
        init_feats_buffer_local->push_back(body_point);
    }
    return true;
}

inline bool freeze_prior_icp_source_if_ready(double current_lidar_beg_time)
{
    if (prior_icp_source_frozen)
        return true;

    if (prior_icp_accum_start_time < 0.0)
        return false;

    if ((current_lidar_beg_time - prior_icp_accum_start_time) < prior_relocalization_accum_time_s)
        return false;

    if (static_cast<int>(init_feats_buffer_local->size()) < prior_icp_min_points)
    {
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "\033[1;32mRelocalization source discarded\033[0m reason=too_few_accumulated_points points=%zu min_points=%d window=[%.3f %.3f] frames=[%" PRIu64 " %" PRIu64 "]",
            init_feats_buffer_local->size(), prior_icp_min_points,
            prior_icp_source_start_time, prior_icp_source_end_time,
            prior_icp_source_first_frame, prior_icp_source_last_frame);
        reset_prior_icp_accumulation("too_few_accumulated_points");
        return false;
    }

    *prior_icp_source_cloud = *init_feats_buffer_local;
    *prior_icp_seed_cloud = *init_feats_buffer_seed;
    prior_icp_source_frozen = true;
    clear_prior_icp_accum_window();
    RCLCPP_INFO(
        rclcpp::get_logger("laser_mapping"),
        "Prior-map ICP source frozen: body_points=%zu seed_points=%zu window=[%.3f %.3f] frames=[%" PRIu64 " %" PRIu64 "] rx=[%" PRIu64 " %" PRIu64 "] ref_frame=%" PRIu64,
        prior_icp_source_cloud->size(),
        prior_icp_seed_cloud->size(),
        prior_icp_source_start_time, prior_icp_source_end_time,
        prior_icp_source_first_frame, prior_icp_source_last_frame,
        prior_icp_source_first_rx, prior_icp_source_last_rx,
        prior_icp_source_ref_frame);
    return true;
}

inline void reject_current_prior_icp_source(const char *reason)
{
    RCLCPP_WARN(
        rclcpp::get_logger("laser_mapping"),
        "\033[1;32mRelocalization source rejected\033[0m reason=%s points=%zu window=[%.3f %.3f] frames=[%" PRIu64 " %" PRIu64 "] rx=[%" PRIu64 " %" PRIu64 "]",
        reason ? reason : "icp_failed",
        prior_icp_source_cloud->size(),
        prior_icp_source_start_time, prior_icp_source_end_time,
        prior_icp_source_first_frame, prior_icp_source_last_frame,
        prior_icp_source_first_rx, prior_icp_source_last_rx);
    reset_prior_icp_accumulation(reason ? reason : "icp_failed");
}

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur(Zero3d);
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;
rclcpp::Clock::SharedPtr node_clock;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

struct PriorReplayFrame
{
    MeasureGroup meas;
    uint64_t frame_index = 0;
    uint64_t rx_index = 0;
};

deque<PriorReplayFrame> prior_replay_frame_queue;
bool prior_replay_snapshot_valid = false;
bool prior_replay_active = false;
state_ikfom prior_replay_start_state;
esekfom::esekf<state_ikfom, 12, input_ikfom>::cov prior_replay_start_cov;
ImuProcess::Snapshot prior_replay_imu_snapshot;
uint64_t prior_replay_start_frame = 0;
double prior_replay_start_lidar_time = -1.0;
std::size_t prior_replay_total_frames = 0;

bool prior_replay_pending_snapshot_valid = false;
state_ikfom prior_replay_pending_state;
esekfom::esekf<state_ikfom, 12, input_ikfom>::cov prior_replay_pending_cov;
ImuProcess::Snapshot prior_replay_pending_imu_snapshot;
uint64_t prior_replay_pending_frame = 0;
double prior_replay_pending_lidar_time = -1.0;

void clear_prior_replay_cache()
{
    prior_replay_snapshot_valid = false;
    prior_replay_active = false;
    prior_replay_frame_queue.clear();
    prior_replay_total_frames = 0;
    prior_replay_start_frame = 0;
    prior_replay_start_lidar_time = -1.0;
    prior_replay_pending_snapshot_valid = false;
    prior_replay_pending_frame = 0;
    prior_replay_pending_lidar_time = -1.0;
}

void capture_prior_replay_pending_snapshot(uint64_t frame_index, double lidar_beg_time)
{
    if (prior_replay_active)
        return;

    prior_replay_pending_state = kf.get_x();
    prior_replay_pending_cov = kf.get_P();
    prior_replay_pending_imu_snapshot = p_imu->CaptureSnapshot();
    prior_replay_pending_frame = frame_index;
    prior_replay_pending_lidar_time = lidar_beg_time;
    prior_replay_pending_snapshot_valid = true;
}

bool begin_prior_replay_candidate_from_pending(uint64_t frame_index, double lidar_beg_time)
{
    if (prior_replay_snapshot_valid)
        return true;

    if (!prior_replay_pending_snapshot_valid)
        return false;

    prior_replay_start_state = prior_replay_pending_state;
    prior_replay_start_cov = prior_replay_pending_cov;
    prior_replay_imu_snapshot = prior_replay_pending_imu_snapshot;
    prior_replay_start_frame = prior_replay_pending_frame;
    prior_replay_start_lidar_time = prior_replay_pending_lidar_time;
    prior_replay_snapshot_valid = true;
    prior_replay_frame_queue.clear();
    prior_replay_total_frames = 0;

    RCLCPP_INFO(
        rclcpp::get_logger("laser_mapping"),
        "Captured FAST-LIO replay snapshot for prior-map source: frame=%" PRIu64 " lidar_t=%.3f request_frame=%" PRIu64 " request_lidar_t=%.3f",
        prior_replay_start_frame, prior_replay_start_lidar_time,
        frame_index, lidar_beg_time);
    return true;
}

void cache_prior_replay_frame(const MeasureGroup &meas, uint64_t frame_index, uint64_t rx_index)
{
    if (!prior_replay_snapshot_valid || prior_replay_active)
        return;

    PriorReplayFrame frame;
    frame.meas = meas;
    frame.frame_index = frame_index;
    frame.rx_index = rx_index;
    prior_replay_frame_queue.push_back(frame);
}

inline void begin_prior_replay_after_relocalization()
{
    if (!prior_replay_snapshot_valid || prior_replay_frame_queue.empty())
    {
        prior_replay_snapshot_valid = false;
        prior_replay_active = false;
        prior_replay_frame_queue.clear();
        return;
    }

    kf.change_x(prior_replay_start_state);
    kf.change_P(prior_replay_start_cov);
    p_imu->RestoreSnapshot(prior_replay_imu_snapshot);
    state_point = kf.get_x();
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
    prior_replay_active = true;
    prior_replay_total_frames = prior_replay_frame_queue.size();
    localization_health_reset_requested = true;

    RCLCPP_INFO(
        rclcpp::get_logger("laser_mapping"),
        "\033[1;32mStart FAST-LIO replay after relocalization\033[0m frames=%zu start_frame=%" PRIu64 " start_lidar_t=%.3f",
        prior_replay_total_frames, prior_replay_start_frame, prior_replay_start_lidar_time);
}

inline bool pop_prior_replay_frame(MeasureGroup &meas, uint64_t &frame_index, uint64_t &rx_index)
{
    if (!prior_replay_active || prior_replay_frame_queue.empty())
        return false;

    const PriorReplayFrame frame = prior_replay_frame_queue.front();
    prior_replay_frame_queue.pop_front();
    meas = frame.meas;
    frame_index = frame.frame_index;
    rx_index = frame.rx_index;
    if (prior_replay_frame_queue.empty())
    {
        prior_replay_active = false;
        prior_replay_snapshot_valid = false;
        RCLCPP_INFO(
            rclcpp::get_logger("laser_mapping"),
            "\033[1;32mFAST-LIO replay caught up\033[0m replayed_frames=%zu",
            prior_replay_total_frames);
    }
    return true;
}

void SigHandle(int sig)
{
    (void)sig;
    flg_exit = true;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

rclcpp::Time pose_output_stamp()
{
    if (pose_output_use_node_clock && node_clock)
    {
        return node_clock->now();
    }
    return get_ros_time(lidar_end_time);
}

inline geometry_msgs::msg::TransformStamped make_identity_static_tf(
    const string &parent_frame, const string &child_frame, const rclcpp::Time &stamp)
{
    geometry_msgs::msg::TransformStamped trans;
    trans.header.stamp = stamp;
    trans.header.frame_id = parent_frame;
    trans.child_frame_id = child_frame;
    trans.transform.translation.x = 0.0;
    trans.transform.translation.y = 0.0;
    trans.transform.translation.z = 0.0;
    trans.transform.rotation.x = 0.0;
    trans.transform.rotation.y = 0.0;
    trans.transform.rotation.z = 0.0;
    trans.transform.rotation.w = 1.0;
    return trans;
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToBaseLink(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);
    V3D p_base_link_imu(BaseLink_R_wrt_LidarIMU * p_body_imu + BaseLink_T_wrt_LidarIMU);

    po->x = p_base_link_imu(0);
    po->y = p_base_link_imu(1);
    po->z = p_base_link_imu(2);
    po->intensity = pi->intensity;
}

inline bool has_latest_imu()
{
    return (!imu_buffer.empty()) && (!Measures.imu.empty()) && (Measures.imu.back() != nullptr);
}

inline void maybe_init_base_link_gravity_map(const V3D &p_C_F, const M3D &R_C_F)
{
    if (map_world_initialized)
        return;

    V3D grav_C = state_point.grav;
    if (grav_C.norm() < 1e-6)
        return;

    const V3D g_C_unit = grav_C.normalized();
    V3D g_F = R_C_F.transpose() * g_C_unit;
    if (g_F.norm() < 1e-6)
        return;

    V3D z_M_F = -g_F.normalized();
    V3D x_F(1.0, 0.0, 0.0);
    V3D x_proj = x_F - z_M_F * (x_F.dot(z_M_F));
    if (x_proj.norm() < 1e-6)
    {
        V3D y_F(0.0, 1.0, 0.0);
        x_proj = y_F - z_M_F * (y_F.dot(z_M_F));
    }
    V3D x_M_F = x_proj.normalized();
    V3D y_M_F = z_M_F.cross(x_M_F).normalized();
    x_M_F = y_M_F.cross(z_M_F).normalized();

    M3D R_F_M;
    R_F_M.col(0) = x_M_F;
    R_F_M.col(1) = y_M_F;
    R_F_M.col(2) = z_M_F;

    R_C_M0 = R_C_F * R_F_M;
    p_C_M0 = p_C_F;
    map_world_initialized = true;

    RCLCPP_DEBUG(
        rclcpp::get_logger("laser_mapping"),
        "Local map origin initialized at current base_link, expressed in camera_init: p_C_M0 = [%.3f, %.3f, %.3f]",
        p_C_M0(0), p_C_M0(1), p_C_M0(2));
}

inline V3D transform_cam_to_local_map(const V3D &p_C)
{
    if (!map_world_initialized)
        return p_C;
    return R_C_M0.transpose() * (p_C - p_C_M0);
}

inline V3D transform_local_map_to_output_map(const V3D &p_M)
{
    if (!prior_map_aligned)
        return p_M;
    return R_P_M * p_M + p_P_M;
}

inline V3D transform_cam_to_map(const V3D &p_C)
{
    return transform_local_map_to_output_map(transform_cam_to_local_map(p_C));
}

inline string resolve_prior_map_path(const string &pcd_path)
{
    if (pcd_path.empty())
        return pcd_path;
    if (pcd_path[0] == '/')
        return pcd_path;
    if (pcd_path.size() >= 2 && pcd_path[0] == '.' && pcd_path[1] == '/')
        return string(ROOT_DIR) + pcd_path.substr(2);
    return string(ROOT_DIR) + pcd_path;
}

inline string derive_scan_context_database_path(const string &configured_path, const string &pcd_path)
{
    if (!configured_path.empty())
        return resolve_prior_map_path(configured_path);

    if (!use_prior_map)
        return string(ROOT_DIR) + "PCD/scans.scd";

    string resolved_pcd_path = resolve_prior_map_path(pcd_path);
    if (resolved_pcd_path.empty())
        resolved_pcd_path = string(ROOT_DIR) + "prior_map/scans.pcd";

    std::filesystem::path db_path(resolved_pcd_path);
    db_path.replace_extension(".scd");
    return db_path.string();
}

inline string derive_manual_loop_session_dir(const string &configured_path)
{
    if (!configured_path.empty())
        return resolve_output_path(configured_path);
    return string(ROOT_DIR) + "PCD/manual_loop_session";
}

inline double normalize_yaw(double yaw)
{
    return wrap_angle_rad(yaw);
}

inline V3D rotation_matrix_to_rpy(const M3D &R)
{
    V3D rpy;
    rpy(0) = atan2(R(2, 1), R(2, 2));
    rpy(1) = asin(std::max(-1.0, std::min(1.0, -R(2, 0))));
    rpy(2) = atan2(R(1, 0), R(0, 0));
    return rpy;
}

inline void compute_base_link_pose_twist_in_cam_init(V3D &p_C_F, M3D &R_C_F, V3D &v_C_F, V3D &w_C_F)
{
    const M3D R_C_I = state_point.rot.toRotationMatrix();
    const M3D R_F_I = BaseLink_R_wrt_LidarIMU;
    const V3D t_F_I = BaseLink_T_wrt_LidarIMU;

    R_C_F = R_C_I * R_F_I.transpose();
    const V3D r_C_IF = -R_C_F * t_F_I;
    p_C_F = state_point.pos + r_C_IF;

    V3D omega_L(Zero3d);
    if (has_latest_imu())
    {
        omega_L << Measures.imu.back()->angular_velocity.x,
                   Measures.imu.back()->angular_velocity.y,
                   Measures.imu.back()->angular_velocity.z;
        omega_L(0) -= state_point.bg[0];
        omega_L(1) -= state_point.bg[1];
        omega_L(2) -= state_point.bg[2];
    }

    w_C_F = R_C_I * omega_L;
    v_C_F = state_point.vel + w_C_F.cross(r_C_IF);
}

bool current_scan_context_gravity_up(V3D &up_B)
{
    const V3D gravity_C = state_point.grav;
    if (!gravity_C.allFinite() || gravity_C.squaredNorm() < 1e-12)
        return false;

    M3D R_C_B = state_point.rot.toRotationMatrix();
    if (use_base_link_output_frame())
    {
        V3D p_C_F, v_C_F, w_C_F;
        M3D R_C_F;
        compute_base_link_pose_twist_in_cam_init(p_C_F, R_C_F, v_C_F, w_C_F);
        R_C_B = R_C_F;
    }

    up_B = -(R_C_B.transpose() * gravity_C).normalized();
    return up_B.allFinite();
}

bool current_scan_context_gravity_rotation(M3D &R_G_B)
{
    R_G_B = Eye3d;
    if (!scan_context_config.gravity_canonicalized)
        return true;

    V3D up_B(Zero3d);
    if (!current_scan_context_gravity_up(up_B))
        return false;
    return sc::makeGravityCanonicalRotation(up_B, R_G_B);
}

bool current_prior_icp_body_pose_in_local_map(M3D &R_M_B, V3D &p_M_B)
{
    if (use_base_link_output_frame())
    {
        V3D p_C_F, v_C_F, w_C_F;
        M3D R_C_F;
        compute_base_link_pose_twist_in_cam_init(p_C_F, R_C_F, v_C_F, w_C_F);
        maybe_init_base_link_gravity_map(p_C_F, R_C_F);
        if (!map_world_initialized)
            return false;

        p_M_B = transform_cam_to_local_map(p_C_F);
        R_M_B = R_C_M0.transpose() * R_C_F;
        return true;
    }

    p_M_B = transform_cam_to_local_map(state_point.pos);
    R_M_B = map_world_initialized
                ? R_C_M0.transpose() * state_point.rot.toRotationMatrix()
                : state_point.rot.toRotationMatrix();
    return true;
}

inline bool get_current_output_pose(sc::Pose &pose, bool initialize_output_map = true)
{
    V3D pos_out = state_point.pos;
    M3D R_out = state_point.rot.toRotationMatrix();

    if (use_base_link_output_frame())
    {
        V3D p_C_F, v_C_F, w_C_F;
        M3D R_C_F;
        compute_base_link_pose_twist_in_cam_init(p_C_F, R_C_F, v_C_F, w_C_F);
        if (initialize_output_map)
            maybe_init_base_link_gravity_map(p_C_F, R_C_F);
        if (!map_world_initialized)
            return false;

        pos_out = transform_cam_to_map(p_C_F);
        R_out = R_C_M0.transpose() * R_C_F;
        if (prior_map_aligned)
            R_out = R_P_M * R_out;
    }

    const V3D rpy = rotation_matrix_to_rpy(R_out);
    pose.x = pos_out(0);
    pose.y = pos_out(1);
    pose.z = pos_out(2);
    pose.roll = rpy(0);
    pose.pitch = rpy(1);
    pose.yaw = normalize_yaw(rpy(2));
    return true;
}

inline Eigen::Matrix3d pose_rpy_to_rotation(const sc::Pose &pose)
{
    const Eigen::AngleAxisd roll_angle(pose.roll, Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd pitch_angle(pose.pitch, Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd yaw_angle(pose.yaw, Eigen::Vector3d::UnitZ());
    return (yaw_angle * pitch_angle * roll_angle).toRotationMatrix();
}

inline Eigen::Matrix4d pose_to_matrix(const sc::Pose &pose)
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = pose_rpy_to_rotation(pose);
    T(0, 3) = pose.x;
    T(1, 3) = pose.y;
    T(2, 3) = pose.z;
    return T;
}

inline Eigen::Quaterniond normalized_quaternion(const Eigen::Matrix3d &rotation)
{
    Eigen::Quaterniond q(rotation);
    q.normalize();
    return q;
}

inline void write_g2o_information(std::ostream &out)
{
    const double diagonal_info[6] = {1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0};
    for (int row = 0; row < 6; ++row)
    {
        for (int col = row; col < 6; ++col)
        {
            out << ' ' << (row == col ? diagonal_info[row] : 0.0);
        }
    }
}

inline bool initialize_manual_loop_export()
{
    if (!manual_loop_export_enable || use_prior_map)
        return true;
    if (manual_loop_export_initialized)
        return true;

    std::error_code ec;
    if (manual_loop_export_overwrite && !manual_loop_session_dir.empty())
    {
        std::filesystem::remove_all(manual_loop_session_dir, ec);
        if (ec)
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "Failed to clear manual loop session directory %s: %s",
                manual_loop_session_dir.c_str(), ec.message().c_str());
            ec.clear();
        }
    }

    std::filesystem::create_directories(manual_loop_keyframe_dir, ec);
    if (ec)
    {
        RCLCPP_ERROR(
            rclcpp::get_logger("laser_mapping"),
            "Failed to create manual loop keyframe directory %s: %s",
            manual_loop_keyframe_dir.c_str(), ec.message().c_str());
        return false;
    }

    {
        std::ofstream g2o(manual_loop_g2o_path, std::ios::trunc);
        if (!g2o)
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("laser_mapping"),
                "Failed to create manual loop pose graph: %s",
                manual_loop_g2o_path.c_str());
            return false;
        }
    }

    {
        std::ofstream tum(manual_loop_tum_path, std::ios::trunc);
        if (!tum)
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("laser_mapping"),
                "Failed to create manual loop TUM trajectory: %s",
                manual_loop_tum_path.c_str());
            return false;
        }
    }

    {
        std::ofstream gravity(manual_loop_gravity_path, std::ios::trunc);
        if (!gravity)
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("laser_mapping"),
                "Failed to create manual loop gravity sidecar: %s",
                manual_loop_gravity_path.c_str());
            return false;
        }
        gravity << "index,stamp,up_x,up_y,up_z\n";
    }

    const std::filesystem::path runtime_params_path =
        std::filesystem::path(manual_loop_session_dir) / "runtime_params.yaml";
    std::ofstream runtime_params(runtime_params_path);
    if (runtime_params)
    {
        runtime_params << std::boolalpha << std::setprecision(17)
                       << "saveResultBodyFrame: true\n"
                       << "keyframe_source: fast_lio_scan_context\n"
                       << "scan_context:\n"
                       << "  num_rings: " << scan_context_config.num_rings << "\n"
                       << "  num_sectors: " << scan_context_config.num_sectors << "\n"
                       << "  max_radius: " << scan_context_config.max_radius << "\n"
                       << "  dual_z_layer_enable: " << scan_context_config.dual_z_layer_enable << "\n"
                       << "  dual_z_split_height: " << scan_context_config.dual_z_split_height << "\n"
                       << "  dual_z_split_auto: " << scan_context_config.dual_z_split_auto << "\n"
                       << "  dual_z_split_auto_min: " << scan_context_config.dual_z_split_auto_min << "\n"
                       << "  dual_z_split_auto_max: " << scan_context_config.dual_z_split_auto_max << "\n"
                       << "  dual_z_split_auto_bin_size: " << scan_context_config.dual_z_split_auto_bin_size << "\n"
                       << "  dual_z_split_auto_histogram_max: " << scan_context_config.dual_z_split_auto_histogram_max << "\n"
                       << "  dual_z_split_auto_min_layer_fraction: " << scan_context_config.dual_z_split_auto_min_layer_fraction << "\n"
                       << "  dual_z_split_auto_min_keyframes: " << scan_context_config.dual_z_split_auto_min_keyframes << "\n"
                       << "  origin_height_from_ground: " << scan_context_config.origin_height_from_ground << "\n"
                       << "  ground_relative_dual_z_split_height: "
                       << sc::effectiveDualZSplitHeight(scan_context_config) << "\n"
                       << "  dual_z_low_weight: " << scan_context_config.dual_z_low_weight << "\n"
                       << "  dual_z_high_weight: " << scan_context_config.dual_z_high_weight << "\n"
                       << "  min_joint_rings: " << scan_context_config.min_joint_rings << "\n"
                       << "  absent_upper_fallback_max_local_fraction: " << scan_context_config.absent_upper_fallback_max_local_fraction << "\n"
                       << "  absent_upper_fallback_radius: " << scan_context_config.absent_upper_fallback_radius << "\n"
                       << "  absent_upper_fallback_min_keyframes: " << scan_context_config.absent_upper_fallback_min_keyframes << "\n"
                       << "  retrieval_height_offset: " << scan_context_config.retrieval_height_offset << "\n"
                       << "  sector_support_exponent: " << scan_context_config.sector_support_exponent << "\n"
                       << "  vertical_boundary_margin: " << scan_context_config.vertical_boundary_margin << "\n"
                       << "  gravity_canonicalization_enable: " << scan_context_config.gravity_canonicalized << "\n"
                       << "  vertical_estimation_enable: " << scan_context_config.vertical_estimation_enable << "\n"
                       << "  vertical_correction_min: " << scan_context_config.vertical_correction_min << "\n"
                       << "  vertical_correction_max: " << scan_context_config.vertical_correction_max << "\n"
                       << "  vertical_stable_fraction: " << scan_context_config.vertical_stable_fraction << "\n"
                       << "  voxel_leaf: " << scan_context_voxel_leaf << "\n"
                       << "  candidate_top_k: " << scan_context_config.candidate_top_k << "\n"
                       << "  yaw_top_k: " << scan_context_config.yaw_top_k << "\n"
                       << "  distance_thresh: " << scan_context_config.distance_thresh << "\n";
    }

    manual_loop_has_last_pose = false;
    manual_loop_export_initialized = true;
    RCLCPP_INFO(
        rclcpp::get_logger("laser_mapping"),
        "Manual loop session will be exported to %s",
        manual_loop_session_dir.c_str());
    return true;
}

inline bool append_manual_loop_keyframe(
    const int index,
    const double stamp,
    const sc::Pose &pose,
    const PointCloudXYZI::Ptr &keyframe_cloud,
    const V3D &gravity_up_body)
{
    if (!manual_loop_export_enable || use_prior_map)
        return true;
    if (!keyframe_cloud || keyframe_cloud->empty())
        return false;
    if (!initialize_manual_loop_export())
        return false;
    if (!gravity_up_body.allFinite() || gravity_up_body.squaredNorm() < 1e-12)
        return false;

    const std::filesystem::path pcd_path =
        std::filesystem::path(manual_loop_keyframe_dir) / (std::to_string(index) + ".pcd");
    const auto save_summary = pcd_save::writeBinary(pcd_path.string(), keyframe_cloud, 0.0);
    if (!save_summary.success)
    {
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "Failed to save manual loop keyframe %s: %s",
            pcd_path.string().c_str(), save_summary.error.c_str());
        return false;
    }

    const Eigen::Matrix4d T_current = pose_to_matrix(pose);
    const Eigen::Quaterniond q_current = normalized_quaternion(T_current.block<3, 3>(0, 0));

    {
        std::ofstream tum(manual_loop_tum_path, std::ios::app);
        if (!tum)
            return false;
        tum << std::fixed << std::setprecision(9)
            << stamp << ' '
            << pose.x << ' ' << pose.y << ' ' << pose.z << ' '
            << q_current.x() << ' ' << q_current.y() << ' ' << q_current.z() << ' ' << q_current.w()
            << '\n';
    }


    {
        const V3D up = gravity_up_body.normalized();
        std::ofstream gravity(manual_loop_gravity_path, std::ios::app);
        if (!gravity)
            return false;
        gravity << std::setprecision(17)
                << index << ',' << stamp << ','
                << up.x() << ',' << up.y() << ',' << up.z() << '\n';
    }

    {
        std::ofstream g2o(manual_loop_g2o_path, std::ios::app);
        if (!g2o)
            return false;
        g2o << std::setprecision(17)
            << "VERTEX_SE3:QUAT " << index << ' '
            << pose.x << ' ' << pose.y << ' ' << pose.z << ' '
            << q_current.x() << ' ' << q_current.y() << ' ' << q_current.z() << ' ' << q_current.w()
            << '\n';
        if (manual_loop_has_last_pose)
        {
            const Eigen::Matrix4d T_relative = manual_loop_last_T.inverse() * T_current;
            const Eigen::Quaterniond q_relative = normalized_quaternion(T_relative.block<3, 3>(0, 0));
            g2o << "EDGE_SE3:QUAT " << (index - 1) << ' ' << index << ' '
                << T_relative(0, 3) << ' ' << T_relative(1, 3) << ' ' << T_relative(2, 3) << ' '
                << q_relative.x() << ' ' << q_relative.y() << ' ' << q_relative.z() << ' ' << q_relative.w();
            write_g2o_information(g2o);
            g2o << '\n';
        }
    }

    manual_loop_last_pose = pose;
    manual_loop_last_T = T_current;
    manual_loop_has_last_pose = true;
    return true;
}

inline PointCloudXYZI::Ptr make_scan_context_body_cloud(const PointCloudXYZI::Ptr &scan_body)
{
    PointCloudXYZI::Ptr scan_context_cloud(new PointCloudXYZI());
    if (!scan_body)
        return scan_context_cloud;

    scan_context_cloud->reserve(scan_body->size());
    for (const auto &point : scan_body->points)
    {
        if (!valid_scan_context_input_point(point))
            continue;

        PointType out;
        if (use_base_link_output_frame())
            RGBpointBodyLidarToBaseLink(&point, &out);
        else
            RGBpointBodyLidarToIMU(&point, &out);
        if (!point_xyz_finite(out))
            continue;
        scan_context_cloud->push_back(out);
    }
    return scan_context_cloud;
}

inline PointCloudXYZI::Ptr downsample_scan_context_cloud(const PointCloudXYZI::Ptr &cloud)
{
    if (!cloud || cloud->empty() || scan_context_voxel_leaf <= 1e-3)
        return cloud;

    PointCloudXYZI::Ptr filtered(new PointCloudXYZI());
    pcl::VoxelGrid<PointType> voxel_filter;
    voxel_filter.setLeafSize(
        static_cast<float>(scan_context_voxel_leaf),
        static_cast<float>(scan_context_voxel_leaf),
        static_cast<float>(scan_context_voxel_leaf));
    voxel_filter.setInputCloud(cloud);
    voxel_filter.filter(*filtered);
    return filtered;
}

inline Eigen::Matrix4f adapt_prior_seed_to_local_source(const Eigen::Matrix4f &T_P_B_f)
{
    if (!prior_icp_source_ref_pose_valid)
        return T_P_B_f;

    Eigen::Matrix4d T_M_B = Eigen::Matrix4d::Identity();
    T_M_B.block<3, 3>(0, 0) = prior_icp_source_ref_R_M_B;
    T_M_B.block<3, 1>(0, 3) = prior_icp_source_ref_p_M_B;

    const Eigen::Matrix4d T_P_B = T_P_B_f.cast<double>();
    const Eigen::Matrix4d T_P_M = T_P_B * T_M_B.inverse();
    return T_P_M.cast<float>();
}

inline bool save_scan_context_database_to_disk(bool warn_if_empty)
{
    if (!scan_context_enable || use_prior_map)
        return true;

    if (scan_context_config.dual_z_layer_enable &&
        scan_context_config.dual_z_split_auto &&
        !scan_context_pending_keyframes.empty())
    {
        const sc::AdaptiveSplitResult estimate =
            scan_context_split_estimator.estimate();
        sc::Config finalized_config = scan_context_config;
        finalized_config.dual_z_layer_enable =
            estimate.dual_layer_enabled;
        finalized_config.dual_z_split_height = estimate.split_height;
        scan_context_config.dual_z_layer_enable =
            estimate.dual_layer_enabled;
        scan_context_config.dual_z_split_height = estimate.split_height;
        scan_context_db.clear();
        scan_context_db.setConfig(finalized_config);
        for (const auto &keyframe : scan_context_pending_keyframes)
        {
            if (!keyframe.cloud || keyframe.cloud->empty())
                continue;
            scan_context_db.addEntry(
                keyframe.stamp, keyframe.pose,
                scan_context_db.makeDescriptor(*keyframe.cloud));
        }
        RCLCPP_INFO(
            rclcpp::get_logger("laser_mapping"),
            "Finalized adaptive Scan Context split: selected=%.2fm "
            "dual_layer=%s adapted=%s keyframes=%zu support=%" PRIu64
            " lower=%.1f%% upper=%.1f%%",
            estimate.split_height,
            estimate.dual_layer_enabled ? "true" : "false",
            estimate.adapted ? "true" : "false",
            estimate.keyframe_count,
            estimate.support_count,
            100.0 * estimate.lower_fraction,
            100.0 * estimate.upper_fraction);
    }

    if (scan_context_db.empty())
    {
        if (warn_if_empty)
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "Scan Context database is empty; skip saving %s",
                scan_context_database_path.c_str());
        }
        return false;
    }

    string error;
    if (!scan_context_db.save(scan_context_database_path, &error))
    {
        RCLCPP_ERROR(
            rclcpp::get_logger("laser_mapping"),
            "Failed to save Scan Context database to %s: %s",
            scan_context_database_path.c_str(), error.c_str());
        return false;
    }

    scan_context_dirty = false;
    RCLCPP_INFO(
        rclcpp::get_logger("laser_mapping"),
        "Saved Scan Context database: %s entries=%zu",
        scan_context_database_path.c_str(), scan_context_db.size());
    return true;
}

inline bool load_prior_map_from_pcd(const string &pcd_path)
{
    if (pcd_path.empty())
    {
        RCLCPP_WARN(rclcpp::get_logger("laser_mapping"), "prior_map enabled but map_file_path is empty.");
        return false;
    }

    const string resolved_pcd_path = resolve_prior_map_path(pcd_path);
    PointCloudXYZI::Ptr loaded(new PointCloudXYZI());
    if (pcl::io::loadPCDFile<PointType>(resolved_pcd_path, *loaded) < 0)
    {
        RCLCPP_ERROR(rclcpp::get_logger("laser_mapping"), "Failed to load prior map PCD: %s", resolved_pcd_path.c_str());
        return false;
    }

    if (loaded->empty())
    {
        RCLCPP_ERROR(rclcpp::get_logger("laser_mapping"), "Prior map PCD is empty: %s", resolved_pcd_path.c_str());
        return false;
    }

    PointCloudXYZI::Ptr loaded_finite(new PointCloudXYZI());
    loaded_finite->reserve(loaded->size());
    for (const auto &point : loaded->points)
    {
        if (point_xyz_finite(point))
            loaded_finite->push_back(point);
    }
    loaded_finite->width = loaded_finite->size();
    loaded_finite->height = 1;
    loaded_finite->is_dense = true;
    if (loaded_finite->empty())
    {
        RCLCPP_ERROR(
            rclcpp::get_logger("laser_mapping"),
            "Prior map PCD has no finite XYZ points: %s",
            resolved_pcd_path.c_str());
        return false;
    }
    if (loaded_finite->size() != loaded->size())
    {
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "Prior map PCD contains non-finite points; using %zu/%zu finite points.",
            loaded_finite->size(), loaded->size());
    }

    const double coarse_leaf = (prior_map_voxel_leaf > 1e-3) ? prior_map_voxel_leaf : filter_size_map_min;
    const double fine_leaf = (prior_map_voxel_leaf_fine > 1e-3) ? prior_map_voxel_leaf_fine : coarse_leaf;

    pcl::VoxelGrid<PointType> coarse_filter;
    coarse_filter.setLeafSize(coarse_leaf, coarse_leaf, coarse_leaf);
    coarse_filter.setInputCloud(loaded_finite);
    coarse_filter.filter(*prior_map_cloud_coarse);

    pcl::VoxelGrid<PointType> fine_filter;
    fine_filter.setLeafSize(fine_leaf, fine_leaf, fine_leaf);
    fine_filter.setInputCloud(loaded_finite);
    fine_filter.filter(*prior_map_cloud_fine);
    *prior_map_cloud = *prior_map_cloud_fine;

    if (prior_map_cloud_coarse->empty() || prior_map_cloud_fine->empty())
    {
        RCLCPP_ERROR(rclcpp::get_logger("laser_mapping"), "Prior map became empty after voxel filtering.");
        return false;
    }

    prior_map_loaded = true;
    RCLCPP_INFO(
        rclcpp::get_logger("laser_mapping"),
        "Loaded prior map: raw=%zu, coarse=%zu (leaf=%.3f), fine=%zu (leaf=%.3f)",
        loaded_finite->size(), prior_map_cloud_coarse->size(), coarse_leaf,
        prior_map_cloud_fine->size(), fine_leaf);
    return true;
}

inline vector<Eigen::Matrix4f> build_prior_icp_seeds(vector<sc::Candidate> *scan_context_candidates = nullptr)
{
    vector<Eigen::Matrix4f> seeds;
    if (scan_context_candidates)
        scan_context_candidates->clear();

    if (scan_context_enable && scan_context_loaded)
    {
        PointCloudXYZI::Ptr query_cloud = prior_icp_seed_cloud;
        if (scan_context_db.config().gravity_canonicalized)
        {
            if (!prior_icp_source_gravity_rotation_valid)
            {
                RCLCPP_WARN(
                    rclcpp::get_logger("laser_mapping"),
                    "Scan Context query has no valid gravity rotation.");
                return seeds;
            }
            query_cloud.reset(new PointCloudXYZI(
                sc::gravityCanonicalize(*query_cloud, prior_icp_source_R_G_B)));
        }
        query_cloud = downsample_scan_context_cloud(query_cloud);
        if (!query_cloud || query_cloud->empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("laser_mapping"), "Scan Context query cloud is empty.");
            return seeds;
        }

        const sc::Descriptor query_descriptor =
            scan_context_query_builder.makeDescriptor(*query_cloud);
        const auto candidates = scan_context_db.queryWithVerticalEstimation(
            query_descriptor, true);
        if (scan_context_candidates)
            *scan_context_candidates = candidates;
        if (candidates.empty())
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "Scan Context found no candidate below distance threshold %.3f.",
                scan_context_db.config().distance_thresh);
            return seeds;
        }

        const double xy_offset = std::max(0.0, scan_context_seed_xy_offset);
        const vector<double> xy_offsets =
            (xy_offset > 1e-3) ? vector<double>{-xy_offset, 0.0, xy_offset} : vector<double>{0.0};
        seeds.reserve(
            candidates.size() *
            static_cast<std::size_t>(std::max(1, scan_context_db.config().yaw_top_k)) *
            xy_offsets.size() * xy_offsets.size());
        for (const auto &candidate : candidates)
        {
            vector<sc::YawMatch> yaw_matches = candidate.yaw_matches;
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

            append_scan_context_candidate_log(candidate, yaw_matches.size(), xy_offset);

            for (const auto &yaw_match : yaw_matches)
            {
                const double seed_yaw = sc::makeCandidateSeedYaw(
                    candidate.pose.canonical_yaw, yaw_match.yaw_shift_rad);
                const Eigen::AngleAxisd yaw_rotation(seed_yaw, Eigen::Vector3d::UnitZ());
                const M3D seed_rotation =
                    yaw_rotation.toRotationMatrix() * prior_icp_source_R_G_B;
                for (const double dx : xy_offsets)
                {
                    for (const double dy : xy_offsets)
                    {
                        seeds.push_back(prior_icp::makeSeedTransform(
                            candidate.pose.x + dx, candidate.pose.y + dy,
                            candidate.pose.z + yaw_match.vertical_shift,
                            seed_rotation));
                    }
                }
            }
        }
        return seeds;
    }

    seeds.reserve(100);
    const double base_x = prior_initial_guess_xy.size() > 0 ? prior_initial_guess_xy[0] : 0.0;
    const double base_y = prior_initial_guess_xy.size() > 1 ? prior_initial_guess_xy[1] : 0.0;
    const double base_yaw = prior_initial_guess_yaw_deg * M_PI / 180.0;
    auto make_seed = [base_x, base_y, base_yaw](double dx, double dy, double yaw_offset_deg) {
        const double yaw = base_yaw + yaw_offset_deg * M_PI / 180.0;
        const Eigen::AngleAxisd yaw_rotation(yaw, Eigen::Vector3d::UnitZ());
        const M3D seed_rotation =
            yaw_rotation.toRotationMatrix() * prior_icp_source_R_G_B;
        return prior_icp::makeSeedTransform(base_x + dx, base_y + dy, 0.0, seed_rotation);
    };

    if (!prior_multi_seed_enable)
    {
        seeds.push_back(make_seed(0.0, 0.0, 0.0));
        return seeds;
    }

    const double xy_step = max(0.1, prior_seed_xy_step);
    const double yaw_step = max(1.0, prior_seed_yaw_step_deg);
    for (double yaw = -prior_seed_yaw_range_deg; yaw <= prior_seed_yaw_range_deg + 1e-6; yaw += yaw_step)
    {
        for (double dx = -prior_seed_xy_range; dx <= prior_seed_xy_range + 1e-6; dx += xy_step)
        {
            for (double dy = -prior_seed_xy_range; dy <= prior_seed_xy_range + 1e-6; dy += xy_step)
            {
                seeds.push_back(make_seed(dx, dy, yaw));
            }
        }
    }

    seeds.push_back(make_seed(0.0, 0.0, 0.0));
    return seeds;
}

inline void add_colored_candidate_point(
    pcl::PointCloud<pcl::PointXYZRGB> &cloud,
    double x, double y, double z,
    uint8_t r, uint8_t g, uint8_t b)
{
    pcl::PointXYZRGB point;
    point.x = static_cast<float>(x);
    point.y = static_cast<float>(y);
    point.z = static_cast<float>(z);
    point.r = r;
    point.g = g;
    point.b = b;
    cloud.push_back(point);
}

inline pcl::PointCloud<pcl::PointXYZRGB>::Ptr make_scan_context_candidate_cloud(
    const vector<sc::Candidate> &candidates)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    const int candidate_count = std::min(3, static_cast<int>(candidates.size()));
    const uint8_t colors[3][3] = {
        {0, 255, 80},
        {255, 220, 0},
        {255, 60, 60},
    };
    const double offsets[][3] = {
        {0.0, 0.0, 0.0},
        {0.25, 0.0, 0.0},
        {-0.25, 0.0, 0.0},
        {0.0, 0.25, 0.0},
        {0.0, -0.25, 0.0},
        {0.0, 0.0, 0.25},
        {0.0, 0.0, -0.25},
    };

    for (int i = 0; i < candidate_count; ++i)
    {
        const auto &candidate = candidates[i];
        const double z = candidate.pose.z + 0.5;
        for (const auto &offset : offsets)
        {
            add_colored_candidate_point(
                *cloud,
                candidate.pose.x + offset[0],
                candidate.pose.y + offset[1],
                z + offset[2],
                colors[i][0], colors[i][1], colors[i][2]);
        }
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

inline void publish_scan_context_candidate_cloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &publisher,
    const vector<sc::Candidate> &candidates)
{
    if (!publisher)
        return;

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*make_scan_context_candidate_cloud(candidates), msg);
    msg.header.stamp = pose_output_stamp();
    msg.header.frame_id = base_link_world_frame_id;
    publisher->publish(msg);
}

inline bool try_align_prior_map_and_build_tree(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubScanContextCandidates)
{
    if (!use_prior_map || prior_map_build_done || !prior_map_loaded || !map_world_initialized)
        return false;

    if (!prior_icp_source_frozen || static_cast<int>(prior_icp_source_cloud->size()) < prior_icp_min_points)
        return false;

    const double relocalization_start_time = omp_get_wtime();

    PointCloudXYZI::Ptr source_coarse(new PointCloudXYZI());
    PointCloudXYZI::Ptr source_fine(new PointCloudXYZI());
    const double coarse_leaf = (prior_map_voxel_leaf > 1e-3) ? prior_map_voxel_leaf : filter_size_map_min;
    const double fine_leaf = (prior_map_voxel_leaf_fine > 1e-3) ? prior_map_voxel_leaf_fine : coarse_leaf;
    pcl::VoxelGrid<PointType> voxel_filter;
    voxel_filter.setLeafSize(coarse_leaf, coarse_leaf, coarse_leaf);
    voxel_filter.setInputCloud(prior_icp_source_cloud);
    voxel_filter.filter(*source_coarse);
    voxel_filter.setLeafSize(fine_leaf, fine_leaf, fine_leaf);
    voxel_filter.setInputCloud(prior_icp_source_cloud);
    voxel_filter.filter(*source_fine);

    if (source_coarse->size() < 50 || source_fine->size() < 50)
    {
        prior_icp_fail_count++;
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "\033[1;32mRelocalization failed\033[0m reason=too_few_downsampled_source_points source_raw=%zu source_coarse=%zu source_fine=%zu min_required=50 fails=%d",
            prior_icp_source_cloud->size(), source_coarse->size(), source_fine->size(), prior_icp_fail_count);
        return false;
    }

    vector<sc::Candidate> scan_context_candidates;
    const double scan_context_start_time = omp_get_wtime();
    const auto seeds = build_prior_icp_seeds(&scan_context_candidates);
    const double scan_context_cost_ms =
        (scan_context_enable && scan_context_loaded)
            ? (omp_get_wtime() - scan_context_start_time) * 1000.0
            : 0.0;
    if (scan_context_enable && scan_context_loaded)
    {
        scan_context_last_icp_candidates = scan_context_candidates;
        scan_context_candidate_cloud_active = !scan_context_last_icp_candidates.empty();
        if (scan_context_candidate_cloud_active)
            scan_context_candidate_cloud_expire_time = omp_get_wtime() + 20.0;
        publish_scan_context_candidate_cloud(pubScanContextCandidates, scan_context_last_icp_candidates);
        static bool logged_candidate_cloud_topic = false;
        if (scan_context_candidate_cloud_active && !logged_candidate_cloud_topic)
        {
            logged_candidate_cloud_topic = true;
            RCLCPP_INFO(
                rclcpp::get_logger("laser_mapping"),
                "Published %zu Scan Context ICP candidate points on /scan_context_icp_candidates.",
                std::min<std::size_t>(3, scan_context_last_icp_candidates.size()));
        }
    }
    if (seeds.empty())
    {
        prior_icp_fail_count++;
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "\033[1;32mRelocalization failed\033[0m reason=no_initial_seeds scan_context_enable=%s scan_context_loaded=%s source_points=%zu fails=%d",
            scan_context_enable ? "true" : "false",
            scan_context_loaded ? "true" : "false",
            prior_icp_source_cloud->size(), prior_icp_fail_count);
        return false;
    }

    const double icp_start_time = omp_get_wtime();
    const prior_icp::Config icp_config{
        prior_icp_max_iterations,
        prior_icp_max_corr_dist,
        prior_icp_min_overlap_ratio,
    };
    vector<int> all_seed_indices;
    all_seed_indices.reserve(seeds.size());
    for (int i = 0; i < static_cast<int>(seeds.size()); ++i)
        all_seed_indices.push_back(i);

    int coarse_converged_count = 0;
    int coarse_valid_count = 0;
    const auto coarse_results = prior_icp::runStage(
        icp_config,
        source_coarse, prior_map_cloud_coarse, seeds, all_seed_indices,
        coarse_converged_count, coarse_valid_count);

    if (coarse_results.empty())
    {
        prior_icp_fail_count++;
        if (coarse_converged_count > 0)
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "\033[1;32mRelocalization failed\033[0m reason=coarse_overlap_rejected seeds=%zu converged=%d required_overlap=%.3f source_coarse=%zu target_coarse=%zu fails=%d",
                seeds.size(), coarse_converged_count, prior_icp_min_overlap_ratio,
                source_coarse->size(), prior_map_cloud_coarse->size(), prior_icp_fail_count);
        }
        else
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "\033[1;32mRelocalization failed\033[0m reason=coarse_not_converged seeds=%zu source_coarse=%zu target_coarse=%zu fails=%d",
                seeds.size(), source_coarse->size(), prior_map_cloud_coarse->size(), prior_icp_fail_count);
        }
        return false;
    }

    const int refine_count = std::min(
        std::max(1, prior_icp_refine_top_k),
        static_cast<int>(coarse_results.size()));
    vector<int> refine_seed_indices;
    refine_seed_indices.reserve(refine_count);
    for (int i = 0; i < refine_count; ++i)
        refine_seed_indices.push_back(coarse_results[i].seed_index);

    int fine_converged_count = 0;
    int fine_valid_count = 0;
    const auto fine_results = prior_icp::runStage(
        icp_config,
        source_fine, prior_map_cloud, seeds, refine_seed_indices,
        fine_converged_count, fine_valid_count);
    const double icp_cost_ms = (omp_get_wtime() - icp_start_time) * 1000.0;

    if (fine_results.empty())
    {
        prior_icp_fail_count++;
        if (fine_converged_count > 0)
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "\033[1;32mRelocalization failed\033[0m reason=fine_overlap_rejected refined=%d converged=%d required_overlap=%.3f source_fine=%zu target_fine=%zu fails=%d",
                refine_count, fine_converged_count, prior_icp_min_overlap_ratio,
                source_fine->size(), prior_map_cloud->size(), prior_icp_fail_count);
        }
        else
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "\033[1;32mRelocalization failed\033[0m reason=fine_not_converged refined=%d source_fine=%zu target_fine=%zu fails=%d",
                refine_count, source_fine->size(), prior_map_cloud->size(), prior_icp_fail_count);
        }
        return false;
    }

    const auto &best_result = fine_results.front();
    const double best_fitness = best_result.fitness;
    const double best_overlap_ratio = best_result.overlap;
    const int best_seed_idx = best_result.seed_index;
    if (best_fitness > prior_icp_fitness_thresh)
    {
        prior_icp_fail_count++;
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "\033[1;32mRelocalization failed\033[0m reason=fitness_too_high best_fitness=%.6f threshold=%.6f best_overlap=%.3f best_seed=%d fails=%d",
            best_fitness, prior_icp_fitness_thresh, best_overlap_ratio, best_seed_idx, prior_icp_fail_count);
        return false;
    }

    const Eigen::Matrix4f T_P_M_f = adapt_prior_seed_to_local_source(best_result.transform);
    R_P_M = T_P_M_f.block<3,3>(0,0).cast<double>();
    p_P_M = T_P_M_f.block<3,1>(0,3).cast<double>();

    PointCloudXYZI::Ptr prior_map_cam_init(new PointCloudXYZI());
    prior_map_cam_init->reserve(prior_map_cloud->size());
    for (const auto &pt_P : prior_map_cloud->points)
    {
        V3D p_P(pt_P.x, pt_P.y, pt_P.z);
        V3D p_M = R_P_M.transpose() * (p_P - p_P_M);
        V3D p_C = R_C_M0 * p_M + p_C_M0;
        PointType out;
        out.x = p_C(0);
        out.y = p_C(1);
        out.z = p_C(2);
        out.intensity = pt_P.intensity;
        if (!point_xyz_finite(out))
            continue;
        prior_map_cam_init->push_back(out);
    }

    if (prior_map_cam_init->size() < 10)
    {
        RCLCPP_ERROR(rclcpp::get_logger("laser_mapping"), "Transformed prior map has too few points for ikdtree build.");
        return false;
    }

    ikdtree.set_downsample_param(filter_size_map_min);
    ikdtree.Build(prior_map_cam_init->points);
    prior_map_aligned = true;
    ikdtree_built = true;
    prior_map_build_done = true;
    localization_output_trusted = true;
    prior_map_ready_for_publish = true;
    prior_icp_fail_count = 0;

    init_feats_buffer->clear();
    init_feats_buffer_local->clear();

    const double relocalization_cost_ms = (omp_get_wtime() - relocalization_start_time) * 1000.0;
    sc::Pose relocalized_pose;
    if (!get_current_output_pose(relocalized_pose))
    {
        V3D p_C_F, v_C_F, w_C_F;
        M3D R_C_F;
        compute_base_link_pose_twist_in_cam_init(p_C_F, R_C_F, v_C_F, w_C_F);
        const V3D p_base_link_map = transform_cam_to_map(p_C_F);
        relocalized_pose.x = p_base_link_map(0);
        relocalized_pose.y = p_base_link_map(1);
        relocalized_pose.z = p_base_link_map(2);
    }
    RCLCPP_INFO(
        rclcpp::get_logger("laser_mapping"),
        "\033[1;32mRelocalization result: success time_ms=%.1f scan_context_ms=%.1f icp_ms=%.1f frame=%s window=[%.3f %.3f] frames=[%" PRIu64 " %" PRIu64 "] pose_xy_yaw=[%.3f %.3f %.2f deg] seeds=%zu refine_top=%d coarse_valid=%d fine_valid=%d best_seed=%d fitness=%.6f overlap=%.3f source_points=%zu target_points=%zu tree_points=%zu. Mapping update enabled.\033[0m",
        relocalization_cost_ms,
        scan_context_cost_ms,
        icp_cost_ms,
        base_link_world_frame_id.c_str(),
        prior_icp_source_start_time, prior_icp_source_end_time,
        prior_icp_source_first_frame, prior_icp_source_last_frame,
        relocalized_pose.x, relocalized_pose.y,
        relocalized_pose.yaw * 180.0 / M_PI,
        seeds.size(), refine_count, coarse_valid_count, fine_valid_count,
        best_seed_idx, best_fitness, best_overlap_ratio,
        source_fine->size(), prior_map_cloud->size(), prior_map_cam_init->size());
    clear_prior_icp_active_source();
    clear_prior_icp_accum_window();
    begin_prior_replay_after_relocalization();
    return true;
}

inline void publish_prior_map_once(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubPriorMap)
{
    if (!prior_map_ready_for_publish || prior_map_pub_once_done || prior_map_cloud->empty())
        return;

    sensor_msgs::msg::PointCloud2 prior_map_msg;
    pcl::toROSMsg(*prior_map_cloud, prior_map_msg);
    prior_map_msg.header.stamp = pose_output_stamp();
    prior_map_msg.header.frame_id = base_link_world_frame_id;
    pubPriorMap->publish(prior_map_msg);

    prior_map_pub_once_done = true;
    prior_map_ready_for_publish = false;
    RCLCPP_INFO(
        rclcpp::get_logger("laser_mapping"),
        "Published prior_map once, points=%zu, frame=%s",
        prior_map_cloud->size(), base_link_world_frame_id.c_str());
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    static uint64_t lidar_rx_index = 0;
    static double last_received_lidar_time = -1.0;
    static double last_received_lidar_end_time = -1.0;
    ++lidar_rx_index;
    static bool fields_checked = false;
    if (!fields_checked)
    {
        msg_is_XYZI = (msg->fields.size() == 4);
        msg_is_XYZIRT = (msg->fields.size() >= 6);
        if (msg_is_XYZIRT)
        {
            RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "AIRY PointCloud2 detected as XYZIRT (%u x %u).",
                        msg->width, msg->height);
        }
        else if (msg_is_XYZI)
        {
            RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                        "AIRY PointCloud2 detected as XYZI only; deskew will be disabled for lidar_type=RSAIRY.");
        }
        else
        {
            RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                        "Unexpected AIRY PointCloud2 fields count=%zu.", msg->fields.size());
        }
        fields_checked = true;
    }
    const double header_time = get_time_sec(msg->header.stamp);
    double pointcloud_begin_time = 0.0, pointcloud_end_time = 0.0;
    const bool has_pointcloud_time_bounds =
        (p_pre->lidar_type == MID360 &&
         get_pointcloud_timestamp_bounds(*msg, p_pre->time_unit, pointcloud_begin_time, pointcloud_end_time) &&
         pointcloud_end_time > pointcloud_begin_time);
    double cur_time = has_pointcloud_time_bounds ? pointcloud_begin_time : header_time;
    double cur_end_time = has_pointcloud_time_bounds ? pointcloud_end_time : 0.0;
    const std::size_t point_count =
        static_cast<std::size_t>(msg->width) * static_cast<std::size_t>(msg->height);
    if (!std::isfinite(cur_time) || (cur_end_time != 0.0 && !std::isfinite(cur_end_time)))
    {
        static int invalid_lidar_time_report_count = 0;
        if (invalid_lidar_time_report_count < 20)
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "discard invalid LiDAR PointCloud2 timestamp: begin=%.9f end=%.9f rx=%" PRIu64,
                cur_time, cur_end_time, lidar_rx_index);
            ++invalid_lidar_time_report_count;
        }
        return;
    }

    {
        std::lock_guard<mutex> lock_buffer(mtx_buffer);
        if (prior_relocalization_sensor_restart_time >= 0.0 &&
            cur_time <= prior_relocalization_sensor_restart_time)
        {
            note_timestamp_rollback_for_localization_restart();
            return;
        }
        if (!is_first_lidar && cur_time < last_timestamp_lidar)
        {
            lidar_timestamp_rollback_events.fetch_add(1, std::memory_order_relaxed);
            note_timestamp_rollback_for_localization_restart();
            static int early_lidar_loopback_drop_report_count = 0;
            if (early_lidar_loopback_drop_report_count < 20)
            {
                RCLCPP_WARN(
                    rclcpp::get_logger("laser_mapping"),
                    "discard out-of-order LiDAR frame before buffering: current=%.9f < last=%.9f rx=%" PRIu64,
                    cur_time, last_timestamp_lidar, lidar_rx_index);
                ++early_lidar_loopback_drop_report_count;
            }
            return;
        }
    }

    warn_lidar_frame_gap_if_needed(
        "PointCloud2", lidar_rx_index,
        last_received_lidar_time, last_received_lidar_end_time,
        cur_time, cur_end_time, point_count, lidar_frame_period_sec);
    last_received_lidar_time = cur_time;
    last_received_lidar_end_time = (cur_end_time > cur_time) ? cur_end_time : cur_time;
    double receive_start_time = omp_get_wtime();

    {
        std::lock_guard<mutex> lock_buffer(mtx_buffer);
        if (prior_relocalization_sensor_restart_time >= 0.0 &&
            cur_time <= prior_relocalization_sensor_restart_time)
        {
            note_timestamp_rollback_for_localization_restart();
            return;
        }

        scan_count ++;
        if (!is_first_lidar && cur_time < last_timestamp_lidar)
        {
            lidar_timestamp_rollback_events.fetch_add(1, std::memory_order_relaxed);
            note_timestamp_rollback_for_localization_restart();
            static int lidar_loopback_drop_report_count = 0;
            if (lidar_loopback_drop_report_count < 20)
            {
                RCLCPP_WARN(
                    rclcpp::get_logger("laser_mapping"),
                    "discard out-of-order LiDAR frame: current=%.9f < last=%.9f rx=%" PRIu64,
                    cur_time, last_timestamp_lidar, lidar_rx_index);
                ++lidar_loopback_drop_report_count;
            }
            return;
        }
        if (is_first_lidar)
        {
            is_first_lidar = false;
        }

        raw_pcl_buffer.push_back(msg);
        time_buffer.push_back(cur_time);
        lidar_end_time_buffer.push_back(cur_end_time);
        lidar_rx_index_buffer.push_back(lidar_rx_index);
        last_timestamp_lidar = cur_time;
        clear_timestamp_rollback_streak();
        record_preprocess_time_sample(scan_count, omp_get_wtime() - receive_start_time);
    }
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
{
    static uint64_t livox_rx_index = 0;
    static double last_received_livox_time = -1.0;
    static double last_received_livox_end_time = -1.0;
    ++livox_rx_index;
    double cur_time = get_time_sec(msg->header.stamp);
    if (!std::isfinite(cur_time))
    {
        static int invalid_livox_time_report_count = 0;
        if (invalid_livox_time_report_count < 20)
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "discard invalid Livox timestamp: begin=%.9f rx=%" PRIu64,
                cur_time, livox_rx_index);
            ++invalid_livox_time_report_count;
        }
        return;
    }
    {
        std::lock_guard<mutex> lock_buffer(mtx_buffer);
        if (prior_relocalization_sensor_restart_time >= 0.0 &&
            cur_time <= prior_relocalization_sensor_restart_time)
        {
            note_timestamp_rollback_for_localization_restart();
            return;
        }
        if (!is_first_lidar && cur_time < last_timestamp_lidar)
        {
            lidar_timestamp_rollback_events.fetch_add(1, std::memory_order_relaxed);
            note_timestamp_rollback_for_localization_restart();
            static int early_livox_loopback_drop_report_count = 0;
            if (early_livox_loopback_drop_report_count < 20)
            {
                RCLCPP_WARN(
                    rclcpp::get_logger("laser_mapping"),
                    "discard out-of-order LiDAR frame before buffering: current=%.9f < last=%.9f rx=%" PRIu64,
                    cur_time, last_timestamp_lidar, livox_rx_index);
                ++early_livox_loopback_drop_report_count;
            }
            return;
        }
    }

    double preprocess_start_time = omp_get_wtime();
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    auto msg_copy = std::make_unique<livox_ros_driver2::msg::CustomMsg>(*msg);
    {
        std::lock_guard<mutex> lock_preprocess(mtx_preprocess);
        p_pre->process(msg_copy, ptr);
    }

    warn_lidar_frame_gap_if_needed(
        "CustomMsg", livox_rx_index,
        last_received_livox_time, last_received_livox_end_time,
        cur_time, 0.0, ptr ? ptr->size() : 0, lidar_frame_period_sec);
    last_received_livox_time = cur_time;
    last_received_livox_end_time = cur_time;

    {
        std::lock_guard<mutex> lock_buffer(mtx_buffer);
        if (prior_relocalization_sensor_restart_time >= 0.0 &&
            cur_time <= prior_relocalization_sensor_restart_time)
        {
            note_timestamp_rollback_for_localization_restart();
            return;
        }

        scan_count ++;
        if (!is_first_lidar && cur_time < last_timestamp_lidar)
        {
            lidar_timestamp_rollback_events.fetch_add(1, std::memory_order_relaxed);
            note_timestamp_rollback_for_localization_restart();
            static int livox_loopback_drop_report_count = 0;
            if (livox_loopback_drop_report_count < 20)
            {
                RCLCPP_WARN(
                    rclcpp::get_logger("laser_mapping"),
                    "discard out-of-order LiDAR frame: current=%.9f < last=%.9f rx=%" PRIu64,
                    cur_time, last_timestamp_lidar, livox_rx_index);
                ++livox_loopback_drop_report_count;
            }
            return;
        }
        if(is_first_lidar)
        {
            is_first_lidar = false;
        }
        last_timestamp_lidar = cur_time;

        if (!time_sync_en && std::fabs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
        {
            static int lidar_imu_unsync_report_count = 0;
            if (lidar_imu_unsync_report_count < 10)
            {
                RCLCPP_WARN(
                    rclcpp::get_logger("laser_mapping"),
                    "IMU and LiDAR not synced: imu_time=%.9f lidar_time=%.9f",
                    last_timestamp_imu, last_timestamp_lidar);
                ++lidar_imu_unsync_report_count;
            }
        }

        if (time_sync_en && !timediff_set_flg && std::fabs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
        {
            timediff_set_flg = true;
            timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
            RCLCPP_INFO(
                rclcpp::get_logger("laser_mapping"),
                "Self-sync IMU and LiDAR: time_diff=%.10f",
                timediff_lidar_wrt_imu);
        }

        lidar_buffer.push_back(ptr);
        time_buffer.push_back(last_timestamp_lidar);
        lidar_end_time_buffer.push_back(0.0);
        lidar_rx_index_buffer.push_back(livox_rx_index);
        clear_timestamp_rollback_streak();

        record_preprocess_time_sample(scan_count, omp_get_wtime() - preprocess_start_time);
    }
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::msg::Imu::SharedPtr msg_in)
{
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

    if (p_pre->lidar_type == RSAIRY && airy_imu_flip_yz)
    {
        msg->angular_velocity.y *= -1.0;
        msg->angular_velocity.z *= -1.0;
        msg->linear_acceleration.y *= -1.0;
        msg->linear_acceleration.z *= -1.0;
    }

    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    if (std::fabs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = get_ros_time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
    }

    double timestamp = get_time_sec(msg->header.stamp);
    const bool imu_values_finite =
        std::isfinite(timestamp) &&
        std::isfinite(msg->angular_velocity.x) &&
        std::isfinite(msg->angular_velocity.y) &&
        std::isfinite(msg->angular_velocity.z) &&
        std::isfinite(msg->linear_acceleration.x) &&
        std::isfinite(msg->linear_acceleration.y) &&
        std::isfinite(msg->linear_acceleration.z);
    if (!imu_values_finite)
    {
        static int invalid_imu_report_count = 0;
        if (invalid_imu_report_count < 20)
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "discard invalid IMU msg with non-finite timestamp/gyro/accel");
            ++invalid_imu_report_count;
        }
        return;
    }

    static double last_received_imu_time = -1.0;
    warn_imu_frame_gap_if_needed(last_received_imu_time, timestamp);
    if (timestamp > last_received_imu_time)
        last_received_imu_time = timestamp;

    {
        std::unique_lock<mutex> lock_buffer(mtx_buffer);

        if (prior_relocalization_sensor_restart_time >= 0.0 &&
            timestamp <= prior_relocalization_sensor_restart_time)
        {
            note_timestamp_rollback_for_localization_restart();
            return;
        }

        if (timestamp <= last_used_imu_time)
        {
            note_timestamp_rollback_for_localization_restart();
            static int stale_imu_report_count = 0;
            if (stale_imu_report_count < 10)
            {
                RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                            "discard stale IMU msg: current %.9f <= used %.9f",
                            timestamp, last_used_imu_time);
                ++stale_imu_report_count;
            }
            return;
        }

        if (last_timestamp_imu >= 0.0 &&
            timestamp + SENSOR_SEVERE_OUT_OF_ORDER_DROP_SEC < last_timestamp_imu)
        {
            imu_timestamp_rollback_events.fetch_add(1, std::memory_order_relaxed);
            note_timestamp_rollback_for_localization_restart();
            static int severe_out_of_order_imu_report_count = 0;
            if (severe_out_of_order_imu_report_count < 20)
            {
                RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                            "discard severely out-of-order IMU msg: current %.9f < newest %.9f",
                            timestamp, last_timestamp_imu);
                ++severe_out_of_order_imu_report_count;
            }
            return;
        }

        last_timestamp_imu = std::max(last_timestamp_imu, timestamp);

        auto insert_pos = std::upper_bound(
            imu_buffer.begin(), imu_buffer.end(), timestamp,
            [](double t, const sensor_msgs::msg::Imu::ConstSharedPtr &imu_msg)
            {
                return t < get_time_sec(imu_msg->header.stamp);
            });
        if (insert_pos != imu_buffer.end())
        {
            static int out_of_order_imu_report_count = 0;
            if (out_of_order_imu_report_count < 10)
            {
                RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                            "insert out-of-order IMU msg into buffer: current %.9f < newest %.9f",
                            timestamp, last_timestamp_imu);
                ++out_of_order_imu_report_count;
            }
        }
        imu_buffer.insert(insert_pos, msg);
        clear_timestamp_rollback_streak();
    }
    sig_buffer.notify_all();
}

void drop_sensor_backlog_for_prior_relocalization_restart(const char *reason, const char *source, bool reset_sensor_time_gate)
{
    std::lock_guard<mutex> lock_buffer(mtx_buffer);
    const std::size_t dropped_raw_lidar = raw_pcl_buffer.size();
    const std::size_t dropped_livox_lidar = lidar_buffer.size();
    const std::size_t dropped_time = time_buffer.size();
    const std::size_t dropped_imu = imu_buffer.size();
    const double lidar_cutoff_time = last_timestamp_lidar;
    const double imu_cutoff_time = last_timestamp_imu;
    const double sensor_cutoff_time = std::max(lidar_cutoff_time, imu_cutoff_time);

    clear_lidar_buffers_locked();
    imu_buffer.clear();
    if (reset_sensor_time_gate)
    {
        prior_relocalization_sensor_restart_time = -1.0;
        last_timestamp_lidar = 0.0;
        last_timestamp_imu = -1.0;
        last_used_imu_time = -1.0;
        is_first_lidar = true;
    }
    else
    {
        if (sensor_cutoff_time > prior_relocalization_sensor_restart_time)
            prior_relocalization_sensor_restart_time = sensor_cutoff_time;
        if (prior_relocalization_sensor_restart_time > last_used_imu_time)
            last_used_imu_time = prior_relocalization_sensor_restart_time;
    }

    if (dropped_raw_lidar > 0 || dropped_livox_lidar > 0 || dropped_imu > 0 || dropped_time > 0)
    {
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "Dropped stale sensor backlog before prior-map relocalization restart: source=%s reason=%s raw_lidar_dropped=%zu livox_lidar_dropped=%zu time_entries_dropped=%zu imu_dropped=%zu lidar_cutoff=%.6f imu_cutoff=%.6f sensor_cutoff=%.6f reset_sensor_time_gate=%s. Waiting for fresh LiDAR+IMU.",
            source ? source : "unknown",
            reason ? reason : "unknown",
            dropped_raw_lidar,
            dropped_livox_lidar,
            dropped_time,
            dropped_imu,
            lidar_cutoff_time,
            imu_cutoff_time,
            sensor_cutoff_time,
            reset_sensor_time_gate ? "true" : "false");
    }
}

void drop_stale_sensor_backlog_after_prior_icp_failure(const char *reason)
{
    drop_sensor_backlog_for_prior_relocalization_restart(reason, "icp_failure");
}

bool reset_prior_relocalization_local_state_after_failure(const char *reason)
{
    if (prior_map_build_done || prior_map_aligned)
        return false;

    state_ikfom clean_state;
    clean_state.offset_T_L_I = Lidar_T_wrt_IMU;
    clean_state.offset_R_L_I = Lidar_R_wrt_IMU;
    kf.change_x(clean_state);

    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov clean_cov = kf.get_P();
    clean_cov.setIdentity();
    kf.change_P(clean_cov);

    p_imu->Reset();
    state_point = kf.get_x();
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
    map_world_initialized = false;
    p_C_M0 = Zero3d;
    R_C_M0 = Eye3d;
    R_P_M = Eye3d;
    p_P_M = Zero3d;
    first_lidar_time = 0.0;
    lidar_end_time = 0.0;
    lidar_mean_scantime = 0.0;
    scan_num = 0;
    flg_first_scan = true;
    flg_EKF_inited = false;
    lidar_pushed = false;
    localization_health_reset_requested = true;
    feats_undistort->clear();
    feats_down_body->clear();
    feats_down_world->clear();

    RCLCPP_WARN(
        rclcpp::get_logger("laser_mapping"),
        "Reset local FAST-LIO/IMU state after prior-map relocalization failure: reason=%s. Next attempt starts from fresh IMU initialization.",
        reason ? reason : "unknown");
    return true;
}

bool restart_prior_relocalization_from_health(
    const char *reason,
    int unhealthy_count,
    bool insufficient_effective_points,
    bool timestamp_rollback,
    int downsampled_points,
    int effective_points,
    double lidar_beg_time,
    uint64_t frame_index,
    uint64_t rx_index)
{
    if (!use_prior_map || !prior_map_loaded)
        return false;

    drop_sensor_backlog_for_prior_relocalization_restart(reason, "localization_health", timestamp_rollback);

    clear_prior_icp_active_source();
    clear_prior_icp_accum_window();
    clear_prior_replay_cache();
    scan_context_last_icp_candidates.clear();
    scan_context_candidate_cloud_active = false;
    prior_icp_fail_count = 0;

    prior_map_aligned = false;
    prior_map_build_done = false;
    localization_output_trusted = false;
    path.poses.clear();
    prior_map_ready_for_publish = true;
    ikdtree_built = false;
    Localmap_Initialized = false;
    cub_needrm.clear();
    Nearest_Points.clear();
    init_feats_buffer->clear();
    feats_undistort->clear();
    feats_down_body->clear();
    feats_down_world->clear();
    featsFromMap->clear();
    laserCloudOri->clear();
    corr_normvect->clear();
    normvec->clear();
    PointVector empty_map;
    ikdtree.Build(empty_map);
    PointVector().swap(ikdtree.PCL_Storage);

    state_ikfom clean_state;
    clean_state.offset_T_L_I = Lidar_T_wrt_IMU;
    clean_state.offset_R_L_I = Lidar_R_wrt_IMU;
    kf.change_x(clean_state);

    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov clean_cov = kf.get_P();
    clean_cov.setIdentity();
    kf.change_P(clean_cov);

    p_imu->Reset();
    state_point = kf.get_x();
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
    map_world_initialized = false;
    p_C_M0 = Zero3d;
    R_C_M0 = Eye3d;
    R_P_M = Eye3d;
    p_P_M = Zero3d;
    first_lidar_time = 0.0;
    lidar_end_time = 0.0;
    lidar_mean_scantime = 0.0;
    scan_num = 0;
    flg_first_scan = true;
    flg_EKF_inited = false;
    lidar_pushed = false;
    is_first_lidar = true;
    localization_health_reset_requested = true;

    RCLCPP_WARN(
        rclcpp::get_logger("laser_mapping"),
        "\033[1;32mLocalization unhealthy, restarting prior-map relocalization\033[0m"
        " reason=%s consecutive=%d frame=%" PRIu64 " rx=%" PRIu64
        " lidar_t=%.3f down=%d effective=%d min_effective=%d"
        " flags=[insufficient_effective=%s timestamp_rollback=%s]",
        reason ? reason : "unknown",
        unhealthy_count,
        frame_index,
        rx_index,
        lidar_beg_time,
        downsampled_points,
        effective_points,
        localization_min_effective_points,
        insufficient_effective_points ? "true" : "false",
        timestamp_rollback ? "true" : "false");
    return true;
}

constexpr double MID360_IMU_REORDER_WAIT_SEC = 0.010;
bool sync_packages(MeasureGroup &meas)
{
    const bool use_raw_pointcloud = (p_pre->lidar_type != AVIA);
    if (use_raw_pointcloud && !lidar_pushed)
    {
        sensor_msgs::msg::PointCloud2::SharedPtr raw_msg;
        double frame_begin_time = 0.0;
        double frame_end_time = 0.0;
        uint64_t rx_index = 0;
        {
            std::lock_guard<mutex> lock_buffer(mtx_buffer);
            if (imu_buffer.empty() || !front_lidar_metadata_ready_locked(use_raw_pointcloud, "raw_preprocess_begin"))
                return false;
            raw_msg = raw_pcl_buffer.front();
            frame_begin_time = time_buffer.front();
            frame_end_time = lidar_end_time_buffer.front();
            rx_index = lidar_rx_index_buffer.front();
        }

        PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
        auto msg_copy = std::make_unique<sensor_msgs::msg::PointCloud2>(*raw_msg);
        {
            std::lock_guard<mutex> lock_preprocess(mtx_preprocess);
            p_pre->process(msg_copy, ptr);
        }

        {
            std::lock_guard<mutex> lock_buffer(mtx_buffer);
            if (!front_lidar_metadata_ready_locked(use_raw_pointcloud, "raw_preprocess_finish") ||
                raw_pcl_buffer.front() != raw_msg)
                return false;
            meas.lidar = ptr;
            meas.lidar_beg_time = frame_begin_time;
            last_synced_lidar_rx_index = rx_index;
            if (source_ray_exporter && source_ray_exporter->enabled())
                source_ray_current_raw_message = raw_msg;
            if (frame_end_time > meas.lidar_beg_time)
            {
                lidar_end_time = frame_end_time;
                const double measured_scan_time = lidar_end_time - meas.lidar_beg_time;
                scan_num ++;
                lidar_mean_scantime += (measured_scan_time - lidar_mean_scantime) / scan_num;
            }
            else if (p_pre->lidar_type == RSAIRY && msg_is_XYZI)
            {
                lidar_end_time = meas.lidar_beg_time;
            }
            else if (meas.lidar->points.size() <= 1)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
                static int too_few_raw_points_report_count = 0;
                if (too_few_raw_points_report_count < 10)
                {
                    RCLCPP_WARN(
                        rclcpp::get_logger("laser_mapping"),
                        "Too few input raw point cloud points: points=%zu rx=%" PRIu64,
                        meas.lidar->points.size(), rx_index);
                    ++too_few_raw_points_report_count;
                }
            }
            else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            }
            else
            {
                scan_num ++;
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
                lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
            }
            meas.lidar_end_time = lidar_end_time;
            lidar_pushed = true;
        }
    }
    else
    {
        std::lock_guard<mutex> lock_buffer(mtx_buffer);
        if (imu_buffer.empty() || !front_lidar_metadata_ready_locked(use_raw_pointcloud, "sync_begin")) {
            return false;
        }

        if(!lidar_pushed)
        {
            meas.lidar = lidar_buffer.front();
            meas.lidar_beg_time = time_buffer.front();
            last_synced_lidar_rx_index = lidar_rx_index_buffer.front();
            const double measured_lidar_end_time = lidar_end_time_buffer.front();
            if (measured_lidar_end_time > meas.lidar_beg_time)
            {
                lidar_end_time = measured_lidar_end_time;
                const double measured_scan_time = lidar_end_time - meas.lidar_beg_time;
                scan_num ++;
                lidar_mean_scantime += (measured_scan_time - lidar_mean_scantime) / scan_num;
            }
            else if (p_pre->lidar_type == RSAIRY && msg_is_XYZI)
            {
                lidar_end_time = meas.lidar_beg_time;
            }
            else if (meas.lidar->points.size() <= 1)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
                static int too_few_livox_points_report_count = 0;
                if (too_few_livox_points_report_count < 10)
                {
                    RCLCPP_WARN(
                        rclcpp::get_logger("laser_mapping"),
                        "Too few input Livox point cloud points: points=%zu rx=%" PRIu64,
                        meas.lidar->points.size(), last_synced_lidar_rx_index);
                    ++too_few_livox_points_report_count;
                }
            }
            else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            }
            else
            {
                scan_num ++;
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
                lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
            }
            meas.lidar_end_time = lidar_end_time;
            lidar_pushed = true;
        }
    }

    {
        std::lock_guard<mutex> lock_buffer(mtx_buffer);
        const double imu_ready_time = lidar_end_time + (p_pre->lidar_type == MID360 ? MID360_IMU_REORDER_WAIT_SEC : 0.0);
        if (last_timestamp_imu < imu_ready_time)
        {
            return false;
        }
    }

    {
        std::lock_guard<mutex> lock_buffer(mtx_buffer);
        if (imu_buffer.empty() || !front_lidar_metadata_ready_locked(use_raw_pointcloud, "imu_window"))
        {
            return false;
        }

        /*** push imu data, and pop from imu buffer ***/
        double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        meas.imu.clear();
        while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
        {
            imu_time = get_time_sec(imu_buffer.front()->header.stamp);
            if(imu_time > lidar_end_time) break;
            meas.imu.push_back(imu_buffer.front());
            last_used_imu_time = imu_time;
            imu_buffer.pop_front();
        }

        if (meas.imu.empty())
        {
            static int empty_imu_window_report_count = 0;
            if (empty_imu_window_report_count < 20)
            {
                const double front_imu_time = imu_buffer.empty() ? -1.0 : get_time_sec(imu_buffer.front()->header.stamp);
                RCLCPP_WARN(
                    rclcpp::get_logger("laser_mapping"),
                    "skip lidar frame without IMU coverage: lidar=[%.6f %.6f] first_buffered_imu=%.6f rx=%" PRIu64,
                    meas.lidar_beg_time, lidar_end_time, front_imu_time, last_synced_lidar_rx_index);
                ++empty_imu_window_report_count;
            }

            pop_lidar_front_locked(use_raw_pointcloud);
            return false;
        }

        pop_lidar_front_locked(use_raw_pointcloud);
    }
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        if (!point_xyz_finite(feats_down_world->points[i]))
            continue;
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

bool write_saved_pcd(const string &path, const PointCloudXYZI::Ptr &cloud)
{
    const pcd_save::SaveSummary summary =
        pcd_save::writeBinary(path, cloud, pcd_save_voxel_leaf);
    if (!summary.success)
    {
        RCLCPP_ERROR(
            rclcpp::get_logger("laser_mapping"),
            "Failed to save PCD %s: %s",
            path.c_str(),
            summary.error.empty() ? "unknown error" : summary.error.c_str());
        return false;
    }

    if (summary.downsampled)
    {
        RCLCPP_INFO(
            rclcpp::get_logger("laser_mapping"),
            "Saved downsampled PCD: %zu -> %zu points, leaf=%.3f m",
            summary.input_points,
            summary.output_points,
            pcd_save_voxel_leaf);
    }
    return true;
}

inline void RGBpointBodyToOutputWorld(PointType const * const pi, PointType * const po)
{
    if (!use_base_link_output_frame())
    {
        RGBpointBodyToWorld(pi, po);
        return;
    }

    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_C(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    V3D p_C_F, v_C_F, w_C_F;
    M3D R_C_F;
    compute_base_link_pose_twist_in_cam_init(p_C_F, R_C_F, v_C_F, w_C_F);
    maybe_init_base_link_gravity_map(p_C_F, R_C_F);

    V3D p_out = transform_cam_to_map(p_C);
    po->x = p_out(0);
    po->y = p_out(1);
    po->z = p_out(2);
    po->intensity = pi->intensity;
}

void save_waiting_pcd_on_exit()
{
    static bool pcd_saved = false;
    static bool scan_context_saved = false;
    const bool need_pcd_save = !use_prior_map && pcd_save_en && pcl_wait_save->size() > 0;
    const bool need_scan_context_save = scan_context_enable && scan_context_dirty;
    if ((!need_pcd_save || pcd_saved) && (!need_scan_context_save || scan_context_saved))
    {
        return;
    }

    if (use_prior_map && !pcd_saved)
    {
        RCLCPP_INFO(
            rclcpp::get_logger("laser_mapping"),
            "use_prior_map is true; skip PCD save on exit.");
        pcd_saved = true;
    }

    if (need_pcd_save && !pcd_saved)
    {
        string file_name = string("scans.pcd");
        std::error_code dir_ec;
        std::filesystem::create_directories(string(ROOT_DIR) + "PCD", dir_ec);
        if (dir_ec)
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "Failed to create PCD directory before saving: %s", dir_ec.message().c_str());
        }
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        RCLCPP_INFO(
            rclcpp::get_logger("laser_mapping"),
            "Saving current scan to /PCD/%s",
            file_name.c_str());
        pcd_saved = write_saved_pcd(all_points_dir, pcl_wait_save);
    }

    if (need_scan_context_save && !scan_context_saved)
    {
        if (save_scan_context_database_to_disk(true))
        {
            scan_context_saved = true;
        }
    }
}

void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI());
        laserCloudWorld->reserve(size);

        for (int i = 0; i < size; i++)
        {
            PointType point_world;
            RGBpointBodyToOutputWorld(&laserCloudFullRes->points[i], &point_world);
            if (point_xyz_finite(point_world))
                laserCloudWorld->push_back(point_world);
        }
        laserCloudWorld->width = laserCloudWorld->size();
        laserCloudWorld->height = 1;
        laserCloudWorld->is_dense = true;

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = use_base_link_output_frame() ? base_link_world_frame_id : "camera_init";
        pubLaserCloudFull->publish(laserCloudmsg);
        if (debug_save_registered_pcd_en)
        {
            static int debug_registered_pcd_frame_count = 0;
            debug_registered_pcd_frame_count++;
            if (debug_registered_pcd_frame_count >= debug_save_registered_pcd_frame_interval)
            {
                debug_registered_pcd_frame_count = 0;
                save_debug_registered_pcd_async(laserCloudWorld, lidar_end_time);
            }
        }
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
     * 2. noted that pcd save will influence the real-time performences */
    if (pcd_save_en && !use_prior_map)
    {
        PointCloudXYZI::Ptr save_cloud_body = (pcd_save_voxel_leaf > 1e-3) ? feats_down_body : feats_undistort;
        int size = save_cloud_body->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI());
        laserCloudWorld->reserve(size);

        for (int i = 0; i < size; i++)
        {
            PointType point_world;
            RGBpointBodyToOutputWorld(&save_cloud_body->points[i], &point_world);
            if (point_xyz_finite(point_world))
                laserCloudWorld->push_back(point_world);
        }
        laserCloudWorld->width = laserCloudWorld->size();
        laserCloudWorld->height = 1;
        laserCloudWorld->is_dense = true;
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            std::error_code dir_ec;
            std::filesystem::create_directories(string(ROOT_DIR) + "PCD", dir_ec);
            if (dir_ec)
            {
                RCLCPP_WARN(
                    rclcpp::get_logger("laser_mapping"),
                    "Failed to create PCD directory before saving: %s", dir_ec.message().c_str());
            }
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            RCLCPP_INFO(
                rclcpp::get_logger("laser_mapping"),
                "Saving current scan to %s",
                all_points_dir.c_str());
            if (write_saved_pcd(all_points_dir, pcl_wait_save))
            {
                pcl_wait_save->clear();
                scan_wait_num = 0;
            }
        }
    }
}

void publish_frame_body(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body,
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr pubScanContextGravity)
{
    if (scan_body_pub_stride_s > 1e-6)
    {
        if (!std::isfinite(lidar_end_time))
            return;
        if (scan_body_pub_first_stamp < 0.0 || lidar_end_time < scan_body_pub_first_stamp)
        {
            scan_body_pub_first_stamp = lidar_end_time;
            scan_body_pub_last_slot = -1;
        }

        const double relative_time = std::max(0.0, lidar_end_time - scan_body_pub_first_stamp);
        // Publish exactly one deskewed LiDAR frame for each stride slot.  The
        // old phase-window test could publish two boundary frames and could
        // miss a slot entirely when the LiDAR period drifted just beyond the
        // configured sample window.
        const double slot_time = relative_time + 0.5 * scan_body_pub_sample_s + 1e-6;
        const int64_t slot = static_cast<int64_t>(std::floor(slot_time / scan_body_pub_stride_s));
        if (slot <= scan_body_pub_last_slot)
            return;
        scan_body_pub_last_slot = slot;
    }

    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI());
    laserCloudIMUBody->reserve(size);

    for (int i = 0; i < size; i++)
    {
        PointType point_body;
        if (use_base_link_output_frame())
        {
            RGBpointBodyLidarToBaseLink(&feats_undistort->points[i],
                                   &point_body);
        }
        else
        {
            RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                                   &point_body);
        }
        if (point_xyz_finite(point_body))
            laserCloudIMUBody->push_back(point_body);
    }
    laserCloudIMUBody->width = laserCloudIMUBody->size();
    laserCloudIMUBody->height = 1;
    laserCloudIMUBody->is_dense = true;

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = use_base_link_output_frame() ? output_body_frame_id : "body";
    pubLaserCloudFull_body->publish(laserCloudmsg);

    V3D up_B(Zero3d);
    if (current_scan_context_gravity_up(up_B))
    {
        geometry_msgs::msg::Vector3Stamped gravity_msg;
        gravity_msg.header = laserCloudmsg.header;
        gravity_msg.vector.x = up_B.x();
        gravity_msg.vector.y = up_B.y();
        gravity_msg.vector.z = up_B.z();
        pubScanContextGravity->publish(gravity_msg);
    }
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI());
    laserCloudWorld->reserve(effct_feat_num);
    for (int i = 0; i < effct_feat_num; i++)
    {
        PointType point_world;
        RGBpointBodyToOutputWorld(&laserCloudOri->points[i], &point_world);
        if (point_xyz_finite(point_world))
            laserCloudWorld->push_back(point_world);
    }
    laserCloudWorld->width = laserCloudWorld->size();
    laserCloudWorld->height = 1;
    laserCloudWorld->is_dense = true;
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
    laserCloudFullRes3.header.frame_id = use_base_link_output_frame() ? base_link_world_frame_id : "camera_init";
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

inline bool should_add_scan_context_keyframe(const sc::Pose &pose)
{
    if (!scan_context_has_last_keyframe)
        return true;

    const double dx = pose.x - scan_context_last_keyframe_pose.x;
    const double dy = pose.y - scan_context_last_keyframe_pose.y;
    const double dz = pose.z - scan_context_last_keyframe_pose.z;
    const double translation = sqrt(dx * dx + dy * dy + dz * dz);
    const double yaw_delta = fabs(normalize_yaw(pose.yaw - scan_context_last_keyframe_pose.yaw));
    return translation >= scan_context_keyframe_meter_gap ||
           yaw_delta >= scan_context_keyframe_yaw_gap_rad;
}

inline void maybe_add_scan_context_keyframe(double stamp, const PointCloudXYZI::Ptr &scan_body)
{
    if (!scan_context_enable || use_prior_map || !scan_body || scan_body->empty())
        return;

    sc::Pose pose;
    if (!get_current_output_pose(pose, false))
    {
        static int pose_warn_count = 0;
        if (pose_warn_count < 5)
        {
            RCLCPP_DEBUG(
                rclcpp::get_logger("laser_mapping"),
                "Skip Scan Context keyframe: output pose is not initialized yet.");
            ++pose_warn_count;
        }
        return;
    }

    if (!should_add_scan_context_keyframe(pose))
        return;

    PointCloudXYZI::Ptr sc_cloud = make_scan_context_body_cloud(scan_body);
    if (!sc_cloud || sc_cloud->empty())
        return;

    PointCloudXYZI::Ptr descriptor_cloud = sc_cloud;
    V3D gravity_up_body(Zero3d);
    if (!current_scan_context_gravity_up(gravity_up_body))
    {
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "Skip Scan Context keyframe: physical gravity estimate is invalid.");
        return;
    }
    M3D R_G_B(Eye3d);
    if (scan_context_config.gravity_canonicalized)
    {
        if (!current_scan_context_gravity_rotation(R_G_B))
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "Skip Scan Context keyframe: gravity estimate is invalid.");
            return;
        }
        descriptor_cloud.reset(new PointCloudXYZI(
            sc::gravityCanonicalize(*sc_cloud, R_G_B)));
    }
    const M3D R_map_body = pose_rpy_to_rotation(pose);
    const M3D R_map_descriptor = R_map_body * R_G_B.transpose();
    pose.canonical_yaw = normalize_yaw(
        std::atan2(R_map_descriptor(1, 0), R_map_descriptor(0, 0)));
    descriptor_cloud = downsample_scan_context_cloud(descriptor_cloud);
    if (!descriptor_cloud || descriptor_cloud->empty())
        return;
    PointCloudXYZI::Ptr manual_loop_cloud = downsample_scan_context_cloud(sc_cloud);
    if (!manual_loop_cloud || manual_loop_cloud->empty())
        return;

    const int keyframe_index = scan_context_keyframe_count;
    if (!append_manual_loop_keyframe(
            keyframe_index, stamp, pose, manual_loop_cloud, gravity_up_body))
    {
        RCLCPP_WARN(
            rclcpp::get_logger("laser_mapping"),
            "Skip Scan Context keyframe: failed to append the synchronized manual-loop record.");
        return;
    }
    if (scan_context_config.dual_z_layer_enable &&
        scan_context_config.dual_z_split_auto)
    {
        scan_context_split_estimator.addScan(*descriptor_cloud);
        scan_context_pending_keyframes.push_back(
            PendingScanContextKeyframe{stamp, pose, descriptor_cloud});
    }
    else
    {
        scan_context_db.addEntry(
            stamp, pose,
            scan_context_db.makeDescriptor(*descriptor_cloud));
    }
    scan_context_last_keyframe_pose = pose;
    scan_context_has_last_keyframe = true;
    scan_context_dirty = true;
    scan_context_keyframe_count++;

    RCLCPP_DEBUG(
        rclcpp::get_logger("laser_mapping"),
        "Scan Context keyframe saved: idx=%d pose=[%.3f %.3f %.3f yaw %.2f deg] scan_points=%zu db_entries=%zu",
        scan_context_keyframe_count - 1,
        pose.x, pose.y, pose.z, pose.yaw * 180.0 / M_PI,
        descriptor_cloud->size(),
        scan_context_config.dual_z_split_auto
            ? scan_context_pending_keyframes.size()
            : scan_context_db.size());
}

template<typename T>
void set_posestamp(T & out)
{
    V3D pos_out = state_point.pos;
    M3D R_out = state_point.rot.toRotationMatrix();

    if (use_base_link_output_frame())
    {
        V3D p_C_F, v_C_F, w_C_F;
        M3D R_C_F;
        compute_base_link_pose_twist_in_cam_init(p_C_F, R_C_F, v_C_F, w_C_F);
        maybe_init_base_link_gravity_map(p_C_F, R_C_F);

        pos_out = transform_cam_to_map(p_C_F);
        R_out = R_C_M0.transpose() * R_C_F;
        if (prior_map_aligned)
        {
            R_out = R_P_M * R_out;
        }
    }

    Eigen::Quaterniond q_out(R_out);
    q_out.normalize();
    out.pose.position.x = pos_out(0);
    out.pose.position.y = pos_out(1);
    out.pose.position.z = pos_out(2);
    out.pose.orientation.x = q_out.x();
    out.pose.orientation.y = q_out.y();
    out.pose.orientation.z = q_out.z();
    out.pose.orientation.w = q_out.w();
}

template<typename T>
void set_twiststamp(T & out)
{
    V3D vel_out = state_point.vel;
    V3D ang_vel_out(Zero3d);
    if (has_latest_imu())
    {
        ang_vel_out << Measures.imu.back()->angular_velocity.x,
                       Measures.imu.back()->angular_velocity.y,
                       Measures.imu.back()->angular_velocity.z;
        ang_vel_out(0) -= state_point.bg[0];
        ang_vel_out(1) -= state_point.bg[1];
        ang_vel_out(2) -= state_point.bg[2];
    }

    if (use_base_link_output_frame())
    {
        V3D p_C_F, v_C_F, w_C_F;
        M3D R_C_F;
        compute_base_link_pose_twist_in_cam_init(p_C_F, R_C_F, v_C_F, w_C_F);
        maybe_init_base_link_gravity_map(p_C_F, R_C_F);
        vel_out = R_C_F.transpose() * v_C_F;
        ang_vel_out = R_C_F.transpose() * w_C_F;
    }

    out.twist.linear.x = vel_out(0);
    out.twist.linear.y = vel_out(1);
    out.twist.linear.z = vel_out(2);
    out.twist.angular.x = ang_vel_out(0);
    out.twist.angular.y = ang_vel_out(1);
    out.twist.angular.z = ang_vel_out(2);
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
    if (use_base_link_output_frame() && !map_world_initialized)
    {
        V3D p_C_F, v_C_F, w_C_F;
        M3D R_C_F;
        compute_base_link_pose_twist_in_cam_init(p_C_F, R_C_F, v_C_F, w_C_F);
        maybe_init_base_link_gravity_map(p_C_F, R_C_F);
        if (!map_world_initialized)
            return;
    }

    const rclcpp::Time stamp = pose_output_stamp();
    odomAftMapped.header.frame_id = use_base_link_output_frame() ? base_link_world_frame_id : "camera_init";
    odomAftMapped.child_frame_id = use_base_link_output_frame() ? output_body_frame_id : "body";
    odomAftMapped.header.stamp = stamp;
    set_posestamp(odomAftMapped.pose);
    set_twiststamp(odomAftMapped.twist);

    std::fill(odomAftMapped.pose.covariance.begin(), odomAftMapped.pose.covariance.end(), 0.0);
    std::fill(odomAftMapped.twist.covariance.begin(), odomAftMapped.twist.covariance.end(), 0.0);

    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }
    pubOdomAftMapped->publish(odomAftMapped);

    if (!publish_tf || !tf_br)
    {
        return;
    }

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = odomAftMapped.header.frame_id;
    trans.child_frame_id = odomAftMapped.child_frame_id;
    trans.header.stamp = stamp;
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    tf_br->sendTransform(trans);
}

void publish_pose(const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pubPose)
{
    if (!publish_pose_topic || !pubPose)
    {
        return;
    }
    if (use_base_link_output_frame() && !map_world_initialized)
    {
        V3D p_C_F, v_C_F, w_C_F;
        M3D R_C_F;
        compute_base_link_pose_twist_in_cam_init(p_C_F, R_C_F, v_C_F, w_C_F);
        maybe_init_base_link_gravity_map(p_C_F, R_C_F);
        if (!map_world_initialized)
            return;
    }

    geometry_msgs::msg::PoseStamped pose_msg;
    set_posestamp(pose_msg);
    pose_msg.header.stamp = pose_output_stamp();
    pose_msg.header.frame_id = use_base_link_output_frame() ? base_link_world_frame_id : "camera_init";
    pubPose->publish(pose_msg);
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose);
    const rclcpp::Time stamp = pose_output_stamp();
    msg_body_pose.header.stamp = stamp; // ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = use_base_link_output_frame() ? base_link_world_frame_id : "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.header.stamp = stamp;
        path.header.frame_id = msg_body_pose.header.frame_id;
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 
    const std::size_t match_capacity = static_cast<std::size_t>(std::max(0, feats_down_size));
    laserCloudOri->resize(match_capacity);
    corr_normvect->resize(match_capacity);
    normvec->resize(match_capacity);
    res_last.assign(match_capacity, 0.0f);
    // FAST-LIO expects points to stay selectable before the first converged nearest-neighbor refresh.
    point_selected_surf.assign(match_capacity, 1);
    int nearest_candidate_count = 0;
    int plane_fit_count = 0;
    int residual_gate_count = 0;
    double kth_dist_sum = 0.0;
    double kth_dist_min = std::numeric_limits<double>::infinity();

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for reduction(+:nearest_candidate_count, plane_fit_count, residual_gate_count, kth_dist_sum) reduction(min:kth_dist_min)
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            if (points_near.size() >= NUM_MATCH_POINTS)
            {
                const double kth_dist = pointSearchSqDis[NUM_MATCH_POINTS - 1];
                kth_dist_sum += kth_dist;
                kth_dist_min = std::min(kth_dist_min, kth_dist);
                if (kth_dist <= 5)
                {
                    point_selected_surf[i] = true;
                    nearest_candidate_count++;
                }
                else
                {
                    point_selected_surf[i] = false;
                }
            }
            else
            {
                point_selected_surf[i] = false;
            }
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            plane_fit_count++;
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            const double body_range = std::max(1e-12, p_body.norm());
            float s = 1 - 0.9 * fabs(pd2) / sqrt(body_range);

            if (s > 0.9)
            {
                residual_gate_count++;
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = std::fabs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        current_measurement_no_effective_points = true;
        static double last_no_effective_report_time = -1.0;
        const V3D euler = SO3ToEuler(s.rot);
        const int map_points = ikdtree.validnum();
        if (last_no_effective_report_time < 0.0 ||
            current_lidar_beg_time - last_no_effective_report_time >= 1.0)
        {
            RCLCPP_WARN(
                rclcpp::get_logger("laser_mapping"),
                "No effective points: frame=%" PRIu64 " rx=%" PRIu64
                " lidar_t=%.3f span=%.1fms raw=%zu undistort=%zu down=%d map=%d"
                " pose=[%.3f %.3f %.3f] rpy_deg=[%.2f %.2f %.2f]"
                " vel_norm=%.3f prev_res_mean=%.4f prior_aligned=%s prior_ready=%s"
                " nearest_gate=%d plane_fit=%d residual_gate=%d kth_sq_min=%.4f kth_sq_mean=%.4f",
                current_lidar_frame_index, current_lidar_rx_index,
                current_lidar_beg_time - first_lidar_time,
                (current_lidar_end_time - current_lidar_beg_time) * 1000.0,
                current_raw_lidar_points, current_undistorted_points, feats_down_size, map_points,
                s.pos(0), s.pos(1), s.pos(2),
                wrap_angle_deg(euler(0)), wrap_angle_deg(euler(1)), wrap_angle_deg(euler(2)),
                s.vel.norm(), res_mean_last,
                prior_map_aligned ? "true" : "false",
                prior_map_build_done ? "true" : "false",
                nearest_candidate_count,
                plane_fit_count,
                residual_gate_count,
                std::isfinite(kth_dist_min) ? kth_dist_min : -1.0,
                feats_down_size > 0 ? kth_dist_sum / static_cast<double>(feats_down_size) : -1.0);
            last_no_effective_report_time = current_lidar_beg_time;
        }
        return;
    }

    current_measurement_no_effective_points = false;
    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<double>("publish.scan_bodyframe_stride_s", 0.0);
        this->declare_parameter<double>("publish.scan_bodyframe_sample_s", 0.2);
        this->declare_parameter<bool>("publish_tf", false);
        this->declare_parameter<bool>("publish_pose_topic", true);
        this->declare_parameter<bool>("pose_output_use_node_clock", false);
        this->declare_parameter<string>("runtime.profile", "mapping");
        this->declare_parameter<bool>("debug.save_registered_pcd_en", false);
        this->declare_parameter<int>("debug.save_registered_pcd_frame_interval", 10);
        this->declare_parameter<string>("debug.registered_pcd_path", "Log/cloud_registered_latest.pcd");
        this->declare_parameter<string>("localization_pose_topic", "/localization_pose");
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<int>("common.lidar_subscribe_qos_depth", 50);
        this->declare_parameter<int>("common.imu_subscribe_qos_depth", 2000);
        this->declare_parameter<string>("common.lidar_qos_reliability", "best_effort");
        this->declare_parameter<string>("common.imu_qos_reliability", "best_effort");
        this->declare_parameter<double>("common.imu_rate", 200.0);
        this->declare_parameter<bool>("common.time_sync_en", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<bool>("common.deskew_en", true);
        this->declare_parameter<bool>("common.airy_imu_flip_yz", false);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<double>("mapping.det_range", 300.0);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("mapping.max_height", 5.0);
        this->declare_parameter<bool>("mapping.transform_to_base_link_frame", false);
        this->declare_parameter<bool>("mapping.output_base_link_origin_odom", false);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T_imu_to_base_link", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R_imu_to_base_link", vector<double>());
        this->declare_parameter<string>("mapping.base_link_world_frame_id", "map");
        this->declare_parameter<string>("mapping.output_body_frame_id", "base_link");
        this->declare_parameter<double>("preprocess.mapping_blind", 2.0);
        this->declare_parameter<double>("preprocess.localization_blind", 0.3);
        this->declare_parameter<string>("preprocess.blind_filter_shape", "sphere");
        this->declare_parameter<double>("preprocess.blind_z_min", -1.0e9);
        this->declare_parameter<double>("preprocess.blind_z_max", 1.0e9);
        this->declare_parameter<double>("preprocess.max_range", -1.0);
        this->declare_parameter<string>("preprocess.tag_filter_mode", "low_confidence");
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<double>("initialization.init_time_s", 0.1);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<int>("pcd_save.interval", -1);
        this->declare_parameter<double>("pcd_save.voxel_leaf", 0.0);
        this->declare_parameter<bool>("manual_loop_export.enable", true);
        this->declare_parameter<string>("manual_loop_export.session_dir", "");
        this->declare_parameter<bool>("manual_loop_export.overwrite", true);
        this->declare_parameter<bool>("source_ray_export.enable", false);
        this->declare_parameter<bool>("source_ray_export.overwrite", false);
        this->declare_parameter<bool>("source_ray_export.save_sensor_pcd", true);
        this->declare_parameter<string>("source_ray_export.manifest_csv", "");
        this->declare_parameter<string>("source_ray_export.pgo_tum", "");
        this->declare_parameter<string>("source_ray_export.output_dir", "");
        this->declare_parameter<int>("source_ray_export.expected_frame_count", 0);
        this->declare_parameter<double>("source_ray_export.timestamp_tolerance_us", 2.0);
        this->declare_parameter<double>("source_ray_export.blind_radius_m", 0.3);
        this->declare_parameter<double>("source_ray_export.blind_z_min_m", -0.5);
        this->declare_parameter<double>("source_ray_export.blind_z_max_m", 2.0);
        this->declare_parameter<double>("source_ray_export.maximum_range_m", 30.0);
        this->declare_parameter<double>("source_ray_export.endpoint_voxel_m", 0.1);
        this->declare_parameter<int>("source_ray_export.deskew_validation_frames", 3);
        this->declare_parameter<vector<double>>(
            "source_ray_export.front_T_input_sensor", vector<double>());
        this->declare_parameter<vector<double>>(
            "source_ray_export.back_T_input_sensor", vector<double>());
        this->declare_parameter<vector<int64_t>>(
            "source_ray_export.selected_pose_indices", vector<int64_t>());
        this->declare_parameter<bool>("prior_map.mapping_use_prior_map", false);
        this->declare_parameter<bool>("prior_map.localization_use_prior_map", true);
        this->declare_parameter<int>("prior_map.icp_max_iterations", 60);
        this->declare_parameter<double>("prior_map.icp_max_corr_dist", 3.0);
        this->declare_parameter<double>("prior_map.icp_fitness_thresh", 0.8);
        this->declare_parameter<double>("prior_map.icp_min_overlap_ratio", 0.5);
        this->declare_parameter<int>("prior_map.icp_min_points", 2000);
        this->declare_parameter<double>("prior_map.voxel_leaf", 0.5);
        this->declare_parameter<double>("prior_map.voxel_leaf_fine", 0.25);
        this->declare_parameter<double>("prior_map.relocalization_accum_time_s", 0.0);
        this->declare_parameter<int>("prior_map.icp_refine_top_k", 3);
        this->declare_parameter<bool>("prior_map.multi_seed_enable", true);
        this->declare_parameter<vector<double>>("prior_map.initial_guess_xy", vector<double>(2, 0.0));
        this->declare_parameter<double>("prior_map.initial_guess_yaw_deg", 0.0);
        this->declare_parameter<double>("prior_map.seed_xy_range", 2.0);
        this->declare_parameter<double>("prior_map.seed_xy_step", 1.0);
        this->declare_parameter<double>("prior_map.seed_yaw_range_deg", 20.0);
        this->declare_parameter<double>("prior_map.seed_yaw_step_deg", 10.0);
        this->declare_parameter<bool>("prior_map.scan_context.enable", true);
        this->declare_parameter<string>("prior_map.scan_context.database_path", "");
        this->declare_parameter<double>("prior_map.scan_context.keyframe_meter_gap", 1.0);
        this->declare_parameter<double>("prior_map.scan_context.keyframe_yaw_gap_deg", 10.0);
        this->declare_parameter<double>("prior_map.scan_context.voxel_leaf", 0.4);
        this->declare_parameter<double>("prior_map.scan_context.seed_xy_offset", 0.5);
        this->declare_parameter<int>("prior_map.scan_context.num_rings", 20);
        this->declare_parameter<int>("prior_map.scan_context.num_sectors", 60);
        this->declare_parameter<double>("prior_map.scan_context.max_radius", 80.0);
        this->declare_parameter<bool>("prior_map.scan_context.dual_z_layer_enable", false);
        this->declare_parameter<double>("prior_map.scan_context.dual_z_split_height", 2.5);
        this->declare_parameter<bool>("prior_map.scan_context.dual_z_split_auto", false);
        this->declare_parameter<double>("prior_map.scan_context.dual_z_split_auto_min", 1.5);
        this->declare_parameter<double>("prior_map.scan_context.dual_z_split_auto_max", 4.5);
        this->declare_parameter<double>("prior_map.scan_context.dual_z_split_auto_bin_size", 0.1);
        this->declare_parameter<double>("prior_map.scan_context.dual_z_split_auto_histogram_max", 8.0);
        this->declare_parameter<double>("prior_map.scan_context.dual_z_split_auto_min_layer_fraction", 0.05);
        this->declare_parameter<int>("prior_map.scan_context.dual_z_split_auto_min_keyframes", 20);
        this->declare_parameter<double>("prior_map.scan_context.origin_height_from_ground", 0.0);
        this->declare_parameter<double>("prior_map.scan_context.dual_z_low_weight", 0.3);
        this->declare_parameter<double>("prior_map.scan_context.dual_z_high_weight", 0.7);
        this->declare_parameter<int>("prior_map.scan_context.min_joint_rings", 2);
        this->declare_parameter<double>("prior_map.scan_context.absent_upper_fallback_max_local_fraction", 0.05);
        this->declare_parameter<double>("prior_map.scan_context.absent_upper_fallback_radius", 10.0);
        this->declare_parameter<int>("prior_map.scan_context.absent_upper_fallback_min_keyframes", 3);
        this->declare_parameter<double>("prior_map.scan_context.retrieval_height_offset", 0.1);
        this->declare_parameter<double>("prior_map.scan_context.sector_support_exponent", 0.5);
        this->declare_parameter<double>("prior_map.scan_context.vertical_boundary_margin", 0.1);
        this->declare_parameter<bool>("prior_map.scan_context.gravity_canonicalization_enable", true);
        this->declare_parameter<bool>("prior_map.scan_context.vertical_estimation_enable", false);
        this->declare_parameter<double>("prior_map.scan_context.vertical_correction_min", -1.5);
        this->declare_parameter<double>("prior_map.scan_context.vertical_correction_max", 1.5);
        this->declare_parameter<double>("prior_map.scan_context.vertical_stable_fraction", 0.5);
        this->declare_parameter<int>("prior_map.scan_context.candidate_top_k", 5);
        this->declare_parameter<int>("prior_map.scan_context.yaw_top_k", 3);
        this->declare_parameter<double>("prior_map.scan_context.distance_thresh", 0.5);
        this->declare_parameter<string>("mapping.lidar_extrinsic_profile", "");
        this->declare_parameter<vector<double>>("mapping.extrinsic_profiles.front.T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_profiles.front.R", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_profiles.fusion.T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_profiles.fusion.R", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());

        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<double>("publish.scan_bodyframe_stride_s", scan_body_pub_stride_s, 0.0);
        this->get_parameter_or<double>("publish.scan_bodyframe_sample_s", scan_body_pub_sample_s, 0.2);
        this->get_parameter_or<bool>("publish_tf", publish_tf, false);
        this->get_parameter_or<bool>("publish_pose_topic", publish_pose_topic, true);
        this->get_parameter_or<bool>("pose_output_use_node_clock", pose_output_use_node_clock, false);
        this->get_parameter_or<string>("runtime.profile", runtime_profile, string("mapping"));
        this->get_parameter_or<bool>("debug.save_registered_pcd_en", debug_save_registered_pcd_en, false);
        this->get_parameter_or<int>("debug.save_registered_pcd_frame_interval", debug_save_registered_pcd_frame_interval, 10);
        this->get_parameter_or<string>("debug.registered_pcd_path", debug_registered_pcd_path, string("Log/cloud_registered_latest.pcd"));
        debug_save_registered_pcd_frame_interval = std::max(1, debug_save_registered_pcd_frame_interval);
        debug_registered_pcd_path = resolve_output_path(debug_registered_pcd_path);
        this->get_parameter_or<string>("localization_pose_topic", localization_pose_topic, string("/localization_pose"));
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/livox/imu");
        int lidar_qos_depth = 50;
        int imu_qos_depth = 2000;
        this->get_parameter_or<int>("common.lidar_subscribe_qos_depth", lidar_qos_depth, 50);
        this->get_parameter_or<int>("common.imu_subscribe_qos_depth", imu_qos_depth, 2000);
        this->get_parameter_or<string>("common.lidar_qos_reliability", lidar_qos_reliability, "best_effort");
        this->get_parameter_or<string>("common.imu_qos_reliability", imu_qos_reliability, "best_effort");
        double imu_rate_hz = 200.0;
        this->get_parameter_or<double>("common.imu_rate", imu_rate_hz, 200.0);
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<bool>("common.deskew_en", deskew_en, true);
        this->get_parameter_or<bool>("common.airy_imu_flip_yz", airy_imu_flip_yz, false);
        this->get_parameter_or<double>("filter_size_corner",filter_size_corner_min,0.5);
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        double det_range_param = DET_RANGE;
        double preprocess_max_range_param = -1.0;
        string blind_filter_shape_param = "sphere";
        double blind_z_min_param = -1.0e9;
        double blind_z_max_param = 1.0e9;
        string tag_filter_mode_param = "low_confidence";
        this->get_parameter_or<double>("mapping.det_range", det_range_param, 300.0);
        this->get_parameter_or<double>("mapping.fov_degree",fov_deg,180.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        this->get_parameter_or<double>("mapping.max_height", MAX_HEIGHT, 5.0);
        this->get_parameter_or<bool>("mapping.transform_to_base_link_frame", transform_to_base_link, false);
        this->get_parameter_or<bool>("mapping.output_base_link_origin_odom", output_base_link_origin_odom, false);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T_imu_to_base_link", extrinT_imu_to_base_link, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R_imu_to_base_link", extrinR_imu_to_base_link, vector<double>());
        this->get_parameter_or<string>("mapping.base_link_world_frame_id", base_link_world_frame_id, string("map"));
        this->get_parameter_or<string>("mapping.output_body_frame_id", output_body_frame_id, string("base_link"));
        this->get_parameter_or<double>("preprocess.mapping_blind", preprocess_mapping_blind, 2.0);
        this->get_parameter_or<double>("preprocess.localization_blind", preprocess_localization_blind, 0.3);
        this->get_parameter_or<string>("preprocess.blind_filter_shape", blind_filter_shape_param, "sphere");
        this->get_parameter_or<double>("preprocess.blind_z_min", blind_z_min_param, -1.0e9);
        this->get_parameter_or<double>("preprocess.blind_z_max", blind_z_max_param, 1.0e9);
        this->get_parameter_or<double>("preprocess.max_range", preprocess_max_range_param, -1.0);
        this->get_parameter_or<string>("preprocess.tag_filter_mode", tag_filter_mode_param, "low_confidence");
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, false);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<double>("initialization.init_time_s", INIT_TIME, 0.1);
        this->get_parameter_or<double>(
            "prior_map.relocalization_accum_time_s", prior_relocalization_accum_time_s, 0.0);
        if (!this->has_parameter("localization_health.auto_relocalize_enable"))
            this->declare_parameter<bool>("localization_health.auto_relocalize_enable", localization_auto_relocalize_enable);
        if (!this->has_parameter("localization_health.unhealthy_consecutive_frames"))
            this->declare_parameter<int>("localization_health.unhealthy_consecutive_frames", localization_unhealthy_consecutive_frames);
        if (!this->has_parameter("localization_health.min_effective_points"))
            this->declare_parameter<int>("localization_health.min_effective_points", localization_min_effective_points);
        if (!this->has_parameter("localization_health.restart_on_timestamp_rollback"))
            this->declare_parameter<bool>("localization_health.restart_on_timestamp_rollback", localization_restart_on_timestamp_rollback);
        this->get_parameter_or<bool>("localization_health.auto_relocalize_enable", localization_auto_relocalize_enable, true);
        this->get_parameter_or<int>("localization_health.unhealthy_consecutive_frames", localization_unhealthy_consecutive_frames, 10);
        this->get_parameter_or<int>("localization_health.min_effective_points", localization_min_effective_points, 100);
        this->get_parameter_or<bool>("localization_health.restart_on_timestamp_rollback", localization_restart_on_timestamp_rollback, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);
        this->get_parameter_or<double>("pcd_save.voxel_leaf", pcd_save_voxel_leaf, 0.0);
        string manual_loop_session_dir_param = "";
        this->get_parameter_or<bool>("manual_loop_export.enable", manual_loop_export_enable, true);
        this->get_parameter_or<string>("manual_loop_export.session_dir", manual_loop_session_dir_param, string(""));
        this->get_parameter_or<bool>("manual_loop_export.overwrite", manual_loop_export_overwrite, true);
        source_ray_export::Config source_ray_export_config;
        vector<double> source_ray_front_transform;
        vector<double> source_ray_back_transform;
        this->get_parameter_or<bool>(
            "source_ray_export.enable", source_ray_export_config.enable, false);
        this->get_parameter_or<bool>(
            "source_ray_export.overwrite", source_ray_export_config.overwrite, false);
        this->get_parameter_or<bool>(
            "source_ray_export.save_sensor_pcd",
            source_ray_export_config.save_sensor_pcd, true);
        this->get_parameter_or<string>(
            "source_ray_export.manifest_csv",
            source_ray_export_config.manifest_csv, string(""));
        this->get_parameter_or<string>(
            "source_ray_export.pgo_tum",
            source_ray_export_config.pgo_tum, string(""));
        this->get_parameter_or<string>(
            "source_ray_export.output_dir",
            source_ray_export_config.output_dir, string(""));
        this->get_parameter_or<int>(
            "source_ray_export.expected_frame_count",
            source_ray_export_config.expected_frame_count, 0);
        this->get_parameter_or<double>(
            "source_ray_export.timestamp_tolerance_us",
            source_ray_export_config.timestamp_tolerance_us, 2.0);
        this->get_parameter_or<double>(
            "source_ray_export.blind_radius_m",
            source_ray_export_config.blind_radius_m, 0.3);
        this->get_parameter_or<double>(
            "source_ray_export.blind_z_min_m",
            source_ray_export_config.blind_z_min_m, -0.5);
        this->get_parameter_or<double>(
            "source_ray_export.blind_z_max_m",
            source_ray_export_config.blind_z_max_m, 2.0);
        this->get_parameter_or<double>(
            "source_ray_export.maximum_range_m",
            source_ray_export_config.maximum_range_m, 30.0);
        this->get_parameter_or<double>(
            "source_ray_export.endpoint_voxel_m",
            source_ray_export_config.endpoint_voxel_m, 0.1);
        this->get_parameter_or<int>(
            "source_ray_export.deskew_validation_frames",
            source_ray_export_config.deskew_validation_frames, 3);
        this->get_parameter_or<vector<double>>(
            "source_ray_export.front_T_input_sensor",
            source_ray_front_transform, vector<double>());
        this->get_parameter_or<vector<double>>(
            "source_ray_export.back_T_input_sensor",
            source_ray_back_transform, vector<double>());
        this->get_parameter_or<vector<int64_t>>(
            "source_ray_export.selected_pose_indices",
            source_ray_export_config.selected_pose_indices, vector<int64_t>());
        bool mapping_use_prior_map = false;
        bool localization_use_prior_map = true;
        this->get_parameter_or<bool>("prior_map.mapping_use_prior_map", mapping_use_prior_map, false);
        this->get_parameter_or<bool>("prior_map.localization_use_prior_map", localization_use_prior_map, true);
        this->get_parameter_or<int>("prior_map.icp_max_iterations", prior_icp_max_iterations, 60);
        this->get_parameter_or<double>("prior_map.icp_max_corr_dist", prior_icp_max_corr_dist, 3.0);
        this->get_parameter_or<double>("prior_map.icp_fitness_thresh", prior_icp_fitness_thresh, 0.8);
        this->get_parameter_or<double>("prior_map.icp_min_overlap_ratio", prior_icp_min_overlap_ratio, 0.5);
        this->get_parameter_or<int>("prior_map.icp_min_points", prior_icp_min_points, 2000);
        this->get_parameter_or<double>("prior_map.voxel_leaf", prior_map_voxel_leaf, 0.5);
        this->get_parameter_or<double>("prior_map.voxel_leaf_fine", prior_map_voxel_leaf_fine, 0.25);
        this->get_parameter_or<int>("prior_map.icp_refine_top_k", prior_icp_refine_top_k, 3);
        this->get_parameter_or<bool>("prior_map.multi_seed_enable", prior_multi_seed_enable, true);
        this->get_parameter_or<vector<double>>("prior_map.initial_guess_xy", prior_initial_guess_xy, vector<double>(2, 0.0));
        this->get_parameter_or<double>("prior_map.initial_guess_yaw_deg", prior_initial_guess_yaw_deg, 0.0);
        this->get_parameter_or<double>("prior_map.seed_xy_range", prior_seed_xy_range, 2.0);
        this->get_parameter_or<double>("prior_map.seed_xy_step", prior_seed_xy_step, 1.0);
        this->get_parameter_or<double>("prior_map.seed_yaw_range_deg", prior_seed_yaw_range_deg, 20.0);
        this->get_parameter_or<double>("prior_map.seed_yaw_step_deg", prior_seed_yaw_step_deg, 10.0);
        string scan_context_database_path_param = "";
        this->get_parameter_or<bool>("prior_map.scan_context.enable", scan_context_enable, true);
        this->get_parameter_or<string>("prior_map.scan_context.database_path", scan_context_database_path_param, string(""));
        this->get_parameter_or<double>("prior_map.scan_context.keyframe_meter_gap", scan_context_keyframe_meter_gap, 1.0);
        this->get_parameter_or<double>("prior_map.scan_context.keyframe_yaw_gap_deg", scan_context_keyframe_yaw_gap_deg, 10.0);
        this->get_parameter_or<double>("prior_map.scan_context.voxel_leaf", scan_context_voxel_leaf, 0.4);
        this->get_parameter_or<double>("prior_map.scan_context.seed_xy_offset", scan_context_seed_xy_offset, 0.5);
        this->get_parameter_or<int>("prior_map.scan_context.num_rings", scan_context_config.num_rings, 20);
        this->get_parameter_or<int>("prior_map.scan_context.num_sectors", scan_context_config.num_sectors, 60);
        this->get_parameter_or<double>("prior_map.scan_context.max_radius", scan_context_config.max_radius, 80.0);
        this->get_parameter_or<bool>("prior_map.scan_context.dual_z_layer_enable", scan_context_config.dual_z_layer_enable, false);
        this->get_parameter_or<double>("prior_map.scan_context.dual_z_split_height", scan_context_config.dual_z_split_height, 2.5);
        this->get_parameter_or<bool>(
            "prior_map.scan_context.dual_z_split_auto",
            scan_context_config.dual_z_split_auto, false);
        this->get_parameter_or<double>(
            "prior_map.scan_context.dual_z_split_auto_min",
            scan_context_config.dual_z_split_auto_min, 1.5);
        this->get_parameter_or<double>(
            "prior_map.scan_context.dual_z_split_auto_max",
            scan_context_config.dual_z_split_auto_max, 4.5);
        this->get_parameter_or<double>(
            "prior_map.scan_context.dual_z_split_auto_bin_size",
            scan_context_config.dual_z_split_auto_bin_size, 0.1);
        this->get_parameter_or<double>(
            "prior_map.scan_context.dual_z_split_auto_histogram_max",
            scan_context_config.dual_z_split_auto_histogram_max, 8.0);
        this->get_parameter_or<double>(
            "prior_map.scan_context.dual_z_split_auto_min_layer_fraction",
            scan_context_config.dual_z_split_auto_min_layer_fraction, 0.05);
        this->get_parameter_or<int>(
            "prior_map.scan_context.dual_z_split_auto_min_keyframes",
            scan_context_config.dual_z_split_auto_min_keyframes, 20);
        this->get_parameter_or<double>(
            "prior_map.scan_context.origin_height_from_ground",
            scan_context_config.origin_height_from_ground, 0.0);
        this->get_parameter_or<double>("prior_map.scan_context.dual_z_low_weight", scan_context_config.dual_z_low_weight, 0.3);
        this->get_parameter_or<double>("prior_map.scan_context.dual_z_high_weight", scan_context_config.dual_z_high_weight, 0.7);
        this->get_parameter_or<int>("prior_map.scan_context.min_joint_rings", scan_context_config.min_joint_rings, 2);
        this->get_parameter_or<double>(
            "prior_map.scan_context.absent_upper_fallback_max_local_fraction",
            scan_context_config.absent_upper_fallback_max_local_fraction, 0.05);
        this->get_parameter_or<double>(
            "prior_map.scan_context.absent_upper_fallback_radius",
            scan_context_config.absent_upper_fallback_radius, 10.0);
        this->get_parameter_or<int>(
            "prior_map.scan_context.absent_upper_fallback_min_keyframes",
            scan_context_config.absent_upper_fallback_min_keyframes, 3);
        this->get_parameter_or<double>(
            "prior_map.scan_context.retrieval_height_offset",
            scan_context_config.retrieval_height_offset, 0.1);
        this->get_parameter_or<double>(
            "prior_map.scan_context.sector_support_exponent",
            scan_context_config.sector_support_exponent, 0.5);
        this->get_parameter_or<double>(
            "prior_map.scan_context.vertical_boundary_margin",
            scan_context_config.vertical_boundary_margin, 0.1);
        this->get_parameter_or<bool>(
            "prior_map.scan_context.gravity_canonicalization_enable",
            scan_context_config.gravity_canonicalized, true);
        this->get_parameter_or<bool>(
            "prior_map.scan_context.vertical_estimation_enable",
            scan_context_config.vertical_estimation_enable, false);
        this->get_parameter_or<double>(
            "prior_map.scan_context.vertical_correction_min",
            scan_context_config.vertical_correction_min, -1.5);
        this->get_parameter_or<double>(
            "prior_map.scan_context.vertical_correction_max",
            scan_context_config.vertical_correction_max, 1.5);
        this->get_parameter_or<double>(
            "prior_map.scan_context.vertical_stable_fraction",
            scan_context_config.vertical_stable_fraction, 0.5);
        this->get_parameter_or<int>("prior_map.scan_context.candidate_top_k", scan_context_config.candidate_top_k, 5);
        this->get_parameter_or<int>("prior_map.scan_context.yaw_top_k", scan_context_config.yaw_top_k, 3);
        this->get_parameter_or<double>("prior_map.scan_context.distance_thresh", scan_context_config.distance_thresh, 0.5);
        string lidar_extrinsic_profile = "";
        vector<double> front_extrinT;
        vector<double> front_extrinR;
        vector<double> fusion_extrinT;
        vector<double> fusion_extrinR;
        this->get_parameter_or<string>("mapping.lidar_extrinsic_profile", lidar_extrinsic_profile, string(""));
        this->get_parameter_or<vector<double>>("mapping.extrinsic_profiles.front.T", front_extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_profiles.front.R", front_extrinR, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_profiles.fusion.T", fusion_extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_profiles.fusion.R", fusion_extrinR, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());

        auto finite_or_default = [this](double value, double fallback, const char *name) {
            if (std::isfinite(value))
                return value;
            RCLCPP_WARN(this->get_logger(), "%s is not finite; using %.6f.", name, fallback);
            return fallback;
        };
        auto vector_has_finite_size = [](const vector<double> &values, std::size_t expected_size) {
            return values.size() == expected_size &&
                   std::all_of(values.begin(), values.end(), [](double value) {
                       return std::isfinite(value);
                   });
        };
        scan_body_pub_stride_s =
            std::max(0.0, finite_or_default(scan_body_pub_stride_s, 0.0, "publish.scan_bodyframe_stride_s"));
        scan_body_pub_sample_s =
            std::max(0.0, finite_or_default(scan_body_pub_sample_s, 0.2, "publish.scan_bodyframe_sample_s"));
        if (scan_body_pub_stride_s > 1e-6)
            scan_body_pub_sample_s = std::min(scan_body_pub_sample_s, scan_body_pub_stride_s);
        if (scan_body_pub_en && scan_body_pub_stride_s > 1e-6)
        {
            RCLCPP_INFO(
                this->get_logger(),
                "Publishing FAST-LIO-undistorted body clouds in %.3fs windows every %.3fs.",
                scan_body_pub_sample_s, scan_body_pub_stride_s);
        }
        auto trim_string = [](string value) {
            auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
            value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
            value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
            return value;
        };
        auto normalize_reliability = [this, &trim_string](string value, const char *name) {
            value = trim_string(value);
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });
            if (value == "reliable" || value == "best_effort")
                return value;
            RCLCPP_WARN(
                this->get_logger(),
                "%s must be 'reliable' or 'best_effort'; got '%s', using 'best_effort'.",
                name, value.c_str());
            return string("best_effort");
        };
        auto parse_tag_filter_mode = [this, &trim_string](string value) {
            value = trim_string(value);
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
            RCLCPP_WARN(
                this->get_logger(),
                "preprocess.tag_filter_mode must be off/other/low_confidence/strict; got '%s', using low_confidence.",
                value.c_str());
            return Preprocess::TAG_FILTER_LOW_CONFIDENCE;
        };
        auto parse_blind_filter_shape = [this, &trim_string](string value) {
            value = trim_string(value);
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });
            if (value == "sphere" || value == "spherical")
                return Preprocess::BLIND_FILTER_SPHERE;
            if (value == "cylinder" || value == "cylindrical")
                return Preprocess::BLIND_FILTER_CYLINDER;
            RCLCPP_WARN(
                this->get_logger(),
                "preprocess.blind_filter_shape must be sphere/cylinder; got '%s', using sphere.",
                value.c_str());
            return Preprocess::BLIND_FILTER_SPHERE;
        };

        lid_topic = trim_string(lid_topic);
        imu_topic = trim_string(imu_topic);
        localization_pose_topic = trim_string(localization_pose_topic);
        runtime_profile = trim_string(runtime_profile);
        std::transform(runtime_profile.begin(), runtime_profile.end(), runtime_profile.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (runtime_profile == "map" || runtime_profile == "mapping")
        {
            runtime_profile = "mapping";
        }
        else if (runtime_profile == "loc" || runtime_profile == "localization" || runtime_profile == "localisation" ||
                 runtime_profile == "prior_map")
        {
            runtime_profile = "localization";
        }
        else
        {
            RCLCPP_WARN(
                this->get_logger(),
                "runtime.profile must be mapping/localization; got '%s', using mapping.",
                runtime_profile.c_str());
            runtime_profile = "mapping";
        }
        map_file_path = trim_string(map_file_path);
        base_link_world_frame_id = trim_string(base_link_world_frame_id);
        output_body_frame_id = trim_string(output_body_frame_id);
        lidar_qos_reliability = normalize_reliability(lidar_qos_reliability, "common.lidar_qos_reliability");
        imu_qos_reliability = normalize_reliability(imu_qos_reliability, "common.imu_qos_reliability");
        lidar_extrinsic_profile = trim_string(lidar_extrinsic_profile);
        std::transform(lidar_extrinsic_profile.begin(), lidar_extrinsic_profile.end(), lidar_extrinsic_profile.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (lidar_extrinsic_profile.empty())
            lidar_extrinsic_profile = "fusion";
        {
            const vector<double> *profile_T = nullptr;
            const vector<double> *profile_R = nullptr;
            string profile_name = lidar_extrinsic_profile;
            if (lidar_extrinsic_profile == "front" || lidar_extrinsic_profile == "lidar_front")
            {
                profile_T = &front_extrinT;
                profile_R = &front_extrinR;
                profile_name = "front";
            }
            else if (lidar_extrinsic_profile == "fusion" || lidar_extrinsic_profile == "fused")
            {
                profile_T = &fusion_extrinT;
                profile_R = &fusion_extrinR;
                profile_name = "fusion";
            }
            else
            {
                RCLCPP_WARN(
                    this->get_logger(),
                    "mapping.lidar_extrinsic_profile must be front or fusion; got '%s', using fusion.",
                    lidar_extrinsic_profile.c_str());
                profile_T = &fusion_extrinT;
                profile_R = &fusion_extrinR;
                profile_name = "fusion";
            }

            if (profile_T != nullptr && profile_R != nullptr)
            {
                if (vector_has_finite_size(*profile_T, 3) && vector_has_finite_size(*profile_R, 9))
                {
                    extrinT = *profile_T;
                    extrinR = *profile_R;
                    RCLCPP_INFO(this->get_logger(), "Using '%s' lidar->imu extrinsic profile.", profile_name.c_str());
                }
                else
                {
                    throw std::runtime_error(
                        "mapping.extrinsic_profiles." + profile_name + ".T/R must contain finite lidar->imu extrinsic values.");
                }
            }
        }

        if (lid_topic.empty())
            throw std::runtime_error("common.lid_topic must not be empty.");
        if (imu_topic.empty())
            throw std::runtime_error("common.imu_topic must not be empty.");
        if (localization_pose_topic.empty())
        {
            RCLCPP_WARN(this->get_logger(), "localization_pose_topic is empty; using /localization_pose.");
            localization_pose_topic = "/localization_pose";
        }
        if (base_link_world_frame_id.empty())
        {
            RCLCPP_WARN(this->get_logger(), "mapping.base_link_world_frame_id is empty; using map.");
            base_link_world_frame_id = "map";
        }
        if (output_body_frame_id.empty())
        {
            RCLCPP_WARN(this->get_logger(), "mapping.output_body_frame_id is empty; using base_link.");
            output_body_frame_id = "base_link";
        }

        NUM_MAX_ITERATIONS = std::max(1, NUM_MAX_ITERATIONS);
        lidar_qos_depth = std::max(1, lidar_qos_depth);
        imu_qos_depth = std::max(1, imu_qos_depth);
        imu_rate_hz = std::max(1e-6, finite_or_default(imu_rate_hz, 200.0, "common.imu_rate"));
        time_diff_lidar_to_imu = finite_or_default(time_diff_lidar_to_imu, 0.0, "common.time_offset_lidar_to_imu");
        filter_size_corner_min = std::max(1e-3, finite_or_default(filter_size_corner_min, 0.5, "filter_size_corner"));
        filter_size_surf_min = std::max(1e-3, finite_or_default(filter_size_surf_min, 0.5, "filter_size_surf"));
        filter_size_map_min = std::max(1e-3, finite_or_default(filter_size_map_min, 0.5, "filter_size_map"));
        cube_len = std::max(1.0, finite_or_default(cube_len, 200.0, "cube_side_length"));
        DET_RANGE = static_cast<float>(std::max(1.0, finite_or_default(det_range_param, 300.0, "mapping.det_range")));
        if (std::isfinite(preprocess_max_range_param) && preprocess_max_range_param > 0.0)
            DET_RANGE = static_cast<float>(std::max(1.0, preprocess_max_range_param));
        fov_deg = std::clamp(finite_or_default(fov_deg, 180.0, "mapping.fov_degree"), 1.0, 360.0);
        gyr_cov = std::max(0.0, finite_or_default(gyr_cov, 0.1, "mapping.gyr_cov"));
        acc_cov = std::max(0.0, finite_or_default(acc_cov, 0.1, "mapping.acc_cov"));
        b_gyr_cov = std::max(0.0, finite_or_default(b_gyr_cov, 0.0001, "mapping.b_gyr_cov"));
        b_acc_cov = std::max(0.0, finite_or_default(b_acc_cov, 0.0001, "mapping.b_acc_cov"));
        MAX_HEIGHT = std::max(0.0, finite_or_default(MAX_HEIGHT, 5.0, "mapping.max_height"));
        preprocess_mapping_blind =
            std::max(0.0, finite_or_default(preprocess_mapping_blind, 2.0, "preprocess.mapping_blind"));
        preprocess_localization_blind =
            std::max(0.0, finite_or_default(preprocess_localization_blind, 0.3, "preprocess.localization_blind"));
        blind_z_min_param = finite_or_default(blind_z_min_param, -1.0e9, "preprocess.blind_z_min");
        blind_z_max_param = finite_or_default(blind_z_max_param, 1.0e9, "preprocess.blind_z_max");
        if (blind_z_min_param > blind_z_max_param)
            std::swap(blind_z_min_param, blind_z_max_param);
        const int parsed_blind_filter_shape = parse_blind_filter_shape(blind_filter_shape_param);
        if (runtime_profile == "localization")
        {
            p_pre->blind = preprocess_localization_blind;
            use_prior_map = localization_use_prior_map;
        }
        else
        {
            p_pre->blind = preprocess_mapping_blind;
            use_prior_map = mapping_use_prior_map;
        }
        localization_output_trusted = !use_prior_map;
        scan_context_blind_filter_shape = parsed_blind_filter_shape;
        scan_context_blind_z_min = blind_z_min_param;
        scan_context_blind_z_max = blind_z_max_param;
        INIT_TIME = std::max(0.0, finite_or_default(INIT_TIME, 0.1, "initialization.init_time_s"));
        prior_relocalization_accum_time_s = std::max(
            0.0,
            finite_or_default(
                prior_relocalization_accum_time_s, 0.0,
                "prior_map.relocalization_accum_time_s"));
        if (scan_context_config.gravity_canonicalized &&
            prior_relocalization_accum_time_s > 1e-9)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Gravity-canonicalized relocalization requires one reference attitude; "
                "forcing prior_map.relocalization_accum_time_s from %.3f to 0.0.",
                prior_relocalization_accum_time_s);
            prior_relocalization_accum_time_s = 0.0;
        }
        localization_unhealthy_consecutive_frames = std::max(1, localization_unhealthy_consecutive_frames);
        localization_min_effective_points = std::max(1, localization_min_effective_points);
        pcd_save_voxel_leaf = std::max(0.0, finite_or_default(pcd_save_voxel_leaf, 0.0, "pcd_save.voxel_leaf"));
        p_pre->SCAN_RATE = std::max(1, p_pre->SCAN_RATE);
        lidar_frame_period_sec = 1.0 / static_cast<double>(p_pre->SCAN_RATE);
        imu_frame_period_sec = 1.0 / imu_rate_hz;

        if (!vector_has_finite_size(extrinT, 3))
        {
            RCLCPP_WARN(this->get_logger(), "mapping.extrinsic_T must have 3 finite elements; using zero translation.");
            extrinT.assign(3, 0.0);
        }
        if (!vector_has_finite_size(extrinR, 9))
        {
            RCLCPP_WARN(this->get_logger(), "mapping.extrinsic_R must have 9 finite elements; using identity rotation.");
            extrinR = {1.0, 0.0, 0.0,
                       0.0, 1.0, 0.0,
                       0.0, 0.0, 1.0};
        }
        const int configured_scan_lines = p_pre->N_SCANS;
        p_pre->N_SCANS = std::clamp(p_pre->N_SCANS, 1, 128);
        if (p_pre->N_SCANS != configured_scan_lines)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "preprocess.scan_line=%d is outside supported range [1, 128]; using %d.",
                configured_scan_lines, p_pre->N_SCANS);
        }
        if (p_pre->lidar_type < AVIA || p_pre->lidar_type > GAZEBO_XYZI)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "preprocess.lidar_type=%d is unsupported; using AVIA(%d).",
                p_pre->lidar_type, AVIA);
            p_pre->lidar_type = AVIA;
        }
        p_imu->deskew_en = deskew_en;
        p_pre->set(p_pre->feature_enabled, p_pre->lidar_type, p_pre->blind, DET_RANGE, MAX_HEIGHT, p_pre->point_filter_num);
        p_pre->set_tag_filter_mode(parse_tag_filter_mode(tag_filter_mode_param));
        p_pre->set_blind_filter(parsed_blind_filter_shape, blind_z_min_param, blind_z_max_param);

        if (source_ray_export_config.enable)
        {
            if (runtime_profile != "mapping" || use_prior_map)
                throw std::runtime_error(
                    "source_ray_export requires mapping mode without a prior map.");
            if (p_pre->lidar_type != MID360)
                throw std::runtime_error(
                    "source_ray_export currently requires preprocess.lidar_type=MID360.");
            if (!vector_has_finite_size(extrinT_imu_to_base_link, 3) ||
                !vector_has_finite_size(extrinR_imu_to_base_link, 9))
            {
                throw std::runtime_error(
                    "source_ray_export requires a finite explicit "
                    "LiDAR-IMU-to-base_link extrinsic.");
            }
            if (!deskew_en)
                throw std::runtime_error(
                    "source_ray_export requires common.deskew_en=true.");
            if (extrinsic_est_en)
                throw std::runtime_error(
                    "source_ray_export requires mapping.extrinsic_est_en=false "
                    "so the frame-end base transform cannot change after "
                    "deskew.");
            if (p_pre->tag_filter_mode != Preprocess::TAG_FILTER_STRICT)
                throw std::runtime_error(
                    "source_ray_export requires "
                    "preprocess.tag_filter_mode=strict.");
            if (p_pre->blind_filter_shape !=
                Preprocess::BLIND_FILTER_CYLINDER)
            {
                throw std::runtime_error(
                    "source_ray_export requires "
                    "preprocess.blind_filter_shape=cylinder.");
            }
            const auto exact_filter_match =
                [](double lhs, double rhs)
            {
                return std::isfinite(lhs) && std::isfinite(rhs) &&
                       std::abs(lhs - rhs) <= 1.0e-12;
            };
            if (!exact_filter_match(
                    source_ray_export_config.blind_radius_m, p_pre->blind) ||
                !exact_filter_match(
                    source_ray_export_config.blind_z_min_m,
                    p_pre->blind_z_min) ||
                !exact_filter_match(
                    source_ray_export_config.blind_z_max_m,
                    p_pre->blind_z_max) ||
                !exact_filter_match(
                    source_ray_export_config.maximum_range_m,
                    p_pre->det_range))
            {
                throw std::runtime_error(
                    "source_ray_export blind/range filters must exactly match "
                    "the active preprocessing filters.");
            }
            source_ray_export_config.scan_line = p_pre->N_SCANS;
            if (!vector_has_finite_size(source_ray_front_transform, 16) ||
                !vector_has_finite_size(source_ray_back_transform, 16))
            {
                throw std::runtime_error(
                    "source_ray_export front/back transforms must each contain "
                    "16 finite row-major values.");
            }
            auto parse_source_transform =
                [](const vector<double> &values, const char *name)
            {
                Eigen::Matrix4d transform;
                for (int row = 0; row < 4; ++row)
                {
                    for (int column = 0; column < 4; ++column)
                    {
                        transform(row, column) =
                            values[static_cast<std::size_t>(row * 4 + column)];
                    }
                }
                const Eigen::Matrix3d rotation =
                    transform.block<3, 3>(0, 0);
                const double orthogonality_error =
                    (rotation.transpose() * rotation -
                     Eigen::Matrix3d::Identity()).norm();
                if (orthogonality_error > 1.0e-5 ||
                    std::abs(rotation.determinant() - 1.0) > 1.0e-5 ||
                    (transform.row(3) -
                     Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0)).norm() >
                        1.0e-9)
                {
                    throw std::runtime_error(
                        string("source_ray_export invalid rigid transform: ") +
                        name);
                }
                return transform;
            };
            source_ray_export_config.T_input_front =
                parse_source_transform(
                    source_ray_front_transform, "front_T_input_sensor");
            source_ray_export_config.T_input_back =
                parse_source_transform(
                    source_ray_back_transform, "back_T_input_sensor");
            source_ray_export_config.manifest_csv =
                trim_string(source_ray_export_config.manifest_csv);
            source_ray_export_config.pgo_tum =
                trim_string(source_ray_export_config.pgo_tum);
            source_ray_export_config.output_dir =
                trim_string(source_ray_export_config.output_dir);
            source_ray_export_config.deskew_validation_frames =
                std::max(0, source_ray_export_config.deskew_validation_frames);
        }

        RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);
        RCLCPP_INFO(
            this->get_logger(),
            "runtime.profile %s use_prior_map=%s preprocess_blind=%.3f mapping_blind=%.3f localization_blind=%.3f",
            runtime_profile.c_str(), use_prior_map ? "true" : "false", p_pre->blind,
            preprocess_mapping_blind, preprocess_localization_blind);
        RCLCPP_INFO(
            this->get_logger(),
            "preprocess.blind_filter_shape %s z=[%.3f %.3f] scan_context_blind=%.3f",
            blind_filter_shape_param.c_str(), blind_z_min_param, blind_z_max_param,
            preprocess_mapping_blind);
        RCLCPP_INFO(this->get_logger(), "preprocess.tag_filter_mode %s", tag_filter_mode_param.c_str());
#ifdef MP_EN
        RCLCPP_INFO(this->get_logger(), "\033[1;32mFAST-LIO OpenMP enabled, MP_PROC_NUM=%d\033[0m", MP_PROC_NUM);
#else
        RCLCPP_INFO(this->get_logger(), "\033[1;33mFAST-LIO OpenMP disabled\033[0m");
#endif

        if (prior_initial_guess_xy.size() != 2)
        {
            RCLCPP_WARN(this->get_logger(), "prior_map.initial_guess_xy must have 2 elements [x, y]. Falling back to [0.0, 0.0].");
            prior_initial_guess_xy.assign(2, 0.0);
        }
        else if (!std::isfinite(prior_initial_guess_xy[0]) || !std::isfinite(prior_initial_guess_xy[1]))
        {
            RCLCPP_WARN(this->get_logger(), "prior_map.initial_guess_xy contains non-finite values. Falling back to [0.0, 0.0].");
            prior_initial_guess_xy.assign(2, 0.0);
        }

        prior_icp_max_iterations = std::max(1, prior_icp_max_iterations);
        prior_icp_max_corr_dist = std::max(1e-3, finite_or_default(prior_icp_max_corr_dist, 3.0, "prior_map.icp_max_corr_dist"));
        prior_icp_fitness_thresh = std::max(0.0, finite_or_default(prior_icp_fitness_thresh, 0.8, "prior_map.icp_fitness_thresh"));
        prior_icp_min_overlap_ratio =
            std::clamp(finite_or_default(prior_icp_min_overlap_ratio, 0.5, "prior_map.icp_min_overlap_ratio"), 0.0, 1.0);
        prior_icp_min_points = std::max(10, prior_icp_min_points);
        prior_map_voxel_leaf = std::max(0.0, finite_or_default(prior_map_voxel_leaf, 0.5, "prior_map.voxel_leaf"));
        prior_map_voxel_leaf_fine = std::max(0.0, finite_or_default(prior_map_voxel_leaf_fine, 0.25, "prior_map.voxel_leaf_fine"));
        prior_icp_refine_top_k = std::max(1, prior_icp_refine_top_k);
        prior_initial_guess_yaw_deg =
            finite_or_default(prior_initial_guess_yaw_deg, 0.0, "prior_map.initial_guess_yaw_deg");
        prior_seed_xy_range = std::max(0.0, finite_or_default(prior_seed_xy_range, 2.0, "prior_map.seed_xy_range"));
        prior_seed_xy_step = std::max(0.1, finite_or_default(prior_seed_xy_step, 1.0, "prior_map.seed_xy_step"));
        prior_seed_yaw_range_deg =
            std::max(0.0, finite_or_default(prior_seed_yaw_range_deg, 20.0, "prior_map.seed_yaw_range_deg"));
        prior_seed_yaw_step_deg =
            std::max(1.0, finite_or_default(prior_seed_yaw_step_deg, 10.0, "prior_map.seed_yaw_step_deg"));
        if (prior_seed_xy_range > 20.0)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "prior_map.seed_xy_range=%.3f is very large; clamping to 20.0m to avoid excessive ICP seeds.",
                prior_seed_xy_range);
            prior_seed_xy_range = 20.0;
        }
        if (prior_seed_yaw_range_deg > 180.0)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "prior_map.seed_yaw_range_deg=%.3f is very large; clamping to 180deg.",
                prior_seed_yaw_range_deg);
            prior_seed_yaw_range_deg = 180.0;
        }
        scan_context_keyframe_meter_gap =
            std::max(0.05, finite_or_default(scan_context_keyframe_meter_gap, 1.0, "prior_map.scan_context.keyframe_meter_gap"));
        scan_context_keyframe_yaw_gap_deg =
            std::max(0.1, finite_or_default(scan_context_keyframe_yaw_gap_deg, 10.0, "prior_map.scan_context.keyframe_yaw_gap_deg"));
        scan_context_keyframe_yaw_gap_rad = scan_context_keyframe_yaw_gap_deg * M_PI / 180.0;
        scan_context_voxel_leaf =
            std::max(0.0, finite_or_default(scan_context_voxel_leaf, 0.4, "prior_map.scan_context.voxel_leaf"));
        scan_context_seed_xy_offset =
            std::max(0.0, finite_or_default(scan_context_seed_xy_offset, 0.5, "prior_map.scan_context.seed_xy_offset"));
        scan_context_config.num_rings = std::max(1, scan_context_config.num_rings);
        scan_context_config.num_sectors = std::max(4, scan_context_config.num_sectors);
        scan_context_config.max_radius =
            std::max(1.0, finite_or_default(scan_context_config.max_radius, 80.0, "prior_map.scan_context.max_radius"));
        scan_context_config.dual_z_split_height =
            finite_or_default(scan_context_config.dual_z_split_height, 2.5, "prior_map.scan_context.dual_z_split_height");
        scan_context_config.dual_z_split_auto_min = std::max(
            0.0, finite_or_default(
                scan_context_config.dual_z_split_auto_min, 1.5,
                "prior_map.scan_context.dual_z_split_auto_min"));
        scan_context_config.dual_z_split_auto_max = std::max(
            scan_context_config.dual_z_split_auto_min + 0.1,
            finite_or_default(
                scan_context_config.dual_z_split_auto_max, 4.5,
                "prior_map.scan_context.dual_z_split_auto_max"));
        scan_context_config.dual_z_split_auto_bin_size = std::max(
            0.01, finite_or_default(
                scan_context_config.dual_z_split_auto_bin_size, 0.1,
                "prior_map.scan_context.dual_z_split_auto_bin_size"));
        scan_context_config.dual_z_split_auto_histogram_max = std::max(
            scan_context_config.dual_z_split_auto_max +
                scan_context_config.dual_z_split_auto_bin_size,
            finite_or_default(
                scan_context_config.dual_z_split_auto_histogram_max, 8.0,
                "prior_map.scan_context.dual_z_split_auto_histogram_max"));
        scan_context_config.dual_z_split_auto_min_layer_fraction = std::clamp(
            finite_or_default(
                scan_context_config.dual_z_split_auto_min_layer_fraction, 0.05,
                "prior_map.scan_context.dual_z_split_auto_min_layer_fraction"),
            0.001, 0.49);
        scan_context_config.dual_z_split_auto_min_keyframes = std::max(
            1, scan_context_config.dual_z_split_auto_min_keyframes);
        scan_context_config.origin_height_from_ground = std::max(
            0.0, finite_or_default(
                scan_context_config.origin_height_from_ground, 0.0,
                "prior_map.scan_context.origin_height_from_ground"));
        scan_context_config.dual_z_low_weight =
            std::max(0.0, finite_or_default(scan_context_config.dual_z_low_weight, 0.3, "prior_map.scan_context.dual_z_low_weight"));
        scan_context_config.dual_z_high_weight =
            std::max(0.0, finite_or_default(scan_context_config.dual_z_high_weight, 0.7, "prior_map.scan_context.dual_z_high_weight"));
        if (scan_context_config.dual_z_low_weight + scan_context_config.dual_z_high_weight <= 1e-12)
        {
            scan_context_config.dual_z_low_weight = 0.3;
            scan_context_config.dual_z_high_weight = 0.7;
        }
        scan_context_config.min_joint_rings = std::clamp(
            scan_context_config.min_joint_rings, 1, scan_context_config.num_rings);
        scan_context_config.absent_upper_fallback_max_local_fraction = std::clamp(
            finite_or_default(
                scan_context_config.absent_upper_fallback_max_local_fraction, 0.05,
                "prior_map.scan_context.absent_upper_fallback_max_local_fraction"),
            0.0, 1.0);
        scan_context_config.absent_upper_fallback_radius = std::max(
            0.1, finite_or_default(
                scan_context_config.absent_upper_fallback_radius, 10.0,
                "prior_map.scan_context.absent_upper_fallback_radius"));
        scan_context_config.absent_upper_fallback_min_keyframes = std::max(
            1, scan_context_config.absent_upper_fallback_min_keyframes);
        scan_context_config.retrieval_height_offset = std::max(
            0.0, finite_or_default(
                scan_context_config.retrieval_height_offset, 0.1,
                "prior_map.scan_context.retrieval_height_offset"));
        scan_context_config.sector_support_exponent = std::max(
            0.0, finite_or_default(
                scan_context_config.sector_support_exponent, 0.5,
                "prior_map.scan_context.sector_support_exponent"));
        scan_context_config.vertical_boundary_margin = std::max(
            0.0, finite_or_default(scan_context_config.vertical_boundary_margin, 0.1,
                                   "prior_map.scan_context.vertical_boundary_margin"));
        scan_context_config.vertical_correction_min = finite_or_default(
            scan_context_config.vertical_correction_min, -1.5,
            "prior_map.scan_context.vertical_correction_min");
        scan_context_config.vertical_correction_max = finite_or_default(
            scan_context_config.vertical_correction_max, 1.5,
            "prior_map.scan_context.vertical_correction_max");
        if (scan_context_config.vertical_correction_min > scan_context_config.vertical_correction_max)
            std::swap(scan_context_config.vertical_correction_min, scan_context_config.vertical_correction_max);
        scan_context_config.vertical_stable_fraction = std::clamp(
            finite_or_default(scan_context_config.vertical_stable_fraction, 0.5,
                              "prior_map.scan_context.vertical_stable_fraction"),
            1e-3, 1.0);
        scan_context_config.candidate_top_k = std::max(1, scan_context_config.candidate_top_k);
        scan_context_config.yaw_top_k = std::max(1, scan_context_config.yaw_top_k);
        scan_context_config.distance_thresh =
            std::max(1e-6, finite_or_default(scan_context_config.distance_thresh, 0.5, "prior_map.scan_context.distance_thresh"));
        scan_context_database_path = derive_scan_context_database_path(scan_context_database_path_param, map_file_path);
        scan_context_db.setConfig(scan_context_config);
        scan_context_query_builder.setConfig(scan_context_config);
        scan_context_split_estimator.setConfig(scan_context_config);
        manual_loop_session_dir = derive_manual_loop_session_dir(manual_loop_session_dir_param);
        manual_loop_keyframe_dir = (std::filesystem::path(manual_loop_session_dir) / "key_point_frame").string();
        manual_loop_g2o_path = (std::filesystem::path(manual_loop_session_dir) / "pose_graph.g2o").string();
        manual_loop_tum_path = (std::filesystem::path(manual_loop_session_dir) / "optimized_poses_tum.txt").string();
        manual_loop_gravity_path = (std::filesystem::path(manual_loop_session_dir) / "scan_context_gravity.csv").string();

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id = use_base_link_output_frame() ? base_link_world_frame_id : "camera_init";

        // /*** variables definition ***/
        // int effect_feat_num = 0, frame_num = 0;
        // double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
        // bool flg_EKF_converged, EKF_stop_flg = 0;

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        const bool base_link_extrinsic_valid =
            vector_has_finite_size(extrinT_imu_to_base_link, 3) &&
            vector_has_finite_size(extrinR_imu_to_base_link, 9);
        if (base_link_extrinsic_valid)
        {
            BaseLink_T_wrt_LidarIMU << VEC_FROM_ARRAY(extrinT_imu_to_base_link);
            BaseLink_R_wrt_LidarIMU << MAT_FROM_ARRAY(extrinR_imu_to_base_link);
            const M3D expected_base_link_R = Lidar_R_wrt_IMU.transpose();
            const V3D expected_base_link_T = -expected_base_link_R * Lidar_T_wrt_IMU;
            const double base_link_R_err = (BaseLink_R_wrt_LidarIMU - expected_base_link_R).norm();
            const double base_link_T_err = (BaseLink_T_wrt_LidarIMU - expected_base_link_T).norm();
            if (base_link_R_err > 1e-3 || base_link_T_err > 1e-3)
            {
                RCLCPP_INFO(
                    this->get_logger(),
                    "Using independent imu->base_link extrinsic; it is not the inverse of mapping.extrinsic_R/T (R_err=%.6f T_err=%.6f).",
                    base_link_R_err, base_link_T_err);
            }
            RCLCPP_INFO(
                this->get_logger(),
                "base_link extrinsic loaded: imu_to_base_t=[%.6f %.6f %.6f] lidar_to_imu_t=[%.6f %.6f %.6f].",
                BaseLink_T_wrt_LidarIMU(0), BaseLink_T_wrt_LidarIMU(1), BaseLink_T_wrt_LidarIMU(2),
                Lidar_T_wrt_IMU(0), Lidar_T_wrt_IMU(1), Lidar_T_wrt_IMU(2));
        }
        else
        {
            BaseLink_R_wrt_LidarIMU = Eye3d;
            BaseLink_T_wrt_LidarIMU = Zero3d;
            if (transform_to_base_link)
            {
                RCLCPP_WARN(this->get_logger(), "base_link frame transformation requested but extrinsic is invalid; disabling transform_to_base_link_frame.");
                transform_to_base_link = false;
            }
        }

        if (use_base_link_output_frame())
        {
            RCLCPP_INFO(this->get_logger(), "Publishing base_link-origin odometry in '%s' frame.", base_link_world_frame_id.c_str());
        }

        source_ray_exporter.reset();
        source_ray_deskew_validation_count = 0;
        if (source_ray_export_config.enable)
        {
            auto exporter =
                std::make_unique<source_ray_export::Exporter>();
            string source_ray_error;
            if (!exporter->configure(
                    source_ray_export_config, source_ray_error))
            {
                throw std::runtime_error(
                    "Failed to configure exact source-ray exporter: " +
                    source_ray_error);
            }
            source_ray_exporter = std::move(exporter);
            RCLCPP_INFO(
                this->get_logger(),
                "Exact source-ray export enabled: output=%s PGO=%s "
                "identity_manifest=%s targets=%d voxel=%.3fm sensor_pcd=%s.",
                source_ray_export_config.output_dir.c_str(),
                source_ray_export_config.pgo_tum.c_str(),
                source_ray_export_config.manifest_csv.c_str(),
                source_ray_export_config.expected_frame_count,
                source_ray_export_config.endpoint_voxel_m,
                source_ray_export_config.save_sensor_pcd ? "true" : "false");
        }

        if (use_prior_map)
        {
            if (!scan_context_enable)
            {
                RCLCPP_INFO(
                    this->get_logger(), "Prior-map ICP fallback initial guess: x=%.3f, y=%.3f, z=0.000, yaw=%.1f deg",
                    prior_initial_guess_xy[0], prior_initial_guess_xy[1], prior_initial_guess_yaw_deg);
            }
            if (!load_prior_map_from_pcd(map_file_path))
            {
                throw std::runtime_error("Prior-map mode requested, but failed to load prior map.");
            }
            if (scan_context_enable)
            {
                string sc_error;
                if (!scan_context_db.load(scan_context_database_path, &sc_error))
                {
                    throw std::runtime_error(
                        "Prior-map Scan Context enabled, but failed to load database '" +
                        scan_context_database_path + "': " + sc_error +
                        ". Re-run mapping to generate the .scd database.");
                }
                if (scan_context_db.empty())
                {
                    throw std::runtime_error(
                        "Prior-map Scan Context database is empty: " +
                        scan_context_database_path +
                        ". Re-run mapping to generate keyframes.");
                }
                if (scan_context_db.config().gravity_canonicalized !=
                    scan_context_config.gravity_canonicalized)
                {
                    throw std::runtime_error(
                        "Scan Context gravity-canonicalization mismatch for '" +
                        scan_context_database_path + "': database=" +
                        (scan_context_db.config().gravity_canonicalized ? "enabled" : "disabled") +
                        ", runtime=" +
                        (scan_context_config.gravity_canonicalized ? "enabled" : "disabled") +
                        ". Regenerate scans.scd with the current configuration.");
                }
                // Descriptor geometry and the map-level adaptive split belong
                // to the map. Preserve only this platform's physical origin
                // height and runtime query controls while constructing query
                // descriptors in the map's exact representation.
                sc::Config query_config = scan_context_config;
                query_config.num_rings = scan_context_db.config().num_rings;
                query_config.num_sectors = scan_context_db.config().num_sectors;
                query_config.max_radius = scan_context_db.config().max_radius;
                query_config.dual_z_layer_enable =
                    scan_context_db.config().dual_z_layer_enable;
                query_config.dual_z_split_height =
                    scan_context_db.config().dual_z_split_height;
                query_config.dual_z_split_auto = false;
                scan_context_query_builder.setConfig(query_config);
                if (scan_context_db.legacyMasksInferred())
                {
                    RCLCPP_WARN(
                        this->get_logger(),
                        "Loaded legacy Scan Context V1 database without explicit validity masks. "
                        "Masks were inferred from nonzero descriptor values; rebuild the map database for exact V2 masks.");
                }
                scan_context_loaded = true;
                RCLCPP_INFO(
                    this->get_logger(),
                    "Scan Context database loaded: %s entries=%zu local_absent_upper_fallback=%zu rings=%d sectors=%d max_radius=%.1f yaw_top_k=%d map_split=%.3f query_split=%.3f retrieval_height_offset=%.3f vertical_boundary_margin=%.3f vertical_stable_fraction=%.3f gravity_canonicalized=%s",
                    scan_context_database_path.c_str(), scan_context_db.size(),
                    scan_context_db.absentUpperFallbackEntryCount(),
                    scan_context_db.config().num_rings,
                    scan_context_db.config().num_sectors,
                    scan_context_db.config().max_radius,
                    scan_context_db.config().yaw_top_k,
                    sc::effectiveDualZSplitHeight(scan_context_db.config()),
                    sc::effectiveDualZSplitHeight(query_config),
                    scan_context_db.config().retrieval_height_offset,
                    scan_context_db.config().vertical_boundary_margin,
                    scan_context_db.config().vertical_stable_fraction,
                    scan_context_db.config().gravity_canonicalized ? "true" : "false");
            }
            RCLCPP_INFO(this->get_logger(), "Prior-map mode enabled. Waiting Scan Context + ICP success before mapping update.");
        }
        else if (scan_context_enable)
        {
            RCLCPP_INFO(
                this->get_logger(),
                "Scan Context keyframe database will be saved to %s (gap=%.2fm/%.1fdeg).",
                scan_context_database_path.c_str(),
                scan_context_keyframe_meter_gap,
                scan_context_keyframe_yaw_gap_deg);
            if (manual_loop_export_enable)
            {
                RCLCPP_INFO(
                    this->get_logger(),
                    "Manual loop export enabled: %s",
                    manual_loop_session_dir.c_str());
            }
        }

        fill(epsi, epsi+23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        /*** debug record ***/
        // FILE *fp;
        std::error_code log_dir_ec;
        std::filesystem::create_directories(root_dir + "/Log", log_dir_ec);
        if (log_dir_ec)
        {
            RCLCPP_WARN(this->get_logger(), "Failed to create Log directory: %s", log_dir_ec.message().c_str());
        }
        std::error_code pcd_dir_ec;
        std::filesystem::create_directories(root_dir + "/PCD", pcd_dir_ec);
        if (pcd_dir_ec)
        {
            RCLCPP_WARN(this->get_logger(), "Failed to create PCD directory: %s", pcd_dir_ec.message().c_str());
        }
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");
        if (!fp)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Failed to open runtime pose log: %s",
                pos_log_dir.c_str());
        }

        // ofstream fout_pre, fout_out, fout_dbg;
        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
        if (fout_pre && fout_out)
            RCLCPP_DEBUG(this->get_logger(), "Debug files opened under %s", ROOT_DIR);
        else
            RCLCPP_WARN(this->get_logger(), "Failed to open debug files under %s", ROOT_DIR);

        /*** ROS subscribe initialization ***/
        lidar_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        imu_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions lidar_sub_options;
        lidar_sub_options.callback_group = lidar_callback_group_;
        rclcpp::SubscriptionOptions imu_sub_options;
        imu_sub_options.callback_group = imu_callback_group_;
        auto lidar_qos = make_sensor_qos(lidar_qos_depth, lidar_qos_reliability);
        auto imu_qos = make_sensor_qos(imu_qos_depth, imu_qos_reliability);
        RCLCPP_INFO(
            this->get_logger(), "Sensor QoS: lidar=%s depth=%d imu=%s depth=%d",
            lidar_qos_reliability.c_str(), lidar_qos_depth,
            imu_qos_reliability.c_str(), imu_qos_depth);
        RCLCPP_INFO(
            this->get_logger(),
            "Sensor gap diagnostics: lidar_rate=%dHz warn>%.3fs imu_rate=%.1fHz warn>%.6fs",
            p_pre->SCAN_RATE,
            3.0 * lidar_frame_period_sec,
            imu_rate_hz,
            3.0 * imu_frame_period_sec);
        if (p_pre->lidar_type == AVIA)
        {
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                lid_topic, lidar_qos, livox_pcl_cbk, lidar_sub_options);
        }
        else
        {
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                lid_topic, lidar_qos, standard_pcl_cbk, lidar_sub_options);
        }
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, imu_qos, imu_cbk, imu_sub_options);
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubScanContextGravity_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>("/scan_context_gravity_up", 20);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        pubPose_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(localization_pose_topic, 20);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        pubPriorMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("prior_map", rclcpp::QoS(1).transient_local());
        pubScanContextCandidates_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/scan_context_icp_candidates", rclcpp::QoS(1).transient_local());
        if (use_prior_map && prior_map_loaded)
        {
            prior_map_ready_for_publish = true;
            publish_prior_map_once(pubPriorMap_);
        }
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
        node_clock = this->get_clock();
        if (use_base_link_output_frame())
        {
            static_tf_broadcaster_->sendTransform(make_identity_static_tf(
                base_link_world_frame_id, base_link_world_frame_id + "_origin", this->get_clock()->now()));
        }

        //------------------------------------------------------------------------------------------------------
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer_ = this->create_wall_timer(period_ms, std::bind(&LaserMappingNode::timer_callback, this));

        auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
        map_pub_timer_ = this->create_wall_timer(map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this));

        if (debug_save_registered_pcd_en)
        {
            RCLCPP_INFO(
                this->get_logger(),
                "Debug registered PCD save enabled: path=%s frame_interval=%d",
                debug_registered_pcd_path.c_str(), debug_save_registered_pcd_frame_interval);
        }

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

    ~LaserMappingNode()
    {
        if (source_ray_exporter && source_ray_exporter->enabled())
        {
            string source_ray_error;
            if (!source_ray_exporter->finalize(source_ray_error))
            {
                RCLCPP_ERROR(
                    this->get_logger(),
                    "Exact source-ray export final validation failed: %s",
                    source_ray_error.c_str());
            }
            else
            {
                RCLCPP_INFO(
                    this->get_logger(),
                    "Exact source-ray export finalized successfully.");
            }
            source_ray_exporter.reset();
        }
        fout_out.close();
        fout_pre.close();
        if (fp)
        {
            fclose(fp);
            fp = nullptr;
        }
    }

private:
    void timer_callback()
    {
        static uint64_t consumed_lidar_frame = 0;
        bool using_prior_replay_frame = false;
        uint64_t replay_source_frame = 0;
        uint64_t replay_source_rx = 0;
        if (localization_restart_pending.exchange(false, std::memory_order_acq_rel))
        {
            const int rollback_streak =
                localization_timestamp_rollback_streak.exchange(0, std::memory_order_acq_rel);
            const bool localization_tracking_active =
                use_prior_map && prior_map_loaded && prior_map_aligned && prior_map_build_done;
            if (localization_auto_relocalize_enable &&
                localization_restart_on_timestamp_rollback &&
                use_prior_map &&
                prior_map_loaded)
            {
                restart_prior_relocalization_from_health(
                    "timestamp_rollback",
                    std::max(rollback_streak, localization_unhealthy_consecutive_frames),
                    false,
                    true,
                    0,
                    0,
                    last_timestamp_lidar - first_lidar_time,
                    consumed_lidar_frame,
                    last_synced_lidar_rx_index);
                return;
            }
            RCLCPP_WARN(
                this->get_logger(),
                "Timestamp rollback restart request ignored: consecutive=%d/%d use_prior=%s loaded=%s aligned=%s build_done=%s auto=%s restart_on_rollback=%s",
                rollback_streak,
                localization_unhealthy_consecutive_frames,
                use_prior_map ? "true" : "false",
                prior_map_loaded ? "true" : "false",
                prior_map_aligned ? "true" : "false",
                prior_map_build_done ? "true" : "false",
                localization_auto_relocalize_enable ? "true" : "false",
                localization_restart_on_timestamp_rollback ? "true" : "false");
        }
        if (pop_prior_replay_frame(Measures, replay_source_frame, replay_source_rx))
        {
            using_prior_replay_frame = true;
            last_synced_lidar_rx_index = replay_source_rx;
        }
        else if (!sync_packages(Measures))
        {
            return;
        }

        {
            ++consumed_lidar_frame;
            const double frame_span_ms = (Measures.lidar_end_time - Measures.lidar_beg_time) * 1000.0;
            current_lidar_frame_index = consumed_lidar_frame;
            current_lidar_rx_index = last_synced_lidar_rx_index;
            current_lidar_beg_time = Measures.lidar_beg_time;
            current_lidar_end_time = Measures.lidar_end_time;
            current_raw_lidar_points = Measures.lidar ? Measures.lidar->size() : 0;
            current_undistorted_points = 0;

            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                source_ray_current_raw_message.reset();
                RCLCPP_DEBUG(this->get_logger(),
                             "skip lidar frame #%" PRIu64 " rx=%" PRIu64 ": first frame used for timestamp init, span=%.3f ms, imu=%zu",
                             consumed_lidar_frame, last_synced_lidar_rx_index, frame_span_ms, Measures.imu.size());
                return;
            }

            if (!using_prior_replay_frame && use_prior_map && !prior_map_build_done)
            {
                capture_prior_replay_pending_snapshot(consumed_lidar_frame, Measures.lidar_beg_time);
            }

            double t0, t5;
            std::unique_ptr<source_ray_export::PendingFrame>
                source_rays_for_frame;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            current_measurement_no_effective_points = false;
            t0 = omp_get_wtime();

            feats_undistort->clear();
            p_imu->Process(Measures, kf, feats_undistort);
            current_undistorted_points = feats_undistort ? feats_undistort->size() : 0;
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (!feats_undistort || feats_undistort->empty())
            {
                source_ray_current_raw_message.reset();
                RCLCPP_DEBUG(this->get_logger(),
                             "skip lidar frame #%" PRIu64 " rx=%" PRIu64 ": no undistorted points, raw=%zu, span=%.3f ms, imu=%zu",
                             consumed_lidar_frame, last_synced_lidar_rx_index,
                             Measures.lidar ? Measures.lidar->size() : 0,
                             frame_span_ms, Measures.imu.size());
                return;
            }

            if (source_ray_exporter && source_ray_exporter->enabled() &&
                source_ray_current_raw_message)
            {
                string source_ray_error;
                source_rays_for_frame =
                    source_ray_exporter->prepare(
                        *source_ray_current_raw_message, source_ray_error);
                if (!source_ray_error.empty())
                {
                    RCLCPP_FATAL(
                        this->get_logger(),
                        "Exact source-ray preparation failed: %s",
                        source_ray_error.c_str());
                    source_ray_current_raw_message.reset();
                    rclcpp::shutdown();
                    return;
                }
                if (source_rays_for_frame)
                {
                    if (source_ray_deskew_validation_count <
                        source_ray_exporter->deskewValidationFrames())
                    {
                        PointCloudXYZI validation_cloud = *Measures.lidar;
                        if (p_imu->deskew_en)
                        {
                            std::sort(
                                validation_cloud.points.begin(),
                                validation_cloud.points.end(), time_list);
                        }
                        if (!p_imu->DeskewOrderedCloudWithCachedTrajectory(
                                validation_cloud, state_point) ||
                            validation_cloud.size() != feats_undistort->size())
                        {
                            source_ray_error =
                                "normal deskew shared-helper validation "
                                "could not reproduce the cloud shape";
                        }
                        else
                        {
                            double maximum_error_m = 0.0;
                            for (std::size_t index = 0;
                                 index < validation_cloud.size(); ++index)
                            {
                                const auto &expected =
                                    feats_undistort->points[index];
                                const auto &actual =
                                    validation_cloud.points[index];
                                const double dx =
                                    static_cast<double>(actual.x) - expected.x;
                                const double dy =
                                    static_cast<double>(actual.y) - expected.y;
                                const double dz =
                                    static_cast<double>(actual.z) - expected.z;
                                maximum_error_m = std::max(
                                    maximum_error_m,
                                    std::sqrt(dx * dx + dy * dy + dz * dz));
                            }
                            if (maximum_error_m > 1.0e-6)
                            {
                                std::ostringstream message;
                                message
                                    << "normal deskew shared-helper mismatch "
                                    << std::setprecision(12)
                                    << maximum_error_m << " m";
                                source_ray_error = message.str();
                            }
                            else
                            {
                                RCLCPP_INFO(
                                    this->get_logger(),
                                    "Exact source-ray shared deskew validation "
                                    "passed: pose=%d points=%zu max_error=%.3e m.",
                                    source_rays_for_frame->pose_index,
                                    validation_cloud.size(),
                                    maximum_error_m);
                                ++source_ray_deskew_validation_count;
                            }
                        }
                    }
                    if (source_ray_error.empty() &&
                        (!p_imu->DeskewOrderedCloudWithCachedTrajectory(
                             source_rays_for_frame->hits_input, state_point) ||
                         !p_imu->DeskewOrderedCloudWithCachedTrajectory(
                             source_rays_for_frame->origins_input, state_point)))
                    {
                        source_ray_error =
                            "exact hit/origin deskew failed";
                    }
                    if (!source_ray_error.empty())
                    {
                        RCLCPP_FATAL(
                            this->get_logger(),
                            "Exact source-ray export failed for pose=%d: %s",
                            source_rays_for_frame->pose_index,
                            source_ray_error.c_str());
                        source_ray_current_raw_message.reset();
                        rclcpp::shutdown();
                        return;
                    }
                }
            }
            source_ray_current_raw_message.reset();

            flg_EKF_inited = ((Measures.lidar_beg_time - first_lidar_time) >= INIT_TIME && ikdtree_built) ? true : false;
            /*** Segment the map in lidar FOV ***/
            if (!use_prior_map)
            {
                lasermap_fov_segment();
            }

            /*** downsample the feature points in a scan ***/
            const double active_scan_leaf =
                (use_prior_map && !prior_map_build_done && prior_map_voxel_leaf > 1e-3)
                    ? prior_map_voxel_leaf
                    : filter_size_surf_min;
            downSizeFilterSurf.setLeafSize(active_scan_leaf, active_scan_leaf, active_scan_leaf);
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }

                    if (use_prior_map)
                    {
                        V3D p_C_F, v_C_F, w_C_F;
                        M3D R_C_F;
                        compute_base_link_pose_twist_in_cam_init(p_C_F, R_C_F, v_C_F, w_C_F);
                        maybe_init_base_link_gravity_map(p_C_F, R_C_F);
                        if (!map_world_initialized)
                        {
                            RCLCPP_WARN(this->get_logger(), "Waiting base_link gravity initialization before prior-map ICP.");
                            return;
                        }

                        const bool cached_for_replay = append_prior_icp_source_frame(
                            feats_undistort,
                            Measures.lidar_beg_time,
                            Measures.lidar_end_time,
                            consumed_lidar_frame,
                            last_synced_lidar_rx_index);
                        if (cached_for_replay)
                        {
                            cache_prior_replay_frame(Measures, consumed_lidar_frame, last_synced_lidar_rx_index);
                        }
                        freeze_prior_icp_source_if_ready(Measures.lidar_beg_time);

                        if (prior_icp_source_frozen)
                        {
                            const bool aligned = try_align_prior_map_and_build_tree(pubScanContextCandidates_);
                            publish_prior_map_once(pubPriorMap_);
                            if (!prior_map_build_done)
                            {
                                RCLCPP_WARN(
                                    this->get_logger(),
                                    "Prior-map ICP pending... source_points=%zu target_points=%zu fails=%d",
                                    prior_icp_source_cloud->size(), prior_map_cloud->size(), prior_icp_fail_count);
                                if (!aligned)
                                {
                                    reject_current_prior_icp_source("icp_failed");
                                }
                            }
                        }
                    }
                    else if (!ikdtree_built)
                    {
                        *init_feats_buffer += *feats_down_world;
                        if ((Measures.lidar_beg_time - first_lidar_time) >= INIT_TIME &&
                            init_feats_buffer->points.size() > 5)
                        {
                            RCLCPP_INFO(
                                this->get_logger(),
                                "Initialization complete: Building ikdtree with %zu points from %.2f seconds of data",
                                init_feats_buffer->points.size(), INIT_TIME);
                            ikdtree.Build(init_feats_buffer->points);
                            ikdtree_built = true;
                            init_feats_buffer->clear();
                        }
                    }
                }
                RCLCPP_DEBUG(this->get_logger(),
                             "skip lidar frame #%" PRIu64 " rx=%" PRIu64 ": map initialization frame, undistort=%zu, down=%d, span=%.3f ms, imu=%zu",
                             consumed_lidar_frame, last_synced_lidar_rx_index,
                             feats_undistort->size(), feats_down_size,
                             frame_span_ms, Measures.imu.size());
                return;
            }
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                RCLCPP_WARN(this->get_logger(),
                            "skip lidar frame #%" PRIu64 " rx=%" PRIu64 ": too few downsampled points, undistort=%zu, down=%d, span=%.3f ms, imu=%zu",
                            consumed_lidar_frame, last_synced_lidar_rx_index,
                            feats_undistort->size(), feats_down_size,
                            frame_span_ms, Measures.imu.size());
                return;
            }
            
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            Nearest_Points.clear();
            Nearest_Points.resize(feats_down_size);

            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            const double vel_norm = state_point.vel.norm();
            static bool have_last_localization_health = false;
            static V3D last_health_pos(0.0, 0.0, 0.0);
            static M3D last_health_rot = M3D::Identity();
            static double last_health_time = 0.0;
            static int last_localization_issue_mask = 0;
            static double last_localization_health_warn_time = -1.0;
            static int localization_unhealthy_count = 0;
            static uint64_t last_seen_lidar_timestamp_rollback_events = 0;
            static uint64_t last_seen_imu_timestamp_rollback_events = 0;
            if (localization_health_reset_requested)
            {
                have_last_localization_health = false;
                last_localization_issue_mask = 0;
                last_localization_health_warn_time = -1.0;
                localization_unhealthy_count = 0;
                last_seen_lidar_timestamp_rollback_events =
                    lidar_timestamp_rollback_events.load(std::memory_order_relaxed);
                last_seen_imu_timestamp_rollback_events =
                    imu_timestamp_rollback_events.load(std::memory_order_relaxed);
                localization_health_reset_requested = false;
            }
            double step_trans = 0.0;
            double step_rot = 0.0;
            double step_dt = 0.0;
            bool pose_jump = false;
            if (have_last_localization_health)
            {
                step_dt = Measures.lidar_beg_time - last_health_time;
                step_trans = (state_point.pos - last_health_pos).norm();
                const M3D delta_rot = last_health_rot.transpose() * state_point.rot.toRotationMatrix();
                const double cos_angle =
                    std::max(-1.0, std::min(1.0, (delta_rot.trace() - 1.0) * 0.5));
                step_rot = std::acos(cos_angle);
                if (step_dt > 1e-3)
                {
                    const double trans_limit = std::max(0.75, 4.0 * step_dt);
                    const double rot_limit = std::max(30.0 * M_PI / 180.0, 180.0 * M_PI / 180.0 * step_dt);
                    pose_jump = step_trans > trans_limit || step_rot > rot_limit;
                }
            }

            constexpr double max_velocity_norm_for_health = 5.0;
            const bool high_residual = effct_feat_num > 0 && res_mean_last > 0.25;
            const bool high_velocity = vel_norm > max_velocity_norm_for_health;
            const bool localization_tracking_active =
                use_prior_map && prior_map_loaded && prior_map_aligned && prior_map_build_done;
            const bool no_effective_points =
                localization_tracking_active && current_measurement_no_effective_points;
            const bool insufficient_effective_points =
                localization_tracking_active &&
                (no_effective_points || effct_feat_num < localization_min_effective_points);
            const uint64_t lidar_rollback_events =
                lidar_timestamp_rollback_events.load(std::memory_order_relaxed);
            const uint64_t imu_rollback_events =
                imu_timestamp_rollback_events.load(std::memory_order_relaxed);
            const bool timestamp_rollback =
                localization_tracking_active &&
                localization_restart_on_timestamp_rollback &&
                (lidar_rollback_events != last_seen_lidar_timestamp_rollback_events ||
                 imu_rollback_events != last_seen_imu_timestamp_rollback_events);
            last_seen_lidar_timestamp_rollback_events = lidar_rollback_events;
            last_seen_imu_timestamp_rollback_events = imu_rollback_events;

            const bool relocalize_unhealthy =
                localization_tracking_active && (insufficient_effective_points || timestamp_rollback);
            const bool localization_output_issue =
                localization_tracking_active &&
                (pose_jump || insufficient_effective_points || high_residual || high_velocity || timestamp_rollback);
            const int localization_issue_mask =
                (pose_jump ? 1 : 0) |
                (insufficient_effective_points ? 2 : 0) |
                (high_residual ? 4 : 0) |
                (high_velocity ? 8 : 0) |
                (timestamp_rollback ? 16 : 0) |
                (no_effective_points ? 32 : 0);
            if (localization_tracking_active && localization_auto_relocalize_enable)
            {
                localization_unhealthy_count =
                    relocalize_unhealthy ? localization_unhealthy_count + 1 : 0;
            }
            else
            {
                localization_unhealthy_count = 0;
            }

            if (pose_jump || no_effective_points || insufficient_effective_points || high_residual || high_velocity || timestamp_rollback)
            {
                const bool issue_changed = localization_issue_mask != last_localization_issue_mask;
                const bool throttle_elapsed =
                    last_localization_health_warn_time < 0.0 ||
                    Measures.lidar_beg_time - last_localization_health_warn_time >= 1.0;
                if (issue_changed || throttle_elapsed)
                {
                    RCLCPP_WARN(
                        this->get_logger(),
                        "Localization health warning: frame=%" PRIu64 " rx=%" PRIu64
                        " lidar_t=%.3f dt=%.3f step=%.3fm rot_step=%.2fdeg"
                        " pose=[%.3f %.3f %.3f] rpy_deg=[%.2f %.2f %.2f]"
                        " vel_norm=%.3f max_vel=%.3f down=%d effective=%d min_effective=%d res_mean=%.4f map=%d"
                        " unhealthy_count=%d/%d"
                        " flags=[pose_jump=%s no_effective=%s insufficient_effective=%s high_residual=%s high_velocity=%s timestamp_rollback=%s]",
                        consumed_lidar_frame, last_synced_lidar_rx_index,
                        Measures.lidar_beg_time - first_lidar_time, step_dt,
                        step_trans, rad2deg(step_rot),
                        state_point.pos(0), state_point.pos(1), state_point.pos(2),
                        wrap_angle_deg(euler_cur(0)),
                        wrap_angle_deg(euler_cur(1)),
                        wrap_angle_deg(euler_cur(2)),
                        vel_norm, max_velocity_norm_for_health,
                        feats_down_size, effct_feat_num, localization_min_effective_points,
                        res_mean_last, ikdtree.validnum(),
                        localization_unhealthy_count,
                        localization_unhealthy_consecutive_frames,
                        pose_jump ? "true" : "false",
                        no_effective_points ? "true" : "false",
                        insufficient_effective_points ? "true" : "false",
                        high_residual ? "true" : "false",
                        high_velocity ? "true" : "false",
                        timestamp_rollback ? "true" : "false");
                    last_localization_issue_mask = localization_issue_mask;
                    last_localization_health_warn_time = Measures.lidar_beg_time;
                }
            }
            if (localization_auto_relocalize_enable &&
                localization_tracking_active &&
                relocalize_unhealthy &&
                localization_unhealthy_count >= localization_unhealthy_consecutive_frames)
            {
                const char *restart_reason =
                    timestamp_rollback ? "timestamp_rollback" : "insufficient_effective";
                restart_prior_relocalization_from_health(
                    restart_reason,
                    localization_unhealthy_count,
                    insufficient_effective_points,
                    timestamp_rollback,
                    feats_down_size,
                    effct_feat_num,
                    Measures.lidar_beg_time - first_lidar_time,
                    consumed_lidar_frame,
                    last_synced_lidar_rx_index);
                return;
            }
            last_health_pos = state_point.pos;
            last_health_rot = state_point.rot.toRotationMatrix();
            last_health_time = Measures.lidar_beg_time;
            have_last_localization_health = true;

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            const bool publish_trusted_localization =
                !use_prior_map ||
                (prior_map_aligned && prior_map_build_done && localization_output_trusted && !localization_output_issue);
            if (publish_trusted_localization)
            {
                publish_odometry(pubOdomAftMapped_, tf_broadcaster_);
                publish_pose(pubPose_);
            }
            else
            {
                static int last_suppressed_issue_mask = -1;
                static double last_suppressed_output_report_time = -1.0;
                const bool suppress_issue_changed = localization_issue_mask != last_suppressed_issue_mask;
                const bool suppress_throttle_elapsed =
                    last_suppressed_output_report_time < 0.0 ||
                    Measures.lidar_beg_time - last_suppressed_output_report_time >= 1.0;
                if (suppress_issue_changed || suppress_throttle_elapsed)
                {
                    RCLCPP_WARN(
                        this->get_logger(),
                        "Suppress localization output while prior-map localization is untrusted/unhealthy: aligned=%s build_done=%s output_trusted=%s flags=[pose_jump=%s no_effective=%s insufficient_effective=%s high_residual=%s high_velocity=%s timestamp_rollback=%s]",
                        prior_map_aligned ? "true" : "false",
                        prior_map_build_done ? "true" : "false",
                        localization_output_trusted ? "true" : "false",
                        pose_jump ? "true" : "false",
                        no_effective_points ? "true" : "false",
                        insufficient_effective_points ? "true" : "false",
                        high_residual ? "true" : "false",
                        high_velocity ? "true" : "false",
                        timestamp_rollback ? "true" : "false");
                    last_suppressed_issue_mask = localization_issue_mask;
                    last_suppressed_output_report_time = Measures.lidar_beg_time;
                }
            }

            /*** add the feature points to map kdtree ***/
            if (!use_prior_map)
            {
                map_incremental();
            }
            else
            {
                add_point_size = 0;
                kdtree_incremental_time = 0.0;
            }
            maybe_add_scan_context_keyframe(Measures.lidar_beg_time, feats_undistort);
            if (source_rays_for_frame)
            {
                // The ray geometry was deskewed with the cached prediction
                // trajectory above.  Write only after the scan-to-map update
                // so physical gravity matches the state used by the normal
                // keyframe/gravity sidecar path.
                const M3D R_base_input =
                    BaseLink_R_wrt_LidarIMU *
                    state_point.offset_R_L_I.toRotationMatrix();
                const V3D t_base_input =
                    BaseLink_R_wrt_LidarIMU *
                        state_point.offset_T_L_I +
                    BaseLink_T_wrt_LidarIMU;
                V3D gravity_up_base = Zero3d;
                if (!current_scan_context_gravity_up(gravity_up_base))
                {
                    RCLCPP_FATAL(
                        this->get_logger(),
                        "Exact source-ray export failed for pose=%d: "
                        "the synchronized Scan Context gravity is invalid.",
                        source_rays_for_frame->pose_index);
                    rclcpp::shutdown();
                    return;
                }

                string source_ray_error;
                if (!source_ray_exporter->writeDeskewed(
                        *source_rays_for_frame,
                        R_base_input,
                        t_base_input,
                        gravity_up_base,
                        source_ray_error))
                {
                    if (source_ray_error.empty())
                        source_ray_error =
                            "exact source-ray shard write failed";
                    RCLCPP_FATAL(
                        this->get_logger(),
                        "Exact source-ray export failed for pose=%d: %s",
                        source_rays_for_frame->pose_index,
                        source_ray_error.c_str());
                    rclcpp::shutdown();
                    return;
                }
                RCLCPP_INFO(
                    this->get_logger(),
                    "Exact source rays exported: pose=%d eligible=%zu "
                    "stamp=%" PRId64 ".",
                    source_rays_for_frame->pose_index,
                    source_rays_for_frame->eligible_count,
                    source_rays_for_frame->frame_reference_timestamp_ns);
            }
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en && publish_trusted_localization) publish_path(pubPath_);
            if (publish_trusted_localization)
            {
                if (scan_pub_en) publish_frame_world(pubLaserCloudFull_);
                if (scan_body_pub_en)
                    publish_frame_body(pubLaserCloudFull_body_, pubScanContextGravity_);
                if (effect_pub_en) publish_effect_world(pubLaserCloudEffect_);
            }
            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                if (time_log_counter < MAXN)
                {
                    T1[time_log_counter] = Measures.lidar_beg_time;
                    s_plot[time_log_counter] = t5 - t0;
                    s_plot2[time_log_counter] = feats_undistort->points.size();
                    s_plot3[time_log_counter] = kdtree_incremental_time;
                    s_plot4[time_log_counter] = kdtree_search_time;
                    s_plot5[time_log_counter] = kdtree_delete_counter;
                    s_plot6[time_log_counter] = kdtree_delete_time;
                    s_plot7[time_log_counter] = kdtree_size_st;
                    s_plot8[time_log_counter] = kdtree_size_end;
                    s_plot9[time_log_counter] = aver_time_consu;
                    s_plot10[time_log_counter] = add_point_size;
                    time_log_counter ++;
                }
                else
                {
                    static bool runtime_log_capacity_warned = false;
                    if (!runtime_log_capacity_warned)
                    {
                        RCLCPP_WARN(
                            this->get_logger(),
                            "Runtime position log reached capacity (%d samples); CSV arrays will stop growing.",
                            MAXN);
                        runtime_log_capacity_warned = true;
                    }
                }
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                if (fp)
                {
                    dump_lio_state_to_log(fp);
                }
            }
        }
    }

    void map_publish_callback()
    {
        if (use_prior_map && scan_context_candidate_cloud_active)
        {
            if (omp_get_wtime() <= scan_context_candidate_cloud_expire_time)
            {
                publish_scan_context_candidate_cloud(pubScanContextCandidates_, scan_context_last_icp_candidates);
            }
            else
            {
                publish_scan_context_candidate_cloud(pubScanContextCandidates_, {});
                scan_context_candidate_cloud_active = false;
            }
        }
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr pubScanContextGravity_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pubPose_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPriorMap_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubScanContextCandidates_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;
    rclcpp::CallbackGroup::SharedPtr lidar_callback_group_;
    rclcpp::CallbackGroup::SharedPtr imu_callback_group_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_pub_timer_;

    bool effect_pub_en = false;
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[23] = {0.001};

    FILE *fp = nullptr;
    ofstream fout_pre, fout_out, fout_dbg;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

    auto node = std::make_shared<LaserMappingNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();
    executor.remove_node(node);

    if (rclcpp::ok())
        rclcpp::shutdown();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
     * 2. pcd save will largely influence the real-time performences */
    save_waiting_pcd_on_exit();

    if (runtime_pos_log)
    {
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        FILE *fp2 = fopen(log_dir.c_str(),"w");
        if (!fp2)
        {
            std::cerr << "Failed to open runtime time log: " << log_dir << std::endl;
        }
        else
        {
            fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
            for (int i = 0;i<time_log_counter; i++){
                fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            }
            fclose(fp2);
        }
    }

    return 0;
}
