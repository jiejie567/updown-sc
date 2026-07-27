#ifndef FAST_LIO_PRIOR_ICP_HPP
#define FAST_LIO_PRIOR_ICP_HPP

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <limits>
#include <vector>

namespace fast_lio::prior_icp
{

using PointType = pcl::PointXYZINormal;
using PointCloudXYZI = pcl::PointCloud<PointType>;

struct Config
{
    int max_iterations = 60;
    double max_corr_dist = 3.0;
    double min_overlap_ratio = 0.5;
};

struct Result
{
    int seed_index = -1;
    double fitness = std::numeric_limits<double>::infinity();
    double overlap = 0.0;
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
};

Eigen::Matrix4f makeSeedTransform(double x, double y, double z, double yaw_rad);
Eigen::Matrix4f makeSeedTransform(double x, double y, double z,
                                  const Eigen::Matrix3d &rotation);

std::vector<Result> runStage(
    const Config &config,
    const PointCloudXYZI::Ptr &source,
    const PointCloudXYZI::Ptr &target,
    const std::vector<Eigen::Matrix4f> &seeds,
    const std::vector<int> &seed_indices,
    int &converged_count,
    int &valid_count);

}  // namespace fast_lio::prior_icp

#endif  // FAST_LIO_PRIOR_ICP_HPP
