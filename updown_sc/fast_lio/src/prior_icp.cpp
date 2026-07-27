#include "prior_icp.hpp"

#include <omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>

#include <algorithm>
#include <cmath>
#include <exception>

namespace fast_lio::prior_icp
{

Eigen::Matrix4f makeSeedTransform(double x, double y, double z, double yaw_rad)
{
    const double c = std::cos(yaw_rad), s = std::sin(yaw_rad);
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T(0, 0) = static_cast<float>(c);
    T(0, 1) = static_cast<float>(-s);
    T(1, 0) = static_cast<float>(s);
    T(1, 1) = static_cast<float>(c);
    T(0, 3) = static_cast<float>(x);
    T(1, 3) = static_cast<float>(y);
    T(2, 3) = static_cast<float>(z);
    return T;
}

Eigen::Matrix4f makeSeedTransform(double x, double y, double z,
                                  const Eigen::Matrix3d &rotation)
{
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    if (!rotation.allFinite())
        return T;
    T.block<3, 3>(0, 0) = rotation.cast<float>();
    T(0, 3) = static_cast<float>(x);
    T(1, 3) = static_cast<float>(y);
    T(2, 3) = static_cast<float>(z);
    return T;
}

namespace
{

double finiteOr(double value, double fallback)
{
    return std::isfinite(value) ? value : fallback;
}

double computeOverlapRatio(
    const PointCloudXYZI &aligned_source,
    pcl::KdTreeFLANN<PointType> &target_kdtree,
    double max_corr_dist)
{
    if (aligned_source.empty() || max_corr_dist <= 0.0)
        return 0.0;

    const float max_corr_dist_sq = static_cast<float>(max_corr_dist * max_corr_dist);
    int matched_points = 0;
    int finite_points = 0;
    std::vector<int> indices(1);
    std::vector<float> sq_distances(1);
    for (const auto &point : aligned_source.points)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            continue;
        ++finite_points;
        if (target_kdtree.nearestKSearch(point, 1, indices, sq_distances) > 0 &&
            sq_distances.front() <= max_corr_dist_sq)
        {
            ++matched_points;
        }
    }

    if (finite_points == 0)
        return 0.0;
    return static_cast<double>(matched_points) / static_cast<double>(finite_points);
}

}  // namespace

std::vector<Result> runStage(
    const Config &config,
    const PointCloudXYZI::Ptr &source,
    const PointCloudXYZI::Ptr &target,
    const std::vector<Eigen::Matrix4f> &seeds,
    const std::vector<int> &seed_indices,
    int &converged_count,
    int &valid_count)
{
    std::vector<Result> results;
    converged_count = 0;
    valid_count = 0;
    if (!source || !target || source->empty() || target->empty() || seeds.empty() || seed_indices.empty())
        return results;

    const int max_iterations = std::max(1, config.max_iterations);
    const double max_corr_dist = std::max(1e-3, finiteOr(config.max_corr_dist, 1.0));
    const double min_overlap_ratio = std::clamp(finiteOr(config.min_overlap_ratio, 0.5), 0.0, 1.0);

    pcl::KdTreeFLANN<PointType> target_kdtree;
    target_kdtree.setInputCloud(target);

#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for schedule(dynamic)
#endif
    for (std::size_t idx = 0; idx < seed_indices.size(); ++idx)
    {
        const int seed_index = seed_indices[idx];
        if (seed_index < 0 || seed_index >= static_cast<int>(seeds.size()))
            continue;
        if (!seeds[seed_index].allFinite())
            continue;

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaximumIterations(max_iterations);
        icp.setMaxCorrespondenceDistance(max_corr_dist);
        icp.setTransformationEpsilon(1e-7);
        icp.setEuclideanFitnessEpsilon(1e-7);
        icp.setRANSACOutlierRejectionThreshold(max_corr_dist * 0.5);
        icp.setInputSource(source);
        icp.setInputTarget(target);

        PointCloudXYZI aligned;
        try
        {
            icp.align(aligned, seeds[seed_index]);
        }
        catch (const std::exception &)
        {
            continue;
        }
        if (!icp.hasConverged() || aligned.empty())
            continue;

        const double fitness = icp.getFitnessScore();
        const Eigen::Matrix4f final_transform = icp.getFinalTransformation();
        if (!std::isfinite(fitness) || !final_transform.allFinite())
            continue;
        double overlap_ratio = 0.0;
#ifdef MP_EN
#pragma omp critical(prior_icp_overlap)
#endif
        {
            ++converged_count;
            overlap_ratio = computeOverlapRatio(aligned, target_kdtree, max_corr_dist);
        }
        if (overlap_ratio < min_overlap_ratio)
            continue;

        Result result;
        result.seed_index = seed_index;
        result.fitness = fitness;
        result.overlap = overlap_ratio;
        result.transform = final_transform;

#ifdef MP_EN
#pragma omp critical(prior_icp_results)
#endif
        {
            ++valid_count;
            results.push_back(result);
        }
    }

    std::sort(results.begin(), results.end(), [](const Result &lhs, const Result &rhs) {
        if (lhs.fitness != rhs.fitness)
            return lhs.fitness < rhs.fitness;
        if (lhs.overlap != rhs.overlap)
            return lhs.overlap > rhs.overlap;
        return lhs.seed_index < rhs.seed_index;
    });
    return results;
}

}  // namespace fast_lio::prior_icp
