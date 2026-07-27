#ifndef FAST_LIO_PCD_SAVE_UTILS_HPP
#define FAST_LIO_PCD_SAVE_UTILS_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstddef>
#include <string>

namespace fast_lio::pcd_save
{

using PointType = pcl::PointXYZINormal;
using PointCloudXYZI = pcl::PointCloud<PointType>;

struct SaveSummary
{
    std::size_t input_points = 0;
    std::size_t output_points = 0;
    bool downsampled = false;
    bool success = false;
    std::string error;
};

PointCloudXYZI::Ptr downsampleForSave(const PointCloudXYZI::Ptr &cloud, double voxel_leaf);

SaveSummary writeBinary(
    const std::string &path,
    const PointCloudXYZI::Ptr &cloud,
    double voxel_leaf);

}  // namespace fast_lio::pcd_save

#endif  // FAST_LIO_PCD_SAVE_UTILS_HPP
