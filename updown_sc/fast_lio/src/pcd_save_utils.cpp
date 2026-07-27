#include "pcd_save_utils.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <exception>
#include <cmath>
#include <filesystem>
#include <string>
#include <system_error>

namespace fast_lio::pcd_save
{

PointCloudXYZI::Ptr downsampleForSave(const PointCloudXYZI::Ptr &cloud, double voxel_leaf)
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

    if (finite_cloud->empty() || !std::isfinite(voxel_leaf) || voxel_leaf <= 1e-3)
        return finite_cloud;

    PointCloudXYZI::Ptr filtered(new PointCloudXYZI());
    pcl::VoxelGrid<PointType> voxel_filter;
    voxel_filter.setLeafSize(
        static_cast<float>(voxel_leaf),
        static_cast<float>(voxel_leaf),
        static_cast<float>(voxel_leaf));
    voxel_filter.setInputCloud(finite_cloud);
    voxel_filter.filter(*filtered);
    return filtered;
}

SaveSummary writeBinary(
    const std::string &path,
    const PointCloudXYZI::Ptr &cloud,
    double voxel_leaf)
{
    SaveSummary summary;
    summary.input_points = cloud ? cloud->size() : 0;
    if (path.empty())
    {
        summary.success = false;
        summary.error = "empty PCD output path";
        return summary;
    }

    const std::filesystem::path output_path(path);
    if (!output_path.parent_path().empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(output_path.parent_path(), ec);
        if (ec)
        {
            summary.success = false;
            summary.error = "failed to create PCD output directory: " + ec.message();
            return summary;
        }
    }

    PointCloudXYZI::Ptr cloud_to_save = downsampleForSave(cloud, voxel_leaf);
    if (!cloud_to_save)
        cloud_to_save.reset(new PointCloudXYZI());

    summary.output_points = cloud_to_save->size();
    summary.downsampled = voxel_leaf > 1e-3 && cloud && !cloud->empty() &&
                          cloud_to_save.get() != cloud.get();

    const std::filesystem::path tmp_path = output_path.string() + ".tmp";
    auto cleanup_tmp = [&tmp_path]() {
        std::error_code cleanup_ec;
        std::filesystem::remove(tmp_path, cleanup_ec);
    };
    try
    {
        pcl::PCDWriter pcd_writer;
        const int ret = pcd_writer.writeBinary(tmp_path.string(), *cloud_to_save);
        if (ret != 0)
        {
            summary.success = false;
            summary.error = "PCDWriter::writeBinary returned " + std::to_string(ret);
            cleanup_tmp();
            return summary;
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
            summary.success = false;
            summary.error = "failed to move temporary PCD into place: " + rename_ec.message();
            cleanup_tmp();
            return summary;
        }
        summary.success = true;
    }
    catch (const std::exception &e)
    {
        summary.success = false;
        summary.error = e.what();
    }
    if (!summary.success)
        cleanup_tmp();
    return summary;
}

}  // namespace fast_lio::pcd_save
