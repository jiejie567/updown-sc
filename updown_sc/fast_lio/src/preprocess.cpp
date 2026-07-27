#include "preprocess.h"

#include <pcl/common/common.h>
#include <algorithm>
#include <limits>

#define RETURN0 0x00
#define RETURN0AND1 0x10

namespace
{
double pointTimeToOffsetMs(double point_time, double header_time, double relative_time_scale)
{
  if (point_time > 1.0e17)       // nanoseconds since epoch
    return (point_time * 1.0e-9 - header_time) * 1000.0;
  if (point_time > 1.0e14)       // microseconds since epoch
    return (point_time * 1.0e-6 - header_time) * 1000.0;
  if (point_time > 1.0e11)       // milliseconds since epoch
    return (point_time * 1.0e-3 - header_time) * 1000.0;
  if (point_time > 1.0e8)        // seconds since epoch
    return (point_time - header_time) * 1000.0;
  return point_time * relative_time_scale;
}
}  // namespace

Preprocess::Preprocess()
    : feature_enabled(0), lidar_type(AVIA), blind(0.01), det_range(100.0), max_height(5.0), point_filter_num(1),
      tag_filter_mode(TAG_FILTER_LOW_CONFIDENCE), blind_filter_shape(BLIND_FILTER_SPHERE),
      blind_z_min(-std::numeric_limits<double>::infinity()), blind_z_max(std::numeric_limits<double>::infinity())
{
  inf_bound = 10;
  N_SCANS = 6;
  SCAN_RATE = 10;
  group_size = 8;
  disA = 0.01;
  disA = 0.1;  // B?
  p2l_ratio = 225;
  limit_maxmid = 6.25;
  limit_midmin = 6.25;
  limit_maxmin = 3.24;
  jump_up_limit = 170.0;
  jump_down_limit = 8.0;
  cos160 = 160.0;
  edgea = 2;
  edgeb = 0.1;
  smallp_intersect = 172.5;
  smallp_ratio = 1.2;
  given_offset_time = false;

  jump_up_limit = cos(jump_up_limit / 180 * M_PI);
  jump_down_limit = cos(jump_down_limit / 180 * M_PI);
  cos160 = cos(cos160 / 180 * M_PI);
  smallp_intersect = cos(smallp_intersect / 180 * M_PI);
}

Preprocess::~Preprocess()
{
}

void Preprocess::set(bool feat_en, int lid_type, double bld, double max_range, double max_z, int pfilt_num)
{
  feature_enabled = feat_en;
  lidar_type = lid_type;
  blind = bld;
  det_range = max_range;
  max_height = max_z;
  point_filter_num = pfilt_num;
}

void Preprocess::set_tag_filter_mode(int mode)
{
  if (mode < TAG_FILTER_OFF || mode > TAG_FILTER_STRICT)
    mode = TAG_FILTER_LOW_CONFIDENCE;
  tag_filter_mode = mode;
}

void Preprocess::set_blind_filter(int shape, double z_min, double z_max)
{
  if (shape < BLIND_FILTER_SPHERE || shape > BLIND_FILTER_CYLINDER)
    shape = BLIND_FILTER_SPHERE;
  if (!std::isfinite(z_min))
    z_min = -std::numeric_limits<double>::infinity();
  if (!std::isfinite(z_max))
    z_max = std::numeric_limits<double>::infinity();
  if (z_min > z_max)
    std::swap(z_min, z_max);
  blind_filter_shape = shape;
  blind_z_min = z_min;
  blind_z_max = z_max;
}

bool Preprocess::valid_range_sq(double range_sq) const
{
  if (!std::isfinite(range_sq))
    return false;
  const double min_range_sq = blind * blind;
  const double max_range_sq = det_range * det_range;
  return range_sq > min_range_sq && range_sq < max_range_sq;
}

bool Preprocess::valid_range_xyz(double x, double y, double z) const
{
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
    return false;

  const double xy_sq = x * x + y * y;
  const double range_sq = xy_sq + z * z;
  const double max_range_sq = det_range * det_range;
  if (range_sq >= max_range_sq)
    return false;

  const double blind_sq = blind * blind;
  if (blind_filter_shape == BLIND_FILTER_CYLINDER)
  {
    const bool inside_xy = xy_sq <= blind_sq;
    const bool inside_z = z >= blind_z_min && z <= blind_z_max;
    return !(inside_xy && inside_z);
  }

  return range_sq > blind_sq;
}

bool Preprocess::valid_livox_tag(uint8_t tag) const
{
  switch (tag_filter_mode)
  {
    case TAG_FILTER_OFF:
      return true;
    case TAG_FILTER_OTHER:
      return (tag & 0x30) == 0x00 || (tag & 0x30) == 0x10;
    case TAG_FILTER_STRICT:
      return (tag & 0x3f) == 0;
    case TAG_FILTER_LOW_CONFIDENCE:
    default:
      return (tag & 0x30) != 0x20 && (tag & 0x0c) != 0x08 && (tag & 0x03) != 0x02;
  }
}

void Preprocess::process(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg, PointCloudXYZI::Ptr& pcl_out)
{
  avia_handler(msg);
  *pcl_out = pl_surf;
}

void Preprocess::process(const sensor_msgs::msg::PointCloud2::UniquePtr &msg, PointCloudXYZI::Ptr& pcl_out)
{
  switch (time_unit)
  {
    case SEC:
      time_unit_scale = 1.e3f;
      break;
    case MS:
      time_unit_scale = 1.f;
      break;
    case US:
      time_unit_scale = 1.e-3f;
      break;
    case NS:
      time_unit_scale = 1.e-6f;
      break;
    default:
      time_unit_scale = 1.f;
      break;
  }

  switch (lidar_type)
  {
    case OUST64:
      oust64_handler(msg);
      break;

    case VELO16:
      velodyne_handler(msg);
      break;

    case MID360:
      mid360_handler(msg);
      break;

    case ROBOSENSE:
    case RSAIRY:
      robosense_airy_handler(msg);
      break;

    case RSM1:
      robosense_m1_handler(msg);
      break;

    case GAZEBO_XYZI:
      default_handler(msg);
      break;

    default:
      default_handler(msg);
      break;
  }
  *pcl_out = pl_surf;
}

void Preprocess::avia_handler(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  double t1 = omp_get_wtime();
  int plsize = msg->point_num;
  // cout<<"plsie: "<<plsize<<endl;

  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  pl_full.resize(plsize);

  for (int i = 0; i < N_SCANS; i++)
  {
    pl_buff[i].clear();
    pl_buff[i].reserve(plsize);
  }
  uint valid_num = 0;

  if (feature_enabled)
  {
    for (uint i = 1; i < plsize; i++)
    {
      if ((msg->points[i].line < N_SCANS) && valid_livox_tag(msg->points[i].tag))
      {
        pl_full[i].x = msg->points[i].x;
        pl_full[i].y = msg->points[i].y;
        pl_full[i].z = msg->points[i].z;
        pl_full[i].intensity = msg->points[i].reflectivity;
        pl_full[i].curvature =
            msg->points[i].offset_time / float(1000000);  // use curvature as time of each laser points

        bool is_new = false;
        if ((abs(pl_full[i].x - pl_full[i - 1].x) > 1e-7) || (abs(pl_full[i].y - pl_full[i - 1].y) > 1e-7) ||
            (abs(pl_full[i].z - pl_full[i - 1].z) > 1e-7))
        {
          pl_buff[msg->points[i].line].push_back(pl_full[i]);
        }
      }
    }
    static int count = 0;
    static double time = 0.0;
    count++;
    double t0 = omp_get_wtime();
    for (int j = 0; j < N_SCANS; j++)
    {
      if (pl_buff[j].size() <= 5)
        continue;
      pcl::PointCloud<PointType>& pl = pl_buff[j];
      plsize = pl.size();
      vector<orgtype>& types = typess[j];
      types.clear();
      types.resize(plsize);
      plsize--;
      for (uint i = 0; i < plsize; i++)
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = sqrt(vx * vx + vy * vy + vz * vz);
      }
      types[plsize].range = sqrt(pl[plsize].x * pl[plsize].x + pl[plsize].y * pl[plsize].y);
      give_feature(pl, types);
      // pl_surf += pl;
    }
    time += omp_get_wtime() - t0;
    printf("Feature extraction time: %lf \n", time / count);
  }
  else
  {
    for (uint i = 1; i < plsize; i++)
    {
      if ((msg->points[i].line < N_SCANS) && valid_livox_tag(msg->points[i].tag))
      {
        valid_num++;
        if (valid_num % point_filter_num == 0)
        {
          pl_full[i].x = msg->points[i].x;
          pl_full[i].y = msg->points[i].y;
          pl_full[i].z = msg->points[i].z;
          pl_full[i].intensity = msg->points[i].reflectivity;
          pl_full[i].curvature = msg->points[i].offset_time /
                                 float(1000000);  // use curvature as time of each laser points, curvature unit: ms

          if (((abs(pl_full[i].x - pl_full[i - 1].x) > 1e-7)
              || (abs(pl_full[i].y - pl_full[i - 1].y) > 1e-7)
              || (abs(pl_full[i].z - pl_full[i - 1].z) > 1e-7))
              && valid_range_xyz(pl_full[i].x, pl_full[i].y, pl_full[i].z))
          {
            pl_surf.push_back(pl_full[i]);
          }
        }
      }
    }
  }
}

void Preprocess::oust64_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();

  const auto process_cloud = [this](const auto &pl_orig, const auto &point_time)
  {
    const int plsize = static_cast<int>(pl_orig.size());
    pl_corn.reserve(plsize);
    pl_surf.reserve(plsize);
    if (feature_enabled)
    {
      for (int i = 0; i < N_SCANS; i++)
      {
        pl_buff[i].clear();
        pl_buff[i].reserve(plsize);
      }

      for (int i = 0; i < plsize; i++)
      {
        const auto &source = pl_orig.points[i];
        if (!valid_range_xyz(source.x, source.y, source.z))
          continue;
        PointType added_pt;
        added_pt.x = source.x;
        added_pt.y = source.y;
        added_pt.z = source.z;
        added_pt.intensity = source.intensity;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.curvature = point_time(source) * time_unit_scale;
        if (source.ring < N_SCANS)
          pl_buff[source.ring].push_back(added_pt);
      }

      for (int j = 0; j < N_SCANS; j++)
      {
        PointCloudXYZI& pl = pl_buff[j];
        const int linesize = static_cast<int>(pl.size());
        if (linesize < 2)
          continue;
        vector<orgtype>& types = typess[j];
        types.clear();
        types.resize(linesize);
        for (int i = 0; i + 1 < linesize; i++)
        {
          types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
          vx = pl[i].x - pl[i + 1].x;
          vy = pl[i].y - pl[i + 1].y;
          vz = pl[i].z - pl[i + 1].z;
          types[i].dista = vx * vx + vy * vy + vz * vz;
        }
        types[linesize - 1].range =
            sqrt(pl[linesize - 1].x * pl[linesize - 1].x +
                 pl[linesize - 1].y * pl[linesize - 1].y);
        give_feature(pl, types);
      }
    }
    else
    {
      for (int i = 0; i < plsize; i++)
      {
        if (i % point_filter_num != 0)
          continue;
        const auto &source = pl_orig.points[i];
        if (!valid_range_xyz(source.x, source.y, source.z))
          continue;

        PointType added_pt;
        added_pt.x = source.x;
        added_pt.y = source.y;
        added_pt.z = source.z;
        added_pt.intensity = source.intensity;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.curvature = point_time(source) * time_unit_scale;
        pl_surf.points.push_back(added_pt);
      }
    }
  };

  const bool uses_time_field = std::any_of(
      msg->fields.begin(), msg->fields.end(),
      [](const sensor_msgs::msg::PointField &field) {
        return field.name == "time";
      });
  if (uses_time_field)
  {
    // The LiDAR/IMU-only Newer College ROS 2 bag uses a compact 22-byte point
    // layout.  Converting all ~100k points to an aligned PCL structure before
    // applying point_filter_num is unnecessarily expensive and can make the
    // ROS subscription drop scans.  In the normal (feature-disabled) FAST-LIO
    // path, decode only the points that will actually be retained.
    if (!feature_enabled && !msg->is_bigendian)
    {
      const auto find_field = [&msg](const char *name)
          -> const sensor_msgs::msg::PointField *
      {
        const auto it = std::find_if(
            msg->fields.begin(), msg->fields.end(),
            [name](const sensor_msgs::msg::PointField &field) {
              return field.name == name;
            });
        return it == msg->fields.end() ? nullptr : &*it;
      };
      const auto *x_field = find_field("x");
      const auto *y_field = find_field("y");
      const auto *z_field = find_field("z");
      const auto *intensity_field = find_field("intensity");
      const auto *time_field = find_field("time");
      const bool supported_layout =
          x_field && y_field && z_field && intensity_field && time_field &&
          x_field->datatype == sensor_msgs::msg::PointField::FLOAT32 &&
          y_field->datatype == sensor_msgs::msg::PointField::FLOAT32 &&
          z_field->datatype == sensor_msgs::msg::PointField::FLOAT32 &&
          intensity_field->datatype == sensor_msgs::msg::PointField::FLOAT32 &&
          time_field->datatype == sensor_msgs::msg::PointField::UINT32 &&
          x_field->offset + sizeof(float) <= msg->point_step &&
          y_field->offset + sizeof(float) <= msg->point_step &&
          z_field->offset + sizeof(float) <= msg->point_step &&
          intensity_field->offset + sizeof(float) <= msg->point_step &&
          time_field->offset + sizeof(uint32_t) <= msg->point_step;
      if (supported_layout && msg->point_step > 0 && msg->width > 0 &&
          msg->height > 0 &&
          msg->row_step >= msg->width * msg->point_step &&
          msg->data.size() >=
              static_cast<std::size_t>(msg->height) * msg->row_step)
      {
        const std::size_t keep_stride =
            static_cast<std::size_t>(std::max(1, point_filter_num));
        const std::size_t point_count =
            static_cast<std::size_t>(msg->width) * msg->height;
        pl_surf.reserve(point_count / keep_stride + 1);
        std::size_t linear_index = 0;
        for (std::size_t row = 0; row < msg->height; ++row)
        {
          for (std::size_t col = 0; col < msg->width;
               ++col, ++linear_index)
          {
            if (linear_index % keep_stride != 0)
              continue;
            const std::size_t offset =
                row * msg->row_step + col * msg->point_step;
            PointType added_pt;
            std::memcpy(
                &added_pt.x, msg->data.data() + offset + x_field->offset,
                sizeof(float));
            std::memcpy(
                &added_pt.y, msg->data.data() + offset + y_field->offset,
                sizeof(float));
            std::memcpy(
                &added_pt.z, msg->data.data() + offset + z_field->offset,
                sizeof(float));
            std::memcpy(
                &added_pt.intensity,
                msg->data.data() + offset + intensity_field->offset,
                sizeof(float));
            uint32_t point_time = 0;
            std::memcpy(
                &point_time, msg->data.data() + offset + time_field->offset,
                sizeof(uint32_t));
            if (!valid_range_xyz(added_pt.x, added_pt.y, added_pt.z))
              continue;
            added_pt.normal_x = 0;
            added_pt.normal_y = 0;
            added_pt.normal_z = 0;
            added_pt.curvature =
                static_cast<float>(point_time) * time_unit_scale;
            pl_surf.push_back(added_pt);
          }
        }
        return;
      }
    }

    pcl::PointCloud<ouster_ros::PointTime> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    process_cloud(pl_orig, [](const ouster_ros::PointTime &point) {
      return point.time;
    });
  }
  else
  {
    pcl::PointCloud<ouster_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    process_cloud(pl_orig, [](const ouster_ros::Point &point) {
      return point.t;
    });
  }
}

void Preprocess::velodyne_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();

  pcl::PointCloud<velodyne_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  if (plsize == 0)
    return;
  pl_surf.reserve(plsize);

  /*** These variables only works when no point timestamps given ***/
  double omega_l = 0.361 * SCAN_RATE;  // scan angular velocity
  std::vector<bool> is_first(N_SCANS, true);
  std::vector<double> yaw_fp(N_SCANS, 0.0);    // yaw of first scan point
  std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
  std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
  /*****************************************************************/

  if (pl_orig.points[plsize - 1].time > 0)
  {
    given_offset_time = true;
  }
  else
  {
    given_offset_time = false;
    double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
    double yaw_end = yaw_first;
    int layer_first = pl_orig.points[0].ring;
    for (uint i = plsize - 1; i > 0; i--)
    {
      if (pl_orig.points[i].ring == layer_first)
      {
        yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
        break;
      }
    }
  }

  if (feature_enabled)
  {
    for (int i = 0; i < N_SCANS; i++)
    {
      pl_buff[i].clear();
      pl_buff[i].reserve(plsize);
    }

    for (int i = 0; i < plsize; i++)
    {
      PointType added_pt;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      int layer = pl_orig.points[i].ring;
      if (layer >= N_SCANS)
        continue;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.curvature = pl_orig.points[i].time * time_unit_scale;  // units: ms
      if (!valid_range_xyz(added_pt.x, added_pt.y, added_pt.z))
        continue;

      if (!given_offset_time)
      {
        double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;
        if (is_first[layer])
        {
          // printf("layer: %d; is first: %d", layer, is_first[layer]);
          yaw_fp[layer] = yaw_angle;
          is_first[layer] = false;
          added_pt.curvature = 0.0;
          yaw_last[layer] = yaw_angle;
          time_last[layer] = added_pt.curvature;
          continue;
        }

        if (yaw_angle <= yaw_fp[layer])
        {
          added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
        }
        else
        {
          added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
        }

        if (added_pt.curvature < time_last[layer])
          added_pt.curvature += 360.0 / omega_l;

        yaw_last[layer] = yaw_angle;
        time_last[layer] = added_pt.curvature;
      }

      pl_buff[layer].points.push_back(added_pt);
    }

    for (int j = 0; j < N_SCANS; j++)
    {
      PointCloudXYZI& pl = pl_buff[j];
      int linesize = pl.size();
      if (linesize < 2)
        continue;
      vector<orgtype>& types = typess[j];
      types.clear();
      types.resize(linesize);
      linesize--;
      for (uint i = 0; i < linesize; i++)
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = vx * vx + vy * vy + vz * vz;
      }
      types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);
      give_feature(pl, types);
    }
  }
  else
  {
    for (int i = 0; i < plsize; i++)
    {
      PointType added_pt;
      // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;

      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.curvature =
          pl_orig.points[i].time * time_unit_scale;  // curvature unit: ms // cout<<added_pt.curvature<<endl;

      if (!given_offset_time)
      {
        int layer = pl_orig.points[i].ring;
        double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

        if (is_first[layer])
        {
          // printf("layer: %d; is first: %d", layer, is_first[layer]);
          yaw_fp[layer] = yaw_angle;
          is_first[layer] = false;
          added_pt.curvature = 0.0;
          yaw_last[layer] = yaw_angle;
          time_last[layer] = added_pt.curvature;
          continue;
        }

        // compute offset time
        if (yaw_angle <= yaw_fp[layer])
        {
          added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
        }
        else
        {
          added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
        }

        if (added_pt.curvature < time_last[layer])
          added_pt.curvature += 360.0 / omega_l;

        yaw_last[layer] = yaw_angle;
        time_last[layer] = added_pt.curvature;
      }

      if (i % point_filter_num == 0)
      {
        if (valid_range_xyz(added_pt.x, added_pt.y, added_pt.z))
        {
          pl_surf.points.push_back(added_pt);
        }
      }
    }
  }
}

void Preprocess::mid360_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();

  struct DecodedPoint
  {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float intensity = 0.0f;
    uint8_t tag = 0;
    int line = 0;
    double offset_ms = 0.0;
    bool has_time = false;
  };

  const bool has_offset_time_field =
      std::any_of(msg->fields.begin(), msg->fields.end(),
                  [](const sensor_msgs::msg::PointField &field) { return field.name == "offset_time"; });
  const double header_time = rclcpp::Time(msg->header.stamp).seconds();
  std::vector<DecodedPoint> decoded;

  if (has_offset_time_field)
  {
    pcl::PointCloud<livox_ros::LivoxPointXyzrto> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    decoded.resize(pl_orig.size());
    for (std::size_t i = 0; i < pl_orig.size(); ++i)
    {
      decoded[i].x = pl_orig.points[i].x;
      decoded[i].y = pl_orig.points[i].y;
      decoded[i].z = pl_orig.points[i].z;
      decoded[i].intensity = pl_orig.points[i].intensity;
      decoded[i].tag = static_cast<uint8_t>(pl_orig.points[i].tag);
      decoded[i].line = static_cast<int>(pl_orig.points[i].line);
      decoded[i].offset_ms = static_cast<double>(pl_orig.points[i].offset_time) * 1.0e-6;
      decoded[i].has_time = true;  // offset_time == 0 is valid for the first point.
    }
  }
  else
  {
    pcl::PointCloud<livox_ros::LivoxPointXyzrtl> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    decoded.resize(pl_orig.size());
    for (std::size_t i = 0; i < pl_orig.size(); ++i)
    {
      decoded[i].x = pl_orig.points[i].x;
      decoded[i].y = pl_orig.points[i].y;
      decoded[i].z = pl_orig.points[i].z;
      decoded[i].intensity = pl_orig.points[i].intensity;
      decoded[i].tag = pl_orig.points[i].tag;
      decoded[i].line = static_cast<int>(pl_orig.points[i].line);
      const double timestamp = pl_orig.points[i].timestamp;
      if (std::isfinite(timestamp) && std::fabs(timestamp) > 1.0e-12)
      {
        decoded[i].offset_ms = pointTimeToOffsetMs(timestamp, header_time, time_unit_scale);
        decoded[i].has_time = true;
      }
    }
  }

  const int plsize = static_cast<int>(decoded.size());
  if (plsize == 0)
    return;
  pl_surf.reserve(plsize);

  /*** These variables only works when no point timestamps given ***/
  double omega_l = 0.361 * SCAN_RATE;  // scan angular velocity
  std::vector<bool> is_first(N_SCANS, true);
  std::vector<double> yaw_fp(N_SCANS, 0.0);    // yaw of first scan point
  std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
  std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
  /*****************************************************************/

  std::vector<double> point_offset_ms(plsize, 0.0);
  std::vector<bool> point_has_timestamp(plsize, false);
  double min_offset_ms = 1.0e300;
  double last_raw_offset_ms = 0.0;
  bool has_point_timestamp = false;
  bool point_timestamp_loop_back = false;
  for (int i = 0; i < plsize; ++i)
  {
    if (decoded[i].has_time && std::isfinite(decoded[i].offset_ms))
    {
      has_point_timestamp = true;
      point_has_timestamp[i] = true;
      point_offset_ms[i] = decoded[i].offset_ms;
      if (i > 0 && point_offset_ms[i] < last_raw_offset_ms)
        point_timestamp_loop_back = true;
      last_raw_offset_ms = point_offset_ms[i];
      if (point_offset_ms[i] < min_offset_ms)
        min_offset_ms = point_offset_ms[i];
    }
  }
  if (has_point_timestamp)
  {
    for (auto &offset_ms : point_offset_ms)
      offset_ms -= min_offset_ms;
  }

  given_offset_time = has_point_timestamp;

  for (uint i = 0; i < plsize; ++i)
  {
    if (!valid_livox_tag(decoded[i].tag))
      continue;

    PointType added_pt;
    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.x = decoded[i].x;
    added_pt.y = decoded[i].y;
    added_pt.z = decoded[i].z;
    added_pt.intensity = decoded[i].intensity;
    added_pt.curvature = 0.;

    int layer = decoded[i].line;
    if (layer < 0 || layer >= N_SCANS)
      continue;
    double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

    if (given_offset_time)
    {
      if (!point_has_timestamp[i])
        continue;
      added_pt.curvature = point_offset_ms[i];
      if (valid_range_xyz(added_pt.x, added_pt.y, added_pt.z))
      {
        pl_surf.push_back(std::move(added_pt));
      }
      continue;
    }

    if (is_first[layer])
    {
      // printf("layer: %d; is first: %d", layer, is_first[layer]);
      yaw_fp[layer] = yaw_angle;
      is_first[layer] = false;
      added_pt.curvature = 0.0;
      yaw_last[layer] = yaw_angle;
      time_last[layer] = added_pt.curvature;
      continue;
    }

    // compute offset time
    if (yaw_angle <= yaw_fp[layer])
    {
      added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
    }
    else
    {
      added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
    }

    if (added_pt.curvature < time_last[layer])
      added_pt.curvature += 360.0 / omega_l;

    yaw_last[layer] = yaw_angle;
    time_last[layer] = added_pt.curvature;

    if (valid_range_xyz(added_pt.x, added_pt.y, added_pt.z))
    {
      pl_surf.push_back(std::move(added_pt));
    }
  }

  if (given_offset_time)
  {
    // Merged front/back MID360 clouds can be concatenated with a timestamp
    // drop at the sensor boundary. Sort by per-point time so FAST-LIO sees a
    // monotonic scan and uses the real latest point as the frame end.
    std::sort(pl_surf.points.begin(), pl_surf.points.end(),
              [](const PointType &a, const PointType &b) { return a.curvature < b.curvature; });
    if (point_timestamp_loop_back)
    {
      static int loop_back_report_count = 0;
      if (loop_back_report_count < 5)
      {
        RCLCPP_DEBUG(rclcpp::get_logger("laser_mapping"),
                     "MID360 point timestamps loop back inside one cloud; sorted points by timestamp");
        ++loop_back_report_count;
      }
    }
  }
}

void Preprocess::robosense_m1_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg)
{
  robosense_airy_handler(msg);
}

void Preprocess::robosense_airy_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();

  pcl::PointCloud<robosense_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  if (plsize == 0)
    return;
  const int cloud_width = static_cast<int>(pl_orig.width);
  const int cloud_height = static_cast<int>(pl_orig.height);
  const bool organized =
      (cloud_width > 1 && cloud_height > 1 && cloud_width * cloud_height == plsize);
  pl_surf.reserve(plsize);
  const double max_range = det_range;

  if (feature_enabled)
  {
    for (int i = 0; i < N_SCANS; ++i)
    {
      pl_buff[i].clear();
      pl_buff[i].reserve(plsize);
    }

    double start_time = 0.0;
    bool start_time_set = false;
    if (organized)
    {
      for (int i_ori_width = 0; i_ori_width < cloud_width; ++i_ori_width)
      {
        for (int i_ori_height = 0; i_ori_height < cloud_height; ++i_ori_height)
        {
          const auto& ori_point = pl_orig.at(i_ori_width, i_ori_height);
          if (!std::isfinite(ori_point.timestamp))
            continue;
          start_time = ori_point.timestamp;
          start_time_set = true;
          break;
        }
        if (start_time_set)
          break;
      }

      for (int i_ori_width = 0; i_ori_width < cloud_width; ++i_ori_width)
      {
        for (int i_ori_height = 0; i_ori_height < cloud_height; ++i_ori_height)
        {
          const auto& ori_point = pl_orig.at(i_ori_width, i_ori_height);
          if (i_ori_height % point_filter_num != 0)
            continue;

          double range = std::sqrt(ori_point.x * ori_point.x + ori_point.y * ori_point.y + ori_point.z * ori_point.z);
          if (!(range < max_range && range > blind) || std::abs(ori_point.z) > max_height)
            continue;

          PointType added_pt;
          added_pt.x = ori_point.x;
          added_pt.y = ori_point.y;
          added_pt.z = ori_point.z;
          added_pt.intensity = ori_point.intensity;
          added_pt.normal_x = 0;
          added_pt.normal_y = 0;
          added_pt.normal_z = 0;
          added_pt.curvature = 0.0f;
          if (start_time_set && std::isfinite(ori_point.timestamp))
          {
            added_pt.curvature = static_cast<float>((ori_point.timestamp - start_time) * time_unit_scale);
          }
          if (i_ori_width < N_SCANS)
          {
            pl_buff[i_ori_width].push_back(added_pt);
          }
        }
      }
    }
    else
    {
      double start_time = 0.0;
      bool start_time_set = false;
      for (int i = 0; i < plsize; ++i)
      {
        if (std::isfinite(pl_orig.points[i].timestamp))
        {
          start_time = pl_orig.points[i].timestamp;
          start_time_set = true;
          break;
        }
      }
      for (int i = 0; i < plsize; ++i)
      {
        const auto& ori_point = pl_orig.points[i];
        if (ori_point.ring >= N_SCANS)
          continue;
        if (i % point_filter_num != 0)
          continue;
        double range = std::sqrt(ori_point.x * ori_point.x + ori_point.y * ori_point.y + ori_point.z * ori_point.z);
        if (!(range < max_range && range > blind) || std::abs(ori_point.z) > max_height)
          continue;
        PointType added_pt;
        added_pt.x = ori_point.x;
        added_pt.y = ori_point.y;
        added_pt.z = ori_point.z;
        added_pt.intensity = ori_point.intensity;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.curvature = 0.0f;
        if (start_time_set && std::isfinite(ori_point.timestamp))
        {
          added_pt.curvature = static_cast<float>((ori_point.timestamp - start_time) * time_unit_scale);
        }
        pl_buff[ori_point.ring].push_back(added_pt);
      }
    }

    for (int j = 0; j < N_SCANS; ++j)
    {
      PointCloudXYZI& pl = pl_buff[j];
      int linesize = pl.size();
      if (linesize < 2)
        continue;
      vector<orgtype>& types = typess[j];
      types.clear();
      types.resize(linesize);
      linesize--;
      for (uint i = 0; i < static_cast<uint>(linesize); ++i)
      {
        types[i].range = std::sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = vx * vx + vy * vy + vz * vz;
      }
      types[linesize].range = std::sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);
      give_feature(pl, types);
    }
    return;
  }

  double start_time = 0.0;
  bool start_time_set = false;
  if (organized)
  {
    for (int i_ori_width = 0; i_ori_width < cloud_width; ++i_ori_width)
    {
      for (int i_ori_height = 0; i_ori_height < cloud_height; ++i_ori_height)
      {
        const auto& ori_point = pl_orig.at(i_ori_width, i_ori_height);
        if (!std::isfinite(ori_point.timestamp))
          continue;
        start_time = ori_point.timestamp;
        start_time_set = true;
        break;
      }
      if (start_time_set)
        break;
    }

    for (int i_ori_width = 0; i_ori_width < cloud_width; ++i_ori_width)
    {
      for (int i_ori_height = 0; i_ori_height < cloud_height; ++i_ori_height)
      {
        const auto& ori_point = pl_orig.at(i_ori_width, i_ori_height);
        if (i_ori_height % point_filter_num != 0)
          continue;

        double range = std::sqrt(ori_point.x * ori_point.x + ori_point.y * ori_point.y + ori_point.z * ori_point.z);
        if (!(range < max_range && range > blind) || std::abs(ori_point.z) > max_height)
          continue;

        PointType added_pt;
        added_pt.x = ori_point.x;
        added_pt.y = ori_point.y;
        added_pt.z = ori_point.z;
        added_pt.intensity = ori_point.intensity;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.curvature = 0.0f;
        if (start_time_set && std::isfinite(ori_point.timestamp))
        {
          added_pt.curvature = static_cast<float>((ori_point.timestamp - start_time) * time_unit_scale);
        }
        pl_surf.points.push_back(added_pt);
      }
    }
    return;
  }

  for (int i = 0; i < plsize; ++i)
  {
    if (std::isfinite(pl_orig.points[i].timestamp))
    {
      start_time = pl_orig.points[i].timestamp;
      start_time_set = true;
      break;
    }
  }

  for (int i = 0; i < plsize; ++i)
  {
    const auto& src = pl_orig.points[i];
    if (i % point_filter_num != 0)
      continue;
    double range = std::sqrt(src.x * src.x + src.y * src.y + src.z * src.z);
    if (!(range < max_range && range > blind) || std::abs(src.z) > max_height)
      continue;

    PointType added_pt;
    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.x = src.x;
    added_pt.y = src.y;
    added_pt.z = src.z;
    added_pt.intensity = src.intensity;
    added_pt.curvature = 0.0f;
    if (start_time_set && std::isfinite(src.timestamp))
    {
      added_pt.curvature = static_cast<float>((src.timestamp - start_time) * time_unit_scale);
    }
    pl_surf.points.push_back(added_pt);
  }
}

void Preprocess::default_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();

  pcl::PointCloud<pcl::PointXYZI> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  if (plsize == 0)
    return;
  pl_surf.reserve(plsize);

  for(uint i = 0; i < plsize; ++i)
  {
    PointType added_pt;
    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    added_pt.curvature = 0.;

    if (valid_range_xyz(added_pt.x, added_pt.y, added_pt.z))
    {
      pl_surf.push_back(std::move(added_pt));
    }
  }
}

void Preprocess::give_feature(pcl::PointCloud<PointType>& pl, vector<orgtype>& types)
{
  int plsize = pl.size();
  int plsize2;
  if (plsize == 0)
  {
    printf("something wrong\n");
    return;
  }
  uint head = 0;

  while (types[head].range < blind)
  {
    head++;
  }

  // Surf
  plsize2 = (plsize > group_size) ? (plsize - group_size) : 0;

  Eigen::Vector3d curr_direct(Eigen::Vector3d::Zero());
  Eigen::Vector3d last_direct(Eigen::Vector3d::Zero());

  uint i_nex = 0, i2;
  uint last_i = 0;
  uint last_i_nex = 0;
  int last_state = 0;
  int plane_type;

  for (uint i = head; i < plsize2; i++)
  {
    if (types[i].range < blind)
    {
      continue;
    }

    i2 = i;

    plane_type = plane_judge(pl, types, i, i_nex, curr_direct);

    if (plane_type == 1)
    {
      for (uint j = i; j <= i_nex; j++)
      {
        if (j != i && j != i_nex)
        {
          types[j].ftype = Real_Plane;
        }
        else
        {
          types[j].ftype = Poss_Plane;
        }
      }

      // if(last_state==1 && fabs(last_direct.sum())>0.5)
      if (last_state == 1 && last_direct.norm() > 0.1)
      {
        double mod = last_direct.transpose() * curr_direct;
        if (mod > -0.707 && mod < 0.707)
        {
          types[i].ftype = Edge_Plane;
        }
        else
        {
          types[i].ftype = Real_Plane;
        }
      }

      i = i_nex - 1;
      last_state = 1;
    }
    else  // if(plane_type == 2)
    {
      i = i_nex;
      last_state = 0;
    }
    // else if(plane_type == 0)
    // {
    //   if(last_state == 1)
    //   {
    //     uint i_nex_tem;
    //     uint j;
    //     for(j=last_i+1; j<=last_i_nex; j++)
    //     {
    //       uint i_nex_tem2 = i_nex_tem;
    //       Eigen::Vector3d curr_direct2;

    //       uint ttem = plane_judge(pl, types, j, i_nex_tem, curr_direct2);

    //       if(ttem != 1)
    //       {
    //         i_nex_tem = i_nex_tem2;
    //         break;
    //       }
    //       curr_direct = curr_direct2;
    //     }

    //     if(j == last_i+1)
    //     {
    //       last_state = 0;
    //     }
    //     else
    //     {
    //       for(uint k=last_i_nex; k<=i_nex_tem; k++)
    //       {
    //         if(k != i_nex_tem)
    //         {
    //           types[k].ftype = Real_Plane;
    //         }
    //         else
    //         {
    //           types[k].ftype = Poss_Plane;
    //         }
    //       }
    //       i = i_nex_tem-1;
    //       i_nex = i_nex_tem;
    //       i2 = j-1;
    //       last_state = 1;
    //     }

    //   }
    // }

    last_i = i2;
    last_i_nex = i_nex;
    last_direct = curr_direct;
  }

  plsize2 = plsize > 3 ? plsize - 3 : 0;
  for (uint i = head + 3; i < plsize2; i++)
  {
    if (types[i].range < blind || types[i].ftype >= Real_Plane)
    {
      continue;
    }

    if (types[i - 1].dista < 1e-16 || types[i].dista < 1e-16)
    {
      continue;
    }

    Eigen::Vector3d vec_a(pl[i].x, pl[i].y, pl[i].z);
    Eigen::Vector3d vecs[2];

    for (int j = 0; j < 2; j++)
    {
      int m = -1;
      if (j == 1)
      {
        m = 1;
      }

      if (types[i + m].range < blind)
      {
        if (types[i].range > inf_bound)
        {
          types[i].edj[j] = Nr_inf;
        }
        else
        {
          types[i].edj[j] = Nr_blind;
        }
        continue;
      }

      vecs[j] = Eigen::Vector3d(pl[i + m].x, pl[i + m].y, pl[i + m].z);
      vecs[j] = vecs[j] - vec_a;

      types[i].angle[j] = vec_a.dot(vecs[j]) / vec_a.norm() / vecs[j].norm();
      if (types[i].angle[j] < jump_up_limit)
      {
        types[i].edj[j] = Nr_180;
      }
      else if (types[i].angle[j] > jump_down_limit)
      {
        types[i].edj[j] = Nr_zero;
      }
    }

    types[i].intersect = vecs[Prev].dot(vecs[Next]) / vecs[Prev].norm() / vecs[Next].norm();
    if (types[i].edj[Prev] == Nr_nor && types[i].edj[Next] == Nr_zero && types[i].dista > 0.0225 &&
        types[i].dista > 4 * types[i - 1].dista)
    {
      if (types[i].intersect > cos160)
      {
        if (edge_jump_judge(pl, types, i, Prev))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    else if (types[i].edj[Prev] == Nr_zero && types[i].edj[Next] == Nr_nor && types[i - 1].dista > 0.0225 &&
             types[i - 1].dista > 4 * types[i].dista)
    {
      if (types[i].intersect > cos160)
      {
        if (edge_jump_judge(pl, types, i, Next))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    else if (types[i].edj[Prev] == Nr_nor && types[i].edj[Next] == Nr_inf)
    {
      if (edge_jump_judge(pl, types, i, Prev))
      {
        types[i].ftype = Edge_Jump;
      }
    }
    else if (types[i].edj[Prev] == Nr_inf && types[i].edj[Next] == Nr_nor)
    {
      if (edge_jump_judge(pl, types, i, Next))
      {
        types[i].ftype = Edge_Jump;
      }
    }
    else if (types[i].edj[Prev] > Nr_nor && types[i].edj[Next] > Nr_nor)
    {
      if (types[i].ftype == Nor)
      {
        types[i].ftype = Wire;
      }
    }
  }

  plsize2 = plsize - 1;
  double ratio;
  for (uint i = head + 1; i < plsize2; i++)
  {
    if (types[i].range < blind || types[i - 1].range < blind || types[i + 1].range < blind)
    {
      continue;
    }

    if (types[i - 1].dista < 1e-8 || types[i].dista < 1e-8)
    {
      continue;
    }

    if (types[i].ftype == Nor)
    {
      if (types[i - 1].dista > types[i].dista)
      {
        ratio = types[i - 1].dista / types[i].dista;
      }
      else
      {
        ratio = types[i].dista / types[i - 1].dista;
      }

      if (types[i].intersect < smallp_intersect && ratio < smallp_ratio)
      {
        if (types[i - 1].ftype == Nor)
        {
          types[i - 1].ftype = Real_Plane;
        }
        if (types[i + 1].ftype == Nor)
        {
          types[i + 1].ftype = Real_Plane;
        }
        types[i].ftype = Real_Plane;
      }
    }
  }

  int last_surface = -1;
  for (uint j = head; j < plsize; j++)
  {
    if (types[j].ftype == Poss_Plane || types[j].ftype == Real_Plane)
    {
      if (last_surface == -1)
      {
        last_surface = j;
      }

      if (j == uint(last_surface + point_filter_num - 1))
      {
        PointType ap;
        ap.x = pl[j].x;
        ap.y = pl[j].y;
        ap.z = pl[j].z;
        ap.intensity = pl[j].intensity;
        ap.curvature = pl[j].curvature;
        pl_surf.push_back(ap);

        last_surface = -1;
      }
    }
    else
    {
      if (types[j].ftype == Edge_Jump || types[j].ftype == Edge_Plane)
      {
        pl_corn.push_back(pl[j]);
      }
      if (last_surface != -1)
      {
        PointType ap;
        for (uint k = last_surface; k < j; k++)
        {
          ap.x += pl[k].x;
          ap.y += pl[k].y;
          ap.z += pl[k].z;
          ap.intensity += pl[k].intensity;
          ap.curvature += pl[k].curvature;
        }
        ap.x /= (j - last_surface);
        ap.y /= (j - last_surface);
        ap.z /= (j - last_surface);
        ap.intensity /= (j - last_surface);
        ap.curvature /= (j - last_surface);
        pl_surf.push_back(ap);
      }
      last_surface = -1;
    }
  }
}

void Preprocess::pub_func(PointCloudXYZI& pl, const rclcpp::Time& ct)
{
  pl.height = 1;
  pl.width = pl.size();
  sensor_msgs::msg::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "livox";
  output.header.stamp = ct;
}

int Preprocess::plane_judge(const PointCloudXYZI& pl, vector<orgtype>& types, uint i_cur, uint& i_nex,
                            Eigen::Vector3d& curr_direct)
{
  double group_dis = disA * types[i_cur].range + disB;
  group_dis = group_dis * group_dis;
  // i_nex = i_cur;

  double two_dis;
  vector<double> disarr;
  disarr.reserve(20);

  for (i_nex = i_cur; i_nex < i_cur + group_size; i_nex++)
  {
    if (types[i_nex].range < blind)
    {
      curr_direct.setZero();
      return 2;
    }
    disarr.push_back(types[i_nex].dista);
  }

  for (;;)
  {
    if ((i_cur >= pl.size()) || (i_nex >= pl.size()))
      break;

    if (types[i_nex].range < blind)
    {
      curr_direct.setZero();
      return 2;
    }
    vx = pl[i_nex].x - pl[i_cur].x;
    vy = pl[i_nex].y - pl[i_cur].y;
    vz = pl[i_nex].z - pl[i_cur].z;
    two_dis = vx * vx + vy * vy + vz * vz;
    if (two_dis >= group_dis)
    {
      break;
    }
    disarr.push_back(types[i_nex].dista);
    i_nex++;
  }

  double leng_wid = 0;
  double v1[3], v2[3];
  for (uint j = i_cur + 1; j < i_nex; j++)
  {
    if ((j >= pl.size()) || (i_cur >= pl.size()))
      break;
    v1[0] = pl[j].x - pl[i_cur].x;
    v1[1] = pl[j].y - pl[i_cur].y;
    v1[2] = pl[j].z - pl[i_cur].z;

    v2[0] = v1[1] * vz - vy * v1[2];
    v2[1] = v1[2] * vx - v1[0] * vz;
    v2[2] = v1[0] * vy - vx * v1[1];

    double lw = v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2];
    if (lw > leng_wid)
    {
      leng_wid = lw;
    }
  }

  if ((two_dis * two_dis / leng_wid) < p2l_ratio)
  {
    curr_direct.setZero();
    return 0;
  }

  uint disarrsize = disarr.size();
  for (uint j = 0; j < disarrsize - 1; j++)
  {
    for (uint k = j + 1; k < disarrsize; k++)
    {
      if (disarr[j] < disarr[k])
      {
        leng_wid = disarr[j];
        disarr[j] = disarr[k];
        disarr[k] = leng_wid;
      }
    }
  }

  if (disarr[disarr.size() - 2] < 1e-16)
  {
    curr_direct.setZero();
    return 0;
  }

  if (lidar_type == AVIA)
  {
    double dismax_mid = disarr[0] / disarr[disarrsize / 2];
    double dismid_min = disarr[disarrsize / 2] / disarr[disarrsize - 2];

    if (dismax_mid >= limit_maxmid || dismid_min >= limit_midmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }
  else
  {
    double dismax_min = disarr[0] / disarr[disarrsize - 2];
    if (dismax_min >= limit_maxmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }

  curr_direct << vx, vy, vz;
  curr_direct.normalize();
  return 1;
}

bool Preprocess::edge_jump_judge(const PointCloudXYZI& pl, vector<orgtype>& types, uint i, Surround nor_dir)
{
  if (nor_dir == 0)
  {
    if (types[i - 1].range < blind || types[i - 2].range < blind)
    {
      return false;
    }
  }
  else if (nor_dir == 1)
  {
    if (types[i + 1].range < blind || types[i + 2].range < blind)
    {
      return false;
    }
  }
  double d1 = types[i + nor_dir - 1].dista;
  double d2 = types[i + 3 * nor_dir - 2].dista;
  double d;

  if (d1 < d2)
  {
    d = d1;
    d1 = d2;
    d2 = d;
  }

  d1 = sqrt(d1);
  d2 = sqrt(d2);

  if (d1 > edgea * d2 || (d1 - d2) > edgeb)
  {
    return false;
  }

  return true;
}
