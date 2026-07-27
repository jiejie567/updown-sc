// #include <ros/ros.h>
#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>

using namespace std;

#define IS_VALID(a) ((abs(a) > 1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE
{
  AVIA = 1,
  VELO16,
  OUST64,
  MID360,
  ROBOSENSE,
  RSM1,
  RSAIRY,
  GAZEBO_XYZI
};
enum TIME_UNIT
{
  SEC = 0,
  MS = 1,
  US = 2,
  NS = 3
};
enum Feature
{
  Nor,
  Poss_Plane,
  Real_Plane,
  Edge_Jump,
  Edge_Plane,
  Wire,
  ZeroPoint
};
enum Surround
{
  Prev,
  Next
};
enum E_jump
{
  Nr_nor,
  Nr_zero,
  Nr_180,
  Nr_inf,
  Nr_blind
};

struct orgtype
{
  double range;
  double dista;
  double angle[2];
  double intersect;
  E_jump edj[2];
  Feature ftype;
  orgtype()
  {
    range = 0;
    edj[Prev] = Nr_nor;
    edj[Next] = Nr_nor;
    ftype = Nor;
    intersect = 2;
  }
};

namespace velodyne_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                                                          intensity)(float, time, time)(uint16_t, ring,
                                                                                                        ring))

namespace ouster_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint8_t ring;
  uint16_t ambient;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// LiDAR/IMU-only Newer College ROS 2 release.  It stores the per-scan
// nanosecond offset as "time" and omits the auxiliary Ouster image fields.
struct EIGEN_ALIGN16 PointTime
{
  PCL_ADD_POINT4D;
  uint32_t time;
  uint8_t ring;
  float intensity;
  uint8_t dynamic;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::PointTime,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (std::uint32_t, time, time)
    (std::uint8_t, ring, ring)
    (float, intensity, intensity)
    (std::uint8_t, dynamic, dynamic)
)

namespace livox_ros
{
typedef struct {
  float x;            /**< X axis, Unit:m */
  float y;            /**< Y axis, Unit:m */
  float z;            /**< Z axis, Unit:m */
  float intensity;
  //float reflectivity; /**< Reflectivity   */
  uint8_t tag;        /**< Livox point tag   */
  uint8_t line;       /**< Laser line id     */
  double timestamp;   /**< Per-point timestamp */
} LivoxPointXyzrtl;

// PointCloud2 layout used by ROS bag conversions of Livox CustomMsg.  The
// original per-point nanosecond offset is retained as offset_time, while tag
// and line are widened to uint32 by the converter.
typedef struct {
  float x;
  float y;
  float z;
  float intensity;
  uint32_t tag;
  uint32_t line;
  uint32_t offset_time;
} LivoxPointXyzrto;
}
POINT_CLOUD_REGISTER_POINT_STRUCT(livox_ros::LivoxPointXyzrtl,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity) 
    //(float, reflectivity, reflectivity)
    (uint8_t, tag, tag)
    (uint8_t, line, line)
    (double, timestamp, timestamp)
)
POINT_CLOUD_REGISTER_POINT_STRUCT(livox_ros::LivoxPointXyzrto,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (std::uint32_t, tag, tag)
    (std::uint32_t, line, line)
    (std::uint32_t, offset_time, offset_time)
)

namespace robosense_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  uint16_t ring;
  double timestamp;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}
POINT_CLOUD_REGISTER_POINT_STRUCT(robosense_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint16_t, ring, ring)
    (double, timestamp, timestamp)
)

class Preprocess
{
  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum TagFilterMode
  {
    TAG_FILTER_OFF = 0,
    TAG_FILTER_OTHER = 1,
    TAG_FILTER_LOW_CONFIDENCE = 2,
    TAG_FILTER_STRICT = 3
  };
  enum BlindFilterShape
  {
    BLIND_FILTER_SPHERE = 0,
    BLIND_FILTER_CYLINDER = 1
  };

  Preprocess();
  ~Preprocess();
  
  void process(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void process(const sensor_msgs::msg::PointCloud2::UniquePtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void set(bool feat_en, int lid_type, double bld, double max_range, double max_z, int pfilt_num);
  void set_tag_filter_mode(int mode);
  void set_blind_filter(int shape, double z_min, double z_max);

  // sensor_msgs::PointCloud2::ConstPtr pointcloud;
  PointCloudXYZI pl_full, pl_corn, pl_surf;
  PointCloudXYZI pl_buff[128]; //maximum 128 line lidar
  vector<orgtype> typess[128]; //maximum 128 line lidar
  float time_unit_scale;
  int lidar_type, point_filter_num, N_SCANS, SCAN_RATE, time_unit, tag_filter_mode, blind_filter_shape;
  double blind, det_range, max_height, blind_z_min, blind_z_max;
  bool feature_enabled, given_offset_time;
  // ros::Publisher pub_full, pub_surf, pub_corn;

private:
  void avia_handler(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg);
  void oust64_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg);
  void velodyne_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg);
  void mid360_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg);
  void robosense_m1_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg);
  void robosense_airy_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg);
  void default_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg);
  bool valid_range_sq(double range_sq) const;
  bool valid_range_xyz(double x, double y, double z) const;
  bool valid_livox_tag(uint8_t tag) const;
  void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);
  void pub_func(PointCloudXYZI &pl, const rclcpp::Time &ct);
  int  plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool small_plane(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);
  
  int group_size;
  double disA, disB, inf_bound;
  double limit_maxmid, limit_midmin, limit_maxmin;
  double p2l_ratio;
  double jump_up_limit, jump_down_limit;
  double cos160;
  double edgea, edgeb;
  double smallp_intersect, smallp_ratio;
  double vx, vy, vz;
};
