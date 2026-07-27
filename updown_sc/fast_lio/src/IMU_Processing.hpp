#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include "use-ikfom.hpp"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  explicit ImuProcess(bool deskew_en);
  ~ImuProcess();
  
  void Reset();
  // void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void Reset(double start_timestamp, const sensor_msgs::msg::Imu::ConstSharedPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q;
  struct Snapshot
  {
    Eigen::Matrix<double, 12, 12> Q;
    V3D cov_acc;
    V3D cov_gyr;
    V3D cov_acc_scale;
    V3D cov_gyr_scale;
    V3D cov_bias_gyr;
    V3D cov_bias_acc;
    double first_lidar_time = 0.0;
    bool deskew_en = true;
    sensor_msgs::msg::Imu::ConstSharedPtr last_imu;
    deque<sensor_msgs::msg::Imu::ConstSharedPtr> v_imu;
    vector<Pose6D> IMUpose;
    vector<M3D> v_rot_pcl;
    M3D Lidar_R_wrt_IMU;
    V3D Lidar_T_wrt_IMU;
    V3D mean_acc;
    V3D mean_gyr;
    V3D angvel_last;
    V3D acc_s_last;
    double start_timestamp = -1.0;
    double last_lidar_end_time = 0.0;
    int init_iter_num = 1;
    bool b_first_frame = true;
    bool imu_need_init = true;
  };
  Snapshot CaptureSnapshot() const;
  void RestoreSnapshot(const Snapshot &snapshot);
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);
  bool DeskewOrderedCloudWithCachedTrajectory(
      PointCloudXYZI &cloud,
      const state_ikfom &frame_end_state) const;

  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;
  bool deskew_en{true};

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;
  // sensor_msgs::ImuConstPtr last_imu_;
  sensor_msgs::msg::Imu::ConstSharedPtr last_imu_;
  deque<sensor_msgs::msg::Imu::ConstSharedPtr> v_imu_;
  vector<Pose6D> IMUpose;
  vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double start_timestamp_;
  double last_lidar_end_time_;
  int    init_iter_num = 1;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;
  bool   moving_init_bias_warned_ = false;
};

ImuProcess::ImuProcess()
    : ImuProcess(true)
{}

ImuProcess::ImuProcess(bool deskew_en)
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1), deskew_en(deskew_en)
{
  init_iter_num = 1;
  Q = process_noise_cov();
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::msg::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  moving_init_bias_warned_ = false;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::msg::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

ImuProcess::Snapshot ImuProcess::CaptureSnapshot() const
{
  Snapshot snapshot;
  snapshot.Q = Q;
  snapshot.cov_acc = cov_acc;
  snapshot.cov_gyr = cov_gyr;
  snapshot.cov_acc_scale = cov_acc_scale;
  snapshot.cov_gyr_scale = cov_gyr_scale;
  snapshot.cov_bias_gyr = cov_bias_gyr;
  snapshot.cov_bias_acc = cov_bias_acc;
  snapshot.first_lidar_time = first_lidar_time;
  snapshot.deskew_en = deskew_en;
  snapshot.last_imu = last_imu_;
  snapshot.v_imu = v_imu_;
  snapshot.IMUpose = IMUpose;
  snapshot.v_rot_pcl = v_rot_pcl_;
  snapshot.Lidar_R_wrt_IMU = Lidar_R_wrt_IMU;
  snapshot.Lidar_T_wrt_IMU = Lidar_T_wrt_IMU;
  snapshot.mean_acc = mean_acc;
  snapshot.mean_gyr = mean_gyr;
  snapshot.angvel_last = angvel_last;
  snapshot.acc_s_last = acc_s_last;
  snapshot.start_timestamp = start_timestamp_;
  snapshot.last_lidar_end_time = last_lidar_end_time_;
  snapshot.init_iter_num = init_iter_num;
  snapshot.b_first_frame = b_first_frame_;
  snapshot.imu_need_init = imu_need_init_;
  return snapshot;
}

void ImuProcess::RestoreSnapshot(const Snapshot &snapshot)
{
  Q = snapshot.Q;
  cov_acc = snapshot.cov_acc;
  cov_gyr = snapshot.cov_gyr;
  cov_acc_scale = snapshot.cov_acc_scale;
  cov_gyr_scale = snapshot.cov_gyr_scale;
  cov_bias_gyr = snapshot.cov_bias_gyr;
  cov_bias_acc = snapshot.cov_bias_acc;
  first_lidar_time = snapshot.first_lidar_time;
  deskew_en = snapshot.deskew_en;
  last_imu_ = snapshot.last_imu;
  v_imu_ = snapshot.v_imu;
  IMUpose = snapshot.IMUpose;
  v_rot_pcl_ = snapshot.v_rot_pcl;
  Lidar_R_wrt_IMU = snapshot.Lidar_R_wrt_IMU;
  Lidar_T_wrt_IMU = snapshot.Lidar_T_wrt_IMU;
  mean_acc = snapshot.mean_acc;
  mean_gyr = snapshot.mean_gyr;
  angvel_last = snapshot.angvel_last;
  acc_s_last = snapshot.acc_s_last;
  start_timestamp_ = snapshot.start_timestamp;
  last_lidar_end_time_ = snapshot.last_lidar_end_time;
  init_iter_num = snapshot.init_iter_num;
  b_first_frame_ = snapshot.b_first_frame;
  imu_need_init_ = snapshot.imu_need_init;
  moving_init_bias_warned_ = false;
  cur_pcl_un_.reset(new PointCloudXYZI());
  if (fout_imu.is_open())
  {
    fout_imu.close();
  }
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N ++;
  }
  state_ikfom init_state = kf_state.get_x();
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  constexpr double MAX_STATIC_GYRO_NORM_FOR_BIAS = 0.1;
  if (mean_gyr.norm() <= MAX_STATIC_GYRO_NORM_FOR_BIAS)
  {
    init_state.bg = mean_gyr;
  }
  else
  {
    init_state.bg = Zero3d;
    if (!moving_init_bias_warned_)
    {
      moving_init_bias_warned_ = true;
      RCLCPP_WARN(
        rclcpp::get_logger("laser_mapping"),
        "IMU initialization detected motion: mean_gyro=[%.3f %.3f %.3f] norm=%.3f rad/s > %.3f. Keep gyro bias at zero instead of treating motion as bias.",
        mean_gyr(0), mean_gyr(1), mean_gyr(2), mean_gyr.norm(),
        MAX_STATIC_GYRO_NORM_FOR_BIAS);
    }
  }
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = 0.00001; 
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();

}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = rclcpp::Time(v_imu.front()->header.stamp).seconds();
  const double &imu_end_time = rclcpp::Time(v_imu.back()->header.stamp).seconds();
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;
  
  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  if (deskew_en)
  {
    sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  }
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr;

  double dt = 0;

  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    double tail_stamp = rclcpp::Time(tail->header.stamp).seconds();
    double head_stamp = rclcpp::Time(head->header.stamp).seconds();

    if (tail_stamp < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    if(head_stamp < last_lidar_end_time_)
    {
      dt = tail_stamp - last_lidar_end_time_;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      dt = tail_stamp - head_stamp;
    }
    
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail_stamp - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);
  
  imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  /*** undistort each lidar point (backward propagation) ***/
  DeskewOrderedCloudWithCachedTrajectory(pcl_out, imu_state);
}

bool ImuProcess::DeskewOrderedCloudWithCachedTrajectory(
    PointCloudXYZI &cloud,
    const state_ikfom &frame_end_state) const
{
  if (!deskew_en || cloud.empty())
    return true;
  if (IMUpose.size() < 2)
    return false;
  if (!std::is_sorted(
          cloud.points.begin(), cloud.points.end(),
          [](const PointType &lhs, const PointType &rhs)
          {
            return lhs.curvature < rhs.curvature;
          }))
  {
    return false;
  }

  auto it_pcl = cloud.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); --it_kp)
  {
    const auto head = it_kp - 1;
    const auto tail = it_kp;
    M3D R_imu;
    R_imu << MAT_FROM_ARRAY(head->rot);
    V3D vel_imu;
    vel_imu << VEC_FROM_ARRAY(head->vel);
    V3D pos_imu;
    pos_imu << VEC_FROM_ARRAY(head->pos);
    V3D acc_imu;
    acc_imu << VEC_FROM_ARRAY(tail->acc);
    V3D angvel_avr;
    angvel_avr << VEC_FROM_ARRAY(tail->gyr);

    for (; it_pcl->curvature / double(1000) > head->offset_time; --it_pcl)
    {
      const double dt =
          it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the scan-end frame. This is the exact compensation used
       * by the normal FAST-LIO path and is intentionally shared with the
       * offline source-ray exporter so physical ray origins and hits cannot
       * drift apart. */
      const M3D R_i(R_imu * Exp(angvel_avr, dt));
      const V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      const V3D T_ei(
          pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt -
          frame_end_state.pos);
      const V3D P_compensate =
          frame_end_state.offset_R_L_I.conjugate() *
          (frame_end_state.rot.conjugate() *
               (R_i *
                    (frame_end_state.offset_R_L_I * P_i +
                     frame_end_state.offset_T_L_I) +
                T_ei) -
           frame_end_state.offset_T_L_I);

      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == cloud.points.begin())
        break;
    }
  }
  return true;
}

void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  assert(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      std::cout << "IMU Initial Done" << std::endl;
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
