cmake_minimum_required(VERSION 3.8)
project(fast_lio)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0144 NEW)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message("Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
set(FAST_LIO_SELECTED_MP_PROC_NUM 8)
add_definitions(-DMP_EN)
add_definitions(-DMP_PROC_NUM=${FAST_LIO_SELECTED_MP_PROC_NUM})
message("core for MP: ${FAST_LIO_SELECTED_MP_PROC_NUM}")

find_package(OpenMP REQUIRED)

# ROS dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(yaml-cpp REQUIRED)

# CI/Docker: PCL→VTK pulls FindMPI with broken /opt/hpcx paths on some images. Do not depend on
# pcl_ros: its ament export links VTK::ChartsCore (requires full VTK). This package only needs
# pcl_conversions + selected PCL modules. Skip VTK and pre-find PCL so pcl_conversions does not expand PCL.
set(CMAKE_DISABLE_FIND_PACKAGE_VTK TRUE)
find_package(PCL REQUIRED COMPONENTS common io filters kdtree registration)

find_package(pcl_conversions REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(dependencies
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  tf2_ros
  rosbag2_cpp
  rosbag2_storage
  pcl_conversions
  livox_ros_driver2
)

# Thirdparty libraries
find_package(Eigen3 REQUIRED)

message(Eigen: ${EIGEN3_INCLUDE_DIR})
message(STATUS "PCL: ${PCL_INCLUDE_DIRS}")

set(msg_files
  "msg/Pose6D.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
)
ament_export_dependencies(rosidl_default_runtime)

add_library(fast_lio_scan_context STATIC src/scan_context.cpp)
target_include_directories(fast_lio_scan_context PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(fast_lio_scan_context ${PCL_LIBRARIES} Eigen3::Eigen OpenMP::OpenMP_CXX)

add_library(fast_lio_prior_icp STATIC src/prior_icp.cpp)
target_include_directories(fast_lio_prior_icp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(fast_lio_prior_icp ${PCL_LIBRARIES} Eigen3::Eigen OpenMP::OpenMP_CXX)

add_library(fast_lio_pcd_save STATIC src/pcd_save_utils.cpp)
target_include_directories(fast_lio_pcd_save PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(fast_lio_pcd_save ${PCL_LIBRARIES})

add_executable(fastlio_mapping
  src/laserMapping.cpp
  src/source_ray_export.cpp
  include/ikd-Tree/ikd_Tree.cpp
  src/preprocess.cpp)
target_include_directories(fastlio_mapping PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(fastlio_mapping ${PCL_LIBRARIES} Eigen3::Eigen fast_lio_scan_context fast_lio_prior_icp fast_lio_pcd_save OpenMP::OpenMP_CXX)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(fastlio_mapping ${cpp_typesupport_target})

ament_target_dependencies(fastlio_mapping ${dependencies})

add_executable(offline_relocalization_exporter src/offline_relocalization_exporter.cpp src/preprocess.cpp)
target_include_directories(offline_relocalization_exporter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(offline_relocalization_exporter ${PCL_LIBRARIES} Eigen3::Eigen fast_lio_scan_context fast_lio_prior_icp OpenMP::OpenMP_CXX yaml-cpp)
ament_target_dependencies(offline_relocalization_exporter ${dependencies} ament_index_cpp)

add_executable(scan_context_convert src/scan_context_convert.cpp)
target_link_libraries(scan_context_convert fast_lio_scan_context)

add_executable(scan_context_subset src/scan_context_subset.cpp)
target_link_libraries(scan_context_subset fast_lio_scan_context)

add_executable(scan_context_cross_sequence_evaluator
  src/scan_context_cross_sequence_evaluator.cpp)
target_link_libraries(scan_context_cross_sequence_evaluator
  fast_lio_scan_context)

add_executable(scan_context_rebuild src/scan_context_rebuild.cpp)
target_link_libraries(scan_context_rebuild
  fast_lio_scan_context yaml-cpp)

add_executable(offline_icp_budget_evaluator src/offline_icp_budget_evaluator.cpp)
target_include_directories(offline_icp_budget_evaluator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(offline_icp_budget_evaluator
  ${PCL_LIBRARIES} Eigen3::Eigen fast_lio_scan_context fast_lio_prior_icp OpenMP::OpenMP_CXX)

if(BUILD_TESTING)
  enable_testing()
  add_executable(scan_context_mask_test test/scan_context_mask_test.cpp)
  target_link_libraries(scan_context_mask_test fast_lio_scan_context)
  add_test(NAME scan_context_mask_test COMMAND scan_context_mask_test)
endif()

# ---------------- Install --------------- #
install(TARGETS fastlio_mapping offline_relocalization_exporter scan_context_convert
  scan_context_subset
  scan_context_cross_sequence_evaluator scan_context_rebuild
  offline_icp_budget_evaluator
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config launch rviz prior_map
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(
  PROGRAMS scripts/plot_offline_relocalization.py
  DESTINATION share/${PROJECT_NAME}/scripts
)

install(
  PROGRAMS
    scripts/export_btc_submaps.py
    scripts/merge_btc_submap_exports.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
