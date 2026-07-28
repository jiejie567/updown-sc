cmake_minimum_required(VERSION 3.16)
project(updown_sc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 REQUIRED)
# Some distro PCL/VTK cmake configs reference MPI targets even when the MPI
# functionality is unused; satisfy them without requiring an MPI install.
find_package(MPI QUIET)
foreach(mpi_target MPI::MPI_C MPI::MPI_CXX)
  if(NOT TARGET ${mpi_target})
    add_library(${mpi_target} INTERFACE IMPORTED)
  endif()
endforeach()
find_package(PCL REQUIRED COMPONENTS common io)
find_package(yaml-cpp REQUIRED)

add_library(updown_sc_core src/scan_context.cpp)
target_include_directories(updown_sc_core PUBLIC include ${PCL_INCLUDE_DIRS})
target_link_libraries(updown_sc_core PUBLIC Eigen3::Eigen ${PCL_LIBRARIES})

foreach(tool
    scan_context_rebuild
    scan_context_cross_sequence_evaluator
    scan_context_convert
    scan_context_subset)
  add_executable(${tool} src/tools/${tool}.cpp)
  target_link_libraries(${tool} PRIVATE updown_sc_core yaml-cpp)
endforeach()
