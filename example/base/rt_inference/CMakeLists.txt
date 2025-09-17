cmake_minimum_required(VERSION 3.5)
project(rt_inference)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)

find_package(OpenCV REQUIRED)
find_package(NumCpp REQUIRED)
find_package(glog REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}   ${CUDAToolkit_INCLUDE_DIRS} ${NumCpp_INCLUDE_DIRS} ${glog_INCLUDE_DIRS})




add_executable(rt_inference rt_from_rt.cpp)
target_link_libraries(rt_inference nvinfer ${OpenCV_LIBS} glog::glog NumCpp::NumCpp  CUDA::cudart)  

