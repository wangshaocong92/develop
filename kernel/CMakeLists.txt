cmake_minimum_required(VERSION 3.18)

project(kernel LANGUAGES CXX CUDA)


find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${GTest_INCLUDE_DIRS}
)

set(DIR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce/reduce.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/softmax/softmax.cpp
)
file(GLOB DIR_HEADER CONFIGURE_DEPENDS
    "*/*.cuh"
)

message(STATUS "DIR_SRCS: ${DIR_SRCS}")

add_library(kernel SHARED ${DIR_SRCS})

target_link_libraries(kernel CUDA::cudart)

# CUDA 设置
set_target_properties(kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

enable_testing()
add_subdirectory(test)

install(TARGETS kernel DESTINATION lib)
install(FILES ${DIR_HEADER} DESTINATION include/kernel)







