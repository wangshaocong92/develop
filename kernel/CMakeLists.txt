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
)
file(GLOB DIR_HEADER CONFIGURE_DEPENDS
    "*/*.cuh"
)

message(STATUS "DIR_SRCS: ${DIR_SRCS}")

add_library(kernel SHARED ${DIR_SRCS})

target_link_libraries(kernel CUDA::cudart)

# CUDA 设置
set_target_properties(kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(reduce_test
    reduce/test_reduce.cu # 或者 test_my_kernel.cpp
)


target_link_libraries(reduce_test PRIVATE ${GTest_LIBRARIES} kernel CUDA::cudart)

# CUDA 设置
set_target_properties(reduce_test PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON 
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)


# 5. 启用测试并发现测试用例
enable_testing()
# 这会自动发现并添加所有 TEST, TEST_F 等用例
include(GoogleTest)
gtest_discover_tests(reduce_test)

install(TARGETS kernel DESTINATION lib)
install(FILES ${DIR_HEADER} DESTINATION include/kernel)







