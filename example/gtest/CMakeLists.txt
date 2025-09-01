cmake_minimum_required(VERSION 3.18)
project(cuda_gtest_demo LANGUAGES CXX CUDA)

# 启用测试
enable_testing()

# 找 GoogleTest
find_package(GTest REQUIRED)

# 添加可执行文件
add_executable(cuda_gtest_demo main.cu)

# 打开 C++14
target_compile_features(cuda_gtest_demo PUBLIC cxx_std_14)

# CUDA 设置
set_target_properties(cuda_gtest_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接 GTest
target_link_libraries(cuda_gtest_demo
    GTest::gtest
    GTest::gtest_main
)

# 注册测试
include(GoogleTest)
gtest_discover_tests(cuda_gtest_demo)
