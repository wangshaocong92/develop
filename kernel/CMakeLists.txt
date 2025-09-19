cmake_minimum_required(VERSION 3.18)

project(kernel LANGUAGES CXX CUDA)


find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${GTest_INCLUDE_DIRS}
)

set(DIR_SRCS
    ./reduce/reduce_with_divergent_warps.cu
)
file(GLOB DIR_HEADER CONFIGURE_DEPENDS
    "*/*.h"
    "*/*.hpp"
    "*.h"
    "*.hpp"
)

message(STATUS "Source files: ${DIR_SRCS}")

add_library(kernel SHARED ${DIR_SRCS})

target_link_libraries(kernel CUDA::cudart)



add_executable(reduce_test
    reduce/test_reduce.cu # 或者 test_my_kernel.cpp
)


target_link_libraries(reduce_test ${GTest_LIBRARIES} kernel CUDA::cudart)


# 5. 启用测试并发现测试用例
enable_testing()
# 这会自动发现并添加所有 TEST, TEST_F 等用例
include(GoogleTest)
include(CTest) 
gtest_discover_tests(reduce_test)



install(TARGETS kernel DESTINATION lib)
install(FILES ${DIR_HEADER} DESTINATION include/kernel)







