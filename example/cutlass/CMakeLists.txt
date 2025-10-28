cmake_minimum_required(VERSION 3.18)
project(cutlass_example)
find_package(NvidiaCutlass REQUIRED)


include_directories(
    ${NvidiaCutlass_INCLUDE_DIRS}
)

add_executable(cutlass_example main.cu)
target_link_libraries(cutlass_example PRIVATE nvidia::cutlass::cutlass)


install(TARGETS cutlass_example DESTINATION bin)