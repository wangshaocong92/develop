cmake_minimum_required(VERSION 3.18)
project(main)
find_package(NvidiaCutlass REQUIRED)

find_package(CUDAToolkit REQUIRED COMPONENTS cublas)

include_directories(
    ${NvidiaCutlass_INCLUDE_DIRS}
)

add_executable(main main.cpp)

# target_link_libraries(gemm-multi-stage PRIVATE nvidia::cutlass::cutlass CUDA::cublas CUDA::cudart)


install(TARGETS main DESTINATION bin)