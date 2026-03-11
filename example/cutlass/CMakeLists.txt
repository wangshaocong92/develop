cmake_minimum_required(VERSION 3.18)
project(cutlass_example)
find_package(NvidiaCutlass REQUIRED)

find_package(CUDAToolkit REQUIRED COMPONENTS cublas)

include_directories(
    ${NvidiaCutlass_INCLUDE_DIRS}
)

add_executable(cutlass_example main.cu)
target_link_libraries(cutlass_example PRIVATE nvidia::cutlass::cutlass)

add_executable(gemm gemm.cu)

target_link_libraries(gemm PRIVATE nvidia::cutlass::cutlass)

add_executable(gemm-multi-stage gemm-multi-stage.cu)

target_link_libraries(gemm-multi-stage PRIVATE nvidia::cutlass::cutlass CUDA::cublas CUDA::cudart)


install(TARGETS cutlass_example DESTINATION bin)
install(TARGETS gemm DESTINATION bin)
install(TARGETS gemm-multi-stage DESTINATION bin)