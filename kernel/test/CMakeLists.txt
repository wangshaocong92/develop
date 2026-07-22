# ============================================================================
# kernel 单元测试
# ============================================================================

include_directories(../)

# GTest Conan 包仅提供 Release 配置;在 Debug 构建下直接指定库路径
set(GTEST_LIBS "${GTest_LIBRARIES}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND TARGET GTest::gtest)
  include("${CMAKE_CURRENT_BINARY_DIR}/GTest-release-x86_64-data.cmake" OPTIONAL)
  if(gtest_PACKAGE_FOLDER_RELEASE)
    set(GTEST_LIBS
        "${gtest_PACKAGE_FOLDER_RELEASE}/lib/libgtest.a"
        "${gtest_PACKAGE_FOLDER_RELEASE}/lib/libgtest_main.a")
  endif()
endif()

# cutlass / cute 头文件路径（softmax 与 attention 共用）
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/flash_attention/csrc/cutlass/include")

# ---- reduce 测试 ----
add_executable(reduce_test
    test_reduce.cu
)

target_link_libraries(reduce_test PRIVATE
    kernel
    ${GTEST_LIBS}
    CUDA::cudart
)

set_target_properties(reduce_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ---- device 测试 ----
add_executable(device_test
    test_device.cu
)

target_link_libraries(device_test PRIVATE
    kernel
    ${GTEST_LIBS}
    CUDA::cudart
)

target_include_directories(device_test PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(device_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ---- softmax 测试 ----
add_executable(softmax_test
    test_softmax.cpp
)

target_link_libraries(softmax_test PRIVATE
    ${GTEST_LIBS}
)

# ---- softmax GPU 测试 ----
add_executable(softmax_gpu_test
    test_softmax_gpu.cu
)

target_link_libraries(softmax_gpu_test PRIVATE
    ${GTEST_LIBS}
    CUDA::cudart
)

target_include_directories(softmax_gpu_test PRIVATE
    ${CUTLASS_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(softmax_gpu_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(softmax_gpu_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# ---- softmax benchmark (无 GTest, 直接跑) ----
add_executable(softmax_bench
    bench_softmax.cu
)

target_link_libraries(softmax_bench PRIVATE
    CUDA::cudart
)

target_include_directories(softmax_bench PRIVATE
    ${CUTLASS_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(softmax_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(softmax_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# ---- attention 测试 ----
add_executable(standard_attention_test
    test_standard_attention.cu
)

target_include_directories(standard_attention_test PRIVATE
    ${CUTLASS_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(standard_attention_test PRIVATE
    ${GTEST_LIBS}
    CUDA::cudart
    CUDA::cublas
)

set_target_properties(standard_attention_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(standard_attention_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# ---- attention benchmark ----
add_executable(standard_attention_bench
    bench_standard_attention.cu
)

target_include_directories(standard_attention_bench PRIVATE
    ${CUTLASS_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(standard_attention_bench PRIVATE
    CUDA::cudart
    CUDA::cublas
)

set_target_properties(standard_attention_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(standard_attention_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# ---- attention benchmark: standard vs flash ----
add_executable(attention_bench
    bench_attention.cu
)

target_include_directories(attention_bench PRIVATE
    ${CUTLASS_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(attention_bench PRIVATE
    CUDA::cudart
)

set_target_properties(attention_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(attention_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# ---- gemm multi-stage 测试 ----
add_executable(gemm_multi_stage_test
    test_gemm_multi_stage.cu
)

target_include_directories(gemm_multi_stage_test PRIVATE
    ${CUTLASS_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(gemm_multi_stage_test PRIVATE
    ${GTEST_LIBS}
    CUDA::cudart
)

set_target_properties(gemm_multi_stage_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(gemm_multi_stage_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# ---- flash attention 测试 ----
add_executable(flash_attention_test
    test_flash_attention.cu
)

target_include_directories(flash_attention_test PRIVATE
    ${CUTLASS_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(flash_attention_test PRIVATE
    ${GTEST_LIBS}
    CUDA::cudart
)

set_target_properties(flash_attention_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(flash_attention_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# ---- GTest 发现 ----
enable_testing()
include(GoogleTest)
gtest_discover_tests(reduce_test)
gtest_discover_tests(device_test)
gtest_discover_tests(softmax_test)
gtest_discover_tests(softmax_gpu_test)
gtest_discover_tests(standard_attention_test)
gtest_discover_tests(gemm_multi_stage_test)
gtest_discover_tests(flash_attention_test)
