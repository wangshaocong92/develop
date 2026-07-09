# ============================================================================
# kernel 单元测试
# ============================================================================

include_directories(../)

# ---- reduce 测试 ----
add_executable(reduce_test
    test_reduce.cu
)

target_link_libraries(reduce_test PRIVATE
    kernel
    ${GTest_LIBRARIES}
    CUDA::cudart
)

set_target_properties(reduce_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ---- softmax 测试 ----
add_executable(softmax_test
    test_softmax.cpp
)

target_link_libraries(softmax_test PRIVATE
    ${GTest_LIBRARIES}
)

# ---- GTest 发现 ----
enable_testing()
include(GoogleTest)
gtest_discover_tests(reduce_test)
gtest_discover_tests(softmax_test)
