/******************************************************************************
 * kernel/softmax/softmax.cuh GPU 单元测试
 *
 * 测试 kernel::gpu::softmax_max 和 softmax_sum
 ******************************************************************************/

#include <cmath>
#include <random>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "softmax/softmax.cuh"

using namespace kernel::gpu;

#define CUDA_CHECK(call) \
    ASSERT_EQ(call, cudaSuccess) << cudaGetErrorString(call)

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

template <int N, int BLOCK_THREADS, int ITEMS_PER_THREAD>
static void run_softmax_max_test() {
    const int ITEMS_PER_BLOCK = BLOCK_THREADS * ITEMS_PER_THREAD;
    const int NUM_BLOCKS = (N + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;

    std::vector<float> h_input(N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(1.f, 10.f);
    for (auto &v : h_input) v = dis(gen);

    float *d_input = nullptr, *d_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max, NUM_BLOCKS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    softmax_max<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<NUM_BLOCKS, BLOCK_THREADS>>>(d_input, d_max, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_max(NUM_BLOCKS);
    CUDA_CHECK(cudaMemcpy(h_max.data(), d_max, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU ref
    std::vector<float> ref(NUM_BLOCKS, -std::numeric_limits<float>::infinity());
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        int start = b * ITEMS_PER_BLOCK;
        int end   = std::min(start + ITEMS_PER_BLOCK, N);
        for (int idx = start; idx < end; ++idx)
            ref[b] = std::max(ref[b], h_input[idx]);
    }

    float diff = max_abs_diff(h_max.data(), ref.data(), NUM_BLOCKS);
    std::cout << "   N=" << N << " BLK=" << BLOCK_THREADS
              << " I=" << ITEMS_PER_THREAD << " max_diff=" << diff << std::endl;
    EXPECT_LT(diff, 1e-4f);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_max));
}

TEST(SoftmaxMax, N1024_B128_I4)   { run_softmax_max_test<1024, 128, 4>(); }
TEST(SoftmaxMax, N1024_B256_I4)   { run_softmax_max_test<1024, 256, 4>(); }
TEST(SoftmaxMax, N1024_B128_I8)   { run_softmax_max_test<1024, 128, 8>(); }
TEST(SoftmaxMax, N5000_B128_I4)   { run_softmax_max_test<5000, 128, 4>(); }
TEST(SoftmaxMax, N1_B128_I4)      { run_softmax_max_test<1,    128, 4>(); }
