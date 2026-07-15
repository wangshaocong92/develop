/******************************************************************************
 * kernel/softmax/softmax.cuh GPU 单元测试
 *
 * 测试: softmax_max / softmax_sum / softmax_forward
 *       host_softmax_forward / host_online_softmax_forward
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

constexpr int BLK = 256;
constexpr int ITM = 4;
constexpr int CHK = BLK * ITM;

// ==========================================================================
// CPU 参照
// ==========================================================================
static float cpu_max(const float *x, int n) {
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) m = fmaxf(m, x[i]);
    return m;
}

static std::vector<float> cpu_softmax(const float *x, int n) {
    float m = cpu_max(x, n);
    std::vector<float> out(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) { out[i] = expf(x[i] - m); sum += out[i]; }
    for (auto &v : out) v = static_cast<float>(v / sum);
    return out;
}

static std::vector<float> make_data(int n, float lo = 1.f, float hi = 10.f) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dis(lo, hi);
    std::vector<float> v(n);
    for (auto &x : v) x = dis(rng);
    return v;
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; ++i) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

// ==========================================================================
// 1. softmax_max
// ==========================================================================
template <int N, int BT, int IT>
static void test_softmax_max() {
    int nb = (N + BT * IT - 1) / (BT * IT);
    auto h_in = make_data(N);
    float *d_in, *d_max;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max, nb * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // M=1 的 2D tensor（行主序），降维当作一维用
    auto layout = cute::Layout<cute::Shape<cute::Int<1>, cute::Int<N>>,
                               cute::Stride<cute::Int<N>, cute::Int<1>>>{};
    auto t = cute::make_tensor(d_in, layout);
    softmax_max<BT, IT, 1, N><<<nb, BT>>>(t, d_max);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(nb);
    CUDA_CHECK(cudaMemcpy(got.data(), d_max, nb * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> ref(nb);
    for (int b = 0; b < nb; ++b) {
        int s = b * BT * IT, e = std::min(s + BT * IT, N);
        ref[b] = -INFINITY;
        for (int i = s; i < e; ++i) ref[b] = fmaxf(ref[b], h_in[i]);
    }
    EXPECT_LT(max_abs_diff(got.data(), ref.data(), nb), 1e-4f);
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_max));
}

TEST(SoftmaxMax, N1024_B128_I4)  { test_softmax_max<1024, 128, 4>(); }
TEST(SoftmaxMax, N1024_B256_I4)  { test_softmax_max<1024, 256, 4>(); }
TEST(SoftmaxMax, N5000_B128_I4)  { test_softmax_max<5000, 128, 4>(); }
TEST(SoftmaxMax, N1_B128_I4)     { test_softmax_max<1,    128, 4>(); }

// ==========================================================================
// 2. softmax_sum
// ==========================================================================
template <int N, int BT, int IT>
static void test_softmax_sum() {
    int cb = BT * IT, nb = (N + cb - 1) / cb;
    auto h_in = make_data(N);
    float gmax = cpu_max(h_in.data(), N);

    float *d_in, *d_sum, *d_max;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, nb * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));  // M=1 的逐行 max
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max, &gmax, sizeof(float), cudaMemcpyHostToDevice));

    // M=1 的 2D tensor（行主序），降维当作一维用
    auto layout = cute::Layout<cute::Shape<cute::Int<1>, cute::Int<N>>,
                               cute::Stride<cute::Int<N>, cute::Int<1>>>{};
    auto t = cute::make_tensor(d_in, layout);
    softmax_sum<BT, IT, 1, N><<<nb, BT>>>(t, d_sum, d_max);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(nb);
    CUDA_CHECK(cudaMemcpy(got.data(), d_sum, nb * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> ref(nb);
    for (int b = 0; b < nb; ++b) {
        int s = b * cb, e = std::min(s + cb, N);
        double sum = 0.0;
        for (int i = s; i < e; ++i) sum += expf(h_in[i] - gmax);
        ref[b] = static_cast<float>(sum);
    }
    EXPECT_LT(max_abs_diff(got.data(), ref.data(), nb), 1e-2f);
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_sum)); CUDA_CHECK(cudaFree(d_max));
}

TEST(SoftmaxSum, N1024_B128_I4)  { test_softmax_sum<1024, 128, 4>(); }
TEST(SoftmaxSum, N1024_B256_I4)  { test_softmax_sum<1024, 256, 4>(); }
TEST(SoftmaxSum, N5000_B128_I4)  { test_softmax_sum<5000, 128, 4>(); }

// ==========================================================================
// 3. softmax_forward (normalize)
// ==========================================================================
template <int N, int BT, int IT>
static void test_softmax_forward() {
    auto h_in = make_data(N);
    float gmax = cpu_max(h_in.data(), N);
    double gsum = 0.0;
    for (int i = 0; i < N; ++i) gsum += expf(h_in[i] - gmax);

    float *d_in;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int nb = (N + BT * IT - 1) / (BT * IT);
    softmax_forward<BT, IT><<<nb, BT>>>(d_in, gsum, static_cast<double>(gmax), N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(N);
    CUDA_CHECK(cudaMemcpy(got.data(), d_in, N * sizeof(float), cudaMemcpyDeviceToHost));

    auto ref = cpu_softmax(h_in.data(), N);
    EXPECT_LT(max_abs_diff(got.data(), ref.data(), N), 1e-4f);
    CUDA_CHECK(cudaFree(d_in));
}

TEST(SoftmaxFwd, N1024_B128_I4)  { test_softmax_forward<1024, 128, 4>(); }
TEST(SoftmaxFwd, N5000_B128_I4)  { test_softmax_forward<5000, 128, 4>(); }

// ==========================================================================
// 4. host_softmax_forward (3-pass, 端到端)
// ==========================================================================
template <int N>
static void test_host_fwd() {
    auto h_in = make_data(N, -5.f, 5.f);
    auto ref = cpu_softmax(h_in.data(), N);

    float *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    host_softmax_forward<N>(d);
    std::vector<float> got(N);
    CUDA_CHECK(cudaMemcpy(got.data(), d, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));

    float err = max_abs_diff(got.data(), ref.data(), N);
    std::cout << "  host_3pass_err=" << err << std::endl;
    EXPECT_LT(err, 1e-2f);
}

TEST(HostSoftmaxFwd, N1024)  { test_host_fwd<1024>(); }
TEST(HostSoftmaxFwd, N5000)  { test_host_fwd<5000>(); }
TEST(HostSoftmaxFwd, N12345) { test_host_fwd<12345>(); }

// ==========================================================================
// 5. host_online_softmax_forward (2-pass, 端到端)
// ==========================================================================
static void test_host_online(int n) {
    auto h_in = make_data(n, -5.f, 5.f);
    auto ref = cpu_softmax(h_in.data(), n);

    float *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    host_online_softmax_forward(d, n);
    std::vector<float> got(n);
    CUDA_CHECK(cudaMemcpy(got.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));

    float err = max_abs_diff(got.data(), ref.data(), n);
    std::cout << "  host_2pass_err=" << err << std::endl;
    EXPECT_LT(err, 1e-2f);
}

TEST(HostOnlineSoftmaxFwd, N1024)  { test_host_online(1024); }
TEST(HostOnlineSoftmaxFwd, N5000)  { test_host_online(5000); }
TEST(HostOnlineSoftmaxFwd, N12345) { test_host_online(12345); }
