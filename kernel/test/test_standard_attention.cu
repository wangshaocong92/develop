/******************************************************************************
 * kernel/attention/standard_attention 单元测试
 ******************************************************************************/

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <cute/tensor.hpp>
#include "attention/standard_attention/standard_attention.h"

using namespace cute;

// ==========================================================================
// CPU 参照
// ==========================================================================
static void cpu_attention(const float *q, const float *k, const float *v, float *out, int M,
                          int N) {
  std::vector<float> S(M * M);
  // S = Q × K^T
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < M; ++j) {
      float acc = 0.f;
      for (int d = 0; d < N; ++d) acc += q[i * N + d] * k[j * N + d];
      S[i * M + j] = acc;
    }
  // softmax per row
  for (int i = 0; i < M; ++i) {
    float mx = -INFINITY;
    for (int j = 0; j < M; ++j) mx = fmaxf(mx, S[i * M + j]);
    double sum = 0.0;
    for (int j = 0; j < M; ++j) {
      S[i * M + j] = expf(S[i * M + j] - mx);
      sum += S[i * M + j];
    }
    for (int j = 0; j < M; ++j) S[i * M + j] /= static_cast<float>(sum);
  }
  // O = softmax(S) × V
  for (int i = 0; i < M; ++i)
    for (int d = 0; d < N; ++d) {
      float acc = 0.f;
      for (int j = 0; j < M; ++j) acc += S[i * M + j] * v[j * N + d];
      out[i * N + d] = acc;
    }
}

static std::vector<float> make_data(int n, float lo = -1.f, float hi = 1.f) {
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
// 模板化测试
// ==========================================================================
template <int M, int N>
static void test_attention() {
  auto h_q = make_data(M * N);
  auto h_k = make_data(M * N);
  auto h_v = make_data(M * N);

  std::vector<float> h_ref(M * N), h_out(M * N);
  cpu_attention(h_q.data(), h_k.data(), h_v.data(), h_ref.data(), M, N);

  // GPU (via Cutlass)
  float *d_q, *d_k, *d_v, *d_out;
  cudaMalloc(&d_q, M * N * sizeof(float));
  cudaMalloc(&d_k, M * N * sizeof(float));
  cudaMalloc(&d_v, M * N * sizeof(float));
  cudaMalloc(&d_out, M * N * sizeof(float));
  cudaMemcpy(d_q, h_q.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, M * N * sizeof(float));

  auto layout = Layout<Shape<Int<M>, Int<N>>, Stride<Int<1>, Int<M>>>{};
  auto tQ = make_tensor(d_q, layout);
  auto tK = make_tensor(d_k, layout);
  auto tV = make_tensor(d_v, layout);
  auto tO = make_tensor(d_out, layout);

  kernel::cpu::standard_attention_forward<M, N>(tQ, tK, tV, tO);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out.data(), d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_out);

  float err = max_abs_diff(h_out.data(), h_ref.data(), M * N);
  std::cout << "  M=" << M << " N=" << N << " err=" << err << std::endl;
  EXPECT_LT(err, 1e-2f);
}

TEST(StandardAttention, M16_N32) {
  test_attention<16, 32>();
}
TEST(StandardAttention, M32_N32) {
  test_attention<32, 32>();
}
TEST(StandardAttention, M32_N64) {
  test_attention<32, 64>();
}
TEST(StandardAttention, M64_N32) {
  test_attention<64, 32>();
}
TEST(StandardAttention, M64_N64) {
  test_attention<64, 64>();
}
TEST(StandardAttention, M128_N64) {
  test_attention<128, 64>();
}
