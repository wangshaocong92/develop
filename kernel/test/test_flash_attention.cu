/******************************************************************************
 * kernel/attention/flash_attention 单元测试
 *
 * 验证单-block 多-stage GEMM 组合成的 flash attention forward:
 *   half 输入 Q/K/V, float 累加/输出, online softmax, 无 1/sqrt(d) 缩放。
 *   与 CPU 参考对拍,误差阈值 1e-2。
 ******************************************************************************/

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include <cute/numeric/numeric_types.hpp>
#include <gtest/gtest.h>

#include "attention/flash_attention/flash_attention.h"

using cute::half_t;

// ==========================================================================
// CPU 参考 (标准 attention: S=Q·Kᵀ, 逐行 softmax, O=P·V, 无 1/sqrt(d) 缩放)
// ==========================================================================
static void cpu_attention(const float *q, const float *k, const float *v, float *out, int M,
                          int N) {
  std::vector<float> S(static_cast<size_t>(M) * M);
  // S = Q × K^T
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < M; ++j) {
      float acc = 0.f;
      for (int d = 0; d < N; ++d) acc += q[i * N + d] * k[j * N + d];
      S[static_cast<size_t>(i) * M + j] = acc;
    }
  // softmax per row
  for (int i = 0; i < M; ++i) {
    float mx = -INFINITY;
    for (int j = 0; j < M; ++j) mx = fmaxf(mx, S[static_cast<size_t>(i) * M + j]);
    double sum = 0.0;
    for (int j = 0; j < M; ++j) {
      S[static_cast<size_t>(i) * M + j] = expf(S[static_cast<size_t>(i) * M + j] - mx);
      sum += S[static_cast<size_t>(i) * M + j];
    }
    for (int j = 0; j < M; ++j) S[static_cast<size_t>(i) * M + j] /= static_cast<float>(sum);
  }
  // O = softmax(S) × V
  for (int i = 0; i < M; ++i)
    for (int d = 0; d < N; ++d) {
      float acc = 0.f;
      for (int j = 0; j < M; ++j) acc += S[static_cast<size_t>(i) * M + j] * v[j * N + d];
      out[i * N + d] = acc;
    }
}

static float max_abs_diff(const float *a, const float *b, int n) {
  float m = 0.f;
  for (int i = 0; i < n; ++i) m = fmaxf(m, fabsf(a[i] - b[i]));
  return m;
}

// ==========================================================================
// Flash Attention 测试主体
// ==========================================================================
template <int M, int N, int q_step, int kv_step>
static void test_flash_attention() {
  static_assert(M % q_step == 0, "M must be divisible by q_step");

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dis(-1.f, 1.f);

  // host data (float for CPU reference)
  std::vector<float> h_qf(M * N), h_kf(M * N), h_vf(M * N), h_ref(M * N);
  for (auto &x : h_qf) x = dis(rng);
  for (auto &x : h_kf) x = dis(rng);
  for (auto &x : h_vf) x = dis(rng);

  cpu_attention(h_qf.data(), h_kf.data(), h_vf.data(), h_ref.data(), M, N);

  // convert to half for GPU
  std::vector<half_t> h_qh(M * N), h_kh(M * N), h_vh(M * N);
  for (int i = 0; i < M * N; ++i) {
    h_qh[i] = static_cast<half_t>(h_qf[i]);
    h_kh[i] = static_cast<half_t>(h_kf[i]);
    h_vh[i] = static_cast<half_t>(h_vf[i]);
  }

  // V_T: transpose V to (N, M) row-major
  std::vector<half_t> h_vth(static_cast<size_t>(N) * M);
  for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m) {
      h_vth[static_cast<size_t>(n) * M + m] = h_vh[m * N + n];
    }

  // device allocation
  half_t *d_q, *d_k, *d_vt, *d_p_scratch;
  float *d_out;
  cudaMalloc(&d_q, sizeof(half_t) * M * N);
  cudaMalloc(&d_k, sizeof(half_t) * M * N);
  cudaMalloc(&d_vt, sizeof(half_t) * N * M);
  cudaMalloc(&d_p_scratch, sizeof(half_t) * M * M);
  cudaMalloc(&d_out, sizeof(float) * M * N);

  cudaMemcpy(d_q, h_qh.data(), sizeof(half_t) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_kh.data(), sizeof(half_t) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vt, h_vth.data(), sizeof(half_t) * N * M, cudaMemcpyHostToDevice);
  cudaMemset(d_p_scratch, 0, sizeof(half_t) * M * M);
  cudaMemset(d_out, 0, sizeof(float) * M * N);

  // launch
  kernel::gpu::host_flash_attention_forward<M, N, q_step, kv_step>(d_q, d_k, d_vt,
                                                                     d_p_scratch, d_out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float> h_out(M * N);
  cudaMemcpy(h_out.data(), d_out, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_vt);
  cudaFree(d_p_scratch);
  cudaFree(d_out);

  float err = max_abs_diff(h_out.data(), h_ref.data(), M * N);
  std::cout << "  M=" << M << " N=" << N << " q_step=" << q_step << " kv_step=" << kv_step
            << " err=" << err << std::endl;
  EXPECT_LT(err, 1e-2f);
}

// ==========================================================================
// 测试用例
// ==========================================================================

// Debug test: verify S=Q·Kᵀ for first KV block
TEST(FlashAttentionForward, DebugGemm1) {
  constexpr int M = 256, N = 64, q_step = 64, kv_step = 32;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dis(-1.f, 1.f);

  std::vector<float> h_qf(M * N), h_kf(M * N);
  for (auto &x : h_qf) x = dis(rng);
  for (auto &x : h_kf) x = dis(rng);

  std::vector<half_t> h_qh(M * N), h_kh(M * N);
  for (int i = 0; i < M * N; ++i) {
    h_qh[i] = static_cast<half_t>(h_qf[i]);
    h_kh[i] = static_cast<half_t>(h_kf[i]);
  }

  half_t *d_q, *d_k;
  float *d_out;
  cudaMalloc(&d_q, sizeof(half_t) * M * N);
  cudaMalloc(&d_k, sizeof(half_t) * M * N);
  cudaMalloc(&d_out, sizeof(float) * M * kv_step);
  cudaMemcpy(d_q, h_qh.data(), sizeof(half_t) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_kh.data(), sizeof(half_t) * M * N, cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, sizeof(float) * M * kv_step);

  // Compute first block's S on GPU
  using Gemm1Config = kernel::config::GemmConfig<half_t, q_step, kv_step, N, 2, 2, kernel::config::GemmMode::kABt, float>;
  constexpr int shm = Gemm1Config::shm_size_AB * sizeof(half_t) + q_step * kv_step * sizeof(float);
  dim3 block(Gemm1Config::kThreadNum);
  dim3 grid(1);  // Only launch block 0
  cudaFuncSetAttribute(kernel::gpu::flash_attention_debug_gemm1<M, N, q_step, kv_step>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
  kernel::gpu::flash_attention_debug_gemm1<M, N, q_step, kv_step><<<grid, block, shm>>>(d_q, d_k, d_out);
  cudaDeviceSynchronize();

  std::vector<float> h_gpu(q_step * kv_step);
  cudaMemcpy(h_gpu.data(), d_out, sizeof(float) * q_step * kv_step, cudaMemcpyDeviceToHost);

  // Compute CPU reference S for first block
  std::vector<float> h_cpu(q_step * kv_step);
  for (int i = 0; i < q_step; ++i)
    for (int j = 0; j < kv_step; ++j) {
      float acc = 0.f;
      for (int d = 0; d < N; ++d) acc += h_qf[i * N + d] * h_kf[j * N + d];
      h_cpu[i * kv_step + j] = acc;
    }

  float err = max_abs_diff(h_gpu.data(), h_cpu.data(), q_step * kv_step);
  std::cout << "  DebugGemm1 err=" << err << std::endl;
  for (int i = 0; i < 5; ++i) {
    std::cout << "    gpu[" << i << "]=" << h_gpu[i] << " cpu[" << i << "]=" << h_cpu[i] << std::endl;
  }
  EXPECT_LT(err, 1e-2f);

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_out);
}

TEST(FlashAttentionForward, M256_N64_q64_kv32) {
  test_flash_attention<256, 64, 64, 32>();
}

TEST(FlashAttentionForward, M512_N64_q64_kv32) {
  test_flash_attention<512, 64, 64, 32>();
}

// ==========================================================================
// 多卡 flash attention(方案1:按 Q 行切分 + K/V 复制)对拍 CPU 参考。
// 单卡机上也应通过(全部行块归卡 0)。
// ==========================================================================
template <int M, int N, int q_step, int kv_step>
static void test_flash_attention_multi_gpu() {
  static_assert(M % q_step == 0, "M must be divisible by q_step");

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dis(-1.f, 1.f);
  std::vector<float> h_qf(M * N), h_kf(M * N), h_vf(M * N), h_ref(M * N);
  for (auto &x : h_qf) x = dis(rng);
  for (auto &x : h_kf) x = dis(rng);
  for (auto &x : h_vf) x = dis(rng);
  cpu_attention(h_qf.data(), h_kf.data(), h_vf.data(), h_ref.data(), M, N);

  // host half 输入:q/k (M,N),vt = Vᵀ (N,M)
  std::vector<half_t> h_qh(M * N), h_kh(M * N), h_vth(static_cast<size_t>(N) * M);
  for (int i = 0; i < M * N; ++i) {
    h_qh[i] = static_cast<half_t>(h_qf[i]);
    h_kh[i] = static_cast<half_t>(h_kf[i]);
  }
  for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m) h_vth[static_cast<size_t>(n) * M + m] = static_cast<half_t>(h_vf[m * N + n]);

  std::vector<float> h_out(M * N, 0.f);
  // 全 host 指针接口:内部自行分卡、复制、算、收回
  kernel::cpu::flash_attention_forward<M, N, q_step, kv_step>(h_qh.data(), h_kh.data(),
                                                              h_vth.data(), h_out.data());

  int dev = 0;
  cudaGetDeviceCount(&dev);
  float err = max_abs_diff(h_out.data(), h_ref.data(), M * N);
  std::cout << "  [multi-gpu] M=" << M << " N=" << N << " devices=" << dev << " err=" << err
            << std::endl;
  EXPECT_LT(err, 1e-2f);
}

TEST(FlashAttentionMultiGpu, M512_N64_q64_kv32) {
  test_flash_attention_multi_gpu<512, 64, 64, 32>();
}
// 块数为奇数:验证不整除时前若干卡多担 1 块
TEST(FlashAttentionMultiGpu, M448_N64_q64_kv32) {
  test_flash_attention_multi_gpu<448, 64, 64, 32>();
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
