/******************************************************************************
 * Standard Attention 性能测试
 ******************************************************************************/

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include "attention/standard_attention/standard_attention.h"

using namespace cute;

static std::vector<float> make_data(int n) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dis(-1.f, 1.f);
  std::vector<float> v(n);
  for (auto &x : v) x = dis(rng);
  return v;
}

// CPU 参照实现: O = softmax(Q·K^T) · V, 行主序 [M,N]
static void cpu_attention(const float *q, const float *k, const float *v, float *out, int M,
                          int N) {
  std::vector<float> S(M * M);
  // S = Q · K^T
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
  // O = S · V
  for (int i = 0; i < M; ++i)
    for (int d = 0; d < N; ++d) {
      float acc = 0.f;
      for (int j = 0; j < M; ++j) acc += S[i * M + j] * v[j * N + d];
      out[i * N + d] = acc;
    }
}

// 计时一个可调用对象, 返回毫秒
template <class Fn>
static double time_ms(Fn &&fn) {
  auto t0 = std::chrono::high_resolution_clock::now();
  fn();
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

template <int M, int N>
static void bench_one(const std::string &label) {
  auto h_q = make_data(M * N);
  auto h_k = make_data(M * N);
  auto h_v = make_data(M * N);

  float *d_q, *d_k, *d_v, *d_out;
  cudaMalloc(&d_q, M * N * sizeof(float));
  cudaMalloc(&d_k, M * N * sizeof(float));
  cudaMalloc(&d_v, M * N * sizeof(float));
  cudaMalloc(&d_out, M * N * sizeof(float));
  cudaMemcpy(d_q, h_q.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, M * N * sizeof(float));

  // warm up cutlass
  static int first = 1;
  if (first) {
    first = 0;
    auto layout = Layout<Shape<Int<M>, Int<N>>, Stride<Int<1>, Int<M>>>{};
    auto tQ = make_tensor(d_q, layout);
    auto tK = make_tensor(d_k, layout);
    auto tV = make_tensor(d_v, layout);
    auto tO = make_tensor(d_out, layout);
    kernel::cpu::standard_attention_forward<M, N>(tQ, tK, tV, tO);
    cudaDeviceSynchronize();
  }

  // GPU
  double gpu_ms = time_ms([&] {
    auto layout = Layout<Shape<Int<M>, Int<N>>, Stride<Int<1>, Int<M>>>{};
    auto tQ = make_tensor(d_q, layout);
    auto tK = make_tensor(d_k, layout);
    auto tV = make_tensor(d_v, layout);
    auto tO = make_tensor(d_out, layout);
    kernel::cpu::standard_attention_forward<M, N>(tQ, tK, tV, tO);
    cudaDeviceSynchronize();
  });

  // CPU
  std::vector<float> h_out(M * N);
  double cpu_ms =
      time_ms([&] { cpu_attention(h_q.data(), h_k.data(), h_v.data(), h_out.data(), M, N); });

  double speedup = cpu_ms / gpu_ms;
  std::cout << "  " << std::left << std::setw(8) << label << std::right << " GPU " << std::setw(9)
            << std::fixed << std::setprecision(3) << gpu_ms << " ms | CPU " << std::setw(9)
            << cpu_ms << " ms | speedup " << std::setw(7) << std::setprecision(2) << speedup << "x"
            << std::endl;

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_out);
}

int main() {
  std::cout << "===== Standard Attention Benchmark =====\n" << std::endl;

  std::cout << "--- head_dim=32 ---" << std::endl;
  bench_one<64, 32>("M=64");
  bench_one<128, 32>("M=128");
  bench_one<256, 32>("M=256");
  bench_one<512, 32>("M=512");

  std::cout << "\n--- head_dim=64 ---" << std::endl;
  bench_one<64, 64>("M=64");
  bench_one<128, 64>("M=128");
  bench_one<256, 64>("M=256");
  bench_one<512, 64>("M=512");

  std::cout << "\n--- head_dim=128 ---" << std::endl;
  bench_one<64, 128>("M=64");
  bench_one<128, 128>("M=128");
  bench_one<256, 128>("M=256");
  bench_one<512, 128>("M=512");

  return 0;
}
