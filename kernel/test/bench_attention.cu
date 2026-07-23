/******************************************************************************
 * Attention 性能基准:standard_attention(CUTLASS, float) vs flash_attention
 * (multi-stage gemm, half 输入/F32 累加)。用 CUDA events 计时,多次迭代取均值。
 *
 * PyTorch 对比见同目录 bench_attention_torch.py(相同 M/N 规模)。
 *
 * 说明:
 *   - standard_attention 走 float 路径(CUTLASS SIMT),输入 (M,N) row-major。
 *   - flash_attention 走 half 路径,输入 Q/K (M,N)、V 需转置成 (N,M),
 *     且受 gemm tile 约束:q_step/kv_step 需满足 MMA 粒度,故规模用 N∈{64},
 *     M∈{256,512,1024},q_step=64,kv_step=32。
 ******************************************************************************/

#include <cstdio>
#include <chrono>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "attention/standard_attention/standard_attention.h"
#include "attention/flash_attention/flash_attention.h"

using namespace cute;
using cute::half_t;

static std::vector<float> make_data(int n) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dis(-1.f, 1.f);
  std::vector<float> v(n);
  for (auto &x : v) x = dis(rng);
  return v;
}

// 用 CUDA events 计时一个 GPU 可调用对象,warmup 后迭代 iters 次取均值(ms/次)。
template <class Fn>
static float time_gpu_ms(Fn &&fn, int iters = 50, int warmup = 10) {
  for (int i = 0; i < warmup; ++i) fn();
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iters; ++i) fn();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / iters;
}

// ----- standard attention(float, CUTLASS) -----
template <int M, int N>
static float bench_standard() {
  auto h_q = make_data(M * N), h_k = make_data(M * N), h_v = make_data(M * N);
  float *d_q, *d_k, *d_v, *d_out;
  cudaMalloc(&d_q, sizeof(float) * M * N);
  cudaMalloc(&d_k, sizeof(float) * M * N);
  cudaMalloc(&d_v, sizeof(float) * M * N);
  cudaMalloc(&d_out, sizeof(float) * M * N);
  cudaMemcpy(d_q, h_q.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, sizeof(float) * M * N);

  auto layout = Layout<Shape<Int<M>, Int<N>>, Stride<Int<1>, Int<M>>>{};
  auto run = [&] {
    auto tQ = make_tensor(d_q, layout);
    auto tK = make_tensor(d_k, layout);
    auto tV = make_tensor(d_v, layout);
    auto tO = make_tensor(d_out, layout);
    kernel::cpu::standard_attention_forward<M, N>(tQ, tK, tV, tO);
  };
  float ms = time_gpu_ms(run);

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_out);
  return ms;
}

// ----- flash attention(half 输入/F32 累加) -----
template <int M, int N, int q_step, int kv_step>
static float bench_flash() {
  auto h_q = make_data(M * N), h_k = make_data(M * N), h_v = make_data(M * N);
  std::vector<half_t> h_qh(M * N), h_kh(M * N), h_vth(N * M);
  for (int i = 0; i < M * N; ++i) {
    h_qh[i] = static_cast<half_t>(h_q[i]);
    h_kh[i] = static_cast<half_t>(h_k[i]);
  }
  // V (M,N) row-major → Vᵀ (N,M) row-major
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) h_vth[c * M + r] = static_cast<half_t>(h_v[r * N + c]);

  half_t *d_q, *d_k, *d_vt;
  float *d_out;
  cudaMalloc(&d_q, sizeof(half_t) * M * N);
  cudaMalloc(&d_k, sizeof(half_t) * M * N);
  cudaMalloc(&d_vt, sizeof(half_t) * N * M);
  cudaMalloc(&d_out, sizeof(float) * M * N);
  cudaMemcpy(d_q, h_qh.data(), sizeof(half_t) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_kh.data(), sizeof(half_t) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vt, h_vth.data(), sizeof(half_t) * N * M, cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, sizeof(float) * M * N);

  auto run = [&] {
    kernel::gpu::host_flash_attention_forward<M, N, q_step, kv_step>(d_q, d_k, d_vt, d_out);
  };
  float ms = time_gpu_ms(run);

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_vt);
  cudaFree(d_out);
  return ms;
}

// ----- flash attention 多卡(方案一分卡):setup 一次,只计 compute -----
// K/V/Q 一次性拷入各卡(构造),run() 只做 kernel + O 的 D2H + 同步,重复计时取均值。
// 剥离一次性 malloc/H2D(真实部署里数据常驻多卡、复用),看分片计算本身的加速。
template <int M, int N, int q_step, int kv_step>
static float bench_flash_multi(int *out_dev, int iters = 20, int warmup = 5) {
  auto h_q = make_data(M * N), h_k = make_data(M * N), h_v = make_data(M * N);
  std::vector<half_t> h_qh(M * N), h_kh(M * N), h_vth(N * M);
  for (int i = 0; i < M * N; ++i) {
    h_qh[i] = static_cast<half_t>(h_q[i]);
    h_kh[i] = static_cast<half_t>(h_k[i]);
  }
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) h_vth[c * M + r] = static_cast<half_t>(h_v[r * N + c]);

  std::vector<float> h_out(M * N);
  kernel::cpu::MultiGpuFlash<M, N, q_step, kv_step> mg(h_qh.data(), h_kh.data(), h_vth.data());
  *out_dev = mg.device_num();

  for (int i = 0; i < warmup; ++i) mg.run();
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) mg.run();
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

// 一行结果:standard / flash 单卡纯 kernel / flash 多卡分片计算(不含一次性拷贝)
template <int M, int N, int q_step, int kv_step>
static void bench_row() {
  float std_ms = bench_standard<M, N>();
  float fl_ms = bench_flash<M, N, q_step, kv_step>();
  int dev = 0;
  float multi_ms = bench_flash_multi<M, N, q_step, kv_step>(&dev);
  printf("  M=%-5d N=%-4d | standard %8.4f | flash-1gpu %8.4f | flash-%dgpu %8.4f ms (计算,不含拷贝)\n",
         M, N, std_ms, fl_ms, dev, multi_ms);
}

int main() {
  printf("===== Attention Benchmark (RTX 4090) =====\n");
  printf("standard=float CUTLASS(纯kernel), flash-1gpu=half 单卡纯kernel\n");
  printf("flash-Ngpu=多卡方案一分片计算(K/V/Q 已常驻各卡,只计 kernel+O回传+同步)\n");
  printf("N=128 = GPT-4 head_dim; M = 序列长度\n\n");

  bench_row<2048, 128, 64, 32>();
  bench_row<4096, 128, 64, 32>();
  bench_row<8192, 128, 64, 32>();
  bench_row<16384, 128, 64, 32>();

  printf("\n(PyTorch SDPA 对比: python3 kernel/test/bench_attention_torch.py)\n");
  return 0;
}
