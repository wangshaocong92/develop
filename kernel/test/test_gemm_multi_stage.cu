/******************************************************************************
 * kernel/gemm/gemm_multi_stage 单元测试
 *
 * 验证从 doc/cute-gemm/gemm-multi-stage.cu 搬入库的 multi-stage GEMM
 * device 函数:D = A · Bᵀ (A:[M,K], B:[N,K], D:[M,N], 均 row-major),
 * 与 CPU 参考对拍。half 精度,阈值放宽。
 ******************************************************************************/

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include <cute/numeric/numeric_types.hpp>
#include <gtest/gtest.h>

#include "gemm/gemm_multi_stage.cuh"

using cute::half_t;

// D = A · Bᵀ 的 CPU 参考(float 累加)
static void cpu_gemm_abt(const std::vector<float> &A, const std::vector<float> &B,
                         std::vector<float> &D, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int p = 0; p < K; ++p) acc += A[i * K + p] * B[j * K + p];
      D[i * N + j] = acc;
    }
}

// D = A · B (B 不转置): A=(M,K) row-major, B=(K,N) row-major, D=(M,N)
static void cpu_gemm_ab(const std::vector<float> &A, const std::vector<float> &B,
                        std::vector<float> &D, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int p = 0; p < K; ++p) acc += A[i * K + p] * B[p * N + j];
      D[i * N + j] = acc;
    }
}

template <int M, int N, int K>
static void test_gemm_multi_stage() {
  using Config = kernel::config::GemmConfig<half_t, 128, 128, 32, 3>;
  static_assert(K % Config::kTileK == 0, "K 必须是 kTileK 的整数倍");

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dis(-1.f, 1.f);

  std::vector<float> hA(M * K), hB(N * K), hRef(M * N);
  for (auto &x : hA) x = dis(rng);
  for (auto &x : hB) x = dis(rng);
  cpu_gemm_abt(hA, hB, hRef, M, N, K);

  // float → half
  std::vector<half_t> hA_h(M * K), hB_h(N * K), hD_h(M * N);
  for (int i = 0; i < M * K; ++i) hA_h[i] = static_cast<half_t>(hA[i]);
  for (int i = 0; i < N * K; ++i) hB_h[i] = static_cast<half_t>(hB[i]);

  half_t *dA, *dB, *dD;
  cudaMalloc(&dA, sizeof(half_t) * M * K);
  cudaMalloc(&dB, sizeof(half_t) * N * K);
  cudaMalloc(&dD, sizeof(half_t) * M * N);
  cudaMemcpy(dA, hA_h.data(), sizeof(half_t) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB_h.data(), sizeof(half_t) * N * K, cudaMemcpyHostToDevice);
  cudaMemset(dD, 0, sizeof(half_t) * M * N);

  kernel::gpu::host_gemm_multi_stage<Config>(dD, dA, dB, M, N, K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  cudaMemcpy(hD_h.data(), dD, sizeof(half_t) * M * N, cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dD);

  float max_err = 0.f;
  for (int i = 0; i < M * N; ++i)
    max_err = std::fmax(max_err, std::fabs(static_cast<float>(hD_h[i]) - hRef[i]));

  std::cout << "  M=" << M << " N=" << N << " K=" << K << " max_err=" << max_err << std::endl;
  // half + K 累加,阈值随 K 放宽
  EXPECT_LT(max_err, 0.5f);
}

TEST(GemmMultiStage, M128_N128_K256) {
  test_gemm_multi_stage<128, 128, 256>();
}
TEST(GemmMultiStage, M256_N128_K128) {
  test_gemm_multi_stage<256, 128, 128>();
}
TEST(GemmMultiStage, M512_N256_K256) {
  test_gemm_multi_stage<512, 256, 256>();
}

// ---------------------------------------------------------------------------
// smem 输出路径:单 block 把 D=A·Bᵀ 结果停在 shared,再拷回 global 供校验。
// 验证 gemm_multi_stage_device 在 D 为 smem tensor 时走 if constexpr 分支。
// ---------------------------------------------------------------------------
template <typename Config>
__global__ void gemm_to_smem_kernel(void *Dptr, const void *Aptr, const void *Bptr, int m, int n,
                                    int k) {
  using namespace cute;
  using T = typename Config::T;
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;

  // 动态 smem:前半供 gemm 内部 A/B 多级暂存,尾部划出输出 tile
  // (extern __shared__ 必须与 gemm 内部同名同类型 char smem_shared[])
  extern __shared__ char smem_shared[];
  T *smem = reinterpret_cast<T *>(smem_shared);
  T *out_tile = smem + Config::shm_size_AB;

  // 本 block 的 A/B 行块 slab:M=kTileM、N=kTileN,故整个 A/B 即 slab
  Tensor gA_slab = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(Int<kTileM>{}, k),
                               make_stride(k, Int<1>{}));  // (kTileM, K)
  Tensor gB_slab = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(Int<kTileN>{}, k),
                               make_stride(k, Int<1>{}));  // (kTileN, K)

  // 输出 tile 作为 smem tensor:(kTileM, kTileN) row-major
  auto sD = make_tensor(make_smem_ptr(out_tile),
                        make_layout(make_shape(Int<kTileM>{}, Int<kTileN>{}),
                                    make_stride(Int<kTileN>{}, Int<1>{})));

  kernel::gpu::gemm_multi_stage_device<Config>(sD, gA_slab, gB_slab);

  // smem tile -> global,供 host 校验
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n), make_stride(n, Int<1>{}));
  for (int i = threadIdx.x; i < kTileM * kTileN; i += blockDim.x) {
    D(i / kTileN, i % kTileN) = sD(i / kTileN, i % kTileN);
  }
}

TEST(GemmMultiStage, SmemOutput_M128_N128_K256) {
  constexpr int M = 128, N = 128, K = 256;
  using Config = kernel::config::GemmConfig<half_t, 128, 128, 32, 3>;

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dis(-1.f, 1.f);
  std::vector<float> hA(M * K), hB(N * K), hRef(M * N);
  for (auto &x : hA) x = dis(rng);
  for (auto &x : hB) x = dis(rng);
  cpu_gemm_abt(hA, hB, hRef, M, N, K);

  std::vector<half_t> hA_h(M * K), hB_h(N * K), hD_h(M * N);
  for (int i = 0; i < M * K; ++i) hA_h[i] = static_cast<half_t>(hA[i]);
  for (int i = 0; i < N * K; ++i) hB_h[i] = static_cast<half_t>(hB[i]);

  half_t *dA, *dB, *dD;
  cudaMalloc(&dA, sizeof(half_t) * M * K);
  cudaMalloc(&dB, sizeof(half_t) * N * K);
  cudaMalloc(&dD, sizeof(half_t) * M * N);
  cudaMemcpy(dA, hA_h.data(), sizeof(half_t) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB_h.data(), sizeof(half_t) * N * K, cudaMemcpyHostToDevice);
  cudaMemset(dD, 0, sizeof(half_t) * M * N);

  // 动态 smem = A/B 暂存 + 输出 tile
  int shm = (Config::shm_size_AB + M * N) * sizeof(half_t);
  cudaFuncSetAttribute(gemm_to_smem_kernel<Config>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
  gemm_to_smem_kernel<Config><<<1, Config::kThreadNum, shm>>>(dD, dA, dB, M, N, K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  cudaMemcpy(hD_h.data(), dD, sizeof(half_t) * M * N, cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dD);

  float max_err = 0.f;
  for (int i = 0; i < M * N; ++i)
    max_err = std::fmax(max_err, std::fabs(static_cast<float>(hD_h[i]) - hRef[i]));
  std::cout << "  [smem] M=" << M << " N=" << N << " K=" << K << " max_err=" << max_err
            << std::endl;
  EXPECT_LT(max_err, 0.5f);
}

// ---------------------------------------------------------------------------
// F32 累加 + smem 输出:输入 half,MMA F32 累加,输出 float smem tile。
// 验证 ComputeType=float 分派(SM80..F32F16F16F32)与 float 输出的 epilogue。
// ---------------------------------------------------------------------------
template <typename Config>
__global__ void gemm_f32_to_smem_kernel(float *Dptr, const half_t *Aptr, const half_t *Bptr,
                                        int m, int n, int k) {
  using namespace cute;
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;

  // A/B 暂存是 half;输出 tile 是 float。两段 smem 分开摆放。
  extern __shared__ char smem_shared[];
  half_t *ab = reinterpret_cast<half_t *>(smem_shared);
  float *out_tile = reinterpret_cast<float *>(smem_shared + Config::shm_size_AB * sizeof(half_t));

  Tensor gA_slab = make_tensor(make_gmem_ptr(Aptr), make_shape(Int<kTileM>{}, k),
                               make_stride(k, Int<1>{}));
  Tensor gB_slab = make_tensor(make_gmem_ptr(Bptr), make_shape(Int<kTileN>{}, k),
                               make_stride(k, Int<1>{}));
  auto sD = make_tensor(make_smem_ptr(out_tile),
                        make_layout(make_shape(Int<kTileM>{}, Int<kTileN>{}),
                                    make_stride(Int<kTileN>{}, Int<1>{})));

  // gemm 内部的 extern smem 也用同一块起始地址(shm_data 是 half),故传同一 base。
  kernel::gpu::gemm_multi_stage_device<Config>(sD, gA_slab, gB_slab);

  Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n), make_stride(n, Int<1>{}));
  for (int i = threadIdx.x; i < kTileM * kTileN; i += blockDim.x)
    D(i / kTileN, i % kTileN) = sD(i / kTileN, i % kTileN);
}

TEST(GemmMultiStage, F32SmemOutput_M64_N64_K256) {
  // 64x64 tile:float 输出 tile(16KB)+A/B 暂存,总 smem 在 RTX4090 99KB 上限内。
  // (128x128 float 输出会超 smem 上限导致 launch 失败——印证重型 tile 不适配小场景)
  constexpr int M = 64, N = 64, K = 256;
  using Config = kernel::config::GemmConfig<half_t, 64, 64, 32, 3, 2,
                                            kernel::config::GemmMode::kABt, float>;

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dis(-1.f, 1.f);
  std::vector<float> hA(M * K), hB(N * K), hRef(M * N);
  for (auto &x : hA) x = dis(rng);
  for (auto &x : hB) x = dis(rng);
  cpu_gemm_abt(hA, hB, hRef, M, N, K);

  std::vector<half_t> hA_h(M * K), hB_h(N * K);
  std::vector<float> hD_f(M * N);
  for (int i = 0; i < M * K; ++i) hA_h[i] = static_cast<half_t>(hA[i]);
  for (int i = 0; i < N * K; ++i) hB_h[i] = static_cast<half_t>(hB[i]);

  half_t *dA, *dB;
  float *dD;
  cudaMalloc(&dA, sizeof(half_t) * M * K);
  cudaMalloc(&dB, sizeof(half_t) * N * K);
  cudaMalloc(&dD, sizeof(float) * M * N);
  cudaMemcpy(dA, hA_h.data(), sizeof(half_t) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB_h.data(), sizeof(half_t) * N * K, cudaMemcpyHostToDevice);
  cudaMemset(dD, 0, sizeof(float) * M * N);

  int shm = Config::shm_size_AB * sizeof(half_t) + M * N * sizeof(float);
  ASSERT_EQ(cudaFuncSetAttribute(gemm_f32_to_smem_kernel<Config>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, shm),
            cudaSuccess);
  gemm_f32_to_smem_kernel<Config><<<1, Config::kThreadNum, shm>>>(dD, dA, dB, M, N, K);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  cudaMemcpy(hD_f.data(), dD, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dD);

  float max_err = 0.f;
  for (int i = 0; i < M * N; ++i) max_err = std::fmax(max_err, std::fabs(hD_f[i] - hRef[i]));
  std::cout << "  [f32 smem] M=" << M << " N=" << N << " K=" << K << " max_err=" << max_err
            << std::endl;
  EXPECT_LT(max_err, 0.5f);
}

// ---------------------------------------------------------------------------
// A·B 变体(B 不转置)+ F32 累加 + smem 输出:验证 GemmMode::kAB 路径。
// A=(kTileM,K) row-major, B=(K,kTileN) row-major, D=(kTileM,kTileN)。
// ---------------------------------------------------------------------------
template <typename Config>
__global__ void gemm_ab_to_smem_kernel(float *Dptr, const half_t *Aptr, const half_t *Bptr,
                                       int m, int n, int k) {
  using namespace cute;
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;

  extern __shared__ char smem_shared[];
  float *out_tile = reinterpret_cast<float *>(smem_shared + Config::shm_size_AB * sizeof(half_t));

  // A slab (kTileM,K) row-major
  Tensor gA_slab = make_tensor(make_gmem_ptr(Aptr), make_shape(Int<kTileM>{}, k),
                               make_stride(k, Int<1>{}));
  // B slab (K,kTileN) row-major —— A·B 不转置,B 的内层连续维是 N
  // A·B 模式:gemm 内部 B slab 物理为 (kTileN, K) row-major(即数学 B=(K,N) 的转置 Bᵀ),
  // MMA 读取阶段再逻辑转置回 (K,N)。故此处按 (N,K) 传入。
  Tensor gB_slab = make_tensor(make_gmem_ptr(Bptr), make_shape(Int<kTileN>{}, k),
                               make_stride(k, Int<1>{}));
  auto sD = make_tensor(make_smem_ptr(out_tile),
                        make_layout(make_shape(Int<kTileM>{}, Int<kTileN>{}),
                                    make_stride(Int<kTileN>{}, Int<1>{})));

  kernel::gpu::gemm_multi_stage_device<Config>(sD, gA_slab, gB_slab);

  Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n), make_stride(n, Int<1>{}));
  for (int i = threadIdx.x; i < kTileM * kTileN; i += blockDim.x)
    D(i / kTileN, i % kTileN) = sD(i / kTileN, i % kTileN);
}

TEST(GemmMultiStage, ABVariant_F32_M64_N64_K256) {
  constexpr int M = 64, N = 64, K = 256;
  using Config = kernel::config::GemmConfig<half_t, 64, 64, 32, 3, 2,
                                            kernel::config::GemmMode::kAB, float>;

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dis(-1.f, 1.f);
  std::vector<float> hA(M * K), hB(K * N), hRef(M * N);
  for (auto &x : hA) x = dis(rng);
  for (auto &x : hB) x = dis(rng);
  cpu_gemm_ab(hA, hB, hRef, M, N, K);

  std::vector<half_t> hA_h(M * K), hBt_h(N * K);  // hBt = Bᵀ, (N,K) row-major
  std::vector<float> hD_f(M * N);
  for (int i = 0; i < M * K; ++i) hA_h[i] = static_cast<half_t>(hA[i]);
  // 数学 B=(K,N) row-major → 转置存成 (N,K):hBt[n,p] = B[p,n]
  for (int p = 0; p < K; ++p)
    for (int nn = 0; nn < N; ++nn) hBt_h[nn * K + p] = static_cast<half_t>(hB[p * N + nn]);

  half_t *dA, *dB;
  float *dD;
  cudaMalloc(&dA, sizeof(half_t) * M * K);
  cudaMalloc(&dB, sizeof(half_t) * K * N);
  cudaMalloc(&dD, sizeof(float) * M * N);
  cudaMemcpy(dA, hA_h.data(), sizeof(half_t) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hBt_h.data(), sizeof(half_t) * N * K, cudaMemcpyHostToDevice);
  cudaMemset(dD, 0, sizeof(float) * M * N);

  int shm = Config::shm_size_AB * sizeof(half_t) + M * N * sizeof(float);
  ASSERT_EQ(cudaFuncSetAttribute(gemm_ab_to_smem_kernel<Config>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, shm),
            cudaSuccess);
  gemm_ab_to_smem_kernel<Config><<<1, Config::kThreadNum, shm>>>(dD, dA, dB, M, N, K);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  cudaMemcpy(hD_f.data(), dD, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dD);

  float max_err = 0.f;
  for (int i = 0; i < M * N; ++i) max_err = std::fmax(max_err, std::fabs(hD_f[i] - hRef[i]));
  std::cout << "  [A*B f32] M=" << M << " N=" << N << " K=" << K << " max_err=" << max_err
            << std::endl;
  EXPECT_LT(max_err, 0.5f);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}