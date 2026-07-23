#pragma once

#include <cutlass/numeric_types.h>
#include <cstddef>
#include <cute/tensor.hpp>
#include <vector>

#include "gemm/gemm_multi_stage.cuh"
#include "gemm/gemm_rs.cuh"

namespace kernel {
using namespace cute;
namespace gpu {

// Debug kernel: only compute first S=Q·Kᵀ tile and write back
// (unchanged)
template <int M, int N, int q_step, int kv_step>
__global__ void flash_attention_debug_gemm1(const half_t *q_ptr, const half_t *k_ptr,
                                            float *out_ptr) {
  using Gemm1Config =
      config::GemmConfig<half_t, q_step, kv_step, N, 1, 2, config::GemmMode::kABt, float>;

  constexpr int shm_size_AB = Gemm1Config::shm_size_AB;

  extern __shared__ char smem_raw[];
  half_t *ab_stage = reinterpret_cast<half_t *>(smem_raw);
  float *s_buf = reinterpret_cast<float *>(smem_raw + shm_size_AB * sizeof(half_t));

  const int q_start = blockIdx.x * q_step;

  auto gQ_slab =
      make_tensor(make_gmem_ptr(q_ptr + q_start * N),
                  make_shape(Int<q_step>{}, Int<N>{}), make_stride(N, Int<1>{}));

  auto s_tensor = make_tensor(make_smem_ptr(s_buf),
                              make_layout(make_shape(Int<q_step>{}, Int<kv_step>{}),
                                          make_stride(Int<kv_step>{}, Int<1>{})));

  // Only first kv block
  auto gK_slab =
      make_tensor(make_gmem_ptr(k_ptr),
                  make_shape(Int<kv_step>{}, Int<N>{}), make_stride(N, Int<1>{}));

  gemm_multi_stage_device<Gemm1Config>(s_tensor, gQ_slab, gK_slab, ab_stage);
  __syncthreads();

  // Write S back to output
  auto out_slab =
      make_tensor(make_gmem_ptr(out_ptr + q_start * kv_step),
                  make_shape(Int<q_step>{}, Int<kv_step>{}),
                  make_stride(Int<kv_step>{}, Int<1>{}));

  for (int idx = threadIdx.x; idx < q_step * kv_step; idx += blockDim.x) {
    int qi = idx / kv_step;
    int kj = idx % kv_step;
    out_slab(qi, kj) = s_tensor(qi, kj);
  }
}

// ============================================================================
// flash_attention_forward 的各阶段 device helper(纯拆分,逻辑与原内联版等价)。
// ============================================================================

// 阶段1:online softmax。读 smem 里的 S(q_step×kv_step),对本 kv 块:
//   逐行(warp 协作 + shuffle 归约)求 max、更新 running m/l、把 P=exp(S-m_new)
//   写入 smem sP,并把本轮的逐行 rescale 因子 scale 存进 scale_arr。
//   m_state/l_state/scale_arr 是长度 q_step 的 smem 数组。
template <int q_step, int kv_step, class STensor, class PTensor>
__device__ __forceinline__ void fa_online_softmax(const STensor &s_tensor, PTensor &sP,
                                                  float *m_state, float *l_state,
                                                  float *scale_arr) {
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int n_warps = blockDim.x / 32;
  for (int qi = warp_id; qi < q_step; qi += n_warps) {
    const float m_old = m_state[qi];

    // 行内 max(各 lane 跨步 + warp 归约)
    float m_cur = -INFINITY;
    for (int kj = lane; kj < kv_step; kj += 32) m_cur = fmaxf(m_cur, s_tensor(qi, kj));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      m_cur = fmaxf(m_cur, __shfl_xor_sync(0xffffffff, m_cur, off));
    const float m_new = fmaxf(m_old, m_cur);
    const float scale = __expf(m_old - m_new);

    // 行内 sum(exp(s-m_new)) + 写 P 到 smem
    float l_part = 0.f;
    for (int kj = lane; kj < kv_step; kj += 32) {
      const float p = __expf(s_tensor(qi, kj) - m_new);
      sP(qi, kj) = static_cast<half_t>(p);
      l_part += p;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) l_part += __shfl_xor_sync(0xffffffff, l_part, off);
    const float l_new = l_state[qi] * scale + l_part;

    if (lane == 0) {
      m_state[qi] = m_new;
      l_state[qi] = l_new;
      scale_arr[qi] = scale;
    }
  }
  __syncthreads();
}

// 阶段2:用逐行因子(rows_factor[qi])缩放寄存器累加器 acc 的每一行。
//   online softmax 的 O rescale 和 epilogue 的 /l 都是这个操作(传不同因子)。
template <int q_step, int N, class ThrMMA, class AccTensor, class RowFn>
__device__ __forceinline__ void fa_scale_acc_rows(AccTensor &acc, ThrMMA &thr_mma, RowFn row_fn) {
  Tensor cO = make_identity_tensor(make_shape(Int<q_step>{}, Int<N>{}));
  auto taccOcO = thr_mma.partition_C(cO);
#pragma unroll
  for (int m = 0; m < size<1>(acc); ++m)
    for (int n = 0; n < size<2>(acc); ++n)
      for (int v = 0; v < size<0>(acc); ++v) {
        int qi = get<0>(taccOcO(v, m, n));
        acc(v, m, n) *= row_fn(qi);
      }
}

// 阶段3:把已在 smem 的 P(sP)经 ldmatrix 读到寄存器 fragment tCrP(MMA-A operand)。
template <class Config, class TiledMMA, class ThrMMA, class PTensor, class RegP>
__device__ __forceinline__ void fa_load_P_to_reg(const PTensor &sP, TiledMMA &tiled_mma,
                                                 ThrMMA &thr_mma, RegP &tCrP) {
  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  auto tAsP = s2r_thr_copy_a.partition_S(sP);
  auto tCrP_view = s2r_thr_copy_a.retile_D(tCrP);
  for (int ik = 0; ik < size<2>(tCrP); ++ik) {
    cute::copy(s2r_copy_a, tAsP(_, _, ik), tCrP_view(_, _, ik));
  }
  __syncthreads();
}

// 阶段4:载入 V 块到 smem 并做 acc_o += P·V(gemm_rs)。
//   gVT_slab:本 kv 块的 Vᵀ (N,kv_step) gmem;sV_ptr:smem 暂存区(复用 ab_stage)。
template <int N, int kv_step, class Config, class TiledMMA, class ThrMMA, class AccTensor,
          class RegP, class VTensor>
__device__ __forceinline__ void fa_pv_gemm(AccTensor &acc_o, const RegP &tCrP,
                                           const VTensor &gVT_slab, half_t *sV_ptr,
                                           TiledMMA &tiled_mma, ThrMMA &thr_mma) {
  using MySmemLayoutV = decltype(tile_to_shape(typename Config::SmemLayoutAtom{},
                                               make_shape(Int<N>{}, Int<kv_step>{})));
  auto sV = make_tensor(make_smem_ptr(sV_ptr), MySmemLayoutV{});

  // gmem → smem(cp.async)
  using G2SCopyB = typename Config::G2SCopyB;
  G2SCopyB g2s_copy_b;
  auto g2s_thr_copy_b = g2s_copy_b.get_slice(threadIdx.x);
  cute::copy(g2s_copy_b, g2s_thr_copy_b.partition_S(gVT_slab), g2s_thr_copy_b.partition_D(sV));
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  // gemm_rs: B(=V) 从 smem 经 LDSM_T 读到 reg,A(=P) 已在 reg
  using MySmemLayoutVNoSwizzle = decltype(get_nonswizzle_portion(MySmemLayoutV{}));
  auto sV_no_swizzle = make_tensor(sV.data().get(), MySmemLayoutVNoSwizzle{});
  auto tOrVt = thr_mma.partition_fragment_B(sV_no_swizzle);
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  auto smem_tiled_copy_V = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_slice(threadIdx.x);
  auto tOsVt = smem_thr_copy_V.partition_S(sV);

  gemm_rs(acc_o, tCrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
  __syncthreads();
}

// 阶段5:epilogue。acc_o 逐行除以 l,写回 gmem out_slab。
template <int q_step, int N, class TiledMMA, class ThrMMA, class AccTensor, class OutTensor>
__device__ __forceinline__ void fa_epilogue(AccTensor &acc_o, OutTensor &out_slab,
                                            const float *l_state, TiledMMA &tiled_mma,
                                            ThrMMA &thr_mma) {
  fa_scale_acc_rows<q_step, N>(acc_o, thr_mma, [&](int qi) { return 1.f / l_state[qi]; });
  auto tCsO = thr_mma.partition_C(out_slab);
#pragma unroll
  for (int m = 0; m < size<1>(acc_o); ++m)
    for (int n = 0; n < size<2>(acc_o); ++n)
      for (int v = 0; v < size<0>(acc_o); ++v)
        tCsO(v, m, n) = static_cast<float>(acc_o(v, m, n));
}

// Main flash attention forward kernel (register-based O)
template <int M, int N, int q_step, int kv_step>
__global__ void flash_attention_forward(const half_t *q_ptr, const half_t *k_ptr,
                                        const half_t *vt_ptr, float *out_ptr) {
  using Gemm1Config =
      config::GemmConfig<half_t, q_step, kv_step, N, 1, 2, config::GemmMode::kABt, float>;
  using Gemm2Config =
      config::GemmConfig<half_t, q_step, N, kv_step, 1, 2, config::GemmMode::kAB, float>;

  static_assert(Gemm1Config::kThreadNum == Gemm2Config::kThreadNum,
                "Gemm1 and Gemm2 must use the same thread count");

  constexpr int shm_size_AB = (Gemm1Config::shm_size_AB > Gemm2Config::shm_size_AB)
                                  ? Gemm1Config::shm_size_AB
                                  : Gemm2Config::shm_size_AB;

  // ---- unified dynamic smem layout ----
  // [0) ab_stage: half[shm_size_AB]      (shared by both gemms + V staging)
  // [1) s_buf:   float[q_step * kv_step] (S = Q·Kᵀ, float)
  extern __shared__ char smem_raw[];

  half_t *ab_stage = reinterpret_cast<half_t *>(smem_raw);
  float *s_buf =
      reinterpret_cast<float *>(smem_raw + shm_size_AB * sizeof(half_t));

  // running max / sum — static smem
  __shared__ float m_state[q_step];
  __shared__ float l_state[q_step];
  __shared__ float scale_arr[q_step];

  const int q_start = blockIdx.x * q_step;

  // Q slab for this block: (q_step, N) row-major
  auto gQ_slab =
      make_tensor(make_gmem_ptr(q_ptr + q_start * N),
                  make_shape(Int<q_step>{}, Int<N>{}), make_stride(N, Int<1>{}));

  // smem tensors
  auto s_tensor = make_tensor(make_smem_ptr(s_buf),
                              make_layout(make_shape(Int<q_step>{}, Int<kv_step>{}),
                                          make_stride(Int<kv_step>{}, Int<1>{})));

  // TiledMMA instance
  using TiledMma = typename Gemm1Config::MMA;
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  // acc_o in registers (O accumulator)
  auto acc_o = partition_fragment_C(tiled_mma, Shape<Int<q_step>, Int<N>>{});
  clear(acc_o);

  // ---- init m, l ----
  for (int qi = threadIdx.x; qi < q_step; qi += blockDim.x) {
    m_state[qi] = -INFINITY;
    l_state[qi] = 0.f;
  }
  __syncthreads();

  // ---- iterate over KV blocks ----
  for (int kv = 0; kv < M; kv += kv_step) {
    // K slab: (kv_step, N) row-major
    auto gK_slab =
        make_tensor(make_gmem_ptr(k_ptr + kv * N),
                    make_shape(Int<kv_step>{}, Int<N>{}), make_stride(N, Int<1>{}));

    // 1) S = Q · Kᵀ  (gemm1: kABt, output float smem)
    gemm_multi_stage_device<Gemm1Config>(s_tensor, gQ_slab, gK_slab, ab_stage);
    __syncthreads();

    // P 写 smem sP(swizzled),复用 ab_stage 区(gemm1 已用完,V 稍后也复用,分时不冲突)。
    using MySmemLayoutP = decltype(tile_to_shape(
        typename Gemm2Config::SmemLayoutAtom{}, make_shape(Int<q_step>{}, Int<kv_step>{})));
    auto sP = make_tensor(make_smem_ptr(ab_stage), MySmemLayoutP{});

    // 2) online softmax:更新 m/l/scale,写 P 到 sP
    fa_online_softmax<q_step, kv_step>(s_tensor, sP, m_state, l_state, scale_arr);

    // 用本轮逐行 scale 缩放已累加的 acc_o(再加入新块 P·V)
    fa_scale_acc_rows<q_step, N>(acc_o, thr_mma, [&](int qi) { return scale_arr[qi]; });
    __syncthreads();

    // 3) P(sP) 读到寄存器
    auto tCrP = thr_mma.partition_fragment_A(sP);
    fa_load_P_to_reg<Gemm2Config>(sP, tiled_mma, thr_mma, tCrP);

    // 4) 载入 V 块到 smem,acc_o += P·V
    auto gVT_slab = make_tensor(make_gmem_ptr(vt_ptr + kv),
                                make_shape(Int<N>{}, Int<kv_step>{}), make_stride(M, Int<1>{}));
    fa_pv_gemm<N, kv_step, Gemm2Config>(acc_o, tCrP, gVT_slab, ab_stage, tiled_mma, thr_mma);
  }

  // ---- epilogue: acc_o 逐行 /l 并写回 gmem ----
  auto out_slab =
      make_tensor(make_gmem_ptr(out_ptr + q_start * N),
                  make_shape(Int<q_step>{}, Int<N>{}), make_stride(N, Int<1>{}));
  fa_epilogue<q_step, N>(acc_o, out_slab, l_state, tiled_mma, thr_mma);
}

// ---- host launcher ----
template <int M, int N, int q_step, int kv_step>
inline void host_flash_attention_forward(const half_t *q, const half_t *k, const half_t *vt,
                                         float *out, int q_rows = M, cudaStream_t stream = 0) {
  using Gemm1Config =
      config::GemmConfig<half_t, q_step, kv_step, N, 1, 2, config::GemmMode::kABt, float>;
  using Gemm2Config =
      config::GemmConfig<half_t, q_step, N, kv_step, 1, 2, config::GemmMode::kAB, float>;

  constexpr int shm_size_AB = (Gemm1Config::shm_size_AB > Gemm2Config::shm_size_AB)
                                  ? Gemm1Config::shm_size_AB
                                  : Gemm2Config::shm_size_AB;

  constexpr int total_smem =
      shm_size_AB * sizeof(half_t) + q_step * kv_step * sizeof(float);

  static_assert(total_smem <= 99000, "Dynamic smem exceeds ~99KB limit");

  dim3 block(Gemm1Config::kThreadNum);
  dim3 grid(q_rows / q_step);

  cudaFuncSetAttribute(flash_attention_forward<M, N, q_step, kv_step>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, total_smem);
  flash_attention_forward<M, N, q_step, kv_step><<<grid, block, total_smem, stream>>>(
      q, k, vt, out);
}

}  // namespace gpu

namespace cpu {

// Multi-GPU flash attention (unchanged)
template <int M, int N, int q_step, int kv_step>
void flash_attention_forward(const half_t *q, const half_t *k, const half_t *vt, float *out) {
  int device_num = 0;
  cudaGetDeviceCount(&device_num);
  if (device_num < 1) return;

  const int total_blocks = M / q_step;
  const int base_blk = total_blocks / device_num;
  const int rem_blk = total_blocks % device_num;

  std::vector<cudaStream_t> streams(device_num);
  std::vector<half_t *> d_q(device_num, nullptr);
  std::vector<half_t *> d_k(device_num, nullptr);
  std::vector<half_t *> d_vt(device_num, nullptr);
  std::vector<float *> d_out(device_num, nullptr);
  std::vector<int> q_rows(device_num, 0);
  std::vector<int> row0(device_num, 0);

  for (int d = 0, blk = 0; d < device_num; ++d) {
    int my_blocks = base_blk + (d < rem_blk ? 1 : 0);
    q_rows[d] = my_blocks * q_step;
    row0[d] = blk * q_step;
    blk += my_blocks;
  }

  for (int d = 0; d < device_num; ++d) {
    cudaSetDevice(d);
    cudaStreamCreate(&streams[d]);
    if (q_rows[d] == 0) continue;

    const int mi = q_rows[d];
    cudaMalloc(&d_k[d], sizeof(half_t) * M * N);
    cudaMalloc(&d_vt[d], sizeof(half_t) * N * M);
    cudaMalloc(&d_q[d], sizeof(half_t) * mi * N);
    cudaMalloc(&d_out[d], sizeof(float) * mi * N);

    cudaMemcpyAsync(d_k[d], k, sizeof(half_t) * M * N, cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_vt[d], vt, sizeof(half_t) * N * M, cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_q[d], q + static_cast<size_t>(row0[d]) * N, sizeof(half_t) * mi * N,
                    cudaMemcpyHostToDevice, streams[d]);

    gpu::host_flash_attention_forward<M, N, q_step, kv_step>(d_q[d], d_k[d], d_vt[d], d_out[d],
                                                             /*q_rows=*/mi, streams[d]);

    cudaMemcpyAsync(out + static_cast<size_t>(row0[d]) * N, d_out[d], sizeof(float) * mi * N,
                    cudaMemcpyDeviceToHost, streams[d]);
  }

  for (int d = 0; d < device_num; ++d) {
    cudaSetDevice(d);
    cudaStreamSynchronize(streams[d]);
    cudaStreamDestroy(streams[d]);
    cudaFree(d_q[d]);
    cudaFree(d_k[d]);
    cudaFree(d_vt[d]);
    cudaFree(d_out[d]);
  }
}

// 多卡 flash 的 setup/compute 分离版:构造时一次性分配各卡显存 + 拷入 K/V/Q,
// run() 只做 kernel launch + O 的 D2H + 同步(可重复计时)。用于衡量"分片计算本身"
// 的加速,把一次性的 malloc/H2D 从每次计算中剥离(真实部署里 K/V/Q 常驻显存、复用多次)。
template <int M, int N, int q_step, int kv_step>
class MultiGpuFlash {
 public:
  MultiGpuFlash(const half_t *q, const half_t *k, const half_t *vt) {
    cudaGetDeviceCount(&device_num_);
    const int total_blocks = M / q_step;
    const int base_blk = total_blocks / device_num_;
    const int rem_blk = total_blocks % device_num_;
    d_q_.assign(device_num_, nullptr);
    d_k_.assign(device_num_, nullptr);
    d_vt_.assign(device_num_, nullptr);
    d_out_.assign(device_num_, nullptr);
    streams_.assign(device_num_, nullptr);
    q_rows_.assign(device_num_, 0);
    row0_.assign(device_num_, 0);

    for (int d = 0, blk = 0; d < device_num_; ++d) {
      int my_blocks = base_blk + (d < rem_blk ? 1 : 0);
      q_rows_[d] = my_blocks * q_step;
      row0_[d] = blk * q_step;
      blk += my_blocks;
    }
    // 一次性 setup:分配 + 拷入(计时时不再重复)
    for (int d = 0; d < device_num_; ++d) {
      cudaSetDevice(d);
      cudaStreamCreate(&streams_[d]);
      if (q_rows_[d] == 0) continue;
      const int mi = q_rows_[d];
      cudaMalloc(&d_k_[d], sizeof(half_t) * M * N);
      cudaMalloc(&d_vt_[d], sizeof(half_t) * N * M);
      cudaMalloc(&d_q_[d], sizeof(half_t) * mi * N);
      cudaMalloc(&d_out_[d], sizeof(float) * mi * N);
      cudaMemcpy(d_k_[d], k, sizeof(half_t) * M * N, cudaMemcpyHostToDevice);
      cudaMemcpy(d_vt_[d], vt, sizeof(half_t) * N * M, cudaMemcpyHostToDevice);
      cudaMemcpy(d_q_[d], q + static_cast<size_t>(row0_[d]) * N, sizeof(half_t) * mi * N,
                 cudaMemcpyHostToDevice);
    }
  }

  // 只做计算:各卡 launch kernel + 同步。结果留各卡 d_out_ 显存(不 D2H),
  // 与单卡纯 kernel 计时口径对齐,衡量分片计算本身的加速。
  void run() {
    for (int d = 0; d < device_num_; ++d) {
      if (q_rows_[d] == 0) continue;
      cudaSetDevice(d);
      const int mi = q_rows_[d];
      gpu::host_flash_attention_forward<M, N, q_step, kv_step>(d_q_[d], d_k_[d], d_vt_[d], d_out_[d],
                                                               mi, streams_[d]);
    }
    for (int d = 0; d < device_num_; ++d) {
      cudaSetDevice(d);
      cudaStreamSynchronize(streams_[d]);
    }
  }

  // 收回结果(计时外单独调,用于正确性校验)
  void gather(float *out) {
    for (int d = 0; d < device_num_; ++d) {
      if (q_rows_[d] == 0) continue;
      cudaSetDevice(d);
      const int mi = q_rows_[d];
      cudaMemcpy(out + static_cast<size_t>(row0_[d]) * N, d_out_[d], sizeof(float) * mi * N,
                 cudaMemcpyDeviceToHost);
    }
  }

  ~MultiGpuFlash() {
    for (int d = 0; d < device_num_; ++d) {
      cudaSetDevice(d);
      if (streams_[d]) cudaStreamDestroy(streams_[d]);
      cudaFree(d_q_[d]);
      cudaFree(d_k_[d]);
      cudaFree(d_vt_[d]);
      cudaFree(d_out_[d]);
    }
  }

  int device_num() const { return device_num_; }

 private:
  int device_num_ = 0;
  std::vector<half_t *> d_q_, d_k_, d_vt_;
  std::vector<float *> d_out_;
  std::vector<cudaStream_t> streams_;
  std::vector<int> q_rows_, row0_;
};

}  // namespace cpu
}  // namespace kernel
