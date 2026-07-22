#pragma once

#include <cute/tensor.hpp>

namespace kernel {
namespace gpu {

// ============================================================================
// 多级流水线(multi-stage / software-pipelined)GEMM:D = A · Bᵀ
//   A: (M, K) row-major        B: (N, K) row-major(计算中按 Bᵀ 使用)
//   D: (M, N) row-major
// 源自 doc/cute-gemm/gemm-multi-stage.cu,改为可被其它 kernel 复用的 __device__
// 函数;去掉了原文件里 cublas / 随机数据 / 比较等测试专用依赖。
//
// 用法:
//   * 组合进别的 kernel:直接在 __global__ 里调用 gemm_multi_stage_device<Cfg>()
//   * 独立启动:用 host_gemm_multi_stage<Cfg>() 或 gemm_multi_stage_kernel<Cfg>
//
// 单 block 乘法原语:用整个 thread block 计算 gD = gA_slab · gB_slabᵀ,内部自行
// 遍历 K。**不引用 blockIdx** —— 由调用方决定 block↔数据的映射并切好数据块。
//   gA_slab: (kTileM, K) row-major   本 block 负责的 A 行块,完整 K
//   gB_slab: (kTileN, K) row-major   本 block 负责的 B 行块,完整 K
//   gD:      (kTileM, kTileN)        本 block 的输出 tile(gmem 或 smem)
// ============================================================================
template <typename Config, typename TensorA, typename TensorB, typename TensorD>
__device__ void gemm_multi_stage_device(TensorD gD, TensorA gA_slab, TensorB gB_slab,
                                        void *smem_ptr = nullptr) {
  using namespace cute;

  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutC = typename Config::SmemLayoutC;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using R2SCopyAtomC = typename Config::R2SCopyAtomC;
  using S2GCopyAtomC = typename Config::S2GCopyAtomC;
  using S2GCopyC = typename Config::S2GCopyC;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;

  // K 维行数(遍历用),由 slab 的 shape 推出
  const int k = size<1>(gA_slab);

  // A/B 暂存 smem:调用方传入指针则用其区域(flash 等外层统一编排 smem),否则用本函数
  // 的 extern __shared__(独立启动)。extern 统一用 char 同名同类型,避免 device-link 冲突。
  extern __shared__ char smem_shared[];
  T *shm_data = (smem_ptr != nullptr) ? reinterpret_cast<T *>(smem_ptr)
                                      : reinterpret_cast<T *>(smem_shared);

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;

  // 把本 block 的 slab 沿 K 拆成 (tile, kTileK, K/kTileK) 三维视图供主循环遍历。
  // slab 只有一个 M/N-tile,故坐标恒取 0;结果 stride 与原 grid 级 gA/gB 逐位一致。
  Tensor gA = local_tile(gA_slab, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                         make_coord(0, _));  // (kTileM, kTileK, k/kTileK)
  // B slab 两模式相同:物理 (kTileN, K) row-major = (K, kTileN) column-major,
  // 恰好是 TN MMA(row.col)要的 B 布局。故 kABt/kAB 用同一切法、同一 LDSM_N,
  // 无需转置视图。差异仅在调用方对 B 的数学解读(见 GemmMode 注释)。
  Tensor gB = local_tile(gB_slab, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                         make_coord(0, _));  // (kTileN, kTileK, k/kTileK)
  // gD 已是调用方给定的 (kTileM, kTileN) 输出 tile,直接使用(gmem 或 smem)。

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{});  // (kTileM, kTileK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{});  // (kTileN, kTileK, kStage)

  // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
  // method
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);           // (MMA, MMA_M, MMA_N)

  // fill zero for accumulator
  clear(tCrD);

  // gmem -cp.async-> shm -ldmatrix-> reg
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);  // ? (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);  // ? (CPY, CPY_M, CPY_K)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);  // ? (CPY, CPY_M, CPY_K, kStage)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);  // ? (CPY, CPY_M, CPY_K)

  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
  auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)
  auto tBsB_copy =
      g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage)

  int itile_to_read = 0;
  int ismem_read = 0;
  int ismem_write = 0;

  int ntile = k / kTileK;

  if constexpr (kStage == 1) {
    // ---- 单缓冲路径(无多级流水):每个 k-tile 顺序 g2s → sync → s2r → mma ----
    // flash 场景 K 只有 1 个 tile,流水线无收益,用它省一半 A/B smem。
#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
      cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile), tAsA_copy(_, _, _, 0));
      cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile), tBsB_copy(_, _, _, 0));
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();

      int nk = size<2>(tCrA);
#pragma unroll
      for (int ik = 0; ik < nk; ++ik) {
        cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, 0), tCrA_view(_, _, ik));
        cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, 0), tCrB_view(_, _, ik));
        cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
      }
      __syncthreads();  // 保护下一 tile 覆盖 smem
    }
  } else {
    // ---- 多级流水线路径(kStage >= 2,双缓冲预取)----
    // submit kStage - 1 tile
    // gmem -> shm
#pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
      cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
                 tAsA_copy(_, _, _, istage));
      cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
                 tBsB_copy(_, _, _, istage));
      cp_async_fence();

      ++itile_to_read;
      ++ismem_write;
    }

    // wait one submitted gmem->smem done
    cp_async_wait<kStage - 2>();
    __syncthreads();

    int ik = 0;
    // smem -> reg
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

    // loop over k: i. load tile, ii. mma
#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
      int nk = size<2>(tCrA);

#pragma unroll
      for (int ik = 0; ik < nk; ++ik) {
        int ik_next = (ik + 1) % nk;

        if (ik == nk - 1) {
          cp_async_wait<kStage - 2>();
          __syncthreads();

          ismem_read = (ismem_read + 1) % kStage;
        }

        // shm -> reg s[itile][ik + 1] -> r[ik + 1]
        cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                   tCrA_view(_, _, ik_next));
        cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                   tCrB_view(_, _, ik_next));

        if (ik == 0) {
          if (itile_to_read < ntile) {
            cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                       tAsA_copy(_, _, _, ismem_write));
            cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                       tBsB_copy(_, _, _, ismem_write));

            ++itile_to_read;
            ismem_write = (ismem_write + 1) % kStage;
          }

          cp_async_fence();
        }

        cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
      }  // for ik
    }    // itile
  }

  if constexpr (is_smem<TensorD>::value) {
    // ---- 输出停在 shared:累加器直接写回调用方给的 smem tile,不碰 global ----
    // 用 MMA 的 C 分区,把每个线程持有的 tCrD 片段写到 gD(smem)的对应位置。
    auto thr_mma_c = tiled_mma.get_slice(idx);
    auto tCsD = thr_mma_c.partition_C(gD);  // (MMA, MMA_M, MMA_N),指向 smem
    using DstT = typename TensorD::value_type;  // 输出元素类型(可能 half 或 float)
    // 按 (V, MMA_M, MMA_N) 逻辑坐标逐元素写:tCrD(紧凑 fragment)与 tCsD(partition_C
    // 坐标映射)的 1D flatten 顺序不一定一致(half 恰好对齐、float 会错位),故用三维坐标。
#pragma unroll
    for (int m = 0; m < size<1>(tCrD); ++m)
#pragma unroll
      for (int n = 0; n < size<2>(tCrD); ++n)
#pragma unroll
        for (int v = 0; v < size<0>(tCrD); ++v)
          tCsD(v, m, n) = static_cast<DstT>(tCrD(v, m, n));
    __syncthreads();  // 确保整块 tile 对后续消费者(如 softmax)可见
  } else {
    // ---- 输出写回 global:reg -> shm(scratchpad) -> global ----
    // use less shared memory as a scratchpad tile to use large wide instuction
    // Dreg -> shm -> reg -> global
    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);   // (CPY, CPY_M, CPY_N)
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe)

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  // (CPY, _1, _1, pipe)
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);  // (CPY, CPY_M, CPY_N)

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN)
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN)

    int step = size<3>(tCsC_r2s);  // pipe
#pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
      // reg -> shm
#pragma unroll
      for (int j = 0; j < step; ++j) {
        // we add a temp tensor to cope with accumulator and output data type
        // difference
        auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
        cute::copy(tCrC_r2sx(_, i + j), t);

        cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
      }
      __syncthreads();

#pragma unroll
      // shm -> global
      for (int j = 0; j < step; ++j) {
        cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
      }

      __syncthreads();
    }
  }
}

// __global__ 包装:承担 grid 级切片 —— 用 blockIdx + 指针算术切出本 block 的
// A/B 行块 slab 和输出 tile,再转调单 block 原语。独立启动时用它;host 端标量参数。
template <typename Config>
__global__ void gemm_multi_stage_kernel(void *Dptr, const void *Aptr, const void *Bptr, int m,
                                        int n, int k) {
  using namespace cute;
  using T = typename Config::T;
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  int ix = blockIdx.x, iy = blockIdx.y;
  (void)m;  // 总行数不再需要(grid.y 已隐含),保留形参仅为接口稳定

  // 本 block 的 A 行块 (kTileM,K)、B 行块 (kTileN,K)、输出 tile (kTileM,kTileN)
  auto gA_slab = make_tensor(make_gmem_ptr((T *)Aptr + iy * kTileM * k),
                             make_shape(Int<kTileM>{}, k), make_stride(k, Int<1>{}));
  auto gB_slab = make_tensor(make_gmem_ptr((T *)Bptr + ix * kTileN * k),
                             make_shape(Int<kTileN>{}, k), make_stride(k, Int<1>{}));
  auto gD_tile = make_tensor(make_gmem_ptr((T *)Dptr + iy * kTileM * n + ix * kTileN),
                             make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(n, Int<1>{}));
  gemm_multi_stage_device<Config>(gD_tile, gA_slab, gB_slab);
}

}  // namespace gpu

namespace config {

using namespace cute;

// gemm 运算模式:kABt = D=A·Bᵀ(B 为 (N,K) row-major,当 Bᵀ 用);
//               kAB  = D=A·B (B 为 (K,N) row-major,不转置)。
enum class GemmMode { kABt, kAB };

template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
          int kStage_ = 5, int kSmemLayoutCBatch_ = 2,
          GemmMode kMode_ = GemmMode::kABt, typename ComputeType_ = T_>
struct GemmConfig {
  using T = T_;
  using ComputeType = ComputeType_;  // 累加/输出计算类型(F32 时启用 float 累加 MMA)
  static constexpr GemmMode kMode = kMode_;
  static constexpr bool kTransposeB = (kMode == GemmMode::kABt);

  // tile configuration
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;
  static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                  make_stride(Int<kTileK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));

  // B 侧 smem 布局:两模式相同。B 物理 (kTileN,kTileK) row-major = (kTileK,kTileN)
  // column-major,正是 TN MMA 要的 B。故复用 K-major atom,无需转置。
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

  // MMA:ComputeType=float 时用 F32 累加(A/B 仍 half,C/D float);否则全 half。
  using mma_op = std::conditional_t<std::is_same_v<ComputeType, float>,
                                    SM80_16x8x16_F32F16F16F32_TN,
                                    SM80_16x8x16_F16F16F16F16_TN>;

  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = typename mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

  using G2SCopyA =
      decltype(make_tiled_copy(g2s_copy_atom{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  // shared memory to register copy
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;

  using S2RCopyAtomA = s2r_copy_atom;
  // B 侧:两模式都用非转置 LDSM_N(B 物理布局已匹配 TN MMA 的 col-major 要求)。
  using S2RCopyAtomB = s2r_copy_atom;

  // epilogue: register to global via shared memory
  using SmemLayoutAtomC = decltype(composition(
      Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                                      make_stride(Int<kMmaPN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

  static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
                    size(SmemLayoutC{}),
                "C shared memory request is large than A's one pipe");

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;

  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC =
      decltype(make_tiled_copy(S2GCopyAtomC{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  static constexpr int kThreadNum = size(MMA{});
  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});

  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);
};

}  // namespace config

namespace gpu {

// Host 启动器:配置 grid/block/动态共享内存并启动 multi-stage GEMM。
//   D = A · Bᵀ,A:(m,k) B:(n,k) D:(m,n),均为 row-major。
// 要求 k % Config::kTileK == 0(与原实现一致,内部按整块遍历 K)。
template <typename Config>
inline void host_gemm_multi_stage(void *Dptr, const void *Aptr, const void *Bptr, int m, int n,
                                  int k, cudaStream_t stream = 0) {
  Config cfg;
  dim3 block(Config::kThreadNum);
  dim3 grid((n + Config::kTileN - 1) / Config::kTileN,
            (m + Config::kTileM - 1) / Config::kTileM);
  int shm_size = Config::kShmSize;

  // 动态共享内存可能超过 48KB 的默认上限,需 opt-in
  cudaFuncSetAttribute(gemm_multi_stage_kernel<Config>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
  gemm_multi_stage_kernel<Config><<<grid, block, shm_size, stream>>>(Dptr, Aptr, Bptr, m, n, k);
}

}  // namespace gpu
}  // namespace kernel
