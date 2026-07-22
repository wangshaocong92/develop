#pragma once

#include <cute/tensor.hpp>

// ============================================================================
// gemm_rs: A-in-registers / B-from-smem(via LDSM) / C-in-registers 的 GEMM。
// 用于 flash attention 的第二个 gemm(P·V):P 已是寄存器 fragment(不落 gmem/smem),
// V 从 smem 经转置 ldmatrix 读入,累加到寄存器里的 O。
//
// 逻辑照搬官方 FlashAttention(csrc/flash_attn/src/utils.h 的 gemm_rs /
// convert_layout_acc_Aregs),接口泛化以适配本项目的 TiledMMA。
// ============================================================================
namespace kernel {
namespace gpu {

using namespace cute;

// 把 MMA accumulator(C)的寄存器 layout 转成 MMA A-operand 需要的 layout。
//   C:  (MMA=4, MMA_M, MMA_N)   —— 每线程 4 个 float 累加器
//   A:  ((4,2), MMA_M, MMA_N/2) —— k=16 时每线程 8 个 half,复用同一块寄存器
// 数据不搬迁,只换 layout 解读(配合 convert_type<half> 后使用)。
template <typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
  static_assert(mma_shape_K == 8 || mma_shape_K == 16);
  if constexpr (mma_shape_K == 8) {
    return acc_layout;
  } else {
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N/2))
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
  }
}

// acc += A · B,A 在寄存器(tCrA,不动),B 从 smem(tCsB)经 copy 预取到寄存器(tCrB)。
// 逐 K-slice 软件流水:先取 K=0,循环内预取下一片再做当前片的 MMA。
template <typename TensorAcc, typename TensorAReg, typename TensorBReg, typename TensorBSmem,
          typename TiledMMA, typename TiledCopyB, typename ThrCopyB>
__forceinline__ __device__ void gemm_rs(TensorAcc &acc, TensorAReg const &tCrA, TensorBReg &tCrB,
                                        TensorBSmem const &tCsB, TiledMMA tiled_mma,
                                        TiledCopyB smem_tiled_copy_B, ThrCopyB smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));   // MMA_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));   // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));  // MMA_K

  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));  // N

  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

}  // namespace gpu
}  // namespace kernel
