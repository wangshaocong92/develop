#include "print.h"
#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>
#include <cute/util/print_latex.hpp>
#include <cutlass/uint128.h>

using namespace cute;



struct GemmConfig
{
  /*
  1. gm -> smem A,B 128 * 32
      smem 的定义: 128 * 32 * n(缓冲区大小)
      定义 smem layout=> smem atom => (Swizzle<2,3,3>,shape<8,32>) => 平铺到 128
  * 32 * n 上 定义 global to shared
  */
  using DataType = cute::half_t;
  static constexpr int MPerBlock = 128;
  static constexpr int NPerBlock = 128;
  static constexpr int KPerBlock = 32;
  /// mma
  using MmaOp = SM80_16x8x16_F16F16F16F16_TN; /// mma operator
  using MmaTraits = MMA_Traits<MmaOp>;        /// mma traits
  /*
      MMA_Atom
      ThrID:      _32:_1
      LayoutA_TV: ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
      LayoutB_TV: ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
      LayoutC_TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
*/
  using MmaAtom =
      MMA_Atom<MmaTraits>; /// mma atom
                           /// 单位mma的计算工具即单个warp计算的矩阵大小

  //// MmaAtom 在整个block上的平铺参数 即 32*16 16*16 =》 32 * 16
  static constexpr int MmaEURepeatM = 2;
  static constexpr int MmaEURepeatN = 2;
  static constexpr int MmaEURepeatK = 1;
  using MmaAtomShape = MmaTraits::Shape_MNK;
  static constexpr int MmaPM = 1 * MmaEURepeatM * get<0>(MmaAtomShape{});
  static constexpr int MmaPN = 2 * MmaEURepeatN * get<1>(MmaAtomShape{});
  static constexpr int MmaPK = 1 * MmaEURepeatK * get<2>(MmaAtomShape{});
  using MmaEURepeatT = decltype(make_layout(make_shape(
      Int<MmaEURepeatM>{}, Int<MmaEURepeatN>{}, Int<MmaEURepeatK>{})));
  using MmaPT = Tile<Int<MmaPM>, Int<MmaPN>, Int<MmaPK>>;
  /*
      TiledMMA
       ThrLayoutVMNK:  (_32,_2,_2,_1):(_1,_32,_64,_0)
       PermutationMNK: ((_1,_2,_1):(_0,_1,_0),_,_)
  */
  using Mma1 = decltype(make_tiled_mma(MmaAtom{}, MmaEURepeatT{},
                                       make_layout(Shape<_1, _2, _1>{})));
  /*
      TiledMMA
          ThrLayoutVMNK:  (_32,_2,_2,_1):(_1,_32,_64,_0)
          PermutationMNK: (_32,_32,_16)
  */
  using Mma = decltype(make_tiled_mma(MmaAtom{}, MmaEURepeatT{}, MmaPT{}));

  /// shared memory layout
  /*
    shared -> register 主要使用ldmatrix指令。

  */
  using SmemLayoutAtom = decltype(composition(
      Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<8>{}, Int<32>{}),
                                      make_stride(Int<32>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<MPerBlock>{}, Int<KPerBlock>{}, Int<3>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<NPerBlock>{}, Int<KPerBlock>{}, Int<3>{})));

  /// 共享内存矩阵c的目标是复用共享内存A或者B的已计算数据。这样可以减少内存申请和暂用
  using SmemLayoutAtomC = decltype(composition(
      Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<4>{}, Int<32>{}),
                                      make_stride(Int<32>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{}, make_shape(Int<32>{}, Int<32>{}, Int<2>{})));

  /// global 2 shared copy
  using G2SCopyOP = SM80_CP_ASYNC_CACHEGLOBAL<cutlass::uint128_t>;
  using G2SCopyTraits = Copy_Traits<G2SCopyOP>;
  using G2SCopyatom =
      Copy_Atom<G2SCopyTraits, DataType>; /// global到shared memory的copy工具
  using G2SCopyA = decltype(make_tiled_copy(
      G2SCopyatom{}, // 1. 拷贝原子操作
      make_layout(
          make_shape(Int<32>{}, Int<4>{}), // 2. 线程布局（Thread Layout）
          make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{})) // 3. 值布局（Value Layout）
      ));

  //// shared to register copy
  /// 主要使用ldmatrix指令,输入一个连续的128bit的数据，输出一个多寄存器的数据
  using S2RCopyOP = SM75_U32x4_LDSM_N;
  using S2RCopyTraits = Copy_Traits<S2RCopyOP>;
  using S2RCopyatom =
      Copy_Atom<S2RCopyTraits, DataType>; /// global到shared memory的copy工具
  /// 将 atom 平铺到整个mma计算的tile上
  using S2RCopyTillA = decltype(make_tiled_copy_A(
      S2RCopyatom{}, Mma{})); /// shared到register的copy工具
  using S2RCopyTillB = decltype(make_tiled_copy_B(
      S2RCopyatom{}, Mma{})); /// shared到register的copy工具

//// C register to shared copy
#if 0
      using R2SCopyOP = SM90_U32x4_STSM_N;
#else
  using R2SCopyOP = UniversalCopy<int>;
#endif
  using R2SCopyTraits = Copy_Traits<R2SCopyOP>;
  using R2SCopyAtomC =
      Copy_Atom<R2SCopyTraits, DataType>; /// register到shared memory的copy工具
  using R2SCopyTillC = decltype(make_tiled_copy_C(
      R2SCopyAtomC{}, Mma{})); /// shared到register的copy工具

  //// shared to global copy /// 32 * 32 = 1024
  /// bit，使用128bit的copy工具需要4个连续的copy指令
  using S2GCopyOP = UniversalCopy<cutlass::uint128_t>;
  using S2GCopyTraits = Copy_Traits<S2GCopyOP>;
  using S2GCopyAtomC =
      Copy_Atom<S2GCopyTraits, DataType>; /// shared memory到global的copy工具
  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{}, // 1. 拷贝原子操作
      make_layout(
          make_shape(Int<32>{}, Int<4>{}), // 2. 线程布局（Thread Layout）
          make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{})) // 3. 值布局（Value Layout）
      ));
};


int main()
{
    /// 代计算矩阵的shape
    constexpr int M = 128 * 128;
    constexpr int N = 128 * 128;
    constexpr int K = 128;
    print_latex(GemmConfig::S2GCopyC{});

    return 0;
}