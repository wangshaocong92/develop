#include <cute/tensor.hpp>
#include "print.h"


using namespace cute;



struct GemmConfig
{
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
    using MmaAtom = MMA_Atom<MmaTraits>;        /// mma atom 单位mma的计算工具即单个warp计算的矩阵大小

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
    using Mma = decltype(make_tiled_mma(MmaAtom{}, MmaEURepeatT{}, make_layout(Shape<_1, _2, _1>{})));
};


int main()
{
    /// 代计算矩阵的shape
    constexpr int M = 128 * 128;
    constexpr int N = 128 * 128;
    constexpr int K = 128;
    printn(GemmConfig::Mma{});

    return 0;
}