#pragma once

#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
namespace kernel {
namespace cpu {
using namespace cute;
template <int M, int N,
          class Tensor2D = Tensor<float, Layout<Shape<Int<M>, Int<N>>, Stride<Int<1>, Int<M>>>>>
void flash_attention_forward(const Tensor2D &q, const Tensor2D &k, const Tensor2D &v,
                             Tensor2D &out) {
  /*
  单个 block 做什么

  假如 kv 可以加载到 global mem。中间的 m * m 矩阵加载不了

  m x m = m x n * n x m
  m x n = m x m * m x n

  m x m 的中间矩阵 如何分到更多的显卡或者机器上去？
  单个device上持有的中间矩阵应该是 mi x m的才能支持整个系统的执行即
  qi > mi x n
  k > m x n
  v > m x n
  out > mi x n
  这样就可以把中间矩阵分到更多的显卡上去,即使单个显卡的内存不够大,也可以支持整个系统的执行
  */
}


}  // namespace cpu
}  // namespace kernel