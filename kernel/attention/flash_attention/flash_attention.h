#pragma once

#include <cutlass/numeric_types.h>
#include <cstddef>
#include <cute/tensor.hpp>
#include <vector>

#include "device/device.cuh"
namespace kernel {
using namespace cute;
namespace gpu {


template <int M, int N,
          // q / out: MI 运行时 → shape[0] 和 stride[1] 用 int
          class QTensor = Tensor<float, Layout<Shape<int, Int<N>>, Stride<Int<1>, int>>>,
          // k / v: M 编译期
          class KVTensor = Tensor<float, Layout<Shape<Int<M>, Int<N>>, Stride<Int<1>, Int<M>>>>>
__global__ void flash_attention_forward(int mi,  // ← 运行时:本卡 q 的行数
                                        const QTensor &q, const KVTensor &k, const KVTensor &v,
                                        QTensor &out) {}

}  // namespace gpu
namespace cpu {

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
  int DEVICE_NUM;
  cudaGetDeviceCount(&DEVICE_NUM);  // = 2
  // 单卡可用于中间矩阵的显存上界,从实际设备能力查询(替代此前硬编码的 2GB)。
  const size_t MAX_GMEM_SIZE_FOR_MI = gpu::get_device().usable_gmem_for_intermediate();
  if (sizeof(float) * M * M > MAX_GMEM_SIZE_FOR_MI) {
    // 需要多卡协同
    std::vector<cudaStream_t> streams(DEVICE_NUM);
    std::vector<cudaEvent_t> done(DEVICE_NUM);
    std::vector<float *> k_ptrs(DEVICE_NUM, nullptr);
    std::vector<float *> v_ptrs(DEVICE_NUM, nullptr);
    std::vector<Tensor2D> k_tensors(DEVICE_NUM);
    std::vector<Tensor2D> v_tensors(DEVICE_NUM);
    for (int d = 0; d < DEVICE_NUM; ++d) {
      cudaSetDevice(d);
      cudaStreamCreate(&streams[d]);
      cudaEventCreate(&done[d]);
      // 将 k v 复制到每个 device 上, 只需要复制一次, 因为 k v 是共享的
      cudaMalloc(&k_ptrs[d], sizeof(float) * M * N);
      cudaMalloc(&v_ptrs[d], sizeof(float) * M * N);
      k_tensors[d] = Tensor2D(k_ptrs[d], Layout<Shape<Int<M>, Int<N>>, Stride<Int<1>, Int<M>>>());
      v_tensors[d] = Tensor2D(v_ptrs[d], Layout<Shape<Int<M>, Int<N>>, Stride<Int<1>, Int<M>>>());
      cudaMemcpy(k_ptrs[d], k.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice, streams[d]);
      cudaMemcpy(v_ptrs[d], v.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice, streams[d]);
    }

    // 单卡计算多少
    const int mi_per_device = M / DEVICE_NUM;
    /// 多卡调度
    // 多卡协同, 并且单卡计算的中间矩阵可以放下, 可以直接计算
    for (int d = 0; d < DEVICE_NUM; ++d) {
      cudaSetDevice(d);  // 关键:当前 device 上下文
                         // 在 streams[d] 上 launch 该卡负责的 MI 行块对应的 kernel
      if (sizeof(float) * mi_per_device * M > MAX_GMEM_SIZE_FOR_MI) {
        /// 需要多卡协同, 并且单卡计算的中间矩阵也太大了, 需要分批计算
      } else {
      }
      /// 调用
      cudaEventRecord(done[d], streams[d]);
    }

  } else {
    // 直接在单卡上执行
  }
}


}  // namespace cpu
}  // namespace kernel