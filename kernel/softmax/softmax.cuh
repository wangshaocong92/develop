#pragma once
#include <cutlass/cutlass.h>

#include <cute/tensor.hpp>
#include <cuda_runtime_api.h>
#include "cute/numeric/integral_constant.hpp"
namespace kernel {
    namespace gpu {
        /*
            输入是一个一维数组，长度为 n，在 GPU 上计算 softmax 的前向传播，在原地输出，长度为 n。
            1. 使用多少个block，单个block的线程数为多少
            2. block 内做reduce


        */
        void __global__ softmax_forward(cute::Tensor<float, cute::Shape<int>> &input)
        {
            //// 获取 block 和 thread 的索引

        }
        void __global__ online_softmax_forward(const float *input, float *output, int n)
        {

        }
    } // namespace gpu
}