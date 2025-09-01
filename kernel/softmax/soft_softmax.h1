
#pragma once
#include <cuda_runtime_api.h>

namespace kernel {

/// cpu safe softmax implementation
void safe_softmax(const float *input, float *output, int length);

/// gpu safe softmax implementation
__global__ void safe_softmax_gpu(const float *input, float *output, int length,
                                 cudaStream_t stream);

} // namespace kernel
