#pragma once
#include <cuda_runtime_api.h>
namespace kernel {
__global__ void reduce_with_divergemnt_warps(int *g_idata, int *g_odata);
}