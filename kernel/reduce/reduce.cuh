#pragma once
#include <cassert>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <new>
namespace kernel {

template <typename T> __device__ T Max(const T &a, const T &b) {
  return a > b ? a : b;
}

template <typename T, typename... Args>
__device__ T Max(const T &a, Args... args) {
  static_assert((std::is_same_v<T, Args> && ...),
                "All arguments must be of the same type");
  auto r = Max(args...);
  return a > r ? a : r;
}
template <typename... Args> struct MaxOp {
  __device__ static auto apply(Args... args) {
    return Max(args...); // 调用上面的 Add
  }
};

template <typename T> __device__ T Min(const T &a, const T &b) {
  return a < b ? a : b;
}

template <typename T, typename... Args>
__device__ T Min(const T &a, Args... args) {
  static_assert((std::is_same_v<T, Args> && ...),
                "All arguments must be of the same type");
  auto r = Min(args...);
  return a < r ? a : r;
}
template <typename... Args> struct MinOp {
  __device__ static auto apply(Args... args) {
    return Min(args...); // 调用上面的 Add
  }
};

template <typename T> __device__ T Add(const T &a, const T &b) { return a + b; }

template <typename T, typename... Args>
__device__ T Add(const T &a, Args... args) {
  static_assert((std::is_same_v<T, Args> && ...),
                "All arguments must be of the same type");
  return a + Add(args...);
}

template <typename... Args> struct AddOp {
  __device__ static auto apply(Args... args) {
    return Add(args...); // 调用上面的 Add
  }
};

template <uint16_t blockSize>
__global__ void reduce_with_divergemnt_warps(int *g_idata, int *g_odata) {
  __shared__ int sdata[blockSize];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();
  // do reduction in shared mem
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

template <uint16_t blockSize>
__global__ void reduce_with_interleaved_addressing(int *g_idata, int *g_odata) {
  __shared__ int sdata[blockSize];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();
  // do reduction in shared mem
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

template <uint16_t blockSize>
__global__ void reduce_with_sequential_addressing(int *g_idata, int *g_odata) {
  __shared__ int sdata[blockSize];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();
  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

template <typename T> struct Type4 {
  T x;
  T y;
  T z;
  T w;
};

struct int4 {
  int32_t x;
  int32_t y;
  int32_t z;
  int32_t w;
};

template <uint16_t blockSize>
__global__ void reduce_add_with_load(int *g_idata, int *g_odata) {
  __shared__ int sdata[blockSize];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  auto p = (int4 *)(g_idata);
  sdata[tid] = Add(p[i].x, p[i].y, p[i].z, p[i].w);
  __syncthreads();
  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}
template <unsigned int blockSize>
__device__ __forceinline__ void warp_reduce(volatile int *sdata, int tid) {
  if (blockSize >= 64)
    sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32)
    sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16)
    sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8)
    sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4)
    sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2)
    sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce(int *g_idata, int *g_odata, unsigned int n) {
  __shared__ int sdata[blockSize];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + tid;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  sdata[tid] = 0;
  while (i < n) {
    sdata[tid] += g_idata[i] + g_idata[i + blockSize];
    i += gridSize;
  }
  __syncthreads();
  if (blockSize == 1024) {
    if (tid < 512) {
      sdata[tid] += sdata[tid + 512];
    }
    __syncthreads();
  }
  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
  }
  if (tid < 32)
    warp_reduce<blockSize>(sdata, tid);
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

template <typename T, unsigned int blockSize, template <typename...> class OP>
__device__ __forceinline__ void warp_reduce(volatile T *sdata, int tid) {
  if (blockSize >= 64)
    sdata[tid] = OP<T, T>::apply(sdata[tid + 32], sdata[tid]);
  if (blockSize >= 32)
    sdata[tid] = OP<T, T>::apply(sdata[tid + 16], sdata[tid]);
  if (blockSize >= 16)
    sdata[tid] = OP<T, T>::apply(sdata[tid + 8], sdata[tid]);
  if (blockSize >= 8)
    sdata[tid] = OP<T, T>::apply(sdata[tid + 4], sdata[tid]);
  if (blockSize >= 4)
    sdata[tid] = OP<T, T>::apply(sdata[tid + 2], sdata[tid]);
  if (blockSize >= 2)
    sdata[tid] = OP<T, T>::apply(sdata[tid + 1], sdata[tid]);
}

template <typename T, uint16_t blockSize, template <typename...> class OP>
__global__ void reduce_with_no_roll_last_warp(T *g_idata, T *g_odata) {
  __shared__ int sdata[blockSize];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  auto p = (Type4<T> *)(g_idata);
  sdata[tid] = OP<T, T, T, T>::apply(p[i].x, p[i].y, p[i].z, p[i].w);
  __syncthreads();
  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s)
      sdata[tid] = OP<T, T>::apply(sdata[tid + s], sdata[tid]);
    __syncthreads();
  }
  if (tid < 32)
    warp_reduce<T, 128, OP>(sdata, tid);

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

} // namespace kernel

#define REDUCE kernel::reduce_with_no_roll_last_warp