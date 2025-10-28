#include "vector_add.cuh"
#include <cassert>
#include <cuda_runtime_api.h>
#include <vector>



int main(){
     cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::vector<half> data(1024 * 8 * 2);
  void *c_data = nullptr;
  cudaMallocAsync(&c_data, data.size() * sizeof(half) , stream);
  cudaMemcpyAsync(c_data, data.data(), 1024 * 8 * 2 * sizeof(half) ,
                  cudaMemcpyHostToDevice);
  void *d_out = nullptr;
  dim3 block(1024);
  dim3 grid(1);
  cudaMallocAsync(&d_out, sizeof(int) * grid.x, stream);
  cudaEventRecord(start);
  vector_add_local_tile_multi_elem_per_thread_half<<<grid,block,0,stream>>>((half * )c_data,1024 *2 * 8,(half *)c_data,(half *)c_data,half(1.0),half(1.0),half(1.0));
  // 停止计时
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("reduce_with_divergemnt_warps CUDA计算耗时: %.3f ms band width %.3f "
         "GB/s \n",
         milliseconds, (data.size() * sizeof(int) * 2) / (milliseconds * 1e6f));

  cudaFreeAsync(c_data, stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  return 0;
}