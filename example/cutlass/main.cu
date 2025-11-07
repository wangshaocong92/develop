#include "vector_add.cuh"
#include <cassert>
#include <cuda_runtime_api.h>
#include <vector>

constexpr int data_size = 1024 << 12;
#define PerThread 64

template <typename T>
void check_register_limits(T kernel, int block_size) {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int registers_per_thread = attr.numRegs;
    int total_registers_needed = registers_per_thread * block_size;
    
    std::cout << "=== 寄存器限制检查 ===" << std::endl;
    std::cout << "Block大小: " << block_size << " 线程" << std::endl;
    std::cout << "需要寄存器: " << total_registers_needed << std::endl;
    std::cout << "可用寄存器: " << prop.regsPerBlock << std::endl;
    
    if (total_registers_needed > prop.regsPerBlock) {
        std::cout << "❌ 错误: 寄存器不足!" << std::endl;
        std::cout << "建议: 减少block大小或优化kernel减少寄存器使用" << std::endl;
    } else {
        std::cout << "✅ 寄存器充足" << std::endl;
        double utilization = (double)total_registers_needed / prop.regsPerBlock * 100;
        std::cout << "利用率: " << utilization << "%" << std::endl;
    }
}

int main(){
  check_register_limits(vector_add_local_tile_multi_elem_per_thread_half<PerThread>, 1024);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::vector<half> data(data_size * PerThread * 2);
  void *c_data = nullptr;
  cudaMallocAsync(&c_data, data.size() * sizeof(half) , stream);
  cudaMemcpyAsync(c_data, data.data(), 1024 * 8 * 2 * sizeof(half) ,
                  cudaMemcpyHostToDevice);
  void *d_out = nullptr;
  dim3 block(1024);
  dim3 grid(data_size / (1024 * PerThread * 2));
  cudaMallocAsync(&d_out, sizeof(int) * grid.x, stream);
  cudaEventRecord(start);
  vector_add_local_tile_multi_elem_per_thread_half<PerThread><<<grid,block,0,stream>>>((half * )c_data,1024 *2 * 8,(half *)c_data,(half *)c_data,half(1.0),half(1.0),half(1.0));
  // 停止计时
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("CUDA计算耗时: %.3f ms band width %.3f "
         "GB/s \n",
         milliseconds, (data.size() * sizeof(int) * 2) / (milliseconds * 1e6f));

  cudaFreeAsync(c_data, stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  return 0;
}