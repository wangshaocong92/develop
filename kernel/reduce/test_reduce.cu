#include "reduce.cuh"
#include <algorithm>
#include <cstring>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
// 简单的测试用例
#define ARRAY_SIZE 1 << 28
#define BLOCK_SIZE 1024
std::vector<int> data(ARRAY_SIZE);
int total = 0;
TEST(TestReduce, reduce_with_divergemnt_warps) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  void *d_data = nullptr;
  cudaMallocAsync(&d_data, data.size() * sizeof(int), stream);
  cudaMemcpyAsync(d_data, data.data(), data.size() * sizeof(int),
                  cudaMemcpyHostToDevice);
  void *d_out = nullptr;
  dim3 block(BLOCK_SIZE);
  dim3 grid((data.size() + block.x - 1) / block.x);
  cudaMallocAsync(&d_out, sizeof(int) * grid.x, stream);
  cudaEventRecord(start);
  kernel::reduce_with_divergemnt_warps<BLOCK_SIZE>
      <<<grid, block, 0, stream>>>((int *)d_data, (int *)d_out);
  // 停止计时
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("reduce_with_divergemnt_warps CUDA计算耗时: %.3f ms band width %.3f "
         "GB/s \n",
         milliseconds, (data.size() * sizeof(int) * 2) / (milliseconds * 1e6f));
  std::vector<int> out(grid.x);
  cudaMemcpyAsync(out.data(), d_out, sizeof(int) * grid.x,
                  cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(d_data, stream);
  cudaFreeAsync(d_out, stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  EXPECT_EQ(total, std::reduce(out.begin(), out.end(), 0)); // 期望相等
}

TEST(TestReduce, reduce_with_interleaved_addressing) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  void *d_data = nullptr;
  cudaMallocAsync(&d_data, data.size() * sizeof(int), stream);
  cudaMemcpyAsync(d_data, data.data(), data.size() * sizeof(int),
                  cudaMemcpyHostToDevice);
  void *d_out = nullptr;
  dim3 block(BLOCK_SIZE);
  dim3 grid((data.size() + block.x - 1) / block.x);
  cudaMallocAsync(&d_out, sizeof(int) * grid.x, stream);
  cudaEventRecord(start);
  kernel::reduce_with_interleaved_addressing<BLOCK_SIZE>
      <<<grid, block, 0, stream>>>((int *)d_data, (int *)d_out);
  // 停止计时
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("reduce_with_interleaved_addressing CUDA计算耗时: %.3f ms band width "
         "%.3f GB/s \n",
         milliseconds, (data.size() * sizeof(int) * 2) / (milliseconds * 1e6f));
  std::vector<int> out(grid.x);
  cudaMemcpyAsync(out.data(), d_out, sizeof(int) * grid.x,
                  cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(d_data, stream);
  cudaFreeAsync(d_out, stream);
  cudaStreamSynchronize(stream);
 cudaStreamDestroy(stream);
 EXPECT_EQ(total, std::reduce(out.begin(), out.end(), 0)); // 期望相等
}

TEST(TestReduce, reduce_with_sequential_addressing) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  void *d_data = nullptr;
  cudaMallocAsync(&d_data, data.size() * sizeof(int), stream);
  cudaMemcpyAsync(d_data, data.data(), data.size() * sizeof(int),
                  cudaMemcpyHostToDevice);
  void *d_out = nullptr;

  dim3 block(BLOCK_SIZE);
  dim3 grid((data.size() + block.x - 1) / block.x);
  cudaMallocAsync(&d_out, sizeof(int) * grid.x, stream);
  cudaEventRecord(start);
  kernel::reduce_with_sequential_addressing<BLOCK_SIZE>
      <<<grid, block, 0, stream>>>((int *)d_data, (int *)d_out);
  // 停止计时
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("reduce_with_sequential_addressing CUDA计算耗时: %.3f ms band width "
         "%.3f GB/s \n",
         milliseconds, (data.size() * sizeof(int) * 2) / (milliseconds * 1e6f));

  std::vector<int> out(grid.x);
  cudaMemcpyAsync(out.data(), d_out, sizeof(int) * grid.x,
                  cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(d_data, stream);
  cudaFreeAsync(d_out, stream);
  cudaStreamSynchronize(stream);
   cudaStreamDestroy(stream);

   EXPECT_EQ(total, std::reduce(out.begin(), out.end(), 0)); // 期望相等
}

TEST(TestReduce, reduce_add_with_load) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  void *d_data = nullptr;
  cudaMallocAsync(&d_data, data.size() * sizeof(int), stream);
  cudaMemcpyAsync(d_data, data.data(), data.size() * sizeof(int),
                  cudaMemcpyHostToDevice);
  void *d_out = nullptr;

  dim3 block(BLOCK_SIZE);
  dim3 grid((data.size() + block.x - 1) / block.x / 4);
  cudaMallocAsync(&d_out, sizeof(int) * grid.x, stream);
  cudaEventRecord(start);
  kernel::reduce_add_with_load<BLOCK_SIZE>
      <<<grid, block, 0, stream>>>((int *)d_data, (int *)d_out);
  // 停止计时
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("reduce_add_with_load CUDA计算耗时: %.3f ms band width %.3f GB/s \n",
         milliseconds, (data.size() * sizeof(int) * 2) / (milliseconds * 1e6f));

  std::vector<int> out(grid.x);
  cudaMemcpyAsync(out.data(), d_out, sizeof(int) * grid.x,
                  cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(d_data, stream);
  cudaFreeAsync(d_out, stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  EXPECT_EQ(total, std::reduce(out.begin(), out.end(), 0)); // 期望相等
}

TEST(TestReduce, reduce_with_no_roll) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  void *d_data = nullptr;
  cudaMallocAsync(&d_data, data.size() * sizeof(int), stream);
  cudaMemcpyAsync(d_data, data.data(), data.size() * sizeof(int),
                  cudaMemcpyHostToDevice);
  void *d_out = nullptr;

  dim3 block(BLOCK_SIZE);
  dim3 grid((data.size() + block.x - 1) / block.x / 4);
  cudaMallocAsync(&d_out, sizeof(int) * grid.x, stream);
  cudaEventRecord(start);
  kernel::reduce_with_no_roll_last_warp<int, BLOCK_SIZE, kernel::AddOp>
      <<<grid, block, 0, stream>>>((int *)d_data, (int *)d_out);
  // 停止计时
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("reduce_with_no_roll CUDA计算耗时: %.3f ms band width %.3f GB/s \n",
         milliseconds, (data.size() * sizeof(int) * 2) / (milliseconds * 1e6f));

  std::vector<int> out(grid.x);
  cudaMemcpyAsync(out.data(), d_out, sizeof(int) * grid.x,
                  cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(d_data, stream);
  cudaFreeAsync(d_out, stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  EXPECT_EQ(total, std::reduce(out.begin(), out.end(), 0)); // 期望相等
}

TEST(TestReduce, reduce) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  void *d_data = nullptr;
  cudaMallocAsync(&d_data, data.size() * sizeof(int), stream);
  cudaMemcpyAsync(d_data, data.data(), data.size() * sizeof(int),
                  cudaMemcpyHostToDevice);
  void *d_out = nullptr;

  dim3 block(BLOCK_SIZE);
  dim3 grid((data.size() + block.x - 1) / block.x / 4);
  cudaMallocAsync(&d_out, sizeof(int) * grid.x, stream);
  cudaEventRecord(start);
  kernel::reduce<BLOCK_SIZE>
      <<<grid, block, 0, stream>>>((int *)d_data, (int *)d_out, data.size());
  // 停止计时
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("reduce CUDA计算耗时: %.3f ms band width %.3f GB/s \n", milliseconds,
         (data.size() * sizeof(int) * 2) / (milliseconds * 1e6f));

  std::vector<int> out(grid.x);
  cudaMemcpyAsync(out.data(), d_out, sizeof(int) * grid.x,
                  cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(d_data, stream);
  cudaFreeAsync(d_out, stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  EXPECT_EQ(total, std::reduce(out.begin(), out.end(), 0)); // 期望相等
}

// 主函数
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  // 启用标准输出捕获（默认开启）
  testing::GTEST_FLAG(catch_exceptions) = false;

  // 设置随机数生成器
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 10); // 设置随机数范围
  // 使用算法填充向量
  std::generate(data.begin(), data.end(), [&]() { return dis(gen); });

  total = std::reduce(data.begin(), data.end(), 0);

  // 运行所有测试
  return RUN_ALL_TESTS();
}
