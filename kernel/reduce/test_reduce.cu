#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <numeric>
#include <vector>
#include <random>
#include <algorithm>
#include "reduce.h"


// 简单的测试用例
TEST(TestReduce, TestTimeConsuming) {
    std::vector<int> data(1 << 20);
      // 设置随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 10); // 设置随机数范围
    // 使用算法填充向量
    std::generate(data.begin(), data.end(), [&]() { return dis(gen); });

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void *d_data = nullptr;
    cudaMallocAsync(&d_data, data.size() * sizeof(int),stream);
    cudaMemcpyAsync(d_data, data.data(), data.size() * sizeof(int), cudaMemcpyHostToDevice,stream);
    void *d_out = nullptr;
    dim3 block(1024);
    dim3 grid((data.size() + block.x - 1) / block.x);
    cudaMallocAsync(&d_out, sizeof(int) *  grid.x,stream);


    kernel::reduce_with_divergemnt_warps<<<grid, block, 0, stream>>>((int *)d_data, (int *)d_out);

    std::vector<int> out(grid.x);
    cudaMemcpyAsync(out.data(), d_out, sizeof(int) * grid.x, cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_data,stream);
    cudaFreeAsync(d_out,stream);
    cudaStreamDestroy(stream);

    EXPECT_EQ(std::reduce(data.begin(), data.end(), 0), std::reduce(out.begin(), out.end(), 0));  // 期望相等
}