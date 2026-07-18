#include "device/device.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstring>
#include <vector>

using kernel::gpu::Device;
using kernel::gpu::DeviceModel;

// ---------------------------------------------------------------------------
// make_rtx4090: 已知规格常量,不依赖运行时,任何环境下都应成立。
// ---------------------------------------------------------------------------
TEST(TestDevice, Rtx4090KnownSpec) {
  Device dev = kernel::gpu::make_rtx4090();

  EXPECT_EQ(dev.model(), DeviceModel::Rtx4090);
  EXPECT_STREQ(dev.name(), "NVIDIA GeForce RTX 4090");
  EXPECT_EQ(dev.compute_capability(), 89);
  EXPECT_EQ(dev.global_mem_size(), 24ull * 1024 * 1024 * 1024);
  EXPECT_EQ(dev.max_shared_mem_per_block(), 99ull * 1024);
  EXPECT_EQ(dev.shared_mem_per_sm(), 100ull * 1024);
  EXPECT_EQ(dev.sm_count(), 128);
  EXPECT_EQ(dev.max_threads_per_block(), 1024);
  EXPECT_EQ(dev.max_threads_per_sm(), 1536);
  EXPECT_EQ(dev.regs_per_sm(), 65536);
  EXPECT_EQ(dev.warp_size(), 32);

  // usable_gmem_for_intermediate() = 80% 全局显存
  EXPECT_EQ(dev.usable_gmem_for_intermediate(), dev.global_mem_size() / 5 * 4);
}

// ---------------------------------------------------------------------------
// 默认构造的 Device 是 Unknown 的空能力。
// ---------------------------------------------------------------------------
TEST(TestDevice, DefaultIsUnknown) {
  Device dev;
  EXPECT_EQ(dev.model(), DeviceModel::Unknown);
  EXPECT_EQ(dev.compute_capability(), 0);
  EXPECT_EQ(dev.global_mem_size(), 0u);
  EXPECT_STREQ(dev.name(), "");
}

// ---------------------------------------------------------------------------
// get_device: 自动检测当前 GPU。无卡环境下跳过(GTEST_SKIP)。
// ---------------------------------------------------------------------------
TEST(TestDevice, GetDeviceAutoDetect) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "无可用 CUDA 设备,跳过自动检测测试";
  }

  Device dev = kernel::gpu::get_device();

  // 无论是否识别为已知型号,以下字段都应由检测填充为正值。
  EXPECT_GT(dev.compute_capability(), 0);
  EXPECT_GT(dev.global_mem_size(), 0u);
  EXPECT_GT(dev.sm_count(), 0);
  EXPECT_GT(dev.max_threads_per_block(), 0);
  EXPECT_EQ(dev.warp_size(), 32);
  EXPECT_NE(dev.name()[0], '\0');

  // 若当前卡就是 4090,应命中已知常量分支。
  if (dev.model() == DeviceModel::Rtx4090) {
    EXPECT_EQ(dev.compute_capability(), 89);
    EXPECT_EQ(dev.sm_count(), 128);
  }
}

// ---------------------------------------------------------------------------
// 设备端可访问性:把 Device 按值传入 kernel,在 device 侧读取其字段。
// 这验证了 __host__ __device__ 访问器 + trivially copyable 的设计。
// ---------------------------------------------------------------------------
__global__ void read_on_device(Device dev, int *out_cc, unsigned long long *out_gmem, int *out_sm) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out_cc = dev.compute_capability();
    *out_gmem = dev.global_mem_size();
    *out_sm = dev.sm_count();
  }
}

TEST(TestDevice, AccessibleOnDevice) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "无可用 CUDA 设备,跳过设备端访问测试";
  }

  Device dev = kernel::gpu::make_rtx4090();

  int *d_cc = nullptr;
  unsigned long long *d_gmem = nullptr;
  int *d_sm = nullptr;
  ASSERT_EQ(cudaMalloc(&d_cc, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_gmem, sizeof(unsigned long long)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_sm, sizeof(int)), cudaSuccess);

  read_on_device<<<1, 32>>>(dev, d_cc, d_gmem, d_sm);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int h_cc = 0, h_sm = 0;
  unsigned long long h_gmem = 0;
  cudaMemcpy(&h_cc, d_cc, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_gmem, d_gmem, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_sm, d_sm, sizeof(int), cudaMemcpyDeviceToHost);

  // device 侧读到的值应与 host 侧一致。
  EXPECT_EQ(h_cc, dev.compute_capability());
  EXPECT_EQ(h_gmem, dev.global_mem_size());
  EXPECT_EQ(h_sm, dev.sm_count());

  cudaFree(d_cc);
  cudaFree(d_gmem);
  cudaFree(d_sm);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
