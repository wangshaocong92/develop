#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>

__global__ void kernel_with_printf() {
  printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

TEST(CudaPrintfTest, Basic) {
  kernel_with_printf<<<2, 4>>>();
  cudaDeviceSynchronize(); // 必须要同步才能 flush printf
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
