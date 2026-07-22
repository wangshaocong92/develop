#include "attention/flash_attention/softmax_reg.cuh"

// Minimal compile-only test for softmax_reg.cuh
__global__ void dummy_kernel() {
  using namespace kernel::gpu;
  // Just instantiate something
  auto t = make_tensor<float>(Shape<_1>{});
}

int main() {
  return 0;
}
