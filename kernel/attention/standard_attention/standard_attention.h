#pragma once
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <stdexcept>
#include "softmax/softmax.cuh"


namespace kernel {
namespace cpu {
using namespace cute;
using CutlassSgemm =
    cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float,
                                cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor,
                                float, cutlass::arch::OpClassSimt, cutlass::arch::Sm89>;
using CutlassSgemm2 =
    cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor,
                                float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt,
                                cutlass::arch::Sm89>;

template <int M, int N,
          class Tensor2D = Tensor<float, Layout<Shape<Int<M>, Int<N>>, Stride<Int<1>, Int<M>>>>>
void standard_attention_forward(const Tensor2D &q, const Tensor2D &k, const Tensor2D &v,
                                Tensor2D &out) {
  // S = Q · K^T, 行主序输出 [M,M]
  float *qk_ptr = nullptr;
  auto layout_qk = Layout<Shape<Int<M>, Int<M>>, Stride<Int<M>, Int<1>>>{};
  cudaMalloc(&qk_ptr, sizeof(float) * M * M);
  auto qk = make_tensor(qk_ptr, layout_qk);

  typename CutlassSgemm::Arguments args({M, M, N}, {q.data(), N},  // A = Q,   RowMajor,    lda = N
                                        {k.data(), N},             // B = K^T, ColumnMajor, ldb = N
                                        {qk.data(), M},            // C,       RowMajor,    ldc = M
                                        {qk.data(), M},            // D
                                        {1.0f, 0.0f});
  CutlassSgemm gemm_op;
  cutlass::Status status = gemm_op(args);
  if (status != cutlass::Status::kSuccess) throw std::runtime_error("gemm1 failed");

  // softmax: 逐行归一化 (qk 为 M×M 行主序，每行 M 个元素连续)
  kernel::gpu::host_softmax_forward<M, M>(qk);

  // O = softmax(S) · V, 行主序输出 [M,N]
  float *ot_ptr = nullptr;
  cudaMalloc(&ot_ptr, sizeof(float) * M * N);
  typename CutlassSgemm2::Arguments args2({M, N, M}, {qk.data(), M},  // A = P, RowMajor, lda = M
                                          {v.data(), N},              // B = V, RowMajor, ldb = N
                                          {ot_ptr, N},                // C,     RowMajor, ldc = N
                                          {ot_ptr, N},                // D
                                          {1.0f, 0.0f});
  status = CutlassSgemm2()(args2);
  if (status != cutlass::Status::kSuccess) throw std::runtime_error("gemm2 failed");

  // GEMM2 已直接产出行主序 O[M,N], 直接拷贝到 out
  cudaMemcpy(out.data(), ot_ptr, sizeof(float) * M * N, cudaMemcpyDeviceToDevice);

  cudaFree(qk_ptr);
  cudaFree(ot_ptr);
  return;
}
}  // namespace cpu
}  // namespace kernel
