#include "gemm/gemm_multi_stage.cuh"

#include <cute/numeric/numeric_types.hpp>

// multi-stage GEMM 全部逻辑在头文件(模板)。此处对默认配置做一次显式实例化,
// 确保模板在库编译期被实际编译、暴露到 libkernel。
namespace kernel {
namespace gpu {

using DefaultGemmConfig = config::GemmConfig<cute::half_t, 128, 128, 32, 3>;

template __global__ void gemm_multi_stage_kernel<DefaultGemmConfig>(void *, const void *,
                                                                    const void *, int, int, int);

}  // namespace gpu
}  // namespace kernel
