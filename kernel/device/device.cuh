#pragma once

#include <cstddef>

namespace kernel {
namespace gpu {

// 已知的 GPU 型号。未识别的卡归为 Unknown,此时能力字段由运行时
// cudaGetDeviceProperties 填充。
enum class DeviceModel : int {
  Unknown = 0,
  Rtx4090,
};

// 描述一块 GPU 的硬件能力,供 flash attention 的分块 / 多卡调度策略查询。
//
// 设计要点:
//   * 纯数据 + trivially copyable —— 可按值传入 kernel,在设备端直接读取。
//   * 所有访问器都是 __host__ __device__ —— host 侧做调度决策,device 侧
//     kernel 内部同样可以查询(例如按 sm_count / 共享内存做分块)。
//   * 不使用虚函数 —— 虚表无法跨 host/device 边界,型号差异通过 model
//     字段 + 工厂函数体现。
struct Device {
  DeviceModel model_ = DeviceModel::Unknown;

  char name_[256] = {};
  int compute_capability_ = 0;  // major * 10 + minor,如 Ada = 89

  std::size_t global_mem_size_ = 0;           // 全局显存总量 (bytes)
  std::size_t max_shared_mem_per_block_ = 0;  // 单 block 最大共享内存 (含 opt-in)
  std::size_t shared_mem_per_sm_ = 0;         // 单 SM 共享内存 (bytes)

  int sm_count_ = 0;               // SM(流多处理器)数量
  int max_threads_per_block_ = 0;  // 单 block 最大线程数
  int max_threads_per_sm_ = 0;     // 单 SM 最大常驻线程数
  int regs_per_sm_ = 0;            // 单 SM 的 32 位寄存器数量
  int warp_size_ = 32;

  // ---- 访问器:host / device 皆可调用 ----
  __host__ __device__ DeviceModel model() const { return model_; }
  __host__ __device__ const char *name() const { return name_; }
  __host__ __device__ int compute_capability() const { return compute_capability_; }

  __host__ __device__ std::size_t global_mem_size() const { return global_mem_size_; }
  __host__ __device__ std::size_t max_shared_mem_per_block() const {
    return max_shared_mem_per_block_;
  }
  __host__ __device__ std::size_t shared_mem_per_sm() const { return shared_mem_per_sm_; }

  __host__ __device__ int sm_count() const { return sm_count_; }
  __host__ __device__ int max_threads_per_block() const { return max_threads_per_block_; }
  __host__ __device__ int max_threads_per_sm() const { return max_threads_per_sm_; }
  __host__ __device__ int regs_per_sm() const { return regs_per_sm_; }
  __host__ __device__ int warp_size() const { return warp_size_; }

  // 给中间矩阵预留的、单卡可用的显存上界(默认取全局显存的 80%)。
  __host__ __device__ std::size_t usable_gmem_for_intermediate() const {
    return global_mem_size_ / 5 * 4;
  }
};

// NVIDIA GeForce RTX 4090 (Ada Lovelace, AD102, SM 8.9) 的已知规格。
// 不依赖运行时,可用于编译期 / 无卡环境下的静态配置。
__host__ Device make_rtx4090();

// 自动检测当前(或指定 ordinal)的 GPU 并返回其能力。
//   ordinal < 0 时使用 cudaGetDevice() 得到的当前设备。
// 识别到已知型号时填充精确常量,否则回退到 cudaGetDeviceProperties 查询值
// (model = Unknown)。
__host__ Device get_device(int ordinal = -1);

}  // namespace gpu
}  // namespace kernel
