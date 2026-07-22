#include "device/device.cuh"

#include <cuda_runtime.h>

#include <cstring>
#include <mutex>
#include <unordered_map>

namespace kernel {
namespace gpu {

namespace {

// 把 name_ 安全地拷进定长缓冲区并保证 '\0' 结尾。
void set_name(Device &dev, const char *name) {
  std::size_t n = std::strlen(name);
  if (n >= sizeof(dev.name_)) { n = sizeof(dev.name_) - 1; }
  std::memcpy(dev.name_, name, n);
  dev.name_[n] = '\0';
}

// 依据 cudaDeviceProp 识别已知型号。
DeviceModel classify(const cudaDeviceProp &prop) {
  if (std::strstr(prop.name, "RTX 4090") != nullptr) { return DeviceModel::Rtx4090; }
  return DeviceModel::Unknown;
}

}  // namespace

Device make_rtx4090() {
  Device dev;
  dev.model_ = DeviceModel::Rtx4090;
  set_name(dev, "NVIDIA GeForce RTX 4090");
  dev.compute_capability_ = 89;                       // Ada Lovelace, SM 8.9
  dev.global_mem_size_ = 24ull * 1024 * 1024 * 1024;  // 24 GB
  dev.max_shared_mem_per_block_ = 99ull * 1024;       // opt-in 上限 99 KB
  dev.shared_mem_per_sm_ = 100ull * 1024;             // 每 SM 100 KB
  dev.sm_count_ = 128;
  dev.max_threads_per_block_ = 1024;
  dev.max_threads_per_sm_ = 1536;
  dev.regs_per_sm_ = 65536;
  dev.warp_size_ = 32;
  return dev;
}

// 从运行时查询值填充能力(未识别型号的回退路径)。
static Device from_prop(const cudaDeviceProp &prop) {
  Device dev;
  dev.model_ = DeviceModel::Unknown;
  set_name(dev, prop.name);
  dev.compute_capability_ = prop.major * 10 + prop.minor;
  dev.global_mem_size_ = prop.totalGlobalMem;
  dev.max_shared_mem_per_block_ = prop.sharedMemPerBlockOptin;
  dev.shared_mem_per_sm_ = prop.sharedMemPerMultiprocessor;
  dev.sm_count_ = prop.multiProcessorCount;
  dev.max_threads_per_block_ = prop.maxThreadsPerBlock;
  dev.max_threads_per_sm_ = prop.maxThreadsPerMultiProcessor;
  dev.regs_per_sm_ = prop.regsPerMultiprocessor;
  dev.warp_size_ = prop.warpSize;
  return dev;
}

// 实际执行一次检测:查询 cudaDeviceProperties 并识别型号。
static Device detect_device(int ordinal) {
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, ordinal) != cudaSuccess) {
    return Device{};  // 查询失败:返回 Unknown 的空能力
  }

  // 识别到已知型号时用精确常量;否则回退到运行时查询值。
  switch (classify(prop)) {
    case DeviceModel::Rtx4090:
      return make_rtx4090();
    case DeviceModel::Unknown:
      break;
  }
  return from_prop(prop);
}

Device get_device(int ordinal) {
  if (ordinal < 0) { cudaGetDevice(&ordinal); }

  // 设备属性在进程生命周期内不变,按 ordinal 缓存,避免重复的
  // cudaGetDeviceProperties 运行时查询。查询失败(Unknown)不写入缓存,
  // 以便后续重试。
  static std::mutex mtx;
  static std::unordered_map<int, Device> cache;

  std::lock_guard<std::mutex> lock(mtx);
  auto it = cache.find(ordinal);
  if (it != cache.end()) { return it->second; }

  Device dev = detect_device(ordinal);
  // 查询失败会返回 compute_capability_ == 0 的空能力,不缓存以便后续重试;
  // 识别成功或"有效但未知型号"(from_prop)都缓存。
  if (dev.compute_capability() > 0) { cache.emplace(ordinal, dev); }
  return dev;
}

}  // namespace gpu
}  // namespace kernel
