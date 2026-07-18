# Device 设备能力抽象

描述一块 GPU 的硬件能力(显存、共享内存、SM 数、线程/寄存器上限等),
供 flash attention 的分块 / 多卡调度策略查询。

- 头文件:[device.cuh](device.cuh)
- 实现:[device.cu](device.cu)
- 单元测试:[../test/test_device.cu](../test/test_device.cu)

## 设计要点

| 要点 | 说明 |
|------|------|
| 纯数据 + trivially copyable | `Device` 只是一个 POD 结构体,可按值拷贝、按值传参 |
| `__host__ __device__` 访问器 | host 侧做调度决策,device 侧 kernel 内部同样可读 |
| 不使用虚函数 | 虚表无法跨 host/device 边界;型号差异用 `DeviceModel` 字段 + 工厂函数表达 |
| 自动检测 + 回退 | 识别到已知型号填精确常量,否则回退到 `cudaGetDeviceProperties` 查询值 |

## 接口

```cpp
namespace kernel::gpu {

enum class DeviceModel { Unknown, Rtx4090 };

struct Device {
  DeviceModel  model() const;
  const char*  name() const;
  int          compute_capability() const;   // major*10+minor,如 Ada = 89
  std::size_t  global_mem_size() const;       // bytes
  std::size_t  max_shared_mem_per_block() const;
  std::size_t  shared_mem_per_sm() const;
  int          sm_count() const;
  int          max_threads_per_block() const;
  int          max_threads_per_sm() const;
  int          regs_per_sm() const;
  int          warp_size() const;
  std::size_t  usable_gmem_for_intermediate() const;  // 默认 80% 全局显存
};

Device make_rtx4090();          // RTX 4090 的编译期已知规格,不依赖运行时
Device get_device(int ordinal = -1);  // 自动检测;ordinal<0 用当前设备
}
```

## 使用说明

### 1. 自动检测当前 GPU(host 侧)

```cpp
#include "device/device.cuh"

kernel::gpu::Device dev = kernel::gpu::get_device();   // 检测当前设备
printf("%s (SM %d), %zu GB, %d SMs\n",
       dev.name(), dev.compute_capability(),
       dev.global_mem_size() >> 30, dev.sm_count());
```

指定某张卡:

```cpp
kernel::gpu::Device dev1 = kernel::gpu::get_device(1);  // ordinal = 1
```

### 2. 用于调度决策

`cpu::flash_attention_forward` 里的单卡显存预算就来自这里,取代了此前硬编码的 2GB:

```cpp
const size_t budget = kernel::gpu::get_device().usable_gmem_for_intermediate();
if (sizeof(float) * M * M > budget) {
  // 中间矩阵放不下 → 需要多卡协同
}
```

### 3. 设备端访问(按值传入 kernel)

`Device` 是 trivially copyable 的,可以直接按值传进 kernel,在设备端读取:

```cpp
__global__ void my_kernel(kernel::gpu::Device dev, /* ... */) {
  int lanes = dev.warp_size();       // device 侧直接查询
  int smem  = dev.shared_mem_per_sm();
  // ... 据此做 block 内分块
}

kernel::gpu::Device dev = kernel::gpu::get_device();
my_kernel<<<grid, block>>>(dev, /* ... */);   // 按值传递即可
```

### 4. 无卡 / 静态场景

需要 RTX 4090 的已知规格而不想触发运行时查询(如编译期配置、无 GPU 的 CI):

```cpp
kernel::gpu::Device dev = kernel::gpu::make_rtx4090();
```

## 扩展新型号

1. 在 `DeviceModel` 枚举中新增型号。
2. 在 [device.cu](device.cu) 中新增 `make_xxx()` 工厂函数,填入规格常量。
3. 在 `classify()` 中依据 `cudaDeviceProp`(如 `prop.name`)识别该型号。
4. 在 `get_device()` 的 `switch` 中把新型号接到对应工厂函数。

## 运行测试

```bash
cmake --build build --target device_test
./build/kernel/test/device_test
```

测试覆盖:已知规格常量、默认 Unknown、自动检测、以及**设备端按值访问**(把
`Device` 传入 kernel 后在 device 侧读回字段并与 host 侧比对)。无 CUDA 设备
时,依赖 GPU 的用例会 `GTEST_SKIP` 跳过。
