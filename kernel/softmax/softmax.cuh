#pragma once
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstring>
#include <cub/cub.cuh>
#include <cute/tensor.hpp>
#include <numeric>
#include <vector>
namespace kernel {
    namespace gpu {
        using namespace cute;
        /*
            输入是一个一维数组，长度为 n，在 GPU 上计算 softmax 的前向传播，在原地输出，长度为 n。
            1. 使用多少个block，单个block的线程数为多少
            2. block 内做reduce
            n 如何拆分呢，多出来的如何做
            3. 单个block 分多少个数据


        */
        // 2D 版：tensor 为 M×N（行主序），逐行归约，d_max[i*gridDim.x + blockIdx.x] 存第 i 行第 blockIdx.x 段的最大值。
        // 一维数组作为 M=1 的退化特例走此 kernel。
        template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int M, int N,
                  class Tensor2D =
                      cute::Tensor<float, Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>>>
        __global__ void softmax_max(Tensor2D tensor, float *d_max) {
          using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
          __shared__ typename BlockReduce::TempStorage temp_storage;
          const int tid = threadIdx.x;
          const int global_index = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD + tid;

          for (auto i = 0; i < M; ++i) {
            // 第 i 行首元素（行主序 stride=1，行内连续）
            auto input = tensor(i, _).data();
            int idx = global_index;

            // 越界填 -inf，避免负值行里 0 被误当成最大值
            float val = (idx < N) ? input[idx] : -INFINITY;
            float thread_max = val;
#pragma unroll
            for (int j = 1; j < ITEMS_PER_THREAD; ++j) {
              idx += BLOCK_THREADS;  // 跨步访问，合并内存读取
              val = (idx < N) ? input[idx] : -INFINITY;
              thread_max = fmaxf(thread_max, val);
            }
            // block 内归约
            float block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
            if (tid == 0) d_max[i * gridDim.x + blockIdx.x] = block_max;
            __syncthreads();  // 保护下一轮 temp_storage 复用
          }
        }

        template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int M, int N,
                  class Tensor2D =
                      cute::Tensor<float, Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>>>
        __global__ void softmax_sum(Tensor2D tensor, float *d_sum, float *max) {
          using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
          __shared__ typename BlockReduce::TempStorage temp_storage;
          const int tid = threadIdx.x;
          const int global_index = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD + tid;

          for (auto i = 0; i < M; ++i) {
            // 加载 + 线程内归约合并
            auto input = tensor(i, _).data();
            // 加载 + 线程内归约合并
            int idx = global_index;

            float val = (idx < N) ? input[idx] : 0.0f;
            float thread_sum = (idx < N) ? expf(val - max[i]) : 0.0f;
#pragma unroll
            for (int j = 1; j < ITEMS_PER_THREAD; ++j) {
              idx += BLOCK_THREADS;  // 跨步访问，合并内存读取
              val = (idx < N) ? input[idx] : 0.0f;
              thread_sum += (idx < N) ? expf(val - max[i]) : 0.0f;
            }
            // block 内归约
            float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
            if (tid == 0) d_sum[i * gridDim.x + blockIdx.x] = block_sum;
            __syncthreads();
          }
        }

        template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
        __global__ void softmax_forward(float *input, double sum,double max,int N) {
            const int tid = threadIdx.x;
            const int global_index = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD + tid;

            // 加载 + 线程内归约合并
            int idx = global_index;
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
              if (idx < N) input[idx] = expf(input[idx] - max) / sum;
              idx += BLOCK_THREADS;  // 跨步访问，合并内存读取
            }
        }

        // 2D 版：tensor 为 M×N（行主序），逐行归一化。sum/max 为 device 上长度 M 的逐行数组。
        // 一维数组作为 M=1 的退化特例走此 kernel。
        template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int M, int N,
                  class Tensor2D =
                      cute::Tensor<float, Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>>>
        __global__ void softmax_forward(Tensor2D tensor, const float *sum, const float *max) {
          const int tid = threadIdx.x;
          const int global_index = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD + tid;

          for (auto i = 0; i < M; ++i) {
            auto input = tensor(i, _).data();
            const float row_sum = sum[i];
            const float row_max = max[i];
            int idx = global_index;
#pragma unroll
            for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
              if (idx < N) input[idx] = expf(input[idx] - row_max) / row_sum;
              idx += BLOCK_THREADS;  // 跨步访问，合并内存读取
            }
          }
        }
        // 2D softmax：tensor 为 M×N（行主序），逐行归一化，原地输出。
        template <int M, int N,
                  class Tensor2D =
                      cute::Tensor<float, Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>>>
        inline void host_softmax_forward(Tensor2D &tensor) {
          constexpr int BLOCK_THREADS = 256;
          constexpr int ITEMS_PER_THREAD = 4;
          constexpr int ITEMS_PER_BLOCK = BLOCK_THREADS * ITEMS_PER_THREAD;
          // grid 沿 N 切段（每行独立），共 NUM_BLOCKS 段
          constexpr int NUM_BLOCKS = (N + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;

          // 每行 NUM_BLOCKS 个部分结果，形状 M×NUM_BLOCKS
          std::vector<float> h_max(M * NUM_BLOCKS);
          std::vector<float> h_sum(M * NUM_BLOCKS);
          float *d_max = nullptr;    // M×NUM_BLOCKS 部分 max
          float *d_sum = nullptr;    // M×NUM_BLOCKS 部分 sum
          float *d_row_max = nullptr;  // 长度 M 的逐行 max
          float *d_row_sum = nullptr;  // 长度 M 的逐行 sum
          cudaMalloc(&d_max, M * NUM_BLOCKS * sizeof(float));
          cudaMalloc(&d_sum, M * NUM_BLOCKS * sizeof(float));
          cudaMalloc(&d_row_max, M * sizeof(float));
          cudaMalloc(&d_row_sum, M * sizeof(float));

          // 1) 逐行分段求 max
          softmax_max<BLOCK_THREADS, ITEMS_PER_THREAD, M, N>
              <<<NUM_BLOCKS, BLOCK_THREADS>>>(tensor, d_max);
          cudaMemcpy(h_max.data(), d_max, M * NUM_BLOCKS * sizeof(float),
                     cudaMemcpyDeviceToHost);
          // host 侧把每行的 NUM_BLOCKS 个部分 max 归约成逐行 max
          std::vector<float> row_max(M);
          for (int i = 0; i < M; ++i) {
            row_max[i] = *std::max_element(h_max.begin() + i * NUM_BLOCKS,
                                           h_max.begin() + (i + 1) * NUM_BLOCKS);
          }
          cudaMemcpy(d_row_max, row_max.data(), M * sizeof(float),
                     cudaMemcpyHostToDevice);

          // 2) 逐行分段求 sum(exp(x - row_max))
          softmax_sum<BLOCK_THREADS, ITEMS_PER_THREAD, M, N>
              <<<NUM_BLOCKS, BLOCK_THREADS>>>(tensor, d_sum, d_row_max);
          cudaMemcpy(h_sum.data(), d_sum, M * NUM_BLOCKS * sizeof(float),
                     cudaMemcpyDeviceToHost);
          std::vector<float> row_sum(M);
          for (int i = 0; i < M; ++i) {
            row_sum[i] = std::accumulate(h_sum.begin() + i * NUM_BLOCKS,
                                         h_sum.begin() + (i + 1) * NUM_BLOCKS, 0.0f);
          }
          cudaMemcpy(d_row_sum, row_sum.data(), M * sizeof(float),
                     cudaMemcpyHostToDevice);

          // 3) 逐行归一化，原地写回
          softmax_forward<BLOCK_THREADS, ITEMS_PER_THREAD, M, N>
              <<<NUM_BLOCKS, BLOCK_THREADS>>>(tensor, d_row_sum, d_row_max);

          cudaFree(d_max);
          cudaFree(d_sum);
          cudaFree(d_row_max);
          cudaFree(d_row_sum);
        }

        // 1D softmax：作为 M=1 的退化特例复用 2D 路径。N 为编译期长度。
        template <int N>
        inline void host_softmax_forward(float *input) {
          cute::Layout<cute::Shape<cute::Int<1>, cute::Int<N>>,
                       cute::Stride<cute::Int<N>, cute::Int<1>>>
              layout{};
          auto tensor = cute::make_tensor(input, layout);
          host_softmax_forward<1, N>(tensor);
        }


          template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
        __global__ void softmax_max_sum(const float *input, float *d_max, float *d_sum, int N) {
            constexpr int CHUNK = BLOCK_THREADS * ITEMS_PER_THREAD;
            using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            __shared__ float smem[CHUNK];
            const int tid = threadIdx.x;
            const int base = blockIdx.x * CHUNK;

            // ---- 一次 global load → SMEM (coalesced) ----
            int idx = base + tid;
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                smem[tid + i * BLOCK_THREADS] = (idx < N) ? input[idx] : 0.0f;
                idx += BLOCK_THREADS;
            }
            __syncthreads();

            // ---- max (from SMEM) ----
            float val = smem[tid];
            float thread_max = val;
            #pragma unroll
            for (int i = 1; i < ITEMS_PER_THREAD; ++i) {
                val = smem[tid + i * BLOCK_THREADS];
                thread_max = fmaxf(thread_max, val);
            }
            float block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());

            // ---- sum (from SMEM) ----
            float thread_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                val = smem[tid + i * BLOCK_THREADS];
                thread_sum += expf(val - block_max);
            }
            float block_sum = BlockReduce(temp_storage).Sum(thread_sum);

            if (tid == 0) {
                d_max[blockIdx.x] = block_max;
                d_sum[blockIdx.x] = block_sum;
            }
        }

        inline void host_online_softmax_forward(float *input, int n){
            constexpr int BLOCK_THREADS = 256;
            constexpr int ITEMS_PER_THREAD = 4;
            const int ITEMS_PER_BLOCK = BLOCK_THREADS * ITEMS_PER_THREAD;
            const int NUM_BLOCKS = (n + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;
            std::vector<float> h_max(NUM_BLOCKS);
            std::vector<float> h_sum(NUM_BLOCKS);
            float *d_max = nullptr;
            float *d_sum = nullptr;
            cudaMalloc(&d_max, NUM_BLOCKS * sizeof(float));
            cudaMalloc(&d_sum, NUM_BLOCKS * sizeof(float));
            softmax_max_sum<BLOCK_THREADS, ITEMS_PER_THREAD><<<NUM_BLOCKS, BLOCK_THREADS>>>(input, d_max, d_sum, n);
            cudaMemcpy(h_max.data(), d_max, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_sum.data(), d_sum, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);
            float max = *std::max_element(h_max.begin(), h_max.end());
            double sum = 0.0;
            for(auto i = 0; i < NUM_BLOCKS; ++i){
                sum += h_sum[i] * expf(h_max[i] - max);
            }
            softmax_forward<BLOCK_THREADS, ITEMS_PER_THREAD><<<NUM_BLOCKS, BLOCK_THREADS>>>(input, sum, max, n);
            cudaFree(d_max);
            cudaFree(d_sum);
        }

        // ====================================================================
        // 方案 A: Cooperative Groups, 单 kernel 内用 grid.sync() 跨 block 同步
        //         全程 device 上完成 max → sum → normalize, 无 host 往返、单次 launch
        // ====================================================================
        // 基于 CAS 的 float atomicMax (对负数也正确, 不能用 int 位重解释的 atomicMax)
        __device__ inline float atomicMaxFloat(float *addr, float val) {
            int *iaddr = (int *)addr;
            int old = *iaddr, assumed;
            do {
                assumed = old;
                float cur = __int_as_float(assumed);
                if (cur >= val) break;  // 已经不小于, 无需更新
                old = atomicCAS(iaddr, assumed, __float_as_int(val));
            } while (old != assumed);
            return __int_as_float(old);
        }

        template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
        __global__ void softmax_coop(float *input, int N, float *g_max, float *g_sum) {
            namespace cg = cooperative_groups;
            cg::grid_group grid = cg::this_grid();
            using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
            __shared__ typename BlockReduce::TempStorage temp_storage;

            const int tid = threadIdx.x;
            const int stride = gridDim.x * BLOCK_THREADS;  // grid-stride: block 数受 occupancy 限制

            // ---- 1) 全局 max: 每 block 局部 max → atomicMaxFloat 到 g_max ----
            float thread_max = -INFINITY;
            for (int idx = blockIdx.x * BLOCK_THREADS + tid; idx < N; idx += stride) {
                thread_max = fmaxf(thread_max, input[idx]);
            }
            float block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
            if (tid == 0) atomicMaxFloat(g_max, block_max);
            grid.sync();  // 等所有 block 写完 max

            // ---- 2) 全局 sum: Σ exp(x - gmax) → atomicAdd 到 g_sum ----
            const float gmax = *g_max;
            float thread_sum = 0.0f;
            for (int idx = blockIdx.x * BLOCK_THREADS + tid; idx < N; idx += stride) {
                thread_sum += expf(input[idx] - gmax);
            }
            __syncthreads();  // 复用 temp_storage 前同步
            float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
            if (tid == 0) atomicAdd(g_sum, block_sum);
            grid.sync();  // 等所有 block 累加完 sum

            // ---- 3) 归一化 ----
            const float gsum = *g_sum;
            for (int idx = blockIdx.x * BLOCK_THREADS + tid; idx < N; idx += stride) {
                input[idx] = expf(input[idx] - gmax) / gsum;
            }
        }

        inline void host_coop_softmax_forward(float *input, int n) {
            constexpr int BLOCK_THREADS = 256;
            constexpr int ITEMS_PER_THREAD = 4;

            float *g_max = nullptr, *g_sum = nullptr;
            cudaMalloc(&g_max, sizeof(float));
            cudaMalloc(&g_sum, sizeof(float));
            // g_max 初始化为 -inf (以 int 位模式写入 atomicMax 用), g_sum 清零
            float ninf = -INFINITY, zero = 0.0f;
            cudaMemcpy(g_max, &ninf, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(g_sum, &zero, sizeof(float), cudaMemcpyHostToDevice);

            // cooperative launch 要求 block 数不超过可同时驻留的上限
            int dev = 0;
            cudaGetDevice(&dev);
            int num_sm = 0;
            cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev);
            int blocks_per_sm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm, softmax_coop<BLOCK_THREADS, ITEMS_PER_THREAD>,
                BLOCK_THREADS, 0);
            int max_blocks = num_sm * blocks_per_sm;
            const int ITEMS_PER_BLOCK = BLOCK_THREADS * ITEMS_PER_THREAD;
            int wanted = (n + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;
            int grid = wanted < max_blocks ? wanted : max_blocks;
            if (grid < 1) grid = 1;

            void *args[] = {&input, &n, &g_max, &g_sum};
            cudaLaunchCooperativeKernel(
                (void *)softmax_coop<BLOCK_THREADS, ITEMS_PER_THREAD>, grid,
                BLOCK_THREADS, args, 0, 0);
            cudaDeviceSynchronize();

            cudaFree(g_max);
            cudaFree(g_sum);
        }

        // ====================================================================
        // 方案 B: 纯 atomic, 不用 grid.sync。两个 kernel:
        //   B1 online-merge: 每 block 产出局部 (m, l), 用 atomicCAS 自旋合并到全局 (M, L)
        //   B2 normalize: 用全局 (M, L) 归一化
        // 依赖链 sum→max 靠 online rescale 公式在合并时就地解决, 无需先定全局 max
        // ====================================================================
        // 把全局 (M, L) 打包进一个 unsigned long long, 用 CAS 原子地读-改-写
        __device__ inline void atomic_merge_ml(unsigned long long *state, float m_i, float l_i) {
            unsigned long long old = *state, assumed;
            do {
                assumed = old;
                float M = __int_as_float((int)(assumed >> 32));
                float L = __int_as_float((int)(assumed & 0xffffffffULL));
                float M_new = fmaxf(M, m_i);
                // 两边 rescale 到 M_new 后相加; 初始 M=-inf 时 exp(-inf)=0 天然正确
                float L_new = L * expf(M - M_new) + l_i * expf(m_i - M_new);
                unsigned long long packed =
                    ((unsigned long long)(unsigned)__float_as_int(M_new) << 32) |
                    (unsigned)__float_as_int(L_new);
                old = atomicCAS(state, assumed, packed);
            } while (old != assumed);
        }

        template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
        __global__ void softmax_atomic_reduce(const float *input, int N,
                                              unsigned long long *state) {
            using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            const int tid = threadIdx.x;
            const int stride = gridDim.x * BLOCK_THREADS;

            // block 内先算局部 max
            float thread_max = -INFINITY;
            for (int idx = blockIdx.x * BLOCK_THREADS + tid; idx < N; idx += stride)
                thread_max = fmaxf(thread_max, input[idx]);
            float m = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
            __shared__ float s_m;
            if (tid == 0) s_m = m;
            __syncthreads();
            m = s_m;

            // 再算相对 block-max 的局部 sum
            float thread_sum = 0.0f;
            for (int idx = blockIdx.x * BLOCK_THREADS + tid; idx < N; idx += stride)
                thread_sum += expf(input[idx] - m);
            float l = BlockReduce(temp_storage).Sum(thread_sum);

            if (tid == 0) atomic_merge_ml(state, m, l);
        }

        template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
        __global__ void softmax_atomic_normalize(float *input, int N,
                                                 const unsigned long long *state) {
            const int tid = threadIdx.x;
            const int stride = gridDim.x * BLOCK_THREADS;
            unsigned long long s = *state;
            const float M = __int_as_float((int)(s >> 32));
            const float L = __int_as_float((int)(s & 0xffffffffULL));
            for (int idx = blockIdx.x * BLOCK_THREADS + tid; idx < N; idx += stride)
                input[idx] = expf(input[idx] - M) / L;
        }

        inline void host_atomic_softmax_forward(float *input, int n) {
            constexpr int BLOCK_THREADS = 256;
            constexpr int ITEMS_PER_THREAD = 4;
            const int ITEMS_PER_BLOCK = BLOCK_THREADS * ITEMS_PER_THREAD;
            const int NUM_BLOCKS = (n + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;

            unsigned long long *state = nullptr;
            cudaMalloc(&state, sizeof(unsigned long long));
            // 初始 (M=-inf, L=0); host 侧用 memcpy 做位重解释 (__float_as_int 是 device-only)
            float ninf = -INFINITY, zero = 0.0f;
            unsigned mi, li;
            std::memcpy(&mi, &ninf, sizeof(float));
            std::memcpy(&li, &zero, sizeof(float));
            unsigned long long init = ((unsigned long long)mi << 32) | li;
            cudaMemcpy(state, &init, sizeof(unsigned long long), cudaMemcpyHostToDevice);

            softmax_atomic_reduce<BLOCK_THREADS, ITEMS_PER_THREAD>
                <<<NUM_BLOCKS, BLOCK_THREADS>>>(input, n, state);
            softmax_atomic_normalize<BLOCK_THREADS, ITEMS_PER_THREAD>
                <<<NUM_BLOCKS, BLOCK_THREADS>>>(input, n, state);
            cudaDeviceSynchronize();
            cudaFree(state);
        }

    } // namespace gpu
}