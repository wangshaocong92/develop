#pragma once
#include <algorithm>
#include <numeric>
#include <vector>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
namespace kernel {
    namespace gpu {
        /*
            输入是一个一维数组，长度为 n，在 GPU 上计算 softmax 的前向传播，在原地输出，长度为 n。
            1. 使用多少个block，单个block的线程数为多少
            2. block 内做reduce
            n 如何拆分呢，多出来的如何做
            3. 单个block 分多少个数据


        */
        template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
        __global__ void softmax_max(float *input, float *d_max,int N) {
            using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            const int tid = threadIdx.x;
            const int global_index = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD + tid;

            // 加载 + 线程内归约合并
            int idx = global_index;
            float val = (idx < N) ? input[idx] : 0.0f;
            float thread_max = val;
            #pragma unroll
            for (int i = 1; i < ITEMS_PER_THREAD; ++i) {
                idx += BLOCK_THREADS; // 跨步访问，合并内存读取
                val = (idx < N) ? input[idx] : 0.0f;
                thread_max = fmaxf(thread_max, val);
            }
            // block 内归约
            float block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());

            if (threadIdx.x == 0) {
                d_max[blockIdx.x] = block_max; // 写入全局内存
            }
        }

        template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
        __global__ void softmax_sum(float *input, float *d_sum, double max,int N) {
            using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            const int tid = threadIdx.x;
            const int global_index = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD + tid;

            // 加载 + 线程内归约合并
            int idx = global_index;
            float val = (idx < N) ? input[idx] : 0.0f;
            float thread_sum = 0.0;
            #pragma unroll
            for (int i = 1; i < ITEMS_PER_THREAD; ++i) {
                idx += BLOCK_THREADS; // 跨步访问，合并内存读取
                val = (idx < N) ? input[idx] : 0.0f;
                thread_sum += (idx < N) ? expf(val - max) : 0.0f;
            }
            // block 内归约
            float block_sum = BlockReduce(temp_storage).Sum(thread_sum);

            if (threadIdx.x == 0) {
                d_sum[blockIdx.x] = block_sum; // 写入全局内存
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
                input[idx] = (idx < N) ? expf(input[idx] - max) / sum : 0.0f;
                idx += BLOCK_THREADS; // 跨步访问，合并内存读取
            }
        }

        inline void host_softmax_forward(float *input, int n) {
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
            softmax_max<BLOCK_THREADS, ITEMS_PER_THREAD><<<NUM_BLOCKS, BLOCK_THREADS>>>(input, d_max, n);
            cudaMemcpy(h_max.data(), d_max, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);
            double max = *std::max_element(h_max.begin(), h_max.end());
            softmax_sum<BLOCK_THREADS, ITEMS_PER_THREAD><<<NUM_BLOCKS, BLOCK_THREADS>>>(input, d_sum, max, n);
            cudaMemcpy(h_sum.data(), d_sum, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);
            double sum = std::accumulate(h_sum.begin(), h_sum.end(), 0.0);
            softmax_forward<BLOCK_THREADS, ITEMS_PER_THREAD><<<NUM_BLOCKS, BLOCK_THREADS>>>(input, sum, max, n);
            cudaFree(d_max);
            cudaFree(d_sum);    
        }


          template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
        __global__ void softmax_max_sum(float *input, float *d_max, float *d_sum, int N) {
            cg::grid_group grid = cg::this_grid();
            using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            const int tid = threadIdx.x;
            const int global_index = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD + tid;

            // 加载 + 线程内归约合并
            int idx = global_index;
            float val = (idx < N) ? input[idx] : 0.0f;
            float thread_max = val;
            #pragma unroll
            for (int i = 1; i < ITEMS_PER_THREAD; ++i) {
                idx += BLOCK_THREADS; // 跨步访问，合并内存读取
                val = (idx < N) ? input[idx] : 0.0f;
                thread_max = fmaxf(thread_max, val);
            }
            // block 内归约
            float block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());

            // 计算 sum
            float thread_sum = 0.0f;
            idx = global_index;
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                val = (idx < N) ? input[idx] : 0.0f;
                thread_sum += (idx < N) ? expf(val - block_max) : 0.0f;
                idx += BLOCK_THREADS; // 跨步访问，合并内存读取
            }
            float block_sum = BlockReduce(temp_storage).Sum(thread_sum);

            if (threadIdx.x == 0) {
                d_max[blockIdx.x] = block_max; // 写入全局内存
                d_sum[blockIdx.x] = block_sum; // 写入全局内存
            }

            grid.sync();  // ← 所有 block 在此等待，确保 gmax 是最终值
            double sum = 0.0;
            float final_max = -std::numeric_limits<float>::infinity();
            if (blockIdx.x == 0 && threadIdx.x == 0) {   
                for (int i = 0; i < gridDim.x; ++i) {
                    final_max = fmaxf(final_max, d_max[i]);
                }
                for(auto i = 0; i < gridDim.x; ++i){
                    sum += d_sum[i] * expf(d_max[i] - final_max);
                }
            }
            grid.sync();  // ← 所有 block 在此等待，确保 gmax 是最终值
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                input[idx] = (idx < N) ? expf(input[idx] - max) / sum : 0.0f;
                idx += BLOCK_THREADS; // 跨步访问，合并内存读取
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

    } // namespace gpu
}