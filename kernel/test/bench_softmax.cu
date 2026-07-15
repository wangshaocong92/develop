/******************************************************************************
 * CPU vs GPU softmax 耗时对比
 ******************************************************************************/

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "softmax/softmax.h"
#include "softmax/softmax.cuh"

static std::vector<float> make_data(int n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dis(-5.f, 5.f);
    std::vector<float> v(n);
    for (auto &x : v) x = dis(rng);
    return v;
}

#define TIME(label, fn)                                                        \
    do {                                                                       \
        auto t0 = std::chrono::high_resolution_clock::now();                   \
        fn();                                                                  \
        auto t1 = std::chrono::high_resolution_clock::now();                   \
        double ms =                                                            \
            std::chrono::duration<double, std::milli>(t1 - t0).count();        \
        std::cout << "  " << std::left << std::setw(30) << (label) << " "     \
                  << std::right << std::setw(8) << std::fixed                  \
                  << std::setprecision(3) << ms << " ms" << std::endl;         \
    } while (0)

template <int N>
static void gpu_3pass(const float *src) {
    float *d = nullptr;
    cudaMalloc(&d, N * sizeof(float));
    cudaMemcpy(d, src, N * sizeof(float), cudaMemcpyHostToDevice);
    kernel::gpu::host_softmax_forward<N>(d);
    cudaMemcpy(const_cast<float*>(src), d, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);
}

static void gpu_2pass(const float *src, int n) {
    float *d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice);
    kernel::gpu::host_online_softmax_forward(d, n);
    cudaMemcpy(const_cast<float*>(src), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);
}

static void gpu_coop(const float *src, int n) {
    float *d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice);
    kernel::gpu::host_coop_softmax_forward(d, n);
    cudaMemcpy(const_cast<float*>(src), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);
}

static void gpu_atomic(const float *src, int n) {
    float *d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice);
    kernel::gpu::host_atomic_softmax_forward(d, n);
    cudaMemcpy(const_cast<float*>(src), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);
}

// 单个编译期尺寸 N 的四项对比。3-pass 需要编译期 N。
template <int N>
static void run_size() {
    std::cout << "\n--- N = " << N << " ---" << std::endl;

    auto base = make_data(N);
    auto input = base;

    TIME("cpu softmax_forward",
         ([&] { kernel::cpu::softmax_forward(input.data(), N); }));

    input = base;
    TIME("cpu online_softmax_forward",
         ([&] { kernel::cpu::online_softmax_forward(input.data(), N); }));

    input = base;
    TIME("gpu softmax (3-pass)",
         ([&] { gpu_3pass<N>(input.data()); }));

    input = base;
    TIME("gpu softmax (2-pass)",
         ([&] { gpu_2pass(input.data(), N); }));

    input = base;
    TIME("gpu softmax (coop/A)",
         ([&] { gpu_coop(input.data(), N); }));

    input = base;
    TIME("gpu softmax (atomic/B)",
         ([&] { gpu_atomic(input.data(), N); }));
}

int main() {
    // 预热
    std::vector<float> warm(1024);
    gpu_3pass<1024>(warm.data());
    gpu_2pass(warm.data(), 1024);
    gpu_coop(warm.data(), 1024);
    gpu_atomic(warm.data(), 1024);

    std::cout << "===== CPU vs GPU softmax =====\n";

    run_size<4096>();
    run_size<65536>();
    run_size<262144>();
    run_size<1048576>();
    run_size<4194304>();
    run_size<16777216>();
    return 0;
}
