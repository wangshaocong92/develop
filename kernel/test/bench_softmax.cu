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

static void gpu_3pass(const float *src, int n) {
    float *d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice);
    kernel::gpu::host_softmax_forward(d, n);
    cudaMemcpy(const_cast<float*>(src), d, n * sizeof(float), cudaMemcpyDeviceToHost);
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

int main() {
    // 预热
    std::vector<float> warm(1024);
    gpu_3pass(warm.data(), 1024);
    gpu_2pass(warm.data(), 1024);

    const int sizes[] = {4096, 65536, 262144, 1048576, 4194304, 16777216};
    std::cout << "===== CPU vs GPU softmax =====\n";

    for (int n : sizes) {
        std::cout << "\n--- N = " << n << " ---" << std::endl;

        auto base = make_data(n);
        auto input = base;

        TIME("cpu softmax_forward",
             ([&] { kernel::cpu::softmax_forward(input.data(), n); }));

        input = base;
        TIME("cpu online_softmax_forward",
             ([&] { kernel::cpu::online_softmax_forward(input.data(), n); }));

        input = base;
        TIME("gpu softmax (3-pass)",
             ([&] { gpu_3pass(input.data(), n); }));

        input = base;
        TIME("gpu softmax (2-pass)",
             ([&] { gpu_2pass(input.data(), n); }));
    }
    return 0;
}
