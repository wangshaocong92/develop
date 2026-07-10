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

constexpr int BLK = 256;
constexpr int ITM = 4;
constexpr int CHK = BLK * ITM;  // 1024

static void launch_online(float *d_in, float *d_max, float *d_sum, int n) {
    int nb = (n + CHK - 1) / CHK;
    void *args[] = {&d_in, &d_max, &d_sum, &n};
    cudaLaunchCooperativeKernel(
        (void*)kernel::gpu::online_softmax_forward<BLK, ITM>,
        dim3(nb), dim3(BLK), args);
}

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

int main() {
    // 预热
    std::vector<float> warm(1024);
    float *d = nullptr, *dm = nullptr, *ds = nullptr;
    cudaMalloc(&d, 1024 * sizeof(float));
    cudaMalloc(&dm, sizeof(float));
    cudaMalloc(&ds, sizeof(float));
    cudaMemcpy(d, warm.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice);
    kernel::gpu::host_softmax_forward(d, 1024);
    launch_online(d, dm, ds, 1024);
    cudaFree(d); cudaFree(dm); cudaFree(ds);

    const int sizes[] = {4096, 65536, 262144, 1048576, 4194304, 16777216};
    std::cout << "===== CPU vs GPU softmax =====\n";

    for (int n : sizes) {
        std::cout << "\n--- N = " << n << " ---" << std::endl;

        auto base = make_data(n);
        std::vector<float> input = base;

        TIME("cpu softmax_forward",
             ([&] { kernel::cpu::softmax_forward(input.data(), n); }));

        input = base;
        TIME("cpu online_softmax_forward",
             ([&] { kernel::cpu::online_softmax_forward(input.data(), n); }));

        // GPU: 3-pass
        input = base;
        TIME("gpu softmax (3-pass)",
             ([&] {
                 float *d_in = nullptr;
                 cudaMalloc(&d_in, n * sizeof(float));
                 cudaMemcpy(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                 kernel::gpu::host_softmax_forward(d_in, n);
                 cudaMemcpy(input.data(), d_in, n * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaFree(d_in);
             }));

        // GPU: online (cooperative, grid sync)
        input = base;
        TIME("gpu softmax (online)",
             ([&] {
                 int nb = (n + CHK - 1) / CHK;
                 float *d_in = nullptr, *d_m = nullptr, *d_s = nullptr;
                 cudaMalloc(&d_in, n * sizeof(float));
                 cudaMalloc(&d_m, nb * sizeof(float));
                 cudaMalloc(&d_s, nb * sizeof(float));
                 cudaMemcpy(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                 launch_online(d_in, d_m, d_s, n);
                 cudaMemcpy(input.data(), d_in, n * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaFree(d_in); cudaFree(d_m); cudaFree(d_s);
             }));
    }
    return 0;
}
