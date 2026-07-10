/******************************************************************************
 * kernel/softmax 单元测试
 *
 * 测试 kernel::cpu::softmax_forward 和 online_softmax_forward
 * 与标准 softmax 公式逐元素对比
 ******************************************************************************/

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "softmax/softmax.h"

// ==========================================================================
// 标准 softmax 参照实现
// ==========================================================================
static void reference_softmax(const float *input, float *output, int n) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i)
        max_val = std::max(max_val, input[i]);

    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += std::exp(static_cast<double>(input[i]) - max_val);

    for (int i = 0; i < n; ++i)
        output[i] = static_cast<float>(
            std::exp(static_cast<double>(input[i]) - max_val) / sum);
}

// ==========================================================================
// 测试辅助
// ==========================================================================
static std::vector<float> make_random_input(int n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-5.f, 5.f);
    std::vector<float> data(n);
    for (auto &v : data) v = dis(gen);
    return data;
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static bool sum_close_to_one(const float *x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += x[i];
    return std::fabs(sum - 1.0) < 1e-5;
}

// ==========================================================================
// softmax_forward 测试
// ==========================================================================
TEST(SoftmaxForward, Small) {
    const int n = 16;
    auto input = make_random_input(n);
    std::vector<float> expected(n);
    reference_softmax(input.data(), expected.data(), n);

    kernel::cpu::softmax_forward(input.data(), n);  // in-place

    EXPECT_TRUE(sum_close_to_one(input.data(), n));
    EXPECT_LT(max_abs_diff(input.data(), expected.data(), n), 1e-5f);
}

TEST(SoftmaxForward, Medium) {
    const int n = 1024;
    auto input = make_random_input(n);
    std::vector<float> expected(n);
    reference_softmax(input.data(), expected.data(), n);

    kernel::cpu::softmax_forward(input.data(), n);

    EXPECT_TRUE(sum_close_to_one(input.data(), n));
    EXPECT_LT(max_abs_diff(input.data(), expected.data(), n), 1e-5f);
}

TEST(SoftmaxForward, Upscale) {
    // 大值场景: 确保数值稳定性
    const int n = 128;
    std::vector<float> input(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(100.f, 200.f);
    for (auto &v : input) v = dis(gen);

    std::vector<float> expected(n);
    reference_softmax(input.data(), expected.data(), n);

    kernel::cpu::softmax_forward(input.data(), n);

    EXPECT_TRUE(sum_close_to_one(input.data(), n));
    EXPECT_LT(max_abs_diff(input.data(), expected.data(), n), 1e-5f);
}

TEST(SoftmaxForward, SingleElement) {
    float x = 3.14f;
    kernel::cpu::softmax_forward(&x, 1);
    EXPECT_NEAR(x, 1.0f, 1e-6f);
}

// ==========================================================================
// online_softmax_forward 测试
// ==========================================================================
TEST(OnlineSoftmaxForward, Small) {
    const int n = 16;
    auto input = make_random_input(n);
    std::vector<float> expected(n);
    reference_softmax(input.data(), expected.data(), n);

    kernel::cpu::online_softmax_forward(input.data(), n);

    EXPECT_TRUE(sum_close_to_one(input.data(), n));
    EXPECT_LT(max_abs_diff(input.data(), expected.data(), n), 1e-5f);
}

TEST(OnlineSoftmaxForward, Medium) {
    const int n = 1024;
    auto input = make_random_input(n);
    std::vector<float> expected(n);
    reference_softmax(input.data(), expected.data(), n);

    kernel::cpu::online_softmax_forward(input.data(), n);

    EXPECT_TRUE(sum_close_to_one(input.data(), n));
    EXPECT_LT(max_abs_diff(input.data(), expected.data(), n), 1e-5f);
}

TEST(OnlineSoftmaxForward, NotMultipleOfStep) {
    // seqlen 非 32 整数倍
    const int n = 100;
    auto input = make_random_input(n);
    std::vector<float> expected(n);
    reference_softmax(input.data(), expected.data(), n);

    kernel::cpu::online_softmax_forward(input.data(), n);

    EXPECT_TRUE(sum_close_to_one(input.data(), n));
    EXPECT_LT(max_abs_diff(input.data(), expected.data(), n), 1e-5f);
}

TEST(OnlineSoftmaxForward, ExactlyOneStep) {
    const int n = 32;
    auto input = make_random_input(n);
    std::vector<float> expected(n);
    reference_softmax(input.data(), expected.data(), n);

    kernel::cpu::online_softmax_forward(input.data(), n);

    EXPECT_TRUE(sum_close_to_one(input.data(), n));
    EXPECT_LT(max_abs_diff(input.data(), expected.data(), n), 1e-5f);
}

TEST(OnlineSoftmaxForward, Large) {
    const int n = 5000000;  // 5M elements
    auto input = make_random_input(n);
    std::vector<float> expected(n);

    auto t0 = std::chrono::high_resolution_clock::now();
    reference_softmax(input.data(), expected.data(), n);
    auto t1 = std::chrono::high_resolution_clock::now();

    kernel::cpu::online_softmax_forward(input.data(), n);
    auto t2 = std::chrono::high_resolution_clock::now();

    double ref_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double impl_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "\n   n=" << n
              << "  reference: " << ref_ms << " ms"
              << "  online_softmax: " << impl_ms << " ms"
              << "  speedup: " << ref_ms / impl_ms << "x\n";

    EXPECT_TRUE(sum_close_to_one(input.data(), n));
    EXPECT_LT(max_abs_diff(input.data(), expected.data(), n), 1e-5f);
}

// ==========================================================================
// 一致性测试: 两个实现的结果应该一致
// ==========================================================================
TEST(SoftmaxConsistency, TwoImplMatch) {
    const int n = 512;
    auto base = make_random_input(n);
    std::vector<float> input1 = base;
    std::vector<float> input2 = base;

    kernel::cpu::softmax_forward(input1.data(), n);
    kernel::cpu::online_softmax_forward(input2.data(), n);

    EXPECT_LT(max_abs_diff(input1.data(), input2.data(), n), 1e-5f);
}
