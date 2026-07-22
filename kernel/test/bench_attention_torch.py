#!/usr/bin/env python3
"""PyTorch attention 基准 —— 与 C++ bench_attention 相同规模对比。

规模:单 batch、单 head,seqlen=M,head_dim=N。与我们的 kernel 一致:
  - 不做 1/sqrt(d) 缩放(scale=1.0),softmax 沿 key 维。
分别测:
  1) 朴素实现(matmul + softmax + matmul),float32 —— 对标 standard_attention
  2) F.scaled_dot_product_attention(FlashAttention 后端),float16 —— 对标 flash
用 CUDA events 计时,warmup + 多次迭代取均值。
"""
import torch
import torch.nn.functional as F

torch.manual_seed(42)
DEVICE = "cuda"


def time_gpu_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def naive_attention(q, k, v):
    # 无缩放,与我们的 kernel 对齐
    s = q @ k.transpose(-2, -1)
    p = torch.softmax(s, dim=-1)
    return p @ v


def bench_row(M, N):
    # float32 朴素(对标 standard_attention)
    q32 = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    k32 = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    v32 = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    naive_ms = time_gpu_ms(lambda: naive_attention(q32, k32, v32))

    # float16 SDPA(FlashAttention 后端,对标我们的 flash)
    q16 = torch.randn(1, 1, M, N, device=DEVICE, dtype=torch.float16)
    k16 = torch.randn(1, 1, M, N, device=DEVICE, dtype=torch.float16)
    v16 = torch.randn(1, 1, M, N, device=DEVICE, dtype=torch.float16)
    sdpa_ms = time_gpu_ms(
        lambda: F.scaled_dot_product_attention(q16, k16, v16, scale=1.0)
    )

    print(
        f"  M={M:<5d} N={N:<4d} | naive-f32 {naive_ms:8.4f} ms | "
        f"SDPA-f16 {sdpa_ms:8.4f} ms | naive/sdpa {naive_ms / sdpa_ms:6.2f}x"
    )


def main():
    assert torch.cuda.is_available(), "需要 CUDA"
    print("===== PyTorch Attention Benchmark =====")
    print(f"device={torch.cuda.get_device_name(0)}, torch={torch.__version__}")
    print("naive=f32 matmul+softmax+matmul, SDPA=f16 scaled_dot_product_attention(scale=1)")
    print("N=128 = GPT-4 head_dim; M = 序列长度(GPT-4 context 8K~32K)\n")
    for M in (2048, 4096, 8192, 16384):
        bench_row(M, 128)


if __name__ == "__main__":
    main()
