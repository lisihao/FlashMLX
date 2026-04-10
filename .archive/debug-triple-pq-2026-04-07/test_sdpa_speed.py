#!/usr/bin/env python3
"""
测试标准SDPA的原始性能
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
import time
import numpy as np

print("=" * 80)
print("标准 SDPA 原始性能测试")
print("=" * 80)
print()

# 测试不同context长度
contexts = [500, 1000, 5000, 10000]

for ctx_len in contexts:
    print(f"测试: {ctx_len} tokens 上下文")

    # 准备数据 (与Qwen3一致)
    q = mx.random.normal((1, 32, 1, 128))  # [B, H, L=1, D]
    k = mx.random.normal((1, 8, ctx_len, 128))  # [B, Hkv, S, D]
    v = mx.random.normal((1, 8, ctx_len, 128))

    # Warmup
    _ = scaled_dot_product_attention(q, k, v, cache=None, scale=1/128**0.5, mask=None)
    mx.eval(_)

    # Benchmark
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        output = scaled_dot_product_attention(q, k, v, cache=None, scale=1/128**0.5, mask=None)
        mx.eval(output)
        times.append(time.perf_counter() - t0)

    mean_time = np.mean(times) * 1000  # ms
    std_time = np.std(times) * 1000

    print(f"  时间: {mean_time:.2f} ± {std_time:.2f} ms")
    print()

print("对比MAC:")
print("  如果SDPA << MAC：MAC慢是正常的")
print("  如果SDPA ≈ MAC：问题在模型其他部分")
