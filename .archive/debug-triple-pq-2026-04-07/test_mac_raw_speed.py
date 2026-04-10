#!/usr/bin/env python3
"""
直接测试MAC wrapper的原始性能
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from flashmlx.mac import MACDecodeWrapper
import time
import numpy as np

print("=" * 80)
print("MAC Wrapper 原始性能测试")
print("=" * 80)
print()

# 创建MAC wrapper
mac = MACDecodeWrapper(
    max_requests=4,
    capacity=8192,
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    threshold=0.3,
    band_r=512,
    window_left=512,
    normalize_queries=True,
)

print("✅ MAC wrapper 创建完成\n")

# 测试不同context长度
contexts = [500, 1000, 5000, 10000]

for ctx_len in contexts:
    print(f"测试: {ctx_len} tokens 上下文")

    # 准备数据 (bf16)
    q = mx.random.normal((1, 32, 128)).astype(mx.bfloat16)  # [B, H, D]
    k = mx.random.normal((1, ctx_len, 8, 128)).astype(mx.bfloat16)  # [B, S, Hkv, D]
    v = mx.random.normal((1, ctx_len, 8, 128)).astype(mx.bfloat16)
    req_ids = mx.array([0], dtype=mx.int32)

    # Warmup
    _ = mac(q, k, v, req_ids)
    mx.eval(_)

    # Benchmark
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        output = mac(q, k, v, req_ids)
        mx.eval(output)  # 强制同步
        times.append(time.perf_counter() - t0)

    mean_time = np.mean(times) * 1000  # ms
    std_time = np.std(times) * 1000

    print(f"  时间: {mean_time:.2f} ± {std_time:.2f} ms")
    print()

print("💡 如果时间很短(<1ms)，说明MAC本身很快，问题在集成")
print("💡 如果时间很长(>10ms)，说明MAC实现本身有问题")
