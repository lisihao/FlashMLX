#!/usr/bin/env python3
"""
Debug MAC hit rate - 为什么初始报告只有 45.2% 命中率？

检查：
1. Warmup 是否充分
2. Threshold 设置是否正确
3. Query drift 是否过大
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx


def sync():
    mx.eval(mx.array(0))


print("=" * 80)
print("MAC Hit Rate Debug")
print("=" * 80)
print()

from flashmlx.mac import MACDecodeWrapper

N, Hq, Hkv, D = 1, 32, 8, 128
S = 8192

# 测试不同配置
configs = [
    ("normalize=True, thr=0.5", True, 0.5),
    ("normalize=True, thr=0.6", True, 0.6),
    ("normalize=True, thr=0.7", True, 0.7),
    ("normalize=False, thr=0.5", False, 0.5),
]

for name, normalize, threshold in configs:
    print(f"{'='*80}")
    print(f"Config: {name}")
    print(f"{'='*80}")

    mac = MACDecodeWrapper(
        max_requests=4,
        capacity=512,
        num_heads=Hq,
        num_kv_heads=Hkv,
        head_dim=D,
        threshold=threshold,
        band_r=256,
        window_left=256,
        normalize_queries=normalize,
    )

    mx.random.seed(42)
    k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    req_ids = mx.array([0], dtype=mx.int32)
    mx.eval(k, v)

    # Warmup with SAME query (perfect match)
    print("  [1] Warmup: 100 steps with same query")
    q_base = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(q_base)

    for i in range(100):
        mac(q_base, k, v, req_ids)
        if i % 50 == 49:
            sync()

    mac.ring_cache.request_length = mx.array([S], dtype=mx.int32)
    mx.eval(mac.ring_cache.request_length)

    # Test with perfect match
    print("  [2] Test: perfect match (same query)")
    mac(q_base, k, v, req_ids)
    stats = mac.last_stats
    if stats:
        print(f"      Hit rate: {stats.hit_rate:.1%}")
        print(f"      Num hits: {stats.num_hits} / {stats.num_total}")

    # Test with slight drift
    print("  [3] Test: slight drift (noise=0.01)")
    noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * 0.01
    q_drift = q_base + noise
    mx.eval(q_drift)
    mac(q_drift, k, v, req_ids)
    stats = mac.last_stats
    if stats:
        print(f"      Hit rate: {stats.hit_rate:.1%}")

    # Test with medium drift
    print("  [4] Test: medium drift (noise=0.1)")
    noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * 0.1
    q_drift = q_base + noise
    mx.eval(q_drift)
    mac(q_drift, k, v, req_ids)
    stats = mac.last_stats
    if stats:
        print(f"      Hit rate: {stats.hit_rate:.1%}")

    # Test with new query
    print("  [5] Test: new random query")
    q_new = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(q_new)
    mac(q_new, k, v, req_ids)
    stats = mac.last_stats
    if stats:
        print(f"      Hit rate: {stats.hit_rate:.1%}")

    print()

print("=" * 80)
print("结论")
print("=" * 80)
print("如果 perfect match 命中率不是 100% → threshold 转换有问题")
print("如果 slight drift 命中率下降很快 → threshold 太严格")
print()
