#!/usr/bin/env python3
"""
MAC 端到端热路径分析 - 精确版本

问题：之前的分解显示理论 1.218ms，实际只有 0.367ms
说明 MLX 自动 fusion 了很多操作

这次直接 profile 端到端，找出真正耗时的部分
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx


def sync():
    mx.eval(mx.array(0))


def measure_ms(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
        sync()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        result = fn()
        mx.eval(result)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


print("=" * 80)
print("MAC 端到端热路径分析")
print("=" * 80)
print()

from flashmlx.mac import MACDecodeWrapper

N, Hq, Hkv, D = 1, 32, 8, 128
S = 8192

mac = MACDecodeWrapper(
    max_requests=4,
    capacity=512,
    num_heads=Hq,
    num_kv_heads=Hkv,
    head_dim=D,
    threshold=0.5,
    band_r=256,
    window_left=256,
    normalize_queries=True,
)

mx.random.seed(42)
k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
req_ids = mx.array([0], dtype=mx.int32)
mx.eval(k, v)

# Warmup
print("Warming up...")
for i in range(100):
    q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(q)
    mac(q, k, v, req_ids)
    if i % 50 == 49:
        sync()

mac.ring_cache.request_length = mx.array([S], dtype=mx.int32)
mx.eval(mac.ring_cache.request_length)

q_test = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
mx.eval(q_test)

print()
print("=" * 80)
print("Profile 1: 完整端到端")
print("=" * 80)


def full_mac():
    return mac(q_test, k, v, req_ids)


t_full = measure_ms(full_mac, warmup=20, iters=100)
print(f"完整 MAC: {t_full:.3f} ms")
print()

# 现在逐步删除各个操作，看性能变化
print("=" * 80)
print("Profile 2: 不同配置下的性能")
print("=" * 80)
print()

# 禁用 normalization
print("[1] 测试：禁用 query normalization")
mac_no_norm = MACDecodeWrapper(
    max_requests=4,
    capacity=512,
    num_heads=Hq,
    num_kv_heads=Hkv,
    head_dim=D,
    threshold=0.5,
    band_r=256,
    window_left=256,
    normalize_queries=False,  # ← 禁用
)

# Warmup
for i in range(50):
    q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(q)
    mac_no_norm(q, k, v, req_ids)
    if i % 25 == 24:
        sync()

mac_no_norm.ring_cache.request_length = mx.array([S], dtype=mx.int32)
mx.eval(mac_no_norm.ring_cache.request_length)


def mac_no_norm_fn():
    return mac_no_norm(q_test, k, v, req_ids)


t_no_norm = measure_ms(mac_no_norm_fn, warmup=20, iters=100)
print(f"  禁用 normalization: {t_no_norm:.3f} ms")
print(f"  节省时间: {t_full - t_no_norm:.3f} ms ({(t_full - t_no_norm) / t_full * 100:.1f}%)")
print()

# 对比：如果全是 miss（start_pos=0，相当于 full attention）
print("[2] 测试：全 miss（full attention）")
mac_miss = MACDecodeWrapper(
    max_requests=4,
    capacity=512,
    num_heads=Hq,
    num_kv_heads=Hkv,
    head_dim=D,
    threshold=0.0,  # ← threshold=0 强制全miss
    band_r=256,
    window_left=256,
    normalize_queries=True,
)

for i in range(50):
    q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(q)
    mac_miss(q, k, v, req_ids)
    if i % 25 == 24:
        sync()

mac_miss.ring_cache.request_length = mx.array([S], dtype=mx.int32)
mx.eval(mac_miss.ring_cache.request_length)


def mac_miss_fn():
    return mac_miss(q_test, k, v, req_ids)


t_miss = measure_ms(mac_miss_fn, warmup=20, iters=100)
print(f"  全 miss（full attn）: {t_miss:.3f} ms")
print(f"  vs 有命中: {t_miss / t_full:.2f}×")
print()

print("=" * 80)
print("结论")
print("=" * 80)
print(f"1. Query normalization 占 {(t_full - t_no_norm) / t_full * 100:.1f}% 时间")
print(f"2. Cache hit 带来 {(t_miss / t_full - 1) * 100:.1f}% 额外开销")
print()
print("如果 normalization 占比大 → 可以优化")
print("如果 hit 开销大 → match/merge 还有优化空间")
print()
