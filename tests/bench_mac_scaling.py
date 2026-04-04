#!/usr/bin/env python3
"""
MAC 性能随序列长度的扩展性

如果性能与 S 成正比 → 内存带宽瓶颈（KV 读取）
如果性能与 S 无关 → 计算瓶颈
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
print("MAC 性能随序列长度扩展")
print("=" * 80)
print()

from flashmlx.mac import MACDecodeWrapper

N, Hq, Hkv, D = 1, 32, 8, 128

configs = [
    ("2K", 2048),
    ("4K", 4096),
    ("8K", 8192),
    ("16K", 16384),
    ("32K", 32768),
]

results = []

for name, S in configs:
    print(f"Testing {name} ({S} tokens)...", end=" ", flush=True)

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
    q_test = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    req_ids = mx.array([0], dtype=mx.int32)
    mx.eval(k, v, q_test)

    # Warmup
    for i in range(20):
        q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
        mx.eval(q)
        mac(q, k, v, req_ids)
        if i % 10 == 9:
            sync()

    mac.ring_cache.request_length = mx.array([S], dtype=mx.int32)
    mx.eval(mac.ring_cache.request_length)

    def mac_fn():
        return mac(q_test, k, v, req_ids)

    t = measure_ms(mac_fn, warmup=10, iters=50)
    results.append((name, S, t))
    print(f"{t:.3f} ms")

print()
print("=" * 80)
print("扩展性分析")
print("=" * 80)
print()
print(f"{'Context':<10} {'Tokens':<10} {'Time (ms)':<12} {'ms/1K tokens':<15} {'vs 2K':<10}")
print("-" * 80)

baseline_time = results[0][2]
baseline_tokens = results[0][1]

for name, S, t in results:
    ms_per_1k = t / (S / 1000)
    vs_baseline = t / baseline_time
    print(
        f"{name:<10} {S:<10} {t:>10.3f}   {ms_per_1k:>13.3f}   {vs_baseline:>8.2f}×"
    )

print()
print("=" * 80)
print("结论")
print("=" * 80)

# 计算时间增长率 vs 序列增长率
time_growth = results[-1][2] / results[0][2]
seq_growth = results[-1][1] / results[0][1]

print(f"序列长度增长: {seq_growth:.1f}× (2K → 32K)")
print(f"时间增长:     {time_growth:.1f}×")
print()

if time_growth < seq_growth * 0.6:
    print("✅ 次线性扩展 - MAC 缓存命中起作用了！")
elif time_growth < seq_growth * 1.2:
    print("😐 线性扩展 - 内存带宽瓶颈（KV 读取）")
else:
    print("❌ 超线性扩展 - 算法问题")

print()
print("如果是内存瓶颈 → 优化方向：")
print("  1. KV cache 量化（降低带宽）")
print("  2. Flash Attention tiling（复用 KV）")
print("  3. 更激进的 partial attention（减少读取范围）")
print()
