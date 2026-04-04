#!/usr/bin/env python3
"""
测试 Partial Attention 是否真的只计算部分

如果 start_pos=S-256 时间 << start_pos=0 时间
→ 说明真的只计算部分

如果时间相近
→ 说明计算了全部然后 mask（没有节省）
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
        mx.eval(*result)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


print("=" * 80)
print("Partial Attention 是否真的部分计算？")
print("=" * 80)
print()

from flashmlx.mac.attention import mac_fused_partial_attention

N, H, D = 1, 32, 128
Hkv = 8
S = 8192

mx.random.seed(42)
queries = mx.random.normal((N, H, D)).astype(mx.bfloat16)
keys = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
values = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
scale = D**-0.5
mx.eval(queries, keys, values)

# Test different start positions
test_configs = [
    ("Full (start=0)", 0),
    ("75% (start=2K)", 2048),
    ("50% (start=4K)", 4096),
    ("25% (start=6K)", 6144),
    ("3% (start=7936)", 7936),  # Only last 256 tokens
]

print("测试不同 start_pos 的性能：")
print("=" * 80)
print(f"{'Config':<20} {'Start Pos':<12} {'Time (ms)':<12} {'vs Full':<10}")
print("-" * 80)

baseline_time = None

for name, start in test_configs:
    start_pos = mx.full((N, H), start, dtype=mx.int32)
    mx.eval(start_pos)

    def run():
        return mac_fused_partial_attention(queries, keys, values, start_pos, scale)

    t = measure_ms(run)

    if baseline_time is None:
        baseline_time = t

    ratio = baseline_time / t if t > 0 else 0
    active_ratio = (S - start) / S

    print(f"{name:<20} {start:<12} {t:>10.3f}   {ratio:>8.2f}× ({active_ratio*100:.0f}% active)")

print()
print("=" * 80)
print("分析")
print("=" * 80)

print()
print("如果时间 ∝ 活跃token比例 → 真的只计算部分 ✅")
print("如果时间不变 → 计算全部然后mask ❌")
print()
print("例如：")
print("  start=6K (25% active) 应该是 start=0 (100% active) 的 ~0.25× 时间")
print("  start=7936 (3% active) 应该是 start=0 的 ~0.03× 时间")
print()
