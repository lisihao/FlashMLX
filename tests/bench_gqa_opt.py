#!/usr/bin/env python3
"""
测试 GQA 优化版本

对比：
1. 原版 (query head 调度)
2. GQA 优化 (KV head 调度)
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
print("GQA 优化测试 (KV Head 调度)")
print("=" * 80)
print()

from flashmlx.mac.attention import (
    mac_fused_partial_attention,
    _mac_partial_attention_reference,
)
from flashmlx.mac.attention_gqa_opt import mac_partial_attention_gqa_opt

N, H, D = 1, 32, 128
Hkv = 8
S = 32768

mx.random.seed(42)
queries = mx.random.normal((N, H, D)).astype(mx.bfloat16)
keys = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
values = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
scale = D**-0.5
mx.eval(queries, keys, values)

print("Test 1: Full attention (start=0)")
print("=" * 80)

start_full = mx.zeros((N, H), dtype=mx.int32)
mx.eval(start_full)

def run_orig():
    return mac_fused_partial_attention(queries, keys, values, start_full, scale)

def run_gqa():
    return mac_partial_attention_gqa_opt(queries, keys, values, start_full, scale)

print("  [1/3] Benchmark original...")
t_orig = measure_ms(run_orig)
print(f"        Time: {t_orig:.3f} ms")

print("  [2/3] Benchmark GQA optimized...")
t_gqa = measure_ms(run_gqa)
print(f"        Time: {t_gqa:.3f} ms")

print("  [3/3] Verify correctness...")
o_ref, lse_ref = _mac_partial_attention_reference(queries, keys, values, start_full, scale)
o_gqa, lse_gqa = run_gqa()
mx.eval(o_ref, lse_ref, o_gqa, lse_gqa)

diff_o = mx.abs(o_ref - o_gqa).max().item()
diff_lse = mx.abs(lse_ref - lse_gqa).max().item()

print()
print(f"  Results:")
print(f"    Original:      {t_orig:.3f} ms")
print(f"    GQA Optimized: {t_gqa:.3f} ms")
print(f"    Speedup:       {t_orig / t_gqa:.2f}×")
print(f"    O diff:        {diff_o:.2e}")
print(f"    LSE diff:      {diff_lse:.2e}")

if diff_o < 1e-2:
    print(f"    ✅ Correct")
else:
    print(f"    ❌ Wrong - 需要调试")

print()
print("=" * 80)
print("分析")
print("=" * 80)
print()
print("如果 speedup > 1.3× → GQA 调度有效！")
print("如果 speedup < 1.1× → 还需要 shared memory 优化")
print()
