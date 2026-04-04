#!/usr/bin/env python3
"""
测试 Block-stride 优化版本 vs 原始版本

对比：
1. 原始：跨度访问 (stride=NUM_WARPS)
2. 优化：块状访问 (连续块)
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
print("Block-Stride 优化测试")
print("=" * 80)
print()

from flashmlx.mac.attention import (
    mac_fused_partial_attention,
    _mac_partial_attention_reference,
)
from flashmlx.mac.attention_opt_block import mac_partial_attention_block_stride

# Test configs
configs = [
    ("8K", 8192),
    ("16K", 16384),
    ("32K", 32768),
]

N, H, D = 1, 32, 128
Hkv = 8

for name, S in configs:
    print(f"{'='*80}")
    print(f"Context: {name} ({S} tokens)")
    print(f"{'='*80}")

    mx.random.seed(42)
    queries = mx.random.normal((N, H, D)).astype(mx.bfloat16)
    keys = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    values = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    start_pos = mx.zeros((N, H), dtype=mx.int32)
    scale = D**-0.5
    mx.eval(queries, keys, values, start_pos)

    # Reference
    def run_ref():
        return _mac_partial_attention_reference(queries, keys, values, start_pos, scale)

    print("  [1/3] Reference (MLX)...")
    t_ref = measure_ms(run_ref)
    o_ref, lse_ref = run_ref()
    mx.eval(o_ref, lse_ref)
    print(f"        Time: {t_ref:.3f} ms")

    # Original fused
    def run_orig():
        return mac_fused_partial_attention(queries, keys, values, start_pos, scale)

    print("  [2/3] Original fused (stride access)...")
    t_orig = measure_ms(run_orig)
    o_orig, lse_orig = run_orig()
    mx.eval(o_orig, lse_orig)
    print(f"        Time: {t_orig:.3f} ms ({t_ref / t_orig:.2f}× vs ref)")

    # Block-stride optimized
    def run_opt():
        return mac_partial_attention_block_stride(queries, keys, values, start_pos, scale)

    print("  [3/3] Block-stride optimized...")
    t_opt = measure_ms(run_opt)
    o_opt, lse_opt = run_opt()
    mx.eval(o_opt, lse_opt)

    # Correctness
    diff_o = mx.abs(o_ref - o_opt).max().item()
    diff_lse = mx.abs(lse_ref - lse_opt).max().item()

    print(f"        Time: {t_opt:.3f} ms ({t_ref / t_opt:.2f}× vs ref)")
    print(f"        Speedup vs original: {t_orig / t_opt:.2f}×")
    print(f"        Diff O: {diff_o:.2e}, LSE: {diff_lse:.2e}")

    if diff_o < 1e-3:
        print(f"        ✅ Correct")
    else:
        print(f"        ❌ Wrong - 需要调试")

    print()

print("=" * 80)
print("总结")
print("=" * 80)
print("如果 block-stride 更快 → 内存访问模式是瓶颈")
print("如果持平/更慢 → 需要其他优化方向（GQA 缓存复用等）")
print()
