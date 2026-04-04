#!/usr/bin/env python3
"""
Merge kernel 详细性能剖析 - 找出端到端变慢的原因

对比：
1. 隔离测试：merge kernel 1.13× 加速
2. 端到端：整体 0.56× 变慢

问题：为什么在实际调用中反而慢了？
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
        if isinstance(result, (tuple, list)):
            mx.eval(*result)
        else:
            mx.eval(result)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


print("=" * 80)
print("Merge Kernel 性能剖析")
print("=" * 80)
print()

# Test data - multiple shapes
configs = [
    ("Tiny", 1, 32, 128),
    ("MAC-typical", 1, 32, 128),  # Same as MAC decode
    ("Larger", 4, 32, 128),
]

from flashmlx.mac.attention import merge_attention_states
from flashmlx.mac.merge_fused import merge_attention_states_fused

for name, N, H, D in configs:
    print(f"{'='*80}")
    print(f"Config: {name} (N={N}, H={H}, D={D})")
    print(f"{'='*80}")

    mx.random.seed(42)
    o_cached = mx.random.normal((N, H, D)).astype(mx.float32)
    lse_cached = mx.random.normal((N, H)).astype(mx.float32)
    o_fresh = mx.random.normal((N, H, D)).astype(mx.float32)
    lse_fresh = mx.random.normal((N, H)).astype(mx.float32)
    mx.eval(o_cached, lse_cached, o_fresh, lse_fresh)

    def run_orig():
        return merge_attention_states(o_cached, lse_cached, o_fresh, lse_fresh)

    def run_fused():
        return merge_attention_states_fused(o_cached, lse_cached, o_fresh, lse_fresh)

    print("  [1/3] Benchmark original (5 MLX ops)...")
    t_orig = measure_ms(run_orig, warmup=20, iters=100)

    print("  [2/3] Benchmark fused (1 Metal kernel)...")
    t_fused = measure_ms(run_fused, warmup=20, iters=100)

    print("  [3/3] Verify correctness...")
    o_ref, lse_ref = run_orig()
    o_test, lse_test = run_fused()
    mx.eval(o_ref, lse_ref, o_test, lse_test)

    diff_o = mx.abs(o_ref - o_test).max().item()
    diff_lse = mx.abs(lse_ref - lse_test).max().item()

    print()
    print(f"  Results:")
    print(f"    Original: {t_orig:.3f} ms")
    print(f"    Fused:    {t_fused:.3f} ms")
    print(f"    Speedup:  {t_orig / t_fused:.2f}×")
    print(f"    O diff:   {diff_o:.2e}")
    print(f"    LSE diff: {diff_lse:.2e}")

    if diff_o < 1e-4:
        print(f"    ✅ Correct")
    else:
        print(f"    ❌ Wrong")

    print()

print("=" * 80)
print("分析")
print("=" * 80)
print()
print("如果这里显示加速，但端到端变慢，说明：")
print("  1. 端到端还有其他开销")
print("  2. MLX 可能自动优化了原始版本的 5 个操作")
print("  3. 或者 wrapper 中的其他部分变慢了")
print()
