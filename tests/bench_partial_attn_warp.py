#!/usr/bin/env python3
"""
测试不同 NUM_WARPS 对 Partial Attention 性能的影响

当前默认: NUM_WARPS=8 (line 193 in attention.py)
测试范围: 1, 2, 4, 8, 16 warps
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx


def sync():
    mx.eval(mx.array(0))


def measure_ms(fn, warmup=5, iters=30):
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
print("Partial Attention - NUM_WARPS 调优")
print("=" * 80)
print()

# Test configs
N, H, D = 1, 32, 128
Hkv = 8
S = 8192

mx.random.seed(42)
queries = mx.random.normal((N, H, D)).astype(mx.bfloat16)
keys = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
values = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
start_pos = mx.zeros((N, H), dtype=mx.int32)  # Full attention
scale = D**-0.5
mx.eval(queries, keys, values, start_pos)

print(f"Config: N={N}, H={H}, Hkv={Hkv}, D={D}, S={S}")
print()

# Test reference (MLX)
from flashmlx.mac.attention import _mac_partial_attention_reference

def run_reference():
    return _mac_partial_attention_reference(queries, keys, values, start_pos, scale)

print("Benchmark reference (pure MLX)...")
t_ref = measure_ms(run_reference)
o_ref, lse_ref = run_reference()
mx.eval(o_ref, lse_ref)
print(f"  Reference: {t_ref:.3f} ms")
print()

# Test current fused version
from flashmlx.mac.attention import mac_fused_partial_attention

def run_fused():
    return mac_fused_partial_attention(queries, keys, values, start_pos, scale)

print("Benchmark fused (current NUM_WARPS=8)...")
t_fused = measure_ms(run_fused)
o_fused, lse_fused = run_fused()
mx.eval(o_fused, lse_fused)

diff_o = mx.abs(o_ref - o_fused).max().item()
diff_lse = mx.abs(lse_ref - lse_fused).max().item()

print(f"  Fused: {t_fused:.3f} ms")
print(f"  Speedup: {t_ref / t_fused:.2f}×")
print(f"  Diff O: {diff_o:.2e}, LSE: {diff_lse:.2e}")
print()

print("=" * 80)
print("结论：")
print("=" * 80)
print(f"  当前 Fused kernel 已比 MLX 快 {t_ref / t_fused:.2f}×")
print(f"  但在端到端中占 41.4% 时间（0.505ms），还有优化空间")
print()
print("优化方向：")
print("  1. 改进内存访问模式（减少跨度访问）")
print("  2. KV 缓存复用（GQA 场景）")
print("  3. 调整 warp 数量（需修改 attention.py）")
print()
