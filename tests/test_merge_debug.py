#!/usr/bin/env python3
"""
Debug merge kernel - 测试所有版本
"""

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
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


print("=" * 80)
print("Merge Kernel Debug - 测试所有版本")
print("=" * 80)
print()

# Test data
N, H, D = 1, 32, 128

mx.random.seed(42)
o_cached = mx.random.normal((N, H, D))
lse_cached = mx.random.normal((N, H))
o_fresh = mx.random.normal((N, H, D))
lse_fresh = mx.random.normal((N, H))
mx.eval(o_cached, lse_cached, o_fresh, lse_fresh)

# Reference: Original MLX implementation
from flashmlx.mac.attention import merge_attention_states


def run_original():
    return merge_attention_states(o_cached, lse_cached, o_fresh, lse_fresh)


print("[1/4] Original (5 MLX ops)...")
t_orig = measure_ms(run_original)
o_ref, lse_ref = run_original()
mx.eval(o_ref, lse_ref)
print(f"      Time: {t_orig:.3f} ms")
print()

# Version 1: Buggy fused kernel
print("[2/4] Fused V1 (buggy)...")
try:
    from flashmlx.mac.merge_fused import merge_attention_states_fused

    def run_v1():
        return merge_attention_states_fused(o_cached, lse_cached, o_fresh, lse_fresh)

    t_v1 = measure_ms(run_v1)
    o_v1, lse_v1 = run_v1()
    mx.eval(o_v1, lse_v1)

    o_diff_v1 = mx.abs(o_ref - o_v1).max().item()
    lse_diff_v1 = mx.abs(lse_ref - lse_v1).max().item()

    print(f"      Time: {t_v1:.3f} ms")
    print(f"      O diff: {o_diff_v1:.2e}, LSE diff: {lse_diff_v1:.2e}")
    if o_diff_v1 < 1e-4:
        print("      ✅ Correct")
    else:
        print("      ❌ Bug detected!")
except Exception as e:
    print(f"      ❌ Failed: {e}")
    t_v1 = None

print()

# Version 2: Fixed with shared memory
print("[3/4] Fused V2 (threadgroup per n,h)...")
try:
    from flashmlx.mac.merge_fused_v2 import merge_attention_states_fused_v2

    def run_v2():
        return merge_attention_states_fused_v2(o_cached, lse_cached, o_fresh, lse_fresh)

    t_v2 = measure_ms(run_v2)
    o_v2, lse_v2 = run_v2()
    mx.eval(o_v2, lse_v2)

    o_diff_v2 = mx.abs(o_ref - o_v2).max().item()
    lse_diff_v2 = mx.abs(lse_ref - lse_v2).max().item()

    print(f"      Time: {t_v2:.3f} ms")
    print(f"      O diff: {o_diff_v2:.2e}, LSE diff: {lse_diff_v2:.2e}")
    if o_diff_v2 < 1e-4:
        print("      ✅ Correct")
        print(f"      Speedup: {t_orig / t_v2:.2f}×")
    else:
        print("      ❌ Bug detected!")
except Exception as e:
    print(f"      ❌ Failed: {e}")
    import traceback

    traceback.print_exc()
    t_v2 = None

print()

# Larger test
print("[4/4] Stress test (N=2, H=32, D=128)...")
N2, H2, D2 = 2, 32, 128
o_c2 = mx.random.normal((N2, H2, D2))
lse_c2 = mx.random.normal((N2, H2))
o_f2 = mx.random.normal((N2, H2, D2))
lse_f2 = mx.random.normal((N2, H2))
mx.eval(o_c2, lse_c2, o_f2, lse_f2)

o_ref2, lse_ref2 = merge_attention_states(o_c2, lse_c2, o_f2, lse_f2)
mx.eval(o_ref2, lse_ref2)

if t_v2 is not None:
    try:
        o_v2_2, lse_v2_2 = merge_attention_states_fused_v2(o_c2, lse_c2, o_f2, lse_f2)
        mx.eval(o_v2_2, lse_v2_2)

        diff_o = mx.abs(o_ref2 - o_v2_2).max().item()
        diff_lse = mx.abs(lse_ref2 - lse_v2_2).max().item()

        print(f"      O diff: {diff_o:.2e}, LSE diff: {diff_lse:.2e}")
        if diff_o < 1e-4:
            print("      ✅ V2 passes stress test")
        else:
            print("      ❌ V2 fails stress test")
    except Exception as e:
        print(f"      ❌ V2 stress test failed: {e}")
else:
    print("      ⏭️  Skipped (V2 not available)")

print()
print("=" * 80)
print("Summary")
print("=" * 80)
print()
print(f"Original:  {t_orig:.3f} ms (reference)")
if t_v1:
    print(f"V1 (buggy): {t_v1:.3f} ms - {t_orig/t_v1:.2f}× {'但有bug!' if o_diff_v1 > 1e-4 else ''}")
if t_v2:
    print(f"V2 (fixed): {t_v2:.3f} ms - {t_orig/t_v2:.2f}×")
print()
print("监护人，V2应该是正确的版本！")
