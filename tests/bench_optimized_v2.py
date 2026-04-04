#!/usr/bin/env python3
"""
测试应用 block-stride 后的性能

对比：
1. 纯 MLX SDPA (baseline)
2. MAC v1 (原始 fused kernel)
3. MAC v2 (+ block-stride)
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
print("MAC v2 优化验证 (Block-Stride)")
print("=" * 80)
print()

from flashmlx.mac import MACDecodeWrapper
from flashmlx.mac.wrapper_optimized import MACDecodeWrapperOptimized

N, Hq, Hkv, D = 1, 32, 8, 128

configs = [("8K", 8192), ("16K", 16384), ("32K", 32768)]

for ctx_name, S in configs:
    print(f"{'='*80}")
    print(f"Context: {ctx_name} ({S} tokens)")
    print(f"{'='*80}")

    mx.random.seed(42)
    k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    req_ids = mx.array([0], dtype=mx.int32)
    mx.eval(k, v)

    # === Baseline: Pure MLX ===
    print("  [1] Baseline: Pure MLX SDPA...")
    groups = Hq // Hkv
    k_exp = mx.repeat(k, groups, axis=2)
    v_exp = mx.repeat(v, groups, axis=2)
    k_exp = mx.transpose(k_exp, (0, 2, 1, 3))
    v_exp = mx.transpose(v_exp, (0, 2, 1, 3))
    mx.eval(k_exp, v_exp)

    q_test = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    q_sdpa = q_test[:, :, None, :]
    mx.eval(q_test, q_sdpa)

    scale = D**-0.5

    def pure_attn():
        scores = (q_sdpa @ mx.transpose(k_exp, (0, 1, 3, 2))) * scale
        weights = mx.softmax(scores, axis=-1)
        return (weights @ v_exp).squeeze(2)

    t_pure = measure_ms(pure_attn)
    print(f"      Time: {t_pure:.3f} ms")

    # === MAC v1: Original ===
    print("  [2] MAC v1: Original fused kernel...")

    mac_v1 = MACDecodeWrapper(
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

    q_base = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(q_base)

    for i in range(100):
        mac_v1(q_base, k, v, req_ids)
        if i % 50 == 49:
            sync()

    mac_v1.ring_cache.request_length = mx.array([S], dtype=mx.int32)
    mx.eval(mac_v1.ring_cache.request_length)

    def mac_v1_fn():
        return mac_v1(q_base, k, v, req_ids)

    t_v1 = measure_ms(mac_v1_fn)
    print(f"      Time: {t_v1:.3f} ms")
    print(f"      Speedup vs baseline: {t_pure / t_v1:.2f}×")

    # === MAC v2: + Block-Stride ===
    print("  [3] MAC v2: + Block-Stride...")

    mac_v2 = MACDecodeWrapperOptimized(
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

    for i in range(100):
        mac_v2(q_base, k, v, req_ids)
        if i % 50 == 49:
            sync()

    mac_v2.ring_cache.request_length = mx.array([S], dtype=mx.int32)
    mx.eval(mac_v2.ring_cache.request_length)

    def mac_v2_fn():
        return mac_v2(q_base, k, v, req_ids)

    t_v2 = measure_ms(mac_v2_fn)
    print(f"      Time: {t_v2:.3f} ms")
    print(f"      Speedup vs baseline: {t_pure / t_v2:.2f}×")
    print(f"      Speedup vs v1: {t_v1 / t_v2:.2f}×")

    print()
    print(f"  Results:")
    print(f"    Pure MLX:  {t_pure:.3f} ms")
    print(f"    MAC v1:    {t_v1:.3f} ms ({t_pure/t_v1:.2f}×)")
    print(f"    MAC v2:    {t_v2:.3f} ms ({t_pure/t_v2:.2f}×)")
    print()

print("=" * 80)
print("总结")
print("=" * 80)
print("目标：32K 达到 8.0× (当前 7.46× → 预期 +7%)")
print()
