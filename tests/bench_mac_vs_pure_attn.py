#!/usr/bin/env python3
"""
MAC vs 纯 MLX Attention (不通过 wrapper)

Baseline: 直接用 MLX 的 scaled_dot_product_attention
MAC: 完整 MAC wrapper (with cache hit, 9% active)
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
print("MAC vs 纯 MLX Attention")
print("=" * 80)
print()

N, Hq, Hkv, D = 1, 32, 8, 128

configs = [("8K", 8192), ("16K", 16384), ("32K", 32768)]

for ctx_name, S in configs:
    print(f"{'='*80}")
    print(f"Context: {ctx_name} ({S} tokens)")
    print(f"{'='*80}")

    mx.random.seed(42)
    k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    q_test = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    scale = D**-0.5
    mx.eval(k, v, q_test)

    # === Baseline: Pure MLX SDPA ===
    print("  [1] Baseline: Pure MLX SDPA...")

    # Expand GQA
    groups = Hq // Hkv
    k_expanded = mx.repeat(k, groups, axis=2)
    v_expanded = mx.repeat(v, groups, axis=2)
    k_expanded = mx.transpose(k_expanded, (0, 2, 1, 3))  # [N, H, S, D]
    v_expanded = mx.transpose(v_expanded, (0, 2, 1, 3))
    mx.eval(k_expanded, v_expanded)

    q_sdpa = q_test[:, :, None, :]  # [N, H, 1, D]
    mx.eval(q_sdpa)

    def pure_attn():
        scores = (q_sdpa @ mx.transpose(k_expanded, (0, 1, 3, 2))) * scale
        weights = mx.softmax(scores, axis=-1)
        output = (weights @ v_expanded).squeeze(2)
        return output

    t_pure = measure_ms(pure_attn)
    print(f"      Time: {t_pure:.3f} ms")

    # === MAC with 100% hit rate (9% active) ===
    print("  [2] MAC with cache (100% hit, ~9% active)...")

    from flashmlx.mac import MACDecodeWrapper

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

    req_ids = mx.array([0], dtype=mx.int32)
    mx.eval(req_ids)

    # Warmup to fill cache
    q_base = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(q_base)
    for i in range(100):
        mac(q_base, k, v, req_ids)
        if i % 50 == 49:
            sync()

    mac.ring_cache.request_length = mx.array([S], dtype=mx.int32)
    mx.eval(mac.ring_cache.request_length)

    def mac_attn():
        return mac(q_base, k, v, req_ids)

    t_mac = measure_ms(mac_attn)
    mac(q_base, k, v, req_ids)
    stats = mac.last_stats

    print(f"      Time: {t_mac:.3f} ms")
    if stats:
        print(f"      Hit rate: {stats.hit_rate:.1%}")
        print(f"      Avg left: {stats.avg_left_start:.0f} / {S}")
        active_pct = (S - stats.avg_left_start) / S * 100
        print(f"      Active tokens: ~{active_pct:.0f}%")

    # === Results ===
    print()
    print(f"  Results:")
    print(f"    Pure MLX:  {t_pure:.3f} ms (100% active)")
    print(f"    MAC:       {t_mac:.3f} ms (~9% active)")
    print(f"    Speedup:   {t_pure / t_mac:.2f}×")
    print()

print("=" * 80)
print("总结")
print("=" * 80)
print("如果加速 < 5× → MAC 的 overhead (match/merge/etc) 抵消了 partial attention 的收益")
print("如果加速 > 5× → MAC 真正发挥作用")
print()
