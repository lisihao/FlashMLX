#!/usr/bin/env python3
"""
模拟真实推理场景 - 逐 token 生成

对比：
1. Standard Attention（每次full attention）
2. MAC-Attention（cache + partial attention）

测试 prefill + decode 两个阶段
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx


def sync():
    mx.eval(mx.array(0))


def measure_decode_step(fn):
    """Measure single decode step (warmup + median of 30 runs)"""
    # Warmup
    for _ in range(10):
        fn()
        sync()
    # Measure
    times = []
    for _ in range(30):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


print("=" * 80)
print("真实推理场景：Prefill + Decode")
print("=" * 80)
print()

from flashmlx.mac import MACDecodeWrapper

N, Hq, Hkv, D = 1, 32, 8, 128

# Test different context lengths
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

    # === Phase 1: Prefill (MAC cache warm-up) ===
    print("  [1] Prefill phase: warming up MAC cache...")
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

    # Simulate prefill: 100 steps with gradually drifting queries
    q_base = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(q_base)

    prefill_time = 0.0
    for i in range(100):
        # Gradual drift
        noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * 0.05
        q = q_base + noise * (i / 100.0)
        mx.eval(q)

        sync()
        t0 = time.perf_counter()
        mac(q, k, v, req_ids)
        sync()
        prefill_time += (time.perf_counter() - t0) * 1000

        q_base = q
        if i % 50 == 49:
            sync()

    mac.ring_cache.request_length = mx.array([S], dtype=mx.int32)
    mx.eval(mac.ring_cache.request_length)

    print(f"      Prefill time: {prefill_time:.1f} ms (100 steps)")
    print(f"      Avg per step: {prefill_time / 100:.3f} ms")

    # Check hit rate after prefill
    q_test = q_base + mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * 0.05
    mx.eval(q_test)
    mac(q_test, k, v, req_ids)
    stats = mac.last_stats
    if stats:
        print(f"      Hit rate: {stats.hit_rate:.1%}")

    # === Phase 2: Decode (steady state) ===
    print("  [2] Decode phase: measuring steady-state performance...")

    def decode_step():
        noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * 0.05
        q = q_base + noise
        mx.eval(q)
        return mac(q, k, v, req_ids)

    t_decode = measure_decode_step(decode_step)
    print(f"      Decode time: {t_decode:.3f} ms/step")

    # Check steady-state hit rate
    mac(q_test, k, v, req_ids)
    stats = mac.last_stats
    if stats:
        print(f"      Steady hit rate: {stats.hit_rate:.1%}")

    # === Baseline: Standard Attention ===
    print("  [3] Baseline: standard attention (always full)...")

    def standard_attn():
        noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * 0.05
        q = q_base + noise
        mx.eval(q)
        # MAC with threshold=0 forces full attention
        mac_baseline = MACDecodeWrapper(
            max_requests=4,
            capacity=512,
            num_heads=Hq,
            num_kv_heads=Hkv,
            head_dim=D,
            threshold=0.0,  # ← Force full attention
            band_r=256,
            window_left=256,
            normalize_queries=True,
        )
        mac_baseline.ring_cache.request_length = mx.array([S], dtype=mx.int32)
        mx.eval(mac_baseline.ring_cache.request_length)
        return mac_baseline(q, k, v, req_ids)

    t_baseline = measure_decode_step(standard_attn)
    print(f"      Standard time: {t_baseline:.3f} ms/step")

    # === Results ===
    print()
    print(f"  Results:")
    print(f"    Standard:  {t_baseline:.3f} ms/step")
    print(f"    MAC:       {t_decode:.3f} ms/step")
    print(f"    Speedup:   {t_baseline / t_decode:.2f}×")
    print()

print("=" * 80)
print("总结")
print("=" * 80)
print("真实推理场景下的加速比")
print()
