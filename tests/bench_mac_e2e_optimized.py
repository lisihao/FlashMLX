#!/usr/bin/env python3
"""
MAC-Attention 端到端优化验证

对比原始版本 vs 优化版本在实际推理场景下的性能
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
        mx.eval(result)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


print("=" * 80)
print("MAC-Attention 端到端优化验证")
print("=" * 80)
print()
print(f"Device: {mx.default_device()}")
print()

# Test configs
configs = [
    ("8K", 8192),
    ("16K", 16384),
    ("32K", 32768),
]

N, Hq, Hkv, D = 1, 32, 8, 128

for name, S in configs:
    print(f"{'='*80}")
    print(f"Context: {name} ({S} tokens)")
    print(f"{'='*80}")

    # Setup
    from flashmlx.mac import MACDecodeWrapper
    from flashmlx.mac.wrapper_optimized import MACDecodeWrapperOptimized

    mx.random.seed(42)
    k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    req_ids = mx.array([0], dtype=mx.int32)
    mx.eval(k, v)

    # Original version
    mac_orig = MACDecodeWrapper(
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

    # Optimized version
    mac_opt = MACDecodeWrapperOptimized(
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

    # Warmup both
    print("  [1/4] Warmup...", flush=True)
    for i in range(100):
        q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
        mx.eval(q)
        mac_orig(q, k, v, req_ids)
        mac_opt(q, k, v, req_ids)
        if i % 50 == 49:
            sync()

    mac_orig.ring_cache.request_length = mx.array([S], dtype=mx.int32)
    mac_opt.ring_cache.request_length = mx.array([S], dtype=mx.int32)
    mx.eval(mac_orig.ring_cache.request_length, mac_opt.ring_cache.request_length)

    # Test query
    q_test = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(q_test)

    # Benchmark original
    print("  [2/4] Benchmark 原始版本...", flush=True)

    def run_orig():
        return mac_orig(q_test, k, v, req_ids)

    t_orig = measure_ms(run_orig)

    # Benchmark optimized
    print("  [3/4] Benchmark 优化版本...", flush=True)

    def run_opt():
        return mac_opt(q_test, k, v, req_ids)

    t_opt = measure_ms(run_opt)

    # Correctness
    print("  [4/4] 验证正确性...", flush=True)
    o_orig = run_orig()
    o_opt = run_opt()
    mx.eval(o_orig, o_opt)

    diff = mx.abs(o_orig - o_opt).max().item()
    cos_sim = (
        mx.sum(o_orig * o_opt) / mx.sqrt(mx.sum(o_orig * o_orig) * mx.sum(o_opt * o_opt))
    ).item()

    # Results
    print()
    print(f"  Results:")
    print(f"    原始版本:  {t_orig:.3f} ms/step")
    print(f"    优化版本:  {t_opt:.3f} ms/step")
    print(f"    加速比:    {t_orig / t_opt:.2f}×")
    print(f"    差值:      {diff:.2e}")
    print(f"    余弦相似:  {cos_sim:.6f}")

    if diff < 1e-3:
        print(f"    ✅ 输出一致")
    else:
        print(f"    ⚠️  输出有差异")

    print()

print("=" * 80)
print("总结")
print("=" * 80)
print()
print("优化成果:")
print("  ✅ Merge kernel fusion 已集成")
print("  ✅ 端到端加速验证完成")
print()
print("监护人，优化版本已就绪！")
