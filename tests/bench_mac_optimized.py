#!/usr/bin/env python3
"""
MAC-Attention 优化验证 Benchmark

对比:
1. 原始实现 vs 优化实现
2. Match kernel 加速比
3. Merge kernel 加速比
4. 端到端加速比

目标:
- Match: 200μs → <50μs (4× faster)
- Merge: 175μs → <30μs (6× faster)
- E2E: 2.07ms → <1.0ms (2× faster) @ 32K context
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
        if isinstance(result, (tuple, list)):
            mx.eval(*result)
        elif result is not None:
            mx.eval(result)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def test_match_kernel():
    """测试 Match kernel 优化"""
    from flashmlx.mac import MACRingCache, mac_ring_match

    print("=" * 72)
    print("Match Kernel 优化测试")
    print("=" * 72)
    print()

    N, Hq, D = 1, 32, 128
    M = 512

    ring_cache = MACRingCache(max_requests=4, capacity=M, num_heads=Hq, head_dim=D)

    # 填满 cache
    mx.random.seed(42)
    for i in range(M):
        q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
        ring_cache.update(q, mx.array([0], dtype=mx.int32), i)

    q_test = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    req_ids = mx.array([0], dtype=mx.int32)
    mx.eval(q_test)
    ring_cache.request_length = mx.array([M], dtype=mx.int32)
    mx.eval(ring_cache.request_length)

    # 原始版本
    def run_original():
        return mac_ring_match(ring_cache, q_test, req_ids, threshold=0.5, band_r=256)

    t_original = measure_ms(run_original)

    print(f"  原始 Match kernel: {t_original:.3f} ms")
    print()

    # 尝试导入优化版本
    try:
        from flashmlx.mac.match_optimized import mac_ring_match_optimized

        def run_optimized():
            return mac_ring_match_optimized(
                ring_cache, q_test, req_ids, threshold=0.5, band_r=256
            )

        t_optimized = measure_ms(run_optimized)

        print(f"  优化 Match kernel: {t_optimized:.3f} ms")
        print(f"  加速比: {t_original / t_optimized:.2f}×")

        # 验证正确性
        hit_o, left_o, idx_o = run_original()
        hit_n, left_n, idx_n = run_optimized()
        mx.eval(hit_o, left_o, idx_o, hit_n, left_n, idx_n)

        hit_match = mx.all(hit_o == hit_n).item()
        left_match = mx.all(left_o == left_n).item()
        idx_match = mx.all(idx_o == idx_n).item()

        if hit_match and left_match and idx_match:
            print("  ✅ 输出完全一致")
        else:
            print("  ❌ 输出不一致！")
            print(f"     hit匹配: {hit_match}, left匹配: {left_match}, idx匹配: {idx_match}")

    except ImportError as e:
        print(f"  ⚠️  优化版本未实现或编译失败: {e}")
        print("  → 需要修复 Metal kernel 语法错误")


def test_merge_kernel():
    """测试 Merge kernel 优化"""
    from flashmlx.mac.attention import merge_attention_states

    print()
    print("=" * 72)
    print("Merge Kernel 优化测试")
    print("=" * 72)
    print()

    N, H, D = 1, 32, 128

    mx.random.seed(42)
    o_cached = mx.random.normal((N, H, D))
    lse_cached = mx.random.normal((N, H))
    o_fresh = mx.random.normal((N, H, D))
    lse_fresh = mx.random.normal((N, H))
    mx.eval(o_cached, lse_cached, o_fresh, lse_fresh)

    # 原始版本（5次MLX操作）
    def run_original():
        return merge_attention_states(o_cached, lse_cached, o_fresh, lse_fresh)

    t_original = measure_ms(run_original)

    print(f"  原始 Merge (5 MLX ops): {t_original:.3f} ms")
    print()

    # 尝试导入优化版本
    try:
        from flashmlx.mac.merge_fused import merge_attention_states_fused

        def run_optimized():
            return merge_attention_states_fused(
                o_cached, lse_cached, o_fresh, lse_fresh
            )

        t_optimized = measure_ms(run_optimized)

        print(f"  优化 Merge (1 Metal kernel): {t_optimized:.3f} ms")
        print(f"  加速比: {t_original / t_optimized:.2f}×")

        # 验证正确性
        o_o, lse_o = run_original()
        o_n, lse_n = run_optimized()
        mx.eval(o_o, lse_o, o_n, lse_n)

        o_diff = mx.abs(o_o - o_n).max().item()
        lse_diff = mx.abs(lse_o - lse_n).max().item()

        print(f"  Max diff - o: {o_diff:.6f}, lse: {lse_diff:.6f}")
        if o_diff < 1e-4 and lse_diff < 1e-4:
            print("  ✅ 输出数值一致")
        else:
            print("  ❌ 输出差异过大！")

    except ImportError as e:
        print(f"  ⚠️  优化版本未实现或编译失败: {e}")
        print("  → 需要修复 Metal kernel 语法错误")


def test_e2e():
    """端到端测试（32K context）"""
    from flashmlx.mac import MACDecodeWrapper

    print()
    print("=" * 72)
    print("端到端测试 (32K context)")
    print("=" * 72)
    print()

    N, Hq, Hkv, D, S = 1, 32, 8, 128, 32768

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

    mx.random.seed(42)
    k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    req_ids = mx.array([0], dtype=mx.int32)
    mx.eval(k, v)

    # Warmup
    for i in range(100):
        q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
        mx.eval(q)
        mac(q, k, v, req_ids)
        sync()

    mac.ring_cache.request_length = mx.array([S], dtype=mx.int32)
    mx.eval(mac.ring_cache.request_length)

    # 测试
    q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(q)

    def run_e2e():
        return mac(q, k, v, req_ids)

    t_e2e = measure_ms(run_e2e)

    print(f"  MAC E2E (32K): {t_e2e:.3f} ms")
    print(f"  预期优化后: ~1.0 ms (当前 {t_e2e:.3f} ms)")
    print()
    print("  ⚠️  完整优化需要：")
    print("     1. Match kernel优化 (-150 μs)")
    print("     2. Merge kernel优化 (-140 μs)")
    print("     3. 异步Rectify/Update (移出关键路径)")


if __name__ == "__main__":
    print(f"Device: {mx.default_device()}")
    print()

    test_match_kernel()
    test_merge_kernel()
    test_e2e()

    print()
    print("=" * 72)
    print("总结")
    print("=" * 72)
    print()
    print("监护人，我已经写好了优化代码：")
    print("  ✅ match_optimized.py - Match kernel 优化 (float8 vectorized)")
    print("  ✅ merge_fused.py - Merge/Downdate 融合 kernel")
    print()
    print("下一步:")
    print("  1. 修复 Metal kernel 编译错误（如果有）")
    print("  2. 验证优化正确性")
    print("  3. 实测加速效果")
    print("  4. 集成到 MACDecodeWrapper")
    print()
    print("预期收益：")
    print("  - Match: 200μs → 50μs (4× faster)")
    print("  - Merge: 175μs → 30μs (6× faster)")
    print("  - E2E: 2.07ms → 1.0ms (2× faster)")
