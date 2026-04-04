#!/usr/bin/env python3
"""
MAC-Attention Metal/MLX Performance Benchmark

Measures:
  1. Match kernel latency at various cache sizes (M)
  2. Full attention vs MAC decode step (hit/miss scenarios)
  3. Scaling: heads, head_dim, batch size
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx


def bench_fn(fn, warmup=5, iters=50, label=""):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()
        mx.eval(mx.array(0))  # sync

    times = []
    for _ in range(iters):
        mx.eval(mx.array(0))  # sync before
        t0 = time.perf_counter()
        result = fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        elif result is not None:
            mx.eval(result)
        else:
            mx.eval(mx.array(0))
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    p10 = times[len(times) // 10]
    p90 = times[int(len(times) * 0.9)]
    return median, p10, p90


def bench_match_kernel():
    """Benchmark match kernel at different cache sizes."""
    from flashmlx.mac import MACRingCache, mac_ring_match

    print("\n" + "=" * 70)
    print("Benchmark 1: Match Kernel Latency (Metal)")
    print("=" * 70)
    print(f"{'M':>6} {'H':>4} {'D':>4} {'N':>4} {'median':>10} {'p10':>10} {'p90':>10}")
    print("-" * 70)

    for M in [64, 256, 512, 1024, 2048, 4096]:
        for H, D in [(8, 128), (32, 128)]:
            N = 1
            R = 4
            cache = MACRingCache(R, M, H, D)

            # Fill cache
            mx.random.seed(0)
            fill_n = min(M, 512)
            for i in range(fill_n):
                q = mx.random.normal((1, H, D)).astype(mx.bfloat16)
                a = mx.random.normal((1, H, D)).astype(mx.bfloat16)
                lse = mx.random.normal((1, H)).astype(mx.float32)
                cache.update(mx.array([0], dtype=mx.int32), q, a, lse)
                mx.eval(cache.query_cache, cache.request_length)

            queries = mx.random.normal((N, H, D)).astype(mx.bfloat16)
            req_ids = mx.array([0], dtype=mx.int32)
            mx.eval(queries)

            def run():
                return mac_ring_match(cache, queries, req_ids, threshold=0.6, band_r=256)

            med, p10, p90 = bench_fn(run, warmup=10, iters=100)
            print(f"{M:>6} {H:>4} {D:>4} {N:>4} {med:>8.3f}ms {p10:>8.3f}ms {p90:>8.3f}ms")


def bench_match_vs_reference():
    """Compare Metal kernel vs pure MLX reference."""
    from flashmlx.mac import MACRingCache, mac_ring_match, mac_ring_match_reference

    print("\n" + "=" * 70)
    print("Benchmark 2: Metal Kernel vs Pure MLX Reference")
    print("=" * 70)

    M, H, D, N, R = 512, 8, 128, 1, 4
    cache = MACRingCache(R, M, H, D)

    mx.random.seed(0)
    for i in range(M):
        q = mx.random.normal((1, H, D)).astype(mx.bfloat16)
        a = mx.random.normal((1, H, D)).astype(mx.bfloat16)
        lse = mx.random.normal((1, H)).astype(mx.float32)
        cache.update(mx.array([0], dtype=mx.int32), q, a, lse)
        mx.eval(cache.query_cache, cache.request_length)

    queries = mx.random.normal((N, H, D)).astype(mx.bfloat16)
    req_ids = mx.array([0], dtype=mx.int32)
    mx.eval(queries)

    def run_metal():
        return mac_ring_match(cache, queries, req_ids, threshold=0.6, band_r=256)

    def run_ref():
        return mac_ring_match_reference(cache, queries, req_ids, threshold=0.6, band_r=256)

    med_m, _, _ = bench_fn(run_metal, warmup=10, iters=100, label="Metal")
    med_r, _, _ = bench_fn(run_ref, warmup=5, iters=20, label="Reference")

    print(f"  Metal kernel:   {med_m:.3f} ms")
    print(f"  MLX reference:  {med_r:.3f} ms")
    print(f"  Speedup:        {med_r / med_m:.1f}x")


def bench_full_attention_baseline():
    """Benchmark full attention (no MAC) as baseline."""
    print("\n" + "=" * 70)
    print("Benchmark 3: Full Attention Baseline (mx.fast.scaled_dot_product_attention)")
    print("=" * 70)
    print(f"{'S':>6} {'Hq':>4} {'Hkv':>4} {'D':>4} {'median':>10}")
    print("-" * 50)

    for S in [256, 1024, 4096, 8192]:
        N, Hq, Hkv, D = 1, 32, 8, 128
        q = mx.random.normal((N, Hq, 1, D)).astype(mx.bfloat16)
        k = mx.random.normal((N, Hkv, S, D)).astype(mx.bfloat16)
        v = mx.random.normal((N, Hkv, S, D)).astype(mx.bfloat16)
        mx.eval(q, k, v)

        scale = D ** -0.5

        def run():
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

        med, _, _ = bench_fn(run, warmup=10, iters=100)
        print(f"{S:>6} {Hq:>4} {Hkv:>4} {D:>4} {med:>8.3f}ms")


def bench_e2e_mac_decode():
    """Benchmark end-to-end MAC decode step."""
    from flashmlx.mac import MACDecodeWrapper

    print("\n" + "=" * 70)
    print("Benchmark 4: E2E MAC Decode Step (match + partial attn + merge + rectify)")
    print("=" * 70)
    print(f"{'S':>6} {'M':>6} {'Hq':>4} {'Hkv':>4} {'D':>4} {'step':>6} {'median':>10} {'hit_rate':>10}")
    print("-" * 75)

    for S in [1024, 4096]:
        N, Hq, Hkv, D, M = 1, 8, 2, 128, 512
        mac = MACDecodeWrapper(
            max_requests=4, capacity=M,
            num_heads=Hq, num_kv_heads=Hkv, head_dim=D,
            threshold=0.5, band_r=256, window_left=256,
        )

        mx.random.seed(42)
        k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
        v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
        req_ids = mx.array([0], dtype=mx.int32)
        mx.eval(k, v)

        # Warm up cache with a few steps
        for _ in range(3):
            q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
            mx.eval(q)
            out = mac(q, k, v, req_ids)
            mx.eval(out)

        # Benchmark miss (new random query)
        q_miss = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
        mx.eval(q_miss)

        def run_miss():
            return mac(q_miss, k, v, req_ids)

        med_miss, _, _ = bench_fn(run_miss, warmup=5, iters=50)
        stats_miss = mac.last_stats
        hr_miss = f"{stats_miss.hit_rate:.0%}" if stats_miss else "?"
        print(f"{S:>6} {M:>6} {Hq:>4} {Hkv:>4} {D:>4} {'miss':>6} {med_miss:>8.3f}ms {hr_miss:>10}")

        # Benchmark hit (reuse same query)
        q_hit = q_miss  # same query → should hit after first pass
        mx.eval(q_hit)

        def run_hit():
            return mac(q_hit, k, v, req_ids)

        med_hit, _, _ = bench_fn(run_hit, warmup=5, iters=50)
        stats_hit = mac.last_stats
        hr_hit = f"{stats_hit.hit_rate:.0%}" if stats_hit else "?"
        print(f"{S:>6} {M:>6} {Hq:>4} {Hkv:>4} {D:>4} {'hit':>6} {med_hit:>8.3f}ms {hr_hit:>10}")


def bench_match_batch_scaling():
    """Benchmark match kernel with increasing batch sizes."""
    from flashmlx.mac import MACRingCache, mac_ring_match

    print("\n" + "=" * 70)
    print("Benchmark 5: Match Kernel Batch Scaling")
    print("=" * 70)
    print(f"{'N':>4} {'M':>6} {'H':>4} {'D':>4} {'median':>10} {'per_query':>12}")
    print("-" * 60)

    M, H, D = 512, 8, 128
    for N in [1, 2, 4, 8, 16]:
        R = max(N, 4)
        cache = MACRingCache(R, M, H, D)

        mx.random.seed(0)
        for r in range(min(N, R)):
            for _ in range(M):
                q = mx.random.normal((1, H, D)).astype(mx.bfloat16)
                a = mx.random.normal((1, H, D)).astype(mx.bfloat16)
                lse = mx.random.normal((1, H)).astype(mx.float32)
                cache.update(mx.array([r], dtype=mx.int32), q, a, lse)
                mx.eval(cache.query_cache, cache.request_length)

        queries = mx.random.normal((N, H, D)).astype(mx.bfloat16)
        req_ids = mx.arange(N, dtype=mx.int32)
        mx.eval(queries, req_ids)

        def run():
            return mac_ring_match(cache, queries, req_ids, threshold=0.6, band_r=256)

        med, _, _ = bench_fn(run, warmup=10, iters=100)
        per_q = med / N
        print(f"{N:>4} {M:>6} {H:>4} {D:>4} {med:>8.3f}ms {per_q:>10.3f}ms/q")


def main():
    print("MAC-Attention Metal/MLX Performance Benchmark")
    print(f"Device: {mx.default_device()}")

    bench_match_kernel()
    bench_match_vs_reference()
    bench_full_attention_baseline()
    bench_e2e_mac_decode()
    bench_match_batch_scaling()

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    sys.path.insert(0, "src")
    sys.path.insert(0, "mlx-lm-source")
    main()
