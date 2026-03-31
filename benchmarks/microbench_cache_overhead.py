#!/usr/bin/env python3
"""
Micro-benchmark: Cache Management Overhead

Direct comparison:
1. Direct memory access (no cache)
2. Cache hit (single tier)
3. Cache miss (3 tier lookups)
"""

import time
import mlx.core as mx
import numpy as np
from flashmlx.cache import HybridCacheManager, HybridCacheConfig, LayerType


def benchmark_direct_access(num_iterations: int = 10000):
    """Benchmark direct memory access without cache."""
    # Create a simple array
    state = mx.array(np.random.randn(256).astype(np.float32))
    stored_states = {i: state for i in range(30)}

    start = time.time()
    for _ in range(num_iterations):
        for i in range(30):
            _ = stored_states[i]  # Direct dict lookup
    end = time.time()

    elapsed_ms = (end - start) * 1000
    ops_per_sec = (num_iterations * 30) / (end - start)

    return {
        'elapsed_ms': elapsed_ms,
        'ops_per_sec': ops_per_sec,
        'latency_us': (elapsed_ms * 1000) / (num_iterations * 30)
    }


def benchmark_cache_hit(num_iterations: int = 10000):
    """Benchmark cache hit (data in Hot tier)."""
    layer_types = {i: LayerType.SSM for i in range(30)}
    config = HybridCacheConfig(total_budget_bytes=10 * 1024 * 1024, compression_ratio=4.0)
    manager = HybridCacheManager(config=config, layer_types=layer_types)

    # Populate cache (all in Hot)
    state = mx.array(np.random.randn(256).astype(np.float32))
    for i in range(30):
        manager.store_ssm(i, state, state.nbytes, 1.0)

    start = time.time()
    for _ in range(num_iterations):
        for i in range(30):
            _ = manager.retrieve_ssm(i)  # Cache hit
    end = time.time()

    elapsed_ms = (end - start) * 1000
    ops_per_sec = (num_iterations * 30) / (end - start)

    return {
        'elapsed_ms': elapsed_ms,
        'ops_per_sec': ops_per_sec,
        'latency_us': (elapsed_ms * 1000) / (num_iterations * 30)
    }


def benchmark_cache_miss(num_iterations: int = 10000):
    """Benchmark cache miss (3 tier lookups)."""
    layer_types = {i: LayerType.SSM for i in range(60)}  # 60 layers
    config = HybridCacheConfig(total_budget_bytes=512 * 1024, compression_ratio=4.0)  # Small budget
    manager = HybridCacheManager(config=config, layer_types=layer_types)

    # Populate cache (will evict most layers)
    state = mx.array(np.random.randn(25600).astype(np.float32))  # 100KB
    for i in range(60):
        manager.store_ssm(i, state, state.nbytes, 1.0)

    # Access layers that are likely evicted
    start = time.time()
    for _ in range(num_iterations):
        for i in range(30, 60):  # Access evicted layers
            _ = manager.retrieve_ssm(i)  # Cache miss (3 tier lookups)
    end = time.time()

    elapsed_ms = (end - start) * 1000
    ops_per_sec = (num_iterations * 30) / (end - start)

    return {
        'elapsed_ms': elapsed_ms,
        'ops_per_sec': ops_per_sec,
        'latency_us': (elapsed_ms * 1000) / (num_iterations * 30)
    }


def main():
    print("="*70)
    print("Micro-benchmark: Cache Management Overhead")
    print("="*70)

    num_iterations = 1000  # 1000 iterations × 30 layers = 30,000 operations

    print(f"\nRunning {num_iterations} iterations × 30 layers = {num_iterations * 30:,} total operations\n")

    # Benchmark 1: Direct access
    print("Benchmark 1: Direct Memory Access (no cache)")
    result_direct = benchmark_direct_access(num_iterations)
    print(f"  Elapsed:    {result_direct['elapsed_ms']:.2f} ms")
    print(f"  Throughput: {result_direct['ops_per_sec']:,.0f} ops/s")
    print(f"  Latency:    {result_direct['latency_us']:.3f} μs/op")

    # Benchmark 2: Cache hit
    print("\nBenchmark 2: Cache Hit (data in Hot tier)")
    result_hit = benchmark_cache_hit(num_iterations)
    print(f"  Elapsed:    {result_hit['elapsed_ms']:.2f} ms")
    print(f"  Throughput: {result_hit['ops_per_sec']:,.0f} ops/s")
    print(f"  Latency:    {result_hit['latency_us']:.3f} μs/op")

    # Benchmark 3: Cache miss
    print("\nBenchmark 3: Cache Miss (3 tier lookups)")
    result_miss = benchmark_cache_miss(num_iterations)
    print(f"  Elapsed:    {result_miss['elapsed_ms']:.2f} ms")
    print(f"  Throughput: {result_miss['ops_per_sec']:,.0f} ops/s")
    print(f"  Latency:    {result_miss['latency_us']:.3f} μs/op")

    # Summary
    print(f"\n{'='*70}")
    print("Overhead Analysis")
    print(f"{'='*70}")

    hit_overhead = ((result_hit['latency_us'] - result_direct['latency_us']) / result_direct['latency_us']) * 100
    miss_overhead = ((result_miss['latency_us'] - result_direct['latency_us']) / result_direct['latency_us']) * 100

    print(f"\nCache Hit Overhead:  {hit_overhead:+.1f}% ({result_hit['latency_us']:.3f} vs {result_direct['latency_us']:.3f} μs)")
    print(f"Cache Miss Overhead: {miss_overhead:+.1f}% ({result_miss['latency_us']:.3f} vs {result_direct['latency_us']:.3f} μs)")

    print(f"\nConclusion:")
    if hit_overhead > 50:
        print(f"  ⚠️  Cache hit overhead ({hit_overhead:.0f}%) is too high!")
        print(f"  → Cache management cost > memory access cost")
        print(f"  → Not cost-effective for small SSM states")
    elif miss_overhead > 200:
        print(f"  ⚠️  Cache miss overhead ({miss_overhead:.0f}%) is excessive!")
        print(f"  → 3-tier lookup is too expensive")
        print(f"  → Need early-exit optimization")
    else:
        print(f"  ✅ Cache overhead is acceptable")

    print(f"\n{'='*70}")
    print("✓ Benchmark Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
