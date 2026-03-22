#!/usr/bin/env python3
"""
Benchmark: SSM Cache Hit Rate Impact on Performance

Compare performance with different cache hit rates:
1. 100% hit (large budget, all layers cached)
2. 50-70% hit (medium budget, frequent layers cached)
3. 0% hit (no cache / disabled)

Measure:
- Latency per access
- Throughput (accesses/second)
- Memory usage
"""

import time
import mlx.core as mx
import numpy as np
from flashmlx.cache import (
    HybridCacheManager,
    HybridCacheConfig,
    LayerType
)
from flashmlx.cache.per_layer_ssm_cache import PerLayerSSMCache


def create_dummy_ssm_state(layer_idx: int, state_size: int = 1024 * 100):
    """Create dummy SSM state (100KB per layer - realistic size) for testing."""
    np.random.seed(layer_idx)
    state = mx.array(np.random.randn(state_size // 4).astype(np.float32))
    return state


def benchmark_scenario(name: str, budget_kb: int, num_layers: int, access_pattern: list):
    """
    Benchmark a specific cache configuration.

    Args:
        name: Scenario name
        budget_kb: Budget in KB
        num_layers: Number of SSM layers
        access_pattern: List of (layer_idx, num_accesses)
    """
    print(f"\n{'='*70}")
    print(f"Scenario: {name}")
    print(f"{'='*70}")
    print(f"Budget: {budget_kb} KB")
    print(f"Layers: {num_layers}")
    print(f"Access pattern: {len(access_pattern)} unique layers\n")

    # Create manager
    layer_types = {i: LayerType.SSM for i in range(num_layers)}
    config = HybridCacheConfig(
        total_budget_bytes=budget_kb * 1024,
        compression_ratio=4.0
    )
    manager = HybridCacheManager(config=config, layer_types=layer_types)

    # Create caches
    caches = []
    for layer_idx in range(num_layers):
        cache = PerLayerSSMCache(manager, layer_idx, size=2)
        cache.enable_managed_cache()
        caches.append(cache)

    # Phase 1: Initial population
    print("Phase 1: Populating cache...")
    for layer_idx in range(num_layers):
        state = create_dummy_ssm_state(layer_idx)
        caches[layer_idx][0] = state

    # Phase 2: Benchmark access pattern
    print("Phase 2: Running benchmark...")
    total_accesses = sum(count for _, count in access_pattern)

    start_time = time.time()

    for layer_idx, num_accesses in access_pattern:
        for _ in range(num_accesses):
            _ = caches[layer_idx][0]

    end_time = time.time()
    elapsed = end_time - start_time

    # Get statistics
    stats = manager.get_statistics()
    ssm_hot = stats['ssm']['hot']
    ssm_warm = stats['ssm']['warm']
    ssm_cold = stats['ssm']['cold']

    # Calculate metrics
    total_hits = (
        ssm_hot.get('total_hits', 0) +
        ssm_warm.get('total_hits', 0) +
        ssm_cold.get('total_hits', 0)
    )
    total_misses = (
        ssm_hot.get('total_misses', 0) +
        ssm_warm.get('total_misses', 0) +
        ssm_cold.get('total_misses', 0)
    )
    total_cache_accesses = total_hits + total_misses
    hit_rate = total_hits / total_cache_accesses if total_cache_accesses > 0 else 0.0

    # Performance metrics
    throughput = total_accesses / elapsed if elapsed > 0 else 0
    latency_ms = (elapsed / total_accesses) * 1000 if total_accesses > 0 else 0

    # Print results
    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")
    print(f"\nCache Statistics:")
    print(f"  Hit rate:       {hit_rate:>8.1%}")
    print(f"  Total hits:     {total_hits:>8}")
    print(f"  Total misses:   {total_misses:>8}")
    print(f"  Cached layers:  {ssm_hot.get('entry_count', 0) + ssm_warm.get('entry_count', 0) + ssm_cold.get('entry_count', 0):>8} / {num_layers}")

    print(f"\nPerformance:")
    print(f"  Total accesses: {total_accesses:>8}")
    print(f"  Elapsed time:   {elapsed:>8.3f} s")
    print(f"  Throughput:     {throughput:>8.0f} accesses/s")
    print(f"  Avg latency:    {latency_ms:>8.3f} ms/access")

    print(f"\nTier Distribution:")
    print(f"  Hot:   {ssm_hot.get('entry_count', 0):>3} layers (hit rate: {ssm_hot.get('hit_rate', 0.0):>6.1%})")
    print(f"  Warm:  {ssm_warm.get('entry_count', 0):>3} layers (hit rate: {ssm_warm.get('hit_rate', 0.0):>6.1%})")
    print(f"  Cold:  {ssm_cold.get('entry_count', 0):>3} layers (hit rate: {ssm_cold.get('hit_rate', 0.0):>6.1%})")

    return {
        'hit_rate': hit_rate,
        'throughput': throughput,
        'latency_ms': latency_ms,
        'elapsed': elapsed,
        'total_accesses': total_accesses
    }


def main():
    print("="*70)
    print("SSM Cache Performance Benchmark")
    print("="*70)

    num_layers = 30

    # Define access pattern (realistic: hot layers accessed more)
    hot_layers = [(i, 100) for i in range(0, 5)]      # 5 hot layers, 100 accesses each
    warm_layers = [(i, 20) for i in range(5, 15)]     # 10 warm layers, 20 accesses each
    cold_layers = [(i, 5) for i in range(15, 30)]     # 15 cold layers, 5 accesses each

    access_pattern = hot_layers + warm_layers + cold_layers
    total_accesses = sum(count for _, count in access_pattern)

    print(f"\nAccess pattern:")
    print(f"  Hot layers (0-4):    5 layers × 100 accesses = 500")
    print(f"  Warm layers (5-14):  10 layers × 20 accesses = 200")
    print(f"  Cold layers (15-29): 15 layers × 5 accesses = 75")
    print(f"  Total: {total_accesses} accesses")

    # Scenario 1: Large budget (100% hit rate)
    result_100 = benchmark_scenario(
        name="100% Hit Rate (Large Budget)",
        budget_kb=10 * 1024,  # 10MB (can hold all 30 layers @ 100KB each)
        num_layers=num_layers,
        access_pattern=access_pattern
    )

    # Scenario 2: Medium budget (partial hit rate)
    result_partial = benchmark_scenario(
        name="Partial Hit Rate (Medium Budget)",
        budget_kb=2 * 1024,   # 2MB (can hold ~15 layers)
        num_layers=num_layers,
        access_pattern=access_pattern
    )

    # Scenario 3: Small budget (low hit rate)
    result_low = benchmark_scenario(
        name="Low Hit Rate (Small Budget)",
        budget_kb=512,    # 512KB (can hold ~5 layers)
        num_layers=num_layers,
        access_pattern=access_pattern
    )

    # Summary comparison
    print(f"\n{'='*70}")
    print("Performance Comparison")
    print(f"{'='*70}")
    print(f"\n{'Scenario':<30} {'Hit Rate':<12} {'Throughput':<15} {'Latency (ms)':<15}")
    print("-"*70)

    scenarios = [
        ("100% Hit Rate", result_100),
        ("Partial Hit Rate", result_partial),
        ("Low Hit Rate", result_low)
    ]

    for name, result in scenarios:
        print(f"{name:<30} {result['hit_rate']:>10.1%}  {result['throughput']:>12.0f}/s  {result['latency_ms']:>12.3f} ms")

    # Calculate performance impact
    print(f"\n{'='*70}")
    print("Performance Impact Analysis")
    print(f"{'='*70}")

    baseline = result_100
    for name, result in scenarios[1:]:
        throughput_impact = ((result['throughput'] - baseline['throughput']) / baseline['throughput']) * 100
        latency_impact = ((result['latency_ms'] - baseline['latency_ms']) / baseline['latency_ms']) * 100

        print(f"\n{name} vs 100% Hit Rate:")
        print(f"  Hit rate:         {result['hit_rate']:.1%} vs {baseline['hit_rate']:.1%}")
        print(f"  Throughput impact: {throughput_impact:+.1f}%")
        print(f"  Latency impact:    {latency_impact:+.1f}%")

    print(f"\n{'='*70}")
    print("✓ Benchmark Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
