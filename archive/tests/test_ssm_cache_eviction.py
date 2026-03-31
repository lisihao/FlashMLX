#!/usr/bin/env python3
"""
Test SSM Cache Eviction and Tiering

This test forces cache eviction by:
1. Creating more layers than budget allows
2. Using different access patterns for Hot/Warm/Cold tiers
3. Verifying tiering mechanism works correctly

Expected results:
- Hot layers stay in Hot tier (high hit rate)
- Warm layers migrate to Warm tier
- Cold layers migrate to Cold tier or get evicted
- Overall hit rate > 70%
"""

import mlx.core as mx
import numpy as np
from flashmlx.cache import (
    HybridCacheManager,
    HybridCacheConfig,
    LayerType
)
from flashmlx.cache.per_layer_ssm_cache import PerLayerSSMCache


def create_dummy_ssm_state(layer_idx: int, state_size: int = 1024):
    """Create dummy SSM state (1KB each) for testing."""
    np.random.seed(layer_idx)
    state = mx.array(np.random.randn(state_size // 4).astype(np.float32))
    return state


def test_ssm_cache_eviction():
    print("="*70)
    print("SSM Cache Eviction and Tiering Test")
    print("="*70)

    # Create layer types (60 SSM layers, each ~1KB)
    num_ssm_layers = 60
    layer_types = {i: LayerType.SSM for i in range(num_ssm_layers)}

    # Create hybrid cache manager with LIMITED budget
    # Total 60 layers × 1KB = 60KB needed
    # Budget: 20KB (forces eviction)
    config = HybridCacheConfig(
        total_budget_bytes=20 * 1024,  # 20KB only (forces eviction!)
        compression_ratio=4.0,
        beta_calibration=True
    )
    manager = HybridCacheManager(config=config, layer_types=layer_types)

    # Get budget allocation
    ssm_ratio = len([t for t in layer_types.values() if t == LayerType.SSM]) / len(layer_types)
    total_budget = config.total_budget_bytes
    ssm_budget = int(total_budget * ssm_ratio)
    hot_budget = int(ssm_budget * 0.15)
    warm_budget = int(ssm_budget * 0.25)
    cold_budget = ssm_budget - hot_budget - warm_budget

    print(f"\nBudget Allocation:")
    print(f"  Total:  {config.total_budget_bytes / 1024:.1f} KB")
    print(f"  Hot:    {hot_budget / 1024:.1f} KB (can hold ~{hot_budget // 1024} layers)")
    print(f"  Warm:   {warm_budget / 1024:.1f} KB (can hold ~{warm_budget // 1024} layers)")
    print(f"  Cold:   {cold_budget / 1024:.1f} KB (can hold ~{cold_budget // 1024} layers)")

    # Create per-layer caches
    caches = []
    for layer_idx in range(num_ssm_layers):
        cache = PerLayerSSMCache(
            manager=manager,
            layer_idx=layer_idx,
            size=2
        )
        cache.enable_managed_cache()
        caches.append(cache)

    print(f"\n✓ Created {num_ssm_layers} SSM layer caches (each ~1KB)")
    print(f"✓ Total memory needed: {num_ssm_layers} KB")
    print(f"✓ Budget: {config.total_budget_bytes / 1024:.1f} KB")
    print(f"✓ Forced to evict: ~{num_ssm_layers - (config.total_budget_bytes // 1024)} layers\n")

    # Define access patterns
    hot_layers = list(range(0, 5))      # 5 layers (very frequent)
    warm_layers = list(range(5, 15))    # 10 layers (frequent)
    cold_layers = list(range(15, 30))   # 15 layers (occasional)
    rare_layers = list(range(30, 60))   # 30 layers (rare, likely evicted)

    print("Access pattern:")
    print(f"  Hot layers (0-4):     5 layers, accessed 200 times each")
    print(f"  Warm layers (5-14):   10 layers, accessed 50 times each")
    print(f"  Cold layers (15-29):  15 layers, accessed 10 times each")
    print(f"  Rare layers (30-59):  30 layers, accessed 2 times each")
    print()

    # Phase 1: Initial population
    print("Phase 1: Initial population...")
    for layer_idx in range(num_ssm_layers):
        state = create_dummy_ssm_state(layer_idx)
        caches[layer_idx][0] = state

    # Phase 2: Access with different patterns
    print("Phase 2: Repeated access with tiering...\n")

    # Hot layers: 200 accesses each
    for _ in range(200):
        for layer_idx in hot_layers:
            _ = caches[layer_idx][0]

    # Warm layers: 50 accesses each
    for _ in range(50):
        for layer_idx in warm_layers:
            _ = caches[layer_idx][0]

    # Cold layers: 10 accesses each
    for _ in range(10):
        for layer_idx in cold_layers:
            _ = caches[layer_idx][0]

    # Rare layers: 2 accesses each
    for _ in range(2):
        for layer_idx in rare_layers:
            _ = caches[layer_idx][0]

    # Get statistics
    stats = manager.get_statistics()
    ssm_hot = stats['ssm']['hot']
    ssm_warm = stats['ssm']['warm']
    ssm_cold = stats['ssm']['cold']

    # Calculate overall
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
    total_accesses = total_hits + total_misses
    overall_hit_rate = total_hits / total_accesses if total_accesses > 0 else 0.0

    # Print results
    print("="*70)
    print("Results")
    print("="*70)

    hot_accesses = ssm_hot.get('total_hits', 0) + ssm_hot.get('total_misses', 0)
    warm_accesses = ssm_warm.get('total_hits', 0) + ssm_warm.get('total_misses', 0)
    cold_accesses = ssm_cold.get('total_hits', 0) + ssm_cold.get('total_misses', 0)

    print(f"\nHot Tier:")
    print(f"  Accesses:   {hot_accesses:>6}")
    print(f"  Hits:       {ssm_hot.get('total_hits', 0):>6}")
    print(f"  Misses:     {ssm_hot.get('total_misses', 0):>6}")
    print(f"  Hit rate:   {ssm_hot.get('hit_rate', 0.0):>6.1%}")
    print(f"  Size:       {ssm_hot.get('entry_count', 0):>6} layers")
    print(f"  Evictions:  {ssm_hot.get('total_evictions', 0):>6}")

    print(f"\nWarm Tier:")
    print(f"  Accesses:   {warm_accesses:>6}")
    print(f"  Hits:       {ssm_warm.get('total_hits', 0):>6}")
    print(f"  Misses:     {ssm_warm.get('total_misses', 0):>6}")
    print(f"  Hit rate:   {ssm_warm.get('hit_rate', 0.0):>6.1%}")
    print(f"  Size:       {ssm_warm.get('entry_count', 0):>6} layers")
    print(f"  Promotions: {ssm_warm.get('total_promotions', 0):>6}")
    print(f"  Demotions:  {ssm_warm.get('total_demotions', 0):>6}")

    print(f"\nCold Tier:")
    print(f"  Accesses:   {cold_accesses:>6}")
    print(f"  Hits:       {ssm_cold.get('total_hits', 0):>6}")
    print(f"  Misses:     {ssm_cold.get('total_misses', 0):>6}")
    print(f"  Hit rate:   {ssm_cold.get('hit_rate', 0.0):>6.1%}")
    print(f"  Size:       {ssm_cold.get('entry_count', 0):>6} layers")
    print(f"  Revivals:   {ssm_cold.get('total_revivals', 0):>6}")

    print(f"\nOverall:")
    print(f"  Total accesses: {total_accesses:>6}")
    print(f"  Total hits:     {total_hits:>6}")
    print(f"  Total misses:   {total_misses:>6}")
    print(f"  Hit rate:       {overall_hit_rate:>6.1%}")

    # Verification
    print(f"\n{'='*70}")
    print("Verification")
    print(f"{'='*70}")

    # Expected: 1000 (hot) + 500 (warm) + 150 (cold) + 60 (rare) = 1710
    expected_accesses = 1000 + 500 + 150 + 60
    print(f"Expected accesses: {expected_accesses}")
    print(f"Actual accesses:   {total_accesses}")

    if abs(total_accesses - expected_accesses) < 50:
        print("✓ Access count matches")
    else:
        print("⚠️  Access count mismatch")

    # Expected: hit rate > 70% (frequent layers should stay cached)
    if overall_hit_rate > 0.7:
        print(f"✓ Hit rate > 70% ({overall_hit_rate:.1%})")
    elif overall_hit_rate > 0.5:
        print(f"⚠️  Hit rate moderate ({overall_hit_rate:.1%})")
    else:
        print(f"❌ Hit rate too low ({overall_hit_rate:.1%})")

    # Check tier distribution
    total_cached = (
        ssm_hot.get('entry_count', 0) +
        ssm_warm.get('entry_count', 0) +
        ssm_cold.get('entry_count', 0)
    )
    print(f"\nTier Distribution:")
    print(f"  Total cached: {total_cached} / {num_ssm_layers} layers ({total_cached/num_ssm_layers*100:.1f}%)")
    print(f"  Hot tier:     {ssm_hot.get('entry_count', 0)} layers")
    print(f"  Warm tier:    {ssm_warm.get('entry_count', 0)} layers")
    print(f"  Cold tier:    {ssm_cold.get('entry_count', 0)} layers")

    print(f"\n{'='*70}")
    print("✓ Test Complete!")
    print(f"{'='*70}")

    return overall_hit_rate, ssm_hot, ssm_warm, ssm_cold


if __name__ == "__main__":
    hit_rate, hot, warm, cold = test_ssm_cache_eviction()

    print(f"\n📊 Summary:")
    print(f"  Overall Hit Rate: {hit_rate:.1%}")
    print(f"  Hot tier hit rate: {hot.get('hit_rate', 0.0):.1%}")
    print(f"  Warm tier hit rate: {warm.get('hit_rate', 0.0):.1%}")
    print(f"  Cold tier hit rate: {cold.get('hit_rate', 0.0):.1%}")

    if hit_rate > 0.8:
        print("\n🎉 Excellent! Hot/Warm/Cold tiering works perfectly!")
    elif hit_rate > 0.6:
        print("\n✅ Good! Tiering mechanism is effective")
    else:
        print("\n⚠️  Needs improvement")
