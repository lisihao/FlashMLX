#!/usr/bin/env python3
"""
Test SSM Cache Hit Rate

This test simulates cross-layer access patterns to verify that
SSM cache Hot/Warm/Cold mechanism works correctly.

Scenario:
- Create 30 SSM layers
- Simulate repeated access to different layers
- Hot layers: accessed frequently (should stay in Hot tier)
- Warm layers: accessed occasionally (should migrate to Warm)
- Cold layers: accessed rarely (should migrate to Cold)

Expected results:
- Hot tier hit rate: > 80%
- Overall hit rate: > 50%
"""

import mlx.core as mx
import numpy as np
from flashmlx.cache import (
    HybridCacheManager,
    HybridCacheConfig,
    LayerType
)
from flashmlx.cache.per_layer_ssm_cache import PerLayerSSMCache


def create_dummy_ssm_state(layer_idx: int, state_size: int = 128):
    """Create dummy SSM state for testing."""
    # Use layer_idx as seed for reproducibility
    np.random.seed(layer_idx)
    state = mx.array(np.random.randn(state_size).astype(np.float32))
    return state


def test_ssm_cache_hit_rate():
    print("="*70)
    print("SSM Cache Hit Rate Test")
    print("="*70)

    # Create layer types (30 SSM layers)
    num_ssm_layers = 30
    layer_types = {i: LayerType.SSM for i in range(num_ssm_layers)}

    # Create hybrid cache manager
    config = HybridCacheConfig(
        total_budget_bytes=128 * 1024 * 1024,  # 128MB
        compression_ratio=4.0,
        beta_calibration=True
    )
    manager = HybridCacheManager(config=config, layer_types=layer_types)

    # Create per-layer caches
    caches = []
    for layer_idx in range(num_ssm_layers):
        cache = PerLayerSSMCache(
            manager=manager,
            layer_idx=layer_idx,
            size=2
        )
        # Enable managed cache mode for testing
        cache.enable_managed_cache()
        caches.append(cache)

    print(f"\n✓ Created {num_ssm_layers} SSM layer caches")
    print(f"✓ Enabled managed cache mode\n")

    # Define access patterns
    # Hot layers: 0-9 (accessed frequently)
    # Warm layers: 10-19 (accessed occasionally)
    # Cold layers: 20-29 (accessed rarely)
    hot_layers = list(range(0, 10))
    warm_layers = list(range(10, 20))
    cold_layers = list(range(20, 30))

    print("Access pattern:")
    print(f"  Hot layers (0-9):    accessed 100 times each")
    print(f"  Warm layers (10-19): accessed 20 times each")
    print(f"  Cold layers (20-29): accessed 5 times each")
    print()

    # Phase 1: Initial population (first access - all misses)
    print("Phase 1: Initial population (first access)...")
    for layer_idx in range(num_ssm_layers):
        state = create_dummy_ssm_state(layer_idx)
        caches[layer_idx][0] = state  # Write (store to managed cache)

    # Phase 2: Repeated access (test hit rate)
    print("Phase 2: Repeated access (testing hit rate)...\n")

    # Hot layers: 100 accesses each
    for _ in range(100):
        for layer_idx in hot_layers:
            _ = caches[layer_idx][0]  # Read (should hit)

    # Warm layers: 20 accesses each
    for _ in range(20):
        for layer_idx in warm_layers:
            _ = caches[layer_idx][0]  # Read (should hit)

    # Cold layers: 5 accesses each
    for _ in range(5):
        for layer_idx in cold_layers:
            _ = caches[layer_idx][0]  # Read (may miss due to eviction)

    # Get statistics
    stats = manager.get_statistics()

    # Extract SSM statistics
    ssm_hot = stats['ssm']['hot']
    ssm_warm = stats['ssm']['warm']
    ssm_cold = stats['ssm']['cold']

    # Calculate overall hit rate
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
    print(f"  Accesses:  {hot_accesses}")
    print(f"  Hits:      {ssm_hot.get('total_hits', 0)}")
    print(f"  Misses:    {ssm_hot.get('total_misses', 0)}")
    print(f"  Hit rate:  {ssm_hot.get('hit_rate', 0.0):.1%}")

    print(f"\nWarm Tier:")
    print(f"  Accesses:  {warm_accesses}")
    print(f"  Hits:      {ssm_warm.get('total_hits', 0)}")
    print(f"  Misses:    {ssm_warm.get('total_misses', 0)}")
    print(f"  Hit rate:  {ssm_warm.get('hit_rate', 0.0):.1%}")

    print(f"\nCold Tier:")
    print(f"  Accesses:  {cold_accesses}")
    print(f"  Hits:      {ssm_cold.get('total_hits', 0)}")
    print(f"  Misses:    {ssm_cold.get('total_misses', 0)}")
    print(f"  Hit rate:  {ssm_cold.get('hit_rate', 0.0):.1%}")

    print(f"\nOverall:")
    print(f"  Total accesses: {total_accesses}")
    print(f"  Total hits:     {total_hits}")
    print(f"  Total misses:   {total_misses}")
    print(f"  Hit rate:       {overall_hit_rate:.1%}")

    # Verification
    print(f"\n{'='*70}")
    print("Verification")
    print(f"{'='*70}")

    # Expected: total accesses = 30 (initial) + 1000 (hot) + 200 (warm) + 50 (cold)
    expected_accesses = 30 + 1000 + 200 + 50
    print(f"Expected accesses: {expected_accesses}")
    print(f"Actual accesses:   {total_accesses}")

    if abs(total_accesses - expected_accesses) < 10:
        print("✓ Access count matches")
    else:
        print("⚠️  Access count mismatch")

    # Expected: hit rate > 50% (after initial misses)
    # Initial 30 misses, then 1250 accesses (most should hit)
    # Hit rate should be around (1250 - evictions) / 1280 ≈ 95%+
    if overall_hit_rate > 0.5:
        print(f"✓ Hit rate > 50% ({overall_hit_rate:.1%})")
    else:
        print(f"⚠️  Hit rate too low ({overall_hit_rate:.1%})")

    # Print tier distribution
    print(f"\nTier Distribution:")
    print(f"  Hot tier size:  {ssm_hot['entry_count']} layers")
    print(f"  Warm tier size: {ssm_warm['entry_count']} layers")
    print(f"  Cold tier size: {ssm_cold['entry_count']} layers")

    print(f"\n{'='*70}")
    print("✓ Test Complete!")
    print(f"{'='*70}")

    return overall_hit_rate


if __name__ == "__main__":
    hit_rate = test_ssm_cache_hit_rate()

    print(f"\nFinal Hit Rate: {hit_rate:.1%}")

    if hit_rate > 0.8:
        print("🎉 Excellent! Hit rate > 80%")
    elif hit_rate > 0.5:
        print("✅ Good! Hit rate > 50%")
    else:
        print("⚠️  Needs improvement. Hit rate < 50%")
