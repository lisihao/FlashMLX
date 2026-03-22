#!/usr/bin/env python3
"""
Debug script to investigate why hybrid cache injection is not working.

Tests:
1. Verify injection mechanism
2. Check monkey patch
3. Verify layer routing
4. Test statistics collection
"""

import mlx.core as mx
from mlx_lm import load, generate
from flashmlx.cache import (
    inject_hybrid_cache_manager,
    get_cache_statistics,
    create_layer_types_from_model,
    HybridCacheConfig,
    LayerType
)
import json


def inspect_model_structure(model):
    """Inspect model layer structure."""
    print("\n" + "="*70)
    print("STEP 1: Model Structure Inspection")
    print("="*70)

    print(f"\nModel type: {type(model)}")
    print(f"Model has {len(model.layers)} layers")

    # Check first few layers
    for i in range(min(5, len(model.layers))):
        layer = model.layers[i]
        print(f"\nLayer {i}:")
        print(f"  Type: {type(layer).__name__}")

        # Check if it has self_attn
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            print(f"  Has self_attn: {type(attn).__name__}")

            # Check if self_attn has cache
            if hasattr(attn, 'cache'):
                cache = attn.cache
                print(f"  Original cache type: {type(cache)}")
                print(f"  Cache object: {cache}")
            else:
                print(f"  ⚠️  No cache attribute found!")

        # Check if it has mamba
        if hasattr(layer, 'mamba'):
            mamba = layer.mamba
            print(f"  Has mamba (SSM): {type(mamba).__name__}")


def test_layer_types_detection(model):
    """Test layer types detection."""
    print("\n" + "="*70)
    print("STEP 2: Layer Types Detection")
    print("="*70)

    layer_types = create_layer_types_from_model(model, attention_layer_pattern="every 4th")

    print(f"\nTotal layers: {len(layer_types)}")
    print(f"SSM layers: {sum(1 for t in layer_types.values() if t == LayerType.SSM)}")
    print(f"Attention layers: {sum(1 for t in layer_types.values() if t == LayerType.ATTENTION)}")

    print("\nLayer type mapping:")
    for i in sorted(layer_types.keys()):
        if i < 10 or i % 10 == 0:  # Print first 10 and every 10th
            print(f"  Layer {i}: {layer_types[i]}")

    return layer_types


def test_injection_with_logging(model, layer_types):
    """Test injection with detailed logging."""
    print("\n" + "="*70)
    print("STEP 3: Hybrid Cache Injection")
    print("="*70)

    config = HybridCacheConfig(
        total_budget_bytes=64 * 1024 * 1024,
        compression_ratio=4.0,
        beta_calibration=True
    )

    print(f"\nConfig:")
    print(f"  Budget: {config.total_budget_bytes / (1024**2):.0f} MB")
    print(f"  Compression: {config.compression_ratio}x")
    print(f"  Beta calibration: {config.beta_calibration}")

    print("\nInjecting hybrid cache manager...")

    # Add debugging to injection
    cache_list = inject_hybrid_cache_manager(
        model=model,
        config=config,
        layer_types=layer_types,
        auto_inject=True
    )

    print(f"\n✓ Injection returned: {type(cache_list).__name__} with {len(cache_list)} elements")

    # Inspect first few cache objects
    print("\nInspecting cache objects:")
    for i in range(min(5, len(cache_list))):
        cache = cache_list[i]
        layer_type = layer_types.get(i, 'unknown')
        print(f"\nCache {i} (type: {layer_type}):")
        print(f"  Class: {type(cache).__name__}")
        print(f"  Module: {type(cache).__module__}")

        # Check if it's our cache
        if 'flashmlx' in type(cache).__module__:
            print(f"  ✓ FlashMLX cache object!")
        else:
            print(f"  ⚠️  Not a FlashMLX cache!")

    return cache_list


def test_cache_statistics(cache_list):
    """Test statistics collection."""
    print("\n" + "="*70)
    print("STEP 4: Statistics Collection Test")
    print("="*70)

    stats = get_cache_statistics(cache_list)

    print("\nRaw statistics structure:")
    print(json.dumps(stats, indent=2, default=str))

    # Check SSM stats
    print("\nSSM statistics:")
    if 'ssm' in stats:
        ssm_stats = stats['ssm']
        print(f"  Keys: {list(ssm_stats.keys())}")

        # Check tier statistics
        for tier in ['hot', 'warm', 'cold']:
            if tier in ssm_stats:
                tier_stats = ssm_stats[tier]
                print(f"  {tier.capitalize()} tier:")
                print(f"    Total accesses: {tier_stats.get('total_accesses', 0)}")
                print(f"    Hits: {tier_stats.get('hits', 0)}")
    else:
        print(f"  ⚠️  No 'ssm' key in statistics!")

    # Check Attention stats
    print("\nAttention statistics:")
    if 'attention' in stats:
        att_stats = stats['attention']
        print(f"  Keys: {list(att_stats.keys())}")
        print(f"  Avg compression: {att_stats.get('avg_compression_ratio', 'NOT FOUND')}")
    else:
        print(f"  ⚠️  No 'attention' key in statistics!")


def test_actual_generation(model, tokenizer, cache_list):
    """Test actual generation and verify cache usage."""
    print("\n" + "="*70)
    print("STEP 5: Actual Generation Test")
    print("="*70)

    # Create a simple prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'hello' and nothing else."}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"\nPrompt length: {len(tokenizer.encode(prompt))} tokens")
    print("\nGenerating response...")

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=10,
        verbose=False
    )

    print(f"Response: {response}")

    # Check statistics after generation
    print("\nStatistics after generation:")
    stats = get_cache_statistics(cache_list)

    # SSM stats (aggregate from all tiers)
    ssm_stats = stats.get('ssm', {})
    total_ssm_accesses = 0
    total_ssm_hits = 0

    for tier in ['hot', 'warm', 'cold']:
        if tier in ssm_stats:
            tier_stats = ssm_stats[tier]
            total_ssm_accesses += tier_stats.get('total_accesses', 0)
            total_ssm_hits += tier_stats.get('hits', 0)

    print(f"\nSSM (all tiers):")
    print(f"  Accesses: {total_ssm_accesses}")
    print(f"  Hits: {total_ssm_hits}")
    if total_ssm_accesses > 0:
        print(f"  Hit rate: {total_ssm_hits / total_ssm_accesses:.2%}")

    # Attention stats
    att_stats = stats.get('attention', {})
    print(f"\nAttention:")
    print(f"  Total compressions: {att_stats.get('total_compressions', 0)}")
    print(f"  Avg compression: {att_stats.get('avg_compression_ratio', 0.0):.2f}x")

    # Per-layer compression
    per_layer = stats.get('per_layer_attention_compression', [])
    if per_layer:
        print(f"\nPer-layer Attention compression:")
        for layer_stats in per_layer[:5]:  # Show first 5
            layer_idx = layer_stats.get('layer_idx', '?')
            compressions = layer_stats.get('total_compressions', 0)
            ratio = layer_stats.get('avg_compression_ratio', 0.0)
            print(f"  Layer {layer_idx}: {compressions} compressions, {ratio:.2f}x avg")


def test_cache_update_method(model, layer_types):
    """Test if cache update method is being called."""
    print("\n" + "="*70)
    print("STEP 6: Cache Update Method Test")
    print("="*70)

    # Check a few attention layers
    attention_indices = [i for i, t in enumerate(layer_types) if t == 'attention']

    if attention_indices:
        idx = attention_indices[0]
        layer = model.layers[idx]

        print(f"\nChecking Attention layer {idx}:")

        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            cache = attn.cache

            print(f"  Cache type: {type(cache).__name__}")
            print(f"  Cache methods: {[m for m in dir(cache) if not m.startswith('_')]}")

            # Check if it has update method
            if hasattr(cache, 'update'):
                print(f"  ✓ Has 'update' method")
                print(f"  Update signature: {cache.update.__doc__ if cache.update.__doc__ else 'No docstring'}")
            else:
                print(f"  ⚠️  No 'update' method found!")


def main():
    print("="*70)
    print("FlashMLX Hybrid Cache Injection Debug Script")
    print("="*70)

    # Load model
    print("\nLoading model...")
    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx/"
    model, tokenizer = load(model_path)
    print("✓ Model loaded")

    # Step 1: Inspect model structure
    inspect_model_structure(model)

    # Step 2: Test layer types detection
    layer_types = test_layer_types_detection(model)

    # Step 3: Test injection with logging
    cache_list = test_injection_with_logging(model, layer_types)

    # Step 4: Test statistics collection
    test_cache_statistics(cache_list)

    # Step 5: Test actual generation
    test_actual_generation(model, tokenizer, cache_list)

    # Step 6: Test cache update method
    test_cache_update_method(model, layer_types)

    print("\n" + "="*70)
    print("Debug Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
