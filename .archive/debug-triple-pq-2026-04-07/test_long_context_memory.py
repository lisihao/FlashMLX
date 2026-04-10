#!/usr/bin/env python3
"""
Test memory savings at long context (32K tokens).

At 32K tokens, KV cache should dominate memory usage.
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def test_config(model, tokenizer, tokens, cache_kwargs, name):
    """Test memory usage for a configuration."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    # Clean state
    mx.clear_cache()
    mx.reset_peak_memory()

    # Create cache
    cache = make_prompt_cache(model, **cache_kwargs)

    # Forward pass (prefill)
    logits = model(tokens, cache=cache)
    mx.eval(logits)

    # Add 1 generation token to trigger flat mode transition
    # (This is when quantization actually happens)
    next_token = mx.argmax(logits[:, -1, :], keepdims=True)
    _ = model(next_token, cache=cache)
    mx.eval(_)

    # Measure
    peak_mb = mx.get_peak_memory() / (1024**2)
    print(f"  Peak Memory: {peak_mb:.1f} MB")

    # Cleanup
    del cache
    del _
    mx.clear_cache()

    return peak_mb


def main():
    MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)

    # Generate 32K tokens
    print("\nGenerating 32K token context...")
    text = "The quick brown fox jumps over the lazy dog. " * 4000
    tokens_list = tokenizer.encode(text)[:32768]
    tokens = mx.array([tokens_list])
    print(f"Context length: {len(tokens_list)} tokens")

    results = {}

    # Test 1: Standard (no compression)
    results['standard'] = test_config(
        model, tokenizer, tokens, {},
        "Standard (no compression)"
    )

    # Test 2: PolarQuant 4-bit
    results['pq4'] = test_config(
        model, tokenizer, tokens,
        {"kv_cache": "triple_pq", "kv_warm_quantizer": "polarquant", "kv_warm_bits": 4},
        "PolarQuant 4-bit"
    )

    # Test 3: PolarQuant 3-bit
    results['pq3'] = test_config(
        model, tokenizer, tokens,
        {"kv_cache": "triple_pq", "kv_warm_quantizer": "polarquant", "kv_warm_bits": 3},
        "PolarQuant 3-bit"
    )

    # Test 4: PolarQuant 2-bit
    results['pq2'] = test_config(
        model, tokenizer, tokens,
        {"kv_cache": "triple_pq", "kv_warm_quantizer": "polarquant", "kv_warm_bits": 2},
        "PolarQuant 2-bit"
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (32K tokens)")
    print(f"{'='*60}\n")

    print(f"{'Config':<25} {'Memory (MB)':>12} {'vs Standard':>15}")
    print("-" * 60)

    baseline = results['standard']

    for name, mem in results.items():
        savings = baseline - mem
        pct = (savings / baseline) * 100
        display_name = {
            'standard': 'Standard',
            'pq4': 'PolarQuant 4-bit',
            'pq3': 'PolarQuant 3-bit',
            'pq2': 'PolarQuant 2-bit',
        }[name]

        print(f"{display_name:<25} {mem:>12.1f} {savings:>9.1f} MB ({pct:+.1f}%)")

    print("\nTheoretical KV cache size (32K tokens, bf16):")
    # 32768 tokens × 32 heads × 128 dim × 36 layers × 2 (K+V) × 2 bytes
    kv_size_bytes = 32768 * 32 * 128 * 36 * 2 * 2
    kv_size_mb = kv_size_bytes / (1024**2)
    print(f"  {kv_size_mb:.1f} MB")

    print(f"\nExpected savings with 4-bit compression:")
    print(f"  {kv_size_mb * 0.75:.1f} MB (75% of KV cache)")

    print(f"\nActual savings:")
    print(f"  {baseline - results['pq4']:.1f} MB")

    if (baseline - results['pq4']) < kv_size_mb * 0.5:
        print("\n⚠️  WARNING: Actual savings much less than expected!")
        print("    Possible reasons:")
        print("    - MLX memory pool not releasing memory")
        print("    - Other buffers taking memory")
        print("    - Peak memory measurement timing issues")


if __name__ == "__main__":
    main()
