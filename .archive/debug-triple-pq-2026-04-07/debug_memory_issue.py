#!/usr/bin/env python3
"""
Debug memory issue with PolarQuant at long context.
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def test_memory(model, tokens_mx, cache_kwargs, name):
    """Test memory usage."""
    print(f"\nTesting: {name}")
    print(f"  Tokens: {tokens_mx.shape[1]}")

    # Clear and reset
    mx.clear_cache()
    mx.reset_peak_memory()

    # Create cache
    cache = make_prompt_cache(model, **cache_kwargs)

    # Forward pass
    logits = model(tokens_mx, cache=cache)
    mx.eval(logits)

    # Measure
    peak_mem_mb = mx.get_peak_memory() / (1024**2)

    print(f"  Peak memory: {peak_mem_mb:.1f} MB")

    # Check cache contents
    if hasattr(cache[0], '_warm_k'):
        warm_k = cache[0]._warm_k
        if warm_k is not None:
            print(f"  Warm cache K shape: {warm_k.shape}")
            warm_k_size_mb = warm_k.size * warm_k.itemsize / (1024**2)
            print(f"  Warm cache K size: {warm_k_size_mb:.1f} MB")

    return peak_mem_mb


def main():
    """Debug memory."""
    MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)
    print(f"Model loaded: {len(model.model.layers)} layers\n")

    # Test at 16K
    print("="*80)
    print("16K Context Test")
    print("="*80)

    # Generate tokens
    text = "The quick brown fox jumps over the lazy dog. " * 2000
    tokens = tokenizer.encode(text)[:16384]
    tokens_mx = mx.array([tokens])

    print(f"Generated {len(tokens)} tokens")

    # Test standard
    mem_std = test_memory(model, tokens_mx, {}, "Standard")

    # Test PolarQuant
    mem_polar = test_memory(model, tokens_mx, {
        "kv_cache": "triple_pq",
        "kv_warm_bits": 4,
    }, "PolarQuant")

    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Standard:   {mem_std:.1f} MB")
    print(f"  PolarQuant: {mem_polar:.1f} MB")
    print(f"  Difference: {mem_polar - mem_std:+.1f} MB")

    if mem_polar > mem_std:
        print(f"  ⚠️  PolarQuant uses MORE memory than standard!")
        print(f"      This suggests a bug or memory leak.")
    else:
        print(f"  ✅ PolarQuant saves {mem_std - mem_polar:.1f} MB")


if __name__ == "__main__":
    main()
