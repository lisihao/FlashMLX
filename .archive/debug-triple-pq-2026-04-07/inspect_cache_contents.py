#!/usr/bin/env python3
"""
Inspect cache contents to find memory leak.
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def inspect_cache(cache, name):
    """Inspect a cache to see what's taking up memory."""
    print(f"\n{name} Cache Inspection:")
    print(f"  Total layers: {len(cache)}")

    # Check first layer in detail
    c = cache[0]

    attrs_to_check = [
        '_recent_k', '_recent_v',
        '_warm_k', '_warm_v',
        '_cold_k', '_cold_v',
        'warm_quantizer',
    ]

    total_mb = 0

    for attr in attrs_to_check:
        if hasattr(c, attr):
            val = getattr(c, attr)
            if val is not None:
                if hasattr(val, 'shape') and hasattr(val, 'size'):
                    size_mb = val.size * val.itemsize / (1024**2)
                    total_mb += size_mb
                    print(f"    {attr}: shape={val.shape}, dtype={val.dtype}, size={size_mb:.1f} MB")
                else:
                    print(f"    {attr}: {type(val)}")

    print(f"  Layer 0 total: {total_mb:.1f} MB")
    print(f"  Estimated all layers: {total_mb * len(cache):.1f} MB")


def main():
    """Debug cache memory."""
    MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)
    print(f"Model loaded\n")

    # Generate 16K tokens
    text = "The quick brown fox jumps over the lazy dog. " * 2000
    tokens = tokenizer.encode(text)[:16384]
    tokens_mx = mx.array([tokens])
    print(f"Tokens: {len(tokens)}\n")

    # Test Standard
    print("="*80)
    print("Standard Cache")
    print("="*80)

    mx.clear_cache()
    cache_std = make_prompt_cache(model)
    _ = model(tokens_mx, cache=cache_std)
    mx.eval(_)

    inspect_cache(cache_std, "Standard")

    # Test PolarQuant
    print("\n" + "="*80)
    print("PolarQuant Cache")
    print("="*80)

    mx.clear_cache()
    cache_pq = make_prompt_cache(model, kv_cache="triple_pq", kv_warm_bits=4)
    _ = model(tokens_mx, cache=cache_pq)
    mx.eval(_)

    inspect_cache(cache_pq, "PolarQuant")


if __name__ == "__main__":
    main()
