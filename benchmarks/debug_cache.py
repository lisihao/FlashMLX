#!/usr/bin/env python3
"""
Debug cache structure
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache


def main():
    print("Loading model...")
    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
    model, tokenizer = load(model_path)
    print(f"✅ Model loaded: {len(model.layers)} layers")
    print()

    # Create cache
    cache = ArraysCache(len(model.layers))

    # Test prompt
    prompt = "Hello, how are you?"
    tokens = mx.array(tokenizer.encode(prompt))
    print(f"Prompt: '{prompt}'")
    print(f"Tokens shape: {tokens.shape}")
    print()

    # Prefill
    print("Running prefill...")
    logits = model(tokens[None], cache=cache)
    mx.eval(logits, cache)
    print(f"✅ Prefill complete")
    print(f"Logits shape: {logits.shape}")
    print()

    # Decode one token
    print("Decoding one token...")
    next_token = mx.argmax(logits[0, -1, :], keepdims=True)
    logits = model(next_token[None], cache=cache)
    mx.eval(logits, cache)
    print(f"✅ Decode complete")
    print(f"Logits shape: {logits.shape}")
    print()

    # Inspect cache for each layer
    print("=" * 60)
    print("Cache Inspection")
    print("=" * 60)
    print()

    for i in range(min(10, len(model.layers))):
        layer = model.layers[i]
        cache_entry = cache[i]

        print(f"Layer {i}:")
        print(f"  Type: {type(layer).__name__}")

        # Check layer attributes
        has_linear_attn = hasattr(layer, 'linear_attn')
        has_mamba_block = hasattr(layer, 'mamba_block')
        has_self_attn = hasattr(layer, 'self_attn')

        print(f"  Attributes: linear_attn={has_linear_attn}, mamba_block={has_mamba_block}, self_attn={has_self_attn}")

        # Check cache
        if cache_entry is None:
            print(f"  Cache: None")
        elif isinstance(cache_entry, list):
            print(f"  Cache: list of {len(cache_entry)} items")
            for j, item in enumerate(cache_entry):
                if item is None:
                    print(f"    [{j}]: None")
                elif isinstance(item, mx.array):
                    print(f"    [{j}]: array {item.shape} {item.dtype}")
                else:
                    print(f"    [{j}]: {type(item)}")
        elif isinstance(cache_entry, mx.array):
            print(f"  Cache: array {cache_entry.shape} {cache_entry.dtype}")
        else:
            print(f"  Cache: {type(cache_entry)}")

        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
