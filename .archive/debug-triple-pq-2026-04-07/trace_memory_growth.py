#!/usr/bin/env python3
"""
Trace memory growth during forward pass.
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def trace_forward_pass(model, tokens_mx, cache_kwargs, name):
    """Trace memory usage during forward pass."""
    print(f"\n{'='*80}")
    print(f"Tracing: {name}")
    print(f"{'='*80}")

    mx.clear_cache()
    mx.reset_peak_memory()

    # Create cache
    cache = make_prompt_cache(model, **cache_kwargs)
    mem_after_cache = mx.get_peak_memory() / (1024**2)
    print(f"  After cache creation: {mem_after_cache:.1f} MB")

    # Process in chunks to see where memory grows
    chunk_size = 4096
    num_tokens = tokens_mx.shape[1]

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        chunk = tokens_mx[:, start:end]

        # Forward pass on chunk
        _ = model(chunk, cache=cache)
        mx.eval(_)

        mem = mx.get_peak_memory() / (1024**2)
        print(f"  After tokens {start:5d}-{end:5d}: {mem:.1f} MB")

    final_mem = mx.get_peak_memory() / (1024**2)
    print(f"\n  Final peak memory: {final_mem:.1f} MB")

    return final_mem


def main():
    """Trace memory growth."""
    MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)
    print("Model loaded\n")

    # Generate 16K tokens
    text = "The quick brown fox jumps over the lazy dog. " * 2000
    tokens = tokenizer.encode(text)[:16384]
    tokens_mx = mx.array([tokens])
    print(f"Tokens: {len(tokens)}\n")

    # Trace Standard
    mem_std = trace_forward_pass(model, tokens_mx, {}, "Standard")

    # Trace PolarQuant
    mem_pq = trace_forward_pass(model, tokens_mx, {
        "kv_cache": "triple_pq",
        "kv_warm_bits": 4,
    }, "PolarQuant")

    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Standard:   {mem_std:.1f} MB")
    print(f"  PolarQuant: {mem_pq:.1f} MB")
    print(f"  Difference: {mem_pq - mem_std:+.1f} MB")


if __name__ == "__main__":
    main()
