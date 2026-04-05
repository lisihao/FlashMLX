#!/usr/bin/env python3
"""
Clean comparison: Standard vs PolarQuant vs TurboAngle

Carefully controlled test.
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.turboangle import TurboAngleQuantizer


def test_method(model, tokens_mx, cache_kwargs, name):
    """Test a single method with clean state."""
    print(f"\n{name}:")

    # Force complete cleanup
    mx.clear_cache()
    mx.metal.clear_cache()
    mx.reset_peak_memory()

    # Create cache
    cache = make_prompt_cache(model, **cache_kwargs)

    # Forward pass
    _ = model(tokens_mx, cache=cache)
    mx.eval(_)

    # Measure
    mem_mb = mx.get_peak_memory() / (1024**2)
    print(f"  Peak Memory: {mem_mb:.1f} MB")

    # Cleanup
    del cache
    del _
    mx.clear_cache()
    mx.metal.clear_cache()

    return mem_mb


def main():
    """Clean comparison."""
    MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)
    num_layers = len(model.model.layers)
    print(f"Model loaded: {num_layers} layers\n")

    # Test at 8K (more manageable)
    print("="*80)
    print("8K Context Comparison")
    print("="*80)

    text = "The quick brown fox jumps over the lazy dog. " * 1000
    tokens = tokenizer.encode(text)[:8192]
    tokens_mx = mx.array([tokens])
    print(f"Tokens: {len(tokens)}\n")

    results = []

    # Test 1: Standard
    mem_std = test_method(model, tokens_mx, {}, "Standard (no compression)")
    results.append(("Standard", mem_std))

    # Test 2: PolarQuant
    mem_pq = test_method(model, tokens_mx, {
        "kv_cache": "triple_pq",
        "kv_warm_bits": 4,
    }, "PolarQuant 4-bit")
    results.append(("PolarQuant", mem_pq))

    # Test 3: TurboAngle
    layer_quantizers = {
        i: TurboAngleQuantizer(n_k=128, n_v=64, head_dim=128)
        for i in range(num_layers)
    }
    mem_ta = test_method(model, tokens_mx, {
        "kv_cache": "triple_pq",
        "kv_layer_quantizers": layer_quantizers,
    }, "TurboAngle Baseline")
    results.append(("TurboAngle", mem_ta))

    # Summary
    print(f"\n{'='*80}")
    print("Summary (8K context):")
    print(f"{'='*80}\n")

    baseline = results[0][1]

    for name, mem in results:
        savings = baseline - mem
        print(f"{name:20s}: {mem:8.1f} MB  (savings: {savings:+7.1f} MB)")

    print(f"\nTheoretical KV cache (8K tokens):")
    print(f"  Standard:   4,608 MB")
    print(f"  PolarQuant: 1,152 MB  (4.0× compression)")
    print(f"  TurboAngle: 1,944 MB  (2.37× compression)")


if __name__ == "__main__":
    main()
