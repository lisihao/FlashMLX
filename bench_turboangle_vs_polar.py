#!/usr/bin/env python3
"""
Direct Comparison: TurboAngle vs PolarQuant

Head-to-head comparison on:
- Perplexity (quality)
- Memory usage
- Compression ratio (actual)
- Speed
"""

import sys
import os
sys.path.insert(0, "mlx-lm-source")

import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.turboangle import TurboAngleQuantizer
import numpy as np


def load_test_text():
    """Load test text."""
    text = """
    The Tower of London is a historic castle located on the north bank of the River Thames
    in central London. It was founded towards the end of 1066 as part of the Norman Conquest
    of England. The White Tower, which gives the entire castle its name, was built by William
    the Conqueror in 1078 and was a resented symbol of oppression, inflicted upon London by
    the new ruling elite. The castle was used as a prison from 1100 until 1952, although that
    was not its primary purpose. A grand palace early in its history, it served as a royal
    residence. As a whole, the Tower is a complex of several buildings set within two concentric
    rings of defensive walls and a moat.
    """ * 5
    return text


def compute_perplexity(model, tokenizer, text, cache=None):
    """Compute perplexity."""
    tokens = tokenizer.encode(text)
    tokens_mx = mx.array([tokens])

    logits = model(tokens_mx, cache=cache)

    # Shift for next-token prediction
    logits_shifted = logits[:, :-1, :]
    targets = mx.array(tokens[1:])

    # Compute log probs
    probs = mx.softmax(logits_shifted, axis=-1)
    log_probs = mx.log(probs + 1e-10)

    # Gather target log probs
    target_log_probs = []
    for i, target in enumerate(targets.tolist()):
        target_log_probs.append(log_probs[0, i, target].item())

    mean_log_prob = np.mean(target_log_probs)
    ppl = np.exp(-mean_log_prob)

    return ppl


def benchmark_method(model, tokenizer, text, name, cache_kwargs):
    """Benchmark a single method."""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print('='*80)

    # Create cache
    cache = make_prompt_cache(model, **cache_kwargs)

    tokens = tokenizer.encode(text)
    tokens_mx = mx.array([tokens])
    print(f"  Tokens: {len(tokens)}")

    # Measure memory
    mx.clear_cache()
    mx.reset_peak_memory()

    # Warmup
    _ = model(tokens_mx[:, :100], cache=cache)
    mx.eval(_)

    # Reset
    mx.clear_cache()
    mx.reset_peak_memory()
    cache = make_prompt_cache(model, **cache_kwargs)

    # Measure speed
    start = time.perf_counter()
    logits = model(tokens_mx, cache=cache)
    mx.eval(logits)
    elapsed = time.perf_counter() - start

    peak_mem = mx.get_peak_memory() / (1024**2)
    tok_per_sec = len(tokens) / elapsed

    # Measure perplexity
    cache = make_prompt_cache(model, **cache_kwargs)
    ppl = compute_perplexity(model, tokenizer, text, cache=cache)

    print(f"\n  Results:")
    print(f"    Perplexity:  {ppl:.6f}")
    print(f"    Peak Memory: {peak_mem:.1f} MB")
    print(f"    Speed:       {tok_per_sec:.1f} tok/s")
    print(f"    Time:        {elapsed:.2f}s")

    return {
        'name': name,
        'ppl': ppl,
        'memory_mb': peak_mem,
        'tok_per_sec': tok_per_sec,
        'time_sec': elapsed,
    }


def main():
    """Run head-to-head comparison."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 22 + "TurboAngle vs PolarQuant" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model not found: {MODEL_PATH}")
        return

    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    num_layers = len(model.model.layers)
    print(f"✅ Model loaded: {num_layers} layers")
    print()

    text = load_test_text()

    # Test configurations
    configs = [
        ("Standard (no compression)", {}),

        ("PolarQuant 4-bit", {
            "kv_cache": "triple_pq",
            "kv_warm_bits": 4,
        }),

        ("TurboAngle Baseline (K128V64)", {
            "kv_cache": "triple_pq",
            "kv_layer_quantizers": {
                i: TurboAngleQuantizer(n_k=128, n_v=64, head_dim=128)
                for i in range(num_layers)
            },
        }),

        ("TurboAngle E4 (K256V128)", {
            "kv_cache": "triple_pq",
            "kv_layer_quantizers": "mistral-7b",
        }),
    ]

    results = []
    for name, kwargs in configs:
        try:
            result = benchmark_method(model, tokenizer, text, name, kwargs)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 30 + "HEAD-TO-HEAD" + " " * 36 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    if not results:
        print("No results")
        return

    baseline_ppl = results[0]['ppl']
    baseline_speed = results[0]['tok_per_sec']

    print(f"{'Method':<35} {'PPL':>10} {'ΔPPL':>10} {'Memory':>10} {'Speed':>10} {'Rel Speed':>10}")
    print("-" * 95)

    for r in results:
        delta_ppl = r['ppl'] - baseline_ppl
        rel_speed = (r['tok_per_sec'] / baseline_speed) * 100

        print(
            f"{r['name']:<35} "
            f"{r['ppl']:>10.6f} "
            f"{delta_ppl:>10.6f} "
            f"{r['memory_mb']:>9.1f}M "
            f"{r['tok_per_sec']:>9.1f} "
            f"{rel_speed:>9.1f}%"
        )

    print()

    # Analysis
    print("Key Findings:")
    print()

    if len(results) >= 3:
        polar = results[1]
        turbo = results[2]

        print(f"PolarQuant vs TurboAngle Baseline:")
        print(f"  Quality:     ΔPPL = {polar['ppl'] - baseline_ppl:.6f} vs {turbo['ppl'] - baseline_ppl:.6f}")
        print(f"  Speed:       {polar['tok_per_sec']:.1f} vs {turbo['tok_per_sec']:.1f} tok/s "
              f"({((polar['tok_per_sec']/turbo['tok_per_sec']-1)*100):+.1f}%)")
        print(f"  Memory:      {polar['memory_mb']:.1f} vs {turbo['memory_mb']:.1f} MB")
        print()

        # Determine winner
        polar_delta = abs(polar['ppl'] - baseline_ppl)
        turbo_delta = abs(turbo['ppl'] - baseline_ppl)

        if polar_delta < 0.001 and turbo_delta < 0.001:
            print("  ✅ Both methods achieve near-zero perplexity loss")
            if polar['tok_per_sec'] > turbo['tok_per_sec']:
                print(f"  → PolarQuant is {((polar['tok_per_sec']/turbo['tok_per_sec']-1)*100):.1f}% faster")
                print("  → **PolarQuant is the winner for most use cases**")
            else:
                print("  → Comparable performance")
        elif turbo_delta < polar_delta:
            print(f"  ✅ TurboAngle has {polar_delta - turbo_delta:.6f} better quality")
            print("  → TurboAngle wins on quality")
        else:
            print(f"  ✅ PolarQuant has {turbo_delta - polar_delta:.6f} better quality")
            print("  → PolarQuant wins on quality")

    print()

    # Theoretical compression ratios
    print("Theoretical Compression Ratios:")
    print("  PolarQuant 4-bit:     4.0×  (4 bits vs bf16 16 bits)")
    print("  TurboAngle Baseline:  2.37× (6.75 bits vs bf16 16 bits)")
    print("  TurboAngle E4:        2.0×  (8.0 bits vs bf16 16 bits)")
    print()


if __name__ == "__main__":
    main()
