#!/usr/bin/env python3
"""
32K Context Comparison: PolarQuant vs TurboAngle

Tests real memory usage and speed at long context.
This is where compression ratios matter!
"""

import sys
import os
sys.path.insert(0, "mlx-lm-source")

import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.turboangle import TurboAngleQuantizer


def generate_long_text(tokenizer, target_tokens):
    """Generate text that reaches target token count."""
    base_text = """
    The Tower of London is a historic castle located on the north bank of the River Thames
    in central London. It was founded towards the end of 1066 as part of the Norman Conquest
    of England. The White Tower, which gives the entire castle its name, was built by William
    the Conqueror in 1078 and was a resented symbol of oppression, inflicted upon London by
    the new ruling elite. The castle was used as a prison from 1100 until 1952, although that
    was not its primary purpose. A grand palace early in its history, it served as a royal
    residence. As a whole, the Tower is a complex of several buildings set within two concentric
    rings of defensive walls and a moat. There have been several phases of expansion, mainly
    under Kings Richard I, Henry III, and Edward I in the 12th and 13th centuries.

    The Tower of London has played a prominent role in English history. It was besieged several
    times, and controlling it has been important to controlling the country. The Tower has served
    variously as an armoury, a treasury, a menagerie, the home of the Royal Mint, a public record
    office, and the home of the Crown Jewels of England. From the early 14th century until the
    reign of Charles II, a procession would be led from the Tower to Westminster Abbey on the
    coronation of a monarch. In the absence of the monarch, the Constable of the Tower is in
    charge of the castle.
    """

    # Repeat until we reach target
    repetitions = (target_tokens // 200) + 1  # ~200 tokens per repetition
    text = base_text * repetitions

    # Tokenize and truncate to exact target
    tokens = tokenizer.encode(text)[:target_tokens]

    return tokens


def benchmark_at_context_length(model, tokenizer, num_layers, context_length):
    """Benchmark all methods at a specific context length."""
    print("\n")
    print("=" * 80)
    print(f"Testing at {context_length:,} tokens context")
    print("=" * 80)
    print()

    # Generate tokens
    print(f"Generating {context_length:,} tokens...")
    tokens = generate_long_text(tokenizer, context_length)
    tokens_mx = mx.array([tokens])
    print(f"✅ Generated {len(tokens):,} tokens")
    print()

    results = []

    # Test configurations
    configs = [
        ("Standard (no compression)", {}),

        ("PolarQuant 4-bit", {
            "kv_cache": "triple_pq",
            "kv_warm_bits": 4,
        }),

        ("TurboAngle Baseline", {
            "kv_cache": "triple_pq",
            "kv_layer_quantizers": {
                i: TurboAngleQuantizer(n_k=128, n_v=64, head_dim=128)
                for i in range(num_layers)
            },
        }),
    ]

    for name, cache_kwargs in configs:
        print(f"Testing: {name}")

        try:
            # Clear memory
            mx.clear_cache()
            mx.reset_peak_memory()

            # Create cache
            cache = make_prompt_cache(model, **cache_kwargs)

            # Run forward pass
            start = time.perf_counter()
            logits = model(tokens_mx, cache=cache)
            mx.eval(logits)
            elapsed = time.perf_counter() - start

            # Get memory
            peak_mem = mx.get_peak_memory() / (1024**2)  # MB
            tok_per_sec = len(tokens) / elapsed

            print(f"  ✅ Peak Memory: {peak_mem:.1f} MB")
            print(f"     Speed:       {tok_per_sec:.1f} tok/s")
            print(f"     Time:        {elapsed:.2f}s")
            print()

            results.append({
                'name': name,
                'memory_mb': peak_mem,
                'tok_per_sec': tok_per_sec,
                'time_sec': elapsed,
            })

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            print()

    return results


def calculate_theoretical_kv_size(num_layers, n_heads, seq_len, head_dim):
    """Calculate theoretical KV cache size."""
    # Standard (bf16): 2 bytes per element
    standard_bytes = num_layers * 2 * n_heads * seq_len * head_dim * 2
    standard_mb = standard_bytes / (1024**2)

    # PolarQuant (4 bits): 0.5 bytes per element
    polar_bytes = num_layers * 2 * n_heads * seq_len * head_dim * 0.5
    polar_mb = polar_bytes / (1024**2)

    # TurboAngle (6.75 bits): 0.84375 bytes per element
    turbo_bytes = num_layers * 2 * n_heads * seq_len * head_dim * (6.75/8)
    turbo_mb = turbo_bytes / (1024**2)

    return {
        'standard': standard_mb,
        'polar': polar_mb,
        'turbo': turbo_mb,
    }


def main():
    """Run 32K context comparison."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 18 + "32K Context: PolarQuant vs TurboAngle" + " " * 22 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model not found: {MODEL_PATH}")
        return

    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    num_layers = len(model.model.layers)

    # Get model config
    attn = model.model.layers[0].self_attn
    n_heads = attn.n_heads
    q_proj_out = attn.q_proj.weight.shape[0]
    head_dim = q_proj_out // n_heads

    print(f"✅ Model loaded")
    print(f"   Layers: {num_layers}")
    print(f"   Heads: {n_heads}")
    print(f"   Head dim: {head_dim}")
    print()

    # Test at different context lengths
    context_lengths = [4096, 8192, 16384, 32768]

    all_results = {}

    for ctx_len in context_lengths:
        results = benchmark_at_context_length(model, tokenizer, num_layers, ctx_len)
        all_results[ctx_len] = results

    # Summary table
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 32 + "SUMMARY" + " " * 40 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    print("Memory Usage Comparison:")
    print()
    print(f"{'Context':<12} {'Standard':>12} {'PolarQuant':>12} {'TurboAngle':>12} {'Polar Savings':>15} {'Turbo Savings':>15}")
    print("-" * 100)

    for ctx_len, results in all_results.items():
        if len(results) >= 3:
            std_mem = results[0]['memory_mb']
            polar_mem = results[1]['memory_mb']
            turbo_mem = results[2]['memory_mb']

            polar_savings = std_mem - polar_mem
            turbo_savings = std_mem - turbo_mem

            print(f"{ctx_len:,} tok  {std_mem:>9.1f} MB {polar_mem:>9.1f} MB "
                  f"{turbo_mem:>9.1f} MB {polar_savings:>11.1f} MB {turbo_savings:>11.1f} MB")

    print()

    # Theoretical KV cache sizes
    print("Theoretical KV Cache Sizes (excluding model params):")
    print()
    print(f"{'Context':<12} {'Standard KV':>13} {'PolarQuant KV':>14} {'TurboAngle KV':>15} "
          f"{'Polar Saves':>12} {'Turbo Saves':>12}")
    print("-" * 100)

    for ctx_len in context_lengths:
        theoretical = calculate_theoretical_kv_size(num_layers, n_heads, ctx_len, head_dim)
        polar_saves = theoretical['standard'] - theoretical['polar']
        turbo_saves = theoretical['standard'] - theoretical['turbo']

        print(f"{ctx_len:,} tok  {theoretical['standard']:>10.1f} MB "
              f"{theoretical['polar']:>11.1f} MB {theoretical['turbo']:>12.1f} MB "
              f"{polar_saves:>9.1f} MB {turbo_saves:>9.1f} MB")

    print()

    # Speed comparison
    print("Speed Comparison:")
    print()
    print(f"{'Context':<12} {'Standard':>12} {'PolarQuant':>12} {'TurboAngle':>12} "
          f"{'Polar vs Std':>13} {'Turbo vs Std':>13}")
    print("-" * 90)

    for ctx_len, results in all_results.items():
        if len(results) >= 3:
            std_speed = results[0]['tok_per_sec']
            polar_speed = results[1]['tok_per_sec']
            turbo_speed = results[2]['tok_per_sec']

            polar_rel = (polar_speed / std_speed - 1) * 100
            turbo_rel = (turbo_speed / std_speed - 1) * 100

            print(f"{ctx_len:,} tok  {std_speed:>9.1f} t/s {polar_speed:>9.1f} t/s "
                  f"{turbo_speed:>9.1f} t/s {polar_rel:>11.1f}% {turbo_rel:>11.1f}%")

    print()

    # Final verdict
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()

    if 32768 in all_results and len(all_results[32768]) >= 3:
        results_32k = all_results[32768]
        std_mem = results_32k[0]['memory_mb']
        polar_mem = results_32k[1]['memory_mb']
        turbo_mem = results_32k[2]['memory_mb']

        polar_savings = std_mem - polar_mem
        turbo_savings = std_mem - turbo_mem

        print(f"At 32K context:")
        print(f"  PolarQuant saves:   {polar_savings:.1f} MB")
        print(f"  TurboAngle saves:   {turbo_savings:.1f} MB")
        print(f"  Difference:         {polar_savings - turbo_savings:.1f} MB (PolarQuant saves more)")
        print()

        if polar_savings > turbo_savings * 1.2:  # 20% more
            print("  → **PolarQuant is clearly superior** (significantly more memory savings)")
        elif polar_savings > turbo_savings:
            print("  → **PolarQuant is better** (more memory savings)")
        else:
            print("  → **TurboAngle is competitive** (comparable savings)")

    print()


if __name__ == "__main__":
    main()
