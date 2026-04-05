#!/usr/bin/env python3
"""
Analyze TurboAngle Compression Ratio

Compares theoretical vs actual compression ratios:
1. Theoretical bit usage per element
2. Actual memory usage at different context lengths
3. Breakdown: model params vs KV cache
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.turboangle import TurboAngleQuantizer


def analyze_theoretical_compression():
    """Calculate theoretical compression ratios."""
    print("=" * 80)
    print("Theoretical Compression Analysis")
    print("=" * 80)
    print()

    # TurboAngle Baseline (K128V64)
    print("TurboAngle Baseline (K128V64):")
    print("  K cache:")
    print("    - Angles: log2(128) = 7 bits per pair → 3.5 bits per element")
    print("    - Norms: 8 bits (linear quantization)")
    print("    - Total K: (3.5 + 8) / 2 = 5.75 bits per element")
    print()
    print("  V cache:")
    print("    - Angles: log2(64) = 6 bits per pair → 3.0 bits per element")
    print("    - Norms: 4 bits (log-space quantization)")
    print("    - Total V: (3.0 + 4) / 2 = 3.5 bits per element")
    print()
    print("  Overall average:")
    print("    - (5.75 + 3.5) / 2 = 4.625 bits per element")
    print("    - But norms are per-pair, so actual: ~6.75 bits (from output)")
    print()
    print("  Compression ratio:")
    print("    - Baseline: bf16 = 16 bits")
    print("    - TurboAngle: 6.75 bits")
    print("    - Ratio: 16 / 6.75 = 2.37×")
    print()

    # TurboAngle E4 (K256V128)
    print("TurboAngle E4 (K256V128):")
    print("  K cache:")
    print("    - Angles: log2(256) = 8 bits per pair → 4.0 bits per element")
    print("    - Norms: 8 bits")
    print("    - Total K: (4.0 + 8) / 2 = 6.0 bits per element")
    print()
    print("  V cache:")
    print("    - Angles: log2(128) = 7 bits per pair → 3.5 bits per element")
    print("    - Norms: 4 bits")
    print("    - Total V: (3.5 + 4) / 2 = 3.75 bits per element")
    print()
    print("  Overall average:")
    print("    - (6.0 + 3.75) / 2 = 4.875 bits per element")
    print("    - Actual: ~8.0 bits (estimated)")
    print()
    print("  Compression ratio:")
    print("    - Baseline: bf16 = 16 bits")
    print("    - TurboAngle E4: 8.0 bits")
    print("    - Ratio: 16 / 8.0 = 2.0×")
    print()

    print("Summary:")
    print("  - TurboAngle Baseline: 2.37× compression")
    print("  - TurboAngle E4: 2.0× compression (higher quality, lower compression)")
    print()


def calculate_kv_cache_size(num_layers, n_heads, seq_len, head_dim, dtype_bytes):
    """Calculate KV cache size in bytes."""
    # K and V each: [B=1, n_heads, seq_len, head_dim]
    k_size = n_heads * seq_len * head_dim * dtype_bytes
    v_size = n_heads * seq_len * head_dim * dtype_bytes
    total_per_layer = k_size + v_size
    total_all_layers = total_per_layer * num_layers
    return total_all_layers


def analyze_memory_breakdown():
    """Analyze memory breakdown at different context lengths."""
    print("=" * 80)
    print("Memory Breakdown Analysis")
    print("=" * 80)
    print()

    MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)

    num_layers = len(model.model.layers)

    # Get model config
    attn = model.model.layers[0].self_attn
    n_heads = attn.n_heads
    q_proj_out = attn.q_proj.weight.shape[0]
    head_dim = q_proj_out // n_heads

    print(f"Model config:")
    print(f"  Layers: {num_layers}")
    print(f"  Heads: {n_heads}")
    print(f"  Head dim: {head_dim}")
    print()

    # Estimate model parameter size
    # model.parameters() returns dict, need to get values
    params_dict = model.parameters()
    if isinstance(params_dict, dict):
        total_params = sum(p.size for p in params_dict.values())
    else:
        total_params = sum(p.size for p in params_dict)
    param_size_mb = total_params * 2 / (1024**2)  # bf16 = 2 bytes
    print(f"Model parameters: {total_params:,} (~{param_size_mb:.1f} MB)")
    print()

    # Calculate KV cache sizes at different context lengths
    context_lengths = [1717, 4096, 8192, 16384, 32768]

    print(f"{'Context':<10} {'Standard KV':>15} {'TurboAngle':>15} {'Savings':>12} {'% Saved':>10}")
    print("-" * 80)

    for seq_len in context_lengths:
        # Standard (bf16)
        standard_bytes = calculate_kv_cache_size(num_layers, n_heads, seq_len, head_dim, 2)
        standard_mb = standard_bytes / (1024**2)

        # TurboAngle baseline (6.75 bits per element)
        turbo_bytes = calculate_kv_cache_size(num_layers, n_heads, seq_len, head_dim, 6.75/8)
        turbo_mb = turbo_bytes / (1024**2)

        savings_mb = standard_mb - turbo_mb
        pct_saved = (savings_mb / standard_mb) * 100

        print(f"{seq_len:<10} {standard_mb:>12.1f} MB {turbo_mb:>12.1f} MB "
              f"{savings_mb:>9.1f} MB {pct_saved:>9.1f}%")

    print()

    # Show total memory (model + KV cache)
    print("Total Memory (Model + KV Cache):")
    print(f"{'Context':<10} {'Standard':>15} {'TurboAngle':>15} {'Savings':>12}")
    print("-" * 80)

    for seq_len in context_lengths:
        standard_bytes = calculate_kv_cache_size(num_layers, n_heads, seq_len, head_dim, 2)
        standard_mb = standard_bytes / (1024**2)
        standard_total = param_size_mb + standard_mb

        turbo_bytes = calculate_kv_cache_size(num_layers, n_heads, seq_len, head_dim, 6.75/8)
        turbo_mb = turbo_bytes / (1024**2)
        turbo_total = param_size_mb + turbo_mb

        savings_mb = standard_total - turbo_total

        print(f"{seq_len:<10} {standard_total:>12.1f} MB {turbo_total:>12.1f} MB "
              f"{savings_mb:>9.1f} MB")

    print()


def measure_actual_compression():
    """Measure actual compression on real model."""
    print("=" * 80)
    print("Actual Compression Measurement")
    print("=" * 80)
    print()

    MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    print(f"✅ Model loaded")
    print()

    # Test at different context lengths
    test_contexts = [
        ("Short (1.7K)", 1717),
        ("Medium (4K)", 4096),
        ("Long (8K)", 8192),
    ]

    results = []

    for name, target_tokens in test_contexts:
        print(f"Testing: {name} ({target_tokens} tokens)")

        # Generate text to reach target length
        text = "The Tower of London is a historic castle. " * (target_tokens // 10)
        tokens = tokenizer.encode(text)[:target_tokens]
        tokens_mx = mx.array([tokens])

        print(f"  Actual tokens: {len(tokens)}")

        # Test 1: Standard
        mx.clear_cache()
        mx.reset_peak_memory()

        cache_std = make_prompt_cache(model)
        _ = model(tokens_mx, cache=cache_std)
        mx.eval(_)

        mem_std = mx.get_peak_memory() / (1024**2)

        # Test 2: TurboAngle Baseline
        mx.clear_cache()
        mx.reset_peak_memory()

        from mlx_lm.models.turboangle import TurboAngleQuantizer
        layer_quantizers = {}
        for i in range(len(model.model.layers)):
            layer_quantizers[i] = TurboAngleQuantizer(n_k=128, n_v=64, head_dim=128)

        cache_turbo = make_prompt_cache(
            model,
            kv_cache="triple_pq",
            kv_layer_quantizers=layer_quantizers,
        )
        _ = model(tokens_mx, cache=cache_turbo)
        mx.eval(_)

        mem_turbo = mx.get_peak_memory() / (1024**2)

        savings = mem_std - mem_turbo
        pct_saved = (savings / mem_std) * 100 if mem_std > 0 else 0

        print(f"  Standard: {mem_std:.1f} MB")
        print(f"  TurboAngle: {mem_turbo:.1f} MB")
        print(f"  Savings: {savings:.1f} MB ({pct_saved:.1f}%)")
        print()

        results.append({
            'name': name,
            'tokens': len(tokens),
            'mem_std': mem_std,
            'mem_turbo': mem_turbo,
            'savings': savings,
            'pct_saved': pct_saved,
        })

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print(f"{'Context':<15} {'Tokens':>8} {'Standard':>12} {'TurboAngle':>12} {'Savings':>12} {'% Saved':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<15} {r['tokens']:>8} {r['mem_std']:>9.1f} MB "
              f"{r['mem_turbo']:>9.1f} MB {r['savings']:>9.1f} MB {r['pct_saved']:>9.1f}%")
    print()


def main():
    """Run all analyses."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "TurboAngle Compression Analysis" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    analyze_theoretical_compression()
    analyze_memory_breakdown()
    measure_actual_compression()


if __name__ == "__main__":
    main()
