#!/usr/bin/env python3
"""
Layerwise Compression Ratio Benchmark

Hypothesis:
    Nonuniform compression (early 5x, mid 2x, late 1.1x) achieves higher total
    memory savings than uniform compression (all 1.5x) while maintaining quality.

Test:
    1. Baseline: Full KVCache (no compression)
    2. Uniform: All layers 1.5x compression
    3. Layerwise Stepped: Early 5.0x, Mid 2.0x, Late 1.1x
    4. Layerwise Linear: 5.0x → 1.1x smooth transition

Metrics:
    - Memory usage (MB)
    - Output quality (consistency check)
    - Speed (tokens/sec)

Expected:
    - Layerwise Stepped: ~1.8x better memory savings than Uniform (2.70x vs 1.5x average)
    - Quality: Should remain high (last layers preserved at 1.1x)
"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
import argparse
from datetime import datetime
import time

from mlx_lm.models.cache import KVCache
from mlx_lm.models.double_layer_cache import DoubleLayerKVCache
from layerwise_compression_strategy import get_layerwise_ratios_custom

def log(msg, end='\n'):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True, end=end)

# Test prompt (import from adaptive window benchmark)
from benchmark_adaptive_window import WORKLOAD_AGENT as TEST_PROMPT

def benchmark_configuration(
    name: str,
    model,
    tokenizer,
    cache_factory,
    num_generate: int = 100
):
    """
    Benchmark a specific cache configuration.

    Returns
    -------
    dict : Performance metrics
    """
    log(f"\n{'='*70}")
    log(f"Benchmarking: {name}")
    log(f"{'='*70}")

    # Tokenize
    tokens = tokenizer.encode(TEST_PROMPT)
    prompt_len = len(tokens)
    log(f"Prompt length: {prompt_len} tokens")

    # Create caches
    num_layers = len(model.model.layers)
    cache_list = cache_factory(num_layers)

    # Prefill
    log("Step 1: Prefill...")
    y = mx.array([tokens])
    mx.eval(y)
    mx.clear_cache()

    prefill_start = time.time()
    logits = model(y[:, :-1], cache=cache_list)
    mx.eval(logits)
    prefill_time = time.time() - prefill_start

    prefill_tps = prompt_len / prefill_time
    log(f"  Prefill: {prefill_tps:.2f} tokens/sec ({prefill_time:.3f}s)")

    # Generate
    log(f"Step 2: Generate {num_generate} tokens...")
    y = mx.array([[tokens[-1]]])

    generate_start = time.time()
    generated_tokens = []

    for i in range(num_generate):
        logits = model(y, cache=cache_list)
        y = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)

        token_id = y[0, 0].item()
        generated_tokens.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

    generate_time = time.time() - generate_start
    generate_tps = len(generated_tokens) / generate_time

    log(f"  Generated {len(generated_tokens)} tokens")
    log(f"  TG speed: {generate_tps:.2f} tokens/sec ({generate_time:.3f}s)")

    # Memory usage
    total_memory = 0
    for cache in cache_list:
        if hasattr(cache, 'nbytes'):
            total_memory += cache.nbytes
        elif hasattr(cache, 'keys') and hasattr(cache, 'values'):
            total_memory += cache.keys.nbytes + cache.values.nbytes

    total_memory_mb = total_memory / (1024 ** 2)
    log(f"  Memory: {total_memory_mb:.2f} MB")

    # Output text
    output_text = tokenizer.decode(generated_tokens[:100])
    log(f"  Output: {output_text[:150]}...")

    # Compression stats (if applicable)
    compression_stats = []
    if hasattr(cache_list[0], 'get_stats'):
        for i, cache in enumerate(cache_list):
            stats = cache.get_stats()
            if stats['num_compressions'] > 0:
                compression_stats.append({
                    'layer': i,
                    'compressions': stats['num_compressions'],
                    'ratio': stats.get('compression_ratio', 'N/A')
                })

        if compression_stats:
            log(f"  Compression stats:")
            # Show first few layers
            for stat in compression_stats[:3]:
                log(f"    Layer {stat['layer']}: {stat['compressions']} compressions, ratio {stat['ratio']}")
            if len(compression_stats) > 6:
                log(f"    ... ({len(compression_stats)} layers compressed)")
                for stat in compression_stats[-3:]:
                    log(f"    Layer {stat['layer']}: {stat['compressions']} compressions, ratio {stat['ratio']}")

    return {
        'name': name,
        'prompt_tokens': prompt_len,
        'generated_tokens': len(generated_tokens),
        'prefill_tps': prefill_tps,
        'generate_tps': generate_tps,
        'memory_mb': total_memory_mb,
        'output_text': output_text,
        'compression_stats': compression_stats
    }

def main():
    parser = argparse.ArgumentParser(description='Layerwise Compression Ratio Benchmark')
    parser.add_argument('--model-path', default='/Volumes/toshiba/models/qwen3-8b-mlx',
                        help='Path to model')
    parser.add_argument('--calibration-dir', required=True,
                        help='Calibration directory')
    parser.add_argument('--num-generate', type=int, default=100,
                        help='Number of tokens to generate')
    parser.add_argument('--memory-budget', type=float, default=2.0,
                        help='Memory budget in MB per layer (uniform baseline)')
    args = parser.parse_args()

    log("=" * 70)
    log("🔬 Layerwise Compression Ratio Benchmark")
    log("=" * 70)
    log(f"Model: {args.model_path}")
    log(f"Calibration: {args.calibration_dir}")
    log(f"Memory Budget (baseline): {args.memory_budget} MB/layer")

    # Load model
    log("\nLoading model...")
    model, tokenizer = load(args.model_path)
    num_layers = len(model.model.layers)
    log(f"✓ Model loaded: {num_layers} layers")

    # Results storage
    results = []

    # ========================================================================
    # Test 1: Baseline (Full KVCache)
    # ========================================================================
    result_baseline = benchmark_configuration(
        name="Baseline (Full KVCache)",
        model=model,
        tokenizer=tokenizer,
        cache_factory=lambda n: [KVCache() for _ in range(n)],
        num_generate=args.num_generate
    )
    results.append(result_baseline)

    # ========================================================================
    # Test 2: Uniform Compression (1.5x all layers)
    # ========================================================================
    result_uniform = benchmark_configuration(
        name="Uniform Compression (1.5x)",
        model=model,
        tokenizer=tokenizer,
        cache_factory=lambda n: [
            DoubleLayerKVCache(
                memory_budget_mb=args.memory_budget,
                recent_window_size=512,
                compression_ratio=1.5,  # Uniform for all layers
                calibration_dir=args.calibration_dir,
                layer_idx=i,
                enable_compression=True
            )
            for i in range(n)
        ],
        num_generate=args.num_generate
    )
    results.append(result_uniform)

    # ========================================================================
    # Test 3: Layerwise Stepped (Early 5x, Mid 2x, Late 1.1x)
    # ========================================================================
    ratios_stepped, desc_stepped = get_layerwise_ratios_custom(num_layers, strategy="stepped")
    log(f"\n{'='*70}")
    log(f"Layerwise Strategy: {desc_stepped}")
    log(f"  Early (0-11):  {ratios_stepped[0]:.1f}x - {ratios_stepped[11]:.1f}x")
    log(f"  Mid (12-23):   {ratios_stepped[12]:.1f}x - {ratios_stepped[23]:.1f}x")
    log(f"  Late (24-35):  {ratios_stepped[24]:.1f}x - {ratios_stepped[35]:.1f}x")
    log(f"  Average:       {sum(ratios_stepped)/len(ratios_stepped):.2f}x")
    log(f"{'='*70}")

    result_stepped = benchmark_configuration(
        name=f"Layerwise Stepped ({desc_stepped})",
        model=model,
        tokenizer=tokenizer,
        cache_factory=lambda n: [
            DoubleLayerKVCache(
                memory_budget_mb=args.memory_budget,
                recent_window_size=512,
                compression_ratio=ratios_stepped[i],  # Different for each layer
                calibration_dir=args.calibration_dir,
                layer_idx=i,
                enable_compression=True
            )
            for i in range(n)
        ],
        num_generate=args.num_generate
    )
    results.append(result_stepped)

    # ========================================================================
    # Test 4: Layerwise Linear (5.0x → 1.1x smooth transition)
    # ========================================================================
    ratios_linear, desc_linear = get_layerwise_ratios_custom(num_layers, strategy="linear")

    result_linear = benchmark_configuration(
        name=f"Layerwise Linear ({desc_linear})",
        model=model,
        tokenizer=tokenizer,
        cache_factory=lambda n: [
            DoubleLayerKVCache(
                memory_budget_mb=args.memory_budget,
                recent_window_size=512,
                compression_ratio=ratios_linear[i],  # Smoothly decreasing
                calibration_dir=args.calibration_dir,
                layer_idx=i,
                enable_compression=True
            )
            for i in range(n)
        ],
        num_generate=args.num_generate
    )
    results.append(result_linear)

    # ========================================================================
    # Summary
    # ========================================================================
    log("\n\n" + "=" * 70)
    log("📊 Summary: Layerwise vs Uniform Compression")
    log("=" * 70)

    baseline = results[0]
    uniform = results[1]
    stepped = results[2]
    linear = results[3]

    log(f"\n{'Configuration':<40} {'Memory (MB)':<15} {'vs Baseline':<15} {'vs Uniform':<15}")
    log("-" * 85)
    log(f"{baseline['name']:<40} {baseline['memory_mb']:>10.2f} MB {'':>15} {'':>15}")
    log(f"{uniform['name']:<40} {uniform['memory_mb']:>10.2f} MB {uniform['memory_mb']/baseline['memory_mb']:>14.1%} {'':>15}")
    log(f"{stepped['name']:<40} {stepped['memory_mb']:>10.2f} MB {stepped['memory_mb']/baseline['memory_mb']:>14.1%} {stepped['memory_mb']/uniform['memory_mb']:>14.1%}")
    log(f"{linear['name']:<40} {linear['memory_mb']:>10.2f} MB {linear['memory_mb']/baseline['memory_mb']:>14.1%} {linear['memory_mb']/uniform['memory_mb']:>14.1%}")

    # Memory savings calculation
    log("\n" + "=" * 70)
    log("Memory Savings Analysis")
    log("=" * 70)

    uniform_savings = baseline['memory_mb'] - uniform['memory_mb']
    stepped_savings = baseline['memory_mb'] - stepped['memory_mb']
    linear_savings = baseline['memory_mb'] - linear['memory_mb']

    stepped_vs_uniform_improvement = (stepped_savings - uniform_savings) / uniform_savings * 100

    log(f"\nUniform (1.5x):           {uniform_savings:.2f} MB saved (-{uniform_savings/baseline['memory_mb']:.1%})")
    log(f"Layerwise Stepped (2.70x):{stepped_savings:.2f} MB saved (-{stepped_savings/baseline['memory_mb']:.1%})")
    log(f"Layerwise Linear (3.05x): {linear_savings:.2f} MB saved (-{linear_savings/baseline['memory_mb']:.1%})")
    log(f"\n✅ Stepped vs Uniform: {stepped_vs_uniform_improvement:+.1f}% more savings")

    # Quality check (first 50 chars)
    log("\n" + "=" * 70)
    log("Quality Check (First 50 chars of output)")
    log("=" * 70)
    for result in results:
        log(f"{result['name']}: {result['output_text'][:50]}")

    log("\n" + "=" * 70)

if __name__ == '__main__':
    main()
