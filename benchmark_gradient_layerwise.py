#!/usr/bin/env python3
"""
Benchmark gradient layerwise compression

Test gradient budgets for different layer groups:
- L0-12: 0.5 (keep 50% - aggressive)
- L12-24: 0.7 (keep 70% - moderate)
- L24-30: 0.85 (keep 85% - light)
- L30-36: 1.0 (keep 100% - no compression)

Compare with:
- Baseline: Uniform R1.5 (all layers)
- Binary: First 50% at R1.5, last 50% no compression
"""

import sys
import time
import json
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.double_layer_cache import DoubleLayerKVCache

# Agent debugging session prompt
AGENT_DEBUG_PROMPT = """You are a senior production support engineer investigating a critical issue. The application has been experiencing intermittent 500 errors over the past 24 hours. The error occurs randomly, affecting about 15% of requests. Initial investigation shows:

1. Database connections are stable
2. Memory usage is normal
3. CPU usage shows occasional spikes
4. Error logs show: "Connection pool exhausted" every few minutes
5. The issue started after deploying version 2.1.3
6. Version 2.1.3 added a new background worker for data processing

Based on the evidence, what is the root cause of the 500 errors?"""


def compute_quality_metrics(text: str) -> dict:
    """Compute quality metrics"""
    lines = text.strip().split('\n')
    if not lines:
        return {"repetition": 0.0, "coherence": 0.0, "relevance": 0.0}

    # Repetition: ratio of repeated lines
    unique_lines = set(lines)
    repetition_ratio = 1.0 - (len(unique_lines) / len(lines))

    # Coherence: ratio of lines with meaningful content (>10 chars)
    meaningful_lines = [l for l in lines if len(l.strip()) > 10]
    coherence_score = len(meaningful_lines) / len(lines) if lines else 0.0

    # Relevance: mentions problem-related keywords
    keywords = ["connection", "pool", "worker", "database", "error", "cause", "issue"]
    relevant_lines = [l for l in lines if any(k in l.lower() for k in keywords)]
    relevance_score = len(relevant_lines) / len(lines) if lines else 0.0

    return {
        "repetition": repetition_ratio,
        "coherence": coherence_score,
        "relevance": relevance_score
    }


def get_layer_budget(layer_idx: int, num_layers: int, config_name: str) -> float:
    """
    Get compression budget for a layer

    Budget = ratio of tokens to keep (1.0 = keep all, 0.5 = keep half)

    Configurations:
    - baseline: 1.0 for all (no budget constraint, let R1.5 compress normally)
    - binary: 1.0 for first 50%, 1.0 for last 50% (R1.5 applied to first half only)
    - gradient: 0.5 (L0-12), 0.7 (L12-24), 0.85 (L24-30), 1.0 (L30-36)
    """
    if config_name == "baseline":
        # Uniform R1.5 for all layers
        return 1.0  # No budget constraint

    elif config_name == "binary":
        # First 50% compressed at R1.5, last 50% not compressed
        if layer_idx < num_layers // 2:
            return 1.0  # Let R1.5 compress
        else:
            return 1.0  # No compression (disable_compression=True)

    elif config_name == "gradient":
        # Gradient budgets
        if layer_idx < 12:
            return 0.5  # Keep 50% of tokens
        elif layer_idx < 24:
            return 0.7  # Keep 70% of tokens
        elif layer_idx < 30:
            return 0.85  # Keep 85% of tokens
        else:
            return 1.0  # Keep all tokens (no compression)

    else:
        raise ValueError(f"Unknown config: {config_name}")


def benchmark_configuration(
    config_name: str,
    model,
    tokenizer,
    num_layers: int = 36,
    num_generate: int = 100
) -> dict:
    """
    Benchmark a specific layerwise compression configuration

    Args:
        config_name: "baseline", "binary", or "gradient"
        model: MLX model
        tokenizer: Tokenizer
        num_layers: Number of layers (36 for Qwen3-8B)
        num_generate: Number of tokens to generate
    """
    print(f"\n{'='*80}")
    print(f"Testing: {config_name.upper()}")
    print(f"{'='*80}")

    # Tokenize prompt
    input_ids = mx.array(tokenizer.encode(AGENT_DEBUG_PROMPT))
    prompt_len = input_ids.shape[0]
    print(f"Prompt length: {prompt_len} tokens")

    # Create layerwise caches
    caches = []
    for layer_idx in range(num_layers):
        budget = get_layer_budget(layer_idx, num_layers, config_name)

        if config_name == "binary" and layer_idx >= num_layers // 2:
            # Binary: last 50% not compressed
            cache = DoubleLayerKVCache(
                memory_budget_mb=2.0,
                recent_window_size=512,  # Keep at 512 (learned from window test)
                compression_ratio=1.0,   # No compression
                calibration_dir="/tmp/am_calibrations_ultra_dense",
                layer_idx=layer_idx,
                enable_compression=False,  # Disable compression
                selection_strategy="nearest"
            )
        else:
            # Baseline or Gradient or Binary first half
            cache = DoubleLayerKVCache(
                memory_budget_mb=2.0,
                recent_window_size=512,  # Keep at 512 (learned from window test)
                compression_ratio=1.5,   # R1.5
                calibration_dir="/tmp/am_calibrations_ultra_dense",
                layer_idx=layer_idx,
                enable_compression=True,
                selection_strategy="nearest",
                compression_budget=budget if config_name == "gradient" else None
            )

        caches.append(cache)

        if layer_idx % 6 == 0:  # Print every 6 layers
            status = "no compression" if (config_name == "binary" and layer_idx >= num_layers // 2) else f"budget={budget}"
            print(f"  Layer {layer_idx:2d}: {status}")

    # Generate
    print(f"\nGenerating {num_generate} tokens...")
    start_time = time.time()

    output = generate(
        model,
        tokenizer,
        prompt=AGENT_DEBUG_PROMPT,
        max_tokens=num_generate,
        verbose=False,
        cache=caches
    )

    gen_time = time.time() - start_time

    # Extract generated text (remove prompt)
    generated_text = output[len(AGENT_DEBUG_PROMPT):].strip()

    # Compute memory usage
    total_memory_mb = 0
    old_prefix_memory_mb = 0
    recent_memory_mb = 0

    for cache in caches:
        if hasattr(cache, 'get_memory_usage'):
            mem = cache.get_memory_usage()
            total_memory_mb += mem['total_mb']
            old_prefix_memory_mb += mem.get('old_prefix_mb', 0)
            recent_memory_mb += mem.get('recent_mb', 0)

    # Compute quality metrics
    quality = compute_quality_metrics(generated_text)

    # Compute tokens per second
    tps = num_generate / gen_time if gen_time > 0 else 0

    # Results
    results = {
        "config": config_name,
        "prompt_len": prompt_len,
        "generated_tokens": num_generate,
        "generation_time_sec": gen_time,
        "tokens_per_second": tps,
        "memory": {
            "total_mb": total_memory_mb,
            "old_prefix_mb": old_prefix_memory_mb,
            "recent_mb": recent_memory_mb
        },
        "quality": quality,
        "output_preview": generated_text[:200]
    }

    # Print results
    print(f"\n{config_name.upper()} Results:")
    print(f"  Memory: {total_memory_mb:.1f} MB (Old: {old_prefix_memory_mb:.1f} MB, Recent: {recent_memory_mb:.1f} MB)")
    print(f"  Speed: {tps:.2f} tok/s")
    print(f"  Quality:")
    print(f"    Repetition: {quality['repetition']*100:.2f}%")
    print(f"    Coherence: {quality['coherence']*100:.2f}%")
    print(f"    Relevance: {quality['relevance']*100:.2f}%")
    print(f"\n  Output (first 200 chars):")
    print(f"  {generated_text[:200]}")

    # Save full output
    output_file = f"/tmp/gradient_{config_name}_output.txt"
    with open(output_file, "w") as f:
        f.write(generated_text)
    print(f"\n  Full output saved to: {output_file}")

    return results


def main():
    print("Gradient Layerwise Compression Benchmark")
    print("=" * 80)

    # Load model
    print("\nLoading Qwen3-8B model...")
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    model, tokenizer = load(model_path)
    num_layers = len(model.model.layers)
    print(f"Model loaded: {num_layers} layers")

    # Test configurations
    configs = ["baseline", "binary", "gradient"]

    all_results = []

    for config in configs:
        result = benchmark_configuration(
            config_name=config,
            model=model,
            tokenizer=tokenizer,
            num_layers=num_layers,
            num_generate=100
        )
        all_results.append(result)

        # Cool down between tests
        if config != configs[-1]:
            print("\nCooling down for 5 seconds...")
            time.sleep(5)

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")

    baseline = all_results[0]

    print(f"\n{'Config':<15} {'Memory (MB)':<15} {'Savings':<12} {'Quality':<30} {'Status':<10}")
    print(f"{'-'*80}")

    for result in all_results:
        mem = result['memory']['total_mb']
        savings = (baseline['memory']['total_mb'] - mem) / baseline['memory']['total_mb'] * 100 if baseline['memory']['total_mb'] > 0 else 0

        quality = result['quality']
        rep = quality['repetition'] * 100
        coh = quality['coherence'] * 100
        rel = quality['relevance'] * 100

        # Quality status
        if rep > 20 or coh < 50 or rel < 30:
            status = "❌ BAD"
        elif rep > 10 or coh < 60 or rel < 40:
            status = "⚠️  FAIR"
        else:
            status = "✅ GOOD"

        print(f"{result['config']:<15} {mem:<15.1f} {f'+{savings:.1f}%':<12} R:{rep:.0f}% C:{coh:.0f}% Rel:{rel:.0f}%{'':<5} {status:<10}")

    # Save results
    results_file = "/tmp/gradient_layerwise_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

    print("\n" + "="*80)
    print("Benchmark complete!")
    print("="*80)


if __name__ == "__main__":
    main()
