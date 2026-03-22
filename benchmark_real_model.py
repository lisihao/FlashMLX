#!/usr/bin/env python3
"""
FlashMLX Hybrid Cache - Real Model Benchmark

Measures:
- PP (Prompt Processing / Prefill): tok/s
- TG (Token Generation / Decode): tok/s
- Memory usage: Before/After/Saved

Target Model: Qwen3.5-35B-Instruct-4bit
"""

import time
import gc
import mlx.core as mx
from mlx_lm import load, generate
from flashmlx.cache import (
    inject_hybrid_cache_manager,
    restore_original_cache,
    create_layer_types_from_model,
    HybridCacheConfig
)


def get_memory_usage():
    """Get current memory usage in MB."""
    gc.collect()
    mx.metal.clear_cache()

    # Get Metal memory stats
    stats = mx.metal.get_active_memory()
    return stats / (1024 ** 2)  # Convert to MB


def measure_performance(model, tokenizer, prompt: str, max_tokens: int = 100):
    """
    Measure PP and TG performance.

    Returns:
        dict with pp_time, tg_time, pp_tok_s, tg_tok_s, total_tokens
    """
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    prompt_length = len(prompt_tokens)

    # Generate with timing
    start_time = time.time()

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )

    end_time = time.time()
    total_time = end_time - start_time

    # Count generated tokens
    response_tokens = tokenizer.encode(response)
    generated_tokens = len(response_tokens) - prompt_length

    # Estimate PP and TG time
    # PP: Time for prefill (assume first ~10% of time)
    # TG: Time for decode (remaining ~90% of time)
    # This is approximate - real split requires instrumentation

    # Better approximation: PP is roughly proportional to prompt length
    # TG is roughly proportional to generated tokens
    estimated_pp_ratio = 0.3  # Conservative estimate

    pp_time = total_time * estimated_pp_ratio
    tg_time = total_time * (1 - estimated_pp_ratio)

    # Calculate tok/s
    pp_tok_s = prompt_length / pp_time if pp_time > 0 else 0
    tg_tok_s = generated_tokens / tg_time if tg_time > 0 else 0

    return {
        "prompt_length": prompt_length,
        "generated_tokens": generated_tokens,
        "total_time": total_time,
        "pp_time": pp_time,
        "tg_time": tg_time,
        "pp_tok_s": pp_tok_s,
        "tg_tok_s": tg_tok_s,
        "response": response
    }


def benchmark_baseline(model, tokenizer, prompts: list):
    """Benchmark without hybrid cache."""
    print("\n" + "=" * 70)
    print("BASELINE (No Hybrid Cache)")
    print("=" * 70)

    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n[Test {i+1}/{len(prompts)}]")
        print(f"Prompt length: ~{len(tokenizer.encode(prompt))} tokens")

        # Measure memory before
        mem_before = get_memory_usage()

        # Measure performance
        perf = measure_performance(model, tokenizer, prompt, max_tokens=100)

        # Measure memory after
        mem_after = get_memory_usage()

        result = {
            "test_id": i + 1,
            "mem_before_mb": mem_before,
            "mem_after_mb": mem_after,
            "mem_used_mb": mem_after - mem_before,
            **perf
        }

        results.append(result)

        print(f"  PP:  {perf['pp_tok_s']:7.2f} tok/s ({perf['pp_time']*1000:6.1f} ms)")
        print(f"  TG:  {perf['tg_tok_s']:7.2f} tok/s ({perf['tg_time']*1000:6.1f} ms)")
        print(f"  Memory: {mem_after:.1f} MB")

        # Clear cache between tests
        gc.collect()
        mx.metal.clear_cache()

    return results


def benchmark_hybrid_cache(model, tokenizer, prompts: list, config: HybridCacheConfig):
    """Benchmark with hybrid cache."""
    print("\n" + "=" * 70)
    print("HYBRID CACHE ENABLED")
    print("=" * 70)
    print(f"Config: {config.compression_ratio}x compression, "
          f"{config.total_budget_bytes / (1024**2):.0f}MB budget")

    # Detect layer types
    layer_types = create_layer_types_from_model(
        model,
        attention_layer_pattern="every 4th"
    )

    # Inject hybrid cache
    cache_wrapper = inject_hybrid_cache_manager(
        model=model,
        config=config,
        layer_types=layer_types,
        auto_inject=True
    )

    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n[Test {i+1}/{len(prompts)}]")
        print(f"Prompt length: ~{len(tokenizer.encode(prompt))} tokens")

        # Measure memory before
        mem_before = get_memory_usage()

        # Measure performance
        perf = measure_performance(model, tokenizer, prompt, max_tokens=100)

        # Measure memory after
        mem_after = get_memory_usage()

        # Get cache statistics
        stats = cache_wrapper.get_statistics()

        result = {
            "test_id": i + 1,
            "mem_before_mb": mem_before,
            "mem_after_mb": mem_after,
            "mem_used_mb": mem_after - mem_before,
            "cache_stats": {
                "ssm_hit_rate": stats['ssm']['local_cache']['hit_rate'],
                "attention_avg_compression": stats['attention']['local_cache']['avg_compression_ratio'],
                "attention_compressions": stats['attention']['local_cache']['total_compressions']
            },
            **perf
        }

        results.append(result)

        print(f"  PP:  {perf['pp_tok_s']:7.2f} tok/s ({perf['pp_time']*1000:6.1f} ms)")
        print(f"  TG:  {perf['tg_tok_s']:7.2f} tok/s ({perf['tg_time']*1000:6.1f} ms)")
        print(f"  Memory: {mem_after:.1f} MB")
        print(f"  Cache: Hit rate {stats['ssm']['local_cache']['hit_rate']:.2%}, "
              f"Compression {stats['attention']['local_cache']['avg_compression_ratio']:.2f}x")

        # Clear cache statistics but keep cache active
        cache_wrapper.clear()
        gc.collect()
        mx.metal.clear_cache()

    # Restore original cache
    restore_original_cache(model, cache_wrapper)

    return results


def print_comparison(baseline_results: list, hybrid_results: list):
    """Print detailed comparison."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    # Average results
    avg_baseline_pp = sum(r['pp_tok_s'] for r in baseline_results) / len(baseline_results)
    avg_baseline_tg = sum(r['tg_tok_s'] for r in baseline_results) / len(baseline_results)
    avg_baseline_mem = sum(r['mem_used_mb'] for r in baseline_results) / len(baseline_results)

    avg_hybrid_pp = sum(r['pp_tok_s'] for r in hybrid_results) / len(hybrid_results)
    avg_hybrid_tg = sum(r['tg_tok_s'] for r in hybrid_results) / len(hybrid_results)
    avg_hybrid_mem = sum(r['mem_used_mb'] for r in hybrid_results) / len(hybrid_results)

    # Calculate overheads
    pp_overhead = (avg_baseline_pp - avg_hybrid_pp) / avg_baseline_pp * 100
    tg_overhead = (avg_baseline_tg - avg_hybrid_tg) / avg_baseline_tg * 100
    mem_saved = (avg_baseline_mem - avg_hybrid_mem) / avg_baseline_mem * 100

    print("\n{:<25} {:<15} {:<15} {:<15}".format(
        "Metric", "Baseline", "Hybrid Cache", "Difference"
    ))
    print("-" * 70)

    print("{:<25} {:>12.2f}  {:>12.2f}  {:>+12.2f}%".format(
        "PP (tok/s)", avg_baseline_pp, avg_hybrid_pp, -pp_overhead
    ))

    print("{:<25} {:>12.2f}  {:>12.2f}  {:>+12.2f}%".format(
        "TG (tok/s)", avg_baseline_tg, avg_hybrid_tg, -tg_overhead
    ))

    print("{:<25} {:>12.1f}  {:>12.1f}  {:>+12.1f}%".format(
        "Memory (MB)", avg_baseline_mem, avg_hybrid_mem, -mem_saved
    ))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nPP (Prompt Processing):")
    print(f"  Baseline: {avg_baseline_pp:.2f} tok/s")
    print(f"  Hybrid:   {avg_hybrid_pp:.2f} tok/s")
    print(f"  Overhead: {pp_overhead:+.1f}%")

    if pp_overhead > 10:
        print(f"  Status:   ⚠️ Exceeds 10% target (expected for this measurement method)")
    else:
        print(f"  Status:   ✓ Within 10% target")

    print(f"\nTG (Token Generation):")
    print(f"  Baseline: {avg_baseline_tg:.2f} tok/s")
    print(f"  Hybrid:   {avg_hybrid_tg:.2f} tok/s")
    print(f"  Overhead: {tg_overhead:+.1f}%")

    if tg_overhead > 10:
        print(f"  Status:   ⚠️ Exceeds 10% target")
    else:
        print(f"  Status:   ✓ Within 10% target")

    print(f"\nMemory:")
    print(f"  Baseline: {avg_baseline_mem:.1f} MB")
    print(f"  Hybrid:   {avg_hybrid_mem:.1f} MB")
    print(f"  Saved:    {mem_saved:.1f}%")

    if mem_saved >= 15:
        print(f"  Status:   ✓ Significant savings (target: 18.8%)")
    else:
        print(f"  Status:   ⚠️ Below expected savings")

    # Average cache statistics
    if hybrid_results and 'cache_stats' in hybrid_results[0]:
        avg_hit_rate = sum(r['cache_stats']['ssm_hit_rate'] for r in hybrid_results) / len(hybrid_results)
        avg_compression = sum(r['cache_stats']['attention_avg_compression'] for r in hybrid_results) / len(hybrid_results)

        print(f"\nCache Statistics:")
        print(f"  SSM hit rate: {avg_hit_rate:.2%}")
        print(f"  Avg compression: {avg_compression:.2f}x")


def main():
    print("=" * 70)
    print("FlashMLX Hybrid Cache - Real Model Benchmark")
    print("=" * 70)
    print("\nTarget: Qwen3.5-35B-Instruct-4bit")
    print("Metrics: PP (tok/s), TG (tok/s), Memory (MB)")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/Qwen3.5-35B-Instruct-4bit")
    print("✓ Model loaded")

    # Test prompts (different lengths)
    prompts = [
        # Short context (~500 tokens)
        """Explain quantum computing in simple terms.""",

        # Medium context (~1500 tokens)
        """Machine learning is a branch of artificial intelligence (AI) and computer science
        which focuses on the use of data and algorithms to imitate the way that humans learn,
        gradually improving its accuracy. Neural networks are a key component of machine learning,
        inspired by the human brain's structure. They consist of layers of interconnected nodes
        that process information in a hierarchical manner. Deep learning, a subset of machine learning,
        uses multiple layers to progressively extract higher-level features from raw input.

        Explain the key differences between supervised and unsupervised learning.""",

        # Long context (~3000 tokens)
        """Artificial intelligence has evolved significantly over the past decades. """ * 50 +
        """

        Based on the above context, summarize the main milestones in AI development and
        predict future trends in the next 5 years."""
    ]

    # Test configurations
    config = HybridCacheConfig(
        total_budget_bytes=64 * 1024 * 1024,  # 64MB
        compression_ratio=4.0,
        beta_calibration=True
    )

    # Run benchmarks
    print("\n" + "=" * 70)
    print("PHASE 1: Baseline Performance")
    print("=" * 70)
    baseline_results = benchmark_baseline(model, tokenizer, prompts)

    print("\n" + "=" * 70)
    print("PHASE 2: Hybrid Cache Performance")
    print("=" * 70)
    hybrid_results = benchmark_hybrid_cache(model, tokenizer, prompts, config)

    # Print comparison
    print_comparison(baseline_results, hybrid_results)

    print("\n" + "=" * 70)
    print("✓ Benchmark Complete!")
    print("=" * 70)

    print("\nNote: PP/TG measurements use approximation (30/70 split).")
    print("For exact TTFT/TBT, use mlx_lm with verbose=True and manual timing.")


if __name__ == "__main__":
    main()
