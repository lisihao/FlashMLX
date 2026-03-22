"""
CompactedKVCache Production Performance Benchmark

Tests key metrics for 35B model with different cache configurations:
- PP (Prompt Processing): tokens/s
- TG (Token Generation): tokens/s
- Memory: peak and average usage
- TTFT (Time To First Token): latency
- TTOT (Time To Output Token): per-token latency

Configurations:
1. Baseline: Standard KVCache
2. Fast Path 5x: CompactedKVCache (Fast Path, 5x compression)
3. Quality Path 5x: CompactedKVCache (Quality Path, 5x compression)
4. Fast Path 10x: CompactedKVCache (Fast Path, 10x compression)

Scenarios:
- Short (512 tokens prompt)
- Medium (2K tokens prompt)
- Long (8K tokens prompt)
- Ultra-long (16K tokens prompt)
"""

import mlx.core as mx
import time
import psutil
import os
import json
from pathlib import Path
from datetime import datetime
from mlx_lm import load, generate
from mlx_lm.models.compacted_cache import CompactedKVCache


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def create_cache_config(config_type, num_layers):
    """
    Create cache configuration based on type.

    Args:
        config_type: 'baseline', 'fast_5x', 'quality_5x', 'fast_10x'
        num_layers: number of model layers

    Returns:
        List of cache objects or None (for baseline)
    """
    if config_type == 'baseline':
        return None  # Use default KVCache

    elif config_type == 'fast_5x':
        return [
            CompactedKVCache(
                max_size=4096,
                compression_ratio=5.0,
                use_quality_path=False,
                enable_compression=True
            )
            for _ in range(num_layers)
        ]

    elif config_type == 'quality_5x':
        return [
            CompactedKVCache(
                max_size=4096,
                compression_ratio=5.0,
                use_quality_path=True,
                quality_fit_beta=True,
                quality_fit_c2=True,
                enable_compression=True
            )
            for _ in range(num_layers)
        ]

    elif config_type == 'fast_10x':
        return [
            CompactedKVCache(
                max_size=4096,
                compression_ratio=10.0,
                use_quality_path=False,
                enable_compression=True
            )
            for _ in range(num_layers)
        ]

    else:
        raise ValueError(f"Unknown config type: {config_type}")


def generate_prompt(length):
    """
    Generate a prompt of approximately the specified length.

    Uses repetitive but coherent text to ensure consistent tokenization.
    """
    base_text = (
        "The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. "
        "The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. "
        "This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. "
        "This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. "
    )

    # Repeat text to reach desired length
    num_repeats = length // 100  # Approximately 100 tokens per base_text
    prompt = base_text * num_repeats

    return prompt


def benchmark_single_run(model, tokenizer, prompt, cache, config_name):
    """
    Run a single benchmark with given configuration.

    Returns:
        dict with metrics
    """
    print(f"\n  Running {config_name}...")

    # Measure initial memory
    mem_before = get_memory_usage()
    mx.metal.clear_cache()

    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    prompt_length = len(tokens)

    # Measure generation
    start_time = time.time()
    ttft_measured = False
    ttft = None
    token_times = []

    generated_tokens = []
    for token in generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=100,  # Generate 100 tokens
        prompt_cache=cache,
        verbose=False
    ):
        current_time = time.time() - start_time

        if not ttft_measured:
            ttft = current_time
            ttft_measured = True

        token_times.append(current_time)
        generated_tokens.append(token)

    total_time = time.time() - start_time

    # Measure final memory
    mem_after = get_memory_usage()
    mem_peak = mem_after  # Approximation (psutil doesn't track peak easily)

    # Calculate metrics
    num_generated = len(generated_tokens)

    # PP (Prompt Processing): time to first token / prompt length
    pp_time = ttft if ttft else total_time
    pp_tokens_per_sec = prompt_length / pp_time if pp_time > 0 else 0

    # TG (Token Generation): (total time - ttft) / num_generated
    tg_time = total_time - ttft if ttft else total_time
    tg_tokens_per_sec = num_generated / tg_time if tg_time > 0 and num_generated > 0 else 0

    # TTOT (Time To Output Token): average per-token latency
    if len(token_times) > 1:
        ttot_avg = sum(token_times[i] - token_times[i-1] for i in range(1, len(token_times))) / (len(token_times) - 1)
    else:
        ttot_avg = tg_time / num_generated if num_generated > 0 else 0

    # Memory usage
    mem_used = mem_after - mem_before

    # Compression stats (if applicable)
    compression_stats = {}
    if cache is not None:
        total_compressions = 0
        avg_compression_ratio = 0
        for c in cache:
            stats = c.get_stats()
            total_compressions += stats['num_compressions']
            if stats['num_compressions'] > 0:
                avg_compression_ratio += stats['avg_compression_ratio']

        if total_compressions > 0:
            avg_compression_ratio /= len([c for c in cache if c.get_stats()['num_compressions'] > 0])

        compression_stats = {
            'total_compressions': total_compressions,
            'avg_compression_ratio': avg_compression_ratio
        }

    return {
        'config_name': config_name,
        'prompt_length': prompt_length,
        'generated_tokens': num_generated,
        'pp_tokens_per_sec': pp_tokens_per_sec,
        'tg_tokens_per_sec': tg_tokens_per_sec,
        'ttft': ttft,
        'ttot_avg': ttot_avg,
        'total_time': total_time,
        'mem_used_mb': mem_used,
        'mem_peak_mb': mem_peak,
        'compression_stats': compression_stats
    }


def run_benchmark_suite(model_path):
    """
    Run complete benchmark suite.
    """
    print("=" * 80)
    print("CompactedKVCache Production Performance Benchmark")
    print("=" * 80)
    print(f"\nModel: {model_path}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load model
    print("Loading model...")
    model, tokenizer = load(model_path)
    num_layers = len(model.layers)
    print(f"Model loaded: {num_layers} layers")

    # Test scenarios
    scenarios = [
        ("Short (512 tokens)", 512),
        ("Medium (2K tokens)", 2048),
        ("Long (8K tokens)", 8192),
        ("Ultra-long (16K tokens)", 16384),
    ]

    # Cache configurations
    configs = [
        ("Baseline (Standard KVCache)", "baseline"),
        ("Fast Path 5x", "fast_5x"),
        ("Quality Path 5x", "quality_5x"),
        ("Fast Path 10x", "fast_10x"),
    ]

    # Results storage
    all_results = []

    # Run benchmarks
    for scenario_name, prompt_length in scenarios:
        print(f"\n{'=' * 80}")
        print(f"Scenario: {scenario_name}")
        print(f"{'=' * 80}")

        prompt = generate_prompt(prompt_length)

        for config_name, config_type in configs:
            try:
                # Create cache
                cache = create_cache_config(config_type, num_layers)

                # Run benchmark
                result = benchmark_single_run(
                    model, tokenizer, prompt, cache, config_name
                )
                result['scenario'] = scenario_name
                result['scenario_prompt_length'] = prompt_length
                all_results.append(result)

                # Print results
                print(f"    PP: {result['pp_tokens_per_sec']:.2f} tok/s")
                print(f"    TG: {result['tg_tokens_per_sec']:.2f} tok/s")
                print(f"    TTFT: {result['ttft']:.3f}s")
                print(f"    TTOT: {result['ttot_avg']*1000:.2f}ms")
                print(f"    Memory: {result['mem_used_mb']:.2f} MB")

                if result['compression_stats']:
                    print(f"    Compressions: {result['compression_stats']['total_compressions']}")
                    if result['compression_stats']['total_compressions'] > 0:
                        print(f"    Avg ratio: {result['compression_stats']['avg_compression_ratio']:.2f}x")

            except Exception as e:
                print(f"    ERROR: {str(e)}")
                import traceback
                traceback.print_exc()

    # Save results
    output_file = Path(__file__).parent / "compacted_cache_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'model': model_path,
            'timestamp': datetime.now().isoformat(),
            'num_layers': num_layers,
            'results': all_results
        }, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 80}")

    # Generate summary
    generate_summary(all_results)

    return all_results


def generate_summary(results):
    """
    Generate summary report from benchmark results.
    """
    print(f"\n{'=' * 80}")
    print("Summary Report")
    print(f"{'=' * 80}\n")

    # Group by scenario
    scenarios = {}
    for r in results:
        scenario = r['scenario']
        if scenario not in scenarios:
            scenarios[scenario] = []
        scenarios[scenario].append(r)

    # Print comparison tables
    for scenario, scenario_results in scenarios.items():
        print(f"\n{scenario}")
        print("-" * 80)
        print(f"{'Config':<25} {'PP (tok/s)':<15} {'TG (tok/s)':<15} {'TTFT (s)':<12} {'Memory (MB)':<15}")
        print("-" * 80)

        baseline = None
        for r in scenario_results:
            print(f"{r['config_name']:<25} "
                  f"{r['pp_tokens_per_sec']:<15.2f} "
                  f"{r['tg_tokens_per_sec']:<15.2f} "
                  f"{r['ttft']:<12.3f} "
                  f"{r['mem_used_mb']:<15.2f}")

            if 'Baseline' in r['config_name']:
                baseline = r

        # Calculate improvements
        if baseline:
            print("\nImprovements vs Baseline:")
            for r in scenario_results:
                if 'Baseline' in r['config_name']:
                    continue

                pp_improvement = ((r['pp_tokens_per_sec'] / baseline['pp_tokens_per_sec']) - 1) * 100
                tg_improvement = ((r['tg_tokens_per_sec'] / baseline['tg_tokens_per_sec']) - 1) * 100
                mem_savings = ((baseline['mem_used_mb'] - r['mem_used_mb']) / baseline['mem_used_mb']) * 100

                print(f"  {r['config_name']:<25} "
                      f"PP: {pp_improvement:+.1f}%  "
                      f"TG: {tg_improvement:+.1f}%  "
                      f"Memory: {mem_savings:+.1f}%")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="CompactedKVCache Production Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-32B-Instruct-4bit",
        help="Path or HF repo to model (default: Qwen2.5-32B)"
    )

    args = parser.parse_args()
    model_path = os.path.expanduser(args.model)

    # Run benchmark
    run_benchmark_suite(model_path)


if __name__ == '__main__':
    main()
