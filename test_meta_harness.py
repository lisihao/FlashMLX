#!/usr/bin/env python3
"""
Quick test of FlashMLX Meta-Harness.

Runs a small optimization on a subset of configurations.
"""

from flashmlx_meta_harness import FlashMLXMetaHarness, BenchmarkConfig

def main():
    MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("Testing FlashMLX Meta-Harness\n")

    # Initialize harness
    harness = FlashMLXMetaHarness(
        model_path=MODEL_PATH,
        test_prompt="The quick brown fox jumps over the lazy dog. " * 200,  # ~200 tokens
    )

    # Quick test with 3 configs
    print("\nRunning quick test with 3 configurations...")

    configs = [
        BenchmarkConfig(kv_cache="standard"),
        BenchmarkConfig(kv_cache="triple_pq", kv_warm_bits=4, strategy="scored_pq"),
        BenchmarkConfig(kv_cache="triple_pq", kv_warm_bits=4, strategy="scored_pq", density_mode="balanced"),
    ]

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]")
        result = harness.benchmark_config(config)
        harness.results.append(result)

    # Print summary
    harness.print_summary()

    # Show Pareto frontier
    frontier = harness.get_pareto_frontier()
    print(f"\nPareto frontier: {len(frontier)} configurations")
    for r in frontier:
        print(f"  - {r.config.strategy or 'standard'}: "
              f"PPL={r.perplexity:.4f}, Speed={r.tokens_per_sec:.1f}/s, "
              f"Memory={r.peak_memory_mb:.1f}MB")

    # Save results
    harness.save_results("test_meta_harness_results.json")

    print("\n✅ Meta-harness test completed!")


if __name__ == "__main__":
    main()
