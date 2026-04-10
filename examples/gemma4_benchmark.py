"""
Gemma 4 Performance Benchmark with FlashMLX

Compares different cache strategies on Gemma 4.
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from flashmlx.vlm_bridge import load_vlm_model, create_vlm_cache, generate_vlm


def benchmark_strategy(model, processor, strategy_name, cache_kwargs, prompt, max_tokens=50):
    """Benchmark a single cache strategy."""
    # Create cache
    cache = create_vlm_cache(model, **cache_kwargs)

    # Warmup
    generate_vlm(model, processor, prompt, cache=cache, max_tokens=5, verbose=False)

    # Actual benchmark (3 runs)
    times = []
    memories = []
    speeds = []

    for i in range(3):
        # Recreate cache for fair comparison
        cache = create_vlm_cache(model, **cache_kwargs)

        response = generate_vlm(
            model, processor,
            prompt=prompt,
            cache=cache,
            max_tokens=max_tokens,
            temperature=0.7,
            verbose=False
        )

        speeds.append(response.generation_tps)
        memories.append(response.peak_memory)

    return {
        "strategy": strategy_name,
        "speed": sum(speeds) / len(speeds),
        "memory": sum(memories) / len(memories),
        "cache_type": type(cache[0]).__name__,
        "cache_layers": len(cache),
    }


def main():
    print("=" * 70)
    print("Gemma 4 Performance Benchmark with FlashMLX")
    print("=" * 70)

    # Load model once
    print("\nLoading Gemma 4...")
    model, processor = load_vlm_model("/Volumes/toshiba/models/gemma-4-E4B")

    # Test prompt (Gemma chat format)
    prompt = (
        "<start_of_turn>user\n"
        "Explain the concept of neural networks in technical detail.\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    # Strategies to test
    strategies = [
        ("Standard (No Compression)", {"strategy": "standard"}),
        ("Triple", {"strategy": "triple"}),
        ("Triple PQ", {"strategy": "triple_pq"}),
    ]

    results = []

    for strategy_name, cache_kwargs in strategies:
        print(f"\n{'='*70}")
        print(f"Testing: {strategy_name}")
        print(f"{'='*70}")

        try:
            result = benchmark_strategy(
                model, processor,
                strategy_name,
                cache_kwargs,
                prompt,
                max_tokens=50
            )
            results.append(result)

            print(f"  Cache type: {result['cache_type']}")
            print(f"  Cache layers: {result['cache_layers']}")
            print(f"  Speed: {result['speed']:.1f} tok/s")
            print(f"  Memory: {result['memory']:.2f} MB")

        except Exception as e:
            print(f"  ❌ Failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if not results:
        print("No successful benchmarks")
        return

    baseline = results[0]

    print(f"\n{'Strategy':<30} {'Speed':<15} {'Memory':<15} {'vs Baseline'}")
    print("-" * 70)

    for r in results:
        speed_change = ((r['speed'] / baseline['speed']) - 1) * 100
        mem_change = ((r['memory'] / baseline['memory']) - 1) * 100

        print(f"{r['strategy']:<30} "
              f"{r['speed']:>6.1f} tok/s   "
              f"{r['memory']:>6.2f} MB     "
              f"{speed_change:+.1f}% / {mem_change:+.1f}%")

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
