"""
Test VLM with FlashMLX Cache

Compares generation performance with different cache strategies.
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import mlx.core as mx

sys.path.insert(0, str(project_root / "src/flashmlx/models"))
sys.path.insert(0, str(project_root / "src/flashmlx/generation"))
sys.path.insert(0, str(project_root / "examples"))

from test_real_weights import download_and_load_model
from vlm_generator import VLMGenerator
from vlm_cache import create_vlm_cache, create_vlm_cache_from_preset, get_vlm_cache_info


def test_cache_strategy(model, tokenizer, config, cache_name: str, preset: str = None):
    """Test a specific cache strategy."""
    print(f"\n{'='*60}")
    print(f"Testing: {cache_name}")
    print(f"{'='*60}")

    # Create cache
    if preset:
        print(f"  Using preset: {preset}")
        cache = create_vlm_cache_from_preset(model, preset)
    else:
        print(f"  Creating cache: {cache_name}")
        cache = create_vlm_cache(model, kv_cache=cache_name)

    # Get cache info
    cache_info = get_vlm_cache_info(cache)
    print(f"  Cache info: {cache_info}")

    # Create generator with cache
    generator = VLMGenerator(
        model=model,
        tokenizer=tokenizer,
        image_token_id=config.image_token_id,
        max_tokens=100,
    )

    # Test text-only generation
    prompt = "What is a vision-language model? Explain in detail."
    print(f"\n  Prompt: {prompt[:50]}...")

    start_time = time.time()
    response = generator.generate(
        prompt=prompt,
        max_tokens=100,
        temperature=0.0,
        cache=cache,
    )
    elapsed = time.time() - start_time

    print(f"\n  Response ({len(response)} chars, {elapsed:.2f}s):")
    print(f"    {response[:150]}...")

    # Calculate tokens/sec (approximate)
    approx_tokens = len(response.split())
    tokens_per_sec = approx_tokens / elapsed if elapsed > 0 else 0

    print(f"\n  Performance:")
    print(f"    Time: {elapsed:.2f}s")
    print(f"    ~{tokens_per_sec:.1f} tokens/sec")

    return {
        "cache_name": cache_name,
        "time": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "response_length": len(response),
        "cache_info": cache_info,
    }


def main():
    """Test VLM with different cache strategies."""
    print("="*60)
    print("VLM Cache Integration Test")
    print("="*60)

    # Load model
    print("\n[1/3] Loading model...")
    model, tokenizer, processor, config = download_and_load_model(use_4bit=False)

    # Test strategies
    print("\n[2/3] Testing cache strategies...")

    caches_to_test = [
        ("standard", None),     # No compression (baseline)
        ("scored_pq", None),    # Route 5: scored PQ (81% savings)
        ("triple_pq", None),    # Triple-layer PQ (72% savings)
    ]

    results = []

    for cache_name, preset in caches_to_test:
        try:
            result = test_cache_strategy(
                model, tokenizer, config,
                cache_name=cache_name,
                preset=preset
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("[3/3] Summary")
    print("="*60)

    if len(results) > 0:
        print("\n| Cache | Time (s) | Tokens/sec | Cache Type |")
        print("|-------|----------|------------|------------|")

        for r in results:
            cache_type = r["cache_info"].get("cache_type", "N/A")
            print(f"| {r['cache_name']:12} | {r['time']:8.2f} | {r['tokens_per_sec']:10.1f} | {cache_type:14} |")

        # Calculate speedup
        if len(results) >= 2:
            baseline_time = results[0]["time"]
            optimized_time = results[1]["time"]
            speedup = ((baseline_time - optimized_time) / baseline_time * 100)

            print(f"\nSpeedup: {speedup:+.1f}% {'(faster)' if speedup > 0 else '(slower)'}")

    print("\n✅ Cache integration test complete!")
    print("\nNext steps:")
    print("  1. Benchmark with longer contexts (4K, 8K)")
    print("  2. Test vision+text generation with cache")
    print("  3. Compare different cache strategies")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
