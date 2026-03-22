#!/usr/bin/env python3
"""
快速质量验证 - 只测试一个简单场景
"""

from mlx_lm import load, generate
from flashmlx.cache import inject_attention_matching
import mlx.core as mx


def test_simple_quality():
    print("="*70)
    print("Quick Quality Test")
    print("="*70)

    # Load model
    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx/"
    print(f"\n加载模型: {model_path}")
    model, tokenizer = load(model_path)

    # Simple prompt
    prompt = "What is 2+2? Answer briefly."

    print(f"\n提示词: {prompt}")

    # Baseline
    print("\n--- Baseline ---")
    model.cache = None  # Clear cache
    baseline_output = generate(
        model, tokenizer, prompt,
        max_tokens=50, verbose=False
    )
    print(baseline_output)

    # With compression
    print("\n--- Compressed (2.0x) ---")
    cache_list, compressor = inject_attention_matching(
        model,
        compression_ratio=2.0,
        beta_calibration=True
    )
    compressed_output = generate(
        model, tokenizer, prompt,
        max_tokens=50, verbose=False
    )
    print(compressed_output)

    # Compare
    print("\n--- Comparison ---")
    baseline_tokens = set(baseline_output.strip().split())
    compressed_tokens = set(compressed_output.strip().split())
    overlap = len(baseline_tokens & compressed_tokens)
    total = len(baseline_tokens | compressed_tokens)
    overlap_pct = overlap / total * 100 if total > 0 else 0

    print(f"Token overlap: {overlap}/{total} ({overlap_pct:.1f}%)")

    # Check compression stats
    print("\n--- Compression Stats ---")
    for i, cache in enumerate(cache_list[:3]):  # First 3 layers
        if hasattr(cache, 'get_stats'):
            stats = cache.get_stats()
            print(f"Layer {i}: {stats['compression_count']} compressions, "
                  f"avg ratio {stats['avg_compression_ratio']:.2f}x")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    test_simple_quality()
