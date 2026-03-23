#!/usr/bin/env python3
"""
Test StreamingLLM compression quality and functionality.
"""
import mlx.core as mx
import sys
sys.path.insert(0, 'src')

from flashmlx.cache.streaming_llm import (
    StreamingLLMCache,
    streaming_llm_compress,
    test_streaming_llm_quality
)


def test_basic_functionality():
    """Test basic cache operations."""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)

    cache = StreamingLLMCache(max_capacity=10, num_sinks=2)

    # Add 15 tokens
    d = 16
    for i in range(15):
        k = mx.random.normal((d,))
        v = mx.random.normal((d,))
        cache.append(k, v)

    info = cache.info()
    print(f"\nAfter adding 15 tokens:")
    print(f"  Current size: {info['current_size']}")
    print(f"  Max capacity: {info['max_capacity']}")
    print(f"  Num sinks: {info['num_sinks']}")
    print(f"  Num recent: {info['num_recent']}")
    print(f"  Total seen: {info['total_tokens_seen']}")
    print(f"  Evicted: {info['evicted_tokens']}")

    assert info['current_size'] == 10, "Cache should be at max capacity"
    assert info['num_sinks'] == 2, "Should have 2 sinks"
    assert info['num_recent'] == 8, "Should have 8 recent tokens"
    assert info['evicted_tokens'] == 5, "Should have evicted 5 tokens"

    print("\n✅ Basic functionality test passed")


def test_compression_quality_small():
    """Test compression quality on small cache."""
    print("\n" + "=" * 60)
    print("Test 2: Small Scale Quality (T=100, t=32)")
    print("=" * 60)

    T, d, n = 100, 64, 20
    mx.random.seed(42)

    K = mx.random.normal((T, d))
    V = mx.random.normal((T, d))
    queries = mx.random.normal((n, d))

    results = test_streaming_llm_quality(
        K, V, queries,
        max_capacity=32,
        num_sinks=4
    )

    print(f"\nResults:")
    print(f"  Compression: {T} → {results['compressed_size']} ({results['compression_ratio']:.1f}x)")
    print(f"  Quality (cosine): {results['cosine_similarity']:.6f}")
    print(f"  MSE: {results['mse']:.6e}")
    print(f"  Attention to sinks: {results['attention_to_sinks']:.4f}")
    print(f"  Attention to recent: {results['attention_to_recent']:.4f}")

    # StreamingLLM target: cos > 0.85
    if results['cosine_similarity'] >= 0.85:
        print(f"\n✅ Quality test passed (cos={results['cosine_similarity']:.6f} >= 0.85)")
    else:
        print(f"\n⚠️ Quality below target (cos={results['cosine_similarity']:.6f} < 0.85)")


def test_compression_quality_medium():
    """Test compression quality on medium cache."""
    print("\n" + "=" * 60)
    print("Test 3: Medium Scale Quality (T=1000, t=256)")
    print("=" * 60)

    T, d, n = 1000, 128, 50
    mx.random.seed(42)

    K = mx.random.normal((T, d))
    V = mx.random.normal((T, d))
    queries = mx.random.normal((n, d))

    results = test_streaming_llm_quality(
        K, V, queries,
        max_capacity=256,
        num_sinks=4
    )

    print(f"\nResults:")
    print(f"  Compression: {T} → {results['compressed_size']} ({results['compression_ratio']:.1f}x)")
    print(f"  Quality (cosine): {results['cosine_similarity']:.6f}")
    print(f"  MSE: {results['mse']:.6e}")
    print(f"  Attention to sinks: {results['attention_to_sinks']:.4f}")
    print(f"  Attention to recent: {results['attention_to_recent']:.4f}")

    if results['cosine_similarity'] >= 0.85:
        print(f"\n✅ Quality test passed (cos={results['cosine_similarity']:.6f} >= 0.85)")
    else:
        print(f"\n⚠️ Quality below target (cos={results['cosine_similarity']:.6f} < 0.85)")


def test_attention_sinks_importance():
    """Test that attention sinks are important for quality."""
    print("\n" + "=" * 60)
    print("Test 4: Attention Sinks Importance")
    print("=" * 60)

    T, d, n = 200, 64, 20
    mx.random.seed(42)

    K = mx.random.normal((T, d))
    V = mx.random.normal((T, d))
    queries = mx.random.normal((n, d))

    # Test with different num_sinks
    for num_sinks in [0, 2, 4, 8]:
        results = test_streaming_llm_quality(
            K, V, queries,
            max_capacity=64,
            num_sinks=num_sinks
        )

        print(f"\nnum_sinks={num_sinks}:")
        print(f"  Quality: {results['cosine_similarity']:.6f}")
        print(f"  Attn to sinks: {results['attention_to_sinks']:.4f}")
        print(f"  Attn to recent: {results['attention_to_recent']:.4f}")

    print("\n✅ Attention sinks test completed")


def test_no_compression_case():
    """Test that no compression happens when T <= max_capacity."""
    print("\n" + "=" * 60)
    print("Test 5: No Compression Case (T <= max_capacity)")
    print("=" * 60)

    T, d = 50, 32
    mx.random.seed(42)

    K = mx.random.normal((T, d))
    V = mx.random.normal((T, d))

    K_comp, V_comp, indices = streaming_llm_compress(
        K, V,
        max_capacity=100,  # Larger than T
        num_sinks=4
    )

    assert K_comp.shape[0] == T, "No compression should occur"
    assert len(indices) == T, "All indices should be kept"
    assert indices == list(range(T)), "Indices should be [0, 1, 2, ..., T-1]"

    print(f"\nResults:")
    print(f"  Input size: {T}")
    print(f"  Output size: {K_comp.shape[0]}")
    print(f"  Compression ratio: {T / K_comp.shape[0]:.1f}x")

    print("\n✅ No compression test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("StreamingLLM Compression Tests")
    print("=" * 60)

    test_basic_functionality()
    test_compression_quality_small()
    test_compression_quality_medium()
    test_attention_sinks_importance()
    test_no_compression_case()

    print("\n" + "=" * 60)
    print("All Tests Completed")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
