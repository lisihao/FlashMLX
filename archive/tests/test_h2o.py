#!/usr/bin/env python3
"""
Test H2O (Heavy-Hitter Oracle) compression quality and functionality.
"""
import mlx.core as mx
import sys
sys.path.insert(0, 'src')

from flashmlx.cache.h2o import (
    H2OCache,
    h2o_compress,
    test_h2o_quality
)


def test_basic_functionality():
    """Test basic H2O cache operations."""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)

    cache = H2OCache(max_capacity=10, recent_ratio=0.3)

    # Add 15 tokens with simulated attention
    d = 16
    for i in range(15):
        k = mx.random.normal((d,))
        v = mx.random.normal((d,))

        # Simulate attention (recent tokens get more attention)
        if len(cache) > 0:
            attn = mx.zeros((len(cache),))
            # Give more attention to recent tokens
            if len(cache) >= 3:
                attn = mx.array([0.1] * (len(cache) - 3) + [0.3, 0.3, 0.3])
                attn = attn / mx.sum(attn)  # Normalize

            cache.append(k, v, attention_weights=attn)
        else:
            cache.append(k, v)

    info = cache.info()
    print(f"\nAfter adding 15 tokens:")
    print(f"  Current size: {info['current_size']}")
    print(f"  Max capacity: {info['max_capacity']}")
    print(f"  Heavy hitters: {info['heavy_size']}")
    print(f"  Recent window: {info['recent_size']}")
    print(f"  Total seen: {info['total_tokens_seen']}")
    print(f"  Eviction count: {info['eviction_count']}")

    if 'avg_attention_heavy' in info:
        print(f"  Avg attention (heavy): {info['avg_attention_heavy']:.4f}")
        print(f"  Avg attention (recent): {info['avg_attention_recent']:.4f}")

    assert info['current_size'] == 10, "Cache should be at max capacity"
    assert info['eviction_count'] == 5, "Should have evicted 5 tokens"

    print("\n✅ Basic functionality test passed")


def test_compression_quality_small():
    """Test H2O compression quality on small cache."""
    print("\n" + "=" * 60)
    print("Test 2: Small Scale Quality (T=100, t=32)")
    print("=" * 60)

    T, d, n = 100, 64, 20
    mx.random.seed(42)

    K = mx.random.normal((T, d))
    V = mx.random.normal((T, d))
    queries = mx.random.normal((n, d))

    results = test_h2o_quality(
        K, V, queries,
        max_capacity=32,
        recent_ratio=0.25
    )

    print(f"\nResults:")
    print(f"  Compression: {T} → {results['compressed_size']} ({results['compression_ratio']:.1f}x)")
    print(f"  Heavy hitters: {results['heavy_hitters']}")
    print(f"  Recent window: {results['recent_window']}")
    print(f"  Quality (cosine): {results['cosine_similarity']:.6f}")
    print(f"  MSE: {results['mse']:.6e}")
    print(f"  Attention to heavy: {results['attention_to_heavy']:.4f}")
    print(f"  Attention to recent: {results['attention_to_recent']:.4f}")

    # H2O target: cos > 0.90
    if results['cosine_similarity'] >= 0.90:
        print(f"\n✅ Quality test passed (cos={results['cosine_similarity']:.6f} >= 0.90)")
    else:
        print(f"\n⚠️ Quality below target (cos={results['cosine_similarity']:.6f} < 0.90)")


def test_compression_quality_medium():
    """Test H2O compression quality on medium cache."""
    print("\n" + "=" * 60)
    print("Test 3: Medium Scale Quality (T=1000, t=256)")
    print("=" * 60)

    T, d, n = 1000, 128, 50
    mx.random.seed(42)

    K = mx.random.normal((T, d))
    V = mx.random.normal((T, d))
    queries = mx.random.normal((n, d))

    results = test_h2o_quality(
        K, V, queries,
        max_capacity=256,
        recent_ratio=0.25
    )

    print(f"\nResults:")
    print(f"  Compression: {T} → {results['compressed_size']} ({results['compression_ratio']:.1f}x)")
    print(f"  Heavy hitters: {results['heavy_hitters']}")
    print(f"  Recent window: {results['recent_window']}")
    print(f"  Quality (cosine): {results['cosine_similarity']:.6f}")
    print(f"  MSE: {results['mse']:.6e}")
    print(f"  Attention to heavy: {results['attention_to_heavy']:.4f}")
    print(f"  Attention to recent: {results['attention_to_recent']:.4f}")

    if results['cosine_similarity'] >= 0.90:
        print(f"\n✅ Quality test passed (cos={results['cosine_similarity']:.6f} >= 0.90)")
    else:
        print(f"\n⚠️ Quality below target (cos={results['cosine_similarity']:.6f} < 0.90)")


def test_recent_ratio_impact():
    """Test impact of recent_ratio parameter."""
    print("\n" + "=" * 60)
    print("Test 4: Recent Ratio Impact")
    print("=" * 60)

    T, d, n = 200, 64, 20
    mx.random.seed(42)

    K = mx.random.normal((T, d))
    V = mx.random.normal((T, d))
    queries = mx.random.normal((n, d))

    # Test different recent_ratio values
    for recent_ratio in [0.1, 0.25, 0.5, 0.75]:
        results = test_h2o_quality(
            K, V, queries,
            max_capacity=64,
            recent_ratio=recent_ratio
        )

        print(f"\nrecent_ratio={recent_ratio:.2f}:")
        print(f"  Heavy: {results['heavy_hitters']}, Recent: {results['recent_window']}")
        print(f"  Quality: {results['cosine_similarity']:.6f}")
        print(f"  Attn to heavy: {results['attention_to_heavy']:.4f}")
        print(f"  Attn to recent: {results['attention_to_recent']:.4f}")

    print("\n✅ Recent ratio test completed")


def test_vs_streaming_llm():
    """Compare H2O vs StreamingLLM."""
    print("\n" + "=" * 60)
    print("Test 5: H2O vs StreamingLLM")
    print("=" * 60)

    T, d, n = 200, 64, 20
    mx.random.seed(42)

    K = mx.random.normal((T, d))
    V = mx.random.normal((T, d))
    queries = mx.random.normal((n, d))

    # Test H2O
    h2o_results = test_h2o_quality(
        K, V, queries,
        max_capacity=64,
        recent_ratio=0.25
    )

    # Test StreamingLLM (simulate)
    from flashmlx.cache.streaming_llm import test_streaming_llm_quality
    stream_results = test_streaming_llm_quality(
        K, V, queries,
        max_capacity=64,
        num_sinks=4
    )

    print(f"\nH2O:")
    print(f"  Quality: {h2o_results['cosine_similarity']:.6f}")
    print(f"  Heavy: {h2o_results['heavy_hitters']}, Recent: {h2o_results['recent_window']}")

    print(f"\nStreamingLLM:")
    print(f"  Quality: {stream_results['cosine_similarity']:.6f}")
    print(f"  Sinks: {stream_results['num_sinks']}, Window: {stream_results['window_size']}")

    if h2o_results['cosine_similarity'] > stream_results['cosine_similarity']:
        print(f"\n✅ H2O better than StreamingLLM (+{h2o_results['cosine_similarity'] - stream_results['cosine_similarity']:.4f})")
    elif h2o_results['cosine_similarity'] < stream_results['cosine_similarity']:
        print(f"\n⚠️ StreamingLLM better than H2O (+{stream_results['cosine_similarity'] - h2o_results['cosine_similarity']:.4f})")
    else:
        print(f"\n✅ H2O and StreamingLLM have similar quality")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("H2O (Heavy-Hitter Oracle) Compression Tests")
    print("=" * 60)

    test_basic_functionality()
    test_compression_quality_small()
    test_compression_quality_medium()
    test_recent_ratio_impact()
    test_vs_streaming_llm()

    print("\n" + "=" * 60)
    print("All Tests Completed")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
