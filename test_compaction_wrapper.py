#!/usr/bin/env python3
"""
Test CompactionWrapper basic functionality
"""

import mlx.core as mx
from flashmlx.compaction import AttentionMatchingWrapper


def test_basic_compression():
    print("="*70)
    print("Testing Compaction Wrapper")
    print("="*70)

    # Create dummy KV cache
    seq_len = 100
    head_dim = 64
    compression_ratio = 3.0

    keys = mx.random.normal((seq_len, head_dim))
    values = mx.random.normal((seq_len, head_dim))

    print(f"\nInput:")
    print(f"  Keys shape: {keys.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Compression ratio: {compression_ratio}x")

    # Create wrapper
    wrapper = AttentionMatchingWrapper(
        compression_ratio=compression_ratio,
        score_method='max',
        beta_method='nnls',
        c2_method='lsq',
    )

    # Compress
    print(f"\nCompressing...")
    C1, beta, C2 = wrapper.compress_kv_cache(keys, values, num_queries=50)

    print(f"\nOutput:")
    print(f"  C1 shape: {C1.shape}")
    print(f"  beta shape: {beta.shape}")
    print(f"  C2 shape: {C2.shape}")
    print(f"  Actual compression ratio: {seq_len / C1.shape[0]:.2f}x")

    # Test applying compacted attention
    print(f"\nTesting compacted attention...")
    query = mx.random.normal((head_dim,))
    output = wrapper.apply_compacted_attention(query, C1, beta, C2)
    print(f"  Query shape: {query.shape}")
    print(f"  Output shape: {output.shape}")

    # Verify output shape
    assert output.shape == (head_dim,), f"Expected shape ({head_dim},), got {output.shape}"

    print(f"\n{'='*70}")
    print("✓ Basic test passed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_basic_compression()
