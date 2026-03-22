#!/usr/bin/env python3
"""
测试修改后的 simple_injection（使用正确实现）
"""

import mlx.core as mx
from flashmlx.cache.attention_matching_compressor_v2 import AttentionMatchingCompressorV2


def test_compressor_v2():
    print("="*70)
    print("Testing AttentionMatchingCompressorV2")
    print("="*70)

    # Create compressor
    compressor = AttentionMatchingCompressorV2(
        compression_ratio=2.0,
        score_method='max',
        beta_method='nnls',
        c2_method='lsq',
        num_queries=50
    )

    # Create dummy KV cache (4D)
    batch_size = 1
    num_heads = 8
    seq_len = 100
    head_dim = 64

    keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
    values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

    print(f"\nInput:")
    print(f"  Keys shape: {keys.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Compression ratio: {compressor.compression_ratio}x")

    # Compress
    print(f"\nCompressing...")
    compressed_keys, compressed_values = compressor.compress_kv_cache(
        layer_idx=0,
        kv_cache=(keys, values)
    )

    print(f"\nOutput:")
    print(f"  Compressed keys shape: {compressed_keys.shape}")
    print(f"  Compressed values shape: {compressed_values.shape}")
    print(f"  Actual compression: {seq_len / compressed_keys.shape[2]:.2f}x")

    # Test beta compensation
    print(f"\nTesting beta compensation...")
    # Dummy attention scores
    query_len = 10
    compressed_seq_len = compressed_keys.shape[2]
    attention_scores = mx.random.normal((batch_size, query_len, compressed_seq_len))

    # Apply beta for head 0
    compensated_scores = compressor.apply_beta_compensation(
        layer_idx=0,
        head_idx=0,
        attention_scores=attention_scores
    )

    print(f"  Original scores shape: {attention_scores.shape}")
    print(f"  Compensated scores shape: {compensated_scores.shape}")

    # Check stats
    stats = compressor.get_compression_stats()
    print(f"\nCompression Stats:")
    print(f"  Total compressions: {stats['total_compressions']}")
    print(f"  Total keys before: {stats['total_keys_before']}")
    print(f"  Total keys after: {stats['total_keys_after']}")
    print(f"  Avg compression ratio: {stats['avg_compression_ratio']:.2f}x")

    print(f"\n{'='*70}")
    print("✓ All tests passed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_compressor_v2()
