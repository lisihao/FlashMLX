#!/usr/bin/env python3
"""
简单的正确性验证 - 不需要加载大模型
验证 CompressorV2 的 beta 是否正确计算
"""

import mlx.core as mx
from flashmlx.cache.attention_matching_compressor_v2 import AttentionMatchingCompressorV2


def compute_attention_output(queries, keys, values):
    """标准 attention"""
    # scores = Q @ K.T / sqrt(d)
    scores = queries @ keys.T
    head_dim = queries.shape[-1]
    scores = scores / mx.sqrt(mx.array(head_dim, dtype=scores.dtype))

    # softmax
    attention_weights = mx.softmax(scores, axis=-1)

    # output = weights @ V
    output = attention_weights @ values
    return output


def test_correctness():
    print("="*70)
    print("Correctness Test - Beta Verification")
    print("="*70)

    # Create test data
    batch_size = 1
    num_heads = 4
    seq_len = 50
    head_dim = 32
    compression_ratio = 2.0

    # Create compressor
    compressor = AttentionMatchingCompressorV2(
        compression_ratio=compression_ratio,
        score_method='max',
        beta_method='nnls',
        c2_method='lsq',
        num_queries=50
    )

    print(f"\nSetup:")
    print(f"  Batch: {batch_size}, Heads: {num_heads}, Seq: {seq_len}, HeadDim: {head_dim}")
    print(f"  Compression ratio: {compression_ratio}x")

    # Create KV cache (4D)
    keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
    values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

    # Compress
    print(f"\nCompressing...")
    compressed_keys, compressed_values = compressor.compress_kv_cache(
        layer_idx=0,
        kv_cache=(keys, values)
    )

    compressed_seq_len = compressed_keys.shape[2]
    print(f"  Compressed from {seq_len} to {compressed_seq_len} tokens")

    # Test attention computation (per-head)
    print(f"\nTesting attention computation...")

    # Take one head
    head_idx = 0
    layer_idx = 0

    # Original keys/values for this head
    orig_keys = mx.squeeze(keys[:, head_idx, :, :], axis=0)  # (seq_len, head_dim)
    orig_values = mx.squeeze(values[:, head_idx, :, :], axis=0)

    # Compressed keys/values
    comp_keys = mx.squeeze(compressed_keys[:, head_idx, :, :], axis=0)  # (t, head_dim)
    comp_values = mx.squeeze(compressed_values[:, head_idx, :, :], axis=0)

    # Create test queries
    n_queries = 10
    queries = mx.random.normal((n_queries, head_dim))

    # Compute original attention
    original_output = compute_attention_output(queries, orig_keys, orig_values)

    # Compute compressed attention (WITH beta!)
    scores = queries @ comp_keys.T  # (n_queries, t)
    scores = scores / mx.sqrt(mx.array(head_dim, dtype=scores.dtype))

    # Apply beta compensation
    scores_with_beta = compressor.apply_beta_compensation(
        layer_idx=layer_idx,
        head_idx=head_idx,
        attention_scores=mx.expand_dims(scores, axis=0)  # Add batch dim
    )
    scores_with_beta = mx.squeeze(scores_with_beta, axis=0)  # Remove batch dim

    # Softmax + apply to values
    attention_weights = mx.softmax(scores_with_beta, axis=-1)
    compressed_output = attention_weights @ comp_values

    # Compare quality
    print(f"\nQuality Metrics:")

    # MSE
    mse = mx.mean((original_output - compressed_output) ** 2)
    print(f"  MSE: {float(mse):.6f}")

    # Cosine similarity (per query)
    cosine_sims = []
    for i in range(n_queries):
        orig = original_output[i]
        comp = compressed_output[i]

        orig_norm = orig / (mx.linalg.norm(orig) + 1e-8)
        comp_norm = comp / (mx.linalg.norm(comp) + 1e-8)

        cos_sim = mx.sum(orig_norm * comp_norm)
        cosine_sims.append(float(cos_sim))

    avg_cosine = sum(cosine_sims) / len(cosine_sims)
    min_cosine = min(cosine_sims)
    max_cosine = max(cosine_sims)

    print(f"  Cosine similarity:")
    print(f"    Average: {avg_cosine:.4f}")
    print(f"    Min: {min_cosine:.4f}")
    print(f"    Max: {max_cosine:.4f}")

    # Relative error
    rel_error = mx.linalg.norm(original_output - compressed_output) / mx.linalg.norm(original_output)
    print(f"  Relative error: {float(rel_error):.4f}")

    # Quality check
    print(f"\nQuality Check:")
    if avg_cosine >= 0.95:
        print(f"  ✅ EXCELLENT (avg cosine >= 0.95)")
    elif avg_cosine >= 0.90:
        print(f"  ✅ GOOD (avg cosine >= 0.90)")
    elif avg_cosine >= 0.80:
        print(f"  ⚠️  ACCEPTABLE (avg cosine >= 0.80)")
    else:
        print(f"  ❌ POOR (avg cosine < 0.80)")

    print(f"\n{'='*70}")

    return avg_cosine >= 0.80


if __name__ == "__main__":
    success = test_correctness()
    exit(0 if success else 1)
