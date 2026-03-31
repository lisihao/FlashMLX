#!/usr/bin/env python3
"""
测试不同 num_queries 对质量的影响
"""

import mlx.core as mx
from flashmlx.cache.attention_matching_compressor_v2 import AttentionMatchingCompressorV2


def compute_attention_output(queries, keys, values):
    """标准 attention"""
    scores = queries @ keys.T
    head_dim = queries.shape[-1]
    scores = scores / mx.sqrt(mx.array(head_dim, dtype=scores.dtype))
    attention_weights = mx.softmax(scores, axis=-1)
    output = attention_weights @ values
    return output


def test_num_queries(num_queries):
    """测试指定 num_queries"""
    # Setup
    batch_size = 1
    num_heads = 4
    seq_len = 100  # 增加到 100
    head_dim = 64  # 增加到 64
    compression_ratio = 2.0

    # Create compressor
    compressor = AttentionMatchingCompressorV2(
        compression_ratio=compression_ratio,
        score_method='max',
        beta_method='nnls',
        c2_method='lsq',
        num_queries=num_queries  # 测试不同值
    )

    # Create KV cache
    keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
    values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

    # Compress
    compressed_keys, compressed_values = compressor.compress_kv_cache(
        layer_idx=0,
        kv_cache=(keys, values)
    )

    # Test attention (per-head)
    head_idx = 0
    layer_idx = 0

    orig_keys = mx.squeeze(keys[:, head_idx, :, :], axis=0)
    orig_values = mx.squeeze(values[:, head_idx, :, :], axis=0)

    comp_keys = mx.squeeze(compressed_keys[:, head_idx, :, :], axis=0)
    comp_values = mx.squeeze(compressed_values[:, head_idx, :, :], axis=0)

    # Test queries
    n_test_queries = 20
    test_queries = mx.random.normal((n_test_queries, head_dim))

    # Original attention
    original_output = compute_attention_output(test_queries, orig_keys, orig_values)

    # Compressed attention (with beta)
    scores = test_queries @ comp_keys.T
    scores = scores / mx.sqrt(mx.array(head_dim, dtype=scores.dtype))
    scores_with_beta = compressor.apply_beta_compensation(
        layer_idx=layer_idx,
        head_idx=head_idx,
        attention_scores=mx.expand_dims(scores, axis=0)
    )
    scores_with_beta = mx.squeeze(scores_with_beta, axis=0)
    attention_weights = mx.softmax(scores_with_beta, axis=-1)
    compressed_output = attention_weights @ comp_values

    # Compute metrics
    cosine_sims = []
    for i in range(n_test_queries):
        orig = original_output[i]
        comp = compressed_output[i]
        orig_norm = orig / (mx.linalg.norm(orig) + 1e-8)
        comp_norm = comp / (mx.linalg.norm(comp) + 1e-8)
        cos_sim = mx.sum(orig_norm * comp_norm)
        cosine_sims.append(float(cos_sim))

    avg_cosine = sum(cosine_sims) / len(cosine_sims)
    return avg_cosine


if __name__ == "__main__":
    print("="*70)
    print("Testing num_queries Impact on Quality")
    print("="*70)

    test_values = [50, 100, 200, 300]

    for num_queries in test_values:
        print(f"\nnum_queries = {num_queries}")

        # Run 3 times and average
        results = []
        for run in range(3):
            cosine = test_num_queries(num_queries)
            results.append(cosine)

        avg = sum(results) / len(results)
        print(f"  Cosine similarity: {avg:.4f} (avg of 3 runs)")

        if avg >= 0.95:
            level = "EXCELLENT"
        elif avg >= 0.90:
            level = "GOOD"
        elif avg >= 0.80:
            level = "ACCEPTABLE"
        else:
            level = "POOR"
        print(f"  Level: {level}")

    print(f"\n{'='*70}")
