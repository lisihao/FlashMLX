#!/usr/bin/env python3
"""
测试正确的 Attention Matching 实现是否能保持质量

对比原始 attention 输出和压缩后的输出
"""

import mlx.core as mx
from flashmlx.compaction import AttentionMatchingWrapper


def compute_attention_output(queries, keys, values):
    """
    计算标准 attention 输出

    Args:
        queries: (n_queries, head_dim)
        keys: (seq_len, head_dim)
        values: (seq_len, head_dim)

    Returns:
        output: (n_queries, head_dim)
    """
    # Compute attention scores
    scores = queries @ keys.T  # (n_queries, seq_len)

    # Scale
    head_dim = queries.shape[-1]
    scores = scores / mx.sqrt(mx.array(head_dim, dtype=scores.dtype))

    # Softmax
    attention_weights = mx.softmax(scores, axis=-1)  # (n_queries, seq_len)

    # Apply to values
    output = attention_weights @ values  # (n_queries, head_dim)

    return output


def test_attention_quality():
    print("="*70)
    print("Testing Correct Attention Matching Implementation")
    print("="*70)

    # Create test data
    seq_len = 100
    head_dim = 64
    n_queries = 20
    compression_ratio = 2.0  # Conservative 2x

    keys = mx.random.normal((seq_len, head_dim))
    values = mx.random.normal((seq_len, head_dim))
    queries = mx.random.normal((n_queries, head_dim))

    print(f"\nSetup:")
    print(f"  Seq len: {seq_len}")
    print(f"  Head dim: {head_dim}")
    print(f"  Num queries: {n_queries}")
    print(f"  Compression ratio: {compression_ratio}x")

    # Compute original attention output
    print(f"\nComputing original attention output...")
    original_output = compute_attention_output(queries, keys, values)
    print(f"  Original output shape: {original_output.shape}")

    # Compress KV cache
    print(f"\nCompressing KV cache...")
    wrapper = AttentionMatchingWrapper(
        compression_ratio=compression_ratio,
        score_method='max',
        beta_method='nnls',
        c2_method='lsq',
    )

    # Use the same queries for training (in real use, would be separate)
    C1, beta, C2 = wrapper.compress_kv_cache(
        keys, values,
        queries=queries,
    )

    print(f"  C1 shape: {C1.shape}")
    print(f"  beta shape: {beta.shape}")
    print(f"  C2 shape: {C2.shape}")
    print(f"  Actual compression: {seq_len / C1.shape[0]:.2f}x")

    # Compute compacted attention output
    print(f"\nComputing compacted attention output...")
    compacted_outputs = []
    for i in range(n_queries):
        query = queries[i]  # (head_dim,)
        output = wrapper.apply_compacted_attention(query, C1, beta, C2)
        compacted_outputs.append(output)

    compacted_output = mx.stack(compacted_outputs)  # (n_queries, head_dim)
    print(f"  Compacted output shape: {compacted_output.shape}")

    # Compute error metrics
    print(f"\nQuality Metrics:")

    # MSE
    mse = mx.mean((original_output - compacted_output) ** 2)
    print(f"  MSE: {float(mse):.6f}")

    # Cosine similarity (per query)
    cosine_sims = []
    for i in range(n_queries):
        orig = original_output[i]
        comp = compacted_output[i]

        # Normalize
        orig_norm = orig / (mx.linalg.norm(orig) + 1e-8)
        comp_norm = comp / (mx.linalg.norm(comp) + 1e-8)

        # Cosine similarity
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
    rel_error = mx.linalg.norm(original_output - compacted_output) / mx.linalg.norm(original_output)
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
    print("✓ Test completed")
    print(f"{'='*70}")

    return avg_cosine >= 0.80


if __name__ == "__main__":
    success = test_attention_quality()
    exit(0 if success else 1)
