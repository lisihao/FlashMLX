#!/usr/bin/env python3
"""
简化质量测试 - 对比有/无 Attention Matching 的 attention 输出质量

测试方法：
1. 创建长序列的 KV cache
2. 对比压缩前后的 attention 输出
3. 计算质量指标（Cosine Similarity, MSE, Relative Error）

测试场景：
- 不同序列长度（512, 1024, 2048）
- 不同压缩比（1.5x, 2.0x, 2.5x, 3.0x）
"""

import mlx.core as mx
from flashmlx.cache.attention_matching_compressor_v2 import AttentionMatchingCompressorV2
import numpy as np


def compute_attention_output(queries, keys, values):
    """标准 attention"""
    scores = queries @ keys.T
    head_dim = queries.shape[-1]
    scores = scores / mx.sqrt(mx.array(head_dim, dtype=scores.dtype))
    attention_weights = mx.softmax(scores, axis=-1)
    output = attention_weights @ values
    return output


def test_quality_at_scale(
    seq_len: int,
    compression_ratio: float,
    num_heads: int = 8,
    head_dim: int = 64,
    num_test_queries: int = 50
):
    """测试指定规模和压缩比的质量"""

    print(f"\n{'='*70}")
    print(f"测试配置: seq_len={seq_len}, compression_ratio={compression_ratio}x")
    print(f"{'='*70}")

    # 创建数据
    batch_size = 1
    keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
    values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

    # 创建压缩器
    compressor = AttentionMatchingCompressorV2(
        compression_ratio=compression_ratio,
        score_method='max',
        beta_method='nnls',
        c2_method='lsq',
        num_queries=100
    )

    # 压缩
    print(f"压缩 KV cache...")
    compressed_keys, compressed_values = compressor.compress_kv_cache(
        layer_idx=0,
        kv_cache=(keys, values)
    )

    compressed_seq_len = compressed_keys.shape[2]
    print(f"✓ 压缩完成: {seq_len} → {compressed_seq_len} tokens ({compression_ratio:.1f}x)")

    # 测试 attention 质量（per-head）
    print(f"\n测试 attention 输出质量...")

    all_cosine_sims = []
    all_mse = []
    all_rel_errors = []

    for head_idx in range(num_heads):
        # Original keys/values for this head
        orig_keys = mx.squeeze(keys[:, head_idx, :, :], axis=0)  # (seq_len, head_dim)
        orig_values = mx.squeeze(values[:, head_idx, :, :], axis=0)

        # Compressed keys/values
        comp_keys = mx.squeeze(compressed_keys[:, head_idx, :, :], axis=0)  # (t, head_dim)
        comp_values = mx.squeeze(compressed_values[:, head_idx, :, :], axis=0)

        # Create test queries
        test_queries = mx.random.normal((num_test_queries, head_dim))

        # Compute original attention
        original_output = compute_attention_output(test_queries, orig_keys, orig_values)

        # Compute compressed attention (WITH beta!)
        scores = test_queries @ comp_keys.T
        scores = scores / mx.sqrt(mx.array(head_dim, dtype=scores.dtype))

        # Apply beta compensation
        scores_with_beta = compressor.apply_beta_compensation(
            layer_idx=0,
            head_idx=head_idx,
            attention_scores=mx.expand_dims(scores, axis=0)
        )
        scores_with_beta = mx.squeeze(scores_with_beta, axis=0)

        attention_weights = mx.softmax(scores_with_beta, axis=-1)
        compressed_output = attention_weights @ comp_values

        # Compute metrics per head
        mse = float(mx.mean((original_output - compressed_output) ** 2))
        rel_error = float(mx.linalg.norm(original_output - compressed_output) / mx.linalg.norm(original_output))

        # Cosine similarity per query
        head_cosine_sims = []
        for i in range(num_test_queries):
            orig = original_output[i]
            comp = compressed_output[i]
            orig_norm = orig / (mx.linalg.norm(orig) + 1e-8)
            comp_norm = comp / (mx.linalg.norm(comp) + 1e-8)
            cos_sim = float(mx.sum(orig_norm * comp_norm))
            head_cosine_sims.append(cos_sim)

        avg_cosine = sum(head_cosine_sims) / len(head_cosine_sims)

        all_cosine_sims.extend(head_cosine_sims)
        all_mse.append(mse)
        all_rel_errors.append(rel_error)

    # 总体统计
    overall_cosine = sum(all_cosine_sims) / len(all_cosine_sims)
    overall_mse = sum(all_mse) / len(all_mse)
    overall_rel_error = sum(all_rel_errors) / len(all_rel_errors)

    print(f"\n质量指标:")
    print(f"  Cosine Similarity (平均): {overall_cosine:.4f}")
    print(f"  MSE (平均): {overall_mse:.6f}")
    print(f"  Relative Error (平均): {overall_rel_error:.4f}")

    # 质量评级
    if overall_cosine >= 0.95:
        quality = "✅ EXCELLENT (≥0.95)"
        token_overlap_est = ">90%"
    elif overall_cosine >= 0.90:
        quality = "✅ GOOD (≥0.90)"
        token_overlap_est = "70-90%"
    elif overall_cosine >= 0.80:
        quality = "⚠️  ACCEPTABLE (≥0.80)"
        token_overlap_est = "50-70%"
    else:
        quality = "❌ POOR (<0.80)"
        token_overlap_est = "<50%"

    print(f"\n质量评级: {quality}")
    print(f"预估 Token Overlap: {token_overlap_est}")

    return {
        "seq_len": seq_len,
        "compression_ratio": compression_ratio,
        "cosine_similarity": overall_cosine,
        "mse": overall_mse,
        "relative_error": overall_rel_error,
        "quality": quality,
        "token_overlap_est": token_overlap_est
    }


def main():
    print("="*70)
    print("简化质量测试 - Attention Matching")
    print("="*70)

    # 测试矩阵
    test_configs = [
        # (seq_len, compression_ratio)
        (512, 2.0),
        (1024, 2.0),
        (2048, 2.0),
        (1024, 1.5),
        (1024, 2.5),
        (1024, 3.0),
    ]

    results = []

    for seq_len, compression_ratio in test_configs:
        result = test_quality_at_scale(
            seq_len=seq_len,
            compression_ratio=compression_ratio
        )
        results.append(result)

    # 总结
    print(f"\n\n{'='*70}")
    print("总结")
    print(f"{'='*70}")

    print(f"\n{'Seq Len':<10} {'Ratio':<8} {'Cosine':<10} {'MSE':<12} {'Rel Err':<10} {'质量'}")
    print("-" * 75)

    for result in results:
        print(f"{result['seq_len']:<10} {result['compression_ratio']:<8.1f} {result['cosine_similarity']:<10.4f} {result['mse']:<12.6f} {result['relative_error']:<10.4f} {result['quality']}")

    # 关键结论
    print(f"\n{'='*70}")
    print("关键结论")
    print(f"{'='*70}")

    # 找出 2.0x 压缩比的平均质量
    ratio_2x_results = [r for r in results if r['compression_ratio'] == 2.0]
    if ratio_2x_results:
        avg_cosine = sum(r['cosine_similarity'] for r in ratio_2x_results) / len(ratio_2x_results)
        print(f"\n2.0x 压缩比平均 Cosine Similarity: {avg_cosine:.4f}")

        if avg_cosine >= 0.90:
            print("✅ 预计文本生成 Token Overlap: ≥70% (超过目标 50%)")
        elif avg_cosine >= 0.80:
            print("⚠️  预计文本生成 Token Overlap: 50-70% (达到目标 50%)")
        else:
            print("❌ 预计文本生成 Token Overlap: <50% (未达到目标)")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
