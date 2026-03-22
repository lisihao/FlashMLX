"""
Correct Quality Evaluation for Self-Attention Compression

论文场景：Self-Attention (keys 查询自己)
正确测试：用所有 keys 作为查询，评估重建质量
"""
import mlx.core as mx
import numpy as np
import time

from flashmlx.compaction.offline_compressor import offline_compress_kv_cache


def compute_attention(queries, keys, values):
    """Standard self-attention"""
    scores = queries @ keys.T
    scale = 1.0 / mx.sqrt(mx.array(queries.shape[-1], dtype=mx.float32))
    scores = scores * scale
    weights = mx.softmax(scores, axis=-1)
    output = weights @ values
    return output


def compute_compressed_attention(queries, C1, beta, C2):
    """Compressed attention"""
    scores = queries @ C1.T
    scale = 1.0 / mx.sqrt(mx.array(queries.shape[-1], dtype=mx.float32))
    scores = scores * scale + beta[None, :]
    weights = mx.softmax(scores, axis=-1)
    output = weights @ C2
    return output


def evaluate_self_attention_compression(
    seq_len: int = 2000,
    head_dim: int = 64,
    compression_ratio: int = 4,
    num_queries: int = 100,
    use_omp: bool = False
):
    """
    评估自注意力压缩质量

    正确方法：用所有 keys 作为查询（自注意力场景）
    """
    print(f"Configuration:")
    print(f"  Sequence: {seq_len} tokens")
    print(f"  Compression: {compression_ratio}x")
    print(f"  Num queries for compression: {num_queries}")
    print(f"  Use OMP: {use_omp}")
    print()

    # Generate synthetic KV cache
    print("Step 1: Generate KV cache...")
    keys = mx.random.normal((seq_len, head_dim))
    values = mx.random.normal((seq_len, head_dim))
    print(f"  ✓ K, V: ({seq_len}, {head_dim})")
    print()

    # Original self-attention (Q = K)
    print("Step 2: Original self-attention (Q = K)...")
    t0 = time.time()
    original_output = compute_attention(keys, keys, values)
    t_orig = time.time() - t0
    print(f"  ✓ Output: {original_output.shape}")
    print(f"  ✓ Time: {t_orig:.4f}s")
    print()

    # Compress
    print("Step 3: Offline compression...")
    t0 = time.time()

    K = keys[None, None, :, :]
    V = values[None, None, :, :]

    C1, beta, C2 = offline_compress_kv_cache(
        K, V,
        compression_ratio=compression_ratio,
        num_queries=num_queries,
        use_omp=use_omp,
        verbose=False
    )

    C1 = C1[0, 0]
    beta = beta[0, 0]
    C2 = C2[0, 0]

    t_comp = time.time() - t0
    budget = seq_len // compression_ratio

    print(f"  ✓ Compressed: {seq_len} -> {budget} tokens")
    print(f"  ✓ Time: {t_comp:.4f}s")
    print()

    # Compressed self-attention (Q = K)
    print("Step 4: Compressed self-attention (Q = K)...")
    t0 = time.time()
    compressed_output = compute_compressed_attention(keys, C1, beta, C2)
    t_comp_attn = time.time() - t0
    print(f"  ✓ Output: {compressed_output.shape}")
    print(f"  ✓ Time: {t_comp_attn:.4f}s")
    print()

    # Quality metrics
    print("=" * 70)
    print("Quality Metrics (Self-Attention)")
    print("=" * 70)

    # Cosine similarity
    orig_flat = mx.reshape(original_output, (-1,))
    comp_flat = mx.reshape(compressed_output, (-1,))

    dot = float(mx.sum(orig_flat * comp_flat))
    norm_o = float(mx.sqrt(mx.sum(orig_flat ** 2)))
    norm_c = float(mx.sqrt(mx.sum(comp_flat ** 2)))
    cos_sim = dot / (norm_o * norm_c) if (norm_o > 0 and norm_c > 0) else 0.0

    # L2 distance
    diff = original_output - compressed_output
    l2_dist = float(mx.sqrt(mx.sum(diff ** 2)))
    rel_error = (l2_dist / norm_o) * 100 if norm_o > 0 else 0.0

    print(f"Cosine Similarity: {cos_sim:.6f}")
    print(f"L2 Distance: {l2_dist:.2f}")
    print(f"Relative Error: {rel_error:.2f}%")
    print()

    # Memory savings
    orig_size = keys.size + values.size
    comp_size = C1.size + beta.size + C2.size
    actual_ratio = orig_size / comp_size
    memory_saved = ((orig_size - comp_size) / orig_size) * 100

    print(f"Compression Ratio: {actual_ratio:.2f}x")
    print(f"Memory Saved: {memory_saved:.1f}%")
    print()

    # Pass criteria
    print("=" * 70)
    print("Evaluation Result")
    print("=" * 70)

    cos_pass = cos_sim >= 0.95
    error_pass = rel_error <= 10.0
    ratio_pass = actual_ratio >= (compression_ratio * 0.95)

    print(f"  Cosine Similarity >= 0.95: {'✅ PASS' if cos_pass else '❌ FAIL'} ({cos_sim:.4f})")
    print(f"  Relative Error <= 10%: {'✅ PASS' if error_pass else '❌ FAIL'} ({rel_error:.2f}%)")
    print(f"  Compression Ratio >= {compression_ratio}x: {'✅ PASS' if ratio_pass else '❌ FAIL'} ({actual_ratio:.2f}x)")
    print()

    if cos_pass and error_pass and ratio_pass:
        print("✅ ALL TESTS PASSED!")
        print("   Compression quality is excellent for self-attention.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("   Quality may need improvement.")

    print("=" * 70)

    return {
        'cosine_similarity': cos_sim,
        'l2_distance': l2_dist,
        'relative_error_pct': rel_error,
        'compression_ratio': actual_ratio,
        'memory_saved_pct': memory_saved
    }


def main():
    """Run comprehensive self-attention evaluation"""
    print("=" * 70)
    print("Self-Attention Compression Quality Evaluation")
    print("=" * 70)
    print()
    print("Testing compression for self-attention (Q = K)")
    print("This is the correct scenario for the paper's algorithm.")
    print()

    # Test configurations
    configs = [
        {'name': 'Small (500 tokens)', 'seq_len': 500, 'ratio': 4, 'queries': 50},
        {'name': 'Medium (2000 tokens)', 'seq_len': 2000, 'ratio': 4, 'queries': 100},
        {'name': 'Large (5000 tokens)', 'seq_len': 5000, 'ratio': 4, 'queries': 100},
        {'name': 'High Compression (8x)', 'seq_len': 2000, 'ratio': 8, 'queries': 100},
        {'name': 'With OMP', 'seq_len': 1000, 'ratio': 4, 'queries': 50, 'use_omp': True}
    ]

    results = []

    for i, config in enumerate(configs):
        print(f"[Test {i+1}/{len(configs)}] {config['name']}")
        print("-" * 70)

        metrics = evaluate_self_attention_compression(
            seq_len=config['seq_len'],
            head_dim=64,
            compression_ratio=config['ratio'],
            num_queries=config['queries'],
            use_omp=config.get('use_omp', False)
        )

        metrics['config_name'] = config['name']
        results.append(metrics)
        print()

    # Summary table
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"{'Test':<30} {'CosSim':<10} {'RelErr%':<10} {'Ratio':<10} {'MemSave%':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['config_name']:<30} "
              f"{r['cosine_similarity']:<10.6f} "
              f"{r['relative_error_pct']:<10.2f} "
              f"{r['compression_ratio']:<10.2f} "
              f"{r['memory_saved_pct']:<10.1f}")

    print()

    # Overall pass/fail
    all_pass = all(
        r['cosine_similarity'] >= 0.95 and
        r['relative_error_pct'] <= 10.0
        for r in results
    )

    if all_pass:
        print("✅ ALL TESTS PASSED!")
        print("   Phase 2 Quality Validation: SUCCESS")
    else:
        print("❌ SOME TESTS FAILED!")
        print("   Phase 2 Quality Validation: NEEDS IMPROVEMENT")

    print("=" * 70)


if __name__ == "__main__":
    main()
