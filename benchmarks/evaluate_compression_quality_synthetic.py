"""
Offline Compression Quality Evaluation - Synthetic Data

使用合成 KV cache 测试压缩质量（数学正确性）

评估指标：
1. Attention Output Similarity (余弦相似度)
2. Reconstruction Error (L2 距离)
3. Memory Savings
4. Compression Ratio

这个方法不依赖真实模型，但可以验证压缩算法的数学正确性。
"""
import mlx.core as mx
import numpy as np
import time

from flashmlx.compaction.offline_compressor import offline_compress_kv_cache
from flashmlx.cache import create_compacted_cache_list


def compute_attention_output(queries, keys, values):
    """
    计算标准 attention 输出

    Args:
        queries: (num_queries, head_dim)
        keys: (seq_len, head_dim)
        values: (seq_len, head_dim)

    Returns:
        Attention output (num_queries, head_dim)
    """
    # Q @ K^T
    scores = queries @ keys.T  # (num_queries, seq_len)

    # Scale
    scale = 1.0 / mx.sqrt(mx.array(queries.shape[-1], dtype=mx.float32))
    scores = scores * scale

    # Softmax
    weights = mx.softmax(scores, axis=-1)

    # Weighted sum of values
    output = weights @ values

    return output


def compute_compressed_attention_output(queries, C1, beta, C2):
    """
    计算压缩 attention 输出

    Args:
        queries: (num_queries, head_dim)
        C1: Compressed keys (budget, head_dim)
        beta: Bias (budget,)
        C2: Compressed values (budget, head_dim)

    Returns:
        Compressed attention output (num_queries, head_dim)
    """
    # Q @ C1^T
    scores = queries @ C1.T  # (num_queries, budget)

    # Scale
    scale = 1.0 / mx.sqrt(mx.array(queries.shape[-1], dtype=mx.float32))
    scores = scores * scale

    # Add bias
    scores = scores + beta[None, :]  # (num_queries, budget)

    # Softmax
    weights = mx.softmax(scores, axis=-1)

    # Weighted sum of compressed values
    output = weights @ C2

    return output


def cosine_similarity(a, b):
    """
    计算余弦相似度

    Args:
        a, b: Arrays of same shape

    Returns:
        Cosine similarity (0-1, higher is better)
    """
    # Flatten
    a_flat = mx.reshape(a, (-1,))
    b_flat = mx.reshape(b, (-1,))

    # Cosine similarity
    dot_product = float(mx.sum(a_flat * b_flat))
    norm_a = float(mx.sqrt(mx.sum(a_flat ** 2)))
    norm_b = float(mx.sqrt(mx.sum(b_flat ** 2)))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)

    return similarity


def l2_distance(a, b):
    """
    计算 L2 距离

    Args:
        a, b: Arrays of same shape

    Returns:
        L2 distance (lower is better)
    """
    diff = a - b
    distance = float(mx.sqrt(mx.sum(diff ** 2)))

    return distance


def evaluate_single_head_quality(
    seq_len: int = 2000,
    head_dim: int = 64,
    compression_ratio: int = 4,
    num_queries: int = 100,
    num_test_queries: int = 50,
    use_omp: bool = False
):
    """
    评估单头压缩质量

    Args:
        seq_len: Sequence length
        head_dim: Head dimension
        compression_ratio: Target compression ratio
        num_queries: Number of queries for compression
        num_test_queries: Number of test queries for evaluation
        use_omp: Use OMP refinement

    Returns:
        Quality metrics dict
    """
    print(f"  Sequence: {seq_len} tokens, dim={head_dim}")
    print(f"  Compression: {compression_ratio}x")
    print(f"  Query generation: {num_queries} queries")
    print(f"  Use OMP: {use_omp}")
    print()

    # Generate synthetic KV cache
    print("  1. Generating synthetic KV cache...")
    keys = mx.random.normal((seq_len, head_dim))
    values = mx.random.normal((seq_len, head_dim))
    print(f"     ✓ Generated")

    # Generate test queries (sample from keys, as in paper)
    print("  2. Generating test queries (from keys)...")
    # Randomly sample some keys as test queries
    test_indices = np.random.choice(seq_len, size=num_test_queries, replace=False)
    test_queries = keys[mx.array(test_indices)]  # Convert indices to MLX array
    print(f"     ✓ Sampled {num_test_queries} test queries from keys")

    # Compute original attention output
    print("  3. Computing original attention...")
    t0 = time.time()
    original_output = compute_attention_output(test_queries, keys, values)
    t_orig = time.time() - t0
    print(f"     ✓ Computed in {t_orig:.4f}s")

    # Offline compression
    print("  4. Offline compression...")
    t0 = time.time()

    # Wrap for multi-head format (B=1, n_heads=1)
    K = keys[None, None, :, :]  # (1, 1, seq_len, head_dim)
    V = values[None, None, :, :]

    C1, beta, C2 = offline_compress_kv_cache(
        K, V,
        compression_ratio=compression_ratio,
        num_queries=num_queries,
        use_omp=use_omp,
        verbose=False
    )

    # Unwrap
    C1 = C1[0, 0]  # (budget, head_dim)
    beta = beta[0, 0]  # (budget,)
    C2 = C2[0, 0]  # (budget, head_dim)

    t_comp = time.time() - t0
    budget = seq_len // compression_ratio

    print(f"     ✓ Compressed {seq_len} -> {budget} tokens in {t_comp:.4f}s")

    # Compute compressed attention output
    print("  5. Computing compressed attention...")
    t0 = time.time()
    compressed_output = compute_compressed_attention_output(test_queries, C1, beta, C2)
    t_comp_attn = time.time() - t0
    print(f"     ✓ Computed in {t_comp_attn:.4f}s")

    # Evaluate quality
    print("  6. Evaluating quality...")

    # Cosine similarity
    cos_sim = cosine_similarity(original_output, compressed_output)

    # L2 distance
    l2_dist = l2_distance(original_output, compressed_output)

    # Relative error
    output_norm = float(mx.sqrt(mx.sum(original_output ** 2)))
    relative_error = (l2_dist / output_norm) * 100 if output_norm > 0 else 0.0

    # Memory savings
    orig_size = keys.size + values.size
    comp_size = C1.size + beta.size + C2.size
    actual_ratio = orig_size / comp_size
    memory_saved_pct = ((orig_size - comp_size) / orig_size) * 100

    metrics = {
        'cosine_similarity': cos_sim,
        'l2_distance': l2_dist,
        'relative_error_pct': relative_error,
        'original_size': orig_size,
        'compressed_size': comp_size,
        'compression_ratio': actual_ratio,
        'memory_saved_pct': memory_saved_pct,
        'compression_time': t_comp,
        'original_attn_time': t_orig,
        'compressed_attn_time': t_comp_attn
    }

    print(f"     Cosine Similarity: {cos_sim:.4f}")
    print(f"     L2 Distance: {l2_dist:.2f}")
    print(f"     Relative Error: {relative_error:.2f}%")
    print(f"     Compression Ratio: {actual_ratio:.2f}x")
    print(f"     Memory Saved: {memory_saved_pct:.1f}%")
    print()

    return metrics


def run_comprehensive_evaluation():
    """
    Run comprehensive evaluation across different settings
    """
    print("=" * 70)
    print("Offline KV Cache Compression - Quality Evaluation (Synthetic Data)")
    print("=" * 70)
    print()

    print("This test evaluates the mathematical correctness of compression")
    print("using synthetic KV caches (not real model generation).")
    print()

    # Test configurations
    configs = [
        {
            'name': 'Small Cache (500 tokens)',
            'seq_len': 500,
            'head_dim': 64,
            'compression_ratio': 4,
            'num_queries': 50,
            'use_omp': False
        },
        {
            'name': 'Medium Cache (2000 tokens)',
            'seq_len': 2000,
            'head_dim': 64,
            'compression_ratio': 4,
            'num_queries': 100,
            'use_omp': False
        },
        {
            'name': 'Large Cache (5000 tokens)',
            'seq_len': 5000,
            'head_dim': 64,
            'compression_ratio': 4,
            'num_queries': 100,
            'use_omp': False
        },
        {
            'name': 'High Compression (8x)',
            'seq_len': 2000,
            'head_dim': 64,
            'compression_ratio': 8,
            'num_queries': 100,
            'use_omp': False
        },
        {
            'name': 'With OMP Refinement',
            'seq_len': 1000,
            'head_dim': 64,
            'compression_ratio': 4,
            'num_queries': 50,
            'use_omp': True
        }
    ]

    results = []

    for i, config in enumerate(configs):
        print(f"[Test {i+1}/{len(configs)}] {config['name']}")
        print("-" * 70)

        metrics = evaluate_single_head_quality(
            seq_len=config['seq_len'],
            head_dim=config['head_dim'],
            compression_ratio=config['compression_ratio'],
            num_queries=config['num_queries'],
            num_test_queries=50,
            use_omp=config['use_omp']
        )

        metrics['config_name'] = config['name']
        results.append(metrics)

        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    print(f"{'Test':<30} {'CosSim':<10} {'RelErr%':<10} {'Ratio':<10} {'MemSave%':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['config_name']:<30} "
              f"{r['cosine_similarity']:<10.4f} "
              f"{r['relative_error_pct']:<10.2f} "
              f"{r['compression_ratio']:<10.2f} "
              f"{r['memory_saved_pct']:<10.1f}")

    print()
    print("Pass Criteria:")
    print()

    # Check pass criteria
    all_pass = True

    for r in results:
        cos_pass = r['cosine_similarity'] >= 0.95
        error_pass = r['relative_error_pct'] <= 10.0

        # Extract expected ratio if specified
        expected_ratio = 4.0  # default
        if 'x)' in r['config_name']:
            try:
                ratio_str = r['config_name'].split('(')[1].split('x')[0].strip()
                expected_ratio = float(ratio_str)
            except:
                pass

        ratio_pass = r['compression_ratio'] >= (expected_ratio * 0.95)  # Allow 5% tolerance

        test_pass = cos_pass and error_pass and ratio_pass

        status = "✅ PASS" if test_pass else "❌ FAIL"
        print(f"  {r['config_name']}: {status}")

        if not test_pass:
            all_pass = False
            print(f"    CosSim: {r['cosine_similarity']:.4f} ({'✅' if cos_pass else '❌ < 0.95'})")
            print(f"    RelErr: {r['relative_error_pct']:.2f}% ({'✅' if error_pass else '❌ > 10%'})")

    print()
    if all_pass:
        print("✅ All tests PASSED!")
        print("   Compression algorithm is mathematically correct.")
    else:
        print("❌ Some tests FAILED!")
        print("   Quality may be below acceptable threshold.")

    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_comprehensive_evaluation()
