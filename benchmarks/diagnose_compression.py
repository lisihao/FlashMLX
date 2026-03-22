"""
Diagnose compression algorithm

Test if compression works correctly when using the SAME queries
that were used for compression.

Expected: Perfect or near-perfect reconstruction
"""
import mlx.core as mx
import numpy as np

from flashmlx.compaction.query_generation import self_study_kmeans
from flashmlx.cache.compaction_algorithm import create_compaction_algorithm


def compute_attention(queries, keys, values):
    """Standard attention"""
    scores = queries @ keys.T
    scale = 1.0 / mx.sqrt(mx.array(queries.shape[-1], dtype=mx.float32))
    scores = scores * scale
    weights = mx.softmax(scores, axis=-1)
    output = weights @ values
    return output, weights


def compute_compressed_attention(queries, C1, beta, C2):
    """Compressed attention"""
    scores = queries @ C1.T
    scale = 1.0 / mx.sqrt(mx.array(queries.shape[-1], dtype=mx.float32))
    scores = scores * scale + beta[None, :]
    weights = mx.softmax(scores, axis=-1)
    output = weights @ C2
    return output, weights


def diagnose():
    """
    Diagnostic test: Use compression queries for testing
    Expected result: Near-perfect reconstruction
    """
    print("=" * 70)
    print("Compression Algorithm Diagnosis")
    print("=" * 70)
    print()

    # Setup
    seq_len = 1000
    head_dim = 64
    compression_ratio = 4
    budget = seq_len // compression_ratio
    num_queries = 100

    print(f"Setup:")
    print(f"  Sequence: {seq_len} tokens")
    print(f"  Compression: {compression_ratio}x ({seq_len} -> {budget} tokens)")
    print(f"  Query generation: {num_queries} queries")
    print()

    # Generate synthetic KV
    print("Step 1: Generate synthetic KV cache...")
    keys = mx.random.normal((seq_len, head_dim))
    values = mx.random.normal((seq_len, head_dim))
    print(f"  ✓ Generated")
    print()

    # Self-Study: Generate queries
    print("Step 2: Self-Study (K-means)...")
    queries = self_study_kmeans(keys, num_queries, verbose=False)
    print(f"  ✓ Generated {queries.shape[0]} queries")
    print()

    # Compute original attention with these queries
    print("Step 3: Original attention (using compression queries)...")
    original_output, original_weights = compute_attention(queries, keys, values)
    print(f"  ✓ Output shape: {original_output.shape}")
    print(f"  ✓ Output norm: {float(mx.sqrt(mx.sum(original_output ** 2))):.2f}")
    print()

    # Compress
    print("Step 4: Compress with these queries...")
    algo = create_compaction_algorithm(
        score_method='mean',
        beta_method='nnls',
        c2_method='lsq',
        c2_ridge_lambda=0.01
    )

    C1, beta, C2, _ = algo.compute_compacted_cache(keys, values, queries, budget)
    print(f"  ✓ C1: {C1.shape}")
    print(f"  ✓ beta: {beta.shape}")
    print(f"  ✓ C2: {C2.shape}")
    print()

    # Compute compressed attention with SAME queries
    print("Step 5: Compressed attention (SAME queries)...")
    compressed_output, compressed_weights = compute_compressed_attention(queries, C1, beta, C2)
    print(f"  ✓ Output shape: {compressed_output.shape}")
    print(f"  ✓ Output norm: {float(mx.sqrt(mx.sum(compressed_output ** 2))):.2f}")
    print()

    # Compare
    print("=" * 70)
    print("Quality Metrics")
    print("=" * 70)

    # Cosine similarity
    orig_flat = mx.reshape(original_output, (-1,))
    comp_flat = mx.reshape(compressed_output, (-1,))
    dot_prod = float(mx.sum(orig_flat * comp_flat))
    norm_orig = float(mx.sqrt(mx.sum(orig_flat ** 2)))
    norm_comp = float(mx.sqrt(mx.sum(comp_flat ** 2)))
    cos_sim = dot_prod / (norm_orig * norm_comp) if (norm_orig > 0 and norm_comp > 0) else 0.0

    # L2 distance
    diff = original_output - compressed_output
    l2_dist = float(mx.sqrt(mx.sum(diff ** 2)))
    rel_error = (l2_dist / norm_orig) * 100 if norm_orig > 0 else 0.0

    print(f"Cosine Similarity: {cos_sim:.6f}")
    print(f"L2 Distance: {l2_dist:.2f}")
    print(f"Relative Error: {rel_error:.2f}%")
    print()

    # Weight comparison
    print("Attention Weights Analysis:")
    print(f"  Original weights sum (per query): {float(mx.mean(mx.sum(original_weights, axis=-1))):.6f}")
    print(f"  Compressed weights sum (per query): {float(mx.mean(mx.sum(compressed_weights, axis=-1))):.6f}")
    print(f"  Original weights shape: {original_weights.shape}")
    print(f"  Compressed weights shape: {compressed_weights.shape}")
    print()

    # Expected result
    print("=" * 70)
    print("Diagnosis Result")
    print("=" * 70)
    print()

    if cos_sim >= 0.95:
        print("✅ PASS: Near-perfect reconstruction achieved!")
        print("   Compression algorithm is working correctly.")
    elif cos_sim >= 0.80:
        print("⚠️  PARTIAL: Reasonable quality but not perfect")
        print(f"   CosSim {cos_sim:.4f} is acceptable but could be better")
        print("   May indicate suboptimal compression parameters")
    else:
        print("❌ FAIL: Poor reconstruction quality!")
        print(f"   CosSim {cos_sim:.4f} is far below expected (>= 0.95)")
        print()
        print("Possible causes:")
        print("  1. Bug in compression algorithm implementation")
        print("  2. Incompatible attention formula")
        print("  3. Numerical instability")
        print("  4. Query generation not representative")

    print("=" * 70)


if __name__ == "__main__":
    diagnose()
