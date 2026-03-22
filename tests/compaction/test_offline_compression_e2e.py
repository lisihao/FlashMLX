"""
End-to-end test for offline compression pipeline
"""
import mlx.core as mx
from flashmlx.compaction.offline_compressor import (
    offline_compress_kv_cache_per_head,
    offline_compress_kv_cache
)


def test_offline_compress_single_head():
    """Test offline compression for single head"""
    print("Test: offline_compress_kv_cache_per_head")

    # Create synthetic KV cache
    seq_len = 1000
    head_dim = 64

    keys = mx.random.normal((seq_len, head_dim))
    values = mx.random.normal((seq_len, head_dim))

    # Compress 4x (1000 -> 250)
    compression_ratio = 4
    num_queries = 50

    print(f"  Input: {seq_len} tokens, {head_dim} dims")
    print(f"  Target: {seq_len // compression_ratio} tokens ({compression_ratio}x)")
    print(f"  Using {num_queries} representative queries")

    # Test with OMP disabled (faster)
    C1, beta, C2 = offline_compress_kv_cache_per_head(
        keys, values,
        compression_ratio=compression_ratio,
        num_queries=num_queries,
        use_omp=False,
        verbose=True
    )

    # Check shapes
    expected_budget = seq_len // compression_ratio
    assert C1.shape == (expected_budget, head_dim), f"C1 shape mismatch: {C1.shape}"
    assert beta.shape == (expected_budget,), f"beta shape mismatch: {beta.shape}"
    assert C2.shape == (expected_budget, head_dim), f"C2 shape mismatch: {C2.shape}"

    print(f"  ✓ C1: {C1.shape}")
    print(f"  ✓ beta: {beta.shape}")
    print(f"  ✓ C2: {C2.shape}")
    print()


def test_offline_compress_multi_head():
    """Test offline compression for multi-head cache"""
    print("Test: offline_compress_kv_cache (multi-head)")

    # Create synthetic multi-head KV cache
    B = 1
    n_heads = 4
    seq_len = 500
    head_dim = 64

    keys = mx.random.normal((B, n_heads, seq_len, head_dim))
    values = mx.random.normal((B, n_heads, seq_len, head_dim))

    compression_ratio = 4
    num_queries = 30

    print(f"  Input: B={B}, heads={n_heads}, seq={seq_len}, dim={head_dim}")
    print(f"  Target: {seq_len // compression_ratio} tokens per head ({compression_ratio}x)")

    # Test with OMP disabled for speed
    C1, beta, C2 = offline_compress_kv_cache(
        keys, values,
        compression_ratio=compression_ratio,
        num_queries=num_queries,
        use_omp=False,
        verbose=True
    )

    # Check shapes
    expected_budget = seq_len // compression_ratio

    print(f"  Actual shapes: C1={C1.shape}, beta={beta.shape}, C2={C2.shape}")
    print(f"  Expected: C1=(1,{n_heads},{expected_budget},{head_dim}), beta=(1,{n_heads},{expected_budget}), C2=(1,{n_heads},{expected_budget},{head_dim})")

    assert C1.shape == (B, n_heads, expected_budget, head_dim), f"C1 shape mismatch: got {C1.shape}, expected {(B, n_heads, expected_budget, head_dim)}"
    assert beta.shape == (B, n_heads, expected_budget), f"beta shape mismatch: got {beta.shape}, expected {(B, n_heads, expected_budget)}"
    assert C2.shape == (B, n_heads, expected_budget, head_dim), f"C2 shape mismatch: got {C2.shape}, expected {(B, n_heads, expected_budget, head_dim)}"

    print(f"  ✓ All heads compressed successfully")
    print()


def test_offline_compress_with_omp():
    """Test offline compression with OMP enabled"""
    print("Test: offline_compress with OMP")

    # Small cache for OMP test
    seq_len = 200
    head_dim = 64

    keys = mx.random.normal((seq_len, head_dim))
    values = mx.random.normal((seq_len, head_dim))

    compression_ratio = 4
    num_queries = 20

    print(f"  Using OMP refinement (simplified)")

    C1, beta, C2 = offline_compress_kv_cache_per_head(
        keys, values,
        compression_ratio=compression_ratio,
        num_queries=num_queries,
        use_omp=True,  # Enable OMP
        verbose=True
    )

    expected_budget = seq_len // compression_ratio
    assert C1.shape == (expected_budget, head_dim)

    print(f"  ✓ OMP refinement works")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("End-to-End Offline Compression Tests")
    print("=" * 70)
    print()

    test_offline_compress_single_head()
    test_offline_compress_multi_head()
    test_offline_compress_with_omp()

    print("=" * 70)
    print("All E2E tests passed!")
    print("=" * 70)
