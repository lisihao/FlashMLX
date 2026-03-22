"""
Unit tests for Query Generation module
"""
import mlx.core as mx
import numpy as np
import pytest


def test_self_study_kmeans():
    """Test K-means query selection"""
    from flashmlx.compaction.query_generation import self_study_kmeans

    # Create test data
    seq_len, head_dim = 256, 64
    keys = mx.random.normal((seq_len, head_dim))

    # Select queries
    num_queries = 50
    queries = self_study_kmeans(keys, num_queries, verbose=False)

    # Assertions
    assert queries.shape == (num_queries, head_dim)
    assert queries.dtype == keys.dtype


def test_self_study_importance_sampling():
    """Test importance sampling query selection"""
    from flashmlx.compaction.query_generation import self_study_importance_sampling

    seq_len, head_dim = 256, 64
    keys = mx.random.normal((seq_len, head_dim))

    num_queries = 50
    queries = self_study_importance_sampling(keys, num_queries, method="norm", verbose=False)

    assert queries.shape == (num_queries, head_dim)


def test_self_study_auto_fallback():
    """Test auto query selection with fallback"""
    from flashmlx.compaction.query_generation import self_study_auto

    seq_len, head_dim = 256, 64
    keys = mx.random.normal((seq_len, head_dim))

    queries = self_study_auto(keys, num_queries=50, verbose=False)

    assert queries.shape[0] == 50
    assert queries.shape[1] == head_dim


def test_compute_attention_output():
    """Test attention output computation"""
    from flashmlx.compaction.query_generation import compute_attention_output

    num_queries, seq_len, head_dim = 10, 100, 64
    queries = mx.random.normal((num_queries, head_dim))
    keys = mx.random.normal((seq_len, head_dim))
    values = mx.random.normal((seq_len, head_dim))

    output = compute_attention_output(queries, keys, values)

    assert output.shape == (num_queries, head_dim)


def test_omp_refine_queries_basic():
    """Test basic OMP query refinement"""
    from flashmlx.compaction.query_generation import omp_refine_queries

    # Small test case
    num_initial, seq_len, head_dim = 20, 100, 64
    budget = 25

    initial_queries = mx.random.normal((num_initial, head_dim))
    keys = mx.random.normal((seq_len, head_dim))
    values = mx.random.normal((seq_len, head_dim))

    # Refine with few iterations for speed
    refined = omp_refine_queries(
        initial_queries, keys, values, budget,
        max_iters=5,
        verbose=False
    )

    assert refined.shape[0] <= num_initial
    assert refined.shape[1] == head_dim


def test_omp_fast_variant():
    """Test fast OMP variant"""
    from flashmlx.compaction.query_generation import omp_refine_queries_fast

    num_initial, seq_len, head_dim = 30, 100, 64
    budget = 25

    initial_queries = mx.random.normal((num_initial, head_dim))
    keys = mx.random.normal((seq_len, head_dim))
    values = mx.random.normal((seq_len, head_dim))

    refined = omp_refine_queries_fast(
        initial_queries, keys, values, budget,
        batch_size=5,
        verbose=False
    )

    assert refined.shape[0] <= budget
    assert refined.shape[1] == head_dim


if __name__ == "__main__":
    print("Running Query Generation tests...")
    test_self_study_kmeans()
    print("OK test_self_study_kmeans")
    
    test_self_study_importance_sampling()
    print("OK test_self_study_importance_sampling")
    
    test_self_study_auto_fallback()
    print("OK test_self_study_auto_fallback")
    
    test_compute_attention_output()
    print("OK test_compute_attention_output")
    
    test_omp_refine_queries_basic()
    print("OK test_omp_refine_queries_basic")
    
    test_omp_fast_variant()
    print("OK test_omp_fast_variant")
    
    print("\nAll tests passed!")
