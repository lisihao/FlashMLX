"""
Self-Study: Select representative queries from full key sequence

Paper Reference: Section 3.2 - Query Generation
Methods:
1. K-means clustering (preferred for quality)
2. Importance sampling (faster alternative)
"""
import mlx.core as mx
import numpy as np
from typing import Literal


def self_study_kmeans(
    keys: mx.array,
    num_queries: int = 100,
    random_state: int = 42,
    verbose: bool = False
) -> mx.array:
    """
    K-means clustering to select representative queries

    Args:
        keys: Full key sequence (seq_len, head_dim)
        num_queries: Number of representative queries to select
        random_state: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Representative queries (num_queries, head_dim)
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError(
            "scikit-learn is required for K-means clustering. "
            "Install with: pip install scikit-learn"
        )

    if verbose:
        print(f"  Self-Study (K-means): {keys.shape[0]} tokens -> {num_queries} queries")

    # Convert MLX array to NumPy (copy to CPU first to avoid buffer format issues)
    import mlx.core as mx
    keys_cpu = mx.array(keys)  # Ensure on CPU
    keys_np = np.array(keys_cpu.tolist())  # Convert via Python list
    num_queries = min(num_queries, keys_np.shape[0])

    kmeans = KMeans(
        n_clusters=num_queries,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    kmeans.fit(keys_np)

    centroids = mx.array(kmeans.cluster_centers_)

    if verbose:
        print(f"    OK Inertia: {kmeans.inertia_:.2f}, Iterations: {kmeans.n_iter_}")

    return centroids


def self_study_importance_sampling(
    keys: mx.array,
    num_queries: int = 100,
    method: Literal["norm", "variance"] = "norm",
    verbose: bool = False
) -> mx.array:
    """
    Importance sampling based on key statistics

    Args:
        keys: Full key sequence (seq_len, head_dim)
        num_queries: Number of queries to sample
        method: "norm" or "variance"
        verbose: Print progress

    Returns:
        Sampled queries (num_queries, head_dim)
    """
    if verbose:
        print(f"  Self-Study (Importance): {keys.shape[0]} tokens -> {num_queries} queries")

    seq_len, head_dim = keys.shape
    num_queries = min(num_queries, seq_len)

    if method == "norm":
        importance = mx.sum(keys ** 2, axis=-1)
    elif method == "variance":
        mean_key = mx.mean(keys, axis=0, keepdims=True)
        importance = mx.sum((keys - mean_key) ** 2, axis=-1)
    else:
        raise ValueError(f"Unknown method: {method}")

    top_indices = mx.argsort(importance)[-num_queries:]

    if verbose:
        top_imp = importance[top_indices]
        print(f"    OK Importance range: [{float(mx.min(top_imp)):.2f}, {float(mx.max(top_imp)):.2f}]")

    return keys[top_indices]


def self_study_auto(
    keys: mx.array,
    num_queries: int = 100,
    prefer_quality: bool = True,
    verbose: bool = False
) -> mx.array:
    """Automatic query selection with fallback"""
    if prefer_quality:
        try:
            return self_study_kmeans(keys, num_queries, verbose=verbose)
        except ImportError:
            if verbose:
                print("  WARN: scikit-learn not available, using importance sampling")
            return self_study_importance_sampling(keys, num_queries, verbose=verbose)
    else:
        return self_study_importance_sampling(keys, num_queries, verbose=verbose)
