"""
Offline KV Cache Compressor (完整论文实现)

用于离线 GC 式内存压缩，不追求速度，追求质量

Paper Implementation:
1. Self-Study: Query generation via K-means
2. OMP: Query refinement
3. Key Selection: Attention-aware top-k
4. Beta Fitting: NNLS
5. Value Fitting: Least squares
"""
import mlx.core as mx
import time
from typing import Tuple, List, Optional

from .query_generation import (
    self_study_auto,
    self_study_kmeans,
    self_study_importance_sampling,
    omp_refine_queries,
    omp_refine_queries_fast,
)
from ..cache.compaction_algorithm import create_compaction_algorithm


def offline_compress_kv_cache_per_head(
    keys: mx.array,      # (seq_len, head_dim)
    values: mx.array,    # (seq_len, head_dim)
    compression_ratio: int = 4,
    num_queries: int = 100,
    use_omp: bool = True,
    use_fast_omp: bool = False,
    max_omp_iters: int = 100,
    verbose: bool = True,
    return_queries: bool = False
) -> Tuple[mx.array, mx.array, mx.array] | Tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Offline compress single head's KV cache (完整论文算法)

    Args:
        keys: Keys for one head (seq_len, head_dim)
        values: Values for one head (seq_len, head_dim)
        compression_ratio: Target ratio (e.g., 4x)
        num_queries: Number of representative queries
        use_omp: Whether to use OMP refinement
        use_fast_omp: Use fast batch OMP instead of greedy
        max_omp_iters: Maximum OMP iterations
        verbose: Print progress

    Returns:
        (C1, beta, C2): Compressed cache
    """
    seq_len, head_dim = keys.shape
    budget = seq_len // compression_ratio

    if verbose:
        print(f"    Sequence: {seq_len} -> {budget} tokens ({compression_ratio}x)")

    # Step 1: Self-Study (Query Generation)
    t0 = time.time()
    queries = self_study_auto(keys, num_queries, prefer_quality=True, verbose=verbose)
    t1 = time.time()

    if verbose:
        print(f"      [Self-Study] {t1-t0:.2f}s")

    # Step 2: OMP Refinement (optional but recommended)
    if use_omp:
        t0 = time.time()
        if use_fast_omp:
            queries = omp_refine_queries_fast(
                queries, keys, values, budget, verbose=verbose
            )
        else:
            queries = omp_refine_queries(
                queries, keys, values, budget,
                max_iters=max_omp_iters,
                verbose=verbose
            )
        t1 = time.time()

        if verbose:
            print(f"      [OMP] {t1-t0:.2f}s")

    # Step 3-6: Existing compression pipeline
    t0 = time.time()
    algo = create_compaction_algorithm(
        score_method='mean',
        beta_method='nnls',
        c2_method='lsq',
        c2_ridge_lambda=0.01
    )

    C1, beta, C2, _ = algo.compute_compacted_cache(
        keys, values, queries, budget
    )
    t1 = time.time()

    if verbose:
        print(f"      [Compression] {t1-t0:.2f}s")

    if return_queries:
        return C1, beta, C2, queries
    else:
        return C1, beta, C2


def offline_compress_kv_cache(
    keys: mx.array,      # (B, n_heads, seq_len, head_dim)
    values: mx.array,    # (B, n_heads, seq_len, head_dim)
    compression_ratio: int = 4,
    num_queries: int = 100,
    use_omp: bool = True,
    use_fast_omp: bool = False,
    max_omp_iters: int = 100,
    verbose: bool = True,
    return_queries: bool = False
) -> Tuple[mx.array, mx.array, mx.array] | Tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Offline compress multi-head KV cache (完整论文实现)

    可以慢慢做，不限制时间。用于离线 GC 式内存压缩。

    Args:
        keys: Full key cache (B, n_heads, seq_len, head_dim)
        values: Full value cache (B, n_heads, seq_len, head_dim)
        compression_ratio: Target compression ratio (e.g., 4x)
        num_queries: Number of representative queries for self-study
        use_omp: Whether to use OMP refinement (slower but better quality)
        use_fast_omp: Use fast batch OMP (faster but slightly lower quality)
        max_omp_iters: Maximum OMP iterations
        verbose: Print progress

    Returns:
        (C1, beta, C2): Compressed cache
            C1: (B, n_heads, budget, head_dim)
            beta: (B, n_heads, budget)
            C2: (B, n_heads, budget, head_dim)

    Example:
        >>> # After generation, compress KV cache offline
        >>> compressed = offline_compress_kv_cache(
        ...     cache.keys, cache.values,
        ...     compression_ratio=4,
        ...     use_omp=True,
        ...     verbose=True
        ... )
        >>> # Replace cache with compressed version
        >>> cache.keys, cache.beta, cache.values = compressed
    """
    B, n_heads, seq_len, head_dim = keys.shape
    budget = seq_len // compression_ratio

    if verbose:
        print("=" * 70)
        print("Offline KV Cache Compression (Paper Implementation)")
        print("=" * 70)
        print(f"  Sequence length: {seq_len} tokens")
        print(f"  Target budget: {budget} tokens ({compression_ratio}x)")
        print(f"  Num heads: {n_heads}")
        print(f"  Query generation: {num_queries} queries")
        print(f"  Use OMP: {use_omp}")
        if use_omp:
            print(f"  OMP variant: {'Fast (batch)' if use_fast_omp else 'Standard (greedy)'}")
        print()

    # Compress each head independently
    C1_list = []
    beta_list = []
    C2_list = []
    queries_list = [] if return_queries else None

    total_time = 0

    for head_idx in range(n_heads):
        if verbose:
            print(f"  [Head {head_idx + 1}/{n_heads}]")

        t_head_start = time.time()

        K_head = keys[0, head_idx]  # (seq_len, head_dim)
        V_head = values[0, head_idx]

        result = offline_compress_kv_cache_per_head(
            K_head, V_head,
            compression_ratio=compression_ratio,
            num_queries=num_queries,
            use_omp=use_omp,
            use_fast_omp=use_fast_omp,
            max_omp_iters=max_omp_iters,
            verbose=verbose,
            return_queries=return_queries
        )

        if return_queries:
            C1, beta, C2, queries = result
            queries_list.append(queries)
        else:
            C1, beta, C2 = result

        C1_list.append(C1)
        beta_list.append(beta)
        C2_list.append(C2)

        t_head_end = time.time()
        head_time = t_head_end - t_head_start
        total_time += head_time

        if verbose:
            print(f"    OK Head {head_idx + 1} compressed in {head_time:.2f}s")
            print()

    # Stack results
    # Each element in C1_list is (budget, head_dim), we want (1, n_heads, budget, head_dim)
    C1_all = mx.stack(C1_list, axis=0)[None, ...]  # stack heads on axis 0, then add batch dim
    beta_all = mx.stack(beta_list, axis=0)[None, ...]  # (1, n_heads, budget)
    C2_all = mx.stack(C2_list, axis=0)[None, ...]  # (1, n_heads, budget, head_dim)

    if verbose:
        print("=" * 70)
        print("Compression Complete")
        print("=" * 70)
        print(f"  Original: {seq_len} tokens")
        print(f"  Compressed: {budget} tokens")
        print(f"  Ratio: {compression_ratio}x")
        print(f"  Memory saved: {(1 - budget/seq_len) * 100:.1f}%")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time per head: {total_time/n_heads:.2f}s")
        print("=" * 70)

    if return_queries:
        # Stack queries: (n_heads, num_queries, head_dim) -> (1, n_heads, num_queries, head_dim)
        queries_all = mx.stack(queries_list, axis=0)[None, ...]
        return C1_all, beta_all, C2_all, queries_all
    else:
        return C1_all, beta_all, C2_all


def offline_compress_mlx_lm_cache(
    cache: List,  # List of KVCache objects from mlx-lm
    compression_ratio: int = 4,
    num_queries: int = 100,
    use_omp: bool = True,
    use_fast_omp: bool = False,
    verbose: bool = True
):
    """
    Compress MLX-LM cache list (wrapper for convenience)

    Args:
        cache: List of KVCache objects from mlx-lm
        compression_ratio: Target ratio
        num_queries: Number of queries for self-study
        use_omp: Use OMP refinement
        use_fast_omp: Use fast OMP variant
        verbose: Print progress

    Returns:
        List of CompactedKVCacheLayer objects
    """
    from ..cache import create_compacted_cache_list

    if verbose:
        print(f"Compressing MLX-LM cache ({len(cache)} layers)")

    compacted_data = []
    original_seq_len = cache[0].keys.shape[-2]

    for layer_idx, layer_cache in enumerate(cache):
        if verbose:
            print(f"\nLayer {layer_idx + 1}/{len(cache)}")

        K = layer_cache.keys  # (B, n_kv_heads, T, head_dim)
        V = layer_cache.values

        C1, beta, C2 = offline_compress_kv_cache(
            K, V,
            compression_ratio=compression_ratio,
            num_queries=num_queries,
            use_omp=use_omp,
            use_fast_omp=use_fast_omp,
            verbose=verbose
        )

        compacted_data.append((C1, beta, C2))

    # Create CompactedKVCache list
    compressed_cache = create_compacted_cache_list(
        compacted_data,
        original_seq_len=original_seq_len
    )

    return compressed_cache
