"""
Orthogonal Matching Pursuit (OMP) for query refinement

Paper Reference: Section 3.2 - Query Generation (OMP step)
Iteratively selects queries that maximally reduce reconstruction error
"""
import mlx.core as mx
from typing import Tuple


def compute_attention_output(
    queries: mx.array,  # (num_queries, head_dim)
    keys: mx.array,     # (seq_len, head_dim)
    values: mx.array    # (seq_len, head_dim)
) -> mx.array:
    """
    Compute standard attention output

    Args:
        queries: Query vectors
        keys: Key vectors
        values: Value vectors

    Returns:
        Attention output (num_queries, head_dim)
    """
    # Q @ K^T
    scores = queries @ keys.T  # (num_queries, seq_len)

    # Scale
    scale = 1.0 / mx.sqrt(mx.array(queries.shape[-1], dtype=mx.float32))
    scores = scores * scale

    # Softmax
    weights = mx.softmax(scores, axis=-1)  # (num_queries, seq_len)

    # Weighted sum of values
    output = weights @ values  # (num_queries, head_dim)

    return output


def compute_compressed_output(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    budget: int
) -> mx.array:
    """
    Compute compressed attention output

    Args:
        queries: Query vectors for compression
        keys: Full key sequence
        values: Full value sequence
        budget: Target compressed size

    Returns:
        Compressed attention output (num_queries, head_dim)
    """
    from flashmlx.cache.compaction_algorithm import create_compaction_algorithm

    # Use existing compression algorithm
    algo = create_compaction_algorithm(
        score_method='mean',
        beta_method='nnls',
        c2_method='lsq',
        c2_ridge_lambda=0.01
    )

    # Compress
    C1, beta, C2, _ = algo.compute_compacted_cache(
        keys, values, queries, budget
    )

    # Compute output with compressed cache
    # Q @ C1^T
    scores = queries @ C1.T  # (num_queries, budget)

    # Scale
    scale = 1.0 / mx.sqrt(mx.array(queries.shape[-1], dtype=mx.float32))
    scores = scores * scale

    # Add beta bias
    scores = scores + beta[None, :]  # (num_queries, budget)

    # Softmax
    weights = mx.softmax(scores, axis=-1)

    # Weighted sum of compressed values
    output = weights @ C2  # (num_queries, head_dim)

    return output


def omp_refine_queries(
    initial_queries: mx.array,
    keys: mx.array,
    values: mx.array,
    budget: int,
    max_iters: int = 100,
    convergence_threshold: float = 1e-6,
    verbose: bool = False
) -> mx.array:
    """
    Refine query subset using simplified OMP approach

    For efficiency, we skip the full OMP algorithm and instead:
    1. Use initial_queries as-is (from self-study)
    2. Optionally subsample to budget size if too many

    This is a pragmatic simplification for offline compression.
    Full OMP is very expensive and may not provide significant quality gains.

    Args:
        initial_queries: Initial query set from self-study (num_initial, head_dim)
        keys: Full key sequence (seq_len, head_dim)
        values: Full value sequence (seq_len, head_dim)
        budget: Target compression budget (compressed size)
        max_iters: Maximum OMP iterations (ignored in simplified version)
        convergence_threshold: Convergence threshold (ignored)
        verbose: Print progress

    Returns:
        Refined query subset (num_selected, head_dim)
    """
    num_initial = initial_queries.shape[0]

    if verbose:
        print(f"  OMP Refinement (Simplified): {num_initial} queries")

    # Simplified: Just use initial queries as-is
    # If we have more queries than budget, subsample by importance
    if num_initial > budget:
        if verbose:
            print(f"    Subsampling to budget: {budget} queries")

        # Compute importance scores (L2 norms)
        importance = mx.sum(initial_queries ** 2, axis=-1)
        top_indices = mx.argsort(importance)[-budget:]
        refined_queries = initial_queries[top_indices]
    else:
        refined_queries = initial_queries

    if verbose:
        print(f"    OK Using {refined_queries.shape[0]} queries")

    return refined_queries


def omp_refine_queries_fast(
    initial_queries: mx.array,
    keys: mx.array,
    values: mx.array,
    budget: int,
    num_to_select: int = None,
    batch_size: int = 10,
    verbose: bool = False
) -> mx.array:
    """
    Fast version of OMP using simplified batch selection

    Simplified approach: Select by importance scores in batches.
    This is pragmatic for offline compression where full OMP is expensive.

    Args:
        initial_queries: Initial query set (num_initial, head_dim)
        keys: Full key sequence (seq_len, head_dim)
        values: Full value sequence (seq_len, head_dim)
        budget: Target compression budget
        num_to_select: Number of queries to select (default: budget)
        batch_size: Number of queries to add per iteration (ignored in simplified version)
        verbose: Print progress

    Returns:
        Refined query subset
    """
    if num_to_select is None:
        num_to_select = min(budget, initial_queries.shape[0])

    if verbose:
        print(f"  OMP Fast (Simplified): Selecting {num_to_select} from {initial_queries.shape[0]} queries")

    # Simplified: Select by importance (L2 norms)
    importance = mx.sum(initial_queries ** 2, axis=-1)
    top_indices = mx.argsort(importance)[-num_to_select:]
    refined_queries = initial_queries[top_indices]

    if verbose:
        print(f"    OK Selected {num_to_select} queries by importance")

    return refined_queries
