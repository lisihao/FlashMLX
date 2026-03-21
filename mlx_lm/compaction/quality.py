"""
Quality Path: Attention-aware KV Cache compression

Phase B implementation: Uses real attention weights for key selection,
adaptive beta fitting, and LSQ for value compression.
"""

import mlx.core as mx
from typing import Optional, Tuple
from .base import compute_attention_output, safe_softmax
from .solvers.nnls import nnls_clamped, nnls_pgd
from .solvers.lsq import compute_C2_lstsq, compute_C2_cholesky, compute_C2_pinv


def select_keys_attention_aware(
    queries: mx.array,
    keys: mx.array,
    budget: int,
    scale: Optional[float] = None
) -> mx.array:
    """
    Select keys based on real attention weights.

    Computes Q·K^T scores and selects top-k keys with highest attention.

    Parameters
    ----------
    queries : mx.array, shape (query_len, head_dim)
        Query vectors
    keys : mx.array, shape (seq_len, head_dim)
        Key vectors
    budget : int
        Number of keys to select
    scale : float, optional
        Attention scale factor (default: 1/sqrt(head_dim))

    Returns
    -------
    indices : mx.array, shape (budget,)
        Indices of selected keys, sorted in ascending order

    Notes
    -----
    This function is the core of Quality Path:
    - Computes attention scores: Q·K^T / scale
    - Aggregates across queries: sum over query dimension
    - Selects top-k keys with highest total attention
    - Time complexity: O(query_len * seq_len * head_dim)
    """
    query_len, head_dim = queries.shape
    seq_len = keys.shape[0]

    if scale is None:
        scale = head_dim ** 0.5

    # Compute attention scores: (query_len, seq_len)
    scores = (queries @ keys.T) / scale

    # Aggregate attention across queries
    # Each key gets a score = sum of attention from all queries
    total_scores = mx.sum(scores, axis=0)  # (seq_len,)

    # Select top-k keys
    if budget >= seq_len:
        # If budget >= seq_len, return all indices
        return mx.arange(seq_len)

    # argsort returns indices in ascending order, we want descending
    # So we negate the scores
    sorted_indices = mx.argsort(-total_scores)
    selected_indices = sorted_indices[:budget]

    # Sort selected indices in ascending order (for consistency)
    selected_indices = mx.sort(selected_indices)

    return selected_indices


def compact_single_head_quality(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    budget: int,
    scale: Optional[float] = None,
    fit_beta: bool = True,
    fit_c2: bool = True,
    nnls_method: str = "clamped",
    lsq_method: str = "lstsq",
    return_indices: bool = False
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Quality Path: Single-head KV cache compression.

    Uses attention-aware selection, adaptive beta, and LSQ fitting.

    Parameters
    ----------
    queries : mx.array, shape (query_len, head_dim)
    keys : mx.array, shape (seq_len, head_dim)
    values : mx.array, shape (seq_len, head_dim)
    budget : int
        Target compressed size
    scale : float, optional
    fit_beta : bool, default=True
        Whether to fit attention bias
    fit_c2 : bool, default=True
        Whether to fit compressed values using LSQ
    nnls_method : str, default="clamped"
        NNLS method: "clamped" or "pgd"
    lsq_method : str, default="lstsq"
        LSQ method: "lstsq", "cholesky", or "pinv"
    return_indices : bool, default=False

    Returns
    -------
    C1 : mx.array, shape (budget, head_dim)
        Compressed keys
    beta : mx.array, shape (budget,)
        Attention bias
    C2 : mx.array, shape (budget, head_dim)
        Compressed values

    Notes
    -----
    Quality Path algorithm:
    1. Select keys using attention-aware selection
    2. Fit beta to match original attention weights (NNLS)
    3. Fit C2 to match original output (LSQ)

    Compared to Fast Path:
    - Fast Path: Heuristic selection (Recent+Random), beta=0, C2=V[indices]
    - Quality Path: Attention-aware selection, fitted beta, fitted C2
    - Quality Path is slower but much more accurate on random data
    """
    query_len, head_dim = queries.shape
    seq_len = keys.shape[0]

    if scale is None:
        scale = head_dim ** 0.5

    # Step 1: Attention-aware key selection
    indices = select_keys_attention_aware(queries, keys, budget, scale)
    C1 = keys[indices]  # (budget, head_dim)

    # Compute original attention output (for comparison)
    # This is what we're trying to approximate
    original_output = compute_attention_output(
        queries, keys, values, beta=None, scale=scale
    )

    # Step 2: Fit beta (if enabled)
    if fit_beta:
        # Following the paper's approach:
        # We want to match the attention mass (sum of exp(scores))
        # Original: exp_scores_original = exp(Q·K^T / scale)
        # Compressed: exp_scores_compressed = exp(Q·C1^T / scale + beta)

        # Compute scores
        compressed_scores = (queries @ C1.T) / scale  # (query_len, budget)
        original_scores = (queries @ keys.T) / scale  # (query_len, seq_len)

        # Compute exp scores (unnormalized attention weights)
        exp_scores_original = mx.exp(original_scores - mx.max(original_scores, axis=1, keepdims=True))
        exp_scores_compressed = mx.exp(compressed_scores - mx.max(compressed_scores, axis=1, keepdims=True))

        # Target: total attention mass for each query
        # target[i] = sum_j exp_scores_original[i, j]
        target_mass = mx.sum(exp_scores_original, axis=1)  # (query_len,)

        # We want: M @ B ≈ target_mass
        # Where M[i, j] = exp_scores_compressed[i, j] (attention from query i to key j)
        # And B[j] = exp(beta[j])
        # This is NNLS: min ||M @ B - target||^2, s.t. B >= 0

        # M = exp_scores_compressed at selected indices
        M = exp_scores_compressed  # (query_len, budget)
        y = target_mass  # (query_len,)

        # Solve NNLS
        if nnls_method == "clamped":
            B = nnls_clamped(M, y, lower_bound=1e-12)
        elif nnls_method == "pgd":
            B = nnls_pgd(M, y, max_iters=100)
        else:
            # Fallback: least squares (may have negative values)
            # compute_C2_lstsq is imported at top of file
            B = compute_C2_lstsq(M[:, :, None], y[:, None]).squeeze()
            B = mx.maximum(B, 1e-12)  # Clamp to positive

        # Convert B to beta: beta = log(B)
        beta = mx.log(B + 1e-12)
        beta = mx.clip(beta, -10.0, 10.0)  # Clip for numerical stability
    else:
        beta = mx.zeros(budget)

    # Step 3: Fit C2 (if enabled)
    if fit_c2:
        # We want: (queries @ C1.T) · C2 ≈ original_output
        # Where: (queries @ C1.T) is attention-like weights
        # This is: W · C2 ≈ Y, where W = softmax(queries @ C1.T / scale + beta)

        # Compute compressed attention weights
        compressed_scores = (queries @ C1.T) / scale + beta[None, :]
        compressed_attn_weights = safe_softmax(compressed_scores, axis=1)  # (query_len, budget)

        # Solve: compressed_attn_weights @ C2 ≈ original_output
        # LSQ problem: min ||W @ C2 - Y||^2
        # Solution: C2 = (W^T W)^{-1} W^T Y

        if lsq_method == "lstsq":
            C2 = compute_C2_lstsq(compressed_attn_weights, original_output)
        elif lsq_method == "cholesky":
            C2 = compute_C2_cholesky(compressed_attn_weights, original_output)
        elif lsq_method == "pinv":
            C2 = compute_C2_pinv(compressed_attn_weights, original_output)
        else:
            # Fallback: direct copy
            C2 = values[indices]

        # C2 shape: (budget, head_dim)
    else:
        # Direct copy
        C2 = values[indices]

    if return_indices:
        return C1, beta, C2, indices
    else:
        return C1, beta, C2


def compact_multi_head_quality(
    keys: mx.array,
    values: mx.array,
    budget: int,
    queries: Optional[mx.array] = None,
    scale: Optional[float] = None,
    fit_beta: bool = True,
    fit_c2: bool = True,
    nnls_method: str = "clamped",
    lsq_method: str = "lstsq"
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Quality Path: Multi-head KV cache compression.

    Parameters
    ----------
    keys : mx.array, shape (n_heads, seq_len, head_dim)
    values : mx.array, shape (n_heads, seq_len, head_dim)
    budget : int
    queries : mx.array, optional, shape (n_heads, query_len, head_dim)
        If None, uses keys as queries (self-attention)
    scale : float, optional
    fit_beta : bool, default=True
    fit_c2 : bool, default=True
    nnls_method : str, default="clamped"
    lsq_method : str, default="lstsq"

    Returns
    -------
    C1 : mx.array, shape (n_heads, budget, head_dim)
    beta : mx.array, shape (n_heads, budget)
    C2 : mx.array, shape (n_heads, budget, head_dim)
    """
    n_heads, seq_len, head_dim = keys.shape

    if queries is None:
        # Use keys as queries (self-attention approximation)
        queries = keys

    if scale is None:
        scale = head_dim ** 0.5

    # Process each head independently
    C1_list = []
    beta_list = []
    C2_list = []

    for h in range(n_heads):
        K_h = keys[h]      # (seq_len, head_dim)
        V_h = values[h]
        Q_h = queries[h]   # (query_len, head_dim)

        C1_h, beta_h, C2_h = compact_single_head_quality(
            Q_h, K_h, V_h, budget, scale,
            fit_beta=fit_beta, fit_c2=fit_c2,
            nnls_method=nnls_method, lsq_method=lsq_method
        )

        C1_list.append(C1_h[None, ...])  # (1, budget, head_dim)
        beta_list.append(beta_h[None, ...])  # (1, budget)
        C2_list.append(C2_h[None, ...])

    # Concatenate
    C1 = mx.concatenate(C1_list, axis=0)  # (n_heads, budget, head_dim)
    beta = mx.concatenate(beta_list, axis=0)  # (n_heads, budget)
    C2 = mx.concatenate(C2_list, axis=0)

    return C1, beta, C2
