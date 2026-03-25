"""
Quality Path: Attention-aware KV Cache compression

Phase B implementation: Uses real attention weights for key selection,
adaptive beta fitting, and LSQ for value compression.

Ported from the author's PyTorch implementation:
- NNLS (Non-Negative Least Squares) for beta fitting
- LSQ (Least Squares) for C2 value fitting
"""

import mlx.core as mx
from typing import Optional, Tuple

# Temporary: Enable debug for first compression only
_DEBUG_COUNTER = 0
_DEBUG_ENABLED = False  # DISABLED: MLX SVD crashes on some matrices
from .base import compute_attention_output, safe_softmax
from .solvers.nnls import nnls_clamped, nnls_pgd
from .solvers.lsq import compute_C2_lstsq, compute_C2_cholesky, compute_C2_pinv


def _bounded_least_squares_scipy(
    M: mx.array,
    y: mx.array,
    bounds: Tuple[float, float] = (-3.0, 3.0),
    method: str = 'trf',
) -> mx.array:
    """
    Bounded least-squares (按照 AM 论文实现)

    Solves: min ||M @ B - y||^2
    Subject to: bounds[0] <= B <= bounds[1]

    论文中对 beta 使用 bounded LS，约束 β ∈ [-3, 3]

    Parameters
    ----------
    M : mx.array, shape (n, t)
        Design matrix (exp_scores)
    y : mx.array, shape (n,)
        Target vector (sum of exp_scores)
    bounds : Tuple[float, float]
        (lower, upper) bounds for B
    method : str
        Solver method: 'trf' (default), 'dogbox', 'lm'

    Returns
    -------
    B : mx.array, shape (t,)
        Solution vector
    """
    try:
        import numpy as np
        from scipy.optimize import least_squares

        M_np = np.array(M.tolist(), dtype=np.float32)
        y_np = np.array(y.tolist(), dtype=np.float32)

        t = M_np.shape[1]

        # Objective function: residual = M @ B - y
        def residual_fn(B):
            return M_np @ B - y_np

        # Initial guess: zeros
        B0 = np.zeros(t)

        # Solve with bounds
        result = least_squares(
            residual_fn,
            B0,
            bounds=(bounds[0], bounds[1]),
            method=method,
            verbose=0
        )

        if not result.success:
            # Fallback: use unbounded LS
            import warnings
            warnings.warn(f"Bounded LS failed: {result.message}, using unbounded fallback")
            B_np = np.linalg.lstsq(M_np, y_np, rcond=None)[0]
            B_np = np.clip(B_np, bounds[0], bounds[1])
        else:
            B_np = result.x

        return mx.array(B_np)

    except ImportError:
        # Fallback: unbounded + clipping
        import numpy as np
        M_np = np.array(M.tolist(), dtype=np.float32)
        y_np = np.array(y.tolist(), dtype=np.float32)
        B_np = np.linalg.lstsq(M_np, y_np, rcond=None)[0]
        B_np = np.clip(B_np, bounds[0], bounds[1])
        return mx.array(B_np)


def _nnls_mlx(
    M: mx.array,
    y: mx.array,
    lower_bound: float = 1e-12,
    upper_bound: Optional[float] = None,
    ridge_lambda: float = 0.0,  # Changed default to 0
    debug: bool = False,
    use_scipy: bool = True,  # Use true NNLS
    use_bounded: bool = False,  # ✅ NEW: Use bounded LS instead of NNLS
    bounds: Tuple[float, float] = (-3.0, 3.0),  # ✅ NEW: Bounds for beta
) -> mx.array:
    """
    Non-Negative Least Squares solver for MLX.

    Solves: min_B 0.5 * ||M B - y||_2^2  s.t. B >= 0

    Now uses scipy.optimize.nnls for true NNLS (not lstsq + clamp).

    Parameters
    ----------
    M : mx.array, shape (n, t)
        Design matrix (exp_scores for selected keys)
    y : mx.array, shape (n,)
        Target vector (total exp_scores sum)
    lower_bound : float
        Minimum value for B (after NNLS, for numerical stability)
    upper_bound : float, optional
        Upper bound for B values (default: None)
    ridge_lambda : float
        Ridge regularization (only used if use_scipy=False)
    debug : bool
        Print diagnostic information (default: False)
    use_scipy : bool
        Use scipy.optimize.nnls (true NNLS) vs lstsq+clamp

    Returns
    -------
    B : mx.array, shape (t,)
        Solution vector (non-negative weights)
    """
    n, t = M.shape

    # ✅ NEW: Use bounded LS if requested (论文推荐)
    if use_bounded:
        return _bounded_least_squares_scipy(M, y, bounds=bounds)

    # Convert to numpy for scipy (if using true NNLS)
    if use_scipy:
        try:
            import numpy as np
            from scipy.optimize import nnls as scipy_nnls

            M_np = np.array(M.tolist(), dtype=np.float32)
            y_np = np.array(y.tolist(), dtype=np.float32)

            if debug:
                print(f"[NNLS] Using scipy.optimize.nnls (true NNLS)")
                print(f"[NNLS] M shape: {M_np.shape}, y shape: {y_np.shape}")

            # True NNLS with increased maxiter and fallback
            # scipy.optimize.nnls默认maxiter = 3*n，可能不够
            # 增加到10*n以提高收敛率
            try:
                try:
                    B_np, residual_norm = scipy_nnls(M_np, y_np, maxiter=10*M_np.shape[1])
                except TypeError:
                    # Old scipy version without maxiter parameter
                    B_np, residual_norm = scipy_nnls(M_np, y_np)
            except RuntimeError as e:
                # NNLS failed to converge, fallback to lstsq + clip
                if debug:
                    print(f"[NNLS] scipy_nnls failed ({e}), using lstsq fallback")
                import warnings
                warnings.warn(f"NNLS convergence failed: {e}, using lstsq + clip fallback")
                B_np = np.linalg.lstsq(M_np, y_np, rcond=None)[0]
                B_np = np.clip(B_np, 0, None)  # Ensure non-negative
                residual_norm = np.linalg.norm(M_np @ B_np - y_np)

            if debug:
                print(f"[NNLS] Residual norm: {residual_norm:.4e}")
                print(f"[NNLS] B range: [{B_np.min():.4e}, {B_np.max():.4e}]")

            # Convert back to MLX
            B = mx.array(B_np)

        except ImportError:
            if debug:
                print(f"[NNLS] scipy not available, falling back to lstsq+clamp")
            use_scipy = False

    # Fallback: lstsq + clamp (if scipy not available)
    if not use_scipy:
        M_fp32 = M.astype(mx.float32)
        y_fp32 = y.astype(mx.float32)

        if debug:
            print(f"[NNLS] Using lstsq + clamp (not true NNLS)")

        # Normal equations with optional ridge
        MtM = M_fp32.T @ M_fp32
        Mty = M_fp32.T @ y_fp32

        if ridge_lambda > 0:
            MtM = MtM + ridge_lambda * mx.eye(t)

        with mx.stream(mx.cpu):
            B = mx.linalg.pinv(MtM) @ Mty

    # Apply lower bound for numerical stability
    if lower_bound is not None and lower_bound > 0:
        B = mx.maximum(B, lower_bound)

    # Apply upper bound if specified
    if upper_bound is not None:
        B = mx.minimum(B, upper_bound)

    return B


def _compute_C2_mlx(
    C1: mx.array,
    beta: mx.array,
    K: mx.array,
    V: mx.array,
    queries: mx.array,
    scale: float,
    ridge_lambda: float = 1e-6
) -> mx.array:
    """
    Compute C2 using LSQ fitting (ported from author's implementation).

    Solves: X @ C2 = Y
    where:
    - Y = softmax(Q·K^T / scale) @ V (original attention output)
    - X = softmax(Q·C1^T / scale + beta) (compressed attention weights)

    Parameters
    ----------
    C1 : mx.array, shape (t, d)
        Compressed keys
    beta : mx.array, shape (t,)
        Attention bias
    K : mx.array, shape (T, d)
        Original keys
    V : mx.array, shape (T, d)
        Original values
    queries : mx.array, shape (n, d)
        Query vectors
    scale : float
        Attention scale (sqrt(head_dim))
    ridge_lambda : float
        Ridge regularization (default: 1e-6)

    Returns
    -------
    C2 : mx.array, shape (t, d)
        Compressed values
    """
    n, d = queries.shape
    t = C1.shape[0]

    # Convert to fp32 for numerical stability
    queries_fp32 = queries.astype(mx.float32)
    K_fp32 = K.astype(mx.float32)
    V_fp32 = V.astype(mx.float32)
    C1_fp32 = C1.astype(mx.float32)
    beta_fp32 = beta.astype(mx.float32)

    # Y = softmax(Q·K^T / scale) @ V (original attention output)
    scores_K = (queries_fp32 @ K_fp32.T) / scale  # (n, T)
    max_K = mx.max(scores_K, axis=1, keepdims=True)  # (n, 1) for numerical stability
    exp_K = mx.exp(scores_K - max_K)  # (n, T)
    sum_exp_K = mx.sum(exp_K, axis=1, keepdims=True)  # (n, 1)
    attn_K = exp_K / sum_exp_K  # (n, T) normalized
    Y = attn_K @ V_fp32  # (n, d)

    # X = softmax(Q·C1^T / scale + beta) (compressed attention weights)
    scores_C = (queries_fp32 @ C1_fp32.T) / scale + beta_fp32  # (n, t)
    max_C = mx.max(scores_C, axis=1, keepdims=True)  # (n, 1)
    exp_C = mx.exp(scores_C - max_C)  # (n, t)
    sum_exp_C = mx.sum(exp_C, axis=1, keepdims=True)  # (n, 1)
    X = exp_C / sum_exp_C  # (n, t) normalized

    # Solve X @ C2 = Y using ridge regression
    # (X^T X + λI) C2 = X^T Y
    try:
        XtX = X.T @ X  # (t, t)
        XtY = X.T @ Y  # (t, d)

        # Add ridge regularization
        XtX_reg = XtX + ridge_lambda * mx.eye(t)

        # Ensure symmetry
        XtX_reg = 0.5 * (XtX_reg + XtX_reg.T)

        # Solve using pinv
        with mx.stream(mx.cpu):
            C2 = mx.linalg.pinv(XtX_reg) @ XtY

    except Exception as e:
        print(f"C2 LSQ solver failed: {e}")
        print(f"  Falling back to direct pinv with increased regularization")

        # Fallback: direct pseudoinverse
        ridge_lambda = 1e-4
        XtX = X.T @ X
        XtX_reg = XtX + ridge_lambda * mx.eye(t)
        XtX_reg = 0.5 * (XtX_reg + XtX_reg.T)
        XtY = X.T @ Y

        with mx.stream(mx.cpu):
            C2 = mx.linalg.pinv(XtX_reg) @ XtY

    # Convert back to original dtype
    C2 = C2.astype(K.dtype)

    return C2


def select_keys_omp_mlx(
    queries: mx.array,
    keys: mx.array,
    budget: int,
    scale: Optional[float] = None,
    k_choice: int = 1,
    verbose: bool = False
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    OMP (Orthogonal Matching Pursuit) key selection with iterative NNLS.

    Greedy algorithm that selects keys to best approximate the partition function.
    Ported from author's _select_keys_omp (omp.py lines 478-718).

    Parameters
    ----------
    queries : mx.array, shape (n, d)
        Query vectors
    keys : mx.array, shape (T, d)
        All keys
    budget : int
        Number of keys to select
    scale : float, optional
        Attention scale (default: sqrt(d))
    k_choice : int, default=1
        Number of keys to select per iteration (for speedup)
    verbose : bool, default=False
        Enable debug logging

    Returns
    -------
    C1 : mx.array, shape (budget, d)
        Selected keys
    beta : mx.array, shape (budget,)
        Log weights
    indices : mx.array, shape (budget,)
        Indices of selected keys

    Notes
    -----
    Algorithm:
    1. Compute exp_scores and target (partition function)
    2. Greedy loop:
       - Compute residual = target - current
       - Find key with highest correlation to residual
       - Solve NNLS: M @ B ≈ target
       - Update current = M @ B
    3. beta = log(B)
    """
    # Enable debug for first compression only
    global _DEBUG_COUNTER
    debug_this = _DEBUG_ENABLED and _DEBUG_COUNTER < 1

    n, d = queries.shape
    T = keys.shape[0]

    if debug_this:
        print(f"[OMP Debug] Input shapes: queries={queries.shape}, keys={keys.shape}")
        print(f"[OMP Debug] n={n}, d={d}, T={T}, budget={budget}")

    if scale is None:
        scale = d ** 0.5

    # Ensure budget < T
    if budget >= T:
        budget = max(T // 2, T - 1)

    # 1. Compute exp_scores and target (partition function)
    scores = (queries @ keys.T) / scale  # (n, T)
    max_scores = mx.max(scores, axis=1, keepdims=True)  # (n, 1)
    exp_scores = mx.exp(scores - max_scores)  # (n, T)
    target = mx.sum(exp_scores, axis=1)  # (n,)

    # 2. Greedy selection
    selected_indices = []
    current = mx.zeros_like(target)  # (n,)
    mask = mx.zeros(T, dtype=mx.bool_)  # Track selected keys

    for i in range(budget):
        # 2a. Compute residual
        residual = target - current  # (n,)

        # Debug: Check shapes on first iteration
        if i == 0 and debug_this:
            print(f"[OMP Debug Loop] exp_scores.shape={exp_scores.shape}, residual.shape={residual.shape}")
            print(f"[OMP Debug Loop] target.shape={target.shape}, current.shape={current.shape}")

        # 2b. Correlation of each key with residual
        # corr[j] = sum_i exp_scores[i,j] * residual[i]
        # Use matrix multiplication to avoid broadcasting issues: residual^T @ exp_scores
        # residual: (n,), exp_scores: (n, T) → corr: (T,)
        corr = residual @ exp_scores  # (T,)

        # Mask already selected keys
        corr = mx.where(mask, -1e9, corr)

        # 2c. Select top k_choice keys
        if k_choice == 1:
            idx = int(mx.argmax(corr).item())
            selected_indices.append(idx)
            mask = mx.where(mx.arange(T) == idx, True, mask)
        else:
            # Select top k_choice
            k = min(k_choice, budget - i)
            top_k_indices = mx.argsort(-corr)[:k]  # Descending order
            for idx in top_k_indices:
                idx_int = int(idx.item())
                if not mask[idx_int]:
                    selected_indices.append(idx_int)
                    mask = mx.where(mx.arange(T) == idx_int, True, mask)
                if len(selected_indices) >= budget:
                    break

        # 2d. Solve NNLS: M @ B ≈ target (with all keys selected so far)
        indices_array = mx.array(selected_indices)
        M = exp_scores[:, indices_array]  # (n, len(selected_indices))
        # Use small ridge to avoid SVD
        B = _nnls_mlx(M, target, lower_bound=None, ridge_lambda=1e-6)

        # 2e. Update current approximation
        current = M @ B

        if verbose and (i + 1) % 5 == 0:
            residual_norm = float(mx.norm(target - current).item())
            print(f"[OMP] Iteration {i+1}/{budget}: residual_norm={residual_norm:.6f}")

    # 3. Final NNLS solve with exactly budget keys to get correct beta
    # (In case we selected more than budget keys due to k_choice > 1)
    indices_array = mx.array(selected_indices[:budget])
    C1 = keys[indices_array]  # (budget, d)

    # Solve NNLS one final time with exactly budget keys
    M_final = exp_scores[:, indices_array]  # (n, budget)

    # Increment debug counter (global already declared at function start)
    _DEBUG_COUNTER += 1

    # ✅ CRITICAL FIX: Use bounded LS to solve beta directly (论文方法)
    # 不再是：NNLS → B → log(B) → beta
    # 而是：Bounded LS → beta ∈ [-3, 3]
    #
    # 论文中明确使用 bounded least-squares，约束 β ∈ [-3, 3]
    # 这避免了 log(0) 和极端负值的问题
    use_bounded_ls = True  # 启用论文的方法

    if use_bounded_ls:
        # 直接求解 beta，约束在 [-3, 3]
        beta = _nnls_mlx(
            M_final, target,
            use_bounded=True,
            bounds=(-3.0, 3.0),
            debug=debug_this or verbose
        )

        if verbose or debug_this:
            print(f"[OMP] Using bounded LS: beta ∈ [-3, 3]")
    else:
        # 旧方法：NNLS + log (已废弃)
        B_final = _nnls_mlx(M_final, target, lower_bound=None, ridge_lambda=1e-6, debug=debug_this or verbose)
        B_min = 0.01
        B_clamped = mx.maximum(B_final, B_min)
        beta = mx.log(B_clamped)

    if verbose or debug_this:
        print(f"[OMP] B_final range: min={float(B_final.min()):.6e}, max={float(B_final.max()):.6e}")
        print(f"[OMP] Beta range: min={float(beta.min()):.4f}, max={float(beta.max()):.4f}, mean={float(beta.mean()):.4f}")

        # Count how many B values were clipped
        num_clipped = int((B_final < B_min).sum())
        if num_clipped > 0:
            print(f"[OMP] Warning: {num_clipped}/{budget} B values clipped (< {B_min:.0e})")

    return C1, beta, indices_array


def select_keys_attention_aware(
    queries: mx.array,
    keys: mx.array,
    budget: int,
    scale: Optional[float] = None
) -> mx.array:
    """
    Select keys based on normalized attention weights (matching author's implementation).

    Computes softmax attention weights and selects top-k keys with highest average attention.

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
    Following the author's HighestAttentionKeysCompaction:
    - Computes softmax attention weights (normalized)
    - Aggregates across queries: mean of normalized attention
    - Selects top-k keys with highest average attention
    - Time complexity: O(query_len * seq_len * head_dim)
    """
    query_len, head_dim = queries.shape
    seq_len = keys.shape[0]

    if scale is None:
        scale = head_dim ** 0.5

    # Compute attention scores: (query_len, seq_len)
    scores = (queries @ keys.T) / scale

    # Compute softmax attention weights (normalized)
    # Following author's approach: per-query max normalization
    max_scores = mx.max(scores, axis=1, keepdims=True)  # (query_len, 1)
    exp_scores = mx.exp(scores - max_scores)  # (query_len, seq_len)
    sum_exp = mx.sum(exp_scores, axis=1, keepdims=True)  # (query_len, 1)
    attention_weights = exp_scores / sum_exp  # (query_len, seq_len) - normalized

    # Aggregate attention across queries: MEAN of normalized attention
    # This gives equal weight to each query (vs sum which favors high-score queries)
    key_scores = mx.mean(attention_weights, axis=0)  # (seq_len,)

    # Select top-k keys
    if budget >= seq_len:
        # If budget >= seq_len, return all indices
        return mx.arange(seq_len)

    # argsort returns indices in ascending order, we want descending
    # So we negate the scores
    sorted_indices = mx.argsort(-key_scores)
    selected_indices = sorted_indices[:budget]

    # ✅ CRITICAL FIX: Preserve chronological order to maintain RoPE consistency!
    #
    # Problem: Qwen3 uses RoPE (Rotary Position Embeddings) applied at generation time
    # based on cache.offset. Each cached key has RoPE(original_position) baked in.
    #
    # If we reorder keys by attention score, we create a mismatch:
    # - Key k50 with RoPE(50) is now at cache position 2
    # - New query at position 148 with RoPE(148) attends to key with RoPE(50) at position 2
    # - This breaks the relative position encoding!
    #
    # Solution: Select top-k by importance, but keep them in chronological order.
    # This way RoPE(i) at cache position i always corresponds to the original position.
    selected_indices = mx.sort(selected_indices)  # MUST keep chronological order for RoPE

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

    # CRITICAL FIX #1: Ensure budget < seq_len to avoid impossible selection
    # Bug: If budget >= seq_len, topk will fail or NNLS will be rank-deficient
    original_budget = budget
    if budget >= seq_len:
        budget = max(seq_len // 2, seq_len - 1)
        # Note: This is a conservative fix, actual usage should adjust budget externally

    # Step 1: OMP key selection with iterative NNLS
    # This returns C1, beta, and indices all at once
    # Beta is already fitted during OMP (no need to recompute)
    use_omp = True  # Flag to enable OMP (vs attention-aware)

    if use_omp:
        # OMP greedy selection + iterative NNLS
        C1, beta_omp, indices = select_keys_omp_mlx(
            queries, keys, budget, scale,
            k_choice=1,  # Greedy (select 1 key per iteration)
            verbose=False
        )

        # OMP already computed beta during selection
        # Decide whether to use fitted beta or force to zero
        if fit_beta:
            beta = beta_omp
            print(f"[OMP] Using fitted beta: min={float(beta.min()):.4f}, max={float(beta.max()):.4f}, mean={float(beta.mean()):.4f}")
        else:
            # zerobeta mode
            beta = mx.zeros_like(beta_omp)
            print(f"[OMP] zerobeta mode: forcing beta=0")
    else:
        # Fallback: Attention-aware selection (old method)
        indices = select_keys_attention_aware(queries, keys, budget, scale)
        C1 = keys[indices]  # (budget, head_dim)

        # Compute original attention output (for comparison)
        original_output = compute_attention_output(
            queries, keys, values, beta=None, scale=scale
        )

        # Step 2: Fit beta (if enabled)
        if fit_beta:
            # Following the author's OMP algorithm (base.py lines 472-605):
            # Solve: M @ B ≈ target, where:
            #   - M = exp_scores_compressed (query_len, budget)
            #   - B = exp(beta) (budget,) - PER-KEY weights
            #   - target = sum of all exp_scores (partition function)

            # Compute exp scores with numerical stability
            original_scores = (queries @ keys.T) / scale  # (query_len, seq_len)
            max_scores = mx.max(original_scores, axis=1, keepdims=True)  # (query_len, 1)
            exp_scores_original = mx.exp(original_scores - max_scores)  # (query_len, seq_len)

            # Select columns for compressed keys
            exp_scores_compressed = exp_scores_original[:, indices]  # (query_len, budget)

            # Target: total attention mass (partition function) for each query
            target = mx.sum(exp_scores_original, axis=1)  # (query_len,)

            # Solve NNLS: M @ B ≈ target, B >= 0
            M = exp_scores_compressed  # (query_len, budget)
            B = _nnls_mlx(M, target, lower_bound=1e-12, ridge_lambda=1e-6)

            # Beta = log(B)
            # Clamp B to avoid log(0)
            B = mx.maximum(B, 1e-12)
            beta = mx.log(B).astype(keys.dtype)

            print(f"[Attention-Aware] Beta fitted: min={float(beta.min()):.6f}, max={float(beta.max()):.6f}, mean={float(beta.mean()):.6f}")
        else:
            beta = mx.zeros(budget, dtype=keys.dtype)

    # Step 3: Fit C2 (if enabled)
    if fit_c2:
        # Following the author's _compute_C2 (base.py lines 61-240):
        # Solve: X @ C2 = Y, where:
        #   - Y = softmax(Q·K^T / scale) @ V (original attention output)
        #   - X = softmax(Q·C1^T / scale + beta) (compressed attention weights)

        C2 = _compute_C2_mlx(
            C1=C1,
            beta=beta,
            K=keys,
            V=values,
            queries=queries,
            scale=scale,
            ridge_lambda=1e-6
        )

        print(f"[AM Debug] C2 fitted: shape={C2.shape}, dtype={C2.dtype}")
    else:
        # Direct copy (fallback)
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
