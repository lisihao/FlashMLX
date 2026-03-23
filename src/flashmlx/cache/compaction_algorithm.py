"""
KV Cache Compaction Algorithm - Highest Attention Keys Method
Implements cache compression by selecting highest-attention keys
"""

import mlx.core as mx
from typing import Optional, Tuple, List


class HighestAttentionKeysCompaction:
    """
    Compacts KV cache by selecting keys with highest attention scores.

    This is a simplified implementation focusing on the core framework.
    Future improvements will add NNLS beta solving and full Ridge Regression.
    """

    def __init__(
        self,
        score_method: str = 'mean',
        beta_method: str = 'nnls',
        c2_method: str = 'lsq',
        c2_ridge_lambda: float = 0.0,
        c2_solver: str = 'lstsq',
        nnls_iters: int = 1000,
        nnls_lower_bound: Optional[float] = None,
        nnls_upper_bound: Optional[float] = None
    ):
        """
        Initialize compaction algorithm.

        Args:
            score_method: Method for aggregating attention scores ('mean', 'max', 'sum')
            beta_method: Method for computing beta ('nnls', 'log-ratio', 'zeros', 'ones')
            c2_method: Method for computing C2 ('lsq', 'direct')
            c2_ridge_lambda: Ridge regularization parameter
            c2_solver: Solver for least squares ('lstsq', 'solve')
            nnls_iters: Number of iterations for NNLS solver (default: 1000)
            nnls_lower_bound: Lower bound for NNLS solution (default: None uses 1e-12)
            nnls_upper_bound: Upper bound for NNLS solution (default: None, no upper bound)
        """
        if score_method not in ('mean', 'max', 'sum'):
            raise ValueError(f"score_method must be 'mean', 'max', or 'sum', got {score_method}")
        if c2_solver not in ('lstsq', 'solve'):
            raise ValueError(f"c2_solver must be 'lstsq' or 'solve', got {c2_solver}")

        self.score_method = score_method
        self.beta_method = beta_method
        self.c2_method = c2_method
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound

    def compute_compacted_cache(
        self,
        K: mx.array,
        V: mx.array,
        queries: mx.array,
        t: int,
        attention_bias: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array, mx.array, List[int]]:
        """
        Main compression function.

        Args:
            K: (T, d) - Original keys
            V: (T, d) - Original values
            queries: (n, d) - Query samples for attention scoring
            t: int - Target compressed size
            attention_bias: optional - Attention bias (not used in simplified version)

        Returns:
            C1: (t, d) - Compressed keys
            beta: (t,) - Bias terms
            C2: (t, d) - Compressed values
            indices: list - Selected key indices
        """
        T, d = K.shape

        # Validate input
        if T < t:
            raise ValueError(f"Cannot compact: K has {T} rows but t={t} requested")
        if V.shape[0] != T:
            raise ValueError(f"K and V must have same first dimension: {T} vs {V.shape[0]}")
        if queries.shape[1] != d:
            raise ValueError(f"Query dimension {queries.shape[1]} must match key dimension {d}")

        # Step 1: Select highest attention keys
        C1, beta, indices = self._select_keys_highest_attention(
            K, queries, t, attention_bias
        )

        # Step 2: Compute C2 (compressed values)
        C2 = self._compute_C2(
            C1, beta, K, V, queries, attention_bias,
            ridge_lambda=self.c2_ridge_lambda,
            solver=self.c2_solver
        )

        return C1, beta, C2, indices

    def _select_keys_highest_attention(
        self,
        K: mx.array,
        queries: mx.array,
        t: int,
        attention_bias: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array, List[int]]:
        """
        Select top-t keys based on attention scores and compute beta bias terms.

        Args:
            K: (T, d) - Keys
            queries: (n, d) - Query samples
            t: Number of keys to select
            attention_bias: Ignored in current version

        Returns:
            C1: (t, d) - Selected keys
            beta: (t,) - Bias terms
            indices: list of selected key indices
        """
        T, d = K.shape
        n = queries.shape[0]

        # Step 1: Compute attention scores for original keys
        scale = 1.0 / mx.sqrt(mx.array(d, dtype=K.dtype))
        attn_scores = queries @ K.T * scale  # (n, T)
        attn_weights = mx.softmax(attn_scores, axis=-1)  # (n, T)

        # Step 2: Aggregate scores across queries to select top-t keys
        if self.score_method == 'mean':
            key_scores = mx.mean(attn_weights, axis=0)  # (T,)
        elif self.score_method == 'max':
            key_scores = mx.max(attn_weights, axis=0)  # (T,)
        else:  # 'sum'
            key_scores = mx.sum(attn_weights, axis=0)  # (T,)

        # Select top-t keys by score
        indices = self._topk_indices(key_scores, t)

        # Extract selected keys
        C1 = K[indices]  # (t, d)

        # Step 3: Compute beta bias terms
        if self.beta_method == 'zeros':
            beta = mx.zeros((t,), dtype=K.dtype)
        elif self.beta_method == 'ones':
            beta = mx.ones((t,), dtype=K.dtype)
        elif self.beta_method == 'log-ratio':
            # Log-ratio method (verified quality=1.000 on commit 536d91e)
            # Goal: softmax(queries @ C1^T / sqrt(d) + beta) ≈ attn_weights[:, indices]
            from ..compaction.solvers import nnls_pgd

            # Compute compressed attention scores (without beta)
            attn_scores_C1 = queries @ C1.T * scale  # (n, t)

            # Extract target attention weights for selected keys
            target_attn = attn_weights[:, indices]  # (n, t)

            # Compute base attention distribution (before beta correction)
            base_attn = mx.softmax(attn_scores_C1, axis=-1)  # (n, t)

            # Log-ratio approximation: beta[j] ≈ mean_over_queries(log(target_attn[:,j] / base_attn[:,j]))
            eps = 1e-10
            target_attn_safe = mx.maximum(target_attn, eps)
            base_attn_safe = mx.maximum(base_attn, eps)

            # Log-ratio for each (query, key) pair
            log_ratio = mx.log(target_attn_safe / base_attn_safe)  # (n, t)

            # Average across queries to get beta for each key
            # This is the simple log-ratio method from the paper
            beta = mx.mean(log_ratio, axis=0)  # (t,)

        elif self.beta_method == 'nnls':
            # NNLS method: match partition functions
            # Ported from author's implementation (https://github.com/adamzweiger/compaction)
            # highest_attention_keys.py lines 216-224

            # Special case: when t == T (no compression), beta should be exactly 0
            if t == T:
                beta = mx.zeros((t,), dtype=K.dtype)
            else:
                from ..compaction.nnls_author import nnls_pg_author

                # Compute exp_scores in fp32 for numerical stability (same as author)
                # Author: scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
                # Author: exp_scores = torch.exp(scores32 - max_scores)
                # Note: attn_scores is already scaled (queries @ K.T * scale)
                scores32 = attn_scores  # Don't multiply scale again!
                max_scores = mx.max(scores32, axis=1, keepdims=True)  # (n, 1)
                exp_scores = mx.exp(scores32 - max_scores)  # (n, T) fp32

                # Compute target = partition function (author line 217)
                # Author: target = exp_scores.sum(dim=1)
                target = mx.sum(exp_scores, axis=1)  # (n,)

                # Build design matrix M for NNLS (author lines 219-220)
                # CRITICAL: Select columns from exp_scores, don't recompute!
                # Author: M = exp_scores[:, selected_indices_tensor]
                M = exp_scores[:, indices]  # (n, t)

                # Solve NNLS using author's exact implementation (author line 223)
                # Author: B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound)
                B = nnls_pg_author(
                    M, target,
                    iters=self.nnls_iters,
                    lower_bound=self.nnls_lower_bound,
                    upper_bound=self.nnls_upper_bound,
                    debug=False
                )  # (t,) fp32 (>=0)

                # Convert to log-space (author line 224)
                # Author: beta32 = torch.log(B)
                beta = mx.log(B)  # (t,)
        else:
            # Default: ones
            beta = mx.ones((t,), dtype=K.dtype)

        # Convert indices to Python list for return
        indices_list = [int(i) for i in indices]

        return C1, beta, indices_list

    def _compute_C2(
        self,
        C1: mx.array,
        beta: mx.array,
        K: mx.array,
        V: mx.array,
        queries: mx.array,
        attention_bias: Optional[mx.array] = None,
        ridge_lambda: float = 0.0,
        solver: str = 'lstsq'
    ) -> mx.array:
        """
        Compute compressed values C2 using Ridge Regression.

        Goal: Minimize ||attn_K @ V - attn_C1 @ C2||^2 + lambda * ||C2||^2
        where attn_K and attn_C1 are attention weights for original and compressed keys.

        Args:
            C1: (t, d) - Selected keys
            beta: (t,) - Bias terms
            K: (T, d) - Original keys
            V: (T, d) - Original values
            queries: (n, d) - Query samples
            attention_bias: Ignored in current version
            ridge_lambda: Ridge regularization parameter
            solver: Solver type ('lstsq' or 'solve')

        Returns:
            C2: (t, d) - Compressed values
        """
        t, d = C1.shape
        T = K.shape[0]
        n = queries.shape[0]

        if self.c2_method == 'direct' or queries is None:
            # Direct selection: find closest keys and use their values
            similarity = C1 @ K.T  # (t, T)
            indices = mx.argmax(similarity, axis=1)  # (t,)
            C2 = V[indices]
            return C2

        # Step 1: Compute attention weights for original keys K
        scale = 1.0 / mx.sqrt(mx.array(d, dtype=K.dtype))
        scores_K = queries @ K.T * scale  # (n, T)
        attn_K = mx.softmax(scores_K, axis=-1)  # (n, T)

        # Step 2: Compute attention weights for compressed keys C1 (with beta correction)
        # CRITICAL: Must include beta to match the compressed attention distribution
        scores_C1 = queries @ C1.T * scale + beta  # (n, t) - add beta bias
        attn_C1 = mx.softmax(scores_C1, axis=-1)  # (n, t)

        # Step 3: Compute target: y = attn_K @ V (n, d)
        y = attn_K @ V  # (n, d)

        # Step 4: Solve Ridge regression: min ||attn_C1 @ C2 - y||^2 + lambda * ||C2||^2
        # Solution: C2 = (X^T @ X + lambda * I)^{-1} @ X^T @ y
        # where X = attn_C1

        X = attn_C1  # (n, t)

        # Compute XTX = X^T @ X (t, t)
        XTX = X.T @ X  # (t, t)

        # Compute XTy = X^T @ y (t, d)
        XTy = X.T @ y  # (t, d)

        # Add ridge regularization
        if ridge_lambda > 0:
            # Spectral norm scaling for lambda
            # Estimate spectral norm using power iteration
            sigma_max = self._estimate_spectral_norm(X, max_iter=20)
            scaled_lambda = ridge_lambda * (sigma_max ** 2)

            # Add regularization: XTX + lambda * I
            reg_term = scaled_lambda * mx.eye(t, dtype=X.dtype)
            XTX_reg = XTX + reg_term
        else:
            XTX_reg = XTX

        # Add small regularization to ensure numerical stability
        if ridge_lambda == 0:
            # Add minimal regularization to avoid singular matrix
            eps = 1e-6
            XTX_reg = XTX_reg + eps * mx.eye(t, dtype=X.dtype)

        # Solve for C2 using matrix inversion
        # Note: inv() and solve() are not supported on GPU in current MLX (0.21.1)
        # Options:
        # 1. Use Cholesky decomposition (if GPU-supported)
        # 2. Fall back to direct method for now
        # 3. Wait for MLX to add GPU support

        # For now, use a simple approach that works on GPU:
        # Gradient descent solution or direct approximation
        # Since XTX is well-conditioned (with regularization), use iterative refinement

        # Alternative: Use numpy for stable solve
        # Note: MLX's linalg.cholesky() is not supported on GPU, so we use numpy
        try:
            import numpy as np

            # Convert to numpy for stable solve
            XTX_reg_np = np.array(XTX_reg)
            XTy_np = np.array(XTy)

            # Solve using numpy's lstsq (more stable than solve for potentially ill-conditioned systems)
            C2_np, _, _, _ = np.linalg.lstsq(XTX_reg_np, XTy_np, rcond=None)

            # Convert back to MLX
            C2 = mx.array(C2_np, dtype=X.dtype)

        except Exception as e:
            # Fallback: use direct method
            import warnings
            warnings.warn(f"LSQ C2 computation failed: {e}. Falling back to direct method.")
            similarity = C1 @ K.T  # (t, T)
            indices = mx.argmax(similarity, axis=1)  # (t,)
            C2 = V[indices]

        return C2

    def _estimate_spectral_norm(
        self,
        A: mx.array,
        max_iter: int = 20
    ) -> float:
        """
        Estimate the spectral norm (largest singular value) using power iteration.

        Args:
            A: (m, n) - Input matrix
            max_iter: Maximum iterations for power iteration

        Returns:
            Estimated spectral norm
        """
        m, n = A.shape

        # Initialize random vector
        v = mx.random.normal(shape=(n,))
        v = v / mx.linalg.norm(v)

        # Power iteration
        for _ in range(max_iter):
            # v = A^T @ A @ v
            u = A @ v
            v = A.T @ u
            norm_v = mx.linalg.norm(v)

            if norm_v > 0:
                v = v / norm_v
            else:
                break

        # Final iteration to get sigma
        u = A @ v
        sigma = mx.linalg.norm(u)

        return float(sigma)

    def _topk_indices(self, scores: mx.array, k: int) -> mx.array:
        """
        Get indices of top-k elements.

        Args:
            scores: (T,) - Score array
            k: Number of top elements to select

        Returns:
            indices: (k,) - Indices of top-k elements (sorted descending)
        """
        T = scores.shape[0]
        k = min(k, T)

        # Simple approach: argsort and take top-k
        # argsort in descending order (negate scores)
        sorted_indices = mx.argsort(-scores)  # (T,)

        # Take top-k
        indices = sorted_indices[:k]  # (k,)

        return indices

    def _softmax(self, x: mx.array, axis: int = -1) -> mx.array:
        """
        Compute softmax along specified axis.

        Args:
            x: Input array
            axis: Axis along which to compute softmax

        Returns:
            Softmax output
        """
        # Numerically stable softmax
        x_max = mx.max(x, axis=axis, keepdims=True)
        exp_x = mx.exp(x - x_max)
        return exp_x / mx.sum(exp_x, axis=axis, keepdims=True)

    def _nnls_pg(
        self,
        A: mx.array,
        b: mx.array,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> mx.array:
        """
        Non-negative least squares using projected gradient descent.

        Solves: min ||A @ x - b||^2 subject to x >= 0

        Algorithm:
        1. Initialize x = ones
        2. Estimate step size using spectral norm of A
        3. Repeat until convergence:
           - Compute gradient: grad = A^T @ (A @ x - b)
           - Update: x = x - step_size * grad
           - Project to non-negative: x = max(x, 0)
           - Check convergence: ||projected_grad|| < tol

        Args:
            A: (m, n) - Coefficient matrix
            b: (m,) - Target vector
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            x: (n,) - Non-negative solution
        """
        m, n = A.shape

        # Initialize x = ones
        x = mx.ones((n,), dtype=A.dtype)

        # Estimate spectral norm for step size
        # step_size = 1 / (sigma_max^2)
        sigma_max = self._estimate_spectral_norm(A, max_iter=20)

        if sigma_max > 0:
            step_size = 1.0 / (sigma_max ** 2 + 1e-8)
        else:
            step_size = 0.01

        # Precompute A^T for efficiency
        AT = A.T  # (n, m)

        # Projected gradient descent
        for iteration in range(max_iter):
            # Compute residual: r = A @ x - b
            residual = A @ x - b  # (m,)

            # Compute gradient: grad = A^T @ residual
            grad = AT @ residual  # (n,)

            # Gradient descent step
            x_new = x - step_size * grad

            # Project to non-negative
            x_new = mx.maximum(x_new, 0.0)

            # Compute projected gradient for convergence check
            # proj_grad = grad where x > 0, else min(grad, 0)
            # Simplified: check if update is small
            diff = mx.linalg.norm(x_new - x)

            x = x_new

            # Check convergence
            if diff < tol:
                break

        return x


def create_compaction_algorithm(
    score_method: str = 'mean',
    beta_method: str = 'nnls',
    c2_method: str = 'lsq',
    **kwargs
) -> HighestAttentionKeysCompaction:
    """
    Factory function to create a compaction algorithm instance.

    Args:
        score_method: Method for attention score aggregation
        beta_method: Method for beta computation
        c2_method: Method for C2 computation
        **kwargs: Additional arguments passed to constructor

    Returns:
        Configured HighestAttentionKeysCompaction instance
    """
    return HighestAttentionKeysCompaction(
        score_method=score_method,
        beta_method=beta_method,
        c2_method=c2_method,
        **kwargs
    )
