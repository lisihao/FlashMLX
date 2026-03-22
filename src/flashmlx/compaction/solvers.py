"""
NNLS (Non-Negative Least Squares) Solvers for MLX

Implementations:
1. nnls_clamped: Fast approximate solver (clamp negative values)
2. nnls_pgd: Projected Gradient Descent (iterative optimization)
3. nnls_auto: Automatic solver selection based on quality requirement

All solvers solve: min_{x ≥ lower_bound} ||Mx - y||^2
"""

import mlx.core as mx
from typing import Optional


def nnls_clamped(
    M: mx.array,
    y: mx.array,
    lower_bound: float = 0.0
) -> mx.array:
    """
    Clamped Least Squares solver (fast approximate NNLS).

    Method:
    1. Solve unconstrained LSQ: x = (M^T M)^{-1} M^T y
    2. Clamp negative values: x = max(x, lower_bound)

    Args:
        M: Design matrix (m, n) - m equations, n unknowns
        y: Target vector (m,)
        lower_bound: Lower bound constraint (default: 0.0)

    Returns:
        x: Solution vector (n,) with x >= lower_bound

    Complexity: O(n^2 m + n^3) - dominated by matrix inversion
    Quality: Approximate (may not be optimal NNLS solution)
    """
    # Validate inputs
    if len(M.shape) != 2:
        raise ValueError(f"M must be 2D matrix, got shape {M.shape}")
    if len(y.shape) != 1:
        raise ValueError(f"y must be 1D vector, got shape {y.shape}")
    if M.shape[0] != y.shape[0]:
        raise ValueError(f"M and y dimensions mismatch: {M.shape[0]} vs {y.shape[0]}")

    m, n = M.shape

    # Compute M^T M and M^T y
    MTM = M.T @ M  # (n, n)
    MTy = M.T @ y  # (n,)

    # Add small regularization for numerical stability
    # This prevents singular matrix issues
    eps = 1e-8
    reg_term = eps * mx.eye(n, dtype=M.dtype)
    MTM_reg = MTM + reg_term

    # Solve (M^T M) x = M^T y
    # Use mx.linalg.solve instead of matrix inversion for stability
    try:
        x = mx.linalg.solve(MTM_reg, MTy)
    except Exception as e:
        # Fallback: use pseudo-inverse if solve fails
        # This can happen if MTM is nearly singular
        try:
            MTM_inv = mx.linalg.inv(MTM_reg)
            x = MTM_inv @ MTy
        except Exception:
            # Last resort: initialize with zeros
            x = mx.zeros((n,), dtype=M.dtype)

    # Clamp negative values
    x = mx.maximum(x, lower_bound)

    return x


def nnls_pgd(
    M: mx.array,
    y: mx.array,
    lower_bound: float = 0.0,
    max_iters: int = 100,
    tol: float = 1e-6,
    step_size: Optional[float] = None,
    verbose: bool = False
) -> mx.array:
    """
    Projected Gradient Descent NNLS solver (iterative optimization).

    Method:
    1. Initialize x = 0 (or project unconstrained solution)
    2. Repeat until convergence:
       - Compute gradient: grad = 2 * M^T (Mx - y)
       - Update: x = x - step_size * grad
       - Project: x = max(x, lower_bound)

    Args:
        M: Design matrix (m, n)
        y: Target vector (m,)
        lower_bound: Lower bound constraint (default: 0.0)
        max_iters: Maximum iterations (default: 100)
        tol: Convergence tolerance (default: 1e-6)
        step_size: Step size (default: auto-computed from Lipschitz constant)
        verbose: Print convergence info (default: False)

    Returns:
        x: Solution vector (n,) with x >= lower_bound

    Complexity: O(max_iters * m * n) - dominated by matrix-vector products
    Quality: High (converges to optimal NNLS solution)
    """
    # Validate inputs
    if len(M.shape) != 2:
        raise ValueError(f"M must be 2D matrix, got shape {M.shape}")
    if len(y.shape) != 1:
        raise ValueError(f"y must be 1D vector, got shape {y.shape}")
    if M.shape[0] != y.shape[0]:
        raise ValueError(f"M and y dimensions mismatch: {M.shape[0]} vs {y.shape[0]}")

    m, n = M.shape

    # Auto-compute step size using Lipschitz constant
    # L = ||M^T M||_2 ≈ largest eigenvalue of M^T M
    # Use power iteration to estimate ||M^T M||_2
    if step_size is None:
        # Estimate spectral norm of M^T M
        # ||M^T M||_2 ≤ ||M^T||_2 * ||M||_2 = ||M||_2^2
        # Use power iteration to estimate ||M||_2
        v = mx.random.normal((n,), dtype=M.dtype)
        v = v / mx.linalg.norm(v)

        for _ in range(10):  # 10 iterations usually sufficient
            Mv = M @ v
            MTMv = M.T @ Mv
            v_new = MTMv / mx.linalg.norm(MTMv)
            v = v_new

        # Estimate ||M||_2
        Mv = M @ v
        spectral_norm_M = mx.linalg.norm(Mv) / mx.linalg.norm(v)

        # Lipschitz constant L = ||M^T M||_2 ≈ ||M||_2^2
        L = spectral_norm_M ** 2

        # Step size = 1 / L (standard choice for gradient descent)
        step_size = 1.0 / (L + 1e-8)

    # Initialize x using clamped solution (warm start)
    x = nnls_clamped(M, y, lower_bound)

    # Precompute M^T M and M^T y for efficiency
    MTM = M.T @ M
    MTy = M.T @ y

    # Iterative optimization
    prev_loss = float('inf')

    for iteration in range(max_iters):
        # Compute gradient: grad = 2 * M^T (Mx - y) = 2 * (M^T M x - M^T y)
        grad = 2.0 * (MTM @ x - MTy)

        # Gradient descent update
        x_new = x - step_size * grad

        # Project to constraint: x >= lower_bound
        x_new = mx.maximum(x_new, lower_bound)

        # Check convergence
        residual = M @ x_new - y
        loss = float(mx.sum(residual ** 2))

        if verbose and iteration % 10 == 0:
            print(f"  [PGD iter {iteration:3d}] loss={loss:.6e}")

        # Convergence check: loss improvement < tol
        if abs(prev_loss - loss) < tol:
            if verbose:
                print(f"  [PGD] Converged at iteration {iteration} (loss={loss:.6e})")
            break

        prev_loss = loss
        x = x_new

    return x


def nnls_auto(
    M: mx.array,
    y: mx.array,
    lower_bound: float = 0.0,
    quality: str = 'medium'
) -> mx.array:
    """
    Automatic NNLS solver selection based on quality requirement.

    Quality levels:
    - 'fast': Use nnls_clamped (O(n^3), approximate)
    - 'medium': Use nnls_pgd with 50 iterations (balanced)
    - 'high': Use nnls_pgd with 100 iterations (optimal)

    Args:
        M: Design matrix (m, n)
        y: Target vector (m,)
        lower_bound: Lower bound constraint (default: 0.0)
        quality: Quality level ('fast', 'medium', 'high')

    Returns:
        x: Solution vector (n,) with x >= lower_bound
    """
    if quality == 'fast':
        return nnls_clamped(M, y, lower_bound)
    elif quality == 'medium':
        return nnls_pgd(M, y, lower_bound, max_iters=50, verbose=False)
    elif quality == 'high':
        return nnls_pgd(M, y, lower_bound, max_iters=100, verbose=False)
    else:
        raise ValueError(f"Invalid quality level: {quality}. Must be 'fast', 'medium', or 'high'")


# Expose public API
__all__ = [
    'nnls_clamped',
    'nnls_pgd',
    'nnls_auto'
]
