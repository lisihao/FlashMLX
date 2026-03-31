"""
NNLS solver - Direct port from author's implementation
Source: https://github.com/adamzweiger/compaction
File: compaction/algorithms/base.py:472-590

Only changes: torch → mx, dim → axis
"""

import mlx.core as mx
from typing import Optional


def nnls_pg_author(
    M: mx.array,
    y: mx.array,
    iters: int = 0,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    debug: bool = False
) -> mx.array:
    """
    Box-constrained non-negative least squares solver with projected gradient.
    Direct port from author's _nnls_pg implementation.

    If iters == 0: Use ridge normal-equations solve + clamp with clamping to bounds.
    If iters > 0: Use projected-gradient descent with specified iterations.

    Solves: min_B 0.5 * ||M B - y||_2^2  s.t. lower_bound <= B <= upper_bound
    Step size 1/L with L ≈ ||M||_2^2 via power iteration.
    Expects fp32 inputs; returns fp32 B with box constraints.

    Parameters
    ----------
    M : array, shape (n, t)
        Design matrix
    y : array, shape (n,)
        Target vector
    iters : int
        Number of projected gradient iterations (0 = use clamped least squares)
    lower_bound : float, optional
        Lower bound for B values (default: 1e-12)
    upper_bound : float, optional
        Upper bound for B values (default: None, no upper bound)
    debug : bool
        Print debug information (default: False)

    Returns
    -------
    B : array, shape (t,)
        Solution vector with box constraints
    """
    n, t = M.shape
    min_val = 1e-12 if lower_bound is None else lower_bound

    # MLX limitation: linalg operations not supported on GPU
    # When iters > 0, use numpy lstsq for initialization (same as author)
    # Convert to numpy, solve, convert back
    import numpy as np

    if iters > 0:
        # Use numpy lstsq for initialization (same as author's approach)
        M_np = np.array(M, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)

        try:
            # Solve: min ||M @ B - y||^2
            B_init_np, _, _, _ = np.linalg.lstsq(M_np, y_np, rcond=None)
            B = mx.array(B_init_np, dtype=M.dtype)

            if debug:
                print(f"[NNLS Debug] Initialized with lstsq (numpy)")
                print(f"[NNLS Debug] B_init range: [{np.min(B_init_np):.6e}, {np.max(B_init_np):.6e}]")

        except Exception as e:
            # Fallback: use heuristic initialization
            if debug:
                print(f"[NNLS Debug] lstsq failed: {e}, using heuristic initialization")
            B_init_val = float(mx.mean(y) / (mx.mean(M) + 1e-12))
            B = mx.ones((t,), dtype=M.dtype) * max(B_init_val, min_val)

    else:
        # For iters == 0, use lstsq and clamp (same as author)
        M_np = np.array(M, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)

        try:
            B_init_np, _, _, _ = np.linalg.lstsq(M_np, y_np, rcond=None)
            B = mx.array(B_init_np, dtype=M.dtype)

            if debug:
                print(f"[NNLS Debug] Initialized with lstsq (numpy), iters=0")

        except Exception as e:
            if debug:
                print(f"[NNLS Debug] lstsq failed: {e}, using heuristic initialization")
            B_init_val = float(mx.mean(y) / (mx.mean(M) + 1e-12))
            B = mx.ones((t,), dtype=M.dtype) * max(B_init_val, min_val)

    # Debug: print statistics before clamping
    if debug:
        n_total = B.size
        n_below_min = int(mx.sum(B < min_val))
        n_above_upper = int(mx.sum(B > upper_bound)) if upper_bound is not None else 0
        print(f"[NNLS Debug] Before clamping: total_values={n_total}, below_min={n_below_min}, above_upper={n_above_upper}")
        print(f"[NNLS Debug] B range before clamping: min={float(mx.min(B)):.6e}, max={float(mx.max(B)):.6e}")

    # Apply bounds
    # Author: B = B.clamp_min_(min_val)
    B = mx.maximum(B, min_val)
    if upper_bound is not None:
        B = mx.minimum(B, upper_bound)

    # Debug: print B range after clamping
    if debug:
        print(f"[NNLS Debug] B range after clamping: min={float(mx.min(B)):.6e}, max={float(mx.max(B)):.6e}")

    if iters == 0:
        if debug:
            residual = M @ B - y
            loss = float(mx.sum(residual ** 2))
            print(f"[NNLS Debug] Initial solution (iters=0): loss={loss:.6e}")
        return B

    # Power iteration for spectral norm
    # Author: u = torch.randn(t, device=M.device, dtype=M.dtype)
    u = mx.random.normal((t,), dtype=M.dtype)
    u = u / (mx.linalg.norm(u) + 1e-12)

    for _ in range(3):  # converges very fast usually
        v = M @ u
        if mx.linalg.norm(v) == 0:
            break
        u_new = M.T @ v
        u_new = u_new / (mx.linalg.norm(u_new) + 1e-12)
        u = u_new

    # Estimate Lipschitz constant L ≈ ||M||_2^2
    Mu = M @ u
    L = float((mx.linalg.norm(Mu)) ** 2)

    if L == 0:
        # M is zero matrix, return clamped zeros
        return mx.maximum(mx.zeros((t,), dtype=M.dtype), min_val)

    # Projected gradient descent
    step_size = 1.0 / L

    for it in range(iters):
        # Gradient: grad = M^T (M B - y)
        grad = M.T @ (M @ B - y)

        # Gradient descent step
        B_new = B - step_size * grad

        # Project to box constraints
        B_new = mx.maximum(B_new, min_val)
        if upper_bound is not None:
            B_new = mx.minimum(B_new, upper_bound)

        B = B_new

    if debug:
        residual = M @ B - y
        loss = float(mx.sum(residual ** 2))
        print(f"[NNLS Debug] Final solution (iters={iters}): loss={loss:.6e}")

    return B
