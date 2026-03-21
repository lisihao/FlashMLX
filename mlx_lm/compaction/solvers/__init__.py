"""
KV Cache Compaction - Math Solvers

提供压缩算法所需的数学求解器。
"""

from .utils import (
    cholesky_solve_mlx,
    spectral_norm_mlx,
    spectral_norm_squared_mlx,
    lstsq_mlx,
    softmax_mlx,
    clip_mlx,
    safe_softmax,
    compute_in_fp32,
)

from .nnls import (
    nnls_clamped,
    nnls_pgd,
    nnls_auto,
)

from .lsq import (
    compute_C2_lstsq,
    compute_C2_cholesky,
    compute_C2_pinv,
    compute_C2_auto,
    evaluate_C2_quality,
)

__all__ = [
    "cholesky_solve_mlx",
    "spectral_norm_mlx",
    "spectral_norm_squared_mlx",
    "lstsq_mlx",
    "softmax_mlx",
    "clip_mlx",
    "safe_softmax",
    "compute_in_fp32",
    "nnls_clamped",
    "nnls_pgd",
    "nnls_auto",
    "compute_C2_lstsq",
    "compute_C2_cholesky",
    "compute_C2_pinv",
    "compute_C2_auto",
    "evaluate_C2_quality",
]
