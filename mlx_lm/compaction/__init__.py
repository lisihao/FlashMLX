"""
KV Cache Compaction for MLX

实现基于 Attention Matching 的 KV Cache 压缩算法。

主要组件:
- solvers: 数学求解器（NNLS, LSQ, utils）
- base: 基础函数（attention computation, key selection）
- fast: Fast Path 压缩算法
"""

__version__ = "0.1.0"

# Export solvers
from .solvers import (
    # Utils
    cholesky_solve_mlx,
    spectral_norm_mlx,
    lstsq_mlx,
    clip_mlx,
    safe_softmax,
    # NNLS
    nnls_clamped,
    nnls_pgd,
    nnls_auto,
    # LSQ
    compute_C2_lstsq,
    compute_C2_cholesky,
    compute_C2_auto,
    evaluate_C2_quality,
)

# Export base functions
from .base import (
    compute_attention_output,
    compute_attention_error,
    select_keys_recent_stride,
    visualize_key_selection,
)

# Export Fast Path
from .fast import (
    compact_single_head_fast,
    compact_single_head_fast_with_queries,
    compact_multi_head_fast,
    estimate_compression_time,
)

__all__ = [
    # Solvers
    "cholesky_solve_mlx",
    "spectral_norm_mlx",
    "lstsq_mlx",
    "clip_mlx",
    "safe_softmax",
    "nnls_clamped",
    "nnls_pgd",
    "nnls_auto",
    "compute_C2_lstsq",
    "compute_C2_cholesky",
    "compute_C2_auto",
    "evaluate_C2_quality",
    # Base
    "compute_attention_output",
    "compute_attention_error",
    "select_keys_recent_stride",
    "visualize_key_selection",
    # Fast Path
    "compact_single_head_fast",
    "compact_single_head_fast_with_queries",
    "compact_multi_head_fast",
    "estimate_compression_time",
]
