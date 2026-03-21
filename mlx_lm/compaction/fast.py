"""
Fast Path Compaction Algorithm

快速压缩算法，优先考虑速度而非最优质量。

特点:
- 无 query generation（0ms）
- Recent + Stride key selection（~10ms）
- Beta = 0（跳过 NNLS）
- C2 = direct（直接复制 values）

时间: < 2s for 60K tokens on M4 Pro
压缩比: 5x
质量: 需验证（< 10% error）
"""

import mlx.core as mx
from .base import select_keys_recent_stride, compute_attention_error


def compact_single_head_fast(
    K: mx.array,
    V: mx.array,
    budget: int,
    recent_ratio: float = 0.25,
    return_indices: bool = False
) -> tuple:
    """
    Fast Path 单头压缩

    算法流程:
    1. Select keys: Recent + Stride selection
    2. Extract C1: C1 = K[indices]
    3. Beta = 0: 跳过 NNLS 优化
    4. Extract C2: C2 = V[indices]

    Parameters
    ----------
    K : mx.array, shape (seq_len, head_dim)
        Original keys
    V : mx.array, shape (seq_len, head_dim)
        Original values
    budget : int
        压缩后的 token 数量（seq_len * compression_ratio）
    recent_ratio : float, default=0.25
        保留最近 tokens 的比例
    return_indices : bool, default=False
        是否返回选中的 indices

    Returns
    -------
    C1 : mx.array, shape (budget, head_dim)
        压缩后的 keys
    beta : mx.array, shape (budget,)
        Attention bias (Fast Path 固定为 0)
    C2 : mx.array, shape (budget, head_dim)
        压缩后的 values
    indices : mx.array, shape (budget,), optional
        选中的 token positions（如果 return_indices=True）

    Time Complexity
    ---------------
    O(budget) - 仅索引操作，无矩阵运算

    Examples
    --------
    >>> K = mx.random.normal((1000, 128))
    >>> V = mx.random.normal((1000, 128))
    >>> C1, beta, C2 = compact_single_head_fast(K, V, budget=200)
    >>> C1.shape, beta.shape, C2.shape
    ((200, 128), (200,), (200, 128))
    """
    seq_len, head_dim = K.shape

    # 1. Select keys: Recent + Stride
    indices = select_keys_recent_stride(seq_len, budget, recent_ratio)

    # 2. Extract C1 (compacted keys)
    C1 = K[indices]

    # 3. Beta = 0 (skip NNLS)
    beta = mx.zeros(budget)

    # 4. Extract C2 (compacted values)
    C2 = V[indices]

    if return_indices:
        return C1, beta, C2, indices
    else:
        return C1, beta, C2


def compact_single_head_fast_with_queries(
    K: mx.array,
    V: mx.array,
    queries: mx.array,
    budget: int,
    recent_ratio: float = 0.25,
    scale: float = None
) -> dict:
    """
    Fast Path 压缩并评估质量

    与 compact_single_head_fast 相同，但额外计算误差指标。

    Parameters
    ----------
    K : mx.array, shape (seq_len, head_dim)
    V : mx.array, shape (seq_len, head_dim)
    queries : mx.array, shape (n_queries, head_dim)
        用于评估质量的 queries
    budget : int
    recent_ratio : float, default=0.25
    scale : float, optional

    Returns
    -------
    result : dict
        包含:
        - 'C1': 压缩后的 keys
        - 'beta': Attention bias
        - 'C2': 压缩后的 values
        - 'indices': 选中的 indices
        - 'metrics': 误差指标 dict

    Examples
    --------
    >>> K = mx.random.normal((1000, 128))
    >>> V = mx.random.normal((1000, 128))
    >>> Q = mx.random.normal((10, 128))
    >>> result = compact_single_head_fast_with_queries(K, V, Q, budget=200)
    >>> print(f"Relative error: {result['metrics']['relative_error']:.4f}")
    """
    # Compress
    C1, beta, C2, indices = compact_single_head_fast(
        K, V, budget, recent_ratio, return_indices=True
    )

    # Evaluate
    metrics = compute_attention_error(
        queries, K, V, C1, C2, beta, scale
    )

    return {
        'C1': C1,
        'beta': beta,
        'C2': C2,
        'indices': indices,
        'metrics': metrics
    }


def compact_multi_head_fast(
    K: mx.array,
    V: mx.array,
    budget: int,
    recent_ratio: float = 0.25
) -> tuple:
    """
    Fast Path 多头压缩

    对每个 head 独立压缩。

    Parameters
    ----------
    K : mx.array, shape (num_heads, seq_len, head_dim)
        Multi-head keys
    V : mx.array, shape (num_heads, seq_len, head_dim)
        Multi-head values
    budget : int
        每个 head 的压缩后 token 数量
    recent_ratio : float, default=0.25

    Returns
    -------
    C1 : mx.array, shape (num_heads, budget, head_dim)
        压缩后的 keys
    beta : mx.array, shape (num_heads, budget)
        Attention bias
    C2 : mx.array, shape (num_heads, budget, head_dim)
        压缩后的 values

    Examples
    --------
    >>> K = mx.random.normal((32, 1000, 128))  # 32 heads
    >>> V = mx.random.normal((32, 1000, 128))
    >>> C1, beta, C2 = compact_multi_head_fast(K, V, budget=200)
    >>> C1.shape
    (32, 200, 128)
    """
    num_heads, seq_len, head_dim = K.shape

    # 对每个 head 独立压缩
    C1_list = []
    beta_list = []
    C2_list = []

    for head_idx in range(num_heads):
        K_head = K[head_idx]  # (seq_len, head_dim)
        V_head = V[head_idx]

        C1_head, beta_head, C2_head = compact_single_head_fast(
            K_head, V_head, budget, recent_ratio
        )

        C1_list.append(C1_head[None, :, :])    # (1, budget, head_dim)
        beta_list.append(beta_head[None, :])   # (1, budget)
        C2_list.append(C2_head[None, :, :])

    # Concatenate
    C1 = mx.concatenate(C1_list, axis=0)  # (num_heads, budget, head_dim)
    beta = mx.concatenate(beta_list, axis=0)
    C2 = mx.concatenate(C2_list, axis=0)

    return C1, beta, C2


def estimate_compression_time(
    seq_len: int,
    budget: int,
    num_heads: int = 1
) -> dict:
    """
    估计 Fast Path 的压缩时间

    基于经验数据（M4 Pro）。

    Parameters
    ----------
    seq_len : int
    budget : int
    num_heads : int, default=1

    Returns
    -------
    timing : dict
        包含各组件的时间估计（秒）

    Examples
    --------
    >>> timing = estimate_compression_time(60000, 12000, num_heads=32)
    >>> print(f"Total time: {timing['total']:.2f}s")
    """
    # 经验常数（基于 profiling）
    time_per_selection = 0.00001  # 10 us per selection
    time_per_extract = 0.00002    # 20 us per extraction

    # Per-head time
    selection_time = time_per_selection * seq_len
    c1_extract_time = time_per_extract * budget
    c2_extract_time = time_per_extract * budget
    beta_time = 0.000001  # zeros() is instant

    per_head_time = selection_time + c1_extract_time + c2_extract_time + beta_time

    # Total time
    total_time = per_head_time * num_heads

    return {
        'selection': selection_time * num_heads,
        'c1_extract': c1_extract_time * num_heads,
        'beta': beta_time * num_heads,
        'c2_extract': c2_extract_time * num_heads,
        'per_head': per_head_time,
        'total': total_time
    }
