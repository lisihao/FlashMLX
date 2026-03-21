"""
Fast Path v2: Hybrid Recent + Random Sampling

改进 Fast Path 以处理非局部性数据。

核心改进：
1. Hybrid Selection：50% Recent + 50% Random uniform
   - Recent：保留最近的 tokens（局部性）
   - Random：均匀随机采样（覆盖全局）
2. 不依赖 query，避免 self-attention 偏差
3. 保持 O(budget) 复杂度

时间：< 2s for 60K tokens（与 v1 相同）
质量：在随机数据和局部性数据上都表现更好
"""

import mlx.core as mx
from .base import compute_attention_output, compute_attention_error
from .solvers import safe_softmax


def compact_single_head_fast_v2(
    K: mx.array,
    V: mx.array,
    budget: int,
    recent_ratio: float = 0.5,
    return_indices: bool = False
) -> tuple:
    """
    Fast Path v2: Hybrid Recent + Random selection

    算法流程：
    1. Select recent：保留最近的 recent_ratio * budget 个 tokens
    2. Select random：均匀随机采样 (1 - recent_ratio) * budget 个 tokens
    3. Merge：合并并排序
    4. Beta = 0（与 v1 相同，保持 Fast）
    5. Extract C2：直接复制 values

    Parameters
    ----------
    K : mx.array, shape (seq_len, head_dim)
    V : mx.array, shape (seq_len, head_dim)
    budget : int
    recent_ratio : float, default=0.5
        保留最近 tokens 的比例
        - v1 使用 0.25（过于偏向 recent）
        - v2 使用 0.5（平衡 recent 和 global）
    return_indices : bool, default=False

    Returns
    -------
    C1, beta, C2 [, indices]

    Time Complexity
    ---------------
    O(budget) - 仅索引操作

    Rationale
    ---------
    v1 的问题：
    - Recent (25%) + Stride (75%)：Stride 是确定性的，不能适应随机分布
    - 在随机数据中，stride 选择的 tokens 可能完全不重要

    v2 的改进：
    - Recent (50%) + Random (50%)：Random 可以均匀覆盖全局
    - 在局部性数据中：Recent 50% 足够捕获重要信息
    - 在随机数据中：Random 50% 提供了均匀覆盖
    - 在混合数据中：两者结合，平衡

    Examples
    --------
    >>> K = mx.random.normal((1000, 128))
    >>> V = mx.random.normal((1000, 128))
    >>> C1, beta, C2 = compact_single_head_fast_v2(K, V, budget=200)
    """
    seq_len, head_dim = K.shape

    # 1. Recent indices
    n_recent = int(budget * recent_ratio)
    recent_start = seq_len - n_recent
    recent_indices = mx.arange(recent_start, seq_len)

    # 2. Random indices (从非 recent 部分采样)
    n_random = budget - n_recent
    if n_random > 0 and recent_start > 0:
        # 均匀随机采样
        random_indices = mx.random.randint(0, recent_start, (n_random,))
    else:
        random_indices = mx.array([], dtype=mx.int32)

    # 3. Merge
    if random_indices.size > 0:
        indices = mx.concatenate([random_indices, recent_indices])
    else:
        indices = recent_indices

    # 确保 budget 个
    if indices.size > budget:
        indices = indices[:budget]
    elif indices.size < budget:
        extra = budget - indices.size
        extra_indices = mx.arange(0, extra)
        indices = mx.concatenate([extra_indices, indices])

    # 4. Extract C1
    C1 = K[indices]

    # 5. Beta = 0 (保持 Fast)
    beta = mx.zeros(budget)

    # 6. Extract C2
    C2 = V[indices]

    if return_indices:
        return C1, beta, C2, indices
    else:
        return C1, beta, C2


def compact_single_head_fast_v2_with_queries(
    K: mx.array,
    V: mx.array,
    queries: mx.array,
    budget: int,
    recent_ratio: float = 0.5,
    scale: float = None
) -> dict:
    """
    Fast Path v2 with quality evaluation

    Parameters
    ----------
    K, V : mx.array
    queries : mx.array, shape (n_queries, head_dim)
        真实的 queries（用于评估质量）
    budget : int
    recent_ratio : float, default=0.5
    scale : float, optional

    Returns
    -------
    result : dict
        包含 C1, beta, C2, indices, metrics
    """
    C1, beta, C2, indices = compact_single_head_fast_v2(
        K, V, budget, recent_ratio, return_indices=True
    )

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


def compact_multi_head_fast_v2(
    K: mx.array,
    V: mx.array,
    budget: int,
    recent_ratio: float = 0.5
) -> tuple:
    """
    Fast Path v2 multi-head compression

    Parameters
    ----------
    K : mx.array, shape (num_heads, seq_len, head_dim)
    V : mx.array, shape (num_heads, seq_len, head_dim)
    budget : int
    recent_ratio : float, default=0.5

    Returns
    -------
    C1, beta, C2
    """
    num_heads, seq_len, head_dim = K.shape

    C1_list = []
    beta_list = []
    C2_list = []

    for head_idx in range(num_heads):
        K_head = K[head_idx]
        V_head = V[head_idx]

        C1_head, beta_head, C2_head = compact_single_head_fast_v2(
            K_head, V_head, budget, recent_ratio
        )

        C1_list.append(C1_head[None, :, :])
        beta_list.append(beta_head[None, :])
        C2_list.append(C2_head[None, :, :])

    C1 = mx.concatenate(C1_list, axis=0)
    beta = mx.concatenate(beta_list, axis=0)
    C2 = mx.concatenate(C2_list, axis=0)

    return C1, beta, C2
