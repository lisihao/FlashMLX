"""
KV Cache Compaction - 基础函数

提供压缩算法的核心辅助函数。
"""

import mlx.core as mx
from .solvers import safe_softmax


def compute_attention_output(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    beta: mx.array = None,
    scale: float = None
) -> mx.array:
    """
    计算 attention 输出

    支持两种模式：
    1. 标准 attention: output = softmax(Q·K^T / scale) · V
    2. 带 beta attention: output = softmax(Q·K^T / scale + beta) · V

    Parameters
    ----------
    queries : mx.array, shape (n_queries, head_dim)
        Query vectors
    keys : mx.array, shape (seq_len, head_dim)
        Key vectors
    values : mx.array, shape (seq_len, head_dim)
        Value vectors
    beta : mx.array, shape (seq_len,), optional
        Attention bias (用于压缩后的 attention)
    scale : float, optional
        Attention scale (默认: sqrt(head_dim))

    Returns
    -------
    output : mx.array, shape (n_queries, head_dim)
        Attention output

    Examples
    --------
    >>> Q = mx.random.normal((10, 128))
    >>> K = mx.random.normal((1000, 128))
    >>> V = mx.random.normal((1000, 128))
    >>> output = compute_attention_output(Q, K, V)
    >>> output.shape
    (10, 128)
    """
    n_queries, head_dim = queries.shape
    seq_len = keys.shape[0]

    # 默认 scale
    if scale is None:
        scale = head_dim ** 0.5

    # Compute attention scores: Q @ K^T / scale
    scores = queries @ keys.T / scale  # (n_queries, seq_len)

    # Add beta if provided
    if beta is not None:
        scores = scores + beta[None, :]  # Broadcast beta to (1, seq_len)

    # Softmax
    attn_weights = safe_softmax(scores, axis=1)  # (n_queries, seq_len)

    # Weighted sum: attn_weights @ V
    output = attn_weights @ values  # (n_queries, head_dim)

    return output


def compute_attention_error(
    queries: mx.array,
    keys_original: mx.array,
    values_original: mx.array,
    keys_compacted: mx.array,
    values_compacted: mx.array,
    beta: mx.array = None,
    scale: float = None
) -> dict:
    """
    计算压缩前后的 attention 误差

    Parameters
    ----------
    queries : mx.array, shape (n_queries, head_dim)
    keys_original : mx.array, shape (seq_len, head_dim)
    values_original : mx.array, shape (seq_len, head_dim)
    keys_compacted : mx.array, shape (budget, head_dim)
    values_compacted : mx.array, shape (budget, head_dim)
    beta : mx.array, shape (budget,), optional
    scale : float, optional

    Returns
    -------
    metrics : dict
        包含以下指标:
        - 'mse': 均方误差
        - 'relative_error': 相对误差
        - 'max_error': 最大误差
        - 'mean_abs_error': 平均绝对误差

    Examples
    --------
    >>> Q = mx.random.normal((10, 128))
    >>> K = mx.random.normal((1000, 128))
    >>> V = mx.random.normal((1000, 128))
    >>> C1 = K[:200]  # 压缩到 200
    >>> C2 = V[:200]
    >>> metrics = compute_attention_error(Q, K, V, C1, C2)
    """
    # Original attention output
    output_original = compute_attention_output(queries, keys_original, values_original, scale=scale)

    # Compacted attention output
    output_compacted = compute_attention_output(queries, keys_compacted, values_compacted, beta, scale)

    # Compute errors
    diff = output_original - output_compacted
    mse = float(mx.mean(diff ** 2))
    relative_error = float(mx.linalg.norm(diff) / mx.linalg.norm(output_original))
    max_error = float(mx.max(mx.abs(diff)))
    mean_abs_error = float(mx.mean(mx.abs(diff)))

    return {
        'mse': mse,
        'relative_error': relative_error,
        'max_error': max_error,
        'mean_abs_error': mean_abs_error
    }


def select_keys_recent_stride(
    seq_len: int,
    budget: int,
    recent_ratio: float = 0.25
) -> mx.array:
    """
    Recent + Stride key selection

    策略:
    1. 保留最近的 recent_ratio * budget 个 tokens
    2. 对剩余部分均匀采样 (1 - recent_ratio) * budget 个 tokens

    这种策略基于假设:
    - 最近的 tokens 通常更重要（recency bias）
    - 远处的 tokens 也需要一些代表性采样

    Parameters
    ----------
    seq_len : int
        原始序列长度
    budget : int
        压缩后的 token 数量
    recent_ratio : float, default=0.25
        保留最近 tokens 的比例

    Returns
    -------
    indices : mx.array, shape (budget,)
        选中的 token positions

    Examples
    --------
    >>> indices = select_keys_recent_stride(1000, 200, recent_ratio=0.25)
    >>> indices.shape
    (200,)
    >>> # 最后 50 个是连续的（recent）
    >>> # 前 150 个是均匀采样的（stride）
    """
    # 计算 recent 和 stride 的数量
    n_recent = int(budget * recent_ratio)
    n_stride = budget - n_recent

    # Recent indices: 最后 n_recent 个
    recent_start = seq_len - n_recent
    recent_indices = mx.arange(recent_start, seq_len)

    # Stride indices: 均匀采样前面的部分
    stride_len = seq_len - n_recent
    if n_stride > 0 and stride_len > 0:
        stride = max(1, stride_len // n_stride)
        stride_indices = mx.arange(0, stride_len, stride)
        # 确保不超过 n_stride
        stride_indices = stride_indices[:n_stride]
    else:
        stride_indices = mx.array([], dtype=mx.int32)

    # 合并并排序
    if stride_indices.size > 0:
        indices = mx.concatenate([stride_indices, recent_indices])
    else:
        indices = recent_indices

    # 确保精确 budget 个
    if indices.size > budget:
        indices = indices[:budget]
    elif indices.size < budget:
        # 不够的话从头开始补
        extra = budget - indices.size
        extra_indices = mx.arange(0, extra)
        indices = mx.concatenate([extra_indices, indices])

    return indices


def visualize_key_selection(
    seq_len: int,
    indices: mx.array,
    attention_scores: mx.array = None
):
    """
    可视化 key selection（用于调试）

    Parameters
    ----------
    seq_len : int
        原始序列长度
    indices : mx.array, shape (budget,)
        选中的 indices
    attention_scores : mx.array, shape (seq_len,), optional
        每个 key 的 attention score（用于对比）

    Returns
    -------
    str
        可视化字符串
    """
    import numpy as np

    selected = np.zeros(seq_len, dtype=bool)
    selected[np.array(indices)] = True

    lines = []
    lines.append(f"Sequence length: {seq_len}")
    lines.append(f"Budget: {len(indices)}")
    lines.append(f"Compression ratio: {seq_len / len(indices):.2f}x")
    lines.append("")

    # 显示选中的位置
    step = max(1, seq_len // 100)  # 最多显示 100 个字符
    vis = ""
    for i in range(0, seq_len, step):
        if selected[i:i+step].any():
            vis += "█"
        else:
            vis += "░"
    lines.append(f"Selection pattern: {vis}")

    # 如果有 attention scores，显示对比
    if attention_scores is not None:
        scores_np = np.array(attention_scores)
        selected_scores = scores_np[np.array(indices)]
        all_scores = scores_np

        lines.append("")
        lines.append(f"Attention scores (selected):")
        lines.append(f"  Mean: {selected_scores.mean():.4f}")
        lines.append(f"  Max:  {selected_scores.max():.4f}")
        lines.append(f"  Min:  {selected_scores.min():.4f}")
        lines.append(f"Attention scores (all):")
        lines.append(f"  Mean: {all_scores.mean():.4f}")
        lines.append(f"  Max:  {all_scores.max():.4f}")

        # Coverage: 选中的 keys 占总 attention mass 的比例
        coverage = selected_scores.sum() / all_scores.sum()
        lines.append(f"Attention coverage: {coverage*100:.1f}%")

    return "\n".join(lines)
