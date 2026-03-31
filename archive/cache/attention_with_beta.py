"""
Custom Attention with Beta Support

在 attention scores 计算后、softmax 前应用 beta。
这是 Attention Matching 压缩的关键。
"""

import mlx.core as mx
from typing import Optional, Dict, Tuple


def scaled_dot_product_attention_with_beta(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[mx.array],
    beta: Optional[mx.array] = None,
) -> mx.array:
    """
    Scaled Dot-Product Attention with optional Beta bias.

    与标准 SDPA 的唯一区别：在 softmax 前加上 beta。

    Args:
        queries: (batch, num_heads, query_len, head_dim)
        keys: (batch, num_heads, key_len, head_dim)
        values: (batch, num_heads, key_len, head_dim)
        scale: 1/sqrt(head_dim)
        mask: Attention mask (batch, 1, query_len, key_len) or None
        beta: Beta bias (batch, num_heads, key_len) or None
            If provided, beta will be broadcast and added to attention scores
            BEFORE softmax.

    Returns:
        Attention output: (batch, num_heads, query_len, head_dim)
    """
    # Standard attention scores: Q @ K.T * scale
    scores = queries @ keys.transpose(0, 1, 3, 2)  # (batch, heads, query_len, key_len)
    scores = scores * scale

    # ✅ 应用 beta（Attention Matching 的关键）
    if beta is not None:
        # beta shape: (batch, num_heads, key_len)
        # 需要 broadcast 到 (batch, num_heads, query_len, key_len)
        beta_expanded = beta[:, :, None, :]  # (batch, heads, 1, key_len)
        scores = scores + beta_expanded

    # Apply mask
    if mask is not None:
        scores = scores + mask

    # Softmax over key dimension
    attention_weights = mx.softmax(scores, axis=-1)

    # Apply attention to values
    output = attention_weights @ values  # (batch, heads, query_len, head_dim)

    return output
