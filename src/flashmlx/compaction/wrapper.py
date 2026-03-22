"""
Wrapper for integrating compaction library with MLX.

Handles tensor conversion between MLX and PyTorch.
"""

import torch
import mlx.core as mx
import numpy as np
from typing import Tuple, Optional

from .algorithms.highest_attention_keys import HighestAttentionKeysCompaction
# Note: QueryGenerator has external dependencies, using simple random queries for now


class AttentionMatchingWrapper:
    """
    Wrapper for Attention Matching compaction with MLX.

    Converts MLX tensors to PyTorch, applies compaction, converts back.
    """

    def __init__(
        self,
        compression_ratio: float = 3.0,
        score_method: str = 'max',
        beta_method: str = 'nnls',
        c2_method: str = 'lsq',
        nnls_iters: int = 0,
        c2_ridge_lambda: float = 0,
    ):
        """
        Initialize Attention Matching wrapper.

        Args:
            compression_ratio: Target compression ratio (e.g., 3.0 for 3x compression)
            score_method: Method to score keys ('max', 'mean', or 'rms')
            beta_method: Method to compute beta ('nnls' or 'zero')
            c2_method: Method to compute C2 ('lsq' or 'direct')
            nnls_iters: Number of NNLS iterations (0 uses lstsq)
            c2_ridge_lambda: Ridge regression lambda for C2
        """
        self.compression_ratio = compression_ratio

        # Create compaction algorithm
        self.algorithm = HighestAttentionKeysCompaction(
            score_method=score_method,
            beta_method=beta_method,
            c2_method=c2_method,
            nnls_iters=nnls_iters,
            c2_ridge_lambda=c2_ridge_lambda,
        )

    def mlx_to_torch(self, arr: mx.array) -> torch.Tensor:
        """Convert MLX array to PyTorch tensor (fixed dtype handling)."""
        # ✅ 必须先转换到 float32，再转 numpy
        if arr.dtype == mx.bfloat16:
            arr = arr.astype(mx.float32)
        elif arr.dtype == mx.float16:
            arr = arr.astype(mx.float32)

        # ✅ 强制转换到 float32，确保 numpy dtype 正确
        np_arr = np.array(arr, dtype=np.float32)

        # ✅ 验证 dtype（防止 PEP 3118 错误）
        if np_arr.dtype != np.float32:
            raise ValueError(f"Expected float32, got {np_arr.dtype}")

        return torch.from_numpy(np_arr)

    def torch_to_mlx(self, tensor: torch.Tensor) -> mx.array:
        """Convert PyTorch tensor to MLX array."""
        # Convert to numpy then to mlx
        np_arr = tensor.cpu().numpy()
        return mx.array(np_arr)

    def compress_kv_cache(
        self,
        keys: mx.array,
        values: mx.array,
        queries: Optional[mx.array] = None,
        num_queries: int = 100,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Compress KV cache using Attention Matching.

        Args:
            keys: Key tensor, shape (seq_len, head_dim)
            values: Value tensor, shape (seq_len, head_dim)
            queries: Optional query tensor for computing attention scores
                     If None, uses random queries
            num_queries: Number of queries to generate if queries=None

        Returns:
            Tuple of (C1, beta, C2):
            - C1: Compressed keys, shape (t, head_dim)
            - beta: Bias terms, shape (t,)
            - C2: Compressed values, shape (t, head_dim)
        """
        seq_len, head_dim = keys.shape
        target_size = int(seq_len / self.compression_ratio)

        # Convert MLX to PyTorch
        K_torch = self.mlx_to_torch(keys)  # (seq_len, head_dim)
        V_torch = self.mlx_to_torch(values)  # (seq_len, head_dim)

        # Generate or convert queries
        if queries is None:
            # Generate random queries (simple fallback)
            queries_torch = torch.randn(
                num_queries, head_dim,
                dtype=K_torch.dtype,
                device=K_torch.device
            )
        else:
            queries_torch = self.mlx_to_torch(queries)  # (n_queries, head_dim)

        # Compute compacted cache using correct algorithm
        C1_torch, beta_torch, C2_torch, indices = self.algorithm.compute_compacted_cache(
            K=K_torch,
            V=V_torch,
            queries=queries_torch,
            t=target_size,
        )

        # Convert back to MLX
        C1_mlx = self.torch_to_mlx(C1_torch)  # (t, head_dim)
        beta_mlx = self.torch_to_mlx(beta_torch)  # (t,)
        C2_mlx = self.torch_to_mlx(C2_torch)  # (t, head_dim)

        return C1_mlx, beta_mlx, C2_mlx

    def apply_compacted_attention(
        self,
        query: mx.array,
        C1: mx.array,
        beta: mx.array,
        C2: mx.array,
    ) -> mx.array:
        """
        Apply compacted attention at inference time.

        Args:
            query: Query tensor, shape (head_dim,) or (num_queries, head_dim)
            C1: Compacted keys, shape (t, head_dim)
            beta: Bias terms, shape (t,)
            C2: Compacted values, shape (t, head_dim)

        Returns:
            Attention output, shape (head_dim,) or (num_queries, head_dim)
        """
        # Ensure query is 2D
        if query.ndim == 1:
            query = mx.expand_dims(query, axis=0)  # (1, head_dim)
            squeeze_output = True
        else:
            squeeze_output = False

        # Compute attention scores: Q @ C1.T + beta
        # scores shape: (num_queries, t)
        scores = query @ C1.T  # (num_queries, t)

        # Scale by 1/sqrt(d)
        head_dim = query.shape[-1]
        scores = scores / mx.sqrt(mx.array(head_dim, dtype=scores.dtype))

        # Add beta bias
        scores = scores + beta[None, :]  # (num_queries, t)

        # Compute attention weights
        attention_weights = mx.softmax(scores, axis=-1)  # (num_queries, t)

        # Apply attention to values
        output = attention_weights @ C2  # (num_queries, head_dim)

        if squeeze_output:
            output = mx.squeeze(output, axis=0)  # (head_dim,)

        return output
