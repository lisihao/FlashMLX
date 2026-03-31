"""
Attention Matching Compressor V2 - Correct Implementation

使用正确的 Attention Matching 算法（带 Beta、优化的 C2、Query Generation）
基于 https://github.com/adamzweiger/compaction
"""

from typing import Dict, Tuple, Optional
import mlx.core as mx
import numpy as np
import torch

from ..compaction import AttentionMatchingWrapper


class AttentionMatchingCompressorV2:
    """
    正确的 Attention Matching Compressor

    使用来自 https://github.com/adamzweiger/compaction 的算法：
    1. 基于 queries 计算 attention scores
    2. 选择 top-t keys
    3. 用 NNLS 求解 Beta（校正 attention 分布）
    4. 用 Ridge Regression 优化 C2（压缩后的 values）

    Args:
        compression_ratio: 压缩比例 (e.g., 2.0 = 压缩到 50%)
        score_method: Attention score 计算方法 ('max', 'mean', 'rms')
        beta_method: Beta 计算方法 ('nnls', 'zero')
        c2_method: C2 计算方法 ('lsq', 'direct')
        num_queries: 生成的 query 数量（用于评估 key 重要性）

    Example:
        >>> compressor = AttentionMatchingCompressorV2(compression_ratio=2.0)
        >>> keys = mx.random.normal((1, 8, 100, 64))  # (batch, heads, seq_len, head_dim)
        >>> values = mx.random.normal((1, 8, 100, 64))
        >>> compressed_keys, compressed_values = compressor.compress_kv_cache(
        ...     layer_idx=0,
        ...     kv_cache=(keys, values)
        ... )
        >>> compressed_keys.shape[2]  # 50 (100 / 2.0)
    """

    def __init__(
        self,
        model,
        compression_ratio: float = 2.0,
        score_method: str = 'max',
        beta_method: str = 'nnls',
        c2_method: str = 'lsq',
        num_queries: int = 100,
    ):
        """初始化正确的 Compressor

        Args:
            model: MLX-LM 模型实例（用于获取 q_norm）
            compression_ratio: 压缩比例
            score_method: Attention score 计算方法
            beta_method: Beta 计算方法
            c2_method: C2 计算方法
            num_queries: Query 数量
        """
        if compression_ratio < 1.0:
            raise ValueError(f"compression_ratio must be >= 1.0, got {compression_ratio}")

        self.model = model  # ✅ 保存 model 引用（用于获取 q_norm）
        self.compression_ratio = compression_ratio
        self.num_queries = num_queries

        # 为每个层创建独立的 Wrapper
        # 因为每层的 head_dim 可能不同
        self.wrappers: Dict[int, AttentionMatchingWrapper] = {}

        # Wrapper 参数
        self.score_method = score_method
        self.beta_method = beta_method
        self.c2_method = c2_method

        # 存储压缩参数 (C1, beta, C2) for each layer
        # 用于 inference 时应用 beta
        self.compressed_params: Dict[int, Tuple[mx.array, mx.array, mx.array]] = {}

        # 统计
        self.compression_stats = {
            "total_compressions": 0,
            "total_keys_before": 0,
            "total_keys_after": 0,
        }

    def _get_wrapper(self, layer_idx: int) -> AttentionMatchingWrapper:
        """获取或创建 layer 的 wrapper"""
        if layer_idx not in self.wrappers:
            self.wrappers[layer_idx] = AttentionMatchingWrapper(
                compression_ratio=self.compression_ratio,
                score_method=self.score_method,
                beta_method=self.beta_method,
                c2_method=self.c2_method,
            )
        return self.wrappers[layer_idx]

    def compress_kv_cache(
        self,
        layer_idx: int,
        kv_cache: Tuple[mx.array, mx.array]
    ) -> Tuple[mx.array, mx.array]:
        """
        压缩 KV cache（正确实现，批量处理优化）

        Args:
            layer_idx: 层索引
            kv_cache: (keys, values) tuple
                      Shape: (batch_size, num_heads, seq_len, head_dim)

        Returns:
            (compressed_keys, compressed_values)
            Shape: (batch_size, num_heads, compressed_seq_len, head_dim)
        """
        keys, values = kv_cache
        batch_size, num_heads, seq_len, head_dim = keys.shape

        # Validate shapes
        if values.shape != keys.shape:
            raise ValueError(f"Keys and values must have same shape, got {keys.shape} vs {values.shape}")

        # Calculate target sequence length
        target_seq_len = max(1, int(seq_len / self.compression_ratio))

        # 如果序列太短，不压缩
        if seq_len < self.compression_ratio or seq_len <= target_seq_len:
            self.compression_stats['total_compressions'] += 1
            self.compression_stats['total_keys_before'] += seq_len
            self.compression_stats['total_keys_after'] += seq_len
            return keys, values

        # 获取 wrapper
        wrapper = self._get_wrapper(layer_idx)

        # 目前只处理 batch=1 的情况
        if batch_size > 1:
            raise NotImplementedError("Batch size > 1 not yet supported")

        # ✅ 批量处理优化：一次性转换所有 heads 到 PyTorch
        # 避免逐 head 循环转换（40 heads × 3 次 = 120 次 → 2 次）

        # Squeeze batch dimension
        keys_3d = mx.squeeze(keys, axis=0)  # (num_heads, seq_len, head_dim)
        values_3d = mx.squeeze(values, axis=0)  # (num_heads, seq_len, head_dim)

        # 一次性转换到 PyTorch（只转换 1 次）
        keys_torch = wrapper.mlx_to_torch(keys_3d)  # (num_heads, seq_len, head_dim)
        values_torch = wrapper.mlx_to_torch(values_3d)  # (num_heads, seq_len, head_dim)

        # 压缩每个 head（在 PyTorch 侧循环）
        C1_list = []
        beta_list = []
        C2_list = []

        for head_idx in range(num_heads):
            # Extract keys/values for this head (PyTorch tensors)
            head_keys_torch = keys_torch[head_idx]  # (seq_len, head_dim)
            head_values_torch = values_torch[head_idx]  # (seq_len, head_dim)

            # ✅ Cache Keys Method: 从 KV cache 中采样 keys 作为 queries
            # 这比 random queries 更准确（论文推荐）

            # 自适应 num_queries：根据序列长度调整
            # - 短序列（<=100）：使用配置的 num_queries
            # - 中长序列：使用序列长度的 50%（更准确）
            # - 超长序列：上限 1000（保证至少 25% 覆盖率）
            if seq_len <= 100:
                num_queries_actual = min(self.num_queries, seq_len)
            else:
                # 自适应增加 queries 数量
                adaptive_num = max(self.num_queries, seq_len // 2)
                num_queries_actual = min(adaptive_num, 1000)  # 上限 1000

            if num_queries_actual < seq_len:
                # 随机采样 indices（PyTorch）
                indices = np.random.choice(seq_len, size=num_queries_actual, replace=False)
                indices = np.sort(indices)  # 保持原始顺序
                indices_mlx = mx.array(indices)

                # ✅ CRITICAL FIX: 应用 q_norm 到采样的 keys
                # Qwen 模型使用 q_norm (RMSNorm) 对 queries 进行归一化
                # 从 cache keys 采样的 queries 也必须应用相同的 q_norm

                # 从 MLX keys 中采样（需要从 keys_3d 获取当前 head）
                head_keys_mlx = keys_3d[head_idx]  # (seq_len, head_dim)
                sampled_keys_mlx = mx.take(head_keys_mlx, indices_mlx, axis=0)  # (num_queries, head_dim)

                # 检查模型是否有 q_norm（Qwen 特有）
                # 动态检测模型结构
                q_norm = None
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    # 结构: model.model.layers[i].self_attn.q_norm
                    if hasattr(self.model.model.layers[layer_idx].self_attn, 'q_norm'):
                        q_norm = self.model.model.layers[layer_idx].self_attn.q_norm
                elif hasattr(self.model, 'layers'):
                    # 结构: model.layers[i].self_attn.q_norm
                    if hasattr(self.model.layers[layer_idx].self_attn, 'q_norm'):
                        q_norm = self.model.layers[layer_idx].self_attn.q_norm

                if q_norm is not None:
                    sampled_queries_mlx = q_norm(sampled_keys_mlx)  # 应用 q_norm scaling
                else:
                    # 没有 q_norm，直接使用 keys
                    sampled_queries_mlx = sampled_keys_mlx

                # 转换到 PyTorch
                sampled_queries_torch = wrapper.mlx_to_torch(sampled_queries_mlx)  # (num_queries, head_dim)
            else:
                # 序列太短，使用全部 keys（也需要 q_norm）
                head_keys_mlx = keys_3d[head_idx]  # (seq_len, head_dim)

                # 动态检测模型结构
                q_norm = None
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    if hasattr(self.model.model.layers[layer_idx].self_attn, 'q_norm'):
                        q_norm = self.model.model.layers[layer_idx].self_attn.q_norm
                elif hasattr(self.model, 'layers'):
                    if hasattr(self.model.layers[layer_idx].self_attn, 'q_norm'):
                        q_norm = self.model.layers[layer_idx].self_attn.q_norm

                if q_norm is not None:
                    sampled_queries_mlx = q_norm(head_keys_mlx)
                    sampled_queries_torch = wrapper.mlx_to_torch(sampled_queries_mlx)
                else:
                    sampled_queries_torch = head_keys_torch

            # Compress using correct algorithm (PyTorch tensors)
            # Returns: C1, beta, C2 (all PyTorch)
            C1_torch, beta_torch, C2_torch, indices = wrapper.algorithm.compute_compacted_cache(
                K=head_keys_torch,
                V=head_values_torch,
                queries=sampled_queries_torch,
                t=target_seq_len,
            )

            # 存储 PyTorch tensors（稍后统一转换）
            C1_list.append(C1_torch)
            beta_list.append(beta_torch)
            C2_list.append(C2_torch)

        # 一次性转换所有结果回 MLX（只转换 1 次）
        C1_torch_stacked = torch.stack(C1_list, dim=0)  # (num_heads, t, head_dim)
        beta_torch_stacked = torch.stack(beta_list, dim=0)  # (num_heads, t)
        C2_torch_stacked = torch.stack(C2_list, dim=0)  # (num_heads, t, head_dim)

        C1_mlx = wrapper.torch_to_mlx(C1_torch_stacked)  # (num_heads, t, head_dim)
        beta_mlx = wrapper.torch_to_mlx(beta_torch_stacked)  # (num_heads, t)
        C2_mlx = wrapper.torch_to_mlx(C2_torch_stacked)  # (num_heads, t, head_dim)

        # 存储压缩参数（用于 inference 时应用 beta）
        for head_idx in range(num_heads):
            C1_head = C1_mlx[head_idx]  # (t, head_dim)
            beta_head = beta_mlx[head_idx]  # (t,)
            C2_head = C2_mlx[head_idx]  # (t, head_dim)
            self.compressed_params[(layer_idx, head_idx)] = (C1_head, beta_head, C2_head)

        # Add batch dimension back
        compressed_keys = mx.expand_dims(C1_mlx, axis=0)  # (1, num_heads, t, head_dim)
        compressed_values = mx.expand_dims(C2_mlx, axis=0)  # (1, num_heads, t, head_dim)

        # Update statistics
        compressed_seq_len = compressed_keys.shape[2]
        self.compression_stats["total_compressions"] += 1
        self.compression_stats["total_keys_before"] += seq_len
        self.compression_stats["total_keys_after"] += compressed_seq_len

        return compressed_keys, compressed_values

    def apply_beta_compensation(
        self,
        layer_idx: int,
        head_idx: int,
        attention_scores: mx.array
    ) -> mx.array:
        """
        应用 Beta 校准（正确实现）

        在 attention scores（softmax 之前）加上 beta

        Args:
            layer_idx: 层索引
            head_idx: Head 索引
            attention_scores: Attention scores before softmax
                              Shape: (batch, query_len, key_len)

        Returns:
            Compensated attention scores (same shape)
        """
        key = (layer_idx, head_idx)
        if key not in self.compressed_params:
            # No compression for this layer/head yet
            return attention_scores

        C1, beta, C2 = self.compressed_params[key]

        # Beta shape: (t,)
        # Broadcast to match attention_scores shape
        # attention_scores: (batch, query_len, t)
        # beta: (t,)

        # Add beta to each attention score
        # Broadcasting: (batch, query_len, t) + (t,) -> (batch, query_len, t)
        compensated_scores = attention_scores + beta[None, None, :]

        return compensated_scores

    def get_compression_stats(self) -> Dict[str, float]:
        """获取压缩统计"""
        total_before = self.compression_stats["total_keys_before"]
        total_after = self.compression_stats["total_keys_after"]

        avg_compression_ratio = total_before / total_after if total_after > 0 else 0.0

        return {
            "total_compressions": self.compression_stats["total_compressions"],
            "total_keys_before": total_before,
            "total_keys_after": total_after,
            "avg_compression_ratio": avg_compression_ratio,
            "configured_compression_ratio": self.compression_ratio,
        }

    def __repr__(self) -> str:
        return (
            f"AttentionMatchingCompressorV2("
            f"compression_ratio={self.compression_ratio}, "
            f"score_method={self.score_method}, "
            f"beta_method={self.beta_method}, "
            f"c2_method={self.c2_method})"
        )
