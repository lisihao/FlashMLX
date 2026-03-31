"""
Simple Attention Matching Injection

为 Attention-only 模型注入 Attention Matching 压缩。
简化版，不依赖已废弃的 HybridCacheManager。
"""

from typing import List, Optional
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from .attention_matching_compressor_v2 import AttentionMatchingCompressorV2


class CompressedArraysCache(ArraysCache):
    """
    扩展 ArraysCache，在存储时自动应用 Attention Matching 压缩。

    拦截 __setitem__ 调用，对 KV cache 进行压缩。
    """

    def __init__(
        self,
        size: int,
        layer_idx: int,
        compressor: AttentionMatchingCompressor,
        left_padding: Optional[List[int]] = None
    ):
        """
        初始化压缩 cache

        Args:
            size: Cache 大小（状态数量）
            layer_idx: 层索引
            compressor: AttentionMatchingCompressor 实例
            left_padding: Left padding（可选）
        """
        super().__init__(size, left_padding)
        self.layer_idx = layer_idx
        self.compressor = compressor

        # RoPE offset（Qwen3.5 需要）
        self.offset = 0

        # 统计
        self.compression_count = 0
        self.total_original_tokens = 0
        self.total_compressed_tokens = 0

    def __setitem__(self, idx, value):
        """
        存储 KV cache，自动应用压缩

        Args:
            idx: Cache 索引
            value: KV cache tuple (keys, values) 或 None
        """
        # 如果是 KV cache tuple，应用压缩
        if value is not None and isinstance(value, tuple) and len(value) == 2:
            keys, values = value

            # 只对 4D tensor 应用压缩（batch, num_heads, seq_len, head_dim）
            if keys.ndim == 4 and values.ndim == 4:
                original_seq_len = keys.shape[2]

                # 应用 Attention Matching 压缩
                try:
                    compressed_keys, compressed_values = self.compressor.compress_kv_cache(
                        layer_idx=self.layer_idx,
                        kv_cache=(keys, values)
                    )

                    compressed_seq_len = compressed_keys.shape[2]

                    # 更新统计
                    self.compression_count += 1
                    self.total_original_tokens += original_seq_len
                    self.total_compressed_tokens += compressed_seq_len

                    # 存储压缩后的 KV
                    value = (compressed_keys, compressed_values)

                except Exception as e:
                    # 如果压缩失败，回退到原始 KV
                    print(f"Warning: Compression failed for layer {self.layer_idx}: {e}")
                    # 保持 value 不变

        # 调用父类存储
        super().__setitem__(idx, value)

    def update_and_fetch(self, keys, values):
        """
        更新并获取 KV cache（自动应用压缩）

        这是 Qwen3.5 等模型的标准 cache 接口。

        Args:
            keys: 新的 keys (batch, num_heads, new_seq_len, head_dim)
            values: 新的 values (batch, num_heads, new_seq_len, head_dim)

        Returns:
            (compressed_keys, compressed_values) 完整的压缩后 KV cache
        """
        # 获取当前存储的 KV（如果有）
        current_kv = self.cache[0]

        if current_kv is None:
            # 首次存储
            full_keys = keys
            full_values = values
        else:
            # Concat 到现有 KV
            current_keys, current_values = current_kv
            full_keys = mx.concatenate([current_keys, keys], axis=-2)
            full_values = mx.concatenate([current_values, values], axis=-2)

        # 应用压缩
        original_seq_len = full_keys.shape[-2]

        try:
            compressed_keys, compressed_values = self.compressor.compress_kv_cache(
                layer_idx=self.layer_idx,
                kv_cache=(full_keys, full_values)
            )

            compressed_seq_len = compressed_keys.shape[-2]

            # 更新统计
            self.compression_count += 1
            self.total_original_tokens += original_seq_len
            self.total_compressed_tokens += compressed_seq_len

        except Exception as e:
            # 如果压缩失败，回退到原始 KV
            print(f"Warning: Compression failed for layer {self.layer_idx}: {e}")
            compressed_keys = full_keys
            compressed_values = full_values
            compressed_seq_len = original_seq_len

        # 存储压缩后的 KV
        self.cache[0] = (compressed_keys, compressed_values)

        # 更新 offset（使用压缩后的长度）
        self.offset = compressed_seq_len

        return compressed_keys, compressed_values

    def make_mask(self, N: int, return_array=None, window_size=None):
        """
        创建 attention mask（兼容 Qwen3.5 的扩展接口）

        Args:
            N: Sequence length
            return_array: 是否返回 array（Qwen3.5 扩展参数）
            window_size: 窗口大小（Qwen3.5 扩展参数）

        Returns:
            Attention mask 或 None
        """
        # 调用父类的 make_mask（只传入 N）
        mask = super().make_mask(N)

        # 如果指定了 return_array=False，Qwen3.5 期望返回 None
        if return_array is False:
            return None

        return mask

    def get_stats(self):
        """获取压缩统计信息"""
        if self.compression_count == 0:
            return {
                'layer_idx': self.layer_idx,
                'compression_count': 0,
                'avg_compression_ratio': 0.0,
                'total_original_tokens': 0,
                'total_compressed_tokens': 0
            }

        avg_ratio = self.total_original_tokens / self.total_compressed_tokens if self.total_compressed_tokens > 0 else 1.0

        return {
            'layer_idx': self.layer_idx,
            'compression_count': self.compression_count,
            'avg_compression_ratio': avg_ratio,
            'total_original_tokens': self.total_original_tokens,
            'total_compressed_tokens': self.total_compressed_tokens
        }


class CacheList(list):
    """
    自定义 list 子类，可以存储额外的属性。
    用于注入到 model.cache。
    """
    pass


def inject_attention_matching(
    model,
    compression_ratio: float = 3.0,
    beta_calibration: bool = True,
    eviction_policy: str = "top_k",
    num_queries: int = 100,
    verbose: bool = True
):
    """
    为模型注入 Attention Matching 压缩。

    自动替换模型的 cache，使所有 Attention 层使用压缩 cache。

    Args:
        model: MLX-LM 模型实例
        compression_ratio: 压缩比例（默认 3.0x）
        beta_calibration: 是否启用 β 校准
        eviction_policy: Token 选择策略 ("top_k" 或 "weighted")
        num_queries: 查询数量（默认 100，用于 Cache Keys 采样）
        verbose: 是否打印注入信息

    Returns:
        restore_fn: 恢复原始 cache 的函数

    Example:
        >>> from mlx_lm import load
        >>> from flashmlx.cache import inject_attention_matching
        >>>
        >>> model, tokenizer = load("path/to/model")
        >>> cache_list, compressor = inject_attention_matching(
        ...     model,
        ...     compression_ratio=3.0,
        ...     beta_calibration=True
        ... )
        >>>
        >>> # 现在 model 会自动使用压缩 cache
        >>> from mlx_lm import generate
        >>> response = generate(model, tokenizer, "Hello", max_tokens=100)
        >>>
        >>> # 查看压缩统计
        >>> for cache in cache_list:
        ...     if hasattr(cache, 'get_stats'):
        ...         print(cache.get_stats())
    """
    # 创建 compressor（使用正确实现）
    # ✅ 传递 model 引用（用于获取 q_norm）
    compressor = AttentionMatchingCompressorV2(
        model=model,  # ✅ 传递 model
        compression_ratio=compression_ratio,
        score_method='max',
        beta_method='nnls' if beta_calibration else 'zero',
        c2_method='lsq',
        num_queries=num_queries
    )

    # 获取层数（假设所有层都是 Attention 层）
    # 对于 Transformer 模型，可以从 model.layers 推断
    if hasattr(model, 'layers'):
        num_layers = len(model.layers)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    else:
        raise ValueError("Cannot determine number of layers from model structure")

    # 创建压缩 cache list
    cache_list = CacheList()

    for layer_idx in range(num_layers):
        # 为每一层创建压缩 cache
        # 每层有 2 个状态（keys, values）
        compressed_cache = CompressedArraysCache(
            size=2,
            layer_idx=layer_idx,
            compressor=compressor
        )
        cache_list.append(compressed_cache)

    # 保存原始 make_cache 方法（如果存在）
    if hasattr(model, 'make_cache'):
        cache_list._original_make_cache = model.make_cache

    # 保存原始 cache
    original_cache = model.cache if hasattr(model, 'cache') else None
    original_make_cache = model.make_cache if hasattr(model, 'make_cache') else None

    # 注入到模型
    model.make_cache = lambda: cache_list
    model.cache = cache_list

    if verbose:
        print(f"✓ Attention Matching 注入成功:")
        print(f"  - 层数: {num_layers}")
        print(f"  - 压缩比例: {compression_ratio}x")
        print(f"  - β 校准: {'启用' if beta_calibration else '禁用'}")
        print(f"  - 选择策略: {eviction_policy}")
        print(f"  - 查询数量: {num_queries}")

    # 返回恢复函数
    def restore():
        """恢复原始 cache"""
        if original_cache is not None:
            model.cache = original_cache
        if original_make_cache is not None:
            model.make_cache = original_make_cache

    return restore


def get_compression_stats(cache_list):
    """
    获取所有层的压缩统计信息

    Args:
        cache_list: 注入的 cache list

    Returns:
        统计信息字典
    """
    stats = []

    for cache in cache_list:
        if hasattr(cache, 'get_stats'):
            stats.append(cache.get_stats())

    # 汇总统计
    total_compressions = sum(s['compression_count'] for s in stats)
    total_original = sum(s['total_original_tokens'] for s in stats)
    total_compressed = sum(s['total_compressed_tokens'] for s in stats)

    overall_ratio = total_original / total_compressed if total_compressed > 0 else 0.0

    return {
        'layer_stats': stats,
        'total_compressions': total_compressions,
        'total_original_tokens': total_original,
        'total_compressed_tokens': total_compressed,
        'overall_compression_ratio': overall_ratio
    }
