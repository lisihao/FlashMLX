"""
Compaction Engine - Offline KV Cache Compression

This module implements the correct approach for AM compression:
- Compression is triggered OUTSIDE the forward pass hot path
- Uses reference queries (Qref) sampled from recent KV cache
- Compresses all layers in a single batch operation
- Follows the AM paper's "one-shot operation" design

Based on user's correction:
"正确做法是：先正常推理，等满足触发条件后，再对'旧的、可压的 KV'做 AM 压缩，
然后继续推理。不是每一层、每一步都在线做 AM。"
"""

import mlx.core as mx
import numpy as np
from typing import List, Optional, Tuple
from .compacted_cache import CompactedKVCache


class CompactionEngine:
    """
    Offline KV Cache Compression Engine.

    Responsibilities:
    1. Monitor KV cache size
    2. Trigger compression when threshold is reached
    3. Sample reference queries (Qref) from recent KV
    4. Compress all layers in a single batch operation

    This is the CORRECT way to use AM compression, as opposed to
    checking and compressing in update_and_fetch() (hot path).

    Parameters
    ----------
    max_size : int
        Maximum cache size before triggering compression
    compression_ratio : float
        Target compression ratio
    num_queries : int, default=128
        Number of reference queries to sample for AM compression
    check_interval : int, default=256
        Check for compaction every N tokens during generation

    Examples
    --------
    >>> # Create engine
    >>> engine = CompactionEngine(max_size=4096, compression_ratio=5.0)
    >>>
    >>> # Check if should compact after prefill
    >>> if engine.should_compact(cache[0]):
    >>>     queries = engine.sample_queries(cache[0])
    >>>     engine.compact_all_layers(cache, queries)
    >>>
    >>> # Periodic check during generation
    >>> if token_idx % engine.check_interval == 0:
    >>>     if engine.should_compact(cache[0]):
    >>>         queries = engine.sample_queries(cache[0])
    >>>         engine.compact_all_layers(cache, queries)
    """

    def __init__(
        self,
        max_size: int = 4096,
        compression_ratio: float = 5.0,
        num_queries: int = 128,
        check_interval: int = 256,
    ):
        self.max_size = max_size
        self.compression_ratio = compression_ratio
        self.num_queries = num_queries
        self.check_interval = check_interval

        # Statistics
        self.total_compactions = 0
        self.total_compaction_time = 0.0

    def should_compact(self, cache: CompactedKVCache) -> bool:
        """
        Check if compression should be triggered.

        Parameters
        ----------
        cache : CompactedKVCache
            Any layer's cache (all layers have same offset)

        Returns
        -------
        bool
            True if cache size exceeds threshold
        """
        return cache.offset > self.max_size

    def sample_queries(
        self,
        cache: CompactedKVCache,
        num_queries: Optional[int] = None,
        sample_from_recent: bool = True,
    ) -> mx.array:
        """
        Sample reference queries (Qref) from KV cache using self-study.

        ✅ CRITICAL FIX: Use ALL recent keys as queries (not random sample)!

        The AM paper emphasizes that Qref quality is crucial:
        - repeat-prefill: Re-run prefill to get real queries (best, but requires original input)
        - self-study: Use keys as queries (good approximation)
        - random vectors: Poor quality

        Previous implementation used RANDOM sampling of keys, which is even worse
        than random vectors! Now we use ALL recent keys, which approximates
        "what the model actually attends to".

        Parameters
        ----------
        cache : CompactedKVCache
            Cache to sample from
        num_queries : int, optional
            IGNORED - we use all recent keys instead of sampling
        sample_from_recent : bool, default=True
            If True, use recent tokens only (last 50%)
            If False, use all tokens

        Returns
        -------
        mx.array
            Query vectors from keys, shape (B, n_heads, num_queries, head_dim)

        Notes
        -----
        This implements "self-study" approach: using the KV cache itself
        as query distribution, which approximates what queries will look like
        in the future.
        """
        B, n_heads, offset, head_dim = cache.keys.shape[0], cache.keys.shape[1], cache.offset, cache.keys.shape[3]

        # ✅ IMPROVED: Use diverse sampling with stride (避免连续token的问题)
        # 论文建议：数千到数万个diverse queries
        # 这里作为快速改进：从整个cache中stride采样，增加diversity

        if sample_from_recent:
            # Use last 50% of tokens
            start_idx = max(0, offset - offset // 2)
        else:
            # Use all tokens
            start_idx = 0

        # ✅ NEW: Stride sampling instead of consecutive
        # 如果 offset 很大，使用 stride=3 采样以增加 diversity
        # 如果 offset 较小，使用所有 tokens
        stride = 3 if offset > 300 else 2 if offset > 100 else 1

        # 采样 indices: start_idx, start_idx+stride, start_idx+2*stride, ...
        indices = list(range(start_idx, offset, stride))

        # 至少保证有一定数量的 queries
        min_queries = 50
        if len(indices) < min_queries and offset > min_queries:
            # Fallback: 均匀采样 min_queries 个
            # 使用 int() 转换确保是 Python int，避免 MLX 的 int64 错误
            indices = [int(x) for x in np.linspace(start_idx, offset-1, min_queries, dtype=int)]

        # Extract queries (使用 mx.take)
        # 确保 indices 是 Python int list，MLX 不接受 int64
        indices_array = mx.array(indices, dtype=mx.int32)
        queries = mx.take(cache.keys, indices_array, axis=2)  # (B, n_heads, num_queries, head_dim)

        if len(indices) > 10:  # 只在有足够queries时打印
            print(f"[CompactionEngine] Sampled {len(indices)} diverse queries (stride={stride}, offset={offset})")

        return queries

    def compact_all_layers(
        self,
        cache_list: List[CompactedKVCache],
        queries: mx.array,
        verbose: bool = False,
    ) -> Tuple[int, float]:
        """
        Compress all layers using the same reference queries.

        This is the core of offline compaction: instead of each layer
        compressing independently in the hot path, we compress all layers
        in a single batch operation outside the forward pass.

        Parameters
        ----------
        cache_list : List[CompactedKVCache]
            List of caches for all layers
        queries : mx.array
            Reference queries (Qref) for AM compression
            Shape: (B, n_heads, num_queries, head_dim)
        verbose : bool, default=False
            Print compression progress

        Returns
        -------
        num_compressed : int
            Number of layers compressed
        total_time : float
            Total compression time (seconds)

        Notes
        -----
        Key advantages over hot path compression:
        1. All layers use the SAME Qref (consistency)
        2. No redundant computation (36 layers don't repeat sampling)
        3. Can be optimized as batch operation
        4. Doesn't block the forward pass
        """
        import time

        start_time = time.time()
        num_compressed = 0

        for i, cache in enumerate(cache_list):
            # Only compress layers that need it
            if not isinstance(cache, CompactedKVCache):
                continue

            if cache.offset <= self.max_size:
                continue

            # Trigger compression with Qref
            success = cache.compact(queries=queries)
            if success:
                num_compressed += 1

                if verbose and i == 0:  # Only print for first layer
                    stats = cache.get_stats()
                    print(f"  Layer 0: {stats['total_tokens_before']} → {stats['total_tokens_after']} tokens")

        total_time = time.time() - start_time

        # Update statistics
        self.total_compactions += 1
        self.total_compaction_time += total_time

        if verbose:
            print(f"  Compressed {num_compressed} layers in {total_time*1000:.2f}ms")

        return num_compressed, total_time

    def get_stats(self) -> dict:
        """
        Get compaction statistics.

        Returns
        -------
        dict
            Statistics including:
            - total_compactions: Number of compaction operations
            - total_compaction_time: Total time spent in compaction
            - avg_compaction_time: Average time per compaction
        """
        avg_time = (
            self.total_compaction_time / self.total_compactions
            if self.total_compactions > 0
            else 0.0
        )

        return {
            "total_compactions": self.total_compactions,
            "total_compaction_time": self.total_compaction_time,
            "avg_compaction_time": avg_time,
        }


def create_compaction_engine(
    max_size: int = 4096,
    compression_ratio: float = 5.0,
    num_queries: int = 128,
    check_interval: int = 256,
) -> CompactionEngine:
    """
    Factory function to create CompactionEngine.

    This is the recommended way to create an engine for use in
    generation loops.

    Parameters
    ----------
    max_size : int, default=4096
        Maximum cache size before triggering compression
    compression_ratio : float, default=5.0
        Target compression ratio
    num_queries : int, default=128
        Number of reference queries to sample
    check_interval : int, default=256
        Check for compaction every N tokens

    Returns
    -------
    CompactionEngine
        Configured compaction engine

    Examples
    --------
    >>> engine = create_compaction_engine(max_size=2048, compression_ratio=3.0)
    >>> # Use in generation loop...
    """
    return CompactionEngine(
        max_size=max_size,
        compression_ratio=compression_ratio,
        num_queries=num_queries,
        check_interval=check_interval,
    )
