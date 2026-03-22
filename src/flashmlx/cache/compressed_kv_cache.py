"""
Compressed KV Cache

Attention layer KV cache wrapper that integrates with AttentionMatchingCompressor.
Provides MLX-LM compatible interface with automatic compression.
"""

from typing import Optional, Tuple, Dict, Any
import mlx.core as mx

from .layer_scheduler import LayerScheduler
from .hybrid_cache_manager import LayerType


class CompressedKVCache:
    """
    Compressed KV cache wrapper for Attention layers using Attention Matching.

    Wraps standard KV cache operations and applies Attention Matching compression
    with β calibration for memory-efficient storage.

    Example:
        >>> from flashmlx.cache import (
        ...     HybridCacheManager, HybridCacheConfig, LayerType,
        ...     LayerScheduler, CompressedKVCache
        ... )
        >>> config = HybridCacheConfig(total_budget_bytes=256 * 1024 * 1024)
        >>> layer_types = {0: LayerType.ATTENTION, 1: LayerType.ATTENTION, ...}
        >>> manager = HybridCacheManager(config=config, layer_types=layer_types)
        >>> scheduler = LayerScheduler(manager)
        >>> cache = CompressedKVCache(scheduler)
        >>> # Store and compress KV cache
        >>> keys = mx.zeros((1, 8, 100, 64))
        >>> values = mx.zeros((1, 8, 100, 64))
        >>> query = mx.zeros((1, 8, 1, 64))
        >>> compressed_k, compressed_v = cache.update_and_fetch(
        ...     layer_idx=0, keys=keys, values=values, query=query
        ... )
        >>> # Retrieve compressed KV cache
        >>> cached = cache.retrieve(0)
    """

    def __init__(self, scheduler: LayerScheduler):
        """
        Initialize Compressed KV Cache.

        Args:
            scheduler: LayerScheduler instance for routing operations
        """
        self.scheduler = scheduler

        # Local cache for compressed KV
        # This stores the compressed KV returned from Attention Matching compression
        self._local_cache: Dict[int, Tuple[mx.array, mx.array]] = {}

        # Statistics
        self.total_updates = 0
        self.total_retrievals = 0
        self.local_cache_hits = 0
        self.total_compression_ratio = 0.0

    def update_and_fetch(
        self,
        layer_idx: int,
        keys: mx.array,
        values: mx.array,
        query: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        """
        Update KV cache with compression and fetch compressed cache.

        Args:
            layer_idx: Layer index
            keys: Key array (batch, num_heads, seq_len, head_dim)
            values: Value array (batch, num_heads, seq_len, head_dim)
            query: Query array for attention matching (optional)
                   Shape: (batch, num_heads, query_len, head_dim)

        Returns:
            (compressed_keys, compressed_values)
            Compressed KV cache with reduced sequence length

        Raises:
            ValueError: If layer is not an Attention layer
        """
        # Verify this is an Attention layer
        layer_type = self.scheduler.get_layer_type(layer_idx)
        if layer_type != LayerType.ATTENTION:
            raise ValueError(
                f"Layer {layer_idx} is not an Attention layer (got {layer_type}). "
                f"Use ManagedArraysCache for SSM layers."
            )

        # Store through scheduler (applies Attention Matching compression)
        compressed_keys, compressed_values = self.scheduler.store(
            layer_idx=layer_idx,
            data=(keys, values),
            query=query
        )

        # Update local cache with compressed KV
        self._local_cache[layer_idx] = (compressed_keys, compressed_values)

        # Update statistics
        self.total_updates += 1

        # Calculate compression ratio
        original_seq_len = keys.shape[2]
        compressed_seq_len = compressed_keys.shape[2]
        compression_ratio = original_seq_len / compressed_seq_len if compressed_seq_len > 0 else 1.0
        self.total_compression_ratio += compression_ratio

        return compressed_keys, compressed_values

    def retrieve(self, layer_idx: int) -> Optional[Tuple[mx.array, mx.array]]:
        """
        Retrieve compressed KV cache.

        Note: For Attention layers, compression happens during store,
        and the compressed KV is cached locally. This method returns
        the locally cached compressed KV.

        Args:
            layer_idx: Layer index

        Returns:
            (compressed_keys, compressed_values) if found, None otherwise

        Raises:
            ValueError: If layer is not an Attention layer
        """
        # Verify this is an Attention layer
        layer_type = self.scheduler.get_layer_type(layer_idx)
        if layer_type != LayerType.ATTENTION:
            raise ValueError(
                f"Layer {layer_idx} is not an Attention layer (got {layer_type}). "
                f"Use ManagedArraysCache for SSM layers."
            )

        self.total_retrievals += 1

        # Return locally cached compressed KV
        if layer_idx in self._local_cache:
            self.local_cache_hits += 1
            return self._local_cache[layer_idx]

        return None

    def contains(self, layer_idx: int) -> bool:
        """
        Check if layer has compressed KV cache.

        Args:
            layer_idx: Layer index

        Returns:
            True if layer has compressed KV cache
        """
        return layer_idx in self._local_cache

    def clear(self, layer_idx: Optional[int] = None):
        """
        Clear compressed KV cache.

        Args:
            layer_idx: If provided, clear only this layer. Otherwise clear all.
        """
        if layer_idx is not None:
            # Clear specific layer
            self._local_cache.pop(layer_idx, None)
        else:
            # Clear all local cache
            self._local_cache.clear()

            # Note: We don't call scheduler.clear() here because
            # Attention compression doesn't use persistent storage
            # (compressed KV is returned immediately during store)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get compression statistics.

        Returns:
            Dictionary with statistics including compression metrics
        """
        # Get managed cache stats (includes Attention compression stats)
        managed_stats = self.scheduler.get_statistics()

        # Calculate average compression ratio
        avg_compression_ratio = (
            self.total_compression_ratio / self.total_updates
            if self.total_updates > 0
            else 1.0
        )

        # Add local cache stats
        stats = {
            "managed": managed_stats,
            "local_cache": {
                "size": len(self._local_cache),
                "total_updates": self.total_updates,
                "total_retrievals": self.total_retrievals,
                "local_cache_hits": self.local_cache_hits,
                "local_cache_hit_rate": (
                    self.local_cache_hits / self.total_retrievals
                    if self.total_retrievals > 0
                    else 0.0
                ),
                "avg_compression_ratio": avg_compression_ratio,
            },
        }

        return stats

    def get_compression_ratio(self) -> float:
        """
        Get average compression ratio across all updates.

        Returns:
            Average compression ratio (original_len / compressed_len)
        """
        if self.total_updates == 0:
            return 1.0
        return self.total_compression_ratio / self.total_updates

    def __len__(self) -> int:
        """Return number of cached layers"""
        return len(self._local_cache)

    def __contains__(self, layer_idx: int) -> bool:
        """Check if layer is cached"""
        return layer_idx in self._local_cache

    def __repr__(self) -> str:
        avg_ratio = self.get_compression_ratio()
        return (
            f"CompressedKVCache("
            f"local_size={len(self._local_cache)}, "
            f"updates={self.total_updates}, "
            f"avg_compression={avg_ratio:.2f}x)"
        )
