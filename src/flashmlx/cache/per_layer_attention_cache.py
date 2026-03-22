"""
Per-Layer Attention Cache

MLX-LM compatible cache for Attention layers with KV cache compression.
Each instance manages cache for a single Attention layer, using Attention
Matching compression via shared HybridCacheManager.
"""

from typing import Optional, List
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from .hybrid_cache_manager import HybridCacheManager, LayerType


class PerLayerAttentionCache(ArraysCache):
    """
    Per-layer cache for Attention layers with KV compression.

    Inherits from MLX-LM's ArraysCache for full compatibility, but applies
    Attention Matching compression when storing keys/values.

    Attention layers typically have 2 cache slots:
    - Slot 0: Keys
    - Slot 1: Values

    Example:
        >>> manager = HybridCacheManager(config, layer_types)
        >>> cache = PerLayerAttentionCache(manager, layer_idx=3, size=2)
        >>> # MLX-LM will call:
        >>> cache[0] = keys      # Automatically compressed
        >>> cache[1] = values    # Automatically compressed
        >>> # Retrieve:
        >>> compressed_keys = cache[0]
        >>> compressed_values = cache[1]
    """

    def __init__(
        self,
        manager: HybridCacheManager,
        layer_idx: int,
        size: int = 2,
        left_padding: Optional[List[int]] = None
    ):
        """
        Initialize per-layer Attention cache.

        Args:
            manager: Shared HybridCacheManager for memory coordination
            layer_idx: Layer index this cache manages
            size: Number of cache slots (default 2: keys + values)
            left_padding: Optional left padding for batched inputs
        """
        # Call parent init
        super().__init__(size, left_padding)

        self.manager = manager
        self.layer_idx = layer_idx

        # Verify this is an Attention layer
        layer_type = manager.get_layer_type(layer_idx)
        if layer_type != LayerType.ATTENTION:
            raise ValueError(
                f"PerLayerAttentionCache used for non-Attention layer {layer_idx} "
                f"(type: {layer_type})"
            )

        # Track last query for compression
        self._last_query = None

        # Track compression stats for this layer
        self._total_compressions = 0
        self._total_compression_ratio = 0.0

    def __setitem__(self, idx: int, value: mx.array):
        """
        Set cache slot value with automatic compression.

        For Attention layers, keys (slot 0) and values (slot 1) are compressed
        using Attention Matching algorithm with β-calibration.

        Args:
            idx: Slot index (0 for keys, 1 for values)
            value: Key or Value array
        """
        if value is None:
            super().__setitem__(idx, value)
            return

        # If we have both keys and values, compress them together
        if idx == 1 and self.cache[0] is not None:
            # Slot 1 (values) - compress with keys
            keys = self.cache[0]
            values = value

            # Apply Attention Matching compression
            compressed_keys, compressed_values = self.manager.store_attention(
                layer_idx=self.layer_idx,
                keys=keys,
                values=values,
                query=self._last_query,  # Use last query if available
                size_bytes=keys.nbytes + values.nbytes if hasattr(keys, 'nbytes') else 0
            )

            # Store compressed KV
            super().__setitem__(0, compressed_keys)
            super().__setitem__(1, compressed_values)

            # Update compression stats
            original_size = keys.shape[2] if len(keys.shape) > 2 else 0
            compressed_size = compressed_keys.shape[2] if len(compressed_keys.shape) > 2 else 0
            if compressed_size > 0:
                ratio = original_size / compressed_size
                self._total_compressions += 1
                self._total_compression_ratio += ratio

        elif idx == 0:
            # Slot 0 (keys) - store temporarily, wait for values
            super().__setitem__(idx, value)

        else:
            # Other slots - store as-is
            super().__setitem__(idx, value)

    def set_query(self, query: mx.array):
        """
        Set current query for Attention Matching compression.

        Args:
            query: Query array for computing attention scores
        """
        self._last_query = query

    @property
    def offset(self) -> int:
        """
        Return current sequence offset (number of cached tokens).

        For Attention layers, offset is the sequence length of cached KV.
        """
        # Check keys (slot 0)
        if self.cache[0] is not None and hasattr(self.cache[0], 'shape'):
            # KV cache shape: (batch, num_heads, seq_len, head_dim)
            # Return seq_len dimension (axis 2)
            if len(self.cache[0].shape) >= 3:
                return self.cache[0].shape[2]
            elif len(self.cache[0].shape) >= 1:
                return self.cache[0].shape[0]

        return 0

    def empty(self) -> bool:
        """Check if cache is empty."""
        return all(c is None for c in self.cache)

    @property
    def nbytes(self) -> int:
        """Return total size in bytes."""
        return sum(c.nbytes for c in self.cache if c is not None)

    def get_compression_stats(self):
        """
        Get compression statistics for this layer.

        Returns:
            dict with compression metrics
        """
        avg_ratio = (
            self._total_compression_ratio / self._total_compressions
            if self._total_compressions > 0
            else 1.0
        )

        return {
            "layer_idx": self.layer_idx,
            "total_compressions": self._total_compressions,
            "avg_compression_ratio": avg_ratio,
        }

    def clear(self):
        """Clear cache."""
        for i in range(len(self.cache)):
            self.cache[i] = None

        self._last_query = None
        # Don't reset compression stats - they're cumulative
