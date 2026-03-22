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
        Set cache slot value.

        Note: Compression is handled in update_and_fetch(), not here.
        This method is only used for direct cache manipulation.

        Args:
            idx: Slot index (0 for keys, 1 for values)
            value: Key or Value array
        """
        # Store as-is (no compression in direct assignment)
        # Compression happens in update_and_fetch()
        super().__setitem__(idx, value)

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """
        Update KV cache with new keys/values and return compressed cache.

        This is the main interface used by MLX-LM Attention layers.

        Args:
            keys: New key array (batch, num_heads, new_seq_len, head_dim)
            values: New value array (batch, num_heads, new_seq_len, head_dim)

        Returns:
            (compressed_keys, compressed_values) - Full KV cache
        """
        # Get current cached keys and values
        cached_keys = self.cache[0]
        cached_values = self.cache[1]

        # Concatenate with new keys/values
        if cached_keys is None:
            # First call - no existing cache
            full_keys = keys
            full_values = values
        else:
            # Concatenate along sequence dimension (axis 2)
            full_keys = mx.concatenate([cached_keys, keys], axis=2)
            full_values = mx.concatenate([cached_values, values], axis=2)

        # Apply Attention Matching compression
        compressed_keys, compressed_values = self.manager.store_attention(
            layer_idx=self.layer_idx,
            keys=full_keys,
            values=full_values,
            query=self._last_query,
            size_bytes=full_keys.nbytes + full_values.nbytes if hasattr(full_keys, 'nbytes') else 0
        )

        # Store compressed KV in cache
        self.cache[0] = compressed_keys
        self.cache[1] = compressed_values

        # Update compression stats
        original_size = full_keys.shape[2] if len(full_keys.shape) > 2 else 0
        compressed_size = compressed_keys.shape[2] if len(compressed_keys.shape) > 2 else 0
        if compressed_size > 0:
            ratio = original_size / compressed_size
            self._total_compressions += 1
            self._total_compression_ratio += ratio

        return compressed_keys, compressed_values

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

    def make_mask(self, N: int, return_array: bool = False, window_size: Optional[int] = None):
        """
        Create attention mask (compatible with MLX-LM).

        Args:
            N: Sequence length
            return_array: If True, always return array instead of string
            window_size: Optional sliding window size

        Returns:
            Attention mask (can be "causal" string or actual mask array)
        """
        # Import create_causal_mask from MLX-LM
        from mlx_lm.models.base import create_causal_mask

        # For single token generation (N=1), no mask needed
        if N == 1:
            return None

        offset = self.offset

        # Use causal mask with offset
        if offset + N > (window_size or float('inf')) or return_array:
            return create_causal_mask(N, offset, window_size=window_size)
        else:
            return "causal"

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
