"""
Per-Layer SSM Cache

MLX-LM compatible cache for SSM layers.
Each instance manages cache for a single SSM layer within a single request.

Design Philosophy:
- SSM states are small and fixed-size (unlike KV cache which grows with seq_len)
- Within a single request: use simple in-memory cache (ArraysCache)
- Across requests: future work for request-level caching with Hot/Warm/Cold tiers

Note: This is different from PerLayerAttentionCache which compresses KV cache
      within a single request to save memory.
"""

from typing import Optional, List
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from .hybrid_cache_manager import HybridCacheManager, LayerType


class PerLayerSSMCache(ArraysCache):
    """
    Per-layer cache for SSM (State-Space Model) layers.

    Inherits from MLX-LM's ArraysCache for full compatibility.
    Uses simple in-memory storage for single-request scenarios.

    SSM layers typically have 2 cache slots:
    - Slot 0: Convolution state
    - Slot 1: SSM state

    Design Note:
        Unlike Attention layers which need KV cache compression (memory pressure),
        SSM states are small and fixed-size. Hot/Warm/Cold tiering is designed
        for cross-request scenarios (e.g., reusing system prompt states),
        which is future work.

    Example:
        >>> manager = HybridCacheManager(config, layer_types)
        >>> cache = PerLayerSSMCache(manager, layer_idx=0, size=2)
        >>> # MLX-LM will call:
        >>> cache[0] = conv_state
        >>> cache[1] = ssm_state
        >>> # Retrieve:
        >>> cached_conv = cache[0]
        >>> cached_ssm = cache[1]
    """

    def __init__(
        self,
        manager: HybridCacheManager,
        layer_idx: int,
        size: int = 2,
        left_padding: Optional[List[int]] = None
    ):
        """
        Initialize per-layer SSM cache.

        Args:
            manager: Shared HybridCacheManager (for future cross-request caching)
            layer_idx: Layer index this cache manages
            size: Number of cache slots (default 2 for SSM: conv + state)
            left_padding: Optional left padding for batched inputs
        """
        # Call parent init
        super().__init__(size, left_padding)

        self.manager = manager
        self.layer_idx = layer_idx

        # Verify this is an SSM layer
        layer_type = manager.get_layer_type(layer_idx)
        if layer_type != LayerType.SSM:
            raise ValueError(
                f"PerLayerSSMCache used for non-SSM layer {layer_idx} "
                f"(type: {layer_type})"
            )

    # Removed __setitem__ and __getitem__ overrides
    # Use default ArraysCache behavior for single-request scenarios

    @property
    def offset(self) -> int:
        """
        Return current sequence offset (number of cached tokens).

        For SSM layers, offset is typically the length of cached state.
        """
        # Check if we have any cached state
        for slot in self.cache:
            if slot is not None and hasattr(slot, 'shape') and len(slot.shape) >= 2:
                # Return sequence length dimension (typically axis 1 for SSM)
                return slot.shape[1] if len(slot.shape) > 1 else slot.shape[0]

        return 0

    def make_mask(self, N: int, **kwargs):
        """
        Create attention mask (compatible with MLX-LM).

        Args:
            N: Sequence length
            **kwargs: Additional arguments (e.g., return_array, window_size)
                     These are ignored for SSM layers as they don't use masks

        Returns:
            None (SSM layers don't use attention masks)
        """
        # SSM layers don't use attention masks, just call parent
        return super().make_mask(N)

    def empty(self) -> bool:
        """Check if cache is empty."""
        return all(c is None for c in self.cache)

    @property
    def nbytes(self) -> int:
        """Return total size in bytes."""
        return sum(c.nbytes for c in self.cache if c is not None)

    def clear(self):
        """Clear cache."""
        # Clear local cache
        for i in range(len(self.cache)):
            self.cache[i] = None
