"""
Per-Layer SSM Cache

MLX-LM compatible cache for SSM layers with hybrid memory management.
Each instance manages cache for a single SSM layer, but shares a global
HybridCacheManager for coordinated budget allocation.
"""

from typing import Optional, List
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from .hybrid_cache_manager import HybridCacheManager, LayerType


class PerLayerSSMCache(ArraysCache):
    """
    Per-layer cache for SSM (State-Space Model) layers.

    Inherits from MLX-LM's ArraysCache for full compatibility, but routes
    storage operations through HybridCacheManager for tiered memory management.

    SSM layers typically have 2 cache slots:
    - Slot 0: Convolution state
    - Slot 1: SSM state

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
            manager: Shared HybridCacheManager for memory coordination
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

        # Track whether we've stored to managed cache
        self._managed_storage = [False] * size

    def __setitem__(self, idx: int, value: mx.array):
        """
        Set cache slot value and store to managed cache.

        Args:
            idx: Slot index (0 for conv_state, 1 for ssm_state)
            value: State array
        """
        # Store locally (in-memory, fast access)
        super().__setitem__(idx, value)

        # Also store in managed cache (tiered Hot/Warm/Cold)
        if value is not None:
            # Estimate size in bytes
            size_bytes = value.nbytes if hasattr(value, 'nbytes') else 0

            # Store to managed cache via HybridCacheManager
            # (This will route to Hot/Warm/Cold tiers based on access patterns)
            self.manager.store_ssm(
                layer_idx=self.layer_idx,
                data=value,
                size_bytes=size_bytes,
                priority=1.0  # Default priority
            )

            self._managed_storage[idx] = True

    def __getitem__(self, idx: int) -> Optional[mx.array]:
        """
        Get cache slot value, trying managed cache if not in local cache.

        Args:
            idx: Slot index

        Returns:
            Cached state array or None
        """
        # Try local cache first
        value = super().__getitem__(idx)

        # If not found locally but was stored to managed cache, try retrieving
        if value is None and idx < len(self._managed_storage) and self._managed_storage[idx]:
            value = self.manager.retrieve_ssm(self.layer_idx)
            if value is not None:
                # Re-populate local cache
                super().__setitem__(idx, value)

        return value

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

    def empty(self) -> bool:
        """Check if cache is empty."""
        return all(c is None for c in self.cache)

    @property
    def nbytes(self) -> int:
        """Return total size in bytes."""
        return sum(c.nbytes for c in self.cache if c is not None)

    def clear(self):
        """Clear both local and managed cache."""
        # Clear local cache
        for i in range(len(self.cache)):
            self.cache[i] = None
            self._managed_storage[i] = False

        # Clear managed cache for this layer
        self.manager.clear_ssm(self.layer_idx)
