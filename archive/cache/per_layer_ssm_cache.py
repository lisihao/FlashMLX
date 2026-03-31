"""
Per-Layer SSM Cache

⚠️  DEPRECATED: 2026-03-22 - SSM cache sealed. See SSM_CACHE_DEPRECATION.md


MLX-LM compatible cache for SSM layers.
Each instance manages cache for a single SSM layer within a single request.

Design Philosophy:
- SSM states are small and fixed-size (unlike KV cache which grows with seq_len)
- Within a single request: use simple in-memory cache (ArraysCache)
- Across requests: use SimplifiedSSMCacheManager for cross-request reuse

Note: This is different from PerLayerAttentionCache which compresses KV cache
      within a single request to save memory.

Changelog (2026-03-22):
- Replaced Hot/Warm/Cold three-tier cache with single-tier SimplifiedSSMCacheManager
- Reduced overhead from 16x to 1x (0.177 μs → 0.011 μs)
- Removed external storage, memory-only caching
"""

from typing import Optional, List, Union
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from .hybrid_cache_manager import HybridCacheManager, LayerType
from .simplified_ssm_cache import SimplifiedSSMCacheManager


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
        manager: Union[HybridCacheManager, SimplifiedSSMCacheManager],
        layer_idx: int,
        size: int = 2,
        left_padding: Optional[List[int]] = None
    ):
        """
        Initialize per-layer SSM cache.

        Args:
            manager: 缓存管理器
                - HybridCacheManager: 兼容旧版（deprecated）
                - SimplifiedSSMCacheManager: 推荐（低开销）
            layer_idx: Layer index this cache manages
            size: Number of cache slots (default 2 for SSM: conv + state)
            left_padding: Optional left padding for batched inputs
        """
        # Call parent init
        super().__init__(size, left_padding)

        self.manager = manager
        self.layer_idx = layer_idx
        self._use_simplified = isinstance(manager, SimplifiedSSMCacheManager)

        # Verify layer type (only if using HybridCacheManager)
        if isinstance(manager, HybridCacheManager):
            layer_type = manager.get_layer_type(layer_idx)
            if layer_type != LayerType.SSM:
                raise ValueError(
                    f"PerLayerSSMCache used for non-SSM layer {layer_idx} "
                    f"(type: {layer_type})"
                )

        # Track access count for testing
        self._access_count = 0
        self._use_managed_cache = False  # Default: simple mode (local cache only)

    def enable_managed_cache(self):
        """Enable managed cache for testing cross-layer access patterns."""
        self._use_managed_cache = True

    def __setitem__(self, idx: int, value: mx.array):
        """
        Set cache slot value.

        Modes:
        - Simple mode (default): Local ArraysCache only
        - Managed mode (opt-in): Store to shared cache for cross-request reuse
        """
        if self._use_managed_cache and value is not None:
            if self._use_simplified:
                # Use SimplifiedSSMCacheManager (single-tier, low overhead)
                self.manager.store(self.layer_idx, value)
            else:
                # Legacy: HybridCacheManager (Hot/Warm/Cold, deprecated)
                size_bytes = value.nbytes if hasattr(value, 'nbytes') else 0
                self.manager.store_ssm(
                    layer_idx=self.layer_idx,
                    data=value,
                    size_bytes=size_bytes,
                    priority=1.0
                )
            # Don't store to local cache - force retrieval from managed cache
        else:
            # Simple mode: use default ArraysCache (fastest)
            super().__setitem__(idx, value)

    def __getitem__(self, idx: int) -> Optional[mx.array]:
        """
        Get cache slot value.

        Modes:
        - Simple mode (default): Local ArraysCache only
        - Managed mode (opt-in): Retrieve from shared cache
        """
        self._access_count += 1

        if self._use_managed_cache:
            if self._use_simplified:
                # Use SimplifiedSSMCacheManager (direct dict lookup)
                value = self.manager.retrieve(self.layer_idx)
            else:
                # Legacy: HybridCacheManager (Hot/Warm/Cold lookup)
                value = self.manager.retrieve_ssm(self.layer_idx)

            if value is not None:
                # Temporarily store in local cache for MLX-LM compatibility
                super().__setitem__(idx, value)
            return value
        else:
            # Simple mode: use default ArraysCache (fastest)
            return super().__getitem__(idx)

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
