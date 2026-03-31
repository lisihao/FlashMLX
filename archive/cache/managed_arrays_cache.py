"""
Managed Arrays Cache

⚠️  DEPRECATED: 2026-03-22 - SSM cache sealed. See SSM_CACHE_DEPRECATION.md


SSM layer cache wrapper that integrates with HybridCacheManager.
Provides MLX-LM compatible interface with automatic memory management.
"""

from typing import Optional, Dict, Any
import mlx.core as mx

from .layer_scheduler import LayerScheduler
from .hybrid_cache_manager import LayerType


class ManagedArraysCache:
    """
    Managed wrapper for SSM layer cache using HybridCacheManager.

    Wraps standard cache operations and routes them through LayerScheduler
    for automatic memory management across Hot/Warm/Cold/Pinned tiers.

    Example:
        >>> from flashmlx.cache import (
        ...     HybridCacheManager, HybridCacheConfig, LayerType,
        ...     LayerScheduler, ManagedArraysCache
        ... )
        >>> config = HybridCacheConfig(total_budget_bytes=256 * 1024 * 1024)
        >>> layer_types = {0: LayerType.SSM, 1: LayerType.SSM, ...}
        >>> manager = HybridCacheManager(config=config, layer_types=layer_types)
        >>> scheduler = LayerScheduler(manager)
        >>> cache = ManagedArraysCache(scheduler)
        >>> # Store SSM state
        >>> state = mx.zeros((10, 64))
        >>> cache.update_and_fetch(layer_idx=0, state=state)
        >>> # Retrieve SSM state
        >>> cached_state = cache.retrieve(0)
    """

    def __init__(self, scheduler: LayerScheduler):
        """
        Initialize Managed Arrays Cache.

        Args:
            scheduler: LayerScheduler instance for routing operations
        """
        self.scheduler = scheduler

        # Local cache for quick access
        # This serves as L0 cache before going to Hot/Warm/Cold tiers
        self._local_cache: Dict[int, mx.array] = {}

        # Statistics
        self.total_updates = 0
        self.total_retrievals = 0
        self.local_cache_hits = 0

    def update_and_fetch(
        self,
        layer_idx: int,
        state: mx.array,
        priority: float = 1.0
    ) -> mx.array:
        """
        Update cache for SSM layer and fetch current state.

        Args:
            layer_idx: Layer index
            state: New SSM state array
            priority: Priority score (higher = more important, default 1.0)

        Returns:
            Current cached state (same as input state after update)

        Raises:
            ValueError: If layer is not an SSM layer
        """
        # Verify this is an SSM layer
        layer_type = self.scheduler.get_layer_type(layer_idx)
        if layer_type != LayerType.SSM:
            raise ValueError(
                f"Layer {layer_idx} is not an SSM layer (got {layer_type}). "
                f"Use CompressedKVCache for Attention layers."
            )

        # Calculate size in bytes
        size_bytes = state.nbytes

        # Store through scheduler (routes to HybridCacheManager)
        success = self.scheduler.store(
            layer_idx=layer_idx,
            data=state,
            size_bytes=size_bytes,
            priority=priority
        )

        if success:
            # Update local cache for quick access
            self._local_cache[layer_idx] = state

        self.total_updates += 1

        return state

    def retrieve(self, layer_idx: int) -> Optional[mx.array]:
        """
        Retrieve cached state for SSM layer.

        Args:
            layer_idx: Layer index

        Returns:
            Cached state if found, None otherwise

        Raises:
            ValueError: If layer is not an SSM layer
        """
        # Verify this is an SSM layer
        layer_type = self.scheduler.get_layer_type(layer_idx)
        if layer_type != LayerType.SSM:
            raise ValueError(
                f"Layer {layer_idx} is not an SSM layer (got {layer_type}). "
                f"Use CompressedKVCache for Attention layers."
            )

        self.total_retrievals += 1

        # Try local cache first (L0 - fastest)
        if layer_idx in self._local_cache:
            self.local_cache_hits += 1
            return self._local_cache[layer_idx]

        # Try scheduler retrieval (routes to Hot/Warm/Cold tiers)
        cached = self.scheduler.retrieve(layer_idx)

        if cached is not None:
            # Update local cache
            self._local_cache[layer_idx] = cached

        return cached

    def contains(self, layer_idx: int) -> bool:
        """
        Check if layer is cached.

        Args:
            layer_idx: Layer index

        Returns:
            True if layer is in cache (local or managed)
        """
        # Check local cache
        if layer_idx in self._local_cache:
            return True

        # Check managed cache
        cached = self.scheduler.retrieve(layer_idx)
        return cached is not None

    def clear(self, layer_idx: Optional[int] = None):
        """
        Clear cache.

        Args:
            layer_idx: If provided, clear only this layer. Otherwise clear all.
        """
        if layer_idx is not None:
            # Clear specific layer
            self._local_cache.pop(layer_idx, None)
        else:
            # Clear all local cache
            self._local_cache.clear()

            # Clear managed cache
            self.scheduler.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with statistics including local and managed cache metrics
        """
        # Get managed cache stats
        managed_stats = self.scheduler.get_statistics()

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
            },
        }

        return stats

    def __len__(self) -> int:
        """Return number of cached layers in local cache"""
        return len(self._local_cache)

    def __contains__(self, layer_idx: int) -> bool:
        """Check if layer is in local cache"""
        return layer_idx in self._local_cache

    def __repr__(self) -> str:
        return (
            f"ManagedArraysCache("
            f"local_size={len(self._local_cache)}, "
            f"updates={self.total_updates}, "
            f"retrievals={self.total_retrievals})"
        )
