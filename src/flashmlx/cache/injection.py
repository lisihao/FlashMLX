"""
Monkey Patch Injection for Hybrid Cache Manager

Provides non-invasive injection of ManagedArraysCache and CompressedKVCache
into MLX-LM models without modifying source code.

Example:
    >>> from flashmlx.cache import (
    ...     inject_hybrid_cache_manager, HybridCacheConfig, LayerType
    ... )
    >>>
    >>> # Define layer types (Qwen3.5: 30 SSM + 10 Attention)
    >>> layer_types = {}
    >>> for i in range(40):
    ...     is_attention = (i + 1) % 4 == 0  # Every 4th layer is Attention
    ...     layer_types[i] = LayerType.ATTENTION if is_attention else LayerType.SSM
    >>>
    >>> # Configure hybrid cache
    >>> config = HybridCacheConfig(
    ...     total_budget_bytes=256 * 1024 * 1024,  # 256MB
    ...     compression_ratio=3.0
    ... )
    >>>
    >>> # Inject into model
    >>> cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)
    >>>
    >>> # Use model normally - cache is automatically managed
    >>> output = model.generate(...)
"""

from typing import Dict, Any, Optional
import mlx.core as mx

from .hybrid_cache_manager import HybridCacheManager, HybridCacheConfig, LayerType
from .layer_scheduler import LayerScheduler
from .managed_arrays_cache import ManagedArraysCache
from .compressed_kv_cache import CompressedKVCache
from .per_layer_ssm_cache import PerLayerSSMCache
from .per_layer_attention_cache import PerLayerAttentionCache


class LayerCacheProxy:
    """
    Per-layer cache proxy that forwards operations to HybridCacheWrapper.

    Makes hybrid cache compatible with MLX-LM's expectation that cache[layer_idx]
    returns a per-layer cache object.
    """

    def __init__(self, wrapper: 'HybridCacheWrapper', layer_idx: int):
        """
        Initialize layer cache proxy.

        Args:
            wrapper: Parent HybridCacheWrapper
            layer_idx: Layer index this proxy represents
        """
        self.wrapper = wrapper
        self.layer_idx = layer_idx
        self.layer_type = wrapper.get_layer_type(layer_idx)
        # Track internal state (for SSM layers with multi-slot cache)
        self._state_slots = {}

    def __getitem__(self, slot: int):
        """
        Get cache state for a slot (SSM layers use slots 0 and 1).

        Args:
            slot: Cache slot index

        Returns:
            Cached state array or None
        """
        return self._state_slots.get(slot)

    def __setitem__(self, slot: int, value: mx.array):
        """
        Set cache state for a slot.

        Args:
            slot: Cache slot index
            value: State array to cache
        """
        self._state_slots[slot] = value


class HybridCacheWrapper:
    """
    Wrapper that holds both SSM and Attention caches.

    Routes cache operations to the appropriate cache based on layer type.
    Provides a unified interface for model integration.
    """

    def __init__(
        self,
        scheduler: LayerScheduler,
        ssm_cache: ManagedArraysCache,
        attention_cache: CompressedKVCache
    ):
        """
        Initialize hybrid cache wrapper.

        Args:
            scheduler: LayerScheduler for routing
            ssm_cache: ManagedArraysCache for SSM layers
            attention_cache: CompressedKVCache for Attention layers
        """
        self.scheduler = scheduler
        self.ssm_cache = ssm_cache
        self.attention_cache = attention_cache
        # Cache proxy objects for each layer
        self._layer_proxies: Dict[int, LayerCacheProxy] = {}

    def update_and_fetch_ssm(
        self,
        layer_idx: int,
        state: mx.array,
        priority: float = 1.0
    ) -> mx.array:
        """
        Update and fetch SSM state.

        Args:
            layer_idx: Layer index
            state: SSM state array
            priority: Priority for cache management

        Returns:
            Updated state array
        """
        return self.ssm_cache.update_and_fetch(
            layer_idx=layer_idx,
            state=state,
            priority=priority
        )

    def update_and_fetch_attention(
        self,
        layer_idx: int,
        keys: mx.array,
        values: mx.array,
        query: Optional[mx.array] = None
    ) -> tuple[mx.array, mx.array]:
        """
        Update and fetch Attention KV cache with compression.

        Args:
            layer_idx: Layer index
            keys: Key array
            values: Value array
            query: Query array for attention matching

        Returns:
            (compressed_keys, compressed_values)
        """
        return self.attention_cache.update_and_fetch(
            layer_idx=layer_idx,
            keys=keys,
            values=values,
            query=query
        )

    def retrieve_ssm(self, layer_idx: int) -> Optional[mx.array]:
        """Retrieve SSM state."""
        return self.ssm_cache.retrieve(layer_idx)

    def retrieve_attention(self, layer_idx: int) -> Optional[tuple[mx.array, mx.array]]:
        """Retrieve Attention KV cache."""
        return self.attention_cache.retrieve(layer_idx)

    def clear(self, layer_idx: Optional[int] = None):
        """
        Clear cache.

        Args:
            layer_idx: If provided, clear only this layer. Otherwise clear all.
        """
        self.ssm_cache.clear(layer_idx)
        self.attention_cache.clear(layer_idx)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with SSM and Attention cache statistics
        """
        return {
            "ssm": self.ssm_cache.get_statistics(),
            "attention": self.attention_cache.get_statistics(),
            "scheduler": self.scheduler.get_statistics(),
        }

    def get_layer_type(self, layer_idx: int) -> LayerType:
        """Get layer type (SSM or ATTENTION)."""
        return self.scheduler.get_layer_type(layer_idx)

    def __getitem__(self, index: int) -> LayerCacheProxy:
        """
        Get cache proxy for a specific layer (makes wrapper indexable like list).

        Args:
            index: Layer index

        Returns:
            LayerCacheProxy for the layer
        """
        # Return cached proxy if exists
        if index in self._layer_proxies:
            return self._layer_proxies[index]

        # Create new proxy
        proxy = LayerCacheProxy(self, index)
        self._layer_proxies[index] = proxy
        return proxy

    def __setitem__(self, index: int, value):
        """
        Set cache for a specific layer (for list compatibility).

        Note: This is a no-op as our hybrid cache manages layers automatically.
        """
        # Our hybrid cache doesn't support per-layer cache replacement
        # This method exists only for interface compatibility
        pass

    def __len__(self) -> int:
        """
        Return number of layers.

        Returns:
            Total number of layers (SSM + Attention)
        """
        return (
            self.scheduler.hybrid_manager.num_ssm_layers +
            self.scheduler.hybrid_manager.num_attention_layers
        )

    def __repr__(self) -> str:
        ssm_stats = self.ssm_cache.get_statistics()
        attn_stats = self.attention_cache.get_statistics()

        return (
            f"HybridCacheWrapper(\n"
            f"  SSM: {ssm_stats['local_cache']['size']} layers, "
            f"{ssm_stats['local_cache']['total_updates']} updates\n"
            f"  Attention: {attn_stats['local_cache']['size']} layers, "
            f"{attn_stats['local_cache']['avg_compression_ratio']:.2f}x compression\n"
            f")"
        )


def inject_hybrid_cache_manager(
    model: Any,
    config: HybridCacheConfig,
    layer_types: Dict[int, LayerType],
    auto_inject: bool = True
) -> List[Any]:
    """
    Inject hybrid cache manager into MLX-LM model.

    This function creates a HybridCacheManager and returns a list of
    per-layer cache objects that are compatible with MLX-LM's expectations.

    Args:
        model: MLX-LM model instance (e.g., Qwen3.5 model)
        config: HybridCacheConfig with budget and compression settings
        layer_types: Dictionary mapping layer_idx → LayerType
        auto_inject: If True, automatically replace model.make_cache (default: True)

    Returns:
        List of per-layer cache objects (PerLayerSSMCache or PerLayerAttentionCache)

    Example:
        >>> # Qwen3.5-35B: 40 layers (30 SSM + 10 Attention)
        >>> layer_types = {}
        >>> for i in range(40):
        ...     # Every 4th layer is Attention
        ...     is_attention = (i + 1) % 4 == 0
        ...     layer_types[i] = LayerType.ATTENTION if is_attention else LayerType.SSM
        >>>
        >>> config = HybridCacheConfig(
        ...     total_budget_bytes=256 * 1024 * 1024,  # 256MB
        ...     compression_ratio=3.0
        ... )
        >>>
        >>> cache_list = inject_hybrid_cache_manager(model, config, layer_types)
        >>>
        >>> # Model now uses hybrid cache automatically
        >>> output = model.generate(prompt, max_tokens=100)
    """
    # Create HybridCacheManager (shared by all layers)
    manager = HybridCacheManager(config=config, layer_types=layer_types)

    # Create per-layer cache objects
    num_layers = len(layer_types)
    cache_list = []

    for layer_idx in range(num_layers):
        layer_type = layer_types.get(layer_idx)

        if layer_type == LayerType.SSM:
            # Create SSM cache for this layer (2 slots: conv_state + ssm_state)
            cache = PerLayerSSMCache(
                manager=manager,
                layer_idx=layer_idx,
                size=2
            )
        elif layer_type == LayerType.ATTENTION:
            # Create Attention cache for this layer (2 slots: keys + values)
            cache = PerLayerAttentionCache(
                manager=manager,
                layer_idx=layer_idx,
                size=2
            )
        else:
            raise ValueError(f"Unknown layer type for layer {layer_idx}: {layer_type}")

        cache_list.append(cache)

    # Auto-inject if requested
    if auto_inject:
        # Store original make_cache method if it exists
        if hasattr(model, 'make_cache'):
            original_make_cache = model.make_cache

            # Store in cache_list for restoration
            cache_list._original_make_cache = original_make_cache

            # Replace make_cache to return our cache list
            model.make_cache = lambda: cache_list

        # Also set model.cache for backward compatibility
        if hasattr(model, 'cache'):
            cache_list._original_cache = model.cache
        model.cache = cache_list

        # Store manager reference for statistics access
        cache_list._manager = manager

    return cache_list


def get_cache_statistics(cache_list: List[Any]) -> Dict[str, Any]:
    """
    Get comprehensive cache statistics from cache list.

    Args:
        cache_list: Per-layer cache list returned by inject_hybrid_cache_manager

    Returns:
        Dictionary with cache statistics

    Example:
        >>> cache_list = inject_hybrid_cache_manager(model, config, layer_types)
        >>> stats = get_cache_statistics(cache_list)
        >>> print(stats['ssm']['hit_rate'])
    """
    if not hasattr(cache_list, '_manager'):
        raise ValueError("Cache list does not have manager reference")

    # Get manager statistics
    manager_stats = cache_list._manager.get_statistics()

    # Add per-layer compression statistics
    attention_compression_stats = []
    for cache in cache_list:
        if isinstance(cache, PerLayerAttentionCache):
            attention_compression_stats.append(cache.get_compression_stats())

    # Combine statistics
    stats = manager_stats.copy()
    stats['per_layer_attention_compression'] = attention_compression_stats

    return stats


def restore_original_cache(model: Any, cache_list: List[Any]):
    """
    Restore original cache from before injection.

    Args:
        model: Model instance
        cache_list: Per-layer cache list with stored original cache

    Example:
        >>> cache_list = inject_hybrid_cache_manager(model, config, layer_types)
        >>> # ... use model ...
        >>> restore_original_cache(model, cache_list)
    """
    # Restore original make_cache method
    if hasattr(cache_list, '_original_make_cache'):
        model.make_cache = cache_list._original_make_cache

    # Restore original cache attribute
    if hasattr(cache_list, '_original_cache'):
        model.cache = cache_list._original_cache
    else:
        # No original cache, set to None
        model.cache = None


def create_layer_types_from_model(
    model: Any,
    attention_layer_indices: Optional[list[int]] = None,
    attention_layer_pattern: Optional[str] = None
) -> Dict[int, LayerType]:
    """
    Automatically detect layer types from model structure.

    Args:
        model: MLX-LM model instance
        attention_layer_indices: Explicit list of Attention layer indices
        attention_layer_pattern: Pattern for Attention layers (e.g., "every 4th")
            Supported patterns:
            - "every Nth" (e.g., "every 4th"): Every N-th layer is Attention
            - "last N" (e.g., "last 10"): Last N layers are Attention

    Returns:
        Dictionary mapping layer_idx → LayerType

    Example:
        >>> # Explicit indices
        >>> layer_types = create_layer_types_from_model(
        ...     model,
        ...     attention_layer_indices=[3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
        ... )
        >>>
        >>> # Pattern-based (Qwen3.5: every 4th layer)
        >>> layer_types = create_layer_types_from_model(
        ...     model,
        ...     attention_layer_pattern="every 4th"
        ... )
        >>>
        >>> # Pattern-based (last 10 layers are Attention)
        >>> layer_types = create_layer_types_from_model(
        ...     model,
        ...     attention_layer_pattern="last 10"
        ... )
    """
    # Get total number of layers
    num_layers = len(model.layers) if hasattr(model, 'layers') else 0

    if num_layers == 0:
        raise ValueError("Could not determine number of layers from model")

    layer_types = {}

    # Method 1: Explicit indices
    if attention_layer_indices is not None:
        attention_set = set(attention_layer_indices)
        for i in range(num_layers):
            layer_types[i] = (
                LayerType.ATTENTION if i in attention_set else LayerType.SSM
            )
        return layer_types

    # Method 2: Pattern-based
    if attention_layer_pattern is not None:
        pattern = attention_layer_pattern.lower().strip()

        # "every Nth" pattern
        if pattern.startswith("every "):
            # Extract N (e.g., "every 4th" → 4)
            n_str = pattern.replace("every ", "").rstrip("thstndrd")
            n = int(n_str)

            for i in range(num_layers):
                # (i+1) % n == 0 means every n-th layer
                is_attention = (i + 1) % n == 0
                layer_types[i] = LayerType.ATTENTION if is_attention else LayerType.SSM

            return layer_types

        # "last N" pattern
        if pattern.startswith("last "):
            # Extract N (e.g., "last 10" → 10)
            n_str = pattern.replace("last ", "")
            n = int(n_str)

            for i in range(num_layers):
                # Last N layers are Attention
                is_attention = i >= (num_layers - n)
                layer_types[i] = LayerType.ATTENTION if is_attention else LayerType.SSM

            return layer_types

        raise ValueError(
            f"Unsupported pattern: {attention_layer_pattern}. "
            f"Supported: 'every Nth', 'last N'"
        )

    # Method 3: Auto-detect from model structure
    # (Check if layer has self.self_attn attribute)
    for i in range(num_layers):
        layer = model.layers[i]
        has_attention = hasattr(layer, 'self_attn')
        layer_types[i] = LayerType.ATTENTION if has_attention else LayerType.SSM

    return layer_types
