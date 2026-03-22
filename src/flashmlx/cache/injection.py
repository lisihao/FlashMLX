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
) -> HybridCacheWrapper:
    """
    Inject hybrid cache manager into MLX-LM model.

    This function creates a HybridCacheManager and wraps it with
    ManagedArraysCache and CompressedKVCache, providing automatic
    routing based on layer type.

    Args:
        model: MLX-LM model instance (e.g., Qwen3.5 model)
        config: HybridCacheConfig with budget and compression settings
        layer_types: Dictionary mapping layer_idx → LayerType
        auto_inject: If True, automatically replace model.cache (default: True)

    Returns:
        HybridCacheWrapper instance for manual control

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
        >>> cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)
        >>>
        >>> # Model now uses hybrid cache automatically
        >>> output = model.generate(prompt, max_tokens=100)
    """
    # Create HybridCacheManager
    manager = HybridCacheManager(config=config, layer_types=layer_types)

    # Create LayerScheduler
    scheduler = LayerScheduler(manager)

    # Create cache wrappers
    ssm_cache = ManagedArraysCache(scheduler)
    attention_cache = CompressedKVCache(scheduler)

    # Create unified wrapper
    wrapper = HybridCacheWrapper(
        scheduler=scheduler,
        ssm_cache=ssm_cache,
        attention_cache=attention_cache
    )

    # Auto-inject if requested
    if auto_inject:
        # Store original cache (for restoration if needed)
        if hasattr(model, 'cache'):
            wrapper._original_cache = model.cache

        # Replace model cache
        model.cache = wrapper

    return wrapper


def restore_original_cache(model: Any, wrapper: HybridCacheWrapper):
    """
    Restore original cache from before injection.

    Args:
        model: Model instance
        wrapper: HybridCacheWrapper with stored original cache

    Example:
        >>> wrapper = inject_hybrid_cache_manager(model, config, layer_types)
        >>> # ... use model ...
        >>> restore_original_cache(model, wrapper)
    """
    if hasattr(wrapper, '_original_cache'):
        model.cache = wrapper._original_cache
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
