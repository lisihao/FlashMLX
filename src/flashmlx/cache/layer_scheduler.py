"""
Layer Scheduler

Automatic routing for hybrid cache management based on layer type.
Provides unified store/retrieve interface that routes to appropriate strategies.
"""

from typing import Optional, Tuple, Union, Dict, Any
import mlx.core as mx

from .hybrid_cache_manager import HybridCacheManager, LayerType


class LayerScheduler:
    """
    Automatic layer routing for hybrid cache management.

    Routes cache operations to appropriate strategies based on layer type:
    - SSM layers → Hybrid Memory Manager v3 (Hot/Warm/Cold/Pinned)
    - Attention layers → Attention Matching compression

    Example:
        >>> from flashmlx.cache import HybridCacheManager, HybridCacheConfig, LayerType, LayerScheduler
        >>> config = HybridCacheConfig(total_budget_bytes=256 * 1024 * 1024)
        >>> layer_types = {0: LayerType.SSM, 1: LayerType.ATTENTION, ...}
        >>> manager = HybridCacheManager(config=config, layer_types=layer_types)
        >>> scheduler = LayerScheduler(manager)
        >>> # Unified interface - automatic routing
        >>> scheduler.store(0, mx.zeros((10, 64)), size_bytes=2560)  # SSM layer
        >>> scheduler.store(1, (keys, values))  # Attention layer
    """

    def __init__(self, hybrid_manager: HybridCacheManager):
        """
        Initialize Layer Scheduler.

        Args:
            hybrid_manager: HybridCacheManager instance to route to
        """
        self.hybrid_manager = hybrid_manager

    def store(
        self,
        layer_idx: int,
        data: Union[mx.array, Tuple[mx.array, mx.array]],
        **kwargs
    ) -> Optional[Union[bool, Tuple[mx.array, mx.array]]]:
        """
        Store cache data for any layer type.

        Automatically routes to:
        - store_ssm() for SSM layers
        - store_attention() for Attention layers

        Args:
            layer_idx: Layer index
            data: Cache data
                  - For SSM layers: mx.array (state array)
                  - For Attention layers: Tuple[mx.array, mx.array] (keys, values)
            **kwargs: Additional arguments passed to underlying method
                     - For SSM: size_bytes, priority
                     - For Attention: query, size_bytes

        Returns:
            - For SSM layers: bool (True if stored successfully)
            - For Attention layers: (compressed_keys, compressed_values)

        Raises:
            ValueError: If layer type is unknown or data format is incorrect
        """
        layer_type = self.hybrid_manager.get_layer_type(layer_idx)

        if layer_type is None:
            raise ValueError(f"Layer {layer_idx} not found in layer_types mapping")

        if layer_type == LayerType.SSM:
            # SSM layer - data should be mx.array
            if isinstance(data, tuple):
                raise ValueError(
                    f"SSM layer {layer_idx} expects mx.array, got tuple. "
                    f"Did you mean to use an Attention layer?"
                )
            return self.hybrid_manager.store_ssm(layer_idx, data, **kwargs)

        elif layer_type == LayerType.ATTENTION:
            # Attention layer - data should be (keys, values) tuple
            if not isinstance(data, tuple):
                raise ValueError(
                    f"Attention layer {layer_idx} expects (keys, values) tuple, got {type(data)}. "
                    f"Did you mean to use an SSM layer?"
                )
            if len(data) != 2:
                raise ValueError(
                    f"Attention layer {layer_idx} expects (keys, values) tuple of length 2, "
                    f"got tuple of length {len(data)}"
                )

            keys, values = data
            return self.hybrid_manager.store_attention(layer_idx, keys, values, **kwargs)

        else:
            raise ValueError(f"Unknown layer type {layer_type} for layer {layer_idx}")

    def retrieve(self, layer_idx: int) -> Optional[mx.array]:
        """
        Retrieve cache data for any layer type.

        Automatically routes to:
        - retrieve_ssm() for SSM layers
        - (Attention layers don't support retrieval - compressed KV is returned during store)

        Args:
            layer_idx: Layer index

        Returns:
            - For SSM layers: mx.array if found, None otherwise
            - For Attention layers: None (compression happens during store)

        Raises:
            ValueError: If layer type is unknown
        """
        layer_type = self.hybrid_manager.get_layer_type(layer_idx)

        if layer_type is None:
            raise ValueError(f"Layer {layer_idx} not found in layer_types mapping")

        if layer_type == LayerType.SSM:
            return self.hybrid_manager.retrieve_ssm(layer_idx)

        elif layer_type == LayerType.ATTENTION:
            # Attention layers don't support retrieval
            # Compressed KV cache is returned during store_attention()
            return None

        else:
            raise ValueError(f"Unknown layer type {layer_type} for layer {layer_idx}")

    def get_layer_type(self, layer_idx: int) -> Optional[LayerType]:
        """
        Get layer type for a given index.

        Args:
            layer_idx: Layer index

        Returns:
            LayerType if found, None otherwise
        """
        return self.hybrid_manager.get_layer_type(layer_idx)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get unified cache statistics.

        Returns:
            Dictionary with comprehensive statistics from HybridCacheManager
        """
        return self.hybrid_manager.get_statistics()

    def clear(self):
        """Clear all caches"""
        self.hybrid_manager.clear()

    def __repr__(self) -> str:
        return f"LayerScheduler(manager={self.hybrid_manager})"
