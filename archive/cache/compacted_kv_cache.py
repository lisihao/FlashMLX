"""
CompactedKVCache - MLX implementation for storing pre-compacted KV cache with beta bias terms.
"""
from typing import List, Tuple, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache


class CompactedKVCacheLayer(KVCache):
    """
    Per-layer CompactedKVCache

    Stores compressed KV cache + beta for a single transformer layer.
    """

    def __init__(
        self,
        c1: mx.array,  # (B, n_kv_heads, t, head_dim)
        beta: mx.array,  # (B, n_kv_heads, t)
        c2: mx.array,  # (B, n_kv_heads, t, head_dim)
        layer_idx: int,
        original_seq_len: Optional[int] = None,
    ):
        """
        Initialize per-layer compacted cache

        Args:
            c1: Compressed keys
            beta: Bias terms
            c2: Compressed values
            layer_idx: Layer index (for debugging)
            original_seq_len: Original sequence length before compaction
        """
        super().__init__()

        self.keys = c1
        self.values = c2
        self.beta = beta
        self.layer_idx = layer_idx
        self._original_seq_len = original_seq_len

        # Set offset to compressed length
        self.offset = c1.shape[-2]

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Update cache with new keys/values by concatenation (no compression)
        """
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=-2)
            self.values = mx.concatenate([self.values, values], axis=-2)

        self.offset = self.keys.shape[-2]
        return self.keys, self.values

    def get_beta(self) -> Optional[mx.array]:
        """
        Get beta for this layer

        Returns:
            Beta array of shape (B, n_kv_heads, t)
        """
        return self.beta

    @property
    def original_seq_len(self) -> Optional[int]:
        """Return the original sequence length before compaction"""
        return self._original_seq_len


class CompactedKVCache(KVCache):
    """
    MLX version of CompactedPrefixCache

    Stores pre-compressed KV cache + beta bias terms for each layer.
    Inherits from KVCache to maintain compatibility with MLX-LM's attention mechanisms.
    """

    def __init__(
        self,
        compacted_cache: List[Tuple[mx.array, mx.array, mx.array]],
        original_seq_len: Optional[int] = None,
    ):
        """
        Initialize CompactedKVCache with pre-computed compacted cache data.

        Args:
            compacted_cache: List of (C1, beta, C2) tuples per layer
                - C1: mx.array of shape (B, n_kv_heads, t, head_dim) - compacted keys
                - beta: mx.array of shape (B, n_kv_heads, t) - bias terms
                - C2: mx.array of shape (B, n_kv_heads, t, head_dim) - compacted values
                where t is the compacted sequence length
            original_seq_len: Original sequence length before compaction (optional)
        """
        super().__init__()

        if not compacted_cache:
            raise ValueError("compacted_cache cannot be empty")

        # Extract keys and values from first layer's compacted cache
        first_layer_c1, first_layer_beta, first_layer_c2 = compacted_cache[0]
        self.keys = first_layer_c1
        self.values = first_layer_c2

        # Initialize beta_cache for storing per-layer beta values
        self.beta_cache: dict[int, mx.array] = {}

        # Store beta for each layer
        for layer_idx, (_, beta, _) in enumerate(compacted_cache):
            self.beta_cache[layer_idx] = beta

        # Set offset to the compacted sequence length
        # shape: (B, n_kv_heads, t, head_dim) -> t is at index -2
        self.offset = first_layer_c1.shape[-2]

        # Store original sequence length if provided (for reference)
        self._original_seq_len = original_seq_len

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Update cache with new keys/values by concatenation (no compression).

        This is a simple append operation, consistent with PyTorch implementation.

        Args:
            keys: mx.array of shape (B, n_kv_heads, new_len, head_dim)
            values: mx.array of shape (B, n_kv_heads, new_len, head_dim)

        Returns:
            Tuple of (keys, values) representing the complete cache after update
        """
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=-2)
            self.values = mx.concatenate([self.values, values], axis=-2)

        # Update offset to reflect new sequence length
        self.offset = self.keys.shape[-2]

        return self.keys, self.values

    def beta_for_layer(self, layer_idx: int) -> Optional[mx.array]:
        """
        Retrieve the beta (bias term) for a specific layer.

        Args:
            layer_idx: Layer index (0-indexed)

        Returns:
            Beta array of shape (B, n_kv_heads, t) if exists, None otherwise
        """
        return self.beta_cache.get(layer_idx)

    @property
    def original_seq_len(self) -> Optional[int]:
        """Return the original sequence length before compaction."""
        return self._original_seq_len


def create_compacted_cache_list(
    compacted_cache: List[Tuple[mx.array, mx.array, mx.array]],
    original_seq_len: Optional[int] = None,
) -> List[CompactedKVCacheLayer]:
    """
    Create a list of per-layer CompactedKVCache objects

    This is the recommended way to create compacted caches for MLX-LM models.

    Args:
        compacted_cache: List of (C1, beta, C2) tuples per layer
        original_seq_len: Original sequence length before compaction

    Returns:
        List of CompactedKVCacheLayer objects, one per layer

    Example:
        >>> compacted_data = [(c1_0, beta_0, c2_0), (c1_1, beta_1, c2_1), ...]
        >>> cache_list = create_compacted_cache_list(compacted_data)
        >>> output = model(input_ids, cache=cache_list)
    """
    cache_list = []
    for layer_idx, (c1, beta, c2) in enumerate(compacted_cache):
        layer_cache = CompactedKVCacheLayer(
            c1=c1,
            beta=beta,
            c2=c2,
            layer_idx=layer_idx,
            original_seq_len=original_seq_len
        )
        cache_list.append(layer_cache)

    return cache_list
