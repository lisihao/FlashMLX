"""Incremental KVTC Cache - Dynamic growth optimization.

This module implements incremental compression for KV cache, avoiding
re-encoding the entire cache when new tokens are added.

Key features:
- Incremental encoding: only encode newly added tokens
- Compression data merging: concatenate compressed payloads
- State tracking: track encoded token positions
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .kvtc_codec import (
    KVTCCodecConfig,
    KVTCSharedCalibration,
    decode_tensor,
    encode_tensor,
    fit_shared_calibration,
)


class IncrementalKVTCCache:
    """Incremental KVTC cache with dynamic growth optimization.

    This cache supports incremental compression:
    - Initial encoding: encode full cache
    - Append: only encode new tokens, merge with existing compressed data
    - Decode: reconstruct full cache from merged compressed data

    Usage:
        # Create from initial cache
        cache = IncrementalKVTCCache.from_cache(initial_cache, calibration)

        # Append new tokens (only encodes new part)
        cache.append(new_keys, new_values)

        # Decode when needed
        keys, values = cache.decode()
    """

    def __init__(self):
        # Compressed data: list of chunks
        # Each chunk: (encoded_keys, encoded_values, num_tokens)
        self._chunks: List[Tuple[Tuple, Tuple, int]] = []

        # Calibration
        self._shared_calibration: Optional[KVTCSharedCalibration] = None
        self._shared_calibration_id: Optional[str] = None

        # State tracking
        self._encoded_tokens: int = 0  # Total number of tokens encoded
        self._batch_size: int = 1
        self._num_heads: int = 0
        self._head_dim: int = 0

        # Metadata
        self._meta: Dict[str, Any] = {}

    @classmethod
    def from_cache(
        cls,
        keys: mx.array,  # [batch, heads, tokens, dim]
        values: mx.array,
        *,
        calibration: Optional[KVTCSharedCalibration] = None,
        energy: float = 0.995,
        rank: Optional[int] = None,
        bits: int = 4,
        group_size: int = 64,
        sample_limit: int = 4096,
        seed: int = 0,
    ):
        """Create incremental cache from initial K/V tensors.

        Args:
            keys: Initial keys [batch, heads, tokens, dim]
            values: Initial values [batch, heads, tokens, dim]
            calibration: Shared calibration (if None, fit new one)
            Other args: calibration parameters

        Returns:
            IncrementalKVTCCache instance
        """
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError(
                "Keys and values must be 4D tensors (batch, heads, tokens, dim)"
            )

        if keys.shape[0] != 1:
            raise ValueError("Incremental cache currently supports batch_size=1 only")

        # Fit calibration if not provided
        if calibration is None:
            codec = KVTCCodecConfig(
                rank=rank if rank is not None else 8,  # Use fixed rank like test_kvtc_cache.py
                bits=bits,
                group_size=group_size,
                sample_limit=sample_limit,
                seed=seed,
                zero_bit_energy_fraction=0.001,  # Prevent zero-bit fallback
            )
            # Use the actual data for calibration to avoid zero-bit allocation
            keys_flat = keys.reshape(-1, keys.shape[-1])
            values_flat = values.reshape(-1, values.shape[-1])
            calibration = fit_shared_calibration(
                [keys_flat],
                [values_flat],
                codec,
            )

        # Encode initial cache
        num_tokens = keys.shape[2]
        encoded_keys = tuple(
            x for x in encode_tensor(keys.reshape(-1, keys.shape[-1]), calibration.keys)
        )
        encoded_values = tuple(
            x for x in encode_tensor(values.reshape(-1, values.shape[-1]), calibration.values)
        )

        # Create instance with chunk-based storage
        obj = cls()
        obj._chunks = [(encoded_keys, encoded_values, num_tokens)]
        obj._shared_calibration = calibration
        obj._shared_calibration_id = calibration.fingerprint()
        obj._encoded_tokens = num_tokens
        obj._batch_size = keys.shape[0]
        obj._num_heads = keys.shape[1]
        obj._head_dim = keys.shape[3]
        obj._meta = {
            "version": 2,  # Version 2: chunk-based storage
            "encoded_tokens": obj._encoded_tokens,
            "batch_size": obj._batch_size,
            "num_heads": obj._num_heads,
            "head_dim": obj._head_dim,
            "num_chunks": len(obj._chunks),
            "shared_calibration_id": obj._shared_calibration_id,
        }

        return obj

    def append(
        self,
        new_keys: mx.array,  # [batch, heads, new_tokens, dim]
        new_values: mx.array,
    ):
        """Append new tokens to cache (incremental encoding).

        This only encodes the new tokens and adds them as a new chunk.

        Args:
            new_keys: New keys to append [batch, heads, new_tokens, dim]
            new_values: New values to append [batch, heads, new_tokens, dim]
        """
        if self._shared_calibration is None:
            raise ValueError("Cannot append to cache without calibration")

        if new_keys.shape[0] != self._batch_size:
            raise ValueError(f"Batch size mismatch: expected {self._batch_size}, got {new_keys.shape[0]}")

        # Encode new tokens only
        new_token_count = new_keys.shape[2]
        new_encoded_keys = tuple(
            x for x in encode_tensor(
                new_keys.reshape(-1, new_keys.shape[-1]),
                self._shared_calibration.keys
            )
        )
        new_encoded_values = tuple(
            x for x in encode_tensor(
                new_values.reshape(-1, new_values.shape[-1]),
                self._shared_calibration.values
            )
        )

        # Add as new chunk (no merging needed)
        self._chunks.append((new_encoded_keys, new_encoded_values, new_token_count))

        # Update state
        self._encoded_tokens += new_token_count

        # Update metadata
        self._meta["encoded_tokens"] = self._encoded_tokens
        self._meta["num_chunks"] = len(self._chunks)

    def decode(self) -> Tuple[mx.array, mx.array]:
        """Decode compressed cache to full K/V tensors.

        Decodes all chunks and concatenates them along the token dimension.

        Returns:
            (keys, values): Reconstructed tensors [batch, heads, tokens, dim]
        """
        if not self._chunks:
            raise ValueError("No encoded data to decode")

        if self._shared_calibration is None:
            raise ValueError("Cannot decode without calibration")

        # Decode all chunks
        decoded_keys_list = []
        decoded_values_list = []

        for encoded_keys, encoded_values, num_tokens in self._chunks:
            # Decode chunk: returns [batch*heads*tokens, dim]
            keys_flat = decode_tensor(encoded_keys, self._shared_calibration.keys)
            values_flat = decode_tensor(encoded_values, self._shared_calibration.values)

            # Reshape to [batch, heads, tokens, dim]
            # Note: keys_flat shape is [batch*heads*num_tokens, dim]
            keys_reshaped = keys_flat.reshape(self._batch_size, self._num_heads, num_tokens, self._head_dim)
            values_reshaped = values_flat.reshape(self._batch_size, self._num_heads, num_tokens, self._head_dim)

            decoded_keys_list.append(keys_reshaped)
            decoded_values_list.append(values_reshaped)

        # Concatenate along token dimension (axis=2)
        keys = mx.array(np.concatenate(decoded_keys_list, axis=2))  # [batch, heads, total_tokens, dim]
        values = mx.array(np.concatenate(decoded_values_list, axis=2))

        return keys, values

    @property
    def encoded_tokens(self) -> int:
        """Number of tokens encoded in the cache."""
        return self._encoded_tokens

    @property
    def state(self):
        """Compressed state for serialization."""
        return self._chunks

    @state.setter
    def state(self, v):
        self._chunks = v

    @property
    def meta_state(self):
        """Metadata for serialization."""
        return json.dumps(self._meta)

    @meta_state.setter
    def meta_state(self, v):
        self._meta = json.loads(v) if isinstance(v, str) else dict(v)

    def empty(self) -> bool:
        """Check if cache is empty."""
        return self._encoded_tokens == 0

    def __repr__(self):
        return (
            f"IncrementalKVTCCache("
            f"tokens={self._encoded_tokens}, "
            f"chunks={len(self._chunks)}, "
            f"shape=[{self._batch_size}, {self._num_heads}, {self._encoded_tokens}, {self._head_dim}])"
        )
