"""KVTC Per-Head Codec - Precision improvement through per-head calibration.

This module implements KVTC P3: Per-Head Calibration, which calibrates each
attention head independently to improve compression accuracy.

Key features:
- Per-head PCA basis: each head learns its own transform
- Better precision: heads focus on different features (position, semantics, etc.)
- Compatible with existing KVTC: can fallback to shared calibration

Design rationale:
- Different attention heads specialize in different aspects:
  * Some heads focus on local patterns
  * Some heads focus on global semantics
  * Some heads focus on positional information
- Shared calibration averages across all heads, losing specialization
- Per-head calibration preserves each head's unique characteristics

Trade-offs:
- ✅ Better accuracy (each head optimized independently)
- ❌ More storage (N calibrations instead of 1)
- ❌ Longer calibration time (N times)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from .kvtc_codec import (
    KVTCCodecConfig,
    KVTCTransformPlan,
    fit_transform_plan,
    encode_tensor,
    decode_tensor,
)


@dataclass
class KVTCPerHeadCalibration:
    """Per-head calibration for keys and values.

    Instead of one shared calibration for all heads, this stores
    a separate calibration for each head.
    """

    key_plans: List[KVTCTransformPlan]  # One plan per head
    value_plans: List[KVTCTransformPlan]  # One plan per head

    @property
    def num_heads(self) -> int:
        return len(self.key_plans)

    def fingerprint(self) -> str:
        """Generate a unique fingerprint for this calibration."""
        import hashlib
        import json

        key_fps = [plan.fingerprint() for plan in self.key_plans]
        value_fps = [plan.fingerprint() for plan in self.value_plans]
        data = {"keys": key_fps, "values": value_fps}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def state(self):
        return {
            "keys": [plan.state for plan in self.key_plans],
            "values": [plan.state for plan in self.value_plans],
        }

    @property
    def meta_state(self):
        return {
            "keys": [plan.meta_state for plan in self.key_plans],
            "values": [plan.meta_state for plan in self.value_plans],
        }


def _to_numpy(x):
    """Convert MLX array or numpy array to numpy array."""
    if isinstance(x, mx.array):
        return np.asarray(x, dtype=np.float32)
    return x


def fit_perhead_calibration(
    key_matrices: Sequence[np.ndarray],  # [batch, heads, tokens, dim]
    value_matrices: Sequence[np.ndarray],
    config: KVTCCodecConfig,
) -> KVTCPerHeadCalibration:
    """Fit per-head calibration for keys and values.

    Args:
        key_matrices: List of key tensors, each [batch, heads, tokens, dim]
        value_matrices: List of value tensors, each [batch, heads, tokens, dim]
        config: Codec configuration

    Returns:
        Per-head calibration with separate plans for each head

    Note:
        This is slower than shared calibration (N times) but provides
        better accuracy by preserving each head's specialization.
    """
    # Get number of heads from first matrix
    first_key = _to_numpy(key_matrices[0])
    if first_key.ndim != 4:
        raise ValueError(f"Expected 4D tensor [batch, heads, tokens, dim], got {first_key.shape}")

    batch, num_heads, tokens, dim = first_key.shape

    # Calibrate each head independently
    key_plans = []
    value_plans = []

    for head_idx in range(num_heads):
        # Extract this head's data from all matrices
        key_head_matrices = []
        value_head_matrices = []

        for key_mat, value_mat in zip(key_matrices, value_matrices):
            key_mat = _to_numpy(key_mat)
            value_mat = _to_numpy(value_mat)

            # Extract head: [batch, heads, tokens, dim] -> [batch, tokens, dim] -> [batch*tokens, dim]
            key_head = key_mat[:, head_idx, :, :].reshape(-1, dim)
            value_head = value_mat[:, head_idx, :, :].reshape(-1, dim)

            key_head_matrices.append(key_head)
            value_head_matrices.append(value_head)

        # Fit calibration for this head
        key_plan = fit_transform_plan(key_head_matrices, config)
        value_plan = fit_transform_plan(value_head_matrices, config)

        key_plans.append(key_plan)
        value_plans.append(value_plan)

    return KVTCPerHeadCalibration(key_plans=key_plans, value_plans=value_plans)


def encode_perhead(
    keys: np.ndarray,  # [batch, heads, tokens, dim]
    values: np.ndarray,
    calibration: KVTCPerHeadCalibration,
) -> Tuple[List, List]:
    """Encode keys and values using per-head calibration.

    Args:
        keys: Key tensor [batch, heads, tokens, dim]
        values: Value tensor [batch, heads, tokens, dim]
        calibration: Per-head calibration

    Returns:
        (encoded_keys, encoded_values): Lists of encoded data, one per head
    """
    keys = _to_numpy(keys)
    values = _to_numpy(values)

    if keys.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {keys.shape}")

    batch, num_heads, tokens, dim = keys.shape

    if num_heads != calibration.num_heads:
        raise ValueError(
            f"Number of heads mismatch: data has {num_heads}, "
            f"calibration has {calibration.num_heads}"
        )

    # Encode each head independently
    encoded_keys = []
    encoded_values = []

    for head_idx in range(num_heads):
        # Extract head: [batch, heads, tokens, dim] -> [batch*tokens, dim]
        key_head = keys[:, head_idx, :, :].reshape(-1, dim)
        value_head = values[:, head_idx, :, :].reshape(-1, dim)

        # Encode with this head's calibration
        enc_key = encode_tensor(key_head, calibration.key_plans[head_idx])
        enc_value = encode_tensor(value_head, calibration.value_plans[head_idx])

        encoded_keys.append(enc_key)
        encoded_values.append(enc_value)

    return encoded_keys, encoded_values


def decode_perhead(
    encoded_keys: List,
    encoded_values: List,
    calibration: KVTCPerHeadCalibration,
    shape: Tuple[int, int, int, int],  # [batch, heads, tokens, dim]
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode keys and values using per-head calibration.

    Args:
        encoded_keys: List of encoded key data, one per head
        encoded_values: List of encoded value data, one per head
        calibration: Per-head calibration
        shape: Original shape [batch, heads, tokens, dim]

    Returns:
        (keys, values): Reconstructed tensors [batch, heads, tokens, dim]
    """
    batch, num_heads, tokens, dim = shape

    if num_heads != calibration.num_heads:
        raise ValueError(
            f"Number of heads mismatch: shape has {num_heads}, "
            f"calibration has {calibration.num_heads}"
        )

    # Decode each head independently
    keys = np.zeros((batch, num_heads, tokens, dim), dtype=np.float32)
    values = np.zeros((batch, num_heads, tokens, dim), dtype=np.float32)

    for head_idx in range(num_heads):
        # Decode with this head's calibration
        key_head = decode_tensor(encoded_keys[head_idx], calibration.key_plans[head_idx])
        value_head = decode_tensor(encoded_values[head_idx], calibration.value_plans[head_idx])

        # Reshape: [batch*tokens, dim] -> [batch, tokens, dim]
        key_head = key_head.reshape(batch, tokens, dim)
        value_head = value_head.reshape(batch, tokens, dim)

        # Insert into result
        keys[:, head_idx, :, :] = key_head
        values[:, head_idx, :, :] = value_head

    return keys, values
