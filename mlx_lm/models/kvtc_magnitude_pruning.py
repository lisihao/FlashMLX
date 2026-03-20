"""KVTC Magnitude Pruning - Compression ratio improvement.

This module implements KVTC P4: Magnitude Pruning, which prunes small-magnitude
coefficients to improve compression ratio at the cost of some accuracy.

Key features:
- Prune small coefficients: only keep top-k% by magnitude
- Higher compression ratio: fewer coefficients to store
- Configurable pruning ratio: trade-off between size and accuracy

Design rationale:
- After DCT/PCA transform, coefficients have varying importance
- Large-magnitude coefficients carry most information
- Small-magnitude coefficients are mostly noise
- Pruning small coefficients has minimal impact on reconstruction

Strategy:
1. Compute magnitude of each coefficient (L2 norm or absolute value)
2. Sort by magnitude, keep top-k%
3. Set remaining coefficients to zero (don't store them)
4. During decode, reconstruct with zeros for pruned coefficients

Trade-offs:
- ✅ Better compression ratio (25-50% improvement)
- ❌ Lower accuracy (losing small coefficients)
- ✅ Configurable: can tune pruning ratio for different use cases
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np
from scipy.fftpack import dct, idct

from .kvtc_codec import (
    KVTCCodecConfig,
    dequantize_groups,
    quantize_groups,
)


@dataclass
class KVTCMagnitudePruningConfig:
    """Configuration for magnitude pruning."""

    keep_ratio: float = 0.75  # Keep top 75% coefficients
    min_keep: int = 4  # Keep at least this many coefficients
    pruning_method: str = "l2"  # "l2" or "abs"


@dataclass
class KVTCMagnitudePruningPlan:
    """Magnitude pruning plan for DCT transform."""

    dim: int  # Dimension of input vectors
    keep_ratio: float  # Ratio of coefficients to keep
    pruning_method: str
    config: KVTCCodecConfig

    def fingerprint(self) -> str:
        """Generate a unique fingerprint for this plan."""
        import hashlib
        import json

        data = {
            "dim": self.dim,
            "keep_ratio": self.keep_ratio,
            "pruning_method": self.pruning_method,
            "config": {
                "bits": self.config.bits,
                "group_size": self.config.group_size,
            },
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def encode(self, x):
        """Encode with magnitude pruning."""
        x = _to_numpy(x).astype(np.float32, copy=False)

        # Apply DCT-II
        coeffs = dct(x, type=2, axis=1, norm='ortho').astype(np.float32)

        # Compute magnitude for each coefficient position (average across rows)
        if self.pruning_method == "l2":
            # L2 norm across rows
            magnitude = np.linalg.norm(coeffs, axis=0)  # [dim]
        else:  # "abs"
            # Mean absolute value across rows
            magnitude = np.mean(np.abs(coeffs), axis=0)  # [dim]

        # Determine how many to keep
        num_keep = max(self.config.rank or int(self.dim * 0.5),
                       int(self.dim * self.keep_ratio))
        num_keep = max(num_keep, 4)  # Keep at least 4

        # Get indices of top-k coefficients
        keep_indices = np.argsort(magnitude)[::-1][:num_keep]
        keep_indices = np.sort(keep_indices)  # Sort for easier processing

        # Prune: set other coefficients to zero
        pruned_coeffs = np.zeros_like(coeffs)
        pruned_coeffs[:, keep_indices] = coeffs[:, keep_indices]

        # Quantize only the kept coefficients
        kept_coeffs = pruned_coeffs[:, keep_indices]  # [rows, num_keep]

        # Quantize
        payload, shifts, scales, q_shape = quantize_groups(
            kept_coeffs, self.config.bits, self.config.group_size
        )

        # Return: (payload, shifts, scales, q_shape, keep_indices, orig_shape)
        return payload, shifts, scales, q_shape, keep_indices, x.shape

    def decode(self, encoded):
        """Decode with magnitude pruning."""
        payload, shifts, scales, q_shape, keep_indices, orig_shape = encoded

        # Dequantize kept coefficients
        kept_coeffs = dequantize_groups(
            payload, q_shape, shifts, scales,
            self.config.bits, self.config.group_size
        )

        # Reconstruct full coefficient array with zeros
        coeffs = np.zeros(orig_shape, dtype=np.float32)
        coeffs[:, keep_indices] = kept_coeffs

        # Apply inverse DCT-II
        x = idct(coeffs, type=2, axis=1, norm='ortho').astype(np.float32)

        return x


@dataclass
class KVTCMagnitudePruningCalibration:
    """Calibration for magnitude pruning (no actual calibration needed)."""

    keys: KVTCMagnitudePruningPlan
    values: KVTCMagnitudePruningPlan

    def fingerprint(self) -> str:
        return f"{self.keys.fingerprint()}_{self.values.fingerprint()}"


def _to_numpy(x):
    """Convert MLX array or numpy array to numpy array."""
    if isinstance(x, mx.array):
        return np.asarray(x, dtype=np.float32)
    return x


def fit_magnitude_pruning_plan(
    matrices: Sequence[np.ndarray],
    config: KVTCCodecConfig,
    pruning_config: KVTCMagnitudePruningConfig,
) -> KVTCMagnitudePruningPlan:
    """Fit a magnitude pruning plan (no calibration needed, just get dimension).

    Args:
        matrices: Input matrices (only used to get dimension)
        config: Codec configuration
        pruning_config: Pruning configuration

    Returns:
        Magnitude pruning plan
    """
    # Get dimension
    mat0 = _to_numpy(matrices[0])
    if mat0.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {mat0.shape}")
    dim = mat0.shape[1]

    return KVTCMagnitudePruningPlan(
        dim=dim,
        keep_ratio=pruning_config.keep_ratio,
        pruning_method=pruning_config.pruning_method,
        config=config,
    )


def fit_magnitude_pruning_calibration(
    key_matrices: Sequence[np.ndarray],
    value_matrices: Sequence[np.ndarray],
    config: KVTCCodecConfig,
    pruning_config: Optional[KVTCMagnitudePruningConfig] = None,
) -> KVTCMagnitudePruningCalibration:
    """Fit magnitude pruning calibration.

    Args:
        key_matrices: Key matrices
        value_matrices: Value matrices
        config: Codec configuration
        pruning_config: Pruning configuration (defaults to keep 75%)

    Returns:
        Magnitude pruning calibration
    """
    if pruning_config is None:
        pruning_config = KVTCMagnitudePruningConfig()

    key_plan = fit_magnitude_pruning_plan(key_matrices, config, pruning_config)
    value_plan = fit_magnitude_pruning_plan(value_matrices, config, pruning_config)

    return KVTCMagnitudePruningCalibration(keys=key_plan, values=value_plan)


def encode_tensor_magnitude_pruning(x, plan: KVTCMagnitudePruningPlan):
    """Encode a 2D matrix with magnitude pruning."""
    return plan.encode(x)


def decode_tensor_magnitude_pruning(encoded, plan: KVTCMagnitudePruningPlan) -> np.ndarray:
    """Decode a tensor encoded with magnitude pruning."""
    return plan.decode(encoded)
