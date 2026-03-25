"""KVTC PCA Codec - Data-driven dimensionality reduction.

This module implements PCA-based compression for KV Cache:
- Learns optimal projection basis from calibration data
- Preserves maximum variance (energy)
- Should be more adaptive than fixed DCT basis

Key differences from DCT:
- DCT: Fixed frequency-domain basis (assumes smooth signals)
- PCA: Data-driven basis (learns from actual KV cache patterns)

Expected advantages:
- Better compression for non-smooth KV cache patterns
- Adaptive to model-specific features
- Potentially better quality at same compression ratio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .kvtc_codec import (
    KVTCCodecConfig,
    _to_numpy,
    dequantize_groups,
    fit_pca_basis,
    project,
    quantize_groups,
    reconstruct,
)


@dataclass
class KVTCPCACalibration:
    """PCA calibration for KV cache compression.

    Stores mean and basis for both keys and values.
    """

    keys: KVTCPCAPlan
    values: KVTCPCAPlan

    def fingerprint(self) -> str:
        return f"{self.keys.fingerprint()}_{self.values.fingerprint()}"


@dataclass
class KVTCPCAPlan:
    """PCA compression plan for a single matrix type (keys or values).

    Attributes:
        mean: Mean vector for centering (shape: [dim])
        basis: PCA basis vectors (shape: [dim, rank])
        config: Codec configuration
    """

    mean: np.ndarray
    basis: np.ndarray
    config: KVTCCodecConfig

    def fingerprint(self) -> str:
        import hashlib
        import json

        data = {
            "mean_norm": float(np.linalg.norm(self.mean)),
            "basis_shape": self.basis.shape,
            "bits": self.config.bits,
            "group_size": self.config.group_size,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def encode(self, x):
        """Encode a 2D matrix using PCA + quantization.

        Args:
            x: Input matrix (shape: [rows, dim])

        Returns:
            Tuple of (payload, shifts, scales, q_shape, mean, basis, orig_shape)
        """
        x = _to_numpy(x).astype(np.float32, copy=False)

        # Project to PCA space
        coeffs = project(x, self.mean, self.basis)

        # Quantize coefficients
        payload, shifts, scales, q_shape = quantize_groups(
            coeffs, self.config.bits, self.config.group_size
        )

        # Return encoded data (store mean and basis for decoding)
        return payload, shifts, scales, q_shape, self.mean, self.basis, x.shape

    def decode(self, encoded):
        """Decode from PCA encoding.

        Args:
            encoded: Tuple of (payload, shifts, scales, q_shape, mean, basis, orig_shape)

        Returns:
            Reconstructed matrix
        """
        payload, shifts, scales, q_shape, mean, basis, orig_shape = encoded

        # Dequantize coefficients
        coeffs = dequantize_groups(
            payload, q_shape, shifts, scales, self.config.bits, self.config.group_size
        )

        # Reconstruct from PCA space
        x = reconstruct(coeffs, mean, basis)

        return x


def fit_pca_plan(
    matrices: Sequence[np.ndarray],
    config: KVTCCodecConfig,
) -> KVTCPCAPlan:
    """Fit a PCA plan for a list of matrices.

    Args:
        matrices: List of 2D matrices to fit PCA on
        config: Codec configuration

    Returns:
        PCA plan with learned mean and basis
    """
    # Concatenate all matrices for fitting
    x = _to_numpy(matrices[0]).astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {x.shape}")

    # Fit PCA basis
    mean, basis = fit_pca_basis(x, config)

    return KVTCPCAPlan(mean=mean, basis=basis, config=config)


def fit_pca_calibration(
    key_matrices: Sequence[np.ndarray],
    value_matrices: Sequence[np.ndarray],
    config: KVTCCodecConfig,
) -> KVTCPCACalibration:
    """Fit PCA calibration for keys and values.

    Args:
        key_matrices: List of key matrices
        value_matrices: List of value matrices
        config: Codec configuration

    Returns:
        PCA calibration with plans for both keys and values
    """
    key_plan = fit_pca_plan(key_matrices, config)
    value_plan = fit_pca_plan(value_matrices, config)

    return KVTCPCACalibration(keys=key_plan, values=value_plan)


def encode_tensor_pca(x, plan: KVTCPCAPlan):
    """Encode a 2D matrix using PCA plan."""
    return plan.encode(x)


def decode_tensor_pca(encoded, plan: KVTCPCAPlan) -> np.ndarray:
    """Decode a tensor encoded with PCA plan."""
    return plan.decode(encoded)
