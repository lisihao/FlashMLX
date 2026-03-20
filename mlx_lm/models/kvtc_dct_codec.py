"""KVTC DCT Codec - No-calibration fast compression using DCT transform.

This module implements KVTC P2: DCT Transform, which replaces PCA with DCT
to eliminate the calibration phase and enable instant compression.

Key features:
- No calibration required: DCT matrix is fixed
- Fast initialization: no need to collect samples
- Online compression: can compress immediately
- Compatible with existing quantization: reuses quantize_groups logic

Design:
- DCT (Discrete Cosine Transform) is a fixed orthogonal transform
- Like PCA, it converts spatial domain to frequency domain
- Low-frequency coefficients contain most information
- High-frequency coefficients can be quantized aggressively
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np
from scipy.fftpack import dct, idct

from .kvtc_codec import (
    KVTCCodecConfig,
    dequantize_groups,
    plan_bit_allocation,
    quantize_groups,
)


@dataclass
class KVTCDCTTransformPlan:
    """DCT-based transform plan (no calibration needed)."""

    dim: int  # Dimension of input vectors
    block_meta: np.ndarray  # Bit allocation plan
    config: KVTCCodecConfig

    @property
    def state(self):
        return (self.dim, self.block_meta)

    def fingerprint(self) -> str:
        """Generate a unique fingerprint for this plan."""
        import hashlib
        import json

        data = {
            "dim": self.dim,
            "block_meta": self.block_meta.tolist(),
            "config": {
                "rank": self.config.rank,
                "bits": self.config.bits,
                "group_size": self.config.group_size,
            },
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def encode(self, x):
        """Encode a 2D matrix with DCT transform."""
        x = _to_numpy(x).astype(np.float32, copy=False)

        # Apply DCT-II along feature dimension (axis=1)
        # DCT-II is the most common DCT variant (used in JPEG)
        coeffs = dct(x, type=2, axis=1, norm='ortho').astype(np.float32)

        # Quantize each block according to bit allocation
        payloads = []
        shifts = []
        scales = []
        q_shapes = []
        offset = 0

        for start, end, bits_i in self.block_meta:
            if bits_i == 0:
                # Zero-bit block: prune (set to zero), don't add to lists
                continue
            else:
                # Quantize block (returns 4 values: payload, shift, scale, q_shape)
                block = coeffs[:, start:end]
                payload, shift, scale, q_shape = quantize_groups(block, bits_i, self.config.group_size)
                payloads.append(payload)
                shifts.append(shift)
                scales.append(scale)
                q_shapes.append(q_shape)

        return tuple(payloads), tuple(shifts), tuple(scales), tuple(q_shapes), x.shape

    def decode(self, encoded):
        """Decode a tensor encoded with DCT transform."""
        if len(encoded) == 5:
            payloads, shifts, scales, q_shapes, orig_shape = encoded
        elif len(encoded) == 4:
            payloads, shifts, scales, orig_shape = encoded
            q_shapes = None
        else:
            raise ValueError(f"Unexpected encoding format: {len(encoded)} elements")

        # Dequantize each block
        coeffs = np.zeros(orig_shape, dtype=np.float32)
        payload_idx = 0  # Separate index for payloads/shifts/scales
        for start, end, bits_i in self.block_meta:
            if bits_i == 0:
                # Zero-bit block: coeffs already zero, don't advance payload_idx
                continue
            else:
                # Dequantize block
                block_shape = q_shapes[payload_idx] if q_shapes else (orig_shape[0], end - start)
                # Note: dequantize_groups signature is (payload, q_shape, shifts, scales, bits, group_size)
                block = dequantize_groups(
                    payloads[payload_idx],
                    block_shape,          # q_shape is 2nd parameter
                    shifts[payload_idx],
                    scales[payload_idx],
                    bits_i,
                    self.config.group_size,
                )
                coeffs[:, start:end] = block
                payload_idx += 1  # Advance index only for non-zero blocks

        # Apply inverse DCT-II along feature dimension (axis=1)
        x = idct(coeffs, type=2, axis=1, norm='ortho').astype(np.float32)

        return x


@dataclass
class KVTCDCTSharedCalibration:
    """Shared DCT calibration for keys and values (no actual calibration needed)."""

    keys: KVTCDCTTransformPlan
    values: KVTCDCTTransformPlan

    def fingerprint(self) -> str:
        """Generate a unique fingerprint for this calibration."""
        return f"{self.keys.fingerprint()}_{self.values.fingerprint()}"


def _to_numpy(x):
    """Convert MLX array or numpy array to numpy array."""
    if isinstance(x, mx.array):
        return np.asarray(x, dtype=np.float32)
    return x


def _subsample_rows(x, limit: int, seed: int):
    """Subsample rows if needed (for bit allocation estimation)."""
    if x.shape[0] <= limit:
        return x
    rng = np.random.default_rng(seed)
    indices = rng.choice(x.shape[0], size=limit, replace=False)
    return x[indices]


def create_fixed_bit_allocation(dim: int, config: KVTCCodecConfig) -> np.ndarray:
    """Create a fixed bit allocation plan based on frequency bands.

    DCT concentrates energy in low frequencies, so we allocate:
    - Low frequencies (first 40%): full bits
    - Mid frequencies (40%-80%): bits-1
    - High frequencies (last 20%): bits-2 (or 0 if bits < 3)

    This avoids needing sample data entirely.
    """
    bits = config.bits
    group_size = config.group_size

    # Effective rank (number of coefficients to keep)
    if config.rank is not None:
        rank = min(config.rank, dim)
    else:
        # Default: keep 50% of coefficients
        rank = max(1, dim // 2)

    # Divide into frequency bands
    low_freq_end = int(rank * 0.4)
    mid_freq_end = int(rank * 0.8)

    blocks = []

    # Low frequency band: full bits
    if low_freq_end > 0:
        for start in range(0, low_freq_end, group_size):
            end = min(start + group_size, low_freq_end)
            blocks.append((start, end, bits))

    # Mid frequency band: bits-1
    if mid_freq_end > low_freq_end:
        mid_bits = max(2, bits - 1)
        for start in range(low_freq_end, mid_freq_end, group_size):
            end = min(start + group_size, mid_freq_end)
            blocks.append((start, end, mid_bits))

    # High frequency band: bits-2 or prune
    if rank > mid_freq_end:
        high_bits = 0 if bits < 3 else (bits - 2)
        for start in range(mid_freq_end, rank, group_size):
            end = min(start + group_size, rank)
            blocks.append((start, end, high_bits))

    # Prune remaining coefficients
    if rank < dim:
        blocks.append((rank, dim, 0))

    return np.array(blocks, dtype=np.int32)


def fit_dct_transform_plan(
    matrices: Sequence[np.ndarray],
    config: KVTCCodecConfig,
    use_fixed_allocation: bool = True
) -> KVTCDCTTransformPlan:
    """Fit a DCT transform plan.

    Args:
        matrices: Input matrices (only used to get dimension if use_fixed_allocation=True)
        config: Codec configuration
        use_fixed_allocation: If True, use fixed bit allocation (no calibration needed).
                              If False, estimate allocation from sample data.

    Returns:
        DCT transform plan
    """
    # Get dimension
    mat0 = _to_numpy(matrices[0]).astype(np.float32, copy=False)
    if mat0.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {mat0.shape}")
    dim = mat0.shape[1]

    if use_fixed_allocation:
        # FAST PATH: Use fixed bit allocation (no sample data needed)
        block_meta = create_fixed_bit_allocation(dim, config)
    else:
        # SLOW PATH: Estimate bit allocation from sample data
        combined = []
        for mat in matrices:
            mat = _to_numpy(mat).astype(np.float32, copy=False)
            if mat.ndim != 2:
                raise ValueError(f"Expected 2D matrix, got shape {mat.shape}")
            combined.append(_subsample_rows(mat, config.sample_limit, config.seed))

        combined = np.concatenate(combined, axis=0)

        # Apply DCT to sample data
        coeffs = dct(combined, type=2, axis=1, norm='ortho').astype(np.float32)

        # Plan bit allocation based on statistics
        block_meta = plan_bit_allocation(coeffs, config)

    return KVTCDCTTransformPlan(dim=dim, block_meta=block_meta, config=config)


def fit_dct_shared_calibration(
    key_matrices: Sequence[np.ndarray],
    value_matrices: Sequence[np.ndarray],
    config: KVTCCodecConfig,
    use_fixed_allocation: bool = True
) -> KVTCDCTSharedCalibration:
    """Fit a shared DCT calibration for keys and values.

    Args:
        key_matrices: Key matrices (only used to get dimension)
        value_matrices: Value matrices (only used to get dimension)
        config: Codec configuration
        use_fixed_allocation: If True, use fixed bit allocation (instant, no calibration).
                              If False, estimate allocation from sample data (slower).

    Returns:
        Shared calibration for keys and values

    Notes:
        With use_fixed_allocation=True, this is essentially instant (no calibration).
        With use_fixed_allocation=False, this estimates bit allocation from sample data.
    """
    key_plan = fit_dct_transform_plan(key_matrices, config, use_fixed_allocation)
    value_plan = fit_dct_transform_plan(value_matrices, config, use_fixed_allocation)
    return KVTCDCTSharedCalibration(keys=key_plan, values=value_plan)


def encode_tensor_dct(x, plan: KVTCDCTTransformPlan):
    """Encode a 2D matrix with DCT transform."""
    return plan.encode(x)


def decode_tensor_dct(encoded, plan: KVTCDCTTransformPlan) -> np.ndarray:
    """Decode a tensor encoded with DCT transform."""
    return plan.decode(encoded)
