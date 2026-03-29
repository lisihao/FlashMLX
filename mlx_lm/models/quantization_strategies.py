"""
Quantization Strategies for KV Cache Warm Layer

Provides pluggable quantization engines for the Warm layer in TripleLayerKVCache.
Supports multiple quantization algorithms:
- Q4_0: 4-bit symmetric quantization (current default)
- PolarQuant: Google's rotation + Lloyd-Max quantization (2-4 bit, data-oblivious)
- GPTQ: Hessian-based quantization (TODO)
- AWQ: Activation-aware weight quantization (TODO)

Example
-------
>>> # Use default Q4_0
>>> cache = TripleLayerKVCache(memory_budget_mb=10.0)
>>>
>>> # Use PolarQuant 4-bit
>>> cache = TripleLayerKVCache(
...     memory_budget_mb=10.0,
...     warm_quantizer=PolarQuantizer(bits=4)
... )
>>>
>>> # Use PolarQuant 3-bit (more aggressive)
>>> cache = TripleLayerKVCache(
...     memory_budget_mb=10.0,
...     warm_quantizer=PolarQuantizer(bits=3)
... )
"""

import math
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import mlx.core as mx


class QuantizationStrategy(ABC):
    """
    Abstract base class for KV cache quantization strategies.

    All quantization implementations must inherit from this class and implement
    the required methods.
    """

    @abstractmethod
    def quantize(
        self,
        keys: mx.array,
        values: mx.array
    ) -> Tuple[mx.array, mx.array, Dict[str, Any]]:
        """
        Quantize KV cache tensors.

        Parameters
        ----------
        keys : mx.array
            Keys tensor, shape (B, n_heads, seq_len, head_dim)
        values : mx.array
            Values tensor, shape (B, n_heads, seq_len, head_dim)

        Returns
        -------
        quant_keys : mx.array
            Quantized keys
        quant_values : mx.array
            Quantized values
        metadata : dict
            Quantization metadata (scales, codebooks, etc.)
            Required for dequantization
        """
        pass

    @abstractmethod
    def dequantize(
        self,
        quant_keys: mx.array,
        quant_values: mx.array,
        metadata: Dict[str, Any]
    ) -> Tuple[mx.array, mx.array]:
        """
        Dequantize KV cache tensors.

        Parameters
        ----------
        quant_keys : mx.array
            Quantized keys
        quant_values : mx.array
            Quantized values
        metadata : dict
            Quantization metadata from quantize()

        Returns
        -------
        keys : mx.array
            Dequantized keys (fp16/fp32)
        values : mx.array
            Dequantized values (fp16/fp32)
        """
        pass

    @property
    def requires_chunk_eviction(self) -> bool:
        """
        Whether this quantizer requires chunk-aware eviction during warm overflow.

        When True, warm overflow pops WHOLE quantized chunks instead of
        dequant→split→re-quantize. This avoids re-quantization error
        amplification for quantizers like TurboQuant where re-quantization
        causes geometric quality degradation.

        Default: False (re-quantize path is fine for Q4_0/PolarQuant).
        """
        return False

    def requantize(
        self,
        keys: mx.array,
        values: mx.array
    ) -> Tuple[mx.array, mx.array, Dict[str, Any]]:
        """
        Re-quantize already-dequantized data (e.g. during warm overflow).

        Default: same as quantize(). Override for quantizers where
        re-quantization causes error amplification (e.g. TurboQuant).
        """
        return self.quantize(keys, values)

    @abstractmethod
    def get_compression_ratio(self) -> float:
        """
        Get compression ratio.

        Returns
        -------
        ratio : float
            Compression ratio (e.g., 2.0 = 2x compression)
        """
        pass

    @abstractmethod
    def estimate_memory(self, num_tokens: int, head_dim: int, num_heads: int) -> int:
        """
        Estimate memory usage in bytes.

        Parameters
        ----------
        num_tokens : int
            Number of tokens
        head_dim : int
            Dimension per head
        num_heads : int
            Number of attention heads

        Returns
        -------
        bytes : int
            Estimated memory usage in bytes
        """
        pass


# ====================================================================
# Q4_0 Quantizer (Current Default)
# ====================================================================

class Q4_0Quantizer(QuantizationStrategy):
    """
    4-bit symmetric quantization.

    Features:
    - Group-based quantization (group_size=32)
    - Symmetric quantization: [-max, +max] → [-8, +7]
    - Per-group scaling factors (fp32)

    Compression: fp16 → int4 = 2.0x

    Parameters
    ----------
    group_size : int
        Quantization group size (default: 32)

    Example
    -------
    >>> quantizer = Q4_0Quantizer(group_size=32)
    >>> quant_k, quant_v, meta = quantizer.quantize(keys, values)
    >>> recovered_k, recovered_v = quantizer.dequantize(quant_k, quant_v, meta)
    """

    def __init__(self, group_size: int = 32):
        self.group_size = group_size

    def quantize(
        self,
        keys: mx.array,
        values: mx.array
    ) -> Tuple[mx.array, mx.array, Dict[str, Any]]:
        """Quantize to 4-bit with per-group scaling."""
        B, n_heads, seq_len, head_dim = keys.shape

        # Quantize keys
        quant_keys, scales_k = self._quantize_symmetric(keys)

        # Quantize values
        quant_values, scales_v = self._quantize_symmetric(values)

        # Metadata for dequantization
        metadata = {
            'scales_k': scales_k,
            'scales_v': scales_v,
            'quant_bits': 4,
            'group_size': self.group_size,
            'seq_len': seq_len  # ← Store actual sequence length to avoid rounding errors
        }

        return quant_keys, quant_values, metadata

    def dequantize(
        self,
        quant_keys: mx.array,
        quant_values: mx.array,
        metadata: Dict[str, Any]
    ) -> Tuple[mx.array, mx.array]:
        """Dequantize from 4-bit."""
        # Dequantize keys
        keys = self._dequantize_symmetric(quant_keys, metadata['scales_k'])

        # Dequantize values
        values = self._dequantize_symmetric(quant_values, metadata['scales_v'])

        return keys, values

    def get_compression_ratio(self) -> float:
        """fp16 → int4 = 2.0x compression."""
        return 2.0

    def estimate_memory(self, num_tokens: int, head_dim: int, num_heads: int) -> int:
        """
        Estimate memory usage.

        Memory breakdown:
        - Quantized data: int4 (0.5 bytes/element)
        - Scales: fp32 (4 bytes per group)
        """
        # Quantized data: int4 = 0.5 bytes/element
        quant_size = num_tokens * head_dim * num_heads * 0.5 * 2  # keys + values

        # Scales: fp32, one per group
        num_groups = (num_tokens * head_dim * num_heads + self.group_size - 1) // self.group_size
        scales_size = num_groups * 4 * 2  # keys + values

        return int(quant_size + scales_size)

    def _quantize_symmetric(self, tensor: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Symmetric quantization: [-max, +max] → [-8, +7].

        Returns
        -------
        quant_tensor : mx.array
            Quantized tensor (int8, will be packed to int4 later)
        scales : mx.array
            Per-group scaling factors (fp32)
        """
        B, n_heads, seq_len, head_dim = tensor.shape

        # Reshape to groups
        total_elements = B * n_heads * seq_len * head_dim
        num_groups = (total_elements + self.group_size - 1) // self.group_size

        # Pad if needed
        padded_size = num_groups * self.group_size
        if total_elements < padded_size:
            flat = mx.reshape(tensor, [-1])
            padding = mx.zeros(padded_size - total_elements, dtype=tensor.dtype)
            flat = mx.concatenate([flat, padding])
        else:
            flat = mx.reshape(tensor, [-1])

        # Reshape to [num_groups, group_size]
        grouped = mx.reshape(flat, [num_groups, self.group_size])

        # Compute per-group max (for symmetric quantization)
        max_vals = mx.max(mx.abs(grouped), axis=1, keepdims=True)
        max_vals = mx.maximum(max_vals, 1e-8)  # Avoid division by zero

        # Quantize: [-max, +max] → [-8, +7]
        scales = max_vals / 7.0
        quant_grouped = mx.round(grouped / scales)
        quant_grouped = mx.clip(quant_grouped, -8, 7)

        # Reshape back
        quant_flat = mx.reshape(quant_grouped, [-1])
        if total_elements < padded_size:
            quant_flat = quant_flat[:total_elements]

        quant_tensor = mx.reshape(quant_flat, tensor.shape)
        scales = mx.reshape(scales, [num_groups])

        return quant_tensor.astype(mx.int8), scales

    def _dequantize_symmetric(self, quant_tensor: mx.array, scales: mx.array) -> mx.array:
        """Dequantize from int8 using per-group scales."""
        B, n_heads, seq_len, head_dim = quant_tensor.shape

        # Reshape to groups
        total_elements = B * n_heads * seq_len * head_dim
        num_groups = scales.shape[0]

        # Pad if needed
        padded_size = num_groups * self.group_size
        if total_elements < padded_size:
            flat = mx.reshape(quant_tensor, [-1])
            padding = mx.zeros(padded_size - total_elements, dtype=quant_tensor.dtype)
            flat = mx.concatenate([flat, padding])
        else:
            flat = mx.reshape(quant_tensor, [-1])

        # Reshape to [num_groups, group_size]
        grouped = mx.reshape(flat, [num_groups, self.group_size])

        # Dequantize
        scales_expanded = mx.expand_dims(scales, axis=1)  # [num_groups, 1]
        dequant_grouped = grouped.astype(mx.float32) * scales_expanded

        # Reshape back
        dequant_flat = mx.reshape(dequant_grouped, [-1])
        if total_elements < padded_size:
            dequant_flat = dequant_flat[:total_elements]

        dequant_tensor = mx.reshape(dequant_flat, quant_tensor.shape)

        return dequant_tensor


# ====================================================================
# Q8_0 Quantizer (8-bit symmetric)
# ====================================================================

class Q8_0Quantizer(QuantizationStrategy):
    """
    8-bit symmetric quantization.

    Features:
    - Group-based quantization (group_size=32)
    - Symmetric quantization: [-max, +max] -> [-128, +127]
    - Per-group scaling factors (fp32)

    Compression: fp16 -> int8 = ~1.78x (with group_size=32 scales overhead)

    Parameters
    ----------
    group_size : int
        Quantization group size (default: 32)

    Example
    -------
    >>> quantizer = Q8_0Quantizer(group_size=32)
    >>> quant_k, quant_v, meta = quantizer.quantize(keys, values)
    >>> recovered_k, recovered_v = quantizer.dequantize(quant_k, quant_v, meta)
    """

    def __init__(self, group_size: int = 32):
        self.group_size = group_size

    def quantize(
        self,
        keys: mx.array,
        values: mx.array
    ) -> Tuple[mx.array, mx.array, Dict[str, Any]]:
        """Quantize to 8-bit with per-group scaling."""
        quant_keys, scales_k = self._quantize_symmetric(keys)
        quant_values, scales_v = self._quantize_symmetric(values)

        metadata = {
            'scales_k': scales_k,
            'scales_v': scales_v,
            'quant_bits': 8,
            'group_size': self.group_size,
            'seq_len': keys.shape[2],
        }

        return quant_keys, quant_values, metadata

    def dequantize(
        self,
        quant_keys: mx.array,
        quant_values: mx.array,
        metadata: Dict[str, Any]
    ) -> Tuple[mx.array, mx.array]:
        """Dequantize from 8-bit."""
        keys = self._dequantize_symmetric(quant_keys, metadata['scales_k'])
        values = self._dequantize_symmetric(quant_values, metadata['scales_v'])
        return keys, values

    def get_compression_ratio(self) -> float:
        """fp16 -> int8 = ~1.78x compression (with group_size=32 scales)."""
        return 16.0 / (8.0 + 32.0 / self.group_size)

    def estimate_memory(self, num_tokens: int, head_dim: int, num_heads: int) -> int:
        """Estimate memory usage."""
        quant_size = num_tokens * head_dim * num_heads * 1 * 2  # int8, K+V
        num_groups = (num_tokens * head_dim * num_heads + self.group_size - 1) // self.group_size
        scales_size = num_groups * 4 * 2  # fp32 scales, K+V
        return int(quant_size + scales_size)

    def _quantize_symmetric(self, tensor: mx.array) -> Tuple[mx.array, mx.array]:
        """Symmetric quantization: [-max, +max] -> [-128, +127]."""
        B, n_heads, seq_len, head_dim = tensor.shape

        total_elements = B * n_heads * seq_len * head_dim
        num_groups = (total_elements + self.group_size - 1) // self.group_size

        padded_size = num_groups * self.group_size
        if total_elements < padded_size:
            flat = mx.reshape(tensor, [-1])
            padding = mx.zeros(padded_size - total_elements, dtype=tensor.dtype)
            flat = mx.concatenate([flat, padding])
        else:
            flat = mx.reshape(tensor, [-1])

        grouped = mx.reshape(flat, [num_groups, self.group_size])

        max_vals = mx.max(mx.abs(grouped), axis=1, keepdims=True)
        max_vals = mx.maximum(max_vals, 1e-8)

        scales = max_vals / 127.0
        quant_grouped = mx.round(grouped / scales)
        quant_grouped = mx.clip(quant_grouped, -128, 127)

        quant_flat = mx.reshape(quant_grouped, [-1])
        if total_elements < padded_size:
            quant_flat = quant_flat[:total_elements]

        quant_tensor = mx.reshape(quant_flat, tensor.shape)
        scales = mx.reshape(scales, [num_groups])

        return quant_tensor.astype(mx.int8), scales

    def _dequantize_symmetric(self, quant_tensor: mx.array, scales: mx.array) -> mx.array:
        """Dequantize from int8 using per-group scales."""
        B, n_heads, seq_len, head_dim = quant_tensor.shape

        total_elements = B * n_heads * seq_len * head_dim
        num_groups = scales.shape[0]

        padded_size = num_groups * self.group_size
        if total_elements < padded_size:
            flat = mx.reshape(quant_tensor, [-1])
            padding = mx.zeros(padded_size - total_elements, dtype=quant_tensor.dtype)
            flat = mx.concatenate([flat, padding])
        else:
            flat = mx.reshape(quant_tensor, [-1])

        grouped = mx.reshape(flat, [num_groups, self.group_size])

        scales_expanded = mx.expand_dims(scales, axis=1)
        dequant_grouped = grouped.astype(mx.float32) * scales_expanded

        dequant_flat = mx.reshape(dequant_grouped, [-1])
        if total_elements < padded_size:
            dequant_flat = dequant_flat[:total_elements]

        return mx.reshape(dequant_flat, quant_tensor.shape)


# ====================================================================
# PolarQuant (Google, AISTATS 2026 / TurboQuant ICLR 2026)
# ====================================================================

class PolarQuantizer(QuantizationStrategy):
    """
    PolarQuant: Random orthogonal rotation + Lloyd-Max optimal scalar quantization.

    From "TurboQuant: Redefining AI Efficiency with Extreme Compression"
    (Google, ICLR 2026, https://arxiv.org/abs/2504.19874).

    Features:
    - Data-oblivious: no calibration data needed
    - Random orthogonal rotation (Haar QR) maps vectors to Gaussian coordinates
    - Lloyd-Max optimal scalar quantizers for each coordinate
    - Bit-packed into uint32 for storage efficiency

    Compression ratios (fp16 baseline):
    - 4-bit: ~3.8x (cosine sim > 0.95)
    - 3-bit: ~4.6x (cosine sim > 0.90)
    - 2-bit: ~6.4x (cosine sim > 0.80)

    Parameters
    ----------
    bits : int
        Bits per coordinate (2, 3, or 4). Default: 4.

    Example
    -------
    >>> quantizer = PolarQuantizer(bits=4)
    >>> quant_k, quant_v, meta = quantizer.quantize(keys, values)
    >>> recovered_k, recovered_v = quantizer.dequantize(quant_k, quant_v, meta)
    """

    # Lloyd-Max optimal centroids for N(0,1), scaled by 1/sqrt(head_dim) at runtime
    _CENTROIDS = {
        2: [-1.5104, -0.4528,  0.4528,  1.5104],
        3: [-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519],
        4: [-2.7331, -2.0698, -1.6189, -1.2570, -0.9431, -0.6573,
            -0.3884, -0.1285,  0.1285,  0.3884,  0.6573,  0.9431,
             1.2570,  1.6189,  2.0698,  2.7331],
    }
    _BOUNDARIES = {
        2: [-5.0, -0.9816, 0.0, 0.9816, 5.0],
        3: [-5.0, -1.7479, -1.0499, -0.5005, 0.0, 0.5005, 1.0499, 1.7479, 5.0],
        4: [-5.0, -2.4015, -1.8443, -1.4380, -1.1001, -0.8002,
            -0.5229, -0.2585,  0.0,    0.2585,  0.5229,  0.8002,
             1.1001,  1.4380,  1.8443,  2.4015, 5.0],
    }

    def __init__(self, bits: int = 4):
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
        self.bits = bits
        self._rotation = None
        self._rotation_t = None
        self._centroids = None
        self._boundaries = None
        self._head_dim = None

    def _ensure_init(self, head_dim: int):
        """Lazy init rotation matrix and codebook for given head_dim."""
        if self._head_dim == head_dim:
            return
        self._head_dim = head_dim
        s = 1.0 / math.sqrt(head_dim)
        self._centroids = mx.array(self._CENTROIDS[self.bits], dtype=mx.float32) * s
        self._boundaries = mx.array(self._BOUNDARIES[self.bits], dtype=mx.float32) * s
        # Haar-distributed random orthogonal matrix
        key = mx.random.key(42)
        g = mx.random.normal(shape=(head_dim, head_dim), key=key)
        q, r = mx.linalg.qr(g, stream=mx.cpu)
        sign = mx.sign(mx.diag(r))
        sign = mx.where(sign == 0, 1, sign)
        self._rotation = q * sign
        self._rotation_t = self._rotation.T

    def _pq_quantize(self, vectors: mx.array) -> Tuple[mx.array, mx.array]:
        """Quantize vectors: norm → rotate → Lloyd-Max threshold → indices."""
        norms = mx.linalg.norm(vectors, axis=-1, keepdims=True)
        rotated = (vectors / mx.maximum(norms, 1e-8)) @ self._rotation_t
        inner = self._boundaries[1:-1]
        indices = mx.zeros(rotated.shape, dtype=mx.uint8)
        for b in range(inner.shape[0]):
            indices = indices + (rotated > inner[b]).astype(mx.uint8)
        return indices, norms

    def _pq_dequantize(self, indices: mx.array, norms: mx.array) -> mx.array:
        """Dequantize: centroids[idx] @ R * norms."""
        return self._centroids[indices] @ self._rotation * norms

    def _pack(self, indices: mx.array) -> mx.array:
        """Pack b-bit indices into uint32."""
        shape = indices.shape
        dim = shape[-1]
        vpi = 32 // self.bits
        n_packed = (dim + vpi - 1) // vpi
        pad_size = n_packed * vpi - dim
        if pad_size > 0:
            indices = mx.concatenate(
                [indices, mx.zeros((*shape[:-1], pad_size), dtype=indices.dtype)],
                axis=-1,
            )
        reshaped = indices.reshape(*shape[:-1], n_packed, vpi).astype(mx.uint32)
        shifts = mx.arange(vpi, dtype=mx.uint32) * self.bits
        shifted = reshaped << shifts
        packed = shifted[..., 0]
        for i in range(1, vpi):
            packed = packed | shifted[..., i]
        return packed

    def _unpack(self, packed: mx.array, dim: int) -> mx.array:
        """Unpack uint32 back to b-bit indices."""
        shape = packed.shape
        vpi = 32 // self.bits
        mask = (1 << self.bits) - 1
        shifts = mx.arange(vpi, dtype=mx.uint32) * self.bits
        extracted = (packed[..., None] >> shifts) & mask
        return extracted.reshape(*shape[:-1], shape[-1] * vpi)[..., :dim].astype(mx.uint8)

    def quantize(
        self,
        keys: mx.array,
        values: mx.array
    ) -> Tuple[mx.array, mx.array, Dict[str, Any]]:
        """Quantize using PolarQuant: rotate → Lloyd-Max → bit-pack."""
        B, n_heads, seq_len, head_dim = keys.shape
        self._ensure_init(head_dim)

        k_idx, k_norms = self._pq_quantize(keys)
        v_idx, v_norms = self._pq_quantize(values)

        packed_k = self._pack(k_idx)
        packed_v = self._pack(v_idx)

        metadata = {
            'k_norms': k_norms,
            'v_norms': v_norms,
            'head_dim': head_dim,
            'bits': self.bits,
            'seq_len': seq_len,
        }

        return packed_k, packed_v, metadata

    def dequantize(
        self,
        quant_keys: mx.array,
        quant_values: mx.array,
        metadata: Dict[str, Any]
    ) -> Tuple[mx.array, mx.array]:
        """Dequantize from PolarQuant packed format."""
        head_dim = metadata['head_dim']
        self._ensure_init(head_dim)

        k_idx = self._unpack(quant_keys, head_dim)
        v_idx = self._unpack(quant_values, head_dim)

        keys = self._pq_dequantize(k_idx, metadata['k_norms'])
        values = self._pq_dequantize(v_idx, metadata['v_norms'])

        return keys, values

    def get_compression_ratio(self) -> float:
        """Compression ratio vs fp16 (16 bits)."""
        # bits per element + norm overhead (negligible for long sequences)
        return 16.0 / (self.bits + 0.25)  # +0.25 for norm amortized

    def estimate_memory(self, num_tokens: int, head_dim: int, num_heads: int) -> int:
        """Estimate memory in bytes."""
        total_elements = num_tokens * head_dim * num_heads

        # Packed indices: bits per element, packed into uint32
        packed_size = (total_elements * self.bits + 31) // 32 * 4 * 2  # keys + values

        # Norms: fp32, one per token per head
        norms_size = num_tokens * num_heads * 4 * 2  # keys + values

        return int(packed_size + norms_size)


# ====================================================================
# QJL Projector (1-bit Quantized Johnson-Lindenstrauss)
# ====================================================================

class QJLProjector:
    """
    1-bit Quantized Johnson-Lindenstrauss transform.

    From Definition 1 in Zandieh et al., ICLR 2026:
        Q_qjl(x) = sign(S · x)
        Q_qjl^{-1}(z) = sqrt(π/2) / d · ||r|| · S^T · z

    Provides unbiased inner product estimates. All ops are standard MLX
    (matmul + sign), no custom Metal kernels needed.
    """

    def __init__(self, seed: int = 137):
        self._S = None
        self._seed = seed
        self._head_dim = None

    def _ensure_init(self, head_dim: int):
        if self._head_dim == head_dim:
            return
        self._head_dim = head_dim
        key = mx.random.key(self._seed)
        self._S = mx.random.normal(shape=(head_dim, head_dim), key=key)

    def quantize(self, residual: mx.array):
        """Project and sign-quantize residual vectors.

        Args:
            residual: (..., head_dim) residual vectors

        Returns:
            signs: (..., head_dim) int8 signs (-1/+1)
            norms: (..., 1) L2 norms
        """
        self._ensure_init(residual.shape[-1])
        norms = mx.linalg.norm(residual, axis=-1, keepdims=True)
        normalized = residual / mx.maximum(norms, 1e-8)
        projected = normalized @ self._S.T
        signs = mx.sign(projected).astype(mx.int8)
        signs = mx.where(signs == 0, mx.array(1, dtype=mx.int8), signs)
        return signs, norms

    def dequantize(self, signs: mx.array, norms: mx.array):
        """Reconstruct: sqrt(π/2) / d · ||r|| · S^T · z.

        Args:
            signs: (..., head_dim) int8 signs
            norms: (..., 1) original norms

        Returns:
            reconstructed: (..., head_dim) float vectors
        """
        d = self._head_dim
        scale = math.sqrt(math.pi / 2.0) / d
        reconstructed = signs.astype(mx.float32) @ self._S
        return (scale * norms * reconstructed).astype(mx.bfloat16)


# ====================================================================
# TurboQuant: PolarQuant (b-1 bit) + QJL residual (1 bit)
# ====================================================================

class TurboQuantizer(QuantizationStrategy):
    """
    Full TurboQuant: PolarQuant + QJL residual correction.

    Algorithm 2 from "TurboQuant: Online Vector Quantization with
    Near-optimal Distortion Rate" (Zandieh et al., ICLR 2026).

    Two-stage approach:
      Stage 1: PolarQuant at (b-1) bits → coarse quantization
      Stage 2: QJL 1-bit on the residual → unbiased correction

    Total: b bits per coordinate with unbiased inner product estimation.

    Uses chunk-aware eviction (requires_chunk_eviction=True) in
    TripleLayerKVCache to avoid re-quantization entirely. Each chunk is
    quantized ONCE when entering warm, and dequantized ONCE when evicted
    to cold. This preserves QJL residual correction quality.

    Parameters
    ----------
    bits : int
        Total bits per coordinate (must be >= 3). Default: 4.
        Uses (bits-1) for PolarQuant + 1 for QJL.
    qjl_alpha : float or None
        Damping factor for QJL residual correction.
        - None (default): auto-calibrate from head_dim using formula
          alpha = d / (d + 1152), calibrated at d=128 → alpha=0.1.
          Larger head_dim → larger alpha (QJL more accurate at higher d).
        - float: fixed alpha for all head_dims.

        QJL variance ∝ 1/d, so optimal alpha increases with head_dim:
          d=64:   alpha≈0.05  (very conservative)
          d=128:  alpha≈0.10  (verified on Qwen3-8B @ 16K)
          d=256:  alpha≈0.18
          d=512:  alpha≈0.31
          d=1024: alpha≈0.47  (approaching paper's 1.0)

        Empirical validation at d=128 (Qwen3-8B):
          alpha=0.0 (PQ-only): correct answer, minor artifacts
          alpha=0.1 (optimal): correct answer, fluent output
          alpha=0.5:           answer lost, text degraded
          alpha=1.0 (paper):   complete garbage output
    """

    @property
    def requires_chunk_eviction(self) -> bool:
        """TQ requires chunk-aware eviction to avoid re-quantization."""
        return True

    # Calibration constant for adaptive alpha: alpha = d / (d + QJL_D0).
    # Derived from: optimal alpha=0.1 at d=128 → d0 = 128/0.1 - 128 = 1152.
    QJL_D0 = 1152

    def __init__(self, bits: int = 4, qjl_alpha: float = None):
        if bits < 3:
            raise ValueError("TurboQuant requires bits >= 3 (PQ needs >= 2 bit)")
        self.bits = bits
        self._qjl_alpha_override = qjl_alpha  # None = auto from head_dim
        self._pq = PolarQuantizer(bits=bits - 1)
        self._qjl = QJLProjector(seed=137)

    def _get_alpha(self, head_dim: int) -> float:
        """Compute QJL damping factor, adaptive to head_dim.

        QJL variance ∝ 1/d, so larger d tolerates more correction.
        Formula: alpha = d / (d + 1152), calibrated at d=128 → 0.1.
        """
        if self._qjl_alpha_override is not None:
            return self._qjl_alpha_override
        return head_dim / (head_dim + self.QJL_D0)

    def _pack_signs(self, signs: mx.array) -> mx.array:
        """Pack int8 signs (-1/+1) into uint32 bitfields."""
        bits = ((signs.astype(mx.int32) + 1) // 2).astype(mx.uint32)
        shape = signs.shape
        d = shape[-1]
        n_packed = (d + 31) // 32
        pad_size = n_packed * 32 - d
        if pad_size > 0:
            bits = mx.concatenate(
                [bits, mx.zeros((*shape[:-1], pad_size), dtype=mx.uint32)],
                axis=-1,
            )
        reshaped = bits.reshape(*shape[:-1], n_packed, 32)
        shifts = mx.arange(32, dtype=mx.uint32)
        shifted = reshaped << shifts
        packed = shifted[..., 0]
        for i in range(1, 32):
            packed = packed | shifted[..., i]
        return packed

    def _unpack_signs(self, packed: mx.array, head_dim: int) -> mx.array:
        """Unpack uint32 bitfields to int8 signs (-1/+1)."""
        shape = packed.shape
        shifts = mx.arange(32, dtype=mx.uint32)
        extracted = (packed[..., None] >> shifts) & 1
        flat = extracted.reshape(*shape[:-1], shape[-1] * 32)[..., :head_dim]
        return (flat.astype(mx.int8) * 2 - 1)

    def quantize(
        self,
        keys: mx.array,
        values: mx.array
    ) -> Tuple[mx.array, mx.array, Dict[str, Any]]:
        """Two-stage quantization: PolarQuant (b-1 bit) + QJL residual (1 bit)."""
        B, n_heads, seq_len, head_dim = keys.shape
        self._qjl._ensure_init(head_dim)

        # Stage 1: PolarQuant at (b-1) bits
        pq_packed_k, pq_packed_v, pq_meta = self._pq.quantize(keys, values)

        # Reconstruct to compute residual
        k_approx, v_approx = self._pq.dequantize(pq_packed_k, pq_packed_v, pq_meta)

        # Compute residuals
        k_residual = keys.astype(mx.float32) - k_approx.astype(mx.float32)
        v_residual = values.astype(mx.float32) - v_approx.astype(mx.float32)

        # Stage 2: QJL on residuals
        k_signs, k_rnorms = self._qjl.quantize(k_residual)
        v_signs, v_rnorms = self._qjl.quantize(v_residual)

        # Pack signs into uint32
        k_signs_packed = self._pack_signs(k_signs)
        v_signs_packed = self._pack_signs(v_signs)

        metadata = {
            'pq_meta': pq_meta,
            'k_signs': k_signs_packed,
            'v_signs': v_signs_packed,
            'k_rnorms': k_rnorms,
            'v_rnorms': v_rnorms,
            'head_dim': head_dim,
            'bits': self.bits,
            'seq_len': seq_len,
        }

        return pq_packed_k, pq_packed_v, metadata

    def requantize(
        self,
        keys: mx.array,
        values: mx.array
    ) -> Tuple[mx.array, mx.array, Dict[str, Any]]:
        """Re-quantize using PQ-only to avoid QJL noise amplification.

        QJL residual correction is designed for *original* data residuals.
        When applied to already-degraded data, the residual is noise →
        QJL amplifies it → geometric error accumulation across overflow cycles.
        """
        return self._pq.quantize(keys, values)

    def dequantize(
        self,
        quant_keys: mx.array,
        quant_values: mx.array,
        metadata: Dict[str, Any]
    ) -> Tuple[mx.array, mx.array]:
        """Dequantize: full TQ (PQ + QJL) or PQ-only (from requantize)."""
        # PQ-only path: metadata from requantize() has no QJL fields
        if 'k_signs' not in metadata:
            return self._pq.dequantize(quant_keys, quant_values, metadata)

        # Full TQ path: PolarQuant + QJL residual correction
        head_dim = metadata['head_dim']
        self._qjl._ensure_init(head_dim)

        # Stage 1: PolarQuant reconstruction
        k_pq, v_pq = self._pq.dequantize(quant_keys, quant_values, metadata['pq_meta'])

        # Stage 2: QJL residual correction
        k_signs = self._unpack_signs(metadata['k_signs'], head_dim)
        v_signs = self._unpack_signs(metadata['v_signs'], head_dim)

        k_correction = self._qjl.dequantize(k_signs, metadata['k_rnorms'])
        v_correction = self._qjl.dequantize(v_signs, metadata['v_rnorms'])

        # Combine: x_hat = x_pq + alpha * x_qjl
        # Damped correction: QJL variance ∝ 1/d, adaptive alpha reduces noise.
        alpha = self._get_alpha(head_dim)
        keys_out = k_pq.astype(mx.float32) + alpha * k_correction.astype(mx.float32)
        values_out = v_pq.astype(mx.float32) + alpha * v_correction.astype(mx.float32)

        return keys_out.astype(mx.bfloat16), values_out.astype(mx.bfloat16)

    def get_compression_ratio(self) -> float:
        """Compression ratio vs fp16 (16 bits)."""
        return 16.0 / (self.bits + 0.5)

    def estimate_memory(self, num_tokens: int, head_dim: int, num_heads: int) -> int:
        """Estimate memory in bytes."""
        total_elements = num_tokens * head_dim * num_heads

        # PQ packed indices: (bits-1) per element
        pq_size = (total_elements * (self.bits - 1) + 31) // 32 * 4 * 2

        # QJL signs: 1 bit per element
        qjl_size = (total_elements + 31) // 32 * 4 * 2

        # Norms: PQ norms + QJL residual norms, fp32
        norms_size = num_tokens * num_heads * 4 * 4  # k_norms + v_norms + k_rnorms + v_rnorms

        return int(pq_size + qjl_size + norms_size)


# ====================================================================
# No-Op Quantizer (For Testing/Comparison)
# ====================================================================

class NoOpQuantizer(QuantizationStrategy):
    """
    No-op quantizer: stores data without quantization.

    Useful for:
    - Ablation studies (measure quantization impact)
    - Debugging (compare with/without quantization)
    - High-quality scenarios (no quality loss tolerance)

    Compression: 1.0x (no compression)

    Example
    -------
    >>> # Disable Warm quantization
    >>> cache = TripleLayerKVCache(warm_quantizer=NoOpQuantizer())
    """

    def quantize(
        self,
        keys: mx.array,
        values: mx.array
    ) -> Tuple[mx.array, mx.array, Dict[str, Any]]:
        """Return data as-is (no quantization)."""
        return keys, values, {}

    def dequantize(
        self,
        quant_keys: mx.array,
        quant_values: mx.array,
        metadata: Dict[str, Any]
    ) -> Tuple[mx.array, mx.array]:
        """Return data as-is."""
        return quant_keys, quant_values

    def get_compression_ratio(self) -> float:
        """No compression."""
        return 1.0

    def estimate_memory(self, num_tokens: int, head_dim: int, num_heads: int) -> int:
        """Full fp16 memory."""
        return num_tokens * head_dim * num_heads * 2 * 2  # keys + values, fp16


# ====================================================================
# Quantizer Registry (For Easy Selection)
# ====================================================================

QUANTIZER_REGISTRY = {
    'q4_0': Q4_0Quantizer,
    'q8_0': Q8_0Quantizer,
    'polarquant': PolarQuantizer,
    'turboquant': TurboQuantizer,  # full TurboQuant: PolarQuant + QJL
    'noop': NoOpQuantizer,
}


def get_quantizer(name: str, **kwargs) -> QuantizationStrategy:
    """
    Get quantizer by name.

    Parameters
    ----------
    name : str
        Quantizer name ('q4_0', 'polarquant', 'turboquant', 'noop')
    **kwargs
        Arguments passed to quantizer constructor

    Returns
    -------
    quantizer : QuantizationStrategy
        Quantizer instance

    Example
    -------
    >>> quantizer = get_quantizer('q4_0', group_size=32)
    >>> quantizer = get_quantizer('polarquant', bits=4)
    >>> quantizer = get_quantizer('polarquant', bits=3)  # more aggressive
    """
    if name not in QUANTIZER_REGISTRY:
        raise ValueError(
            f"Unknown quantizer: {name}. "
            f"Available: {list(QUANTIZER_REGISTRY.keys())}"
        )

    return QUANTIZER_REGISTRY[name](**kwargs)
