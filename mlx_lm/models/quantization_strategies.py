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


# Backward-compatible alias
TurboQuantizer = PolarQuantizer


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
    'polarquant': PolarQuantizer,
    'turboquant': PolarQuantizer,  # backward-compatible alias
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
