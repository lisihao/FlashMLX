"""
Triple-Layer KV Cache: Recent + Warm + Cold

Architecture:
    L0 (Recent) → 0-512 tokens    → exact (quality guarantee)
    L1 (Warm)   → 512-2048 tokens → KV quantization (Q4_0, ~2x compression)
    L2 (Cold)   → 2048+ tokens    → AM compression (R1.5, ~1.5x compression)

Key Innovations:
    1. Age-based layering: tokens move through layers as they age
    2. Multi-method compression: exact + quant + AM
    3. Quality-aware: most important (recent) tokens are exact
    4. Memory-efficient: older tokens use aggressive compression

Example:
    >>> cache = TripleLayerKVCache(
    ...     recent_size=512,
    ...     warm_size=1536,  # 2048 - 512
    ...     calibration_dir="/path/to/calibrations",
    ...     layer_idx=0,
    ...     compression_ratio=1.5
    ... )
"""

from typing import Optional, Tuple, Dict, Any
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import _BaseCache
from mlx_lm.models.double_layer_cache import CalibrationRegistry
from mlx_lm.models.quantization_strategies import (
    QuantizationStrategy,
    Q4_0Quantizer,
    get_quantizer
)
import numpy as np


class TripleLayerKVCache(_BaseCache):
    """
    Triple-layer KV cache with age-based compression.

    Layers:
        - L0 (Recent): 0-512 tokens, exact (no compression)
        - L1 (Warm): 512-2048 tokens, KV quantization (~2x compression)
        - L2 (Cold): 2048+ tokens, AM compression (~1.5x compression)

    Class Attributes:
        _shared_surprise: Cross-layer surprise scores from layer 0's keys.
        _shared_surprise_tag: (eviction_count, scorable_len) to prevent stale reads.

    Parameters
    ----------
    recent_size : int
        Size of L0 (Recent) layer (default: 512)
    warm_size : int
        Size of L1 (Warm) layer (default: 1536, i.e., 2048 - 512)
    calibration_dir : str, optional
        Directory containing AM calibration files for L2 (Cold)
    layer_idx : int, optional
        Layer index (required if using AM calibration)
    compression_ratio : float
        AM compression ratio for L2 (Cold) (default: 1.5)
    selection_strategy : str
        Calibration selection strategy: "ceil", "floor", "nearest" (default: "ceil")
    quant_bits : int
        Quantization bits for L1 (Warm) (default: 4)
    enable_warm_quant : bool
        Enable quantization for Warm layer (default: True)
    enable_cold_am : bool
        Enable AM compression for Cold layer (default: True)

    Example
    -------
    >>> # Production setup
    >>> cache = TripleLayerKVCache(
    ...     recent_size=512,
    ...     warm_size=1536,
    ...     calibration_dir="/tmp/am_calibrations_ultra_dense",
    ...     layer_idx=0,
    ...     compression_ratio=1.5
    ... )
    """

    # Cross-layer surprise sharing: layer 0 computes, all layers use
    _shared_surprise = None   # np.ndarray of windowed z-scores
    _shared_surprise_tag = None  # (eviction_count, scorable_len) freshness check

    # Attention probe sharing: layer 0 runs probe, all layers use scores
    _shared_probe = None                # H0Probe instance (set by cache_factory)
    _shared_attn_importance = None      # np.ndarray from probe
    _shared_attn_importance_tag = None  # freshness tag

    # Auto-reconstruction: stored by cache_factory for auto-reconstruct trigger
    _shared_inner_model = None          # inner model reference
    _shared_cache_list = None           # all TripleLayerKVCache instances

    def __init__(
        self,
        memory_budget_mb: float = 10.0,  # Memory budget in MB (legacy, secondary)
        warm_quantizer: Optional[QuantizationStrategy] = None,  # Pluggable quantizer
        recent_size: int = 512,
        warm_size: int = 1536,  # 2048 - 512
        calibration_dir: Optional[str] = None,
        calibration_file: Optional[str] = None,  # Direct calibration file path
        layer_idx: Optional[int] = None,
        compression_ratio: float = 1.5,
        selection_strategy: str = "ceil",
        quant_bits: int = 4,  # DEPRECATED: use warm_quantizer instead
        enable_warm_quant: bool = True,
        enable_cold_am: bool = True,
        enable_cold_quant: bool = True,  # Quantize Cold layer (when AM disabled)
        warm_overflow_threshold: int = 64,  # Batch Warm→Cold migration
        lazy_prefill_threshold: int = 8192,  # Skip Q4_0 during prefill for contexts below this
        scored_mode: bool = False,  # Architecture D: AM-scored on clean bf16
        scored_prefill_chunk_evict: bool = False,  # Enable prefill eviction for PP memory savings
        scored_prefill_max_cache: int = 4096,  # Eviction trigger: evict when cache exceeds this
        flat_quant: Optional[str] = None,  # Flat buffer quantization: None='bf16', 'q8_0'
        pinned_tokens: int = 0,  # First N tokens are never evicted (system prompt protection)
        density_mode: Optional[str] = None,  # Route 0: off | balanced | ultra_long | recall_first
        density_scale: float = 0.0,  # Route 0: log2 space bias (+1 = double compression)
    ):
        # Memory budget (PRIMARY trigger)
        self.memory_budget_mb = memory_budget_mb
        self.memory_budget_bytes = int(memory_budget_mb * 1024 * 1024)

        # Pluggable quantization strategy (NEW)
        if warm_quantizer is None:
            # Default: Q4_0 with specified quant_bits
            self.warm_quantizer = Q4_0Quantizer(group_size=32)
        else:
            self.warm_quantizer = warm_quantizer

        # Layer sizes (DEPRECATED: only used as hints)
        self.recent_size = recent_size
        self.warm_size = warm_size
        self.layer_idx = layer_idx

        # Compression settings
        self.compression_ratio = compression_ratio
        self.selection_strategy = selection_strategy
        self.quant_bits = quant_bits  # DEPRECATED
        self.enable_warm_quant = enable_warm_quant
        self.enable_cold_am = enable_cold_am
        # When AM is enabled, Cold still uses Q4_0 during PREFILL for memory savings.
        # AM compression is applied at promotion time (first TG token), not during prefill.
        self.enable_cold_quant = enable_cold_quant or enable_cold_am
        self.warm_overflow_threshold = warm_overflow_threshold
        self.lazy_prefill_threshold = lazy_prefill_threshold
        self.adaptive_ratio = (compression_ratio == 0)  # 0 = auto-select

        # L0 (Recent): exact storage
        self.recent_keys = None
        self.recent_values = None

        # L1 (Warm): quantized storage
        self.warm_keys = None
        self.warm_values = None
        self.warm_metadata = []  # List of quantization metadata (from warm_quantizer)
        # DEPRECATED: kept for backward compatibility
        self.warm_scales_k = None
        self.warm_scales_v = None

        # L2 (Cold): AM compressed storage (batch compression)
        # Split into compressed (already AM compressed) and pending (waiting for batch)
        self.cold_compressed_keys = None      # Already AM compressed
        self.cold_compressed_values = None
        self.cold_pending_keys = None         # Accumulating for batch compression
        self.cold_pending_values = None
        self.cold_batch_threshold = 512       # Default: compress when pending >= 512 tokens
        self.cold_pending_metadata = []      # Quantization metadata for Cold pending

        # Legacy attribute for backward compatibility
        self.cold_keys = None
        self.cold_values = None

        # AM calibration: direct file OR registry
        self._am_calibration = None  # Direct: {layer_idx: {beta, selected_indices, ...}}
        self.calibration_registry = None
        if enable_cold_am:
            if calibration_file:
                self._load_am_calibration(calibration_file)
            elif calibration_dir:
                self.calibration_registry = CalibrationRegistry(calibration_dir, auto_scan=True)
                available_lengths = self.calibration_registry.get_available_lengths(ratio=compression_ratio)
                if available_lengths:
                    self.cold_batch_threshold = max(available_lengths)
                    if layer_idx == 0:
                        print(f"[TripleLayerKVCache] Registry: cold_batch_threshold = {self.cold_batch_threshold}")
                elif layer_idx == 0:
                    print(f"[TripleLayerKVCache] Warning: No calibrations found for ratio {compression_ratio}")

        # Flat mode: after first TG token, promote to pre-allocated bf16 buffer
        # Uses same strategy as Standard KVCache: pre-alloc + slice assignment + view
        self._flat_keys = None
        self._flat_values = None
        self._flat_mode = False
        self._flat_offset = 0
        self._flat_step = 256
        self._needs_cleanup = False  # Deferred cleanup of quantized data
        self._true_offset = 0  # Original token count (before AM), used for RoPE

        # Statistics
        self.num_warm_compressions = 0
        self.num_cold_compressions = 0
        self._total_offset = 0

        # Architecture D P2: Scored-as-Warm-tier
        # AM scores on clean bf16 → important(bf16 flat) + unimportant(dropped)
        self.scored_mode = scored_mode
        self._scored_active = False
        if scored_mode:
            self.primary_quantizer = None  # P2: important tokens go bf16 directly
        else:
            self.primary_quantizer = None

        # Chunked prefill eviction: score and evict during PP to bound memory
        self._scored_prefill_chunk_evict = scored_prefill_chunk_evict
        self._scored_prefill_max_cache = scored_prefill_max_cache
        self._prefill_tokens_seen = 0
        self._prefill_eviction_count = 0

        # Flat buffer quantization: None (bf16), 'q8_0' (int8 + per-token scales),
        # 'q4_0' (nibble-packed uint8 + per-group scales, group_size=32),
        # or 'turboquant' (PolarQuant packed uint32 + per-token norms)
        self._flat_quant = flat_quant
        self._flat_keys_scales = None
        self._flat_values_scales = None
        self._q4_group_size = 32
        # TurboQuant flat buffer: PolarQuantizer for quantize-on-write/dequantize-on-read
        # norm_correction=False: safe for head_dim>=128 (gated by _TURBOQUANT_MIN_HEAD_DIM),
        # yields ~52% dequant speedup (verified 15/15 token match on Qwen3-8B).
        self._flat_pq = None
        self._flat_pq_head_dim = None
        if flat_quant == 'turboquant':
            from mlx_lm.models.quantization_strategies import PolarQuantizer
            self._flat_pq = PolarQuantizer(bits=4, norm_correction=False)

        # Pinned prefix: first N tokens are never evicted (system prompt protection).
        # When AM scoring/compression runs, pinned tokens are unconditionally kept.
        # Typical usage: system prompt (500-2000 tokens) is pinned for multi-agent reuse.
        self.pinned_tokens = pinned_tokens

        # Route 5: Scored KV-Direct fusion — h^(0) archive reference
        self._h0_store = None  # Set by cache_factory for scored_kv_direct strategy

        # H0Probe attention-based eviction
        self._probe_eviction_enabled = False  # Set by cache_factory

        # Auto-reconstruction: trigger h^(0) reconstruction after prefill
        self._auto_reconstruct = False  # Set by cache_factory

        # Route 5: Reconstructed K/V injection
        self._recon_keys = None
        self._recon_values = None
        self._recon_persistent = True  # persist across TG steps (not just first token)

        # Merge strategy: track prefix-era tokens in flat buffer for dedup.
        # After _scored_compress_prefix, the flat buffer layout is:
        #   [pinned (0:pin_n)] [hot scored] [recent] [TG appended...]
        # When recall injects [0:N], tokens in [pinned + hot scored] overlap.
        # _flat_prefix_token_count = pinned + hot scored (tokens to skip on recall).
        self._flat_prefix_token_count = 0

        # Route 0: Density Router parameters + signal
        self._density_mode = density_mode  # None = off
        self._density_scale = density_scale
        self._density_signal = None  # populated by _scored_compress_prefix

    # -------------------------------------------------------------------
    # Route 0: Density signal extraction
    # -------------------------------------------------------------------

    def _extract_density_signal(
        self,
        importance_masks: list,
        total_len: int,
        scorable_len: int,
        n_hot: int,
        n_dropped: int,
    ) -> dict:
        """Extract density signal from AM importance masks.

        Computes per-chunk and aggregate density metrics from the AM scoring
        results. These metrics feed Route 0's discrete compression level
        selection (see config.DensityLevel + snap_to_nearest).

        Zero-cost: operates on already-computed numpy boolean arrays.

        Returns dict with:
            keep_ratio: fraction of scorable tokens kept (= 1/effective_ratio)
            concentration: mean gap coefficient-of-variation across chunks
                (low = uniform distribution, high = clustered important tokens)
            log2_ratio: log2 of effective compression ratio
            n_chunks: number of full chunks scored
            chunk_concentrations: per-chunk concentration values
        """
        import math

        if not importance_masks or scorable_len == 0:
            return {
                "keep_ratio": 1.0,
                "concentration": 0.0,
                "log2_ratio": 0.0,
                "n_chunks": 0,
                "chunk_concentrations": [],
            }

        actual_ratio = scorable_len / max(n_hot, 1)
        log2_ratio = math.log2(max(actual_ratio, 1.01))
        keep_ratio = n_hot / scorable_len

        # Per-chunk concentration: coefficient of variation of gaps between
        # important token positions.  Low CV = evenly spread (uniform density),
        # high CV = clustered (variable density within chunk).
        chunk_concentrations = []
        for mask in importance_masks:
            positions = np.where(mask)[0]
            if len(positions) > 1:
                gaps = np.diff(positions).astype(np.float32)
                mean_gap = gaps.mean()
                cv = float(gaps.std() / max(mean_gap, 1e-6))
                chunk_concentrations.append(cv)
            else:
                chunk_concentrations.append(0.0)

        mean_concentration = float(np.mean(chunk_concentrations)) if chunk_concentrations else 0.0

        return {
            "keep_ratio": float(keep_ratio),
            "concentration": mean_concentration,
            "log2_ratio": float(log2_ratio),
            "n_chunks": len(importance_masks),
            "chunk_concentrations": chunk_concentrations,
        }

    def inject_reconstruction(self, recon_keys, recon_values):
        """Inject reconstructed K/V for next attention step.

        The injected K/V is prepended to the flat buffer output on the
        next _fetch_flat call, then automatically cleared.

        Args:
            recon_keys: (B, n_kv_heads, N, head_dim) reconstructed keys
            recon_values: (B, n_kv_heads, N, head_dim) reconstructed values
        """
        self._recon_keys = recon_keys
        self._recon_values = recon_values

    def clear_reconstruction(self):
        """Explicitly clear persistent reconstruction data.

        Call this when generation is complete or cache is being reset,
        to free the memory held by reconstructed K/V tensors.
        """
        self._recon_keys = None
        self._recon_values = None

    def _load_am_calibration(self, filepath: str):
        """Load AM calibration file directly (bypasses CalibrationRegistry)."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Handle both formats:
        # Format 1 (offline): {model_name, calibration: {layer_idx: {...}}, ...}
        # Format 2 (on-policy): {layer_idx: {beta, selected_indices, ...}}
        if 'calibration' in data:
            self._am_calibration = data['calibration']
        else:
            self._am_calibration = data

        # Derive batch threshold from calibration metadata
        # Use first available layer (hybrid models only have attention layer indices)
        first_key = min(self._am_calibration.keys())
        first_layer = self._am_calibration[first_key]
        budget = first_layer['budget']
        ratio = first_layer['compression_ratio']
        self._am_prefix_len = int(budget * ratio)
        self.cold_batch_threshold = self._am_prefix_len

        if self.layer_idx == first_key:
            print(f"[TripleLayerKVCache] AM calibration loaded: {filepath}")
            print(f"  Prefix: {self._am_prefix_len} → Budget: {budget} ({ratio}x)")

    @property
    def offset(self):
        """Total cache size (for RoPE positioning)."""
        if self._scored_active:
            return self._true_offset
        if self._flat_mode:
            # Use original offset for correct RoPE, even if AM compressed the buffer
            return self._true_offset
        # Chunked prefill with eviction: physical cache < total tokens seen
        if self._prefill_tokens_seen > 0:
            return self._prefill_tokens_seen
        cold_size = 0
        if self.cold_compressed_keys is not None:
            cold_size += self.cold_compressed_keys.shape[2]
        if self.cold_pending_keys is not None:
            cold_size += self.cold_pending_keys.shape[2]
        warm_size = self.warm_keys.shape[2] if self.warm_keys is not None else 0
        recent_size = self.recent_keys.shape[2] if self.recent_keys is not None else 0
        return cold_size + warm_size + recent_size

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Update cache with new keys/values and return full cache.

        Two-phase design:
            - Prefill: triple-layer quantized path (memory-efficient)
            - TG: pre-allocated bf16 buffer with slice assignment (Standard KVCache speed)
        """
        # FLAT FAST PATH: pre-allocated buffer — same strategy as Standard KVCache
        # (Architecture D scored mode also transitions to flat mode after promotion)
        if self._flat_mode:
            prev = self._flat_offset
            # Grow buffer if needed (rare: only every _flat_step tokens)
            if prev + keys.shape[2] > self._flat_keys.shape[2]:
                n_steps = (self._flat_step + keys.shape[2] - 1) // self._flat_step
                B, n_heads, _, head_dim = self._flat_keys.shape
                new_k = mx.zeros((B, n_heads, n_steps * self._flat_step, head_dim), dtype=self._flat_keys.dtype)
                new_v = mx.zeros((B, n_heads, n_steps * self._flat_step, head_dim), dtype=self._flat_values.dtype)
                self._flat_keys = mx.concatenate([self._flat_keys, new_k], axis=2)
                self._flat_values = mx.concatenate([self._flat_values, new_v], axis=2)
                if self._flat_quant in ('q8_0', 'q4_0', 'turboquant') and self._flat_keys_scales is not None:
                    scale_dim = self._flat_keys_scales.shape[-1]  # 1 for q8_0/turboquant, D//G for q4_0
                    scale_dtype = self._flat_keys_scales.dtype  # bf16 for q8/q4, float32 for turboquant
                    new_sk = mx.zeros((B, n_heads, n_steps * self._flat_step, scale_dim), dtype=scale_dtype)
                    new_sv = mx.zeros((B, n_heads, n_steps * self._flat_step, scale_dim), dtype=scale_dtype)
                    self._flat_keys_scales = mx.concatenate([self._flat_keys_scales, new_sk], axis=2)
                    self._flat_values_scales = mx.concatenate([self._flat_values_scales, new_sv], axis=2)

            # Slice assignment — O(1) for bf16, quantize-on-write for Q8_0 (PP only)
            self._flat_offset += keys.shape[2]
            self._true_offset += keys.shape[2]
            self._write_flat(prev, self._flat_offset, keys, values)
            return self._fetch_flat(self._flat_offset)

        return self._update_slow_path(keys, values)

    def _cleanup_quantized(self):
        """Free quantized data after flat buffer is materialized."""
        self.cold_pending_keys = None
        self.cold_pending_values = None
        self.cold_pending_metadata = []
        self.cold_compressed_keys = None
        self.cold_compressed_values = None
        self.warm_keys = None
        self.warm_values = None
        self.warm_metadata = []
        self.recent_keys = None
        self.recent_values = None

    # ── Q8_0 flat buffer helpers ─────────────────────────────────────────

    def _q8_quantize(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Per-token absmax quantization to int8.

        Args:
            x: (B, H, S, D) bf16/float32 tensor
        Returns:
            quant: (B, H, S, D) int8 tensor
            scales: (B, H, S, 1) bf16 tensor
        """
        max_val = mx.max(mx.abs(x), axis=-1, keepdims=True)
        scales = (max_val / 127.0).astype(mx.bfloat16)
        scales = mx.maximum(scales, mx.array(1e-8, dtype=mx.bfloat16))
        quant = mx.round(x / scales).astype(mx.int8)
        return quant, scales

    def _q8_dequantize(self, quant: mx.array, scales: mx.array) -> mx.array:
        """Dequantize int8 back to bf16.

        Args:
            quant: (B, H, S, D) int8
            scales: (B, H, S, 1) bf16
        Returns:
            (B, H, S, D) bf16
        """
        return quant.astype(mx.bfloat16) * scales

    # ── Q4_0 flat buffer helpers (nibble-packed, per-group scales) ──────

    def _q4_quantize(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Per-group 4-bit symmetric quantization with nibble packing.

        Args:
            x: (B, H, S, D) bf16 tensor, D must be divisible by group_size
        Returns:
            packed: (B, H, S, D//2) uint8 (two 4-bit values per byte)
            scales: (B, H, S, D//group_size) bf16 (per-group scales)
        """
        B, H, S, D = x.shape
        G = self._q4_group_size
        num_groups = D // G

        # Reshape into groups: (B, H, S, num_groups, G)
        grouped = x.reshape(B, H, S, num_groups, G)

        # Per-group absmax → scale
        max_val = mx.max(mx.abs(grouped), axis=-1, keepdims=True)
        scales = (max_val / 7.0).astype(mx.bfloat16)
        scales = mx.maximum(scales, mx.array(1e-8, dtype=mx.bfloat16))

        # Quantize to [-7, 7], shift to [1, 15] unsigned for packing
        quant = mx.clip(mx.round(grouped / scales), -7, 7)
        quant_u = (quant + 8).astype(mx.uint8)  # [1, 15]

        # Flatten groups back to (B, H, S, D)
        quant_flat = quant_u.reshape(B, H, S, D)

        # Nibble pack: pair adjacent → one uint8
        # Use multiply+add to avoid potential bitwise issues on uint8
        high = quant_flat[..., 0::2].astype(mx.uint32)
        low = quant_flat[..., 1::2].astype(mx.uint32)
        packed = (high * 16 + low).astype(mx.uint8)  # (B, H, S, D//2)

        return packed, scales.squeeze(-1)  # scales: (B, H, S, num_groups)

    def _q4_dequantize(self, packed: mx.array, scales: mx.array) -> mx.array:
        """Dequantize nibble-packed 4-bit data back to bf16.

        Args:
            packed: (B, H, S, D//2) uint8
            scales: (B, H, S, num_groups) bf16
        Returns:
            (B, H, S, D) bf16
        """
        B, H, S, half_D = packed.shape
        D = half_D * 2
        G = self._q4_group_size
        num_groups = D // G

        # Unpack nibbles via integer division/modulo
        p = packed.astype(mx.int32)
        high = p // 16 - 8   # high nibble → signed [-8, 7]
        low = (p % 16) - 8   # low nibble → signed [-8, 7]

        # Interleave: (B, H, S, D//2, 2) → (B, H, S, D)
        interleaved = mx.stack([high, low], axis=-1)
        unpacked = interleaved.reshape(B, H, S, D).astype(mx.bfloat16)

        # Expand scales: (B, H, S, num_groups) → (B, H, S, num_groups, G) → (B, H, S, D)
        scales_expanded = scales[..., :, None]  # (B, H, S, num_groups, 1)
        # Reshape unpacked into groups, multiply, reshape back
        grouped = unpacked.reshape(B, H, S, num_groups, G)
        result = (grouped * scales_expanded).reshape(B, H, S, D)

        return result

    # TurboQuant requires head_dim >= 128 for usable attention quality.
    # PolarQuant's random rotation converges to Gaussian in high dimensions;
    # below 128 the per-vector error destroys attention patterns in practice
    # (verified: 0.5B/head_dim=64 → 2/10 token match, 8B/head_dim=128 → 10/10).
    _TURBOQUANT_MIN_HEAD_DIM = 128

    def _alloc_flat_buffer(self, B: int, n_heads: int, alloc_len: int, head_dim: int):
        """Allocate flat buffer (bf16, Q8_0, Q4_0, or TurboQuant based on flat_quant)."""
        # Auto-downgrade turboquant for small head_dim models
        if self._flat_quant == 'turboquant' and head_dim < self._TURBOQUANT_MIN_HEAD_DIM:
            if self.layer_idx == 0:
                print(f"[TripleLayerKVCache] turboquant requires head_dim≥{self._TURBOQUANT_MIN_HEAD_DIM}, "
                      f"got {head_dim}. Auto-downgrading to q4_0.")
            self._flat_quant = 'q4_0'
            self._flat_pq = None
            self._flat_pq_head_dim = None
        if self._flat_quant == 'q8_0':
            self._flat_keys = mx.zeros((B, n_heads, alloc_len, head_dim), dtype=mx.int8)
            self._flat_values = mx.zeros((B, n_heads, alloc_len, head_dim), dtype=mx.int8)
            self._flat_keys_scales = mx.zeros((B, n_heads, alloc_len, 1), dtype=mx.bfloat16)
            self._flat_values_scales = mx.zeros((B, n_heads, alloc_len, 1), dtype=mx.bfloat16)
        elif self._flat_quant == 'q4_0':
            G = self._q4_group_size
            self._flat_keys = mx.zeros((B, n_heads, alloc_len, head_dim // 2), dtype=mx.uint8)
            self._flat_values = mx.zeros((B, n_heads, alloc_len, head_dim // 2), dtype=mx.uint8)
            self._flat_keys_scales = mx.zeros((B, n_heads, alloc_len, head_dim // G), dtype=mx.bfloat16)
            self._flat_values_scales = mx.zeros((B, n_heads, alloc_len, head_dim // G), dtype=mx.bfloat16)
        elif self._flat_quant == 'turboquant':
            packed_dim = self._flat_pq.flat_packed_dim(head_dim)
            self._flat_keys = mx.zeros((B, n_heads, alloc_len, packed_dim), dtype=mx.uint32)
            self._flat_values = mx.zeros((B, n_heads, alloc_len, packed_dim), dtype=mx.uint32)
            # Per-token norms (float32 for precision, 1 per token per head)
            self._flat_keys_scales = mx.zeros((B, n_heads, alloc_len, 1), dtype=mx.float32)
            self._flat_values_scales = mx.zeros((B, n_heads, alloc_len, 1), dtype=mx.float32)
            self._flat_pq_head_dim = head_dim
        else:
            self._flat_keys = mx.zeros((B, n_heads, alloc_len, head_dim), dtype=mx.bfloat16)
            self._flat_values = mx.zeros((B, n_heads, alloc_len, head_dim), dtype=mx.bfloat16)

    def _write_flat(self, start: int, end: int, keys: mx.array, values: mx.array):
        """Write to flat buffer (handles quantize-on-write for Q8_0/Q4_0/TurboQuant)."""
        if self._flat_quant == 'q8_0':
            qk, sk = self._q8_quantize(keys)
            qv, sv = self._q8_quantize(values)
            self._flat_keys[..., start:end, :] = qk
            self._flat_values[..., start:end, :] = qv
            self._flat_keys_scales[..., start:end, :] = sk
            self._flat_values_scales[..., start:end, :] = sv
        elif self._flat_quant == 'q4_0':
            qk, sk = self._q4_quantize(keys)
            qv, sv = self._q4_quantize(values)
            self._flat_keys[..., start:end, :] = qk
            self._flat_values[..., start:end, :] = qv
            self._flat_keys_scales[..., start:end, :] = sk
            self._flat_values_scales[..., start:end, :] = sv
        elif self._flat_quant == 'turboquant':
            pk, nk = self._flat_pq.flat_quantize(keys)
            pv, nv = self._flat_pq.flat_quantize(values)
            self._flat_keys[..., start:end, :] = pk
            self._flat_values[..., start:end, :] = pv
            self._flat_keys_scales[..., start:end, :] = nk
            self._flat_values_scales[..., start:end, :] = nv
        else:
            self._flat_keys[..., start:end, :] = keys
            self._flat_values[..., start:end, :] = values

    def _fetch_flat(self, end: int) -> Tuple[mx.array, mx.array]:
        """Fetch from flat buffer (dequant Q8/Q4/TurboQuant on every read — keeps TG memory minimal)."""
        if self._flat_quant == 'q8_0':
            k = self._flat_keys[..., :end, :].astype(mx.bfloat16) * self._flat_keys_scales[..., :end, :]
            v = self._flat_values[..., :end, :].astype(mx.bfloat16) * self._flat_values_scales[..., :end, :]
        elif self._flat_quant == 'q4_0':
            k = self._q4_dequantize(self._flat_keys[..., :end, :], self._flat_keys_scales[..., :end, :])
            v = self._q4_dequantize(self._flat_values[..., :end, :], self._flat_values_scales[..., :end, :])
        elif self._flat_quant == 'turboquant':
            k = self._flat_pq.flat_dequantize(
                self._flat_keys[..., :end, :], self._flat_keys_scales[..., :end, :], self._flat_pq_head_dim)
            v = self._flat_pq.flat_dequantize(
                self._flat_values[..., :end, :], self._flat_values_scales[..., :end, :], self._flat_pq_head_dim)
        else:
            k, v = self._flat_keys[..., :end, :], self._flat_values[..., :end, :]

        # Route 5: Prepend reconstructed K/V if available.
        # Dedup: recall covers [0:N] which overlaps with prefix-era hot tokens
        # in the flat buffer. Skip those to avoid duplicate attention entries.
        # Layout after merge: [recall [0:N]] + [non-prefix tokens from flat buffer]
        if self._recon_keys is not None:
            skip = self._flat_prefix_token_count
            if skip > 0 and skip < end:
                # Skip prefix-era tokens (covered by recall), keep rest
                k_rest = k[..., skip:, :]
                v_rest = v[..., skip:, :]
                k = mx.concatenate([self._recon_keys, k_rest], axis=2)
                v = mx.concatenate([self._recon_values, v_rest], axis=2)
            else:
                # No prefix tokens to skip (or skip covers everything)
                k = mx.concatenate([self._recon_keys, k], axis=2)
                v = mx.concatenate([self._recon_values, v], axis=2)
            if not self._recon_persistent:
                self._recon_keys = None
                self._recon_values = None

        return k, v

    def _eval_flat_buffer(self):
        """Evaluate flat buffer arrays (includes scales/norms for Q8_0/Q4_0/TurboQuant)."""
        mx.eval(self._flat_keys, self._flat_values)
        if self._flat_quant in ('q8_0', 'q4_0', 'turboquant'):
            mx.eval(self._flat_keys_scales, self._flat_values_scales)

    def _update_slow_path(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """Triple-layer path: prefill + first TG token promotion."""
        # 1. Append to Recent
        if self.recent_keys is None:
            self.recent_keys = keys
            self.recent_values = values
        else:
            self.recent_keys = mx.concatenate([self.recent_keys, keys], axis=2)
            self.recent_values = mx.concatenate([self.recent_values, values], axis=2)

        # Track total tokens for RoPE positioning (chunked prefill eviction)
        if self._scored_prefill_chunk_evict:
            self._prefill_tokens_seen += keys.shape[2]

        # 2. LAZY PREFILL: during prefill (multi-token), skip Q4_0 quantization
        #    when context is small enough. Tokens stay in bf16 Recent for PP speed.
        #    Aging + quantization is deferred to promotion (first TG token).
        #    For long contexts (> threshold), do incremental aging to avoid TTFT spike.
        if keys.shape[2] > 1:
            # Scored chunked prefill eviction: bound PP memory by evicting cold tokens
            if self.scored_mode and self._scored_prefill_chunk_evict:
                total = self.recent_keys.shape[2]
                if total > self._scored_prefill_max_cache:
                    self._scored_prefill_evict()
                return self.recent_keys, self.recent_values

            if self.recent_keys.shape[2] <= self.lazy_prefill_threshold:
                return self.recent_keys, self.recent_values
            # Long context: do incremental aging during prefill
            if self.recent_keys.shape[2] > self.recent_size:
                self._manage_aging()
            return self._concat_all_layers()

        # 3. First TG token — SCORED FAST PROMOTION:
        #    When scored mode has all data in bf16 Recent (lazy prefill),
        #    skip aging entirely. Go directly from bf16 to AM scoring.
        #    Avoids pointless Q4_0 quantize → immediate dequant roundtrip.
        has_am = self.enable_cold_am and (self._am_calibration is not None or self.calibration_registry is not None)
        no_warm_cold = self.warm_keys is None and self.cold_pending_keys is None

        if self.scored_mode and has_am and no_warm_cold and self.recent_keys.shape[2] > self.recent_size:
            if self._prefill_tokens_seen > 0:
                # Chunked prefill eviction: already scored during PP, just promote to flat
                self._true_offset = self._prefill_tokens_seen
                self._promote_to_flat_buffer(self.recent_keys, self.recent_values)
            else:
                # Full scored compression (original path: single-pass prefill)
                full_keys = self.recent_keys
                full_values = self.recent_values
                B, n_heads, cache_len, head_dim = full_keys.shape
                self._true_offset = cache_len
                recent_len = self.recent_size
                mx.eval(full_keys, full_values)
                self.recent_keys = None
                self.recent_values = None
                self._scored_compress_prefix(full_keys, full_values, recent_len)
                del full_keys, full_values
            self._scored_active = True

            # Auto-reconstruction: layer 0 triggers h^(0) → K/V for all layers
            if self.layer_idx == 0:
                self._auto_reconstruct_if_enabled()

            return self._fetch_flat(self._flat_offset)

        # 3b. First TG token (non-scored): reorganize accumulated tokens into layers
        if self.recent_keys.shape[2] > self.recent_size:
            self._manage_aging()

        # Save recent length before concat (needed for AM prefix split)
        recent_len = self.recent_keys.shape[2] if self.recent_keys is not None else 0

        # 4. Concatenate all layers (dequant Cold+Warm)
        full_keys, full_values = self._concat_all_layers()

        # 5. Promote to pre-allocated flat buffer on first TG token
        #    Trigger: layers exist from aging, OR large Recent from lazy prefill
        has_layers = self.warm_keys is not None or self.cold_pending_keys is not None
        if has_layers:
            B, n_heads, cache_len, head_dim = full_keys.shape

            # Record original cache length for correct RoPE positioning
            self._true_offset = cache_len

            # Stage 1: Materialize dequanted data, then free quantized originals
            # This prevents peak memory from holding both Q4_0 + bf16 simultaneously
            mx.eval(full_keys, full_values)
            self._cleanup_quantized()

            # Stage 2: Branch based on architecture mode
            if self.scored_mode and has_am:
                # Architecture D: Score on clean bf16 → hot/cold split → flat buffer
                self._scored_compress_prefix(full_keys, full_values, recent_len)
                del full_keys, full_values
                self._scored_active = True
                return self._fetch_flat(self._flat_offset)

            # Pipeline mode (original): AM prune → flat buffer
            if has_am:
                full_keys, full_values = self._am_compress_prefix(
                    full_keys, full_values, recent_len
                )
                mx.eval(full_keys, full_values)
                cache_len = full_keys.shape[2]  # Update to compressed length

            # Stage 3: Allocate flat buffer and copy (quantized data already freed)
            alloc_len = ((cache_len + self._flat_step - 1) // self._flat_step + 1) * self._flat_step
            self._alloc_flat_buffer(B, n_heads, alloc_len, head_dim)
            self._write_flat(0, cache_len, full_keys, full_values)
            self._eval_flat_buffer()  # Force eval before freeing source data
            del full_keys, full_values  # Free intermediate bf16 (flat buffer owns the data now)
            self._flat_offset = cache_len
            self._flat_mode = True
            # Track prefix tokens for recall dedup
            self._flat_prefix_token_count = cache_len - recent_len
            return self._fetch_flat(self._flat_offset)

        return full_keys, full_values

    def _get_effective_ratio(self, context_len: int) -> float:
        """Select optimal compression ratio based on context length and Route 0 density.

        Without Route 0 (density_mode=None):
            Adaptive: <= 16K → 3.0x, > 16K → 1.5x
            Explicit: use self.compression_ratio directly

        With Route 0 (density_mode set):
            Base ratio (adaptive or explicit) → log2 → + density_scale →
            snap_to_nearest DensityLevel → use that level's compression_ratio.
        """
        # Step 1: Get base ratio
        if not self.adaptive_ratio:
            base_ratio = self.compression_ratio
        elif context_len <= 16384:
            base_ratio = 3.0
        else:
            base_ratio = 1.5

        # Step 2: Apply Route 0 density_scale (if active)
        if self._density_mode is not None and self._density_mode != "off":
            if self._density_scale == 0.0:
                return base_ratio  # passthrough: no discretization at scale=0
            import math
            from flashmlx.config import snap_to_nearest
            base_log2 = math.log2(max(base_ratio, 1.01))
            level = snap_to_nearest(base_log2, scale=self._density_scale)
            return level.compression_ratio

        return base_ratio

    def _am_compress_prefix(
        self,
        keys: mx.array,
        values: mx.array,
        recent_len: int
    ) -> Tuple[mx.array, mx.array]:
        """
        Apply AM compression to old prefix (non-recent tokens) at promotion time.

        Splits the full cache into [pinned | scorable | recent], compresses
        scorable region in chunks of cold_batch_threshold (512), reassembles.

        Pinned tokens (first self.pinned_tokens) are never compressed.
        Only compresses FULL chunks. Partial remainder is kept uncompressed.
        """
        B, n_heads, total_len, head_dim = keys.shape
        old_len = total_len - recent_len

        if old_len <= 0:
            return keys, values

        # Split: [pinned | scorable | recent]
        pin_n = min(self.pinned_tokens, old_len)
        pinned_keys = keys[:, :, :pin_n, :] if pin_n > 0 else None
        pinned_values = values[:, :, :pin_n, :] if pin_n > 0 else None
        scorable_keys = keys[:, :, pin_n:old_len, :]
        scorable_values = values[:, :, pin_n:old_len, :]
        recent_keys = keys[:, :, old_len:, :]
        recent_values = values[:, :, old_len:, :]
        scorable_len = old_len - pin_n

        # Compression parameters — adaptive ratio selects based on context length
        effective_ratio = self._get_effective_ratio(total_len)
        chunk_size = self.cold_batch_threshold  # 512
        budget = int(chunk_size / effective_ratio)

        compressed_k = []
        compressed_v = []
        n_full_chunks = 0

        # Pinned tokens go first (unconditionally)
        if pin_n > 0:
            compressed_k.append(pinned_keys)
            compressed_v.append(pinned_values)

        # Compress scorable region
        for offset in range(0, scorable_len, chunk_size):
            chunk_end = min(offset + chunk_size, scorable_len)
            chunk_k = scorable_keys[:, :, offset:chunk_end, :]
            chunk_v = scorable_values[:, :, offset:chunk_end, :]
            chunk_len = chunk_end - offset

            if chunk_len == chunk_size:
                # Full chunk: apply AM compression
                comp_k, comp_v = self._compress_with_am(
                    chunk_k, chunk_v, budget, chunk_size
                )
                compressed_k.append(comp_k)
                compressed_v.append(comp_v)
                n_full_chunks += 1
                self.num_cold_compressions += 1
            else:
                # Partial chunk: keep uncompressed
                compressed_k.append(chunk_k)
                compressed_v.append(chunk_v)

        # Reassemble: pinned + compressed_scorable + recent
        compressed_k.append(recent_keys)
        compressed_v.append(recent_values)

        result_keys = mx.concatenate(compressed_k, axis=2)
        result_values = mx.concatenate(compressed_v, axis=2)

        if self.layer_idx == 0:
            saved = total_len - result_keys.shape[2]
            ratio_info = f" (adaptive→{effective_ratio}x)" if self.adaptive_ratio else ""
            pin_info = f"{pin_n} pinned + " if pin_n > 0 else ""
            print(f"[AM Promotion{ratio_info}] {pin_info}{scorable_len} scorable + {recent_len} recent → "
                  f"{result_keys.shape[2]} tokens "
                  f"({n_full_chunks} chunks compressed, saved {saved} tokens)")

        return result_keys, result_values

    def _calculate_memory(self) -> int:
        """
        Calculate current memory usage in bytes.

        Uses quantizer's estimate_memory() for accurate Warm layer calculation.

        Returns
        -------
        total_bytes : int
            Estimated memory usage in bytes
        """
        total_bytes = 0

        # Flat mode (pipeline AM or scored mode): flat bf16 / Q8_0 / Q4_0 / TurboQuant buffer
        if self._flat_mode and self._flat_keys is not None:
            B, H, _, D = self._flat_keys.shape
            if self._flat_quant == 'q8_0':
                # int8 data (1 byte) + bf16 scales per token (2 bytes), K+V
                head_dim = D  # D is head_dim for Q8_0
                total_bytes += self._flat_offset * H * (head_dim * 1 + 1 * 2) * 2
            elif self._flat_quant == 'q4_0':
                # D is head_dim//2 for Q4_0 (nibble-packed)
                half_dim = D
                head_dim = half_dim * 2
                G = self._q4_group_size
                # uint8 packed (0.5 byte/val) + bf16 scales per group (2 bytes * head_dim/G), K+V
                total_bytes += self._flat_offset * H * (half_dim * 1 + (head_dim // G) * 2) * 2
            elif self._flat_quant == 'turboquant':
                # D is packed_dim (uint32). Per token: packed_dim*4 bytes + 1 norm (4 bytes), K+V
                packed_dim = D
                total_bytes += self._flat_offset * H * (packed_dim * 4 + 1 * 4) * 2
            else:
                head_dim = D
                total_bytes += self._flat_offset * H * head_dim * 2 * 2  # K+V, bf16
            return total_bytes

        # Recent layer (fp32/fp16)
        if self.recent_keys is not None:
            dtype_size = 4 if self.recent_keys.dtype == mx.float32 else 2
            recent_size = self.recent_keys.size * dtype_size * 2  # keys + values
            total_bytes += recent_size

        # Warm layer (quantized) - use quantizer's estimate
        if self.warm_keys is not None:
            if self.enable_warm_quant:
                B, n_heads, seq_len, _ = self.warm_keys.shape
                # Get real head_dim from recent (packed quantizers change last dim)
                if self.recent_keys is not None:
                    head_dim = self.recent_keys.shape[-1]
                elif self.warm_metadata and 'head_dim' in self.warm_metadata[0]:
                    head_dim = self.warm_metadata[0]['head_dim']
                else:
                    head_dim = self.warm_keys.shape[-1]  # fallback (Q4_0 keeps original dim)
                warm_bytes = self.warm_quantizer.estimate_memory(
                    num_tokens=seq_len,
                    head_dim=head_dim,
                    num_heads=n_heads
                )
                total_bytes += warm_bytes
            else:
                # Not quantized
                dtype_size = 4 if self.warm_keys.dtype == mx.float32 else 2
                warm_size = self.warm_keys.size * dtype_size * 2
                total_bytes += warm_size

        # Cold layer (compressed + pending)
        if self.cold_compressed_keys is not None:
            total_bytes += self.cold_compressed_keys.nbytes + self.cold_compressed_values.nbytes
        if self.cold_pending_keys is not None:
            total_bytes += self.cold_pending_keys.nbytes + self.cold_pending_values.nbytes
            # Include quantization metadata (scales)
            if self.enable_cold_quant:
                for meta in self.cold_pending_metadata:
                    if 'scales_k' in meta:
                        total_bytes += meta['scales_k'].nbytes
                    if 'scales_v' in meta:
                        total_bytes += meta['scales_v'].nbytes

        return total_bytes

    def _manage_aging(self):
        """
        Manage token aging: move tokens between layers.

        Flow:
            Recent overflow → Warm (with quantization)
            Warm overflow → Cold (with AM compression)
        """
        # Recent → Warm (if Recent overflows)
        if self.recent_keys is not None:
            recent_len = self.recent_keys.shape[2]
            if recent_len > self.recent_size:
                overflow = recent_len - self.recent_size

                # Extract overflow tokens (oldest from Recent)
                overflow_keys = self.recent_keys[:, :, :overflow, :]
                overflow_values = self.recent_values[:, :, :overflow, :]

                # Remove from Recent
                self.recent_keys = self.recent_keys[:, :, overflow:, :]
                self.recent_values = self.recent_values[:, :, overflow:, :]

                # Add to Warm (with quantization)
                self._append_warm_with_quant(overflow_keys, overflow_values)

        # Warm → Cold (batch threshold: wait until overflow >= threshold)
        if self.warm_keys is not None:
            warm_len = self.warm_keys.shape[2]
            if warm_len > self.warm_size + self.warm_overflow_threshold:
                overflow = warm_len - self.warm_size

                # Choose eviction strategy based on quantizer capability
                if self.enable_warm_quant and len(self.warm_metadata) > 0 \
                        and self.warm_quantizer.requires_chunk_eviction:
                    # CHUNK-AWARE EVICTION: Pop whole quantized chunks.
                    # Zero re-quantization — each chunk quantized once (entering warm),
                    # dequantized once (leaving warm to cold). Preserves QJL quality.
                    overflow_keys_dequant, overflow_values_dequant = \
                        self._evict_warm_chunks(overflow)

                elif self.enable_warm_quant and len(self.warm_metadata) > 0:
                    # DEQUANT-SPLIT-REQUANTIZE: Dequantize all, split, re-quantize
                    # remaining. Works well for Q4_0/PolarQuant (cos stays >0.995).
                    overflow_keys_dequant, overflow_values_dequant = \
                        self._evict_warm_requantize(overflow)

                else:
                    # Not quantized — simple slice
                    overflow_keys_dequant = self.warm_keys[:, :, :overflow, :]
                    overflow_values_dequant = self.warm_values[:, :, :overflow, :]
                    self.warm_keys = self.warm_keys[:, :, overflow:, :]
                    self.warm_values = self.warm_values[:, :, overflow:, :]

                # Add to Cold (with AM compression or direct storage)
                overflow_len = overflow_keys_dequant.shape[2]
                if self.enable_cold_am and overflow_len > self.cold_batch_threshold:
                    for off in range(0, overflow_len, self.cold_batch_threshold):
                        chunk_end = min(off + self.cold_batch_threshold, overflow_len)
                        chunk_keys = overflow_keys_dequant[:, :, off:chunk_end, :]
                        chunk_values = overflow_values_dequant[:, :, off:chunk_end, :]
                        self._append_cold_with_am(chunk_keys, chunk_values)
                else:
                    self._append_cold_with_am(overflow_keys_dequant, overflow_values_dequant)

    def _evict_warm_chunks(self, overflow: int):
        """
        Chunk-aware eviction: pop whole quantized chunks from warm.

        Zero re-quantization path for TurboQuant and similar quantizers
        where re-quantization causes error amplification.

        Each chunk was quantized ONCE when entering warm. Here we dequantize
        only the evicted chunks and leave remaining chunks untouched.

        If overflow doesn't align to chunk boundaries, we evict the minimum
        number of whole chunks that cover >= overflow tokens.

        Returns (overflow_keys_dequant, overflow_values_dequant) in bf16.
        """
        # Get sequence length for each chunk from metadata
        chunk_seq_lens = self._get_chunk_seq_lens()

        # Find minimum number of whole chunks to cover >= overflow tokens
        evict_tokens = 0
        evict_count = 0
        for seq_len in chunk_seq_lens:
            evict_tokens += seq_len
            evict_count += 1
            if evict_tokens >= overflow:
                break

        # Dequantize only the evicted chunks
        dequant_keys_list = []
        dequant_values_list = []
        offset = 0

        for i in range(evict_count):
            seq_len = chunk_seq_lens[i]
            meta = self.warm_metadata[i]
            chunk_keys = self.warm_keys[:, :, offset:offset+seq_len, :]
            chunk_values = self.warm_values[:, :, offset:offset+seq_len, :]

            dequant_k, dequant_v = self.warm_quantizer.dequantize(
                chunk_keys, chunk_values, meta
            )
            dequant_keys_list.append(dequant_k)
            dequant_values_list.append(dequant_v)
            offset += seq_len

        # Concatenate evicted chunks
        overflow_keys = mx.concatenate(dequant_keys_list, axis=2)
        overflow_values = mx.concatenate(dequant_values_list, axis=2)

        # Keep remaining chunks in warm UNTOUCHED (no re-quantization!)
        if evict_count < len(self.warm_metadata):
            self.warm_keys = self.warm_keys[:, :, offset:, :]
            self.warm_values = self.warm_values[:, :, offset:, :]
            self.warm_metadata = self.warm_metadata[evict_count:]
        else:
            self.warm_keys = None
            self.warm_values = None
            self.warm_metadata = []

        return overflow_keys, overflow_values

    def _evict_warm_requantize(self, overflow: int):
        """
        Dequant-split-requantize eviction for Q4_0/PolarQuant.

        Dequantizes all warm chunks, extracts overflow tokens,
        and re-quantizes remaining as a single chunk.
        Works well for quantizers where re-quantization is stable
        (cos stays >0.995 after multiple cycles).

        Returns (overflow_keys_dequant, overflow_values_dequant) in bf16.
        """
        chunk_seq_lens = self._get_chunk_seq_lens()

        # Dequantize each chunk separately
        dequant_keys_list = []
        dequant_values_list = []
        offset = 0

        for seq_len, meta in zip(chunk_seq_lens, self.warm_metadata):
            chunk_keys = self.warm_keys[:, :, offset:offset+seq_len, :]
            chunk_values = self.warm_values[:, :, offset:offset+seq_len, :]

            dequant_k, dequant_v = self.warm_quantizer.dequantize(
                chunk_keys, chunk_values, meta
            )
            dequant_keys_list.append(dequant_k)
            dequant_values_list.append(dequant_v)
            offset += seq_len

        # Concatenate all dequantized chunks
        full_warm_keys = mx.concatenate(dequant_keys_list, axis=2)
        full_warm_values = mx.concatenate(dequant_values_list, axis=2)

        # Extract overflow from dequantized
        overflow_keys_dequant = full_warm_keys[:, :, :overflow, :]
        overflow_values_dequant = full_warm_values[:, :, :overflow, :]

        # Re-quantize remaining tokens as a single chunk
        remaining_keys = full_warm_keys[:, :, overflow:, :]
        remaining_values = full_warm_values[:, :, overflow:, :]

        if remaining_keys.shape[2] > 0:
            quant_k, quant_v, new_meta = self.warm_quantizer.requantize(
                remaining_keys, remaining_values
            )
            self.warm_keys = quant_k
            self.warm_values = quant_v
            self.warm_metadata = [new_meta]
        else:
            self.warm_keys = None
            self.warm_values = None
            self.warm_metadata = []

        return overflow_keys_dequant, overflow_values_dequant

    def _get_chunk_seq_lens(self):
        """Get sequence length for each quantized chunk in warm."""
        B, n_heads, _, _ = self.warm_keys.shape
        chunk_seq_lens = []
        for meta in self.warm_metadata:
            if 'seq_len' in meta:
                chunk_seq_lens.append(meta['seq_len'])
            else:
                group_size = meta['group_size']
                num_groups = meta['scales_k'].shape[0]
                total_elements = num_groups * group_size
                # Get real head_dim from recent_keys or metadata
                if self.recent_keys is not None:
                    head_dim = self.recent_keys.shape[-1]
                elif 'head_dim' in meta:
                    head_dim = meta['head_dim']
                else:
                    head_dim = self.warm_keys.shape[-1]
                chunk_tokens = total_elements // (B * n_heads * head_dim)
                chunk_seq_lens.append(chunk_tokens)
        return chunk_seq_lens

    # Max tokens per quantization chunk for chunk-aware eviction.
    # Smaller = finer eviction granularity, but more metadata overhead.
    # 256 tokens balances granularity vs overhead (at 8 heads * 128 dim = 128KB/chunk bf16).
    CHUNK_EVICT_SIZE = 256

    def _append_warm_with_quant(self, keys: mx.array, values: mx.array):
        """
        Append tokens to Warm layer with pluggable quantization.

        For quantizers that require chunk-aware eviction (e.g. TurboQuant),
        splits large inputs into fixed-size chunks so that eviction has
        fine-grained control. Each chunk is independently quantized.
        """
        if self.enable_warm_quant:
            seq_len = keys.shape[2]

            # Split into smaller chunks for chunk-aware eviction quantizers
            if self.warm_quantizer.requires_chunk_eviction and seq_len > self.CHUNK_EVICT_SIZE:
                for off in range(0, seq_len, self.CHUNK_EVICT_SIZE):
                    end = min(off + self.CHUNK_EVICT_SIZE, seq_len)
                    self._append_warm_single_chunk(
                        keys[:, :, off:end, :],
                        values[:, :, off:end, :]
                    )
            else:
                self._append_warm_single_chunk(keys, values)

            self.num_warm_compressions += 1
        else:
            # No quantization (store exact)
            if self.warm_keys is None:
                self.warm_keys = keys
                self.warm_values = values
                self.warm_metadata = []
            else:
                self.warm_keys = mx.concatenate([self.warm_keys, keys], axis=2)
                self.warm_values = mx.concatenate([self.warm_values, values], axis=2)

    def _append_warm_single_chunk(self, keys: mx.array, values: mx.array):
        """Quantize and append a single chunk to warm."""
        quant_keys, quant_values, metadata = self.warm_quantizer.quantize(keys, values)

        if self.warm_keys is None:
            self.warm_keys = quant_keys
            self.warm_values = quant_values
            self.warm_metadata = [metadata]
        else:
            self.warm_keys = mx.concatenate([self.warm_keys, quant_keys], axis=2)
            self.warm_values = mx.concatenate([self.warm_values, quant_values], axis=2)
            self.warm_metadata.append(metadata)

    def _append_cold_with_am(self, keys: mx.array, values: mx.array):
        """
        Append tokens to Cold layer with batch AM compression.

        Strategy:
        1. Accumulate tokens in cold_pending
        2. When pending >= threshold (512), compress entire pending batch
        3. Merge compressed result into cold_compressed
        """
        # AM compression is applied at promotion time, NOT during prefill.
        # During prefill, Cold always uses Q4_0 for memory savings.
        if True:
            # No AM compression
            if self.enable_cold_quant:
                # Quantize Cold for memory savings.
                # Use requantize() instead of quantize() so that TQ falls back
                # to PQ-only (avoids double QJL noise on already-dequantized data).
                keys_bf16 = keys.astype(mx.bfloat16) if keys.dtype == mx.float32 else keys
                values_bf16 = values.astype(mx.bfloat16) if values.dtype == mx.float32 else values
                quant_keys, quant_values, metadata = self.warm_quantizer.requantize(keys_bf16, values_bf16)

                if self.cold_pending_keys is None:
                    self.cold_pending_keys = quant_keys
                    self.cold_pending_values = quant_values
                    self.cold_pending_metadata = [metadata]
                else:
                    self.cold_pending_keys = mx.concatenate([self.cold_pending_keys, quant_keys], axis=2)
                    self.cold_pending_values = mx.concatenate([self.cold_pending_values, quant_values], axis=2)
                    self.cold_pending_metadata.append(metadata)
            else:
                # Store exact (cast to bfloat16 to avoid float32 bloat)
                keys_bf16 = keys.astype(mx.bfloat16) if keys.dtype == mx.float32 else keys
                values_bf16 = values.astype(mx.bfloat16) if values.dtype == mx.float32 else values

                if self.cold_pending_keys is None:
                    self.cold_pending_keys = keys_bf16
                    self.cold_pending_values = values_bf16
                else:
                    self.cold_pending_keys = mx.concatenate([self.cold_pending_keys, keys_bf16], axis=2)
                    self.cold_pending_values = mx.concatenate([self.cold_pending_values, values_bf16], axis=2)

    def _compress_with_am(
        self,
        keys: mx.array,
        values: mx.array,
        target_len: int,
        prefix_len: int
    ) -> Tuple[mx.array, mx.array]:
        """
        Compress using AM (Attention Matching) algorithm.

        Supports two calibration sources:
        1. Direct calibration (self._am_calibration) - loaded from calibration_file
        2. CalibrationRegistry (self.calibration_registry) - loaded from calibration_dir
        """
        # Get layer calibration from either source
        layer_calib = None
        if self._am_calibration is not None:
            layer_calib = self._am_calibration.get(self.layer_idx)
        elif self.calibration_registry is not None:
            calib_file = self.calibration_registry.get_calibration(
                length=prefix_len,
                ratio=self.compression_ratio,
                strategy=self.selection_strategy
            )
            if calib_file is not None:
                layer_calib = calib_file['calibration'][self.layer_idx]

        if layer_calib is None:
            return keys, values

        selected_indices = layer_calib['selected_indices']
        actual_len = keys.shape[2]

        if isinstance(selected_indices, mx.array):
            selected_indices = np.array(selected_indices)

        # Dynamic clipping: only keep indices < actual_len
        valid_mask = selected_indices < actual_len
        clipped_indices = selected_indices[valid_mask]

        if self.layer_idx == 0 and self.num_cold_compressions == 0:
            print(f"[AM] {actual_len} → {min(len(clipped_indices), target_len)} tokens "
                  f"(indices range: [0, {clipped_indices[-1] if len(clipped_indices) > 0 else 'N/A'}])")

        if len(clipped_indices) > target_len:
            clipped_indices = clipped_indices[:target_len]
        elif len(clipped_indices) == 0:
            return keys, values

        indices = mx.array(clipped_indices, dtype=mx.int32)
        compressed_keys = keys[:, :, indices, :]
        compressed_values = values[:, :, indices, :]

        return compressed_keys, compressed_values

    # ── Architecture D: Scored Differential Compression ──────────────────

    def _get_importance_mask(self, chunk_len: int, budget: int, prefix_len: int,
                             surprise_scores: "np.ndarray | None" = None,
                             attention_scores: "np.ndarray | None" = None) -> np.ndarray:
        """
        Get boolean importance mask from AM calibration + surprise/attention protection.

        Priority order:
        1. attention_scores (H2O-style, from H0Probe) — if available, top-K wins
        2. surprise_scores (key-norm z-scores) — merged with AM calibration
        3. AM calibration only — position-based fallback

        Returns:
            mask: np.ndarray of shape (chunk_len,) where True = important (PQ4)
        """
        # Hybrid path: attention ranking + surprise protection.
        # When BOTH are available, attention provides base ranking (top-K)
        # and surprise provides protection overlay (z >= 2.0 tokens are kept).
        if (attention_scores is not None and len(attention_scores) >= chunk_len
                and surprise_scores is not None):
            surprise_threshold = 2.0
            surprise_mask = surprise_scores >= surprise_threshold
            n_surprise = int(surprise_mask.sum())

            if 0 < n_surprise <= budget:
                # Surprise tokens get guaranteed slots; fill rest with attention top-K
                remaining = budget - n_surprise
                attn_copy = attention_scores[:chunk_len].copy()
                attn_copy[surprise_mask] = -np.inf  # don't double-count
                attn_top = np.argsort(-attn_copy)[:remaining]
                importance_mask = surprise_mask.copy()
                importance_mask[attn_top] = True
                return importance_mask
            elif n_surprise > budget:
                # Even surprise alone exceeds budget; take top-K by z-score
                top_idx = np.argsort(-surprise_scores)[:budget]
                importance_mask = np.zeros(chunk_len, dtype=bool)
                importance_mask[top_idx] = True
                return importance_mask
            # n_surprise == 0: no surprising tokens, fall through to pure attention

        # Pure attention path (no surprise tokens, or surprise not available).
        if attention_scores is not None and len(attention_scores) >= chunk_len:
            top_k_idx = np.argsort(-attention_scores[:chunk_len])[:budget]
            importance_mask = np.zeros(chunk_len, dtype=bool)
            importance_mask[top_k_idx] = True
            return importance_mask
        layer_calib = None
        if self._am_calibration is not None:
            layer_calib = self._am_calibration.get(self.layer_idx)
        elif self.calibration_registry is not None:
            calib_file = self.calibration_registry.get_calibration(
                length=prefix_len,
                ratio=self.compression_ratio,
                strategy=self.selection_strategy
            )
            if calib_file is not None:
                layer_calib = calib_file['calibration'][self.layer_idx]

        if layer_calib is None:
            return np.ones(chunk_len, dtype=bool)

        # Use ranked_indices (full ranking) if available, else selected_indices
        if 'ranked_indices' in layer_calib:
            ranked = layer_calib['ranked_indices']
            if isinstance(ranked, mx.array):
                ranked = np.array(ranked)
            valid = ranked[ranked < chunk_len]
            clipped_indices = valid[:budget]
        else:
            selected_indices = layer_calib['selected_indices']
            if isinstance(selected_indices, mx.array):
                selected_indices = np.array(selected_indices)

            valid_mask = selected_indices < chunk_len
            clipped_indices = selected_indices[valid_mask]
            if len(clipped_indices) > budget:
                clipped_indices = clipped_indices[:budget]
            elif len(clipped_indices) < budget:
                # Budget exceeds calibration capacity (ratio < calibration ratio).
                # Fill remaining slots with evenly-spaced unselected indices.
                all_indices = np.arange(chunk_len)
                unselected = np.setdiff1d(all_indices, clipped_indices)
                extra_needed = budget - len(clipped_indices)
                if extra_needed <= len(unselected):
                    # Evenly spaced: picks from across the chunk, not random
                    step = max(1, len(unselected) // extra_needed)
                    extra = unselected[::step][:extra_needed]
                    clipped_indices = np.sort(np.concatenate([clipped_indices, extra]))

        importance_mask = np.zeros(chunk_len, dtype=bool)
        importance_mask[clipped_indices] = True

        # Surprise protection: keep tokens with unusual key norms (z-score >= 2.0).
        # RoPE preserves L2 norms, so key norms reflect token content, not position.
        if surprise_scores is not None:
            surprise_threshold = 2.0
            surprise_mask = surprise_scores >= surprise_threshold
            n_surprise = int(surprise_mask.sum())
            if n_surprise > 0:
                combined = importance_mask | surprise_mask
                n_combined = int(combined.sum())
                if n_combined <= budget:
                    importance_mask = combined
                elif n_surprise <= budget:
                    # Surprise fits in budget; fill rest with AM picks
                    remaining = budget - n_surprise
                    am_only = importance_mask & ~surprise_mask
                    am_idx = np.where(am_only)[0][:remaining]
                    importance_mask = surprise_mask.copy()
                    importance_mask[am_idx] = True
                else:
                    # Even surprise alone exceeds budget; take top-k by z-score
                    top_idx = np.argsort(-surprise_scores)[:budget]
                    importance_mask = np.zeros(chunk_len, dtype=bool)
                    importance_mask[top_idx] = True

        return importance_mask

    def _scored_prefill_evict(self):
        """Score and evict during prefill to bound KV cache memory.

        When KV cache exceeds scored_prefill_max_cache during chunked prefill,
        AM-score accumulated tokens and keep only hot ones. This bounds PP peak
        memory to ~max_cache tokens instead of growing to full context length.

        Pinned tokens (first self.pinned_tokens) are never evicted.
        """
        keys = self.recent_keys
        values = self.recent_values
        B, n_heads, total_len, head_dim = keys.shape

        # Split: [old_prefix | recent_window]
        recent_window = min(self.recent_size, total_len)
        old_len = total_len - recent_window

        if old_len <= 0:
            return

        # Split old into [pinned | scorable]
        pin_n = min(self.pinned_tokens, old_len)
        scorable_start = pin_n
        scorable_len = old_len - pin_n

        pinned_k = keys[:, :, :pin_n, :] if pin_n > 0 else None
        pinned_v = values[:, :, :pin_n, :] if pin_n > 0 else None
        scorable_k = keys[:, :, scorable_start:old_len, :]
        scorable_v = values[:, :, scorable_start:old_len, :]
        recent_k = keys[:, :, old_len:, :]
        recent_v = values[:, :, old_len:, :]

        # AM scoring on scorable tokens only (pinned are unconditionally kept)
        effective_ratio = self._get_effective_ratio(self._prefill_tokens_seen)
        chunk_size = self.cold_batch_threshold  # 512
        budget = int(chunk_size / effective_ratio)

        # Attention probe scoring: if probe is available and enabled, use
        # H2O-style cumulative attention scores instead of key-norm surprise.
        # Layer 0 runs the probe once, shares results with all layers.
        attn_importance = None
        evict_tag = (self._prefill_eviction_count + 1, scorable_len)
        if self._probe_eviction_enabled and scorable_len > 0:
            if self.layer_idx == 0:
                probe = TripleLayerKVCache._shared_probe
                if probe is not None and self._h0_store is not None:
                    attn_scores = probe.score_tokens(self._h0_store)
                    # Slice to scorable range (skip pinned prefix)
                    if len(attn_scores) >= scorable_start + scorable_len:
                        attn_importance = attn_scores[scorable_start:scorable_start + scorable_len]
                    else:
                        attn_importance = attn_scores[:scorable_len]
                    TripleLayerKVCache._shared_attn_importance = attn_importance
                    TripleLayerKVCache._shared_attn_importance_tag = evict_tag
                    if self.num_cold_compressions == 0:
                        print(f"[Probe] Scored {len(attn_importance)} tokens via "
                              f"{probe._n_layers}-layer attention probe")
            elif (TripleLayerKVCache._shared_attn_importance_tag == evict_tag and
                  TripleLayerKVCache._shared_attn_importance is not None):
                attn_importance = TripleLayerKVCache._shared_attn_importance

        # Cross-layer key-norm surprise scoring.
        # Always computed (even when probe available) for hybrid eviction.
        surprise_all = None
        if scorable_len > 0:
            if self.layer_idx == 0:
                kn = mx.linalg.norm(scorable_k.astype(mx.float32), axis=-1)
                kn = kn.mean(axis=1).squeeze(0)
                kn_np = np.array(kn)
                per_chunk_z = np.zeros(scorable_len, dtype=np.float32)
                for ci in range(0, scorable_len, chunk_size):
                    ce = min(ci + chunk_size, scorable_len)
                    chunk = kn_np[ci:ce]
                    c_mean, c_std = chunk.mean(), chunk.std() + 1e-8
                    per_chunk_z[ci:ce] = np.abs(chunk - c_mean) / c_std
                half_w = 16
                padded = np.pad(per_chunk_z, (half_w, half_w), constant_values=0)
                from numpy.lib.stride_tricks import sliding_window_view
                surprise_all = sliding_window_view(padded, 2 * half_w + 1).max(axis=-1)
                TripleLayerKVCache._shared_surprise = surprise_all
                TripleLayerKVCache._shared_surprise_tag = evict_tag
            elif (TripleLayerKVCache._shared_surprise_tag == evict_tag and
                  TripleLayerKVCache._shared_surprise is not None):
                surprise_all = TripleLayerKVCache._shared_surprise

        global_indices = []
        for offset in range(0, scorable_len, chunk_size):
            chunk_end = min(offset + chunk_size, scorable_len)
            chunk_len = chunk_end - offset
            if chunk_len == chunk_size:
                chunk_surprise = surprise_all[offset:chunk_end] if surprise_all is not None else None
                chunk_attn = attn_importance[offset:chunk_end] if attn_importance is not None else None
                imp_mask = self._get_importance_mask(chunk_len, budget, chunk_size,
                                                    surprise_scores=chunk_surprise,
                                                    attention_scores=chunk_attn)
                global_indices.append(np.where(imp_mask)[0] + offset)
            else:
                # Partial chunk: keep all tokens
                global_indices.append(np.arange(offset, chunk_end))

        parts = []
        if pin_n > 0:
            parts.append(pinned_k)

        if global_indices:
            all_indices = np.concatenate(global_indices)
            idx = mx.array(all_indices, dtype=mx.int32)
            parts.append(scorable_k[:, :, idx, :])
            n_hot = len(all_indices)
        else:
            all_indices = np.array([], dtype=np.int32)
            n_hot = 0

        parts.append(recent_k)

        parts_v = []
        if pin_n > 0:
            parts_v.append(pinned_v)
        if n_hot > 0:
            parts_v.append(scorable_v[:, :, idx, :])
        parts_v.append(recent_v)

        # Replace recent with [pinned | hot | recent_window]
        self.recent_keys = mx.concatenate(parts, axis=2)
        self.recent_values = mx.concatenate(parts_v, axis=2)
        mx.eval(self.recent_keys, self.recent_values)

        self._prefill_eviction_count += 1
        if self.layer_idx == 0:
            pin_info = f"{pin_n} pinned + " if pin_n > 0 else ""
            print(f"[Scored Prefill Evict #{self._prefill_eviction_count}] "
                  f"{total_len} → {self.recent_keys.shape[2]} tokens "
                  f"(kept {pin_info}{n_hot} hot + {recent_window} recent, "
                  f"ratio={effective_ratio}x)")

    def _promote_to_flat_buffer(self, keys, values):
        """Copy bf16 buffer to pre-allocated flat buffer (no AM scoring).

        Used after chunked prefill eviction: cache is already scored/pruned,
        just needs to transition to flat mode for O(1) TG appends.
        """
        B, n_heads, cache_len, head_dim = keys.shape
        alloc_len = ((cache_len + self._flat_step - 1) // self._flat_step + 1) * self._flat_step
        self._alloc_flat_buffer(B, n_heads, alloc_len, head_dim)
        self._write_flat(0, cache_len, keys, values)
        self._flat_offset = cache_len
        self._flat_mode = True
        # After chunked eviction, all prefix tokens are already scored
        recent_len = min(self.recent_size, cache_len)
        self._flat_prefix_token_count = cache_len - recent_len
        self.recent_keys = None
        self.recent_values = None
        self._eval_flat_buffer()

    def _auto_reconstruct_if_enabled(self):
        """Trigger h^(0) reconstruction after prefill, if enabled.

        Called by layer 0 only, after flat mode transition. Reconstructs
        evicted tokens from h^(0) and injects K/V into ALL layers' caches.
        """
        if not self._auto_reconstruct:
            return
        if self._h0_store is None or self._h0_store.count <= 0:
            return

        inner_model = TripleLayerKVCache._shared_inner_model
        if inner_model is None:
            return

        import time as _time
        n_evicted = self._h0_store.count

        # Use probe importance scores for targeted reconstruction if available
        importance_scores = None
        probe = TripleLayerKVCache._shared_probe
        if probe is not None:
            importance_scores = probe.score_tokens(self._h0_store)

        mode = "targeted" if importance_scores is not None else "full"
        print(f"[AutoRecon] Reconstructing {n_evicted} tokens ({mode}) from h^(0)...")
        t0 = _time.perf_counter()

        from mlx_lm.models.kv_direct_cache import _run_reconstruction

        caches = TripleLayerKVCache._shared_cache_list
        if caches is None:
            return

        kv_direct_indices = list(range(len(caches)))
        _run_reconstruction(inner_model, caches, self._h0_store, n_evicted,
                            kv_direct_indices, importance_scores=importance_scores)

        # Eval recon arrays and update prefix counts
        recon_arrays = []
        for c in caches:
            if getattr(c, '_recon_keys', None) is not None:
                recon_arrays.extend([c._recon_keys, c._recon_values])
        if recon_arrays:
            mx.eval(*recon_arrays)
        for c in caches:
            if getattr(c, '_recon_keys', None) is not None:
                c._flat_prefix_token_count = max(
                    getattr(c, '_flat_prefix_token_count', 0), n_evicted)

        recon_ms = (_time.perf_counter() - t0) * 1000
        print(f"[AutoRecon] Done in {recon_ms:.0f}ms — injected into "
              f"{len(recon_arrays) // 2} layers")

    def _scored_compress_prefix(
        self,
        keys: mx.array,
        values: mx.array,
        recent_len: int
    ):
        """
        Architecture D (P2): AM score on clean bf16 → hot/drop split.

        Scored-as-Warm-tier architecture:
        - Pinned tokens (system prompt) → flat buffer bf16 (always kept)
        - Important tokens (AM selected) → flat buffer bf16 (HOT attention path)
        - Unimportant tokens → dropped (same as Pipeline)
        - Recent tokens → flat buffer bf16 (exact)

        Key insight: flat buffer only has ~50% of old tokens (important + recent).
        This gives pipeline-equivalent TG speed (~8K tokens → ~23.5 tok/s).

        Advantage over Pipeline: AM scores on clean bf16 data (not degraded PQ4).
        NO PQ roundtrip — important tokens go bf16→flat directly.
        """
        B, n_heads, total_len, head_dim = keys.shape
        old_len = total_len - recent_len

        self._true_offset = total_len

        # Split: [pinned | scorable | recent]
        pin_n = min(self.pinned_tokens, old_len)
        scorable_start = pin_n
        scorable_len = old_len - pin_n

        pinned_k = keys[:, :, :pin_n, :] if pin_n > 0 else None
        pinned_v = values[:, :, :pin_n, :] if pin_n > 0 else None
        scorable_k = keys[:, :, scorable_start:old_len, :]
        scorable_v = values[:, :, scorable_start:old_len, :]
        recent_k = keys[:, :, old_len:, :]
        recent_v = values[:, :, old_len:, :]

        # Process scorable region: build global importance index (vectorized)
        effective_ratio = self._get_effective_ratio(total_len)
        chunk_size = self.cold_batch_threshold  # 512
        budget = int(chunk_size / effective_ratio)

        # Phase 1: Build global index array (numpy, cheap)
        global_indices = []
        importance_masks = []  # Route 0: collect for density signal
        n_full_chunks = 0

        for offset in range(0, scorable_len, chunk_size):
            chunk_end = min(offset + chunk_size, scorable_len)
            chunk_len = chunk_end - offset

            if chunk_len == chunk_size:
                imp_mask = self._get_importance_mask(chunk_len, budget, chunk_size)
                global_indices.append(np.where(imp_mask)[0] + offset)
                importance_masks.append(imp_mask)
                n_full_chunks += 1
            else:
                # Partial chunk: keep all tokens
                global_indices.append(np.arange(offset, chunk_end))

        if global_indices:
            all_indices = np.concatenate(global_indices)
        else:
            all_indices = np.array([], dtype=np.int32)
        n_hot_tokens = len(all_indices)
        n_dropped_tokens = scorable_len - n_hot_tokens

        # Route 0: Extract density signal (zero cost — numpy ops on existing masks)
        self._density_signal = self._extract_density_signal(
            importance_masks, total_len, scorable_len, n_hot_tokens, n_dropped_tokens,
        )

        # Phase 2: Allocate flat buffer and write: [pinned | hot_scored | recent]
        cache_len = pin_n + n_hot_tokens + recent_len

        if self.layer_idx == 0:
            ratio_info = f" (adaptive→{effective_ratio}x)" if self.adaptive_ratio else ""
            pin_info = f"{pin_n} pinned + " if pin_n > 0 else ""
            density_info = f" density={self._density_signal['keep_ratio']:.2f}"
            print(f"[Scored P2{ratio_info}] {old_len} old + {recent_len} recent → "
                  f"{pin_info}{n_hot_tokens} hot(bf16) + {n_dropped_tokens} dropped + {recent_len} recent "
                  f"({n_full_chunks} chunks scored{density_info})")

        alloc_len = ((cache_len + self._flat_step - 1) // self._flat_step + 1) * self._flat_step
        self._alloc_flat_buffer(B, n_heads, alloc_len, head_dim)

        write_offset = 0
        # Write pinned tokens first (unconditionally kept)
        if pin_n > 0:
            self._write_flat(0, pin_n, pinned_k, pinned_v)
            write_offset = pin_n

        # Write AM-scored hot tokens
        if n_hot_tokens > 0:
            global_idx = mx.array(all_indices, dtype=mx.int32)
            self._write_flat(write_offset, write_offset + n_hot_tokens,
                             scorable_k[:, :, global_idx, :], scorable_v[:, :, global_idx, :])
            write_offset += n_hot_tokens

        # Write recent tokens
        self._write_flat(write_offset, cache_len, recent_k, recent_v)
        self._flat_offset = cache_len
        self._flat_mode = True
        # Track prefix-era token count for recall dedup (pinned + hot scored)
        self._flat_prefix_token_count = pin_n + n_hot_tokens
        self._eval_flat_buffer()

    def _dequant_scored_chunks(
        self,
        quant_k: mx.array,
        quant_v: mx.array,
        meta_list: list,
        quantizer: 'QuantizationStrategy'
    ) -> Tuple[mx.array, mx.array]:
        """Dequantize multiple scored chunks using the given quantizer."""
        if len(meta_list) == 1:
            return quantizer.dequantize(quant_k, quant_v, meta_list[0])

        dk_list, dv_list = [], []
        offset = 0
        for meta in meta_list:
            seq_len = meta['seq_len']
            ck = quant_k[:, :, offset:offset + seq_len, :]
            cv = quant_v[:, :, offset:offset + seq_len, :]
            dk, dv = quantizer.dequantize(ck, cv, meta)
            dk_list.append(dk)
            dv_list.append(dv)
            offset += seq_len

        return mx.concatenate(dk_list, axis=2), mx.concatenate(dv_list, axis=2)

    # ── End Architecture D ───────────────────────────────────────────────

    def _quantize(self, x: mx.array, bits: int = 4) -> Tuple[mx.array, mx.array]:
        """
        Quantize tensor to lower bits using per-token quantization.

        Input shape: (B, num_heads, seq_len, head_dim)
        Quantize per-token (compute scale across head_dim for each token)

        Returns:
            quantized: quantized tensor (float16, not int8 for MLX compatibility)
            scales: quantization scales (float32)
        """
        # Per-token symmetric quantization
        # For each token, compute scale = max(abs(x)) across head_dim
        # Shape of x: (B, num_heads, seq_len, head_dim)

        # Compute max across head_dim (axis=-1) for each token
        max_val = mx.max(mx.abs(x), axis=-1, keepdims=True)  # (B, num_heads, seq_len, 1)

        # Compute scale
        qmax = 2 ** (bits - 1) - 1  # 7 for 4-bit
        scale = max_val / qmax
        scale = mx.maximum(scale, 1e-8)  # Avoid division by zero

        # Quantize
        quantized = x / scale  # Normalize
        quantized = mx.round(quantized)  # Round to nearest integer
        quantized = mx.clip(quantized, -qmax - 1, qmax)  # Clip to [-8, 7] for 4-bit

        # Store as float16 (MLX doesn't support int8 operations well)
        # The values are integers but stored as floats
        quantized = quantized.astype(mx.float16)

        return quantized, scale

    def _dequantize(self, quantized: mx.array, scales: mx.array) -> mx.array:
        """
        Dequantize tensor.

        Input:
            quantized: (B, num_heads, seq_len, head_dim) float16 with integer values
            scales: (B, num_heads, seq_len, 1) float32

        Dequantization: x = Q * scale
        """
        # Dequantize: multiply by scale and convert back to float32
        dequantized = quantized.astype(mx.float32) * scales
        return dequantized

    def _dequant_chunks(self, keys, values, metadata_list):
        """Dequantize multiple quantized chunks using per-chunk metadata."""
        dequant_keys_list = []
        dequant_values_list = []
        offset = 0

        for meta in metadata_list:
            seq_len = meta['seq_len']
            chunk_keys = keys[:, :, offset:offset+seq_len, :]
            chunk_values = values[:, :, offset:offset+seq_len, :]

            dequant_k, dequant_v = self.warm_quantizer.dequantize(
                chunk_keys, chunk_values, meta
            )
            dequant_keys_list.append(dequant_k)
            dequant_values_list.append(dequant_v)
            offset += seq_len

        return (
            mx.concatenate(dequant_keys_list, axis=2).astype(mx.bfloat16),
            mx.concatenate(dequant_values_list, axis=2).astype(mx.bfloat16)
        )

    def _merged_dequant(self, data, metadata_list):
        """
        Vectorized dequant: merge scales with head-interleaving, single matmul.

        When chunks are concatenated on axis=2 (seq), the flattened group order
        becomes [h0_c1, h0_c2, h1_c1, h1_c2, ...] per (batch, head).
        But naive scale concat gives [h0_c1, h1_c1, ..., h0_c2, h1_c2, ...].

        Fix: reshape each chunk's scales to (B*n_heads, groups_per_head),
        concat on axis=1, then flatten. This matches the concatenated tensor layout.
        """
        B, n_heads, total_seq, head_dim = data.shape
        group_size = metadata_list[0]['group_size']
        groups_per_token = head_dim // group_size  # = 4 for head_dim=128, group_size=32

        # Merge scales with correct head-interleaving
        scale_chunks = []
        for meta in metadata_list:
            seq_len = meta['seq_len']
            gph = seq_len * groups_per_token  # groups per (batch*head)
            # Reshape from flat (B*n_heads*gph,) to (B*n_heads, gph)
            scale_chunks.append(meta['scales_k'].reshape(B * n_heads, gph))

        # Concat within each (batch, head) → (B*n_heads, total_gph)
        merged_scales = mx.concatenate(scale_chunks, axis=1).reshape(-1)

        # Single vectorized dequant
        n_groups = merged_scales.shape[0]
        dequant = (data.reshape(n_groups, group_size).astype(mx.float32)
                   * merged_scales.reshape(n_groups, 1))
        return dequant.reshape(data.shape).astype(mx.bfloat16)

    def _generic_dequant_kv(self, keys, values, metadata_list):
        """Dequant keys and values using pluggable quantizer.

        Uses merged vectorized path for Q4_0, per-chunk for other quantizers.
        """
        # Fast path: Q4_0 with merged scales (vectorized, no Python loop)
        if metadata_list and 'group_size' in metadata_list[0]:
            return self._merged_dequant_kv(keys, values, metadata_list)

        # Generic path: per-chunk dequantization via quantizer interface
        dequant_keys_list = []
        dequant_values_list = []
        offset = 0

        for meta in metadata_list:
            seq_len = meta['seq_len']
            chunk_keys = keys[:, :, offset:offset+seq_len, :]
            chunk_values = values[:, :, offset:offset+seq_len, :]
            dk, dv = self.warm_quantizer.dequantize(
                chunk_keys, chunk_values, meta
            )
            dequant_keys_list.append(dk)
            dequant_values_list.append(dv)
            offset += seq_len

        if len(dequant_keys_list) == 1:
            return dequant_keys_list[0], dequant_values_list[0]
        return (mx.concatenate(dequant_keys_list, axis=2),
                mx.concatenate(dequant_values_list, axis=2))

    def _merged_dequant_kv(self, keys, values, metadata_list):
        """Dequant both keys and values using merged scales (Q4_0 fast path)."""
        B, n_heads, total_seq, head_dim = keys.shape
        group_size = metadata_list[0]['group_size']
        groups_per_token = head_dim // group_size

        # Build merged scales for K and V
        k_chunks = []
        v_chunks = []
        for meta in metadata_list:
            seq_len = meta['seq_len']
            gph = seq_len * groups_per_token
            k_chunks.append(meta['scales_k'].reshape(B * n_heads, gph))
            v_chunks.append(meta['scales_v'].reshape(B * n_heads, gph))

        merged_k = mx.concatenate(k_chunks, axis=1).reshape(-1)
        merged_v = mx.concatenate(v_chunks, axis=1).reshape(-1)

        n_groups = merged_k.shape[0]
        dk = (keys.reshape(n_groups, group_size).astype(mx.float32)
              * merged_k.reshape(n_groups, 1)).reshape(keys.shape).astype(mx.bfloat16)
        dv = (values.reshape(n_groups, group_size).astype(mx.float32)
              * merged_v.reshape(n_groups, 1)).reshape(values.shape).astype(mx.bfloat16)
        return dk, dv

    def _concat_all_layers(self) -> Tuple[mx.array, mx.array]:
        """
        Concatenate all layers: Cold + Warm + Recent.

        Uses merged vectorized dequant (no Python loop) for speed.

        Returns:
            keys, values: Full cache (bfloat16)
        """
        layers_k = []
        layers_v = []

        # L2 (Cold) - compressed + pending
        if self.cold_compressed_keys is not None:
            layers_k.append(self.cold_compressed_keys.astype(mx.bfloat16))
            layers_v.append(self.cold_compressed_values.astype(mx.bfloat16))
        if self.cold_pending_keys is not None:
            if self.enable_cold_quant and len(self.cold_pending_metadata) > 0:
                dk, dv = self._generic_dequant_kv(
                    self.cold_pending_keys, self.cold_pending_values,
                    self.cold_pending_metadata
                )
                layers_k.append(dk)
                layers_v.append(dv)
            else:
                layers_k.append(self.cold_pending_keys.astype(mx.bfloat16))
                layers_v.append(self.cold_pending_values.astype(mx.bfloat16))

        # L1 (Warm) - pluggable dequant (vectorized for Q4_0, per-chunk for others)
        if self.warm_keys is not None:
            if self.enable_warm_quant and len(self.warm_metadata) > 0:
                dk, dv = self._generic_dequant_kv(
                    self.warm_keys, self.warm_values, self.warm_metadata
                )
                layers_k.append(dk)
                layers_v.append(dv)
            else:
                layers_k.append(self.warm_keys.astype(mx.bfloat16))
                layers_v.append(self.warm_values.astype(mx.bfloat16))

        # L0 (Recent) - always exact bfloat16
        if self.recent_keys is not None:
            layers_k.append(self.recent_keys.astype(mx.bfloat16))
            layers_v.append(self.recent_values.astype(mx.bfloat16))

        # Concatenate
        if len(layers_k) == 0:
            return None, None

        full_keys = mx.concatenate(layers_k, axis=2) if len(layers_k) > 1 else layers_k[0]
        full_values = mx.concatenate(layers_v, axis=2) if len(layers_v) > 1 else layers_v[0]

        return full_keys, full_values

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            dict: Memory usage for each layer
        """
        def array_size_mb(arr):
            if arr is None:
                return 0.0
            return arr.nbytes / (1024 * 1024)

        cold_mb = array_size_mb(self.cold_keys) + array_size_mb(self.cold_values)
        warm_mb = (
            array_size_mb(self.warm_keys) + array_size_mb(self.warm_values) +
            array_size_mb(self.warm_scales_k) + array_size_mb(self.warm_scales_v)
        )
        recent_mb = array_size_mb(self.recent_keys) + array_size_mb(self.recent_values)

        return {
            "total_mb": cold_mb + warm_mb + recent_mb,
            "cold_mb": cold_mb,
            "warm_mb": warm_mb,
            "recent_mb": recent_mb,
            "cold_tokens": self.cold_keys.shape[2] if self.cold_keys is not None else 0,
            "warm_tokens": self.warm_keys.shape[2] if self.warm_keys is not None else 0,
            "recent_tokens": self.recent_keys.shape[2] if self.recent_keys is not None else 0,
            "num_warm_compressions": self.num_warm_compressions,
            "num_cold_compressions": self.num_cold_compressions
        }

    def __repr__(self):
        mem = self.get_memory_usage()
        return (
            f"TripleLayerKVCache(layer={self.layer_idx}, "
            f"recent={mem['recent_tokens']}, warm={mem['warm_tokens']}, cold={mem['cold_tokens']}, "
            f"total_mem={mem['total_mb']:.1f}MB)"
        )
