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
        lazy_prefill_threshold: int = 8192  # Skip Q4_0 during prefill for contexts below this
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
        self._flat_step = 256  # Pre-allocation chunk size (same as KVCache.step)
        self._needs_cleanup = False  # Deferred cleanup of quantized data
        self._true_offset = 0  # Original token count (before AM), used for RoPE

        # Statistics
        self.num_warm_compressions = 0
        self.num_cold_compressions = 0
        self._total_offset = 0

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
        layer0 = self._am_calibration[0]
        budget = layer0['budget']
        ratio = layer0['compression_ratio']
        self._am_prefix_len = int(budget * ratio)
        self.cold_batch_threshold = self._am_prefix_len

        if self.layer_idx == 0:
            print(f"[TripleLayerKVCache] AM calibration loaded: {filepath}")
            print(f"  Prefix: {self._am_prefix_len} → Budget: {budget} ({ratio}x)")

    @property
    def offset(self):
        """Total cache size (for RoPE positioning)."""
        if self._flat_mode:
            # Use original offset for correct RoPE, even if AM compressed the buffer
            return self._true_offset
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
        # FAST PATH: pre-allocated buffer — same strategy as Standard KVCache
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

            # Slice assignment — O(1), no allocation, no copy
            self._flat_offset += keys.shape[2]
            self._true_offset += keys.shape[2]
            self._flat_keys[..., prev:self._flat_offset, :] = keys
            self._flat_values[..., prev:self._flat_offset, :] = values
            return self._flat_keys[..., :self._flat_offset, :], self._flat_values[..., :self._flat_offset, :]

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

    def _update_slow_path(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """Triple-layer path: prefill + first TG token promotion."""
        # 1. Append to Recent
        if self.recent_keys is None:
            self.recent_keys = keys
            self.recent_values = values
        else:
            self.recent_keys = mx.concatenate([self.recent_keys, keys], axis=2)
            self.recent_values = mx.concatenate([self.recent_values, values], axis=2)

        # 2. LAZY PREFILL: during prefill (multi-token), skip Q4_0 quantization
        #    when context is small enough. Tokens stay in bf16 Recent for PP speed.
        #    Aging + quantization is deferred to promotion (first TG token).
        #    For long contexts (> threshold), do incremental aging to avoid TTFT spike.
        if keys.shape[2] > 1:
            if self.recent_keys.shape[2] <= self.lazy_prefill_threshold:
                return self.recent_keys, self.recent_values
            # Long context: do incremental aging during prefill
            if self.recent_keys.shape[2] > self.recent_size:
                self._manage_aging()
            return self._concat_all_layers()

        # 3. First TG token: reorganize accumulated tokens into layers
        #    This handles both lazy prefill (all in Recent) and normal aging
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

            # Stage 2: Apply AM compression to old prefix (non-recent tokens)
            if self.enable_cold_am and (self._am_calibration is not None or self.calibration_registry is not None):
                full_keys, full_values = self._am_compress_prefix(
                    full_keys, full_values, recent_len
                )
                mx.eval(full_keys, full_values)
                cache_len = full_keys.shape[2]  # Update to compressed length

            # Stage 3: Allocate flat buffer and copy (quantized data already freed)
            alloc_len = ((cache_len + self._flat_step - 1) // self._flat_step + 1) * self._flat_step
            self._flat_keys = mx.zeros((B, n_heads, alloc_len, head_dim), dtype=mx.bfloat16)
            self._flat_values = mx.zeros((B, n_heads, alloc_len, head_dim), dtype=mx.bfloat16)
            self._flat_keys[..., :cache_len, :] = full_keys
            self._flat_values[..., :cache_len, :] = full_values
            del full_keys, full_values  # Free intermediate bf16 (flat buffer owns the data now)
            self._flat_offset = cache_len
            self._flat_mode = True
            return self._flat_keys[..., :self._flat_offset, :], self._flat_values[..., :self._flat_offset, :]

        return full_keys, full_values

    def _get_effective_ratio(self, context_len: int) -> float:
        """Select optimal compression ratio based on context length.

        Data-driven mapping (Qwen3-8B benchmarks):
            <= 16K: 3.0x — better TG (+5-23%), +59-66% memory savings
            > 16K:  2.0x — stable TG, avoids 32K regression
        """
        if not self.adaptive_ratio:
            return self.compression_ratio
        if context_len <= 16384:
            return 3.0
        return 2.0

    def _am_compress_prefix(
        self,
        keys: mx.array,
        values: mx.array,
        recent_len: int
    ) -> Tuple[mx.array, mx.array]:
        """
        Apply AM compression to old prefix (non-recent tokens) at promotion time.

        Splits the full cache into [old_prefix | recent], compresses old_prefix
        in chunks of cold_batch_threshold (512), reassembles with recent.

        Only compresses FULL chunks. Partial remainder is kept uncompressed.
        """
        B, n_heads, total_len, head_dim = keys.shape
        old_len = total_len - recent_len

        if old_len <= 0:
            return keys, values

        # Split: [old_prefix | recent]
        old_keys = keys[:, :, :old_len, :]
        old_values = values[:, :, :old_len, :]
        recent_keys = keys[:, :, old_len:, :]
        recent_values = values[:, :, old_len:, :]

        # Compression parameters — adaptive ratio selects based on context length
        effective_ratio = self._get_effective_ratio(total_len)
        chunk_size = self.cold_batch_threshold  # 512
        budget = int(chunk_size / effective_ratio)

        compressed_k = []
        compressed_v = []
        n_full_chunks = 0

        for offset in range(0, old_len, chunk_size):
            chunk_end = min(offset + chunk_size, old_len)
            chunk_k = old_keys[:, :, offset:chunk_end, :]
            chunk_v = old_values[:, :, offset:chunk_end, :]
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

        # Reassemble: compressed_old + recent
        compressed_k.append(recent_keys)
        compressed_v.append(recent_values)

        result_keys = mx.concatenate(compressed_k, axis=2)
        result_values = mx.concatenate(compressed_v, axis=2)

        if self.layer_idx == 0:
            saved = total_len - result_keys.shape[2]
            ratio_info = f" (adaptive→{effective_ratio}x)" if self.adaptive_ratio else ""
            print(f"[AM Promotion{ratio_info}] {old_len} old + {recent_len} recent → "
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

        # Recent layer (fp32/fp16)
        if self.recent_keys is not None:
            dtype_size = 4 if self.recent_keys.dtype == mx.float32 else 2
            recent_size = self.recent_keys.size * dtype_size * 2  # keys + values
            total_bytes += recent_size

        # Warm layer (quantized) - use quantizer's estimate
        if self.warm_keys is not None:
            if self.enable_warm_quant:
                B, n_heads, seq_len, head_dim = self.warm_keys.shape
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

                # Extract overflow tokens (oldest from Warm)
                overflow_keys = self.warm_keys[:, :, :overflow, :]
                overflow_values = self.warm_values[:, :, :overflow, :]

                # Dequantize before moving to Cold using pluggable quantizer
                if self.enable_warm_quant and len(self.warm_metadata) > 0:
                    # 🔧 FIX: Dequantize each chunk individually using its original metadata
                    # then extract overflow. This is the CORRECT way to handle multi-chunk quantization.
                    # (The old approach of merging scales was WRONG - it broke the group structure)

                    B, n_heads, _, head_dim = self.warm_keys.shape

                    # Get sequence length for each chunk from metadata
                    chunk_seq_lens = []
                    for meta in self.warm_metadata:
                        if 'seq_len' in meta:
                            chunk_seq_lens.append(meta['seq_len'])
                        else:
                            # Fallback: infer from scales shape
                            group_size = meta['group_size']
                            num_groups = meta['scales_k'].shape[0]
                            total_elements = num_groups * group_size
                            chunk_tokens = total_elements // (B * n_heads * head_dim)
                            chunk_seq_lens.append(chunk_tokens)

                    # Dequantize each chunk separately
                    dequant_keys_list = []
                    dequant_values_list = []
                    offset = 0

                    for seq_len, meta in zip(chunk_seq_lens, self.warm_metadata):
                        # Extract this chunk from concatenated quantized data
                        chunk_keys = self.warm_keys[:, :, offset:offset+seq_len, :]
                        chunk_values = self.warm_values[:, :, offset:offset+seq_len, :]

                        # Dequantize using original metadata
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

                    # Keep remaining in Warm (need to re-quantize as a SINGLE chunk)
                    remaining_keys = full_warm_keys[:, :, overflow:, :]
                    remaining_values = full_warm_values[:, :, overflow:, :]

                    # Re-quantize remaining tokens as a single chunk
                    if remaining_keys.shape[2] > 0:
                        quant_remaining_k, quant_remaining_v, new_metadata = self.warm_quantizer.quantize(
                            remaining_keys, remaining_values
                        )
                        self.warm_keys = quant_remaining_k
                        self.warm_values = quant_remaining_v
                        self.warm_metadata = [new_metadata]  # Single chunk now
                    else:
                        self.warm_keys = None
                        self.warm_values = None
                        self.warm_metadata = []
                else:
                    # Not quantized
                    overflow_keys_dequant = overflow_keys
                    overflow_values_dequant = overflow_values

                    # Remove from Warm
                    self.warm_keys = self.warm_keys[:, :, overflow:, :]
                    self.warm_values = self.warm_values[:, :, overflow:, :]

                # Add to Cold (with AM compression or direct storage)
                # 🔧 CHUNKED OVERFLOW: Split large overflow into batch-sized chunks
                # to prevent accumulating more tokens than calibration can handle
                overflow_len = overflow_keys_dequant.shape[2]
                if self.enable_cold_am and overflow_len > self.cold_batch_threshold:
                    # Split into chunks
                    for offset in range(0, overflow_len, self.cold_batch_threshold):
                        chunk_end = min(offset + self.cold_batch_threshold, overflow_len)
                        chunk_keys = overflow_keys_dequant[:, :, offset:chunk_end, :]
                        chunk_values = overflow_values_dequant[:, :, offset:chunk_end, :]
                        self._append_cold_with_am(chunk_keys, chunk_values)
                else:
                    # Small overflow, append directly
                    self._append_cold_with_am(overflow_keys_dequant, overflow_values_dequant)

    def _append_warm_with_quant(self, keys: mx.array, values: mx.array):
        """
        Append tokens to Warm layer with pluggable quantization.

        Uses the configured warm_quantizer strategy (default: Q4_0).
        Supports multiple quantization algorithms:
        - Q4_0: 4-bit symmetric quantization (~2x)
        - TurboQuant: Adaptive bitwidth (~3x, TODO)
        - Custom: User-provided QuantizationStrategy
        """
        if self.enable_warm_quant:
            # Use pluggable quantization strategy
            quant_keys, quant_values, metadata = self.warm_quantizer.quantize(keys, values)

            # Append quantized
            if self.warm_keys is None:
                self.warm_keys = quant_keys
                self.warm_values = quant_values
                self.warm_metadata = [metadata]  # List of metadata dicts
            else:
                self.warm_keys = mx.concatenate([self.warm_keys, quant_keys], axis=2)
                self.warm_values = mx.concatenate([self.warm_values, quant_values], axis=2)
                self.warm_metadata.append(metadata)  # Track metadata per chunk

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
                # Quantize Cold for memory savings (Q4_0, same as Warm)
                keys_bf16 = keys.astype(mx.bfloat16) if keys.dtype == mx.float32 else keys
                values_bf16 = values.astype(mx.bfloat16) if values.dtype == mx.float32 else values
                quant_keys, quant_values, metadata = self.warm_quantizer.quantize(keys_bf16, values_bf16)

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
            layer_calib = self._am_calibration[self.layer_idx]
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
