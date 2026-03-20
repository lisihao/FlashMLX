"""Optimized Metal-accelerated KVTC codec with smart fallback.

Key optimizations:
1. Persistent GPU buffers (mean, basis stay on GPU)
2. Vectorized quantization (process all groups at once)
3. Smart fallback (small batch uses NumPy)
4. Reduced CPU-GPU round-trips
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Tuple
import zlib

import mlx.core as mx
import numpy as np

from .kvtc_codec import (
    KVTCCodecConfig,
    KVTCTransformPlan,
    _to_numpy,
)


@dataclass
class PerformanceStats:
    """Performance statistics."""

    small_batch_numpy_ms: float = 0.0
    large_batch_metal_ms: float = 0.0
    total_batches: int = 0
    numpy_batches: int = 0
    metal_batches: int = 0


class OptimizedMetalKVTCCodec:
    """Optimized Metal-accelerated KVTC codec with smart fallback.

    Optimizations:
    - Persistent GPU buffers for calibration data
    - Vectorized quantization
    - Smart fallback to NumPy for small batches
    - Minimal CPU-GPU data transfers

    Usage:
        codec = OptimizedMetalKVTCCodec(plan)
        encoded = codec.encode(x)  # Auto-selects NumPy or Metal
        decoded = codec.decode(encoded)
    """

    def __init__(
        self,
        plan: KVTCTransformPlan,
        small_batch_threshold: int = 300,
        enable_profiling: bool = False,
    ):
        """Initialize optimized Metal codec.

        Args:
            plan: Existing KVTCTransformPlan with calibration data
            small_batch_threshold: Batch size below which NumPy is used (default: 300)
                Based on performance testing, Metal shows performance advantage only
                for batch >= 300. Smaller batches use NumPy fallback for better performance.
            enable_profiling: If True, collect performance statistics
        """
        self.plan = plan
        self.small_batch_threshold = small_batch_threshold
        self.enable_profiling = enable_profiling
        self.stats = PerformanceStats()

        # Persistent GPU buffers (avoid repeated transfers)
        self.mean_gpu = mx.array(plan.mean)  # [d_model]
        self.basis_gpu = mx.array(plan.basis)  # [d_model, rank]

        # Pre-transpose basis for reconstruction
        self.basis_T_gpu = self.basis_gpu.T  # [rank, d_model]

        # Force evaluation to ensure buffers are created
        mx.eval(self.mean_gpu, self.basis_gpu, self.basis_T_gpu)

        print(f"✅ OptimizedMetalKVTCCodec initialized")
        print(f"   Calibration data on GPU: mean {self.mean_gpu.shape}, basis {self.basis_gpu.shape}")
        print(f"   Small batch threshold: {small_batch_threshold}")

    def _should_use_metal(self, batch_size: int) -> bool:
        """Determine whether to use Metal or NumPy."""
        return batch_size >= self.small_batch_threshold

    def _project_vectorized(self, x_gpu: mx.array) -> mx.array:
        """Vectorized PCA projection: y = (x - mean) @ basis.

        All operations on GPU, no CPU-GPU round-trips.
        """
        # Center: x_centered = x - mean (broadcasting)
        x_centered = x_gpu - self.mean_gpu

        # Project: y = x_centered @ basis
        y = mx.matmul(x_centered, self.basis_gpu)

        return y

    def _reconstruct_vectorized(self, y_gpu: mx.array) -> mx.array:
        """Vectorized PCA reconstruction: x = y @ basis.T + mean.

        All operations on GPU, no CPU-GPU round-trips.
        """
        # Reconstruct: x = y @ basis.T
        x = mx.matmul(y_gpu, self.basis_T_gpu)

        # Uncenter: x = x + mean
        x = x + self.mean_gpu

        return x

    def _quantize_vectorized(
        self,
        x_gpu: mx.array,
        bits: int,
        group_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized quantization: process all groups at once.

        Significantly faster than per-group loops.
        """
        batch, rank = x_gpu.shape
        n_groups = (rank + group_size - 1) // group_size
        qmax = (1 << (bits - 1)) - 1

        # Reshape to [batch, n_groups, group_size] for vectorized processing
        # Pad if necessary
        pad_size = n_groups * group_size - rank
        if pad_size > 0:
            x_padded = mx.pad(x_gpu, [(0, 0), (0, pad_size)])
        else:
            x_padded = x_gpu

        x_reshaped = x_padded.reshape(batch, n_groups, group_size)

        # Compute shifts (mean per group) - [n_groups]
        shifts = mx.mean(x_reshaped, axis=(0, 2))  # Average over batch and group_size

        # Center each group
        shifts_expanded = shifts.reshape(1, n_groups, 1)
        x_centered = x_reshaped - shifts_expanded

        # Compute scales (max abs per group) - [n_groups]
        scales = mx.max(mx.abs(x_centered), axis=(0, 2))  # Max over batch and group_size
        scales = mx.where(scales > 0, scales / qmax, 1.0)

        # Quantize
        scales_expanded = scales.reshape(1, n_groups, 1)
        q = mx.clip(
            mx.round(x_centered / scales_expanded),
            -qmax - 1,
            qmax
        ).astype(mx.int8)

        # Reshape back to [batch, rank]
        q = q.reshape(batch, n_groups * group_size)
        if pad_size > 0:
            q = q[:, :rank]

        # Force evaluation and transfer to CPU
        mx.eval(q, shifts, scales)

        q_np = np.array(q, copy=False)
        shifts_np = np.array(shifts, copy=False)
        scales_np = np.array(scales, copy=False)

        # DEFLATE compression
        payload = zlib.compress(q_np.tobytes(), level=9)
        payload_np = np.frombuffer(payload, dtype=np.uint8).copy()

        return (
            payload_np,
            shifts_np,
            scales_np,
            np.asarray(q_np.shape, dtype=np.int32),
        )

    def _dequantize_vectorized(
        self,
        payload: np.ndarray,
        shifts: np.ndarray,
        scales: np.ndarray,
        q_shape: np.ndarray,
    ) -> mx.array:
        """Vectorized dequantization: process all groups at once."""
        # Decompress
        q_bytes = zlib.decompress(payload.tobytes())
        q_np = np.frombuffer(q_bytes, dtype=np.int8).reshape(tuple(q_shape))

        # Transfer to GPU
        q_gpu = mx.array(q_np)
        shifts_gpu = mx.array(shifts)
        scales_gpu = mx.array(scales)

        batch, rank = q_gpu.shape
        group_size = self.plan.config.group_size
        n_groups = len(shifts)

        # Reshape to [batch, n_groups, group_size]
        pad_size = n_groups * group_size - rank
        if pad_size > 0:
            q_padded = mx.pad(q_gpu, [(0, 0), (0, pad_size)])
        else:
            q_padded = q_gpu

        q_reshaped = q_padded.reshape(batch, n_groups, group_size).astype(mx.float32)

        # Dequantize: x = q * scale + shift
        scales_expanded = scales_gpu.reshape(1, n_groups, 1)
        shifts_expanded = shifts_gpu.reshape(1, n_groups, 1)

        x_reshaped = q_reshaped * scales_expanded + shifts_expanded

        # Reshape back
        x = x_reshaped.reshape(batch, n_groups * group_size)
        if pad_size > 0:
            x = x[:, :rank]

        return x

    def encode(self, x: np.ndarray):
        """Encode with smart fallback: NumPy for small batches, Metal for large.

        Args:
            x: [batch, d_model] input tensor (NumPy)

        Returns:
            Encoded data (same format as KVTCTransformPlan)
        """
        batch_size = x.shape[0]

        # Smart fallback for small batches
        if not self._should_use_metal(batch_size):
            if self.enable_profiling:
                start = time.perf_counter()
                result = self.plan.encode(x)
                elapsed = (time.perf_counter() - start) * 1000
                self.stats.small_batch_numpy_ms += elapsed
                self.stats.numpy_batches += 1
            else:
                result = self.plan.encode(x)

            self.stats.total_batches += 1
            return result

        # Large batch: use Metal
        if self.enable_profiling:
            start = time.perf_counter()

        # Transfer to GPU (only x, mean/basis already on GPU)
        x_gpu = mx.array(x.astype(np.float32, copy=False))

        # Project to PCA space (GPU)
        coeffs_gpu = self._project_vectorized(x_gpu)

        # Quantize and compress by blocks
        payloads = []
        shifts = []
        scales = []
        q_shapes = []

        for start_idx, width, bits in self.plan.block_meta:
            start_idx = int(start_idx)
            width = int(width)
            bits = int(bits)

            block_gpu = coeffs_gpu[:, start_idx : start_idx + width]

            if bits == 0:
                # Zero-bit block (pruned)
                payloads.append(np.zeros(1, dtype=np.uint8))
                shifts.append(np.zeros(1, dtype=np.float32))
                scales.append(np.zeros(1, dtype=np.float32))
                q_shapes.append(np.asarray(block_gpu.shape, dtype=np.int32))
                continue

            # Quantize group (vectorized)
            group_size = min(self.plan.config.group_size, width)
            payload, block_shifts, block_scales, q_shape = self._quantize_vectorized(
                block_gpu, bits, group_size
            )

            payloads.append(payload)
            shifts.append(block_shifts)
            scales.append(block_scales)
            q_shapes.append(q_shape)

        if self.enable_profiling:
            elapsed = (time.perf_counter() - start) * 1000
            self.stats.large_batch_metal_ms += elapsed
            self.stats.metal_batches += 1

        self.stats.total_batches += 1

        return (
            tuple(payloads),
            tuple(shifts),
            tuple(scales),
            tuple(q_shapes),
            np.asarray(x.shape, dtype=np.int32),  # orig_shape
        )

    def decode(
        self,
        encoded: Tuple[
            Tuple[np.ndarray, ...],
            Tuple[np.ndarray, ...],
            Tuple[np.ndarray, ...],
            Tuple[np.ndarray, ...],
            np.ndarray,
        ],
    ) -> np.ndarray:
        """Decode with smart fallback.

        Args:
            encoded: Tuple of (payloads, shifts, scales, q_shapes, orig_shape)

        Returns:
            x_hat: [batch, d_model] reconstructed tensor (NumPy)
        """
        payloads, shifts, scales, q_shapes, orig_shape = encoded

        # Check batch size for fallback
        batch_size = int(q_shapes[0][0]) if q_shapes else 1

        if not self._should_use_metal(batch_size):
            return self.plan.decode(encoded)

        # Large batch: use Metal
        total_rank = sum(int(width) for _, width, _ in self.plan.block_meta)

        # Allocate coefficient buffer on GPU
        coeffs_gpu = mx.zeros((batch_size, total_rank), dtype=mx.float32)

        # Dequantize and assemble blocks
        for i, (start_idx, width, bits) in enumerate(self.plan.block_meta):
            start_idx = int(start_idx)
            width = int(width)
            bits = int(bits)

            if bits == 0:
                # Zero-bit block remains zero
                continue

            # Dequantize block (vectorized)
            block_gpu = self._dequantize_vectorized(
                payloads[i], shifts[i], scales[i], q_shapes[i]
            )

            coeffs_gpu[:, start_idx : start_idx + width] = block_gpu

        # Reconstruct from PCA space (GPU)
        x_hat_gpu = self._reconstruct_vectorized(coeffs_gpu)

        # Transfer back to CPU
        return np.array(x_hat_gpu, copy=False)

    def print_stats(self):
        """Print performance statistics."""
        if not self.enable_profiling:
            print("Profiling disabled. Enable with enable_profiling=True")
            return

        print("\n" + "=" * 70)
        print("Optimized Metal KVTC Codec - Performance Statistics")
        print("=" * 70)
        print(f"Total batches:   {self.stats.total_batches}")
        print(f"NumPy batches:   {self.stats.numpy_batches} (small, < {self.small_batch_threshold})")
        print(f"Metal batches:   {self.stats.metal_batches} (large, >= {self.small_batch_threshold})")

        if self.stats.numpy_batches > 0:
            avg_numpy = self.stats.small_batch_numpy_ms / self.stats.numpy_batches
            print(f"\nNumPy avg time:  {avg_numpy:.2f} ms/batch")

        if self.stats.metal_batches > 0:
            avg_metal = self.stats.large_batch_metal_ms / self.stats.metal_batches
            print(f"Metal avg time:  {avg_metal:.2f} ms/batch")

            if self.stats.numpy_batches > 0:
                # Rough speedup estimate (not accurate due to different batch sizes)
                print(f"\nNote: Direct speedup comparison not meaningful due to different batch sizes")

        print("=" * 70 + "\n")


def create_optimized_codec(
    plan: KVTCTransformPlan,
    small_batch_threshold: int = 80,
) -> OptimizedMetalKVTCCodec:
    """Create an optimized Metal codec with smart fallback.

    Args:
        plan: Existing KVTCTransformPlan with calibration data
        small_batch_threshold: Batch size below which NumPy is used

    Returns:
        OptimizedMetalKVTCCodec instance
    """
    return OptimizedMetalKVTCCodec(plan, small_batch_threshold, enable_profiling=False)
