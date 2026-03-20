#!/usr/bin/env python3
"""Detailed debug of KVTC encoding/decoding."""

import sys
sys.path.insert(0, '.')

import numpy as np
import mlx.core as mx

from mlx_lm.models.kvtc_codec import (
    KVTCCodecConfig,
    fit_shared_calibration,
    project,
    reconstruct,
)

np.random.seed(42)

# Create test data
batch, heads, tokens, dim = 1, 8, 10, 64
keys = mx.random.normal((batch, heads, tokens, dim))
keys_flat = keys.reshape(-1, dim)

print("=" * 70)
print("Step 1: Original Data")
print("=" * 70)
print(f"Shape: {keys_flat.shape}")
print(f"First 3 rows, first 5 cols:\n{keys_flat[:3, :5]}")
print(f"Row-wise mean: {keys_flat.mean(axis=1)[:5]}")
print(f"Row-wise std: {keys_flat.std(axis=1)[:5]}")

# Convert to numpy first
keys_np = np.asarray(keys_flat, dtype=np.float32)

# Calibration
print("\n" + "=" * 70)
print("Step 2: Calibration")
print("=" * 70)
config = KVTCCodecConfig(energy=0.99, bits=4, group_size=32, sample_limit=100)
# Use the actual data for calibration instead of separate random data
calibration = fit_shared_calibration([keys_np], [keys_np], config)

plan = calibration.keys
print(f"PCA mean shape: {plan.mean.shape}")
print(f"PCA mean values (first 5): {plan.mean[0, :5]}")
print(f"PCA basis shape: {plan.basis.shape}")
print(f"Rank: {plan.basis.shape[1]}")
print(f"Block meta: {plan.block_meta}")

# Manual projection
print("\n" + "=" * 70)
print("Step 3: PCA Projection")
print("=" * 70)
coeffs = project(keys_np, plan.mean, plan.basis)
print(f"Coeffs shape: {coeffs.shape}")
print(f"First 3 rows:\n{coeffs[:3]}")
print(f"Row-wise mean: {coeffs.mean(axis=1)[:5]}")
print(f"Row-wise std: {coeffs.std(axis=1)[:5]}")
print(f"All rows identical? {np.allclose(coeffs[0], coeffs[1])}")

# Manual reconstruction (before quantization)
print("\n" + "=" * 70)
print("Step 4: Reconstruction (without quantization)")
print("=" * 70)
recon_no_quant = reconstruct(coeffs, plan.mean, plan.basis)
print(f"Reconstructed shape: {recon_no_quant.shape}")
print(f"First 3 rows, first 5 cols:\n{recon_no_quant[:3, :5]}")
print(f"Max diff vs original: {np.max(np.abs(keys_np - recon_no_quant)):.6f}")

# Full encode
print("\n" + "=" * 70)
print("Step 5: Full Encode (with quantization)")
print("=" * 70)
from mlx_lm.models.kvtc_codec import encode_tensor, decode_tensor

encoded = encode_tensor(keys_np, plan)
payloads, shifts, scales, q_shapes, orig_shape = encoded
print(f"Number of blocks: {len(payloads)}")
print(f"q_shapes: {[tuple(qs) for qs in q_shapes]}")
print(f"shifts: {shifts}")
print(f"scales: {scales}")

# Full decode
print("\n" + "=" * 70)
print("Step 6: Full Decode")
print("=" * 70)
decoded = decode_tensor(encoded, plan)
print(f"Decoded shape: {decoded.shape}")
print(f"First 3 rows, first 5 cols:\n{decoded[:3, :5]}")
print(f"Row-wise mean: {decoded.mean(axis=1)[:5]}")
print(f"All rows identical? {np.allclose(decoded[0], decoded[1])}")
print(f"\nMax diff vs original: {np.max(np.abs(keys_np - decoded)):.6f}")
