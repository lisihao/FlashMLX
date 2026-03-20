#!/usr/bin/env python3
"""Simple test to verify KVTC codec without zero-bit allocation."""

import sys
sys.path.insert(0, '.')

import numpy as np
import mlx.core as mx

from mlx_lm.models.kvtc_codec import (
    KVTCCodecConfig,
    fit_shared_calibration,
    encode_tensor,
    decode_tensor,
)

np.random.seed(42)

# Create test data
batch, heads, tokens, dim = 1, 8, 10, 64
keys = mx.random.normal((batch, heads, tokens, dim))
keys_flat = keys.reshape(-1, dim)
keys_np = np.asarray(keys_flat, dtype=np.float32)

print("Original shape:", keys_np.shape)
print("Original sample (first 3 rows, first 5 cols):")
print(keys_np[:3, :5])

# Calibration with much larger budget to avoid zero-bit fallback
# The DP algorithm needs sufficient budget to allocate bits
config = KVTCCodecConfig(
    rank=16,  # Even smaller rank
    bits=8,  # Higher bits = larger budget
    group_size=8,  # Smaller group size for better allocation    sample_limit=100,
    allowed_block_sizes=(4, 8, 16),  # Fine-grained blocks
)
calibration = fit_shared_calibration([keys_np], [keys_np], config)

print("\nCalibration:")
print(f"PCA rank: {calibration.keys.basis.shape[1]}")
print(f"Block meta: {calibration.keys.block_meta}")

# Encode
encoded = encode_tensor(keys_np, calibration.keys)
payloads, shifts, scales, q_shapes, orig_shape = encoded
print("\nEncoded:")
print(f"Number of blocks: {len(payloads)}")
print(f"Block bits: {[block[2] for block in calibration.keys.block_meta]}")
print(f"scales (first block): {scales[0][:3]}")

# Decode
decoded = decode_tensor(encoded, calibration.keys)
print("\nDecoded shape:", decoded.shape)
print("Decoded sample (first 3 rows, first 5 cols):")
print(decoded[:3, :5])

# Check correctness
diff = np.max(np.abs(keys_np - decoded))
print(f"\nMax diff: {diff:.6f}")

if diff < 0.5:
    print("✅ Test PASSED!")
else:
    print("❌ Test FAILED!")
