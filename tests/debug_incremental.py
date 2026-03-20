#!/usr/bin/env python3
"""Debug incremental KVTC cache."""

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

print("Original keys shape:", keys.shape)
print("Original keys sample:", keys[0, 0, :3, :5])

# Reshape for encoding
keys_flat = keys.reshape(-1, dim)
print("\nFlattened shape:", keys_flat.shape)
print("Flattened sample:", keys_flat[:3, :5])

# Calibration
config = KVTCCodecConfig(energy=0.99, bits=4, group_size=32, sample_limit=100)
cal_keys = mx.random.normal((100, dim))
calibration = fit_shared_calibration([cal_keys], [cal_keys], config)

# Encode
print("\nEncoding...")
encoded_raw = encode_tensor(keys_flat, calibration.keys)
print(f"Encoded type: {type(encoded_raw)}")
print(f"Encoded length: {len(encoded_raw) if isinstance(encoded_raw, tuple) else 'N/A'}")

# Check if we need to wrap it
if isinstance(encoded_raw, tuple):
    encoded = encoded_raw  # Already a tuple, don't re-wrap
    print(f"Encoded format: ({len(encoded[0])} payloads, {len(encoded[1])} shifts, {len(encoded[2])} scales, {len(encoded[3])} q_shapes, orig_shape={encoded[4]})")
else:
    encoded = tuple(x for x in encoded_raw)  # Generator, convert to tuple
    print(f"Converted to tuple, length: {len(encoded)}")

# Decode
print("\nDecoding...")
decoded_flat = decode_tensor(encoded, calibration.keys)
print("Decoded flat shape:", decoded_flat.shape)
print("Decoded flat sample:", decoded_flat[:3, :5])

# Reshape back
decoded = mx.array(decoded_flat.reshape(batch, heads, tokens, dim))
print("\nDecoded shape:", decoded.shape)
print("Decoded sample:", decoded[0, 0, :3, :5])

# Compare
diff = mx.max(mx.abs(keys - decoded))
print(f"\nMax diff: {diff:.6f}")
