#!/usr/bin/env python3
"""Test incremental cache chunk logic without compression."""

import sys
sys.path.insert(0, '.')

import numpy as np
import mlx.core as mx

# Simple mock encoding/decoding for testing
def mock_encode(x):
    """Simple mock encoder - just returns the data as-is in a tuple."""
    return (x.tobytes(), x.shape)

def mock_decode(encoded):
    """Simple mock decoder - reconstructs from bytes."""
    data_bytes, shape = encoded
    return np.frombuffer(data_bytes, dtype=np.float32).reshape(shape)

# Test chunk-based storage
print("=" * 70)
print("Test: Chunk-based Storage and Concatenation")
print("=" * 70)

# Initial data: 5 tokens
batch, heads, initial_tokens, dim = 1, 8, 5, 64
keys_initial = np.random.randn(batch * heads, initial_tokens, dim).astype(np.float32)
values_initial = np.random.randn(batch * heads, initial_tokens, dim).astype(np.float32)

print(f"Initial keys shape: {keys_initial.shape}")
print(f"Initial keys sample: {keys_initial[0, :3, :5]}")

# Store as first chunk
chunks = []
encoded_keys_init = mock_encode(keys_initial)
encoded_values_init = mock_encode(values_initial)
chunks.append((encoded_keys_init, encoded_values_init, initial_tokens))

print(f"✅ Stored chunk 1: {initial_tokens} tokens")

# Append new tokens: 3 tokens
new_tokens = 3
keys_new = np.random.randn(batch * heads, new_tokens, dim).astype(np.float32)
values_new = np.random.randn(batch * heads, new_tokens, dim).astype(np.float32)

print(f"\nAppending {new_tokens} new tokens...")
print(f"New keys shape: {keys_new.shape}")
print(f"New keys sample: {keys_new[0, :3, :5]}")

# Store as second chunk
encoded_keys_new = mock_encode(keys_new)
encoded_values_new = mock_encode(values_new)
chunks.append((encoded_keys_new, encoded_values_new, new_tokens))

print(f"✅ Stored chunk 2: {new_tokens} tokens")

# Decode all chunks and concatenate
print("\nDecoding all chunks...")
decoded_keys_list = []
decoded_values_list = []

for encoded_keys, encoded_values, num_tokens in chunks:
    decoded_keys_list.append(mock_decode(encoded_keys))
    decoded_values_list.append(mock_decode(encoded_values))

# Concatenate along token dimension (axis=1)
keys_concat = np.concatenate(decoded_keys_list, axis=1)
values_concat = np.concatenate(decoded_values_list, axis=1)

print(f"Concatenated keys shape: {keys_concat.shape}")
print(f"Expected shape: ({batch * heads}, {initial_tokens + new_tokens}, {dim})")

# Verify correctness
expected_keys = np.concatenate([keys_initial, keys_new], axis=1)
expected_values = np.concatenate([values_initial, values_new], axis=1)

key_diff = np.max(np.abs(expected_keys - keys_concat))
value_diff = np.max(np.abs(expected_values - values_concat))

print(f"\nKey diff: {key_diff:.10f}")
print(f"Value diff: {value_diff:.10f}")

if key_diff < 1e-6 and value_diff < 1e-6:
    print("✅ Chunk logic test PASSED!")
else:
    print("❌ Chunk logic test FAILED!")
