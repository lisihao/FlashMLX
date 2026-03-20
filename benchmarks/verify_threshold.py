#!/usr/bin/env python3
"""Verify threshold adjustment."""

import sys
sys.path.insert(0, '.')

import numpy as np

from mlx_lm.models.kvtc_codec import (
    KVTCCodecConfig,
    fit_shared_calibration,
)
from mlx_lm.models.optimized_metal_kvtc import OptimizedMetalKVTCCodec


print("=== Threshold Verification ===")
print()

# Quick calibration
np.random.seed(42)
cal_keys = np.random.randn(50, 128).astype(np.float32)
cal_values = np.random.randn(50, 128).astype(np.float32)

config = KVTCCodecConfig(energy=0.99, bits=4, group_size=32, sample_limit=50)
calibration = fit_shared_calibration([cal_keys], [cal_values], config)

# Create codec with default threshold
codec = OptimizedMetalKVTCCodec(calibration.keys, enable_profiling=True)

print(f"✅ Default threshold: {codec.small_batch_threshold}")
print()

# Test different batch sizes
test_cases = [
    (50, "Should use NumPy"),
    (100, "Should use NumPy"),
    (200, "Should use NumPy"),
    (300, "Should use Metal"),
    (500, "Should use Metal"),
]

for batch_size, expected in test_cases:
    x = np.random.randn(batch_size, 128).astype(np.float32)
    _ = codec.encode(x)

    used_numpy = codec.stats.numpy_batches > 0
    used_metal = codec.stats.metal_batches > 0

    actual = "NumPy" if used_numpy and codec.stats.numpy_batches == codec.stats.total_batches else "Metal"
    status = "✅" if expected.endswith(actual) else "❌"

    print(f"Batch {batch_size:3d}: {actual:5s} {status} ({expected})")

    # Reset stats
    codec.stats.numpy_batches = 0
    codec.stats.metal_batches = 0
    codec.stats.total_batches = 0

print()
print("=== Verification Complete ===")
