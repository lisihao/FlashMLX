#!/usr/bin/env python3
"""Quick correctness test for optimized Metal KVTC codec."""

import sys
sys.path.insert(0, '.')

import numpy as np

from mlx_lm.models.kvtc_codec import (
    KVTCCodecConfig,
    fit_shared_calibration,
)
from mlx_lm.models.optimized_metal_kvtc import OptimizedMetalKVTCCodec


def quick_test():
    """Quick correctness test."""
    print("=" * 60)
    print("Quick Correctness Test")
    print("=" * 60)

    # Small calibration data
    np.random.seed(42)
    cal_keys = np.random.randn(50, 128).astype(np.float32)
    cal_values = np.random.randn(50, 128).astype(np.float32)

    config = KVTCCodecConfig(energy=0.99, bits=4, group_size=32, sample_limit=50)
    print("Calibrating...")
    calibration = fit_shared_calibration([cal_keys], [cal_values], config)

    numpy_codec = calibration.keys
    metal_codec = OptimizedMetalKVTCCodec(calibration.keys, small_batch_threshold=50)

    # Test data
    x = np.random.randn(10, 128).astype(np.float32)

    # Encode/decode with both codecs
    print("Testing NumPy codec...")
    numpy_enc = numpy_codec.encode(x)
    numpy_dec = numpy_codec.decode(numpy_enc)

    print("Testing Metal codec...")
    metal_enc = metal_codec.encode(x)
    metal_dec = metal_codec.decode(metal_enc)

    # Compare results
    max_diff = np.max(np.abs(numpy_dec - metal_dec))
    mean_diff = np.mean(np.abs(numpy_dec - metal_dec))
    mse = np.mean((numpy_dec - metal_dec) ** 2)

    print(f"\nMax diff:  {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    print(f"MSE:       {mse:.6f}")

    if max_diff < 1e-4:
        print("\n✅ Quick test PASSED!")
        return 0
    else:
        print("\n❌ Quick test FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(quick_test())
