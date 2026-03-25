#!/usr/bin/env python3
"""Test KVTC PCA compression.

Quick validation of PCA-based KV cache compression.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from mlx_lm.models.kvtc_codec import KVTCCodecConfig
from mlx_lm.models.kvtc_pca_codec import (
    fit_pca_calibration,
    encode_tensor_pca,
    decode_tensor_pca,
)


def generate_test_data(rows, dim, latent_dim=8, seed=42):
    """Generate low-rank structured data."""
    rng = np.random.default_rng(seed)

    # Generate low-rank structure
    latent = rng.normal(size=(rows, latent_dim)).astype(np.float32)
    basis = rng.normal(size=(latent_dim, dim)).astype(np.float32)

    # Add small noise
    data = latent @ basis
    data += 0.01 * rng.normal(size=data.shape).astype(np.float32)

    return data


def test_pca_codec():
    """Test PCA compression on synthetic data."""
    print("=" * 70)
    print("KVTC PCA Codec Test")
    print("=" * 70)
    print()

    # Generate test data
    rows, dim = 1024, 128
    keys = generate_test_data(rows, dim)
    values = generate_test_data(rows, dim, seed=43)

    print(f"Data shape: {keys.shape}")
    print(f"Original size: {keys.nbytes / 1024:.2f} KB")
    print()

    # Test different configurations
    configs = [
        ("PCA-8 (rank=8, 4-bit)", KVTCCodecConfig(rank=8, bits=4, group_size=16)),
        ("PCA-16 (rank=16, 4-bit)", KVTCCodecConfig(rank=16, bits=4, group_size=16)),
        ("PCA-32 (rank=32, 4-bit)", KVTCCodecConfig(rank=32, bits=4, group_size=16)),
    ]

    results = []

    for name, config in configs:
        print("-" * 70)
        print(f"Testing: {name}")
        print("-" * 70)

        # Fit calibration
        calibration = fit_pca_calibration([keys], [values], config)

        # Encode
        encoded = encode_tensor_pca(keys, calibration.keys)

        # Measure size
        payload, shifts, scales, q_shape, mean, basis, orig_shape = encoded
        compressed_size = payload.nbytes + shifts.nbytes + scales.nbytes + mean.nbytes + basis.nbytes

        # Decode
        decoded = decode_tensor_pca(encoded, calibration.keys)

        # Measure error
        rel_error = np.linalg.norm(decoded - keys) / np.linalg.norm(keys)
        compression_ratio = keys.nbytes / compressed_size

        print(f"✓ Rank: {basis.shape[1]}")
        print(f"✓ Relative error: {rel_error:.6f}")
        print(f"✓ Compressed size: {compressed_size / 1024:.2f} KB")
        print(f"✓ Compression ratio: {compression_ratio:.2f}x")
        print()

        results.append((name, compression_ratio, rel_error))

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    print(f"{'Configuration':30s}  {'Compression':>12s}  {'Rel. Error':>12s}")
    print(f"{'-'*30}  {'-'*12}  {'-'*12}")
    for name, comp, err in results:
        print(f"{name:30s}  {comp:11.2f}x  {err:12.6f}")
    print()

    # Validation
    all_passed = all(err < 1.0 for _, _, err in results)

    if all_passed:
        print("✅ All tests PASSED!")
        return 0
    else:
        print("❌ Some tests FAILED!")
        return 1


def main():
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "KVTC PCA Codec Test" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        return test_pca_codec()
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
