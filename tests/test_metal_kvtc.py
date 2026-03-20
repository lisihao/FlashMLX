"""Unit tests for Metal-accelerated KVTC codec."""

import numpy as np
import pytest

from mlx_lm.models.kvtc_codec import (
    KVTCCodecConfig,
    KVTCSharedCalibration,
    fit_shared_calibration,
)
from mlx_lm.models.metal_kvtc_codec import MetalKVTCCodec


@pytest.fixture
def sample_calibration_data():
    """Generate sample KV tensors for calibration."""
    np.random.seed(42)
    # Simulate key/value tensors: [batch, d_model]
    keys = np.random.randn(1000, 512).astype(np.float32)
    values = np.random.randn(1000, 512).astype(np.float32)
    return keys, values


@pytest.fixture
def calibrated_codec(sample_calibration_data):
    """Create a calibrated codec for testing."""
    keys, values = sample_calibration_data

    config = KVTCCodecConfig(
        energy=0.99,
        bits=4,
        group_size=64,
        sample_limit=500,
    )

    calibration = fit_shared_calibration([keys], [values], config)
    return calibration


def test_metal_codec_creation(calibrated_codec):
    """Test creating Metal codec from calibration."""
    # Create Metal codec
    metal_codec = MetalKVTCCodec(calibrated_codec.keys, enable_profiling=True)

    assert metal_codec.plan is not None
    assert metal_codec.mean_mx is not None
    assert metal_codec.basis_mx is not None
    assert metal_codec.metal_available  # Should be True on M-series Mac


def test_metal_codec_round_trip(calibrated_codec):
    """Test encode-decode round trip maintains correctness."""
    np.random.seed(123)
    x_original = np.random.randn(10, 512).astype(np.float32)

    # Encode with Metal codec
    metal_codec = MetalKVTCCodec(calibrated_codec.keys, enable_profiling=False)
    encoded = metal_codec.encode(x_original)

    # Decode
    x_decoded = metal_codec.decode(encoded)

    # Check shape
    assert x_decoded.shape == x_original.shape

    # Check reconstruction error
    mse = np.mean((x_original - x_decoded) ** 2)
    relative_error = mse / (np.mean(x_original ** 2) + 1e-8)

    print(f"\nMSE: {mse:.6f}")
    print(f"Relative error: {relative_error:.6f}")

    # Error should be small (< 1% for 4-bit quantization)
    assert relative_error < 0.01, f"Reconstruction error too large: {relative_error}"


def test_metal_codec_vs_numpy(calibrated_codec):
    """Compare Metal codec output with NumPy baseline."""
    np.random.seed(456)
    x_original = np.random.randn(20, 512).astype(np.float32)

    # Encode with NumPy codec (baseline)
    numpy_encoded = calibrated_codec.keys.encode(x_original)
    numpy_decoded = calibrated_codec.keys.decode(numpy_encoded)

    # Encode with Metal codec
    metal_codec = MetalKVTCCodec(calibrated_codec.keys, enable_profiling=False)
    metal_encoded = metal_codec.encode(x_original)
    metal_decoded = metal_codec.decode(metal_encoded)

    # Outputs should be very close (within floating point precision)
    np.testing.assert_allclose(
        metal_decoded,
        numpy_decoded,
        rtol=1e-4,  # Relative tolerance
        atol=1e-5,  # Absolute tolerance
        err_msg="Metal and NumPy codecs produce different results",
    )


def test_metal_codec_compression_ratio(calibrated_codec):
    """Verify Metal codec achieves similar compression as NumPy."""
    np.random.seed(789)
    x_original = np.random.randn(100, 512).astype(np.float32)

    # Encode with Metal codec
    metal_codec = MetalKVTCCodec(calibrated_codec.keys, enable_profiling=False)
    encoded = metal_codec.encode(x_original)

    # Calculate sizes
    original_bytes = x_original.nbytes
    payloads, _, _, _ = encoded
    compressed_bytes = sum(p.nbytes for p in payloads)

    compression_ratio = original_bytes / compressed_bytes

    print(f"\nOriginal: {original_bytes:,} bytes")
    print(f"Compressed: {compressed_bytes:,} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x")

    # Should achieve at least 4x compression with 4-bit quantization
    assert compression_ratio >= 4.0, f"Compression ratio too low: {compression_ratio}"


def test_metal_codec_profiling(calibrated_codec):
    """Test profiling functionality."""
    np.random.seed(101112)
    x = np.random.randn(50, 512).astype(np.float32)

    # Create codec with profiling
    metal_codec = MetalKVTCCodec(calibrated_codec.keys, enable_profiling=True)

    # Encode
    encoded = metal_codec.encode(x)

    # Decode
    decoded = metal_codec.decode(encoded)

    # Check stats were collected
    assert metal_codec.stats.project_time_ms > 0
    assert metal_codec.stats.quantize_time_ms > 0
    assert metal_codec.stats.deflate_time_ms > 0
    assert metal_codec.stats.transfer_time_ms > 0
    assert metal_codec.stats.dequantize_time_ms > 0
    assert metal_codec.stats.reconstruct_time_ms > 0

    # Print stats
    metal_codec.print_stats()

    # Total encode time should be sum of components
    expected_encode = (
        metal_codec.stats.project_time_ms
        + metal_codec.stats.quantize_time_ms
        + metal_codec.stats.deflate_time_ms
    )
    assert abs(metal_codec.stats.total_encode_time_ms - expected_encode) < 0.1


def test_metal_codec_batch_sizes(calibrated_codec):
    """Test Metal codec with various batch sizes."""
    metal_codec = MetalKVTCCodec(calibrated_codec.keys, enable_profiling=False)

    for batch_size in [1, 10, 100, 500]:
        np.random.seed(batch_size)
        x = np.random.randn(batch_size, 512).astype(np.float32)

        # Encode-decode
        encoded = metal_codec.encode(x)
        decoded = metal_codec.decode(encoded)

        # Check shape preserved
        assert decoded.shape == x.shape

        # Check reconstruction quality
        mse = np.mean((x - decoded) ** 2)
        relative_error = mse / (np.mean(x ** 2) + 1e-8)
        assert relative_error < 0.01


def test_metal_codec_zero_bit_blocks(calibrated_codec):
    """Test handling of zero-bit (pruned) blocks."""
    # This config might create zero-bit blocks
    config = KVTCCodecConfig(
        energy=0.95,  # Lower energy threshold
        bits=4,
        group_size=64,
        zero_bit_energy_fraction=0.1,  # More aggressive pruning
    )

    # Re-calibrate with new config
    np.random.seed(42)
    keys = np.random.randn(1000, 512).astype(np.float32)
    values = np.random.randn(1000, 512).astype(np.float32)
    calibration = fit_shared_calibration([keys], [values], config)

    # Test encode-decode
    metal_codec = MetalKVTCCodec(calibration.keys, enable_profiling=False)
    x = np.random.randn(10, 512).astype(np.float32)

    encoded = metal_codec.encode(x)
    decoded = metal_codec.decode(encoded)

    assert decoded.shape == x.shape
    # Reconstruction might be less accurate with pruning
    mse = np.mean((x - decoded) ** 2)
    relative_error = mse / (np.mean(x ** 2) + 1e-8)
    assert relative_error < 0.05  # Allow slightly higher error with pruning


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
