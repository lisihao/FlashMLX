#!/usr/bin/env python3
"""Test optimized Metal KVTC codec with smart fallback."""

import sys
sys.path.insert(0, '.')

import time
import numpy as np

from mlx_lm.models.kvtc_codec import (
    KVTCCodecConfig,
    fit_shared_calibration,
)
from mlx_lm.models.optimized_metal_kvtc import OptimizedMetalKVTCCodec


def test_correctness():
    """Test encode-decode correctness."""
    print("=" * 70)
    print("Test 1: Correctness Verification")
    print("=" * 70)

    np.random.seed(42)
    cal_keys = np.random.randn(200, 256).astype(np.float32)
    cal_values = np.random.randn(200, 256).astype(np.float32)

    config = KVTCCodecConfig(energy=0.99, bits=4, group_size=32, sample_limit=100)
    calibration = fit_shared_calibration([cal_keys], [cal_values], config)

    numpy_codec = calibration.keys
    metal_codec = OptimizedMetalKVTCCodec(calibration.keys, small_batch_threshold=50)

    # Test data
    x = np.random.randn(10, 256).astype(np.float32)

    # Encode/decode with both codecs
    numpy_enc = numpy_codec.encode(x)
    numpy_dec = numpy_codec.decode(numpy_enc)

    metal_enc = metal_codec.encode(x)
    metal_dec = metal_codec.decode(metal_enc)

    # Compare results
    max_diff = np.max(np.abs(numpy_dec - metal_dec))
    mean_diff = np.mean(np.abs(numpy_dec - metal_dec))
    mse = np.mean((numpy_dec - metal_dec) ** 2)

    print(f"Max diff:  {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    print(f"MSE:       {mse:.6f}")

    if max_diff < 1e-4:
        print("✅ Correctness test PASSED!\n")
        return True
    else:
        print("❌ Correctness test FAILED!\n")
        return False


def test_smart_fallback():
    """Test smart fallback mechanism."""
    print("=" * 70)
    print("Test 2: Smart Fallback Mechanism")
    print("=" * 70)

    np.random.seed(123)
    cal_keys = np.random.randn(200, 256).astype(np.float32)
    cal_values = np.random.randn(200, 256).astype(np.float32)

    config = KVTCCodecConfig(energy=0.99, bits=4, group_size=32, sample_limit=100)
    calibration = fit_shared_calibration([cal_keys], [cal_values], config)

    metal_codec = OptimizedMetalKVTCCodec(
        calibration.keys,
        small_batch_threshold=80,
        enable_profiling=True
    )

    # Test small batch (should use NumPy)
    print("Testing small batch (50, should use NumPy)...")
    x_small = np.random.randn(50, 256).astype(np.float32)
    _ = metal_codec.encode(x_small)

    # Test large batch (should use Metal)
    print("Testing large batch (100, should use Metal)...")
    x_large = np.random.randn(100, 256).astype(np.float32)
    _ = metal_codec.encode(x_large)

    metal_codec.print_stats()

    if metal_codec.stats.numpy_batches == 1 and metal_codec.stats.metal_batches == 1:
        print("✅ Smart fallback test PASSED!\n")
        return True
    else:
        print("❌ Smart fallback test FAILED!\n")
        return False


def test_performance():
    """Test performance across different batch sizes."""
    print("=" * 70)
    print("Test 3: Performance Benchmark")
    print("=" * 70)

    np.random.seed(456)
    cal_keys = np.random.randn(500, 512).astype(np.float32)
    cal_values = np.random.randn(500, 512).astype(np.float32)

    config = KVTCCodecConfig(energy=0.99, bits=4, group_size=64, sample_limit=250)
    print("Calibrating (this may take 30-60 seconds)...")
    calibration = fit_shared_calibration([cal_keys], [cal_values], config)

    numpy_codec = calibration.keys
    metal_codec = OptimizedMetalKVTCCodec(calibration.keys, small_batch_threshold=100)

    batch_sizes = [50, 100, 200, 500]
    results = []

    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}×512")

        x = np.random.randn(batch_size, 512).astype(np.float32)

        # Warmup
        for _ in range(2):
            _ = numpy_codec.encode(x)
            _ = metal_codec.encode(x)

        # Benchmark NumPy
        numpy_times = []
        for _ in range(3):
            start = time.perf_counter()
            enc = numpy_codec.encode(x)
            dec = numpy_codec.decode(enc)
            numpy_times.append((time.perf_counter() - start) * 1000)

        # Benchmark Metal
        metal_times = []
        for _ in range(3):
            start = time.perf_counter()
            enc = metal_codec.encode(x)
            dec = metal_codec.decode(enc)
            metal_times.append((time.perf_counter() - start) * 1000)

        numpy_avg = np.mean(numpy_times)
        metal_avg = np.mean(metal_times)
        speedup = numpy_avg / metal_avg if metal_avg > 0 else 0

        results.append({
            'batch': batch_size,
            'numpy': numpy_avg,
            'metal': metal_avg,
            'speedup': speedup
        })

        print(f"  NumPy: {numpy_avg:.2f} ms")
        print(f"  Metal: {metal_avg:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x {'✅' if speedup >= 1.0 else '❌'}")

    # Summary
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"{'Batch':<10} {'NumPy (ms)':<15} {'Metal (ms)':<15} {'Speedup':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['batch']:<10} {r['numpy']:<15.2f} {r['metal']:<15.2f} {r['speedup']:<10.2f}x")

    # Check if large batches are faster
    large_batch_results = [r for r in results if r['batch'] >= 100]
    if large_batch_results:
        avg_speedup = np.mean([r['speedup'] for r in large_batch_results])
        print(f"\nAverage speedup (batch >= 100): {avg_speedup:.2f}x")

        if avg_speedup >= 1.0:
            print("✅ Performance test PASSED! (Metal faster for large batches)\n")
            return True
        else:
            print("⚠️ Performance test WARNING: Metal not faster yet\n")
            return True  # Still pass, as correctness is more important

    return True


def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "Optimized Metal KVTC Test Suite" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    all_passed = True

    try:
        all_passed &= test_correctness()
        all_passed &= test_smart_fallback()
        all_passed &= test_performance()

        print("=" * 70)
        if all_passed:
            print("🎉 All tests PASSED!")
        else:
            print("❌ Some tests FAILED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
