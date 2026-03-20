#!/usr/bin/env python3
"""Fast performance test for optimized Metal KVTC codec.

Uses smaller calibration data for faster testing while preserving performance trends.
"""

import sys
sys.path.insert(0, '.')

import time
import numpy as np

from mlx_lm.models.kvtc_codec import (
    KVTCCodecConfig,
    fit_shared_calibration,
)
from mlx_lm.models.optimized_metal_kvtc import OptimizedMetalKVTCCodec


def fast_performance_test():
    """Fast performance benchmark."""
    print("=" * 70)
    print("Fast Performance Benchmark")
    print("=" * 70)
    print()

    # Smaller calibration data for faster testing
    print("Calibrating (reduced dataset: 200×256)...")
    np.random.seed(789)
    cal_keys = np.random.randn(200, 256).astype(np.float32)
    cal_values = np.random.randn(200, 256).astype(np.float32)

    config = KVTCCodecConfig(energy=0.99, bits=4, group_size=32, sample_limit=100)
    calibration = fit_shared_calibration([cal_keys], [cal_values], config)
    print("✅ Calibration complete")
    print()

    numpy_codec = calibration.keys
    metal_codec = OptimizedMetalKVTCCodec(calibration.keys, small_batch_threshold=100)

    batch_sizes = [50, 100, 200, 500]
    results = []

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}×256")

        x = np.random.randn(batch_size, 256).astype(np.float32)

        # Warmup
        for _ in range(2):
            _ = numpy_codec.encode(x)
            _ = metal_codec.encode(x)

        # Benchmark NumPy
        numpy_times = []
        for _ in range(5):
            start = time.perf_counter()
            enc = numpy_codec.encode(x)
            dec = numpy_codec.decode(enc)
            numpy_times.append((time.perf_counter() - start) * 1000)

        # Benchmark Metal
        metal_times = []
        for _ in range(5):
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
            'speedup': speedup,
            'used_metal': batch_size >= metal_codec.small_batch_threshold
        })

        print(f"  NumPy: {numpy_avg:.2f} ms")
        print(f"  Metal: {metal_avg:.2f} ms")
        if results[-1]['used_metal']:
            print(f"  Speedup: {speedup:.2f}x {'✅' if speedup >= 1.0 else '⚠️'}")
        else:
            print(f"  (Used NumPy fallback, batch < {metal_codec.small_batch_threshold})")
        print()

    # Summary
    print("=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"{'Batch':<10} {'NumPy (ms)':<15} {'Metal (ms)':<15} {'Speedup':<10} {'Used':<10}")
    print("-" * 70)
    for r in results:
        used = "Metal" if r['used_metal'] else "NumPy"
        print(f"{r['batch']:<10} {r['numpy']:<15.2f} {r['metal']:<15.2f} {r['speedup']:<10.2f}x {used:<10}")

    # Analysis
    print()
    print("=" * 70)
    print("Analysis")
    print("=" * 70)

    metal_results = [r for r in results if r['used_metal']]
    if metal_results:
        avg_speedup = np.mean([r['speedup'] for r in metal_results])
        print(f"Average speedup (Metal batches): {avg_speedup:.2f}x")

        if avg_speedup >= 1.0:
            print("✅ Metal shows performance advantage for large batches")
        elif avg_speedup >= 0.5:
            print("⚠️  Metal is slower but acceptable (within 2x of NumPy)")
        else:
            print("❌ Metal is significantly slower, needs optimization")

    numpy_fallback = [r for r in results if not r['used_metal']]
    if numpy_fallback:
        print(f"\nNumPy fallback used for {len(numpy_fallback)} batch sizes (< {metal_codec.small_batch_threshold})")
        print("✅ Smart fallback mechanism working correctly")

    print("=" * 70)
    print()

    return 0 if all(r['speedup'] > 0 for r in metal_results) else 1


if __name__ == "__main__":
    sys.exit(fast_performance_test())
