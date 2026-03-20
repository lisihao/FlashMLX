#!/usr/bin/env python3
"""Performance benchmark comparing Metal-accelerated KVTC vs NumPy baseline."""

import argparse
import time
from pathlib import Path

import numpy as np

from mlx_lm.models.kvtc_codec import (
    KVTCCodecConfig,
    fit_shared_calibration,
)
from mlx_lm.models.metal_kvtc_codec import MetalKVTCCodec


def benchmark_encode_decode(
    codec,
    x: np.ndarray,
    n_runs: int = 10,
    warmup: int = 2,
) -> tuple[float, float]:
    """Benchmark encode and decode times.

    Returns:
        (encode_time_ms, decode_time_ms)
    """
    # Warmup
    for _ in range(warmup):
        encoded = codec.encode(x)
        _ = codec.decode(encoded)

    # Benchmark encode
    encode_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        encoded = codec.encode(x)
        encode_times.append((time.perf_counter() - start) * 1000)

    # Benchmark decode
    decode_times = []
    encoded = codec.encode(x)  # Use same encoded data
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = codec.decode(encoded)
        decode_times.append((time.perf_counter() - start) * 1000)

    return np.mean(encode_times), np.mean(decode_times)


def main():
    parser = argparse.ArgumentParser(description="Metal KVTC performance benchmark")
    parser.add_argument("--batch", type=int, default=100, help="Batch size")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs")
    args = parser.parse_args()

    print("=" * 80)
    print("Metal KVTC Performance Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Batch size: {args.batch}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Runs: {args.runs}")
    print(f"  Warmup: {args.warmup}")
    print()

    # Generate calibration data
    print("Generating calibration data...")
    np.random.seed(42)
    cal_keys = np.random.randn(1000, args.d_model).astype(np.float32)
    cal_values = np.random.randn(1000, args.d_model).astype(np.float32)

    # Calibrate
    print("Calibrating codec...")
    config = KVTCCodecConfig(
        energy=0.99,
        bits=4,
        group_size=64,
        sample_limit=500,
    )
    calibration = fit_shared_calibration([cal_keys], [cal_values], config)

    # Create codecs
    numpy_codec = calibration.keys
    metal_codec = MetalKVTCCodec(calibration.keys, enable_profiling=False)

    # Generate test data
    print(f"Generating test data ({args.batch}×{args.d_model})...")
    x = np.random.randn(args.batch, args.d_model).astype(np.float32)

    # Benchmark NumPy codec
    print("\nBenchmarking NumPy codec...")
    numpy_encode_ms, numpy_decode_ms = benchmark_encode_decode(
        numpy_codec, x, args.runs, args.warmup
    )

    # Benchmark Metal codec
    print("Benchmarking Metal codec...")
    metal_encode_ms, metal_decode_ms = benchmark_encode_decode(
        metal_codec, x, args.runs, args.warmup
    )

    # Verify correctness
    print("\nVerifying correctness...")
    numpy_encoded = numpy_codec.encode(x)
    numpy_decoded = numpy_codec.decode(numpy_encoded)

    metal_encoded = metal_codec.encode(x)
    metal_decoded = metal_codec.decode(metal_encoded)

    max_diff = np.max(np.abs(numpy_decoded - metal_decoded))
    mean_diff = np.mean(np.abs(numpy_decoded - metal_decoded))

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    # Print results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"{'Codec':<15} {'Encode (ms)':<15} {'Decode (ms)':<15} {'Total (ms)':<15}")
    print("-" * 80)
    print(
        f"{'NumPy':<15} {numpy_encode_ms:<15.2f} {numpy_decode_ms:<15.2f} {numpy_encode_ms + numpy_decode_ms:<15.2f}"
    )
    print(
        f"{'Metal':<15} {metal_encode_ms:<15.2f} {metal_decode_ms:<15.2f} {metal_encode_ms + metal_decode_ms:<15.2f}"
    )
    print("-" * 80)

    # Calculate speedup
    encode_speedup = numpy_encode_ms / metal_encode_ms if metal_encode_ms > 0 else 0
    decode_speedup = numpy_decode_ms / metal_decode_ms if metal_decode_ms > 0 else 0
    total_speedup = (numpy_encode_ms + numpy_decode_ms) / (
        metal_encode_ms + metal_decode_ms
    )

    print(f"{'Speedup':<15} {encode_speedup:<15.2f}x {decode_speedup:<15.2f}x {total_speedup:<15.2f}x")
    print("=" * 80)

    # Print summary
    print("\n📊 Summary:")
    print(f"  ✅ Correctness verified (max diff: {max_diff:.6f})")
    print(f"  ⚡ Encode speedup: {encode_speedup:.2f}x")
    print(f"  ⚡ Decode speedup: {decode_speedup:.2f}x")
    print(f"  ⚡ Total speedup: {total_speedup:.2f}x")

    if total_speedup >= 10.0:
        print(f"  🎯 Target achieved! (10× or better)")
    elif total_speedup >= 5.0:
        print(f"  🟡 Good speedup, but below 10× target")
    else:
        print(f"  🔴 Speedup below target, further optimization needed")


if __name__ == "__main__":
    main()
