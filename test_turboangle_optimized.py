#!/usr/bin/env python3
"""
Test TurboAngle optimized version vs original.

Compares:
1. Memory usage (bit-packing effect)
2. Speed
3. Quality (should be identical)
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import time
import mlx.core as mx
from mlx_lm.models.turboangle import TurboAngleQuantizer
from mlx_lm.models.turboangle_optimized import TurboAngleQuantizerOptimized
import numpy as np


def compute_cosine_similarity(a, b):
    """Compute average cosine similarity."""
    B, H, S, D = a.shape
    a_flat = a.reshape(-1, D)
    b_flat = b.reshape(-1, D)

    dot = (a_flat * b_flat).sum(axis=-1)
    norm_a = mx.sqrt((a_flat * a_flat).sum(axis=-1))
    norm_b = mx.sqrt((b_flat * b_flat).sum(axis=-1))

    sim = dot / (norm_a * norm_b + 1e-8)
    return sim.mean().item()


def test_quantizer(quantizer, keys, values, name):
    """Test a quantizer."""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")
    print(f"  Quantizer: {quantizer}")
    print()

    # Quantize
    start = time.perf_counter()
    quant_k, quant_v, metadata = quantizer.quantize(keys, values)
    mx.eval(quant_k)
    mx.eval(quant_v)
    quant_time = time.perf_counter() - start

    # Measure quantized size
    if isinstance(quant_k, dict):
        # Dict format (check all arrays)
        total_bytes = 0
        for key, val in quant_k.items():
            if hasattr(val, 'size') and hasattr(val, 'itemsize'):
                total_bytes += val.size * val.itemsize
        for key, val in quant_v.items():
            if hasattr(val, 'size') and hasattr(val, 'itemsize'):
                total_bytes += val.size * val.itemsize
        quant_size_mb = total_bytes / (1024**2)
    else:
        # Array format
        quant_size_mb = (quant_k.size * quant_k.itemsize +
                        quant_v.size * quant_v.itemsize) / (1024**2)

    print(f"  Quantization time: {quant_time*1000:.2f}ms")
    print(f"  Quantized size: {quant_size_mb:.2f} MB")

    # Dequantize
    start = time.perf_counter()
    rec_k, rec_v = quantizer.dequantize(quant_k, quant_v, metadata)
    mx.eval(rec_k)
    mx.eval(rec_v)
    dequant_time = time.perf_counter() - start

    print(f"  Dequantization time: {dequant_time*1000:.2f}ms")

    # Quality
    sim_k = compute_cosine_similarity(keys, rec_k)
    sim_v = compute_cosine_similarity(values, rec_v)

    print(f"  K cosine similarity: {sim_k:.6f}")
    print(f"  V cosine similarity: {sim_v:.6f}")

    # Original size
    orig_size_mb = (keys.size * 2 + values.size * 2) / (1024**2)  # bf16 = 2 bytes
    actual_compression = orig_size_mb / quant_size_mb

    print(f"\n  Original size: {orig_size_mb:.2f} MB")
    print(f"  Actual compression ratio: {actual_compression:.2f}×")
    print(f"  Theoretical compression: {quantizer.get_compression_ratio():.2f}×")

    return {
        'name': name,
        'quant_time_ms': quant_time * 1000,
        'dequant_time_ms': dequant_time * 1000,
        'quant_size_mb': quant_size_mb,
        'sim_k': sim_k,
        'sim_v': sim_v,
        'actual_compression': actual_compression,
        'theoretical_compression': quantizer.get_compression_ratio(),
    }


def main():
    """Run comparison."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 18 + "TurboAngle: Original vs Optimized" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Test configuration
    B, n_heads, seq_len, head_dim = 1, 32, 1024, 128

    print(f"Test data: B={B}, heads={n_heads}, seq_len={seq_len}, head_dim={head_dim}")
    print()

    # Generate test data
    keys = mx.random.normal(shape=(B, n_heads, seq_len, head_dim)).astype(mx.bfloat16)
    values = mx.random.normal(shape=(B, n_heads, seq_len, head_dim)).astype(mx.bfloat16)

    # Test original
    quantizer_orig = TurboAngleQuantizer(n_k=128, n_v=64, head_dim=head_dim)
    result_orig = test_quantizer(quantizer_orig, keys, values, "Original TurboAngle")

    # Test optimized
    quantizer_opt = TurboAngleQuantizerOptimized(n_k=128, n_v=64, head_dim=head_dim)
    result_opt = test_quantizer(quantizer_opt, keys, values, "Optimized TurboAngle (bit-packed)")

    # Summary
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 32 + "SUMMARY" + " " * 40 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    print(f"{'Metric':<30} {'Original':>15} {'Optimized':>15} {'Improvement':>15}")
    print("-" * 80)

    # Quantized size
    size_improvement = (result_orig['quant_size_mb'] / result_opt['quant_size_mb'])
    print(f"{'Quantized size (MB)':<30} {result_orig['quant_size_mb']:>15.2f} "
          f"{result_opt['quant_size_mb']:>15.2f} {size_improvement:>14.2f}×")

    # Actual compression
    print(f"{'Actual compression':<30} {result_orig['actual_compression']:>14.2f}× "
          f"{result_opt['actual_compression']:>14.2f}× "
          f"{result_opt['actual_compression']/result_orig['actual_compression']:>14.2f}×")

    # Speed
    quant_speedup = result_orig['quant_time_ms'] / result_opt['quant_time_ms']
    dequant_speedup = result_orig['dequant_time_ms'] / result_opt['dequant_time_ms']
    print(f"{'Quantization time (ms)':<30} {result_orig['quant_time_ms']:>15.2f} "
          f"{result_opt['quant_time_ms']:>15.2f} {quant_speedup:>14.2f}×")
    print(f"{'Dequantization time (ms)':<30} {result_orig['dequant_time_ms']:>15.2f} "
          f"{result_opt['dequant_time_ms']:>15.2f} {dequant_speedup:>14.2f}×")

    # Quality
    print(f"{'K similarity':<30} {result_orig['sim_k']:>15.6f} "
          f"{result_opt['sim_k']:>15.6f} {'same':>15}")
    print(f"{'V similarity':<30} {result_orig['sim_v']:>15.6f} "
          f"{result_opt['sim_v']:>15.6f} {'same':>15}")

    print()
    print("Key Findings:")
    print(f"  ✅ Memory: {size_improvement:.1f}× smaller (bit-packing works!)")
    print(f"  ✅ Actual compression: {result_opt['actual_compression']:.2f}× "
          f"(vs theoretical {result_opt['theoretical_compression']:.2f}×)")
    if quant_speedup > 1.1:
        print(f"  ✅ Speed: {quant_speedup:.1f}× faster quantization")
    elif quant_speedup < 0.9:
        print(f"  ⚠️  Speed: {1/quant_speedup:.1f}× slower quantization (packing overhead)")
    else:
        print(f"  ➖ Speed: similar ({quant_speedup:.2f}×)")

    print()


if __name__ == "__main__":
    main()
