#!/usr/bin/env python3
"""
Final comparison: PolarQuant vs TurboAngle Optimized

Fair comparison with both implementations optimized.
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import time
import mlx.core as mx
from mlx_lm.models.quantization_strategies import PolarQuantizer
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
    print(f"{name}")
    print('='*80)

    # Warmup
    _ = quantizer.quantize(keys[:, :, :10, :], values[:, :, :10, :])
    mx.eval(_)

    # Quantize
    start = time.perf_counter()
    quant_k, quant_v, metadata = quantizer.quantize(keys, values)
    mx.eval(quant_k)
    mx.eval(quant_v)
    quant_time = time.perf_counter() - start

    # Measure size
    def get_size(obj):
        if isinstance(obj, dict):
            total = 0
            for val in obj.values():
                if hasattr(val, 'size') and hasattr(val, 'itemsize'):
                    total += val.size * val.itemsize
            return total
        elif hasattr(obj, 'size') and hasattr(obj, 'itemsize'):
            return obj.size * obj.itemsize
        return 0

    quant_size_bytes = get_size(quant_k) + get_size(quant_v)
    quant_size_mb = quant_size_bytes / (1024**2)

    # Dequantize
    start = time.perf_counter()
    rec_k, rec_v = quantizer.dequantize(quant_k, quant_v, metadata)
    mx.eval(rec_k)
    mx.eval(rec_v)
    dequant_time = time.perf_counter() - start

    # Quality
    sim_k = compute_cosine_similarity(keys, rec_k)
    sim_v = compute_cosine_similarity(values, rec_v)

    # Stats
    orig_size_mb = (keys.size * 2 + values.size * 2) / (1024**2)
    compression = orig_size_mb / quant_size_mb

    print(f"  Compression:    {compression:.2f}× (theoretical: {quantizer.get_compression_ratio():.2f}×)")
    print(f"  Quantized size: {quant_size_mb:.2f} MB (vs {orig_size_mb:.2f} MB original)")
    print(f"  Quality:        K={sim_k:.6f}, V={sim_v:.6f}")
    print(f"  Speed:          Q={quant_time*1000:.2f}ms, DQ={dequant_time*1000:.2f}ms")

    return {
        'name': name,
        'compression': compression,
        'size_mb': quant_size_mb,
        'sim_k': sim_k,
        'sim_v': sim_v,
        'quant_ms': quant_time * 1000,
        'dequant_ms': dequant_time * 1000,
    }


def main():
    """Run final comparison."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "FINAL: PolarQuant vs TurboAngle" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Test at larger scale
    B, n_heads, seq_len, head_dim = 1, 32, 4096, 128

    print(f"Test: B={B}, heads={n_heads}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"Original size: {(2*B*n_heads*seq_len*head_dim*2)/(1024**2):.1f} MB (bf16)\n")

    # Generate data
    keys = mx.random.normal(shape=(B, n_heads, seq_len, head_dim)).astype(mx.bfloat16)
    values = mx.random.normal(shape=(B, n_heads, seq_len, head_dim)).astype(mx.bfloat16)

    results = []

    # Test PolarQuant
    pq = PolarQuantizer(bits=4)
    results.append(test_quantizer(pq, keys, values, "PolarQuant 4-bit"))

    # Test TurboAngle Optimized
    ta = TurboAngleQuantizerOptimized(n_k=128, n_v=64, head_dim=head_dim)
    results.append(test_quantizer(ta, keys, values, "TurboAngle Optimized (K128V64)"))

    # TurboAngle E4
    ta_e4 = TurboAngleQuantizerOptimized(n_k=256, n_v=128, head_dim=head_dim)
    results.append(test_quantizer(ta_e4, keys, values, "TurboAngle Optimized E4 (K256V128)"))

    # Summary
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 30 + "VERDICT" + " " * 41 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    baseline = results[0]  # PolarQuant

    print(f"{'Method':<35} {'Compression':>12} {'Quality':>12} {'Speed':>12}")
    print("-" * 80)

    for r in results:
        avg_sim = (r['sim_k'] + r['sim_v']) / 2
        total_time = r['quant_ms'] + r['dequant_ms']

        # Relative to PolarQuant
        comp_vs_pq = f"{r['compression']/baseline['compression']:.2f}×" if r != baseline else "baseline"

        print(f"{r['name']:<35} {r['compression']:>10.2f}× "
              f"{avg_sim:>11.5f} {total_time:>9.1f}ms")

    print()
    print("Analysis:")
    print()

    pq_res = results[0]
    ta_res = results[1]
    ta_e4_res = results[2]

    # Compression
    print(f"Compression:")
    print(f"  PolarQuant:       {pq_res['compression']:.2f}×")
    print(f"  TurboAngle:       {ta_res['compression']:.2f}× "
          f"({pq_res['compression']/ta_res['compression']:.1f}× less than PolarQuant)")
    print(f"  TurboAngle E4:    {ta_e4_res['compression']:.2f}×")
    print()

    # Quality
    pq_sim = (pq_res['sim_k'] + pq_res['sim_v']) / 2
    ta_sim = (ta_res['sim_k'] + ta_res['sim_v']) / 2

    print(f"Quality (avg similarity):")
    print(f"  PolarQuant:       {pq_sim:.5f}")
    print(f"  TurboAngle:       {ta_sim:.5f} ({ta_sim - pq_sim:+.5f})")

    if abs(ta_sim - pq_sim) < 0.001:
        print(f"  → Essentially identical quality")
    elif ta_sim > pq_sim:
        print(f"  → TurboAngle slightly better")
    else:
        print(f"  → PolarQuant slightly better")
    print()

    # Speed
    pq_time = pq_res['quant_ms'] + pq_res['dequant_ms']
    ta_time = ta_res['quant_ms'] + ta_res['dequant_ms']

    print(f"Speed (total Q+DQ):")
    print(f"  PolarQuant:       {pq_time:.1f}ms")
    print(f"  TurboAngle:       {ta_time:.1f}ms ({ta_time/pq_time:.2f}×)")

    if ta_time < pq_time * 1.2:
        print(f"  → Competitive speed")
    else:
        print(f"  → PolarQuant faster")
    print()

    # Final verdict
    print("="*80)
    print("CONCLUSION:")
    print("="*80)
    print()

    if pq_res['compression'] > ta_res['compression'] * 1.3:
        print("✅ **PolarQuant wins on compression** (significantly better)")
        print(f"   Saves {pq_res['size_mb'] - ta_res['size_mb']:.1f} MB more than TurboAngle")
    else:
        print("➖ Compression: comparable")

    if abs(ta_sim - pq_sim) < 0.001:
        print("➖ Quality: identical (both near-lossless)")
    elif ta_sim > pq_sim + 0.002:
        print("✅ **TurboAngle wins on quality** (measurably better)")
    elif pq_sim > ta_sim + 0.002:
        print("✅ **PolarQuant wins on quality** (measurably better)")
    else:
        print("➖ Quality: essentially identical")

    if pq_time < ta_time * 0.8:
        print("✅ **PolarQuant wins on speed** (significantly faster)")
    elif ta_time < pq_time * 0.8:
        print("✅ **TurboAngle wins on speed** (significantly faster)")
    else:
        print("➖ Speed: competitive")

    print()


if __name__ == "__main__":
    main()
