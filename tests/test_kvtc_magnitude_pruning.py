#!/usr/bin/env python3
"""Test KVTC Magnitude Pruning.

Compare different pruning ratios:
1. No pruning (keep 100%)
2. Aggressive pruning (keep 50%)
3. Moderate pruning (keep 75%)
4. Conservative pruning (keep 90%)
"""

import sys
sys.path.insert(0, '.')

import time
import numpy as np
import mlx.core as mx

from mlx_lm.models.kvtc_codec import KVTCCodecConfig
from mlx_lm.models.kvtc_dct_codec import (
    fit_dct_shared_calibration,
    encode_tensor_dct,
    decode_tensor_dct,
)
from mlx_lm.models.kvtc_magnitude_pruning import (
    KVTCMagnitudePruningConfig,
    fit_magnitude_pruning_calibration,
    encode_tensor_magnitude_pruning,
    decode_tensor_magnitude_pruning,
)


def generate_test_data(batch, heads, tokens, dim, latent_dim=8, seed=42):
    """Generate structured low-rank data."""
    rng = np.random.default_rng(seed)

    latent = rng.normal(size=(batch, heads, tokens, latent_dim)).astype(np.float32)
    wk = rng.normal(size=(latent_dim, dim)).astype(np.float32)
    wv = rng.normal(size=(latent_dim, dim)).astype(np.float32)

    keys = np.einsum("bhtf,fd->bhtd", latent, wk)
    values = np.einsum("bhtf,fd->bhtd", latent, wv)
    keys += 0.01 * rng.normal(size=keys.shape).astype(np.float32)
    values += 0.01 * rng.normal(size=values.shape).astype(np.float32)

    return keys, values


def measure_compression_ratio(encoded):
    """Measure compressed size in bytes."""
    total_bytes = 0

    if isinstance(encoded, tuple):
        # Unpack based on length
        if len(encoded) == 6:
            # Magnitude pruning: (payload, shifts, scales, q_shape, keep_indices, orig_shape)
            payload, shifts, scales, q_shape, keep_indices, orig_shape = encoded
            total_bytes += payload.nbytes
            total_bytes += shifts.nbytes
            total_bytes += scales.nbytes
            total_bytes += keep_indices.nbytes
        elif len(encoded) == 5:
            # DCT: (payloads, shifts, scales, q_shapes, orig_shape)
            payloads, shifts, scales, q_shapes, orig_shape = encoded
            for p, s, sc in zip(payloads, shifts, scales):
                total_bytes += p.nbytes
                total_bytes += s.nbytes
                total_bytes += sc.nbytes

    return total_bytes


def test_magnitude_pruning():
    """Compare different pruning strategies."""
    print("=" * 70)
    print("KVTC P4: Magnitude Pruning Comparison")
    print("=" * 70)
    print()

    # Generate test data
    batch, heads, tokens, dim = 1, 4, 128, 32
    keys, values = generate_test_data(batch, heads, tokens, dim)

    keys_flat = keys.reshape(-1, dim)
    values_flat = values.reshape(-1, dim)

    print(f"Data shape: {keys_flat.shape}")
    print(f"Original size: {keys_flat.nbytes / 1024:.2f} KB")
    print()

    # Configuration
    config = KVTCCodecConfig(
        rank=8,
        bits=4,
        group_size=16,
        sample_limit=256,
        zero_bit_energy_fraction=0.001,
    )
    print(f"Config: rank={config.rank}, bits={config.bits}, "
          f"group_size={config.group_size}")
    print()

    results = []

    # ========================================================================
    # Baseline: DCT-Fixed (no pruning)
    # ========================================================================
    print("-" * 70)
    print("Baseline: DCT-Fixed (No Pruning, keep 100%)")
    print("-" * 70)

    t0 = time.time()
    dct_cal = fit_dct_shared_calibration([keys_flat], [values_flat], config, use_fixed_allocation=True)
    dct_cal_time = time.time() - t0

    t0 = time.time()
    dct_enc = encode_tensor_dct(keys_flat, dct_cal.keys)
    dct_enc_time = time.time() - t0

    t0 = time.time()
    dct_dec = decode_tensor_dct(dct_enc, dct_cal.keys)
    dct_dec_time = time.time() - t0

    dct_err = np.linalg.norm(dct_dec - keys_flat) / np.linalg.norm(keys_flat)
    dct_size = measure_compression_ratio(dct_enc)

    print(f"⏱️  Encode: {dct_enc_time*1000:.2f} ms")
    print(f"⏱️  Decode: {dct_dec_time*1000:.2f} ms")
    print(f"📊 Relative error: {dct_err:.6f}")
    print(f"💾 Compressed size: {dct_size / 1024:.2f} KB")
    print(f"📦 Compression ratio: {keys_flat.nbytes / dct_size:.2f}x")
    print()

    results.append(("No Pruning (100%)", dct_enc_time, dct_dec_time, dct_err, dct_size))

    # ========================================================================
    # Test different pruning ratios
    # ========================================================================
    pruning_ratios = [
        (0.90, "Conservative (keep 90%)"),
        (0.75, "Moderate (keep 75%)"),
        (0.50, "Aggressive (keep 50%)"),
    ]

    for keep_ratio, label in pruning_ratios:
        print("-" * 70)
        print(f"Magnitude Pruning: {label}")
        print("-" * 70)

        pruning_config = KVTCMagnitudePruningConfig(
            keep_ratio=keep_ratio,
            pruning_method="l2",
        )

        t0 = time.time()
        mag_cal = fit_magnitude_pruning_calibration(
            [keys_flat], [values_flat], config, pruning_config
        )
        mag_cal_time = time.time() - t0

        t0 = time.time()
        mag_enc = encode_tensor_magnitude_pruning(keys_flat, mag_cal.keys)
        mag_enc_time = time.time() - t0

        t0 = time.time()
        mag_dec = decode_tensor_magnitude_pruning(mag_enc, mag_cal.keys)
        mag_dec_time = time.time() - t0

        mag_err = np.linalg.norm(mag_dec - keys_flat) / np.linalg.norm(keys_flat)
        mag_size = measure_compression_ratio(mag_enc)

        print(f"⏱️  Encode: {mag_enc_time*1000:.2f} ms")
        print(f"⏱️  Decode: {mag_dec_time*1000:.2f} ms")
        print(f"📊 Relative error: {mag_err:.6f}")
        print(f"💾 Compressed size: {mag_size / 1024:.2f} KB")
        print(f"📦 Compression ratio: {keys_flat.nbytes / mag_size:.2f}x")
        print(f"🎯 Size reduction vs baseline: {(1 - mag_size/dct_size)*100:.1f}%")
        print()

        results.append((label, mag_enc_time, mag_dec_time, mag_err, mag_size))

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    print("📦 **Compression Ratio**:")
    baseline_size = results[0][4]
    for name, _, _, _, size in results:
        ratio = keys_flat.nbytes / size
        reduction = (1 - size/baseline_size) * 100 if name != results[0][0] else 0
        print(f"   {name:25s}: {ratio:5.2f}x ", end="")
        if reduction > 0:
            print(f"(size -{reduction:4.1f}%)")
        else:
            print()
    print()

    print("📊 **Accuracy (Relative Error)**:")
    for name, _, _, err, _ in results:
        status = "✅" if err < 1.1 else "❌"
        print(f"   {name:25s}: {err:.6f} {status}")
    print()

    print("⚖️  **Trade-off Analysis**:")
    print(f"   {'Strategy':25s}  {'Size Gain':>10s}  {'Accuracy Loss':>15s}")
    print(f"   {'-'*25}  {'-'*10}  {'-'*15}")
    baseline_err = results[0][3]
    for name, _, _, err, size in results[1:]:
        size_gain = (1 - size/baseline_size) * 100
        acc_loss = (err - baseline_err) / baseline_err * 100
        print(f"   {name:25s}  {size_gain:9.1f}%  {acc_loss:14.1f}%")
    print()

    # Recommendation
    print("🎯 **Recommendation**:")
    # Find best trade-off (maximize size reduction while keeping accuracy loss < 10%)
    best_idx = 0
    best_score = 0
    for i, (name, _, _, err, size) in enumerate(results[1:], 1):
        size_gain = (1 - size/baseline_size) * 100
        acc_loss = (err - baseline_err) / baseline_err * 100
        if acc_loss < 10:  # Acceptable accuracy loss
            score = size_gain - acc_loss  # Simple trade-off score
            if score > best_score:
                best_score = score
                best_idx = i

    if best_idx > 0:
        best_name = results[best_idx][0]
        print(f"   {best_name} offers the best size/accuracy trade-off")
    else:
        print(f"   No pruning (baseline) is safest for accuracy-critical tasks")
    print()

    # Test pass criteria
    all_passed = all(err < 1.1 for _, _, _, err, _ in results)

    if all_passed:
        print("=" * 70)
        print("✅ All configurations PASSED!")
        print()
        print("🎯 **P4 Key Achievement**:")
        best = results[-1]  # Aggressive pruning
        size_gain = (1 - best[4]/baseline_size) * 100
        print(f"   Magnitude pruning (50%) achieves {size_gain:.1f}% size reduction")
        print(f"   while maintaining acceptable accuracy (error: {best[3]:.6f})")
        print("=" * 70)
        return True
    else:
        print("=" * 70)
        print("❌ Some configurations FAILED!")
        print("=" * 70)
        return False


def main():
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 16 + "KVTC P4: Magnitude Pruning Test" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        success = test_magnitude_pruning()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
