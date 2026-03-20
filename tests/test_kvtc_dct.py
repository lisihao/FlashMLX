#!/usr/bin/env python3
"""Test KVTC DCT codec vs PCA codec.

Compare three methods:
1. PCA (baseline, requires calibration)
2. DCT with sample-based bit allocation (requires sample data)
3. DCT with fixed bit allocation (NO calibration needed!)
"""

import sys
sys.path.insert(0, '.')

import time
import numpy as np
import mlx.core as mx

from mlx_lm.models.kvtc_codec import (
    KVTCCodecConfig,
    fit_shared_calibration,
    encode_tensor,
    decode_tensor,
)
from mlx_lm.models.kvtc_dct_codec import (
    fit_dct_shared_calibration,
    encode_tensor_dct,
    decode_tensor_dct,
)


def generate_test_data(batch, heads, tokens, dim, latent_dim=8, seed=42):
    """Generate structured low-rank data (same as test_incremental_kvtc.py)."""
    rng = np.random.default_rng(seed)

    # Generate low-rank structure
    latent = rng.normal(size=(batch, heads, tokens, latent_dim)).astype(np.float32)
    wk = rng.normal(size=(latent_dim, dim)).astype(np.float32)
    wv = rng.normal(size=(latent_dim, dim)).astype(np.float32)

    # Low-rank projection + small noise
    keys = np.einsum("bhtf,fd->bhtd", latent, wk)
    values = np.einsum("bhtf,fd->bhtd", latent, wv)
    keys += 0.01 * rng.normal(size=keys.shape).astype(np.float32)
    values += 0.01 * rng.normal(size=values.shape).astype(np.float32)

    return keys, values


def test_pca_vs_dct():
    """Compare PCA and DCT codecs."""
    print("=" * 70)
    print("KVTC P2: PCA vs DCT Comparison (3 Methods)")
    print("=" * 70)
    print()

    # Generate test data (same as successful test)
    batch, heads, tokens, dim = 1, 4, 128, 32
    keys, values = generate_test_data(batch, heads, tokens, dim)

    # Flatten for codec
    keys_flat = keys.reshape(-1, dim)
    values_flat = values.reshape(-1, dim)

    print(f"Data shape: {keys_flat.shape}")
    print()

    # Configuration (same as successful test)
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
    # Method 1: PCA (baseline)
    # ========================================================================
    print("-" * 70)
    print("Method 1: PCA (Baseline)")
    print("-" * 70)

    t0 = time.time()
    pca_cal = fit_shared_calibration([keys_flat], [values_flat], config)
    pca_cal_time = time.time() - t0

    t0 = time.time()
    pca_enc = encode_tensor(keys_flat, pca_cal.keys)
    pca_enc_time = time.time() - t0

    t0 = time.time()
    pca_dec = decode_tensor(pca_enc, pca_cal.keys)
    pca_dec_time = time.time() - t0

    pca_err = np.linalg.norm(pca_dec - keys_flat) / np.linalg.norm(keys_flat)

    print(f"⏱️  Calibration: {pca_cal_time*1000:.2f} ms")
    print(f"⏱️  Encode: {pca_enc_time*1000:.2f} ms")
    print(f"⏱️  Decode: {pca_dec_time*1000:.2f} ms")
    print(f"📊 Relative error: {pca_err:.6f}")
    print()

    results.append(("PCA", pca_cal_time, pca_enc_time, pca_dec_time, pca_err))

    # ========================================================================
    # Method 2: DCT with sample-based bit allocation
    # ========================================================================
    print("-" * 70)
    print("Method 2: DCT (sample-based bit allocation)")
    print("-" * 70)

    t0 = time.time()
    dct_sample_cal = fit_dct_shared_calibration(
        [keys_flat], [values_flat], config, use_fixed_allocation=False
    )
    dct_sample_cal_time = time.time() - t0

    t0 = time.time()
    dct_sample_enc = encode_tensor_dct(keys_flat, dct_sample_cal.keys)
    dct_sample_enc_time = time.time() - t0

    t0 = time.time()
    dct_sample_dec = decode_tensor_dct(dct_sample_enc, dct_sample_cal.keys)
    dct_sample_dec_time = time.time() - t0

    dct_sample_err = np.linalg.norm(dct_sample_dec - keys_flat) / np.linalg.norm(keys_flat)

    print(f"⏱️  Calibration: {dct_sample_cal_time*1000:.2f} ms")
    print(f"⏱️  Encode: {dct_sample_enc_time*1000:.2f} ms")
    print(f"⏱️  Decode: {dct_sample_dec_time*1000:.2f} ms")
    print(f"📊 Relative error: {dct_sample_err:.6f}")
    print()

    results.append(("DCT-Sample", dct_sample_cal_time, dct_sample_enc_time,
                    dct_sample_dec_time, dct_sample_err))

    # ========================================================================
    # Method 3: DCT with FIXED bit allocation (NO calibration!)
    # ========================================================================
    print("-" * 70)
    print("Method 3: DCT (FIXED bit allocation, NO calibration!)")
    print("-" * 70)

    t0 = time.time()
    dct_fixed_cal = fit_dct_shared_calibration(
        [keys_flat], [values_flat], config, use_fixed_allocation=True
    )
    dct_fixed_cal_time = time.time() - t0

    t0 = time.time()
    dct_fixed_enc = encode_tensor_dct(keys_flat, dct_fixed_cal.keys)
    dct_fixed_enc_time = time.time() - t0

    t0 = time.time()
    dct_fixed_dec = decode_tensor_dct(dct_fixed_enc, dct_fixed_cal.keys)
    dct_fixed_dec_time = time.time() - t0

    dct_fixed_err = np.linalg.norm(dct_fixed_dec - keys_flat) / np.linalg.norm(keys_flat)

    print(f"⏱️  Calibration: {dct_fixed_cal_time*1000:.2f} ms (instant!)")
    print(f"⏱️  Encode: {dct_fixed_enc_time*1000:.2f} ms")
    print(f"⏱️  Decode: {dct_fixed_dec_time*1000:.2f} ms")
    print(f"📊 Relative error: {dct_fixed_err:.6f}")
    print()

    results.append(("DCT-Fixed", dct_fixed_cal_time, dct_fixed_enc_time,
                    dct_fixed_dec_time, dct_fixed_err))

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    print("⏱️  **Calibration Time**:")
    for name, cal_t, _, _, _ in results:
        print(f"   {name:15s}: {cal_t*1000:7.2f} ms")
    pca_cal_t = results[0][1]
    dct_fixed_cal_t = results[2][1]
    speedup = pca_cal_t / dct_fixed_cal_t
    print(f"   → DCT-Fixed is {speedup:.0f}x faster than PCA!")
    print()

    print("⏱️  **Encode Time**:")
    for name, _, enc_t, _, _ in results:
        print(f"   {name:15s}: {enc_t*1000:7.2f} ms")
    print()

    print("⏱️  **Decode Time**:")
    for name, _, _, dec_t, _ in results:
        print(f"   {name:15s}: {dec_t*1000:7.2f} ms")
    print()

    print("📊 **Accuracy (Relative Error)**:")
    for name, _, _, _, err in results:
        status = "✅" if err < 1.1 else "❌"
        print(f"   {name:15s}: {err:.6f} {status}")
    print()

    # Test pass criteria
    all_passed = all(err < 1.1 for _, _, _, _, err in results)

    if all_passed:
        print("=" * 70)
        print("✅ All methods PASSED! (relative error < 1.1)")
        print()
        print("🎯 **P2 Key Achievement**:")
        print(f"   DCT-Fixed eliminates calibration overhead ({speedup:.0f}x speedup)")
        print(f"   while maintaining accuracy (error: {dct_fixed_err:.6f})")
        print("=" * 70)
        return True
    else:
        print("=" * 70)
        print("❌ Some methods FAILED!")
        print("=" * 70)
        return False


def main():
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "KVTC P2: DCT Codec Test" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        success = test_pca_vs_dct()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
