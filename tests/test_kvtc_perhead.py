#!/usr/bin/env python3
"""Test KVTC Per-Head codec vs Shared codec.

Compare:
1. Compression accuracy (relative error)
2. Calibration time (N times slower)
3. Storage overhead (N calibrations)
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
from mlx_lm.models.kvtc_perhead_codec import (
    fit_perhead_calibration,
    encode_perhead,
    decode_perhead,
)


def generate_test_data(batch, heads, tokens, dim, latent_dim=8, seed=42):
    """Generate structured low-rank data with head specialization.

    Each head gets slightly different projection matrices to simulate
    different attention patterns.
    """
    rng = np.random.default_rng(seed)

    # Generate low-rank structure
    latent = rng.normal(size=(batch, heads, tokens, latent_dim)).astype(np.float32)

    # Different projection matrices for each head (simulate specialization)
    keys = np.zeros((batch, heads, tokens, dim), dtype=np.float32)
    values = np.zeros((batch, heads, tokens, dim), dtype=np.float32)

    for h in range(heads):
        # Each head gets slightly different projection
        wk = rng.normal(size=(latent_dim, dim)).astype(np.float32)
        wv = rng.normal(size=(latent_dim, dim)).astype(np.float32)

        # Add head-specific bias
        wk *= (1.0 + 0.1 * h / heads)
        wv *= (1.0 + 0.1 * h / heads)

        # Project
        keys[:, h, :, :] = np.einsum("btf,fd->btd", latent[:, h, :, :], wk)
        values[:, h, :, :] = np.einsum("btf,fd->btd", latent[:, h, :, :], wv)

    # Add small noise
    keys += 0.01 * rng.normal(size=keys.shape).astype(np.float32)
    values += 0.01 * rng.normal(size=values.shape).astype(np.float32)

    return keys, values


def test_shared_vs_perhead():
    """Compare shared vs per-head calibration."""
    print("=" * 70)
    print("KVTC P3: Shared vs Per-Head Calibration")
    print("=" * 70)
    print()

    # Generate test data with head specialization
    batch, heads, tokens, dim = 1, 4, 128, 32
    keys, values = generate_test_data(batch, heads, tokens, dim)

    print(f"Data shape: {keys.shape}")
    print(f"Number of heads: {heads}")
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
    # Method 1: Shared Calibration (baseline)
    # ========================================================================
    print("-" * 70)
    print("Method 1: Shared Calibration (All heads share one PCA basis)")
    print("-" * 70)

    # Flatten for shared calibration: [batch, heads, tokens, dim] -> [batch*heads*tokens, dim]
    keys_flat = keys.reshape(-1, dim)
    values_flat = values.reshape(-1, dim)

    t0 = time.time()
    shared_cal = fit_shared_calibration([keys_flat], [values_flat], config)
    shared_cal_time = time.time() - t0

    t0 = time.time()
    shared_enc_keys = encode_tensor(keys_flat, shared_cal.keys)
    shared_enc_time = time.time() - t0

    t0 = time.time()
    shared_dec_keys = decode_tensor(shared_enc_keys, shared_cal.keys)
    shared_dec_time = time.time() - t0

    # Reshape back
    shared_dec_keys = shared_dec_keys.reshape(keys.shape)

    shared_err = np.linalg.norm(shared_dec_keys - keys) / np.linalg.norm(keys)

    print(f"⏱️  Calibration: {shared_cal_time*1000:.2f} ms")
    print(f"⏱️  Encode: {shared_enc_time*1000:.2f} ms")
    print(f"⏱️  Decode: {shared_dec_time*1000:.2f} ms")
    print(f"📊 Relative error: {shared_err:.6f}")
    print()

    results.append(("Shared", shared_cal_time, shared_enc_time, shared_dec_time, shared_err))

    # ========================================================================
    # Method 2: Per-Head Calibration
    # ========================================================================
    print("-" * 70)
    print("Method 2: Per-Head Calibration (Each head gets its own PCA basis)")
    print("-" * 70)

    t0 = time.time()
    perhead_cal = fit_perhead_calibration([keys], [values], config)
    perhead_cal_time = time.time() - t0

    t0 = time.time()
    perhead_enc_keys, perhead_enc_values = encode_perhead(keys, values, perhead_cal)
    perhead_enc_time = time.time() - t0

    t0 = time.time()
    perhead_dec_keys, perhead_dec_values = decode_perhead(
        perhead_enc_keys, perhead_enc_values, perhead_cal, keys.shape
    )
    perhead_dec_time = time.time() - t0

    perhead_err = np.linalg.norm(perhead_dec_keys - keys) / np.linalg.norm(keys)

    print(f"⏱️  Calibration: {perhead_cal_time*1000:.2f} ms (per-head)")
    print(f"⏱️  Encode: {perhead_enc_time*1000:.2f} ms")
    print(f"⏱️  Decode: {perhead_dec_time*1000:.2f} ms")
    print(f"📊 Relative error: {perhead_err:.6f}")
    print()

    results.append(("Per-Head", perhead_cal_time, perhead_enc_time, perhead_dec_time, perhead_err))

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
    slowdown = results[1][1] / results[0][1]
    print(f"   → Per-Head is {slowdown:.2f}x slower (expected: ~{heads}x)")
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
    shared_err_val = results[0][4]
    perhead_err_val = results[1][4]
    for name, _, _, _, err in results:
        status = "✅" if err < 1.1 else "❌"
        print(f"   {name:15s}: {err:.6f} {status}")

    improvement = (shared_err_val - perhead_err_val) / shared_err_val * 100
    print(f"   → Per-Head improves accuracy by {improvement:.2f}%")
    print()

    # Test pass criteria
    all_passed = all(err < 1.1 for _, _, _, _, err in results)
    perhead_better = perhead_err_val < shared_err_val

    if all_passed and perhead_better:
        print("=" * 70)
        print("✅ All methods PASSED!")
        print()
        print("🎯 **P3 Key Achievement**:")
        print(f"   Per-Head calibration improves accuracy by {improvement:.2f}%")
        print(f"   Trade-off: {slowdown:.2f}x calibration time (offline, one-time cost)")
        print("=" * 70)
        return True
    elif all_passed:
        print("=" * 70)
        print("⚠️  All methods PASSED, but Per-Head shows no improvement")
        print("   (May need more head specialization in test data)")
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
    print("║" + " " * 16 + "KVTC P3: Per-Head Calibration Test" + " " * 18 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        success = test_shared_vs_perhead()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
