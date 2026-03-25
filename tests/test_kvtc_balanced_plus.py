#!/usr/bin/env python3
"""Test KVTC Balanced+ (30% + 6-bit 超高精度).

目标：
- 压缩率：35-45x (推理速度最优区间)
- 精度：相对误差 0.45-0.55 (比方案 A 的 0.83 好 40%+)

策略：
- 保留 30% 系数
- 使用 6-bit 超高精度
- 精细分级：6-bit / 5-bit / 4-bit / 3-bit / 2-bit / 0-bit
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
    KVTCMagnitudeTieredConfig,
    fit_magnitude_tiered_calibration,
    encode_tensor_magnitude_tiered,
    decode_tensor_magnitude_tiered,
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
        if len(encoded) == 6:
            # Tiered: (tier_payloads, tier_shifts, tier_scales, tier_q_shapes, tier_indices, orig_shape)
            tier_payloads, tier_shifts, tier_scales, tier_q_shapes, tier_indices, orig_shape = encoded
            for p, s, sc, idx in zip(tier_payloads, tier_shifts, tier_scales, tier_indices):
                total_bytes += p.nbytes
                total_bytes += s.nbytes
                total_bytes += sc.nbytes
                total_bytes += idx.nbytes
        elif len(encoded) == 5:
            # DCT: (payloads, shifts, scales, q_shapes, orig_shape)
            payloads, shifts, scales, q_shapes, orig_shape = encoded
            for p, s, sc in zip(payloads, shifts, scales):
                total_bytes += p.nbytes
                total_bytes += s.nbytes
                total_bytes += sc.nbytes

    return total_bytes


def test_balanced_plus():
    """Test Balanced+ (30% + 6-bit)."""
    print("=" * 70)
    print("KVTC Balanced+: 30% + 6-bit 超高精度")
    print("=" * 70)
    print()

    # Generate test data
    batch, heads, tokens, dim = 1, 4, 1024, 32
    keys, values = generate_test_data(batch, heads, tokens, dim)

    keys_flat = keys.reshape(-1, dim)
    values_flat = values.reshape(-1, dim)

    print(f"Data shape: {keys_flat.shape}")
    print(f"Original size: {keys_flat.nbytes / 1024:.2f} KB")
    print()

    results = []

    # ========================================================================
    # Baseline: DCT-Fixed
    # ========================================================================
    print("-" * 70)
    print("Baseline: DCT-Fixed")
    print("-" * 70)

    config = KVTCCodecConfig(rank=8, bits=4, group_size=16)

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

    results.append(("DCT-Fixed", dct_enc_time, dct_dec_time, dct_err, dct_size))

    # ========================================================================
    # Balanced+ (30% + 6-bit 超高精度)
    # ========================================================================
    print("-" * 70)
    print("Balanced+ (30% + 6-bit 超高精度)")
    print("说明: Top 5% → 6-bit, 5-10% → 5-bit, 10-20% → 4-bit,")
    print("      20-25% → 3-bit, 25-30% → 2-bit, 30-100% → 0-bit")
    print("目标: 压缩率 35-45x, 精度 0.45-0.55")
    print("-" * 70)

    tiered_config = KVTCMagnitudeTieredConfig(
        tier_ratios=(0.05, 0.10, 0.20, 0.25, 0.30),
        tier_bits=(6, 5, 4, 3, 2, 0),
        pruning_method="l2",
    )

    config_6bit = KVTCCodecConfig(rank=8, bits=6, group_size=16)

    t0 = time.time()
    tier_cal = fit_magnitude_tiered_calibration([keys_flat], [values_flat], config_6bit, tiered_config)
    tier_cal_time = time.time() - t0

    t0 = time.time()
    tier_enc = encode_tensor_magnitude_tiered(keys_flat, tier_cal.keys)
    tier_enc_time = time.time() - t0

    t0 = time.time()
    tier_dec = decode_tensor_magnitude_tiered(tier_enc, tier_cal.keys)
    tier_dec_time = time.time() - t0

    tier_err = np.linalg.norm(tier_dec - keys_flat) / np.linalg.norm(keys_flat)
    tier_size = measure_compression_ratio(tier_enc)

    print(f"⏱️  Encode: {tier_enc_time*1000:.2f} ms")
    print(f"⏱️  Decode: {tier_dec_time*1000:.2f} ms")
    print(f"📊 Relative error: {tier_err:.6f}")
    print(f"💾 Compressed size: {tier_size / 1024:.2f} KB")
    print(f"📦 Compression ratio: {keys_flat.nbytes / tier_size:.2f}x")
    print(f"🎯 vs DCT-Fixed: 压缩率 {(tier_size/dct_size - 1)*100:+.1f}%, "
          f"精度提升 {(1 - tier_err/dct_err)*100:.1f}%")
    print()

    results.append(("Balanced+ (30%)", tier_enc_time, tier_dec_time, tier_err, tier_size))

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    print("📊 **结果对比**:")
    print(f"   {'Strategy':25s}  {'Compression':>12s}  {'Accuracy':>15s}")
    print(f"   {'-'*25}  {'-'*12}  {'-'*15}")
    for name, _, _, err, size in results:
        comp_ratio = keys_flat.nbytes / size
        print(f"   {name:25s}  {comp_ratio:11.2f}x  {err:14.6f}")
    print()

    # Check if target is met
    target_comp = 35  # 目标压缩率
    target_err = 0.55  # 目标误差

    balanced_comp = keys_flat.nbytes / tier_size
    balanced_err = tier_err

    print("🎯 **目标达成情况**:")
    if balanced_comp >= target_comp and balanced_err <= target_err:
        print(f"   ✅ 压缩率: {balanced_comp:.2f}x (目标 ≥{target_comp}x)")
        print(f"   ✅ 精度: {balanced_err:.6f} (目标 ≤{target_err})")
        print(f"   ✅ 相比 DCT-Fixed 精度提升: {(1 - balanced_err/dct_err)*100:.1f}%")
        success = True
    else:
        print(f"   压缩率: {balanced_comp:.2f}x (目标 ≥{target_comp}x) {'✅' if balanced_comp >= target_comp else '❌'}")
        print(f"   精度: {balanced_err:.6f} (目标 ≤{target_err}) {'✅' if balanced_err <= target_err else '❌'}")
        success = balanced_comp >= target_comp * 0.8 and balanced_err <= target_err * 1.2
    print()

    return success


def main():
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 16 + "KVTC Balanced+ (30% + 6-bit)" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        success = test_balanced_plus()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
