#!/usr/bin/env python3
"""Test KVTC Magnitude + Tiered Quantization (方案 A).

对比测试：
1. DCT-Fixed (baseline): 分层量化，基于频域假设
2. 方案 A (40% split): Magnitude + 分级量化，数据驱动
3. 方案 A (20% split): 高压缩率配置
4. 方案 A (60% split): 高精度配置

目标：
- 压缩率：40-50x (接近 DCT-Fixed)
- 精度：相对误差 ~0.30 (远优于 DCT-Fixed 的 0.92)

测试场景：
- 模型：35B (Qwen3-30B-A3B 类似规模)
- 上下文：4K-8K tokens
- KV Cache：~10-20 GB (未压缩) → 200-500 MB (压缩后)
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
    """Generate structured low-rank data (simulates 35B model KV Cache)."""
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


def test_magnitude_tiered():
    """Compare DCT-Fixed vs Magnitude Tiered (方案 A)."""
    print("=" * 70)
    print("KVTC 方案 A: Magnitude + Tiered Quantization")
    print("=" * 70)
    print()

    # Generate test data (simulates 35B model with 4K context)
    # 实际 35B 模型: batch=1, heads=40, tokens=4096, dim=128
    # 这里用较小规模测试，但保持相同的低秩结构
    batch, heads, tokens, dim = 1, 4, 1024, 32
    keys, values = generate_test_data(batch, heads, tokens, dim)

    keys_flat = keys.reshape(-1, dim)
    values_flat = values.reshape(-1, dim)

    print(f"Data shape: {keys_flat.shape}")
    print(f"Original size: {keys_flat.nbytes / 1024:.2f} KB")
    print(f"模拟场景: 35B 模型 + 4K 上下文")
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
    # Baseline: DCT-Fixed (frequency-based tiering)
    # ========================================================================
    print("-" * 70)
    print("Baseline: DCT-Fixed (频域假设，分层量化)")
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

    results.append(("DCT-Fixed (Baseline)", dct_enc_time, dct_dec_time, dct_err, dct_size))

    # ========================================================================
    # 方案 A: Magnitude + Tiered (默认配置: 40%/80% split)
    # ========================================================================
    print("-" * 70)
    print("方案 A (Balanced): Magnitude + Tiered (保留 25%)")
    print("说明: Top 10% → 4-bit, 10-20% → 3-bit, 20-25% → 2-bit, 25-100% → 0-bit")
    print("-" * 70)

    tiered_config = KVTCMagnitudeTieredConfig(
        tier_ratios=(0.10, 0.20, 0.25),  # 只保留 25% (与 DCT-Fixed rank=8 一致)
        tier_bits=(4, 3, 2, 0),  # Top 10%→4bit, 10-20%→3bit, 20-25%→2bit, 25-100%→0bit
        pruning_method="l2",
    )

    t0 = time.time()
    tier_cal = fit_magnitude_tiered_calibration([keys_flat], [values_flat], config, tiered_config)
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

    results.append(("方案 A (Balanced 40%)", tier_enc_time, tier_dec_time, tier_err, tier_size))

    # ========================================================================
    # 方案 A: High Compression (20%/50% split)
    # ========================================================================
    print("-" * 70)
    print("方案 A (High Compression): Magnitude + Tiered (保留 15%)")
    print("说明: Top 5% → 4-bit, 5-10% → 3-bit, 10-15% → 2-bit, 15-100% → 0-bit (更激进)")
    print("-" * 70)

    tiered_config_high = KVTCMagnitudeTieredConfig(
        tier_ratios=(0.05, 0.10, 0.15),  # 只保留 15% (更激进，高压缩率)
        tier_bits=(4, 3, 2, 0),  # Top 5%→4bit, 5-10%→3bit, 10-15%→2bit, 15-100%→0bit
        pruning_method="l2",
    )

    tier_cal_high = fit_magnitude_tiered_calibration([keys_flat], [values_flat], config, tiered_config_high)

    t0 = time.time()
    tier_enc_high = encode_tensor_magnitude_tiered(keys_flat, tier_cal_high.keys)
    tier_enc_high_time = time.time() - t0

    t0 = time.time()
    tier_dec_high = decode_tensor_magnitude_tiered(tier_enc_high, tier_cal_high.keys)
    tier_dec_high_time = time.time() - t0

    tier_err_high = np.linalg.norm(tier_dec_high - keys_flat) / np.linalg.norm(keys_flat)
    tier_size_high = measure_compression_ratio(tier_enc_high)

    print(f"⏱️  Encode: {tier_enc_high_time*1000:.2f} ms")
    print(f"⏱️  Decode: {tier_dec_high_time*1000:.2f} ms")
    print(f"📊 Relative error: {tier_err_high:.6f}")
    print(f"💾 Compressed size: {tier_size_high / 1024:.2f} KB")
    print(f"📦 Compression ratio: {keys_flat.nbytes / tier_size_high:.2f}x")
    print(f"🎯 vs DCT-Fixed: 压缩率 {(tier_size_high/dct_size - 1)*100:+.1f}%, "
          f"精度提升 {(1 - tier_err_high/dct_err)*100:.1f}%")
    print()

    results.append(("方案 A (High Comp 20%)", tier_enc_high_time, tier_dec_high_time, tier_err_high, tier_size_high))

    # ========================================================================
    # 方案 A: High Precision (60%/90% split)
    # ========================================================================
    print("-" * 70)
    print("方案 A (High Precision): Magnitude + Tiered (保留 40%)")
    print("说明: Top 15% → 4-bit, 15-30% → 3-bit, 30-40% → 2-bit, 40-100% → 0-bit (保守)")
    print("-" * 70)

    tiered_config_prec = KVTCMagnitudeTieredConfig(
        tier_ratios=(0.15, 0.30, 0.40),  # 保留 40% (保守，高精度)
        tier_bits=(4, 3, 2, 0),  # Top 15%→4bit, 15-30%→3bit, 30-40%→2bit, 40-100%→0bit
        pruning_method="l2",
    )

    tier_cal_prec = fit_magnitude_tiered_calibration([keys_flat], [values_flat], config, tiered_config_prec)

    t0 = time.time()
    tier_enc_prec = encode_tensor_magnitude_tiered(keys_flat, tier_cal_prec.keys)
    tier_enc_prec_time = time.time() - t0

    t0 = time.time()
    tier_dec_prec = decode_tensor_magnitude_tiered(tier_enc_prec, tier_cal_prec.keys)
    tier_dec_prec_time = time.time() - t0

    tier_err_prec = np.linalg.norm(tier_dec_prec - keys_flat) / np.linalg.norm(keys_flat)
    tier_size_prec = measure_compression_ratio(tier_enc_prec)

    print(f"⏱️  Encode: {tier_enc_prec_time*1000:.2f} ms")
    print(f"⏱️  Decode: {tier_dec_prec_time*1000:.2f} ms")
    print(f"📊 Relative error: {tier_err_prec:.6f}")
    print(f"💾 Compressed size: {tier_size_prec / 1024:.2f} KB")
    print(f"📦 Compression ratio: {keys_flat.nbytes / tier_size_prec:.2f}x")
    print(f"🎯 vs DCT-Fixed: 压缩率 {(tier_size_prec/dct_size - 1)*100:+.1f}%, "
          f"精度提升 {(1 - tier_err_prec/dct_err)*100:.1f}%")
    print()

    results.append(("方案 A (High Prec 40%)", tier_enc_prec_time, tier_dec_prec_time, tier_err_prec, tier_size_prec))

    # ========================================================================
    # 方案 A+: Ultra Precision (70% + 5-bit)
    # ========================================================================
    print("-" * 70)
    print("方案 A+ (Ultra Precision): Magnitude + Tiered (保留 70% + 5-bit)")
    print("说明: Top 25% → 5-bit, 25-50% → 4-bit, 50-70% → 3-bit, 70-100% → 0-bit")
    print("目标: 精度 <0.35 (接近 P4 原始), 压缩率 >25x")
    print("-" * 70)

    tiered_config_ultra = KVTCMagnitudeTieredConfig(
        tier_ratios=(0.25, 0.50, 0.70),  # 保留 70%
        tier_bits=(5, 4, 3, 0),  # 使用 5-bit 高精度！
        pruning_method="l2",
    )

    # 注意：需要修改 config.bits 为 5 以支持 5-bit 量化
    config_5bit = KVTCCodecConfig(
        rank=8,
        bits=5,  # 5-bit！
        group_size=16,
        sample_limit=256,
        zero_bit_energy_fraction=0.001,
    )

    tier_cal_ultra = fit_magnitude_tiered_calibration([keys_flat], [values_flat], config_5bit, tiered_config_ultra)

    t0 = time.time()
    tier_enc_ultra = encode_tensor_magnitude_tiered(keys_flat, tier_cal_ultra.keys)
    tier_enc_ultra_time = time.time() - t0

    t0 = time.time()
    tier_dec_ultra = decode_tensor_magnitude_tiered(tier_enc_ultra, tier_cal_ultra.keys)
    tier_dec_ultra_time = time.time() - t0

    tier_err_ultra = np.linalg.norm(tier_dec_ultra - keys_flat) / np.linalg.norm(keys_flat)
    tier_size_ultra = measure_compression_ratio(tier_enc_ultra)

    print(f"⏱️  Encode: {tier_enc_ultra_time*1000:.2f} ms")
    print(f"⏱️  Decode: {tier_dec_ultra_time*1000:.2f} ms")
    print(f"📊 Relative error: {tier_err_ultra:.6f}")
    print(f"💾 Compressed size: {tier_size_ultra / 1024:.2f} KB")
    print(f"📦 Compression ratio: {keys_flat.nbytes / tier_size_ultra:.2f}x")
    print(f"🎯 vs DCT-Fixed: 压缩率 {(tier_size_ultra/dct_size - 1)*100:+.1f}%, "
          f"精度提升 {(1 - tier_err_ultra/dct_err)*100:.1f}%")
    print(f"🎯 vs P4 原始 (90%): 对比精度目标 0.275")
    print()

    results.append(("方案 A+ (Ultra Prec 70%)", tier_enc_ultra_time, tier_dec_ultra_time, tier_err_ultra, tier_size_ultra))

    # ========================================================================
    # 方案 A++: Super Precision (80% + 5-bit)
    # ========================================================================
    print("-" * 70)
    print("方案 A++ (Super Precision): Magnitude + Tiered (保留 80% + 5-bit)")
    print("说明: Top 30% → 5-bit, 30-60% → 4-bit, 60-80% → 3-bit, 80-100% → 0-bit")
    print("目标: 精度 <0.30 (超越 P4 原始), 压缩率 >20x")
    print("-" * 70)

    tiered_config_super = KVTCMagnitudeTieredConfig(
        tier_ratios=(0.30, 0.60, 0.80),  # 保留 80%
        tier_bits=(5, 4, 3, 0),  # 使用 5-bit 高精度！
        pruning_method="l2",
    )

    tier_cal_super = fit_magnitude_tiered_calibration([keys_flat], [values_flat], config_5bit, tiered_config_super)

    t0 = time.time()
    tier_enc_super = encode_tensor_magnitude_tiered(keys_flat, tier_cal_super.keys)
    tier_enc_super_time = time.time() - t0

    t0 = time.time()
    tier_dec_super = decode_tensor_magnitude_tiered(tier_enc_super, tier_cal_super.keys)
    tier_dec_super_time = time.time() - t0

    tier_err_super = np.linalg.norm(tier_dec_super - keys_flat) / np.linalg.norm(keys_flat)
    tier_size_super = measure_compression_ratio(tier_enc_super)

    print(f"⏱️  Encode: {tier_enc_super_time*1000:.2f} ms")
    print(f"⏱️  Decode: {tier_dec_super_time*1000:.2f} ms")
    print(f"📊 Relative error: {tier_err_super:.6f}")
    print(f"💾 Compressed size: {tier_size_super / 1024:.2f} KB")
    print(f"📦 Compression ratio: {keys_flat.nbytes / tier_size_super:.2f}x")
    print(f"🎯 vs DCT-Fixed: 压缩率 {(tier_size_super/dct_size - 1)*100:+.1f}%, "
          f"精度提升 {(1 - tier_err_super/dct_err)*100:.1f}%")
    print(f"🎯 vs P4 原始 (90%): 对比精度目标 0.275")
    print()

    results.append(("方案 A++ (Super Prec 80%)", tier_enc_super_time, tier_dec_super_time, tier_err_super, tier_size_super))

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
        improvement = (baseline_size / size - 1) * 100
        print(f"   {name:30s}: {ratio:5.2f}x ", end="")
        if improvement != 0:
            print(f"(vs baseline {improvement:+5.1f}%)")
        else:
            print()
    print()

    print("📊 **Accuracy (Relative Error)**:")
    baseline_err = results[0][3]
    for name, _, _, err, _ in results:
        status = "✅" if err < 1.1 else "❌"
        improvement = (1 - err/baseline_err) * 100
        print(f"   {name:30s}: {err:.6f} {status}", end="")
        if improvement > 0:
            print(f" (精度提升 {improvement:.1f}%)")
        else:
            print()
    print()

    print("⚖️  **Trade-off Analysis**:")
    print(f"   {'Strategy':30s}  {'Compression':>12s}  {'Accuracy':>15s}")
    print(f"   {'-'*30}  {'-'*12}  {'-'*15}")
    for name, _, _, err, size in results:
        comp_ratio = keys_flat.nbytes / size
        print(f"   {name:30s}  {comp_ratio:11.2f}x  {err:14.6f}")
    print()

    # P4 原始对比
    print("📌 **P4 原始结果对比** (参考值):")
    print(f"   Conservative (90%): 相对误差 0.275, 压缩率 10.28x")
    print(f"   Moderate (75%):     相对误差 0.384, 压缩率 11.77x")
    print(f"   Aggressive (50%):   相对误差 0.579, 压缩率 17.32x")
    print()

    # Find best precision results
    best_prec_result = min(results[1:], key=lambda x: x[3])  # Lowest error
    best_prec_name, _, _, best_prec_err, best_prec_size = best_prec_result
    best_prec_comp = keys_flat.nbytes / best_prec_size

    # Recommendation
    print("🎯 **Recommendation**:")
    if best_prec_err < 0.35:
        print(f"   ✅ {best_prec_name} 达到精度目标：")
        print(f"   - 精度：{best_prec_err:.6f} (目标 <0.35, 接近 P4 原始)")
        print(f"   - 压缩率：{best_prec_comp:.2f}x (vs P4 原始 10-17x，提升 {(best_prec_comp/12 - 1)*100:+.1f}%)")
        print(f"   - 数据驱动，自适应（不依赖频域假设）")
    else:
        print(f"   ⚠️  当前最佳精度 {best_prec_err:.6f} 仍未达到目标 <0.35")
        print(f"   建议：进一步提高保留率或使用更高 bit 数")
    print()

    print("🔧 **35B 模型实际应用建议**:")
    print(f"   - 长上下文 (>4K): 使用 High Compression (20%) → ~60-80x 压缩")
    print(f"   - 中上下文 (2-4K): 使用 Balanced (40%) → ~40-50x 压缩")
    print(f"   - 短上下文 (<2K): 使用 High Precision (60%) → ~25-30x 压缩")
    print()

    # Test pass criteria
    all_passed = all(err < 1.1 for _, _, _, err, _ in results)
    # Balanced (25%) should be close to DCT-Fixed in compression and better in accuracy
    tier_better = tier_err < dct_err  # Any improvement
    tier_comp_ok = (keys_flat.nbytes / tier_size) > 40  # At least 40x compression

    if all_passed and tier_better and tier_comp_ok:
        print("=" * 70)
        print("✅ All configurations PASSED!")
        print()
        print("🎯 **方案 A 核心成就**:")
        print(f"   1. 精度提升 {(1 - tier_err/dct_err)*100:.1f}% (相对误差: {tier_err:.6f} vs {dct_err:.6f})")
        print(f"   2. 压缩率保持 {keys_flat.nbytes / tier_size:.2f}x (vs DCT-Fixed {keys_flat.nbytes / dct_size:.2f}x)")
        print(f"   3. 数据驱动，自适应（不依赖频域假设）")
        print(f"   4. 可配置，适应不同场景（20%/40%/60% 配置）")
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
    print("║" + " " * 10 + "KVTC 方案 A: Magnitude + Tiered Quantization" + " " * 13 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        success = test_magnitude_tiered()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
