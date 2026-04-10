#!/usr/bin/env python3
"""
尝试复现FlashMLX v1.x benchmark (32K, scored_pq)

Expected results:
  Standard: PP 5,079 MB, TG 4,572 MB
  FlashMLX: PP 774 MB (-84.8%), TG 147 MB (-96.8%)
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"
CALIBRATION = "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl"

print("="*70)
print("复现FlashMLX v1.x Benchmark")
print("原始代码（无任何修改）")
print("="*70)

model, tokenizer = load(MODEL_PATH)

# 32K context
text = "The quick brown fox jumps over the lazy dog. " * 4000
tokens_list = tokenizer.encode(text)[:32768]
tokens = mx.array([tokens_list])
print(f"\nContext: {len(tokens_list)} tokens")

def test_config(name, cache_kwargs):
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")

    mx.clear_cache()
    mx.reset_peak_memory()

    try:
        cache = make_prompt_cache(model, **cache_kwargs)

        # Prefill
        print("Running prefill...")
        logits = model(tokens, cache=cache)
        mx.eval(logits)

        pp_peak = mx.get_peak_memory() / (1024**2)
        pp_active = mx.get_active_memory() / (1024**2)
        print(f"✅ PP: Peak {pp_peak:.0f} MB, Active {pp_active:.0f} MB")

        # Generate tokens
        print("Generating tokens...")
        for i in range(10):
            next_token = mx.argmax(logits[:, -1:, :], axis=-1)
            logits = model(next_token, cache=cache)
            mx.eval(logits)

        tg_peak = mx.get_peak_memory() / (1024**2)
        tg_active = mx.get_active_memory() / (1024**2)
        print(f"✅ TG: Peak {tg_peak:.0f} MB, Active {tg_active:.0f} MB")

        del cache, logits
        mx.clear_cache()

        return pp_peak, tg_active, True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return 0, 0, False

# Test 1: Standard
pp_std, tg_std, ok_std = test_config("Standard (baseline)", {})

# Test 2: scored_pq (FlashMLX v1.x config)
pp_spq, tg_spq, ok_spq = test_config("scored_pq + Q8 (FlashMLX v1.x)", {
    "kv_cache": "scored_pq",
    "kv_calibration": CALIBRATION,
    "kv_flat_quant": "q8_0",
    "kv_compression_ratio": 0.0,  # Auto-adaptive
})

# Summary
print(f"\n{'='*70}")
print("对比结果")
print(f"{'='*70}")
print(f"\n{'Config':<30} {'PP Peak':>12} {'TG Active':>12}")
print("-"*70)

if ok_std:
    print(f"{'Standard':<30} {pp_std:>12.0f} {tg_std:>12.0f}")

if ok_spq:
    print(f"{'scored_pq':<30} {pp_spq:>12.0f} {tg_spq:>12.0f}")

    pp_diff = ((pp_std - pp_spq) / pp_std * 100) if ok_std else 0
    tg_diff = ((tg_std - tg_spq) / tg_std * 100) if ok_std else 0

    print(f"\n{'Compression':<30} {pp_diff:>11.1f}% {tg_diff:>11.1f}%")

print(f"\n{'='*70}")
print("FlashMLX v1.x Benchmark (期望结果)")
print(f"{'='*70}")
print(f"Standard:  PP 5,079 MB, TG 4,572 MB")
print(f"FlashMLX:  PP 774 MB (-84.8%), TG 147 MB (-96.8%)")

print(f"\n{'='*70}")
print("结论")
print(f"{'='*70}")

if ok_spq:
    if pp_diff > 70 and tg_diff > 70:
        print("✅ 原始代码达到benchmark性能")
        print("   → 说明我的修改破坏了性能")
    else:
        print(f"❌ 原始代码未达到benchmark性能")
        print(f"   实际: PP {pp_diff:.1f}%, TG {tg_diff:.1f}%")
        print(f"   期望: PP -84.8%, TG -96.8%")
        print(f"   → 说明benchmark用的是不同配置/模型")
else:
    print("❌ scored_pq运行失败（OOM或其他错误）")
    print("   → 原始代码本身有问题")
