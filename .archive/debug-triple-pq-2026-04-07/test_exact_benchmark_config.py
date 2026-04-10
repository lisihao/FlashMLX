#!/usr/bin/env python3
"""
用model card的精确配置测试
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

# Model card的精确配置
CARD_CONFIG = {
    "strategy": "scored_pq",
    "flat_quant": "q8_0",
    "compression_ratio": 0.0,  # Auto-adaptive
    "calibration_file": "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl",
    "recent_size": 512,
    "warm_size": 2048,
    "scored_max_cache": 2048,
}

print("用Model Card精确配置测试")
print("="*70)

model, tokenizer = load(MODEL_PATH)

# 32K context
text = "The quick brown fox jumps over the lazy dog. " * 4000
tokens_list = tokenizer.encode(text)[:32768]
tokens = mx.array([tokens_list])
print(f"Context: {len(tokens_list)} tokens\n")

def test(name, cache_kwargs):
    print(f"{name}")
    print("-"*70)

    mx.clear_cache()
    mx.reset_peak_memory()

    try:
        cache = make_prompt_cache(model, **cache_kwargs)

        # Prefill
        logits = model(tokens, cache=cache)
        mx.eval(logits)

        pp_peak = mx.get_peak_memory() / (1024**2)
        pp_active = mx.get_active_memory() / (1024**2)

        # TG
        for _ in range(10):
            next_token = mx.argmax(logits[:, -1:, :], axis=-1)
            logits = model(next_token, cache=cache)
            mx.eval(logits)

        tg_active = mx.get_active_memory() / (1024**2)

        print(f"✅ PP Peak: {pp_peak:.0f} MB, TG Active: {tg_active:.0f} MB\n")

        del cache, logits
        mx.clear_cache()

        return pp_peak, tg_active, True

    except Exception as e:
        print(f"❌ ERROR: {e}\n")
        return 0, 0, False

# Test 1: Standard
pp_std, tg_std, ok1 = test("Standard", {})

# Test 2: Model card配置
pp_mc, tg_mc, ok2 = test("Model Card Config", {
    "kv_cache": CARD_CONFIG["strategy"],
    "kv_calibration": CARD_CONFIG["calibration_file"],
    "kv_flat_quant": CARD_CONFIG["flat_quant"],
    "kv_compression_ratio": CARD_CONFIG["compression_ratio"],
})

print("="*70)
print("对比")
print("="*70)
print(f"Standard:   PP {pp_std:.0f} MB, TG {tg_std:.0f} MB")

if ok2:
    print(f"scored_pq:  PP {pp_mc:.0f} MB, TG {tg_mc:.0f} MB")

    pp_pct = ((pp_std - pp_mc) / pp_std * 100) if pp_std > 0 else 0
    tg_pct = ((tg_std - tg_mc) / tg_std * 100) if tg_std > 0 else 0

    print(f"Savings:    PP {pp_pct:+.1f}%, TG {tg_pct:+.1f}%")

    print(f"\nModel Card期望: PP -89%, TG -89%")

    if pp_pct > 60 and tg_pct > 60:
        print("✅ 接近benchmark性能")
    else:
        print(f"⚠️  性能未达标")
else:
    print("scored_pq: 运行失败")
