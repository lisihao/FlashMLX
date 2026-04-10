#!/usr/bin/env python3
"""Test fix with both triple_pq and scored_pq."""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"
CALIBRATION = "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl"

print("Testing FIXED code")
print("="*60)

model, tokenizer = load(MODEL_PATH)

text = "The quick brown fox jumps over the lazy dog. " * 2000
tokens_list = tokenizer.encode(text)[:16384]
tokens = mx.array([tokens_list])
print(f"Context: {len(tokens_list)} tokens\n")

def test(name, cache_kwargs):
    print(f"\n{name}")
    print("-"*60)
    mx.clear_cache()
    mx.reset_peak_memory()

    cache = make_prompt_cache(model, **cache_kwargs)
    logits = model(tokens, cache=cache)
    mx.eval(logits)

    pp_peak = mx.get_peak_memory() / (1024**2)
    pp_active = mx.get_active_memory() / (1024**2)

    for _ in range(5):
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(next_token, cache=cache)
        mx.eval(logits)

    tg_peak = mx.get_peak_memory() / (1024**2)
    tg_active = mx.get_active_memory() / (1024**2)

    print(f"  PP:  Peak {pp_peak:.0f} MB, Active {pp_active:.0f} MB")
    print(f"  TG:  Peak {tg_peak:.0f} MB, Active {tg_active:.0f} MB")

    del cache, logits
    mx.clear_cache()
    return pp_peak, tg_active

pp_std, tg_std = test("1. Standard", {})

pp_tpq, tg_tpq = test("2. triple_pq + Q8 (FIXED)", {
    "kv_cache": "triple_pq",
    "kv_warm_quantizer": "polarquant",
    "kv_warm_bits": 4,
    "kv_flat_quant": "q8_0"
})

try:
    pp_spq, tg_spq = test("3. scored_pq + Q8 (should still work)", {
        "kv_cache": "scored_pq",
        "kv_calibration": CALIBRATION,
        "kv_flat_quant": "q8_0",
    })
except Exception as e:
    print(f"   Error: {e}")
    pp_spq, tg_spq = 0, 0

print(f"\n{'='*60}")
print("COMPARISON (16K tokens)")
print(f"{'='*60}")
print(f"{'Config':<25} {'PP Peak':>10} {'Change':>10} {'Status':>10}")
print("-"*70)
print(f"{'Standard':<25} {pp_std:>10.0f} {'—':>10} {'baseline':>10}")

tpq_change = ((pp_tpq - pp_std) / pp_std * 100)
tpq_status = "✅ FIXED" if abs(tpq_change) < 10 else "❌ STILL BAD"
print(f"{'triple_pq (before +55%)':<25} {pp_tpq:>10.0f} {tpq_change:>+9.1f}% {tpq_status:>10}")

if pp_spq > 0:
    spq_change = ((pp_spq - pp_std) / pp_std * 100)
    spq_status = "✅ GOOD" if abs(spq_change) < 10 else "❌ BROKE"
    print(f"{'scored_pq (before -2%)':<25} {pp_spq:>10.0f} {spq_change:>+9.1f}% {spq_status:>10}")

print(f"\nOriginal bug (triple_pq): +55.3% PP memory")
print(f"Expected after fix: ~0% (same as standard)")
