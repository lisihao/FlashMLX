#!/usr/bin/env python3
"""Test ORIGINAL code with both triple_pq and scored_pq."""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"
CALIBRATION = "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl"

print("Testing ORIGINAL code (before fix)")
print("="*60)

model, tokenizer = load(MODEL_PATH)

# Test at 16K (manageable size)
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

    # Generate 5 tokens
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

# Test 1: Standard
pp_std, tg_std = test("1. Standard", {})

# Test 2: triple_pq (has bug?)
pp_tpq, tg_tpq = test("2. triple_pq + Q8", {
    "kv_cache": "triple_pq",
    "kv_warm_quantizer": "polarquant",
    "kv_warm_bits": 4,
    "kv_flat_quant": "q8_0"
})

# Test 3: scored_pq (should work)
try:
    pp_spq, tg_spq = test("3. scored_pq + Q8", {
        "kv_cache": "scored_pq",
        "kv_calibration": CALIBRATION,
        "kv_flat_quant": "q8_0",
    })
except Exception as e:
    print(f"   Error: {e}")
    pp_spq, tg_spq = 0, 0

# Summary
print(f"\n{'='*60}")
print("SUMMARY (16K tokens, ORIGINAL code)")
print(f"{'='*60}")
print(f"{'Config':<20} {'PP Peak':>10} {'TG Active':>10} {'vs Std':>10}")
print("-"*60)
print(f"{'Standard':<20} {pp_std:>10.0f} {tg_std:>10.0f} {'—':>10}")

if pp_tpq > 0:
    tpq_diff = ((pp_tpq - pp_std) / pp_std * 100)
    print(f"{'triple_pq + Q8':<20} {pp_tpq:>10.0f} {tg_tpq:>10.0f} {tpq_diff:>+9.1f}%")

if pp_spq > 0:
    spq_diff = ((pp_spq - pp_std) / pp_std * 100)
    print(f"{'scored_pq + Q8':<20} {pp_spq:>10.0f} {tg_spq:>10.0f} {spq_diff:>+9.1f}%")

print(f"\nExpected:")
print(f"  triple_pq: Should have bug (+20-70% memory)")
print(f"  scored_pq: Should work well (-70-90% memory)")
