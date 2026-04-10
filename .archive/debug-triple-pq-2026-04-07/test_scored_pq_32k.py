#!/usr/bin/env python3
"""Test with scored_pq configuration (original FlashMLX v1.x)."""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"
CALIBRATION = "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl"

print("Loading model...")
model, tokenizer = load(MODEL_PATH)

# Generate 32K tokens
print("\nGenerating 32K token context...")
text = "The quick brown fox jumps over the lazy dog. " * 4000
tokens_list = tokenizer.encode(text)[:32768]
tokens = mx.array([tokens_list])
print(f"Context length: {len(tokens_list)} tokens\n")

def test_config(name, cache_kwargs):
    """Test a configuration."""
    print("="*60)
    print(name)
    print("="*60)

    mx.clear_cache()
    mx.reset_peak_memory()

    # Create cache
    cache = make_prompt_cache(model, **cache_kwargs)

    # Prefill
    logits = model(tokens, cache=cache)
    mx.eval(logits)

    pp_peak = mx.get_peak_memory() / (1024**2)
    pp_active = mx.get_active_memory() / (1024**2)
    print(f"After prefill:")
    print(f"  Peak:   {pp_peak:.1f} MB")
    print(f"  Active: {pp_active:.1f} MB")

    # Generate 5 tokens
    for i in range(5):
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(next_token, cache=cache)
        mx.eval(logits)

    tg_peak = mx.get_peak_memory() / (1024**2)
    tg_active = mx.get_active_memory() / (1024**2)
    print(f"After TG:")
    print(f"  Peak:   {tg_peak:.1f} MB")
    print(f"  Active: {tg_active:.1f} MB\n")

    del cache, logits
    mx.clear_cache()

    return pp_peak, pp_active, tg_peak, tg_active

# Test 1: Standard
print("\n" + "="*60)
print("TEST 1: Standard (no compression)")
print("="*60)
pp_peak_std, pp_active_std, tg_peak_std, tg_active_std = test_config(
    "Standard", {})

# Test 2: scored_pq (original FlashMLX v1.x config)
print("\n" + "="*60)
print("TEST 2: scored_pq + Q8 flat (FlashMLX v1.x)")
print("="*60)
try:
    pp_peak_spq, pp_active_spq, tg_peak_spq, tg_active_spq = test_config(
        "scored_pq + Q8", {
            "kv_cache": "scored_pq",
            "kv_calibration": CALIBRATION,
            "kv_flat_quant": "q8_0",
            "kv_compression_ratio": 0.0,  # Auto-adaptive
        })
except Exception as e:
    print(f"Error: {e}")
    print("Calibration file may not exist. Trying without...")
    pp_peak_spq, pp_active_spq, tg_peak_spq, tg_active_spq = test_config(
        "scored_pq (no calibration)", {
            "kv_cache": "scored_pq",
            "kv_flat_quant": "q8_0",
        })

# Summary
print("\n" + "="*60)
print("SUMMARY (32K tokens)")
print("="*60)
print(f"\n{'Config':<25} {'PP Peak':>12} {'PP Active':>12} {'TG Active':>12}")
print("-" * 70)
print(f"{'Standard':<25} {pp_peak_std:>12.1f} {pp_active_std:>12.1f} {tg_active_std:>12.1f}")
print(f"{'scored_pq + Q8':<25} {pp_peak_spq:>12.1f} {pp_active_spq:>12.1f} {tg_active_spq:>12.1f}")

pp_savings = ((pp_peak_std - pp_peak_spq) / pp_peak_std * 100)
tg_savings = ((tg_active_std - tg_active_spq) / tg_active_std * 100)

print(f"\n{'Savings':<25} {pp_savings:>12.1f}% {'-':>12} {tg_savings:>12.1f}%")

print(f"\nFlashMLX v1.x benchmark (M4 Pro 48GB):")
print(f"  Standard: PP 5,079 MB, TG 4,572 MB")
print(f"  FlashMLX: PP 774 MB (-84.8%), TG 147 MB (-96.8%)")
print(f"\nExpected vs Actual:")
print(f"  PP: Expected -84.8%, Got {pp_savings:.1f}%")
print(f"  TG: Expected -96.8%, Got {tg_savings:.1f}%")

if pp_savings > 70 and tg_savings > 70:
    print(f"\n✅ scored_pq working as expected!")
elif pp_savings > 10:
    print(f"\n⚠️  Some compression, but less than expected")
else:
    print(f"\n❌ My fix broke scored_pq!")
