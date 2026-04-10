#!/usr/bin/env python3
"""Test with FlashMLX v1.x configuration to reproduce benchmark results."""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

print("Loading model...")
model, tokenizer = load(MODEL_PATH)

# Generate 32K tokens (matching benchmark)
print("\nGenerating 32K token context...")
text = "The quick brown fox jumps over the lazy dog. " * 4000
tokens_list = tokenizer.encode(text)[:32768]
tokens = mx.array([tokens_list])
print(f"Context length: {len(tokens_list)} tokens\n")

def test_config(name, cache_kwargs):
    """Test memory usage for a configuration."""
    print(f"{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    mx.clear_cache()
    mx.reset_peak_memory()

    # Create cache
    cache = make_prompt_cache(model, **cache_kwargs)

    # Prefill
    logits = model(tokens, cache=cache)
    mx.eval(logits)

    pp_mem = mx.get_peak_memory() / (1024**2)
    print(f"  PP Peak Memory: {pp_mem:.1f} MB")

    # Generate 1 token to trigger flat mode
    next_token = mx.argmax(logits[:, -1, :], keepdims=True)
    logits = model(next_token, cache=cache)
    mx.eval(logits)

    tg_mem = mx.get_peak_memory() / (1024**2)
    print(f"  After 1 TG token: {tg_mem:.1f} MB\n")

    del cache, logits
    mx.clear_cache()

    return pp_mem, tg_mem

# Test 1: Standard
print("\n" + "="*60)
print("TEST 1: Standard (no compression)")
print("="*60)
pp_std, tg_std = test_config("Standard", {})

# Test 2: scored_pq (what FlashMLX v1.x might have used)
print("\n" + "="*60)
print("TEST 2: scored_pq + Q8 flat")
print("="*60)
# First need calibration
print("Note: scored_pq requires calibration. Testing triple_pq instead...")

# Test 3: triple_pq + Q8 flat
pp_pq, tg_pq = test_config("Triple PQ + Q8 flat", {
    "kv_cache": "triple_pq",
    "kv_warm_quantizer": "polarquant",
    "kv_warm_bits": 4,
    "kv_flat_quant": "q8_0"
})

# Summary
print("\n" + "="*60)
print("SUMMARY (32K tokens)")
print("="*60)
print(f"\n{'Config':<25} {'PP Memory':>12} {'TG Memory':>12}")
print("-" * 60)
print(f"{'Standard':<25} {pp_std:>12.1f} {tg_std:>12.1f}")
print(f"{'Triple PQ + Q8':<25} {pp_pq:>12.1f} {tg_pq:>12.1f}")

pp_savings = pp_std - pp_pq
tg_savings = tg_std - tg_pq
pp_pct = (pp_savings / pp_std) * 100
tg_pct = (tg_savings / tg_std) * 100

print(f"\n{'Savings':<25} {pp_savings:>12.1f} ({pp_pct:>+6.1f}%) {tg_savings:>12.1f} ({tg_pct:>+6.1f}%)")

print(f"\nFlashMLX v1.x benchmark (M4 Pro 48GB):")
print(f"  PP: 5,079 MB → 774 MB (-84.8%)")
print(f"  TG: 4,572 MB → 147 MB (-96.8%)")

if pp_pct < -20:
    print(f"\n❌ BUG: Compressed uses MORE memory during PP ({pp_pct:+.1f}%)")
elif pp_pct < 50:
    print(f"\n⚠️  LOW COMPRESSION: Only {pp_pct:.1f}% savings (expected ~85%)")
else:
    print(f"\n✅ GOOD COMPRESSION: {pp_pct:.1f}% PP savings")
