#!/usr/bin/env python3
"""Test fix at 32K tokens - where KV cache dominates memory."""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

print("Loading model...")
model, tokenizer = load(MODEL_PATH)

# Generate 32K tokens
print("\nGenerating 32K token context...")
text = "The quick brown fox jumps over the lazy dog. " * 4000
tokens_list = tokenizer.encode(text)[:32768]
tokens = mx.array([tokens_list])
print(f"Context length: {len(tokens_list)} tokens\n")

def test_32k(name, cache_kwargs):
    """Test at 32K tokens."""
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
    print(f"After prefill (32K tokens):")
    print(f"  Peak:   {pp_peak:.1f} MB")
    print(f"  Active: {pp_active:.1f} MB")

    # Generate 5 tokens
    for i in range(5):
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(next_token, cache=cache)
        mx.eval(logits)

    tg_peak = mx.get_peak_memory() / (1024**2)
    tg_active = mx.get_active_memory() / (1024**2)
    print(f"After 5 TG tokens:")
    print(f"  Peak:   {tg_peak:.1f} MB")
    print(f"  Active: {tg_active:.1f} MB\n")

    del cache, logits
    mx.clear_cache()

    return pp_peak, pp_active, tg_peak, tg_active

# Test Standard
pp_peak_std, pp_active_std, tg_peak_std, tg_active_std = test_32k(
    "Standard (no compression)", {})

# Test PolarQuant + Q8 flat
pp_peak_pq, pp_active_pq, tg_peak_pq, tg_active_pq = test_32k(
    "PolarQuant 4-bit + Q8 flat", {
        "kv_cache": "triple_pq",
        "kv_warm_quantizer": "polarquant",
        "kv_warm_bits": 4,
        "kv_flat_quant": "q8_0"
    })

# Summary
print("="*60)
print("SUMMARY (32K tokens)")
print("="*60)
print(f"\n{'Config':<25} {'PP Active':>12} {'TG Active':>12}")
print("-" * 60)
print(f"{'Standard':<25} {pp_active_std:>12.1f} {tg_active_std:>12.1f}")
print(f"{'PolarQuant + Q8':<25} {pp_active_pq:>12.1f} {tg_active_pq:>12.1f}")

pp_savings = pp_active_std - pp_active_pq
tg_savings = tg_active_std - tg_active_pq
pp_pct = (pp_savings / pp_active_std) * 100 if pp_active_std > 0 else 0
tg_pct = (tg_savings / tg_active_std) * 100 if tg_active_std > 0 else 0

print(f"\n{'Savings':<25} {pp_savings:>12.1f} ({pp_pct:>+6.1f}%) {tg_savings:>12.1f} ({tg_pct:>+6.1f}%)")

print(f"\nTheoretical KV cache at 32K (bf16):")
print(f"  32K × 8 heads × 128 dim × 36 layers × 2 (K+V) × 2 bytes = ~9.4 GB")
print(f"  Q8 compression: ~4.7 GB")
print(f"  Expected savings: ~4.7 GB")

if tg_pct > 20:
    print(f"\n✅ Compression working! TG active memory reduced by {tg_pct:.1f}%")
elif abs(tg_pct) < 5:
    print(f"\n⚠️  Low compression: Only {tg_pct:.1f}% savings")
else:
    print(f"\n❌ Bug: Compressed uses more memory!")

print(f"\nFlashMLX v1.x (scored_pq) at 32K:")
print(f"  PP: 526 MB, TG: 529 MB (-89% vs standard 4840 MB)")
print(f"\nNote: scored_pq uses AM eviction (keeps only hot tokens),")
print(f"      triple_pq keeps all tokens (just quantizes them).")
