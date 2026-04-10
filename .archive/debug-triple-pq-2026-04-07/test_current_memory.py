#!/usr/bin/env python3
"""Test current vs peak memory to see if compression is working."""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

print("Loading model...")
model, tokenizer = load(MODEL_PATH)

# Generate 16K tokens
print("\nGenerating 16K token context...")
text = "The quick brown fox jumps over the lazy dog. " * 2000
tokens_list = tokenizer.encode(text)[:16384]
tokens = mx.array([tokens_list])
print(f"Context length: {len(tokens_list)} tokens\n")

def test_with_current_mem(name, cache_kwargs):
    """Test with current memory tracking."""
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

    peak_after_pp = mx.get_peak_memory() / (1024**2)
    active_after_pp = mx.get_active_memory() / (1024**2)
    print(f"After prefill:")
    print(f"  Peak:   {peak_after_pp:.1f} MB")
    print(f"  Active: {active_after_pp:.1f} MB")

    # Generate 5 tokens (triggers flat mode transition)
    for i in range(5):
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(next_token, cache=cache)
        mx.eval(logits)

    peak_after_tg = mx.get_peak_memory() / (1024**2)
    active_after_tg = mx.get_active_memory() / (1024**2)
    print(f"After 5 TG tokens:")
    print(f"  Peak:   {peak_after_tg:.1f} MB")
    print(f"  Active: {active_after_tg:.1f} MB\n")

    del cache, logits
    mx.clear_cache()

    return peak_after_tg, active_after_tg

# Test Standard
print("\nTest 1: Standard")
peak_std, active_std = test_with_current_mem("Standard (no compression)", {})

# Test PolarQuant + Q8 flat
peak_pq, active_pq = test_with_current_mem("PolarQuant 4-bit + Q8 flat", {
    "kv_cache": "triple_pq",
    "kv_warm_quantizer": "polarquant",
    "kv_warm_bits": 4,
    "kv_flat_quant": "q8_0"
})

# Summary
print("="*60)
print("SUMMARY")
print("="*60)
print(f"\n{'Config':<25} {'Peak (MB)':>12} {'Active (MB)':>12}")
print("-" * 60)
print(f"{'Standard':<25} {peak_std:>12.1f} {active_std:>12.1f}")
print(f"{'PolarQuant + Q8':<25} {peak_pq:>12.1f} {active_pq:>12.1f}")

peak_savings = peak_std - peak_pq
active_savings = active_std - active_pq
peak_pct = (peak_savings / peak_std) * 100 if peak_std > 0 else 0
active_pct = (active_savings / active_std) * 100 if active_std > 0 else 0

print(f"\n{'Savings':<25} {peak_savings:>12.1f} ({peak_pct:>+6.1f}%) {active_savings:>12.1f} ({active_pct:>+6.1f}%)")

if active_pct > 30:
    print(f"\n✅ Compression working! Active memory reduced by {active_pct:.1f}%")
elif abs(active_pct) < 5:
    print(f"\n⚠️  No compression: Active memory unchanged")
else:
    print(f"\n❌ Bug: Compressed uses MORE active memory!")
