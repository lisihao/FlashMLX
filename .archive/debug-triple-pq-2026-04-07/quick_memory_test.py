#!/usr/bin/env python3
"""Quick memory test: Standard vs PolarQuant 4-bit (prefill only)."""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

print("Loading model...")
model, tokenizer = load(MODEL_PATH)

# Generate 16K tokens (manageable size)
print("\nGenerating 16K token context...")
text = "The quick brown fox jumps over the lazy dog. " * 2000
tokens_list = tokenizer.encode(text)[:16384]
tokens = mx.array([tokens_list])
print(f"Context length: {len(tokens_list)} tokens\n")

# Test 1: Standard
print("="*60)
print("Test 1: Standard (no compression)")
print("="*60)
mx.clear_cache()
mx.reset_peak_memory()

cache_std = make_prompt_cache(model)
_ = model(tokens, cache=cache_std)
mx.eval(_)

mem_std = mx.get_peak_memory() / (1024**2)
print(f"Peak Memory: {mem_std:.1f} MB\n")

del cache_std, _
mx.clear_cache()

# Test 2: PolarQuant 4-bit
print("="*60)
print("Test 2: PolarQuant 4-bit")
print("="*60)
mx.reset_peak_memory()

cache_pq = make_prompt_cache(
    model,
    kv_cache="triple_pq",
    kv_warm_quantizer="polarquant",
    kv_warm_bits=4
)
_ = model(tokens, cache=cache_pq)
mx.eval(_)

mem_pq = mx.get_peak_memory() / (1024**2)
print(f"Peak Memory: {mem_pq:.1f} MB\n")

del cache_pq, _

# Summary
print("="*60)
print("SUMMARY")
print("="*60)
print(f"Standard:        {mem_std:.1f} MB")
print(f"PolarQuant 4bit: {mem_pq:.1f} MB")
savings = mem_std - mem_pq
pct = (savings / mem_std) * 100
print(f"Savings:         {savings:+.1f} MB ({pct:+.1f}%)")

if savings > 1000:  # > 1GB savings
    print("\n✅ Fix successful! Memory reduced as expected.")
elif abs(savings) < 100:  # < 100MB difference
    print("\n⚠️  Fix ineffective: Memory unchanged.")
else:  # Negative savings
    print("\n❌ Fix failed: Memory increased!")
