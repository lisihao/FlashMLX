#!/usr/bin/env python3
"""Test memory with prefill + generation (trigger flat mode transition)."""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

print("Loading model...")
model, tokenizer = load(MODEL_PATH)

# Use 16K tokens
print("\nGenerating 16K token context...")
text = "The quick brown fox jumps over the lazy dog. " * 2000
tokens_list = tokenizer.encode(text)[:16384]
tokens = mx.array([tokens_list])
print(f"Context length: {len(tokens_list)} tokens\n")

def test_with_gen(name, cache_kwargs):
    """Test with prefill + 5 generation tokens."""
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

    mem_after_prefill = mx.get_peak_memory() / (1024**2)
    print(f"After prefill: {mem_after_prefill:.1f} MB")

    # Generate 500 tokens to see memory growth
    for i in range(500):
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(next_token, cache=cache)
        mx.eval(logits)

        if i == 4:  # After flat mode transition
            mem_after_flat = mx.get_peak_memory() / (1024**2)
            print(f"After 5 tokens (flat mode): {mem_after_flat:.1f} MB")

    mem_final = mx.get_peak_memory() / (1024**2)
    print(f"After 500 tokens: {mem_final:.1f} MB")
    print(f"Peak memory: {mem_final:.1f} MB\n")

    del cache, logits
    mx.clear_cache()

    return mem_final

# Test Standard
mem_std = test_with_gen("Standard (no compression)", {})

# Test PolarQuant 4-bit + Q8 flat buffer
mem_pq = test_with_gen("PolarQuant 4-bit + Q8 flat", {
    "kv_cache": "triple_pq",
    "kv_warm_quantizer": "polarquant",
    "kv_warm_bits": 4,
    "kv_flat_quant": "q8_0"  # Enable flat buffer quantization
})

# Summary
print("="*60)
print("SUMMARY (16K prefill + 500 generation tokens)")
print("="*60)
print(f"Standard:        {mem_std:.1f} MB")
print(f"PolarQuant 4bit + Q8: {mem_pq:.1f} MB")
savings = mem_std - mem_pq
pct = (savings / mem_std) * 100
print(f"Savings:         {savings:+.1f} MB ({pct:+.1f}%)")

if savings > 200:  # > 200MB savings
    print("\n✅ Compression working! Memory reduced during generation.")
elif abs(savings) < 100:
    print("\n⚠️  No compression: Memory identical.")
else:
    print("\n❌ Bug still present: Compressed uses MORE memory!")
