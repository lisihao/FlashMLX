#!/usr/bin/env python3
"""
CORRECT decode profiling with proper KV cache usage
"""

import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as mlx_cache

MODEL_PATH = Path.home() / "models" / "/Volumes/toshiba/models/qwen3.5-2b-opus-distilled"

def profile_decode_correctly():
    """
    Measure decode with CORRECT KV cache usage (like stream_generate does)
    """
    print("=" * 80)
    print("CORRECT Decode Profiling (with KV Cache)")
    print("=" * 80)
    print()

    print("Loading model...")
    model, tokenizer = load(str(MODEL_PATH))

    # Test prompt
    prompt = "The quick brown fox jumps over the lazy dog. " * 100
    tokens = mx.array(tokenizer.encode(prompt))
    print(f"Prompt tokens: {len(tokens)}")
    print()

    # Step 1: Prefill phase (compute KV cache)
    print("=" * 80)
    print("STEP 1: PREFILL (compute KV cache)")
    print("=" * 80)

    # Create cache using the CORRECT API
    kv_cache = mlx_cache.make_prompt_cache(model, max_kv_size=None)

    # Warmup
    for _ in range(2):
        _ = model(tokens[None, :], cache=kv_cache)
    mx.eval(model.parameters())

    # Measure prefill
    prefill_times = []
    for i in range(3):
        # Reset cache for each run
        kv_cache = mlx_cache.make_prompt_cache(model, max_kv_size=None)

        start = time.perf_counter()
        logits = model(tokens[None, :], cache=kv_cache)
        mx.eval(logits)
        duration = time.perf_counter() - start

        prefill_times.append(duration)
        print(f"  Run {i+1}: {duration*1000:.2f} ms")

    avg_prefill = sum(prefill_times) / len(prefill_times) * 1000
    prefill_tps = len(tokens) / (avg_prefill / 1000)
    print(f"\nAverage prefill: {avg_prefill:.2f} ms")
    print(f"Prefill throughput: {prefill_tps:.1f} tok/s")
    print()

    # Step 2: Decode phase (single token with cache)
    print("=" * 80)
    print("STEP 2: DECODE (single token with KV cache)")
    print("=" * 80)

    # Prepare cache from prefill
    kv_cache = mlx_cache.make_prompt_cache(model, max_kv_size=None)
    _ = model(tokens[None, :], cache=kv_cache)
    mx.eval(kv_cache)

    # Get last token as input for decode
    last_token = mx.argmax(model(tokens[None, :], cache=kv_cache)[:, -1, :], axis=-1)
    mx.eval(last_token)

    # Warmup decode
    for _ in range(5):
        _ = model(last_token[None, :], cache=kv_cache)
    mx.eval(kv_cache)

    # Measure decode
    decode_times = []
    for i in range(100):
        # Use last generated token as input
        start = time.perf_counter()
        logits = model(last_token[None, :], cache=kv_cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)
        duration = time.perf_counter() - start

        decode_times.append(duration)
        last_token = next_token

        if i < 10:
            print(f"  Token {i+1}: {duration*1000:.3f} ms")

    print()
    avg_decode = sum(decode_times) / len(decode_times) * 1000
    decode_tps = 1000 / avg_decode

    print(f"Average decode: {avg_decode:.3f} ms/token")
    print(f"Decode throughput: {decode_tps:.1f} tok/s")
    print()

    # Step 3: Compare with baseline
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()

    baseline_tg = 60.1  # From stream_generate
    print(f"Baseline (stream_generate):     {baseline_tg:.1f} tok/s")
    print(f"Correct decode (with cache):    {decode_tps:.1f} tok/s")
    print(f"Difference: {decode_tps - baseline_tg:+.1f} tok/s ({(decode_tps/baseline_tg - 1)*100:+.1f}%)")
    print()

    # Analysis
    if abs(decode_tps - baseline_tg) < 3:
        print("✅ PERFECT MATCH - measurement is accurate")
        print("   → This is the real GPU execution time per token")
    elif decode_tps > baseline_tg:
        print("⚠️ Decode is faster than baseline")
        print(f"   → Missing overhead: {(baseline_tg/decode_tps - 1)*-100:.1f}%")
    else:
        print("⚠️ Decode is slower than baseline")
        print(f"   → Extra overhead: {(decode_tps/baseline_tg - 1)*-100:.1f}%")

    print()
    print("=" * 80)
    print("TIME BREAKDOWN (per token)")
    print("=" * 80)
    print()

    total_time = 1000 / baseline_tg  # ms
    gpu_time = avg_decode  # ms
    overhead = total_time - gpu_time

    print(f"Total (measured by stream_generate): {total_time:.3f} ms (100.0%)")
    print(f"GPU execution (model forward):        {gpu_time:.3f} ms ({gpu_time/total_time*100:.1f}%)")
    print(f"Overhead (tokenization + dispatch):   {overhead:.3f} ms ({overhead/total_time*100:.1f}%)")

    return {
        "prefill_ms": avg_prefill,
        "prefill_tps": prefill_tps,
        "decode_ms": avg_decode,
        "decode_tps": decode_tps,
        "baseline_tps": baseline_tg,
    }

if __name__ == "__main__":
    results = profile_decode_correctly()
