#!/usr/bin/env python3
"""
CORRECT decode profiling with proper KV cache usage - 30B A3B Model
"""

import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as mlx_cache

MODEL_PATH = Path.home() / "models" / "qwen3.5-35b-mlx"

def profile_decode_correctly():
    """
    Measure decode with CORRECT KV cache usage (like stream_generate does)
    """
    print("=" * 80)
    print("CORRECT Decode Profiling (with KV Cache) - 30B A3B Model")
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

    # Exclude token 1 (cold start)
    avg_decode_stable = sum(decode_times[1:]) / len(decode_times[1:]) * 1000
    decode_tps_stable = 1000 / avg_decode_stable

    print(f"Stable state (tokens 2-100): {avg_decode_stable:.3f} ms/token ({decode_tps_stable:.1f} tok/s)")
    print()

    return {
        "prefill_ms": avg_prefill,
        "prefill_tps": prefill_tps,
        "decode_ms": avg_decode,
        "decode_tps": decode_tps,
        "decode_stable_ms": avg_decode_stable,
        "decode_stable_tps": decode_tps_stable,
    }

if __name__ == "__main__":
    results = profile_decode_correctly()
