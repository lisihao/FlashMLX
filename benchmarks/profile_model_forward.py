#!/usr/bin/env python3
"""
Profile model.forward() calls directly with mx.eval() to measure GPU time
"""

import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load

# Configuration
MODEL_PATH = Path.home() / "models" / "/Volumes/toshiba/models/qwen3.5-2b-opus-distilled"

def profile_forward_passes():
    """Profile individual forward passes with GPU sync"""
    print("=" * 80)
    print("Direct Model Forward Profiling (GPU Synced)")
    print("=" * 80)
    print()

    print("Loading model...")
    model, tokenizer = load(str(MODEL_PATH))

    # Create test inputs
    prompt = "The quick brown fox jumps over the lazy dog. " * 100
    tokens = mx.array(tokenizer.encode(prompt))

    print(f"Prompt tokens: {len(tokens)}")
    print()

    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = model(tokens[None, :])  # Add batch dimension
    mx.eval(model.parameters())  # Ensure everything is loaded
    print("Warmup done")
    print()

    # Profile prefill (full sequence)
    print("=" * 80)
    print("PREFILL (full sequence)")
    print("=" * 80)

    prefill_times = []
    for i in range(5):
        start = time.perf_counter()
        output = model(tokens[None, :])
        mx.eval(output)  # Force GPU sync
        duration = time.perf_counter() - start
        prefill_times.append(duration)
        print(f"Run {i+1}: {duration*1000:.2f} ms")

    avg_prefill = sum(prefill_times) / len(prefill_times) * 1000
    print(f"\nAverage prefill: {avg_prefill:.2f} ms")
    print(f"Throughput: {len(tokens) / (avg_prefill/1000):.1f} tok/s")
    print()

    # Profile decode (single token)
    print("=" * 80)
    print("DECODE (single token)")
    print("=" * 80)

    # Get KV cache from a forward pass
    cache = []
    _ = model(tokens[None, :], cache=cache)
    mx.eval(cache)

    # Now do single-token forward passes
    decode_times = []
    next_token = tokens[-1:]  # Last token as next input

    for i in range(50):
        start = time.perf_counter()
        output = model(next_token[None, :], cache=cache)
        mx.eval(output)  # Force GPU sync
        duration = time.perf_counter() - start
        decode_times.append(duration)

        if i < 10:
            print(f"Token {i+1}: {duration*1000:.2f} ms")

    print()
    avg_decode = sum(decode_times) / len(decode_times) * 1000
    decode_tps = 1000 / avg_decode
    print(f"Average decode: {avg_decode:.2f} ms/token")
    print(f"Throughput: {decode_tps:.1f} tok/s")
    print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    baseline_tg = 60.1  # From earlier profiling
    print(f"Baseline TG (stream_generate): {baseline_tg:.1f} tok/s")
    print(f"Actual TG (forward + eval):    {decode_tps:.1f} tok/s")
    print(f"Difference: {decode_tps - baseline_tg:.1f} tok/s ({(decode_tps/baseline_tg - 1)*100:.1f}%)")
    print()

    if abs(decode_tps - baseline_tg) < 5:
        print("✅ Sync overhead is minimal - GPU execution is the bottleneck")
        print("   → GEMV optimization will directly improve TG")
    elif decode_tps > baseline_tg * 1.1:
        print("⚠️ Forward is faster than stream_generate")
        print(f"   → Extra overhead: {baseline_tg/decode_tps * 100 - 100:.1f}% in generation loop")
        print("   → concat/norm dispatch overhead confirmed!")
    else:
        print("⚠️ Forward is slower - possible measurement issue")

    print()
    print("=" * 80)
    print("TIME BREAKDOWN")
    print("=" * 80)
    print()

    total_time_per_token = 1000 / baseline_tg  # ms
    gpu_time_per_token = avg_decode  # ms
    dispatch_overhead = total_time_per_token - gpu_time_per_token

    print(f"Total time per token:     {total_time_per_token:.2f} ms (100%)")
    print(f"GPU execution time:       {gpu_time_per_token:.2f} ms ({gpu_time_per_token/total_time_per_token*100:.1f}%)")
    print(f"Dispatch/Python overhead: {dispatch_overhead:.2f} ms ({dispatch_overhead/total_time_per_token*100:.1f}%)")

if __name__ == "__main__":
    profile_forward_passes()
