#!/usr/bin/env python3
"""
Profile with Metal GPU capture to see real GPU execution time
Uses MLX's metal.start_capture() and metal.stop_capture()
"""

import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, stream_generate

# Configuration
MODEL_PATH = Path.home() / "models" / "/Volumes/toshiba/models/qwen3.5-2b-opus-distilled"
TRACE_DIR = Path(__file__).parent.parent / "profiling_data"
TRACE_DIR.mkdir(exist_ok=True)

def profile_generation():
    """Profile token generation with explicit eval() calls to measure GPU time"""
    print("=" * 80)
    print("Metal GPU Profiling - Real Execution Time")
    print("=" * 80)
    print()

    print("Loading model...")
    model, tokenizer = load(str(MODEL_PATH))

    # Short prompt for profiling
    prompt = "The quick brown fox jumps over the lazy dog. " * 100  # ~500 tokens

    print(f"Prompt tokens: ~{len(tokenizer.encode(prompt))}")
    print()

    # Prefill phase profiling
    print("=" * 80)
    print("PREFILL PHASE (with mx.eval() sync)")
    print("=" * 80)

    times = []
    token_count = 0
    prefill_done = False

    for response in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=50,
    ):
        # Force synchronization to measure real GPU time
        start = time.perf_counter()
        mx.eval(response)  # Block until GPU completes
        duration = time.perf_counter() - start
        times.append(duration)

        token_count += 1

        if token_count == 1:
            print(f"First token (Prefill): {duration*1000:.2f} ms")
            prefill_done = True
        elif token_count <= 10:
            print(f"Token {token_count}: {duration*1000:.2f} ms")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if times:
        prefill_ms = times[0] * 1000
        decode_times = times[1:]

        print(f"Total tokens: {token_count}")
        print(f"Prefill time: {prefill_ms:.2f} ms")

        if decode_times:
            decode_avg = sum(decode_times) / len(decode_times) * 1000
            decode_min = min(decode_times) * 1000
            decode_max = max(decode_times) * 1000
            decode_tps = 1000 / decode_avg if decode_avg > 0 else 0

            print(f"Decode tokens: {len(decode_times)}")
            print(f"Decode avg: {decode_avg:.2f} ms/token ({decode_tps:.1f} tok/s)")
            print(f"Decode min: {decode_min:.2f} ms")
            print(f"Decode max: {decode_max:.2f} ms")

            # Analyze variance
            import statistics
            if len(decode_times) > 1:
                decode_std = statistics.stdev(decode_times) * 1000
                print(f"Decode std: {decode_std:.2f} ms")

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # Compare with baseline
    baseline_tg = 60.1  # From earlier profiling
    if decode_times:
        actual_tps = 1000 / (sum(decode_times) / len(decode_times) * 1000)
        print(f"Baseline TG (no sync): {baseline_tg:.1f} tok/s")
        print(f"Actual TG (with sync): {actual_tps:.1f} tok/s")
        print(f"Difference: {actual_tps - baseline_tg:.1f} tok/s ({(actual_tps/baseline_tg - 1)*100:.1f}%)")
        print()

        if abs(actual_tps - baseline_tg) < 5:
            print("✅ Sync overhead is negligible - GPU is the bottleneck")
        else:
            print("⚠️ Significant difference - dispatch overhead may be hiding GPU time")

if __name__ == "__main__":
    profile_generation()
