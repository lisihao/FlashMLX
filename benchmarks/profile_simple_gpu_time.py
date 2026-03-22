#!/usr/bin/env python3
"""
Simple GPU time measurement: measure model() calls with mx.eval() sync
"""

import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load

MODEL_PATH = Path.home() / "models" / "/Volumes/toshiba/models/qwen3.5-2b-opus-distilled"

def measure_gpu_time():
    print("=" * 80)
    print("Simple GPU Time Measurement")
    print("=" * 80)
    print()

    print("Loading model...")
    model, tokenizer = load(str(MODEL_PATH))

    # Test with different sequence lengths
    test_cases = [
        ("Short (100 tokens)", 100),
        ("Medium (500 tokens)", 500),
        ("Long (1000 tokens)", 1000),
    ]

    print()
    print("=" * 80)
    print("FORWARD PASS TIMING (with mx.eval() sync)")
    print("=" * 80)
    print()

    for name, token_count in test_cases:
        prompt = "Hello world " * (token_count // 2)
        tokens = mx.array(tokenizer.encode(prompt))
        actual_tokens = len(tokens)

        # Warmup
        _ = model(tokens[None, :])
        mx.eval(model.parameters())

        # Measure
        times = []
        for _ in range(3):
            start = time.perf_counter()
            output = model(tokens[None, :])
            mx.eval(output)  # Force GPU sync
            duration = time.perf_counter() - start
            times.append(duration)

        avg_time = sum(times) / len(times) * 1000
        throughput = actual_tokens / (avg_time / 1000)

        print(f"{name}:")
        print(f"  Tokens:      {actual_tokens}")
        print(f"  Avg time:    {avg_time:.2f} ms")
        print(f"  Throughput:  {throughput:.1f} tok/s")
        print()

    # Now measure decode-like scenario (single token input)
    print("=" * 80)
    print("SINGLE TOKEN FORWARD (decode-like)")
    print("=" * 80)
    print()

    single_token = mx.array(tokenizer.encode("Hello"))[:1]

    # Warmup
    for _ in range(5):
        _ = model(single_token[None, :])
    mx.eval(model.parameters())

    # Measure
    times = []
    for i in range(100):
        start = time.perf_counter()
        output = model(single_token[None, :])
        mx.eval(output)
        duration = time.perf_counter() - start
        times.append(duration)

        if i < 10:
            print(f"  Token {i+1}: {duration*1000:.3f} ms")

    avg_time = sum(times) / len(times) * 1000
    throughput = 1000 / avg_time

    print()
    print(f"Average single-token forward: {avg_time:.3f} ms")
    print(f"Throughput: {throughput:.1f} tok/s")
    print()

    # Compare with baseline
    print("=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)
    print()

    baseline_tg = 60.1  # From stream_generate
    print(f"Baseline (stream_generate): {baseline_tg:.1f} tok/s")
    print(f"Direct forward + eval:      {throughput:.1f} tok/s")
    print()

    if throughput > baseline_tg * 1.2:
        overhead_pct = (baseline_tg / throughput - 1) * -100
        print(f"⚠️ stream_generate has {overhead_pct:.1f}% extra overhead")
        print("   → Likely from tokenization + concat/norm dispatch")
        print("   → Concat/Norm fusion optimization is HIGH priority!")
    elif abs(throughput - baseline_tg) < 5:
        print("✅ Overhead is minimal")
        print("   → GPU execution is the main bottleneck")
        print("   → GEMV optimization will directly help")
    else:
        print("⚠️ forward() is slower - unexpected")

if __name__ == "__main__":
    measure_gpu_time()
