#!/usr/bin/env python3
"""
Detailed profiling script for FlashMLX.

This script uses the FlashMLX Profiler to analyze performance bottlenecks
in the token generation pipeline, with focus on:
- GEMV operations
- Flash Attention
- GatedDeltaNet cache/concat
- Memory access patterns
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

# Import FlashMLX profiler if available
try:
    from flashmlx.profiler import Profiler, ProfilerConfig, InstrumentationLevel
    PROFILER_AVAILABLE = True
except ImportError:
    print("Warning: FlashMLX profiler not available, using basic timing")
    PROFILER_AVAILABLE = False


DEFAULT_MODEL_PATH = Path.home() / "models" / "/Volumes/toshiba/models/qwen3.5-2b-opus-distilled"
DEFAULT_PROMPT_LENGTH = 2981  # Medium length for balanced analysis


def generate_prompt(length: int) -> str:
    """Generate a prompt with approximately the requested token length."""
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In the realm of artificial intelligence, large language models "
        "have demonstrated remarkable capabilities across diverse tasks. "
    )
    words_needed = int(length * 1.3)
    words = base_text.split()
    prompt_words = []
    while len(prompt_words) < words_needed:
        prompt_words.extend(words)
    return " ".join(prompt_words[:words_needed])


def run_basic_profiling(model, tokenizer, prompt: str, max_tokens: int) -> dict:
    """Run basic profiling without FlashMLX profiler."""
    import time

    print("\nRunning basic profiling (no FlashMLX profiler)...")

    start_time = time.perf_counter()
    first_token_time = None
    generated_tokens = 0

    for response in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    ):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        generated_tokens += 1

    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time

    ttft_s = first_token_time - start_time
    total_s = end_time - start_time
    decode_s = max(end_time - first_token_time, 1e-9)
    decode_tokens = max(generated_tokens - 1, 0)
    decode_tps = decode_tokens / decode_s if decode_tokens > 0 else 0.0

    prompt_tokens = len(tokenizer.encode(prompt))
    prompt_tps = prompt_tokens / ttft_s if ttft_s > 0 else 0.0

    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "ttft_ms": round(ttft_s * 1000, 1),
        "prompt_tps": round(prompt_tps, 1),
        "decode_tps": round(decode_tps, 1),
        "total_time_s": round(total_s, 2),
    }


def run_detailed_profiling(model, tokenizer, prompt: str, max_tokens: int) -> dict:
    """Run detailed profiling with FlashMLX profiler."""
    print("\nRunning detailed profiling with FlashMLX profiler...")

    config = ProfilerConfig(
        name="baseline_detailed",
        level=InstrumentationLevel.FULL,
        capture_memory=True,
        capture_kernels=True,
        capture_stack=True,
        min_function_time_ms=0.0,  # Capture all calls (MLX is lazy)
    )

    with Profiler("baseline_detailed", config=config) as profiler:
        generated_tokens = 0
        for response in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        ):
            generated_tokens += 1

    # Get profiler results
    results = profiler.get_results()

    # Extract key metrics
    function_times = {}
    if "performance" in results:
        for func_name, func_data in results["performance"].items():
            if "total_time" in func_data:
                function_times[func_name] = func_data["total_time"]

    # Sort by time
    sorted_functions = sorted(
        function_times.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Print top 10 functions
    print("\nTop 10 time-consuming functions:")
    print(f"{'Function':<50} {'Time (ms)':<15} {'%':<10}")
    print("-" * 75)

    total_time = sum(function_times.values())
    for func_name, func_time in sorted_functions[:10]:
        percentage = (func_time / total_time * 100) if total_time > 0 else 0
        print(f"{func_name:<50} {func_time*1000:<15.2f} {percentage:<10.1f}")

    return {
        "profiler_results": results,
        "function_times": function_times,
        "sorted_functions": sorted_functions[:20],  # Top 20
        "generated_tokens": generated_tokens,
    }


def analyze_results(basic_results: dict, detailed_results: dict | None) -> dict:
    """Analyze and summarize profiling results."""
    analysis = {
        "basic_metrics": basic_results,
    }

    if detailed_results:
        # Analyze function categories
        gemv_time = 0
        attention_time = 0
        concat_time = 0
        norm_time = 0
        other_time = 0

        for func_name, func_time in detailed_results["function_times"].items():
            func_lower = func_name.lower()
            if "gemv" in func_lower or "matmul" in func_lower:
                gemv_time += func_time
            elif "attention" in func_lower or "flash" in func_lower:
                attention_time += func_time
            elif "concat" in func_lower or "cache" in func_lower:
                concat_time += func_time
            elif "norm" in func_lower or "rms" in func_lower:
                norm_time += func_time
            else:
                other_time += func_time

        total_time = gemv_time + attention_time + concat_time + norm_time + other_time

        analysis["category_breakdown"] = {
            "gemv": {"time_ms": gemv_time * 1000, "percentage": gemv_time / total_time * 100 if total_time > 0 else 0},
            "attention": {"time_ms": attention_time * 1000, "percentage": attention_time / total_time * 100 if total_time > 0 else 0},
            "concat_cache": {"time_ms": concat_time * 1000, "percentage": concat_time / total_time * 100 if total_time > 0 else 0},
            "norm": {"time_ms": norm_time * 1000, "percentage": norm_time / total_time * 100 if total_time > 0 else 0},
            "other": {"time_ms": other_time * 1000, "percentage": other_time / total_time * 100 if total_time > 0 else 0},
        }

        print("\n" + "=" * 80)
        print("Category Breakdown")
        print("=" * 80)
        print(f"{'Category':<20} {'Time (ms)':<15} {'Percentage':<15}")
        print("-" * 50)
        for category, data in analysis["category_breakdown"].items():
            print(f"{category:<20} {data['time_ms']:<15.2f} {data['percentage']:<15.1f}%")

    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlashMLX detailed profiling")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the MLX model",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=DEFAULT_PROMPT_LENGTH,
        help="Target prompt length",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Tokens to generate",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "profiling_results.json"),
        help="Where to write JSON results",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 80)
    print("FlashMLX Detailed Profiling")
    print("=" * 80)
    print("\nConfiguration")
    print(f"  model_path:      {args.model_path}")
    print(f"  prompt_length:   {args.prompt_length}")
    print(f"  max_tokens:      {args.max_tokens}")
    print(f"  profiler:        {'FlashMLX' if PROFILER_AVAILABLE else 'Basic'}")

    print("\nLoading model...")
    try:
        model, tokenizer = load(args.model_path)
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        return 1

    # Generate prompt
    prompt = generate_prompt(args.prompt_length)

    # Run basic profiling
    basic_results = run_basic_profiling(model, tokenizer, prompt, args.max_tokens)

    # Run detailed profiling if available
    detailed_results = None
    if PROFILER_AVAILABLE:
        try:
            detailed_results = run_detailed_profiling(model, tokenizer, prompt, args.max_tokens)
        except Exception as exc:
            print(f"Warning: Detailed profiling failed: {exc}")

    # Analyze results
    analysis = analyze_results(basic_results, detailed_results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "model_path": args.model_path,
            "prompt_length": args.prompt_length,
            "max_tokens": args.max_tokens,
        },
        "analysis": analysis,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n\nSaved results to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
