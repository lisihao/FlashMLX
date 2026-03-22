#!/usr/bin/env python3
"""
Profile mlx-lm generation with FlashMLX's profiler.

This script is meant to answer:
- Where does mlx-lm spend time during prompt prefill and decoding?
- Which MLX primitives dominate the runtime?
- How does the profile change with prompt length?
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx_lm import load, stream_generate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flashmlx.profiler import ProfileAnalyzer, Profiler, ProfilerConfig, InstrumentationLevel


DEFAULT_MODEL_PATH = Path.home() / "models" / "/Volumes/toshiba/models/qwen3.5-2b-opus-distilled"

reset_peak_memory = getattr(mx, "reset_peak_memory", mx.metal.reset_peak_memory)


def generate_prompt(length: int) -> str:
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


def parse_lengths(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def summarize_hotspots(analyzer: ProfileAnalyzer) -> None:
    stats = analyzer.get_function_stats()
    if not stats:
        print("No function events captured.")
        return

    categories = defaultdict(float)
    for name, data in stats.items():
        total = data["total_ms"]
        if "scaled_dot_product_attention" in name:
            categories["attention"] += total
        elif "rms_norm" in name:
            categories["rms_norm"] += total
        elif "matmul" in name:
            categories["matmul"] += total
        elif "softmax" in name or name.endswith("exp"):
            categories["softmax/exp"] += total
        elif "sum" in name:
            categories["reduction"] += total
        elif "concatenate" in name or "transpose" in name or "reshape" in name:
            categories["tensor_manipulation"] += total
        elif "rope" in name:
            categories["rope"] += total
        else:
            categories["other"] += total

    total_ms = sum(categories.values())
    if total_ms <= 0:
        total_ms = sum(data["total_ms"] for data in stats.values())

    print("\nCategory breakdown")
    print("-" * 80)
    for name, total in sorted(categories.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{name:<24} {total:>12.2f} ms  {total / total_ms * 100:>6.1f}%")

    print("\nTop hotspots")
    print("-" * 80)
    print(f"{'Function':<34} {'Total ms':>12} {'Calls':>8} {'%':>8}")
    for hotspot in analyzer.get_top_hotspots(15):
        print(
            f"{hotspot['name']:<34} "
            f"{hotspot['total_ms']:>12.2f} "
            f"{hotspot['count']:>8} "
            f"{hotspot['percent']:>7.1f}%"
        )


def summarize_input_shapes(analyzer: ProfileAnalyzer, names: tuple[str, ...]) -> None:
    shape_counts = defaultdict(int)
    shape_time = defaultdict(float)

    for event in analyzer.events:
        if event.get("event_type") != "function_call":
            continue
        name = event.get("name", "")
        if name not in names:
            continue
        shapes = json.dumps(event.get("input_shapes"), sort_keys=True)
        shape_counts[(name, shapes)] += 1
        shape_time[(name, shapes)] += float(event.get("duration_ms") or 0.0)

    if not shape_counts:
        print("\nNo shape data captured for targeted functions.")
        return

    print("\nShape breakdown")
    print("-" * 80)
    print(f"{'Function':<34} {'Calls':>8} {'Total ms':>12}  Shapes")
    ranked = sorted(shape_counts.items(), key=lambda kv: shape_time[kv[0]], reverse=True)
    for (name, shapes), count in ranked[:20]:
        total = shape_time[(name, shapes)]
        print(f"{name:<34} {count:>8} {total:>12.2f}  {shapes}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile mlx-lm generate with FlashMLX profiler")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--prompt-length", type=int, default=2981)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", default="/tmp/flashmlx_mlx_lm_profile.json")
    parser.add_argument("--profile-name", default="mlx_lm_generate")
    parser.add_argument("--level", choices=["basic", "detailed", "full"], default="full")
    parser.add_argument("--capture-memory", action="store_true", default=True)
    parser.add_argument("--capture-args", action="store_true", default=False)
    args = parser.parse_args()

    level_map = {
        "basic": InstrumentationLevel.BASIC,
        "detailed": InstrumentationLevel.DETAILED,
        "full": InstrumentationLevel.FULL,
    }

    config = ProfilerConfig(
        level=level_map[args.level],
        capture_memory=args.capture_memory,
        capture_args=args.capture_args,
    )
    prompt = generate_prompt(args.prompt_length)

    print("=" * 80)
    print("FlashMLX mlx-lm Profiling")
    print("=" * 80)
    print(f"model_path:   {args.model_path}")
    print(f"prompt_len:   {args.prompt_length}")
    print(f"max_tokens:   {args.max_tokens}")
    print(f"profile_lvl:  {args.level}")

    print("\nLoading model...")
    model, tokenizer = load(args.model_path)

    reset_peak_memory()

    with Profiler(args.profile_name, config=config) as profiler:
        with profiler.region("mlx_lm_stream_generate"):
            first_token_seen = False
            generated = 0
            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                prefill_step_size=2048,
            ):
                if not first_token_seen:
                    profiler.logger.log_event(
                        event_type="milestone",
                        name="first_token",
                        duration_ms=0.0,
                    )
                    first_token_seen = True
                generated += 1

    analyzer = ProfileAnalyzer(str(profiler.output_file))
    analyzer.print_summary()
    summarize_hotspots(analyzer)
    if args.capture_args:
        summarize_input_shapes(analyzer, ("mx.concatenate", "mx.fast.rms_norm", "mx.fast.scaled_dot_product_attention"))

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(
            {
                "profile_file": str(profiler.output_file),
                "model_path": args.model_path,
                "prompt_length": args.prompt_length,
                "max_tokens": args.max_tokens,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nProfile metadata saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
