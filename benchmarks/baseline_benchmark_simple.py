#!/usr/bin/env python3
"""
Generation strategy benchmark for FlashMLX / mlx-lm.

This script is meant to answer practical questions:
- Is TTFT scaling clean?
- What does decode throughput look like?
- Does KV cache configuration change the tradeoff?
- How far can we push this machine with mlx-lm knobs?

The script keeps the baseline path aligned with current mlx-lm APIs:
- stream_generate for streaming measurement
- optional max_kv_size / kv_bits knobs for cache experiments
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import mlx.core as mx
from mlx_lm import load, stream_generate


DEFAULT_MODEL_PATH = Path.home() / "models" / "/Volumes/toshiba/models/qwen3.5-2b-opus-distilled"
DEFAULT_PROMPT_LENGTHS = [745, 2981, 11926]

reset_peak_memory = getattr(mx, "reset_peak_memory", mx.metal.reset_peak_memory)
get_peak_memory = getattr(mx, "get_peak_memory", mx.metal.get_peak_memory)


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


def parse_lengths(raw: str) -> list[int]:
    lengths = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        lengths.append(int(item))
    if not lengths:
        raise ValueError("At least one prompt length must be provided")
    return lengths


def run_single_test(
    model,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int,
    test_name: str,
    mode: str,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    max_kv_size: Optional[int] = None,
    prefill_step_size: int = 2048,
) -> dict:
    """Measure one prompt once using stream_generate."""
    print(f"\n{'=' * 80}")
    print(f"{test_name}")
    print(f"{'=' * 80}")

    reset_peak_memory()

    start_time = time.perf_counter()
    first_token_time = None
    end_time = None

    generated_tokens = 0
    gen_kwargs = {
        "max_tokens": max_tokens,
        "prefill_step_size": prefill_step_size,
        "kv_bits": kv_bits,
        "kv_group_size": kv_group_size,
        "quantized_kv_start": quantized_kv_start,
    }
    if max_kv_size is not None:
        gen_kwargs["max_kv_size"] = max_kv_size

    for response in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        **gen_kwargs,
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
    peak_memory_gb = get_peak_memory() / (1024**3)
    prompt_tps = prompt_tokens / ttft_s if ttft_s > 0 else 0.0

    result = {
        "test_name": test_name,
        "mode": mode,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "ttft_ms": round(ttft_s * 1000, 1),
        "prompt_tps": round(prompt_tps, 1),
        "decode_tps": round(decode_tps, 1),
        "total_time_s": round(total_s, 2),
        "peak_memory_gb": round(peak_memory_gb, 2),
    }

    print("\nResults")
    print(f"  prompt_tokens:    {result['prompt_tokens']}")
    print(f"  generated_tokens: {result['generated_tokens']}")
    print(f"  TTFT:             {result['ttft_ms']:.1f} ms")
    print(f"  prompt_tps:       {result['prompt_tps']:.1f} tok/s")
    print(f"  decode_tps:       {result['decode_tps']:.1f} tok/s")
    print(f"  peak_memory_gb:    {result['peak_memory_gb']:.2f} GB")

    return result


def benchmark_mode(
    *,
    mode_name: str,
    model,
    tokenizer,
    prompt_lengths: Iterable[int],
    max_tokens: int,
    runs: int,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    max_kv_size: Optional[int] = None,
    prefill_step_size: int = 2048,
) -> list[dict]:
    results = []

    for prompt_length in prompt_lengths:
        prompt = generate_prompt(prompt_length)
        prompt_tokens = len(tokenizer.encode(prompt))
        for run_idx in range(runs):
            test_name = (
                f"{mode_name.upper()} | prompt={prompt_length} "
                f"(~{prompt_tokens} tok) | run={run_idx + 1}/{runs}"
            )
            result = run_single_test(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                test_name=test_name,
                mode=mode_name,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
                max_kv_size=max_kv_size,
                prefill_step_size=prefill_step_size,
            )
            result["prompt_length"] = prompt_length
            result["run_idx"] = run_idx + 1
            result["prompt_tokens"] = prompt_tokens
            results.append(result)

    return results


def summarize_mode(mode_name: str, results: list[dict]) -> None:
    if not results:
        return

    ttft_values = [r["ttft_ms"] for r in results]
    decode_values = [r["decode_tps"] for r in results]
    memory_values = [r["peak_memory_gb"] for r in results]
    p95_index = max(0, int(len(ttft_values) * 0.95) - 1)

    print("\n" + "=" * 80)
    print(f"{mode_name.upper()} SUMMARY")
    print("=" * 80)
    print(f"runs:            {len(results)}")
    print(f"TTFT avg/p95:     {statistics.mean(ttft_values):.1f} ms / {sorted(ttft_values)[p95_index]:.1f} ms")
    print(f"decode_tps avg:  {statistics.mean(decode_values):.1f} tok/s")
    print(f"peak_mem avg:     {statistics.mean(memory_values):.2f} GB")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlashMLX generation strategy benchmark")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the main MLX model",
    )
    parser.add_argument(
        "--prompt-lengths",
        default="745,2981,11926",
        help="Comma-separated target prompt lengths",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Tokens to generate per run",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="How many times to repeat each prompt length",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="Optional KV cache quantization bits",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help="KV cache quantization group size",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=0,
        help="Step at which KV quantization starts",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Optional rotating KV cache size",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Prompt prefill chunk size",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "generation_strategy_results.json"),
        help="Where to write JSON results",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    prompt_lengths = parse_lengths(args.prompt_lengths)

    print("=" * 80)
    print("FlashMLX Generation Strategy Benchmark")
    print("=" * 80)
    print("\nConfiguration")
    print(f"  model_path:         {args.model_path}")
    print(f"  prompt_lengths:      {prompt_lengths}")
    print(f"  max_tokens:          {args.max_tokens}")
    print(f"  runs:                {args.runs}")
    print(f"  kv_bits:             {args.kv_bits}")
    print(f"  kv_group_size:       {args.kv_group_size}")
    print(f"  quantized_kv_start:  {args.quantized_kv_start}")
    print(f"  max_kv_size:         {args.max_kv_size}")
    print(f"  prefill_step_size:   {args.prefill_step_size}")

    print("\nLoading main model...")
    try:
        model, tokenizer = load(args.model_path)
    except Exception as exc:
        print(f"Failed to load main model: {exc}")
        return 1

    all_results = {}

    baseline_results = benchmark_mode(
        mode_name="baseline",
        model=model,
        tokenizer=tokenizer,
        prompt_lengths=prompt_lengths,
        max_tokens=args.max_tokens,
        runs=args.runs,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        max_kv_size=args.max_kv_size,
        prefill_step_size=args.prefill_step_size,
    )
    all_results["baseline"] = baseline_results
    summarize_mode("baseline", baseline_results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "model_path": args.model_path,
            "prompt_lengths": prompt_lengths,
            "max_tokens": args.max_tokens,
            "runs": args.runs,
            "kv_bits": args.kv_bits,
            "kv_group_size": args.kv_group_size,
            "quantized_kv_start": args.quantized_kv_start,
            "max_kv_size": args.max_kv_size,
            "prefill_step_size": args.prefill_step_size,
        },
        "results": all_results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
