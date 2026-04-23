#!/usr/bin/env python3
"""
TEP Phase B — Hot-Swap Quality Recovery Benchmark

Tests whether dynamic pool hot-swap can recover quality lost by K-1 clamping.

Three configurations compared:
  1. Standard (reference) — no offloading, full quality
  2. Compact + K-1 only — production-realistic, PP correct, TG K-1 clamped
  3. Compact + hot-swap — PP correct, compact after 1st token, hot-swap at token N

The hot-swap approach:
  - After compact, TG runs with K-1 clamped pool
  - After N tokens, analyze TG expert usage (zero-sync buffered)
  - Swap cold pool experts for hot non-pool experts
  - Continue TG with improved pool → quality should recover

Usage:
    python3 benchmarks/bench_hot_swap.py [--model PATH] [--tokens N]
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

MLX_LM_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mlx-lm-source")
sys.path.insert(0, MLX_LM_SOURCE)

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate


def compare_texts(reference: str, candidate: str) -> dict:
    """Compare two generated texts."""
    min_len = min(len(reference), len(candidate))
    max_len = max(len(reference), len(candidate))
    if max_len == 0:
        return {"match_ratio": 1.0, "first_diverge_char": -1, "first_diverge_word": -1}

    matches = sum(1 for a, b in zip(reference, candidate) if a == b)
    match_ratio = matches / max_len

    first_diverge = -1
    for i in range(min_len):
        if reference[i] != candidate[i]:
            first_diverge = i
            break
    if first_diverge == -1 and len(reference) != len(candidate):
        first_diverge = min_len

    ref_words = reference.split()
    cand_words = candidate.split()
    first_diverge_word = -1
    for i in range(min(len(ref_words), len(cand_words))):
        if ref_words[i] != cand_words[i]:
            first_diverge_word = i
            break
    if first_diverge_word == -1 and len(ref_words) != len(cand_words):
        first_diverge_word = min(len(ref_words), len(cand_words))

    return {
        "match_ratio": match_ratio,
        "first_diverge_char": first_diverge,
        "first_diverge_word": first_diverge_word,
    }


def run_standard(model_path: str, prompt: str, max_tokens: int) -> dict:
    """Standard inference reference."""
    print("\n  Loading model (standard)...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    mx.random.seed(42)
    mx.metal.reset_peak_memory()
    t0 = time.perf_counter()
    text = ""
    token_count = 0
    tg_tps = 0

    for response in stream_generate(model, tokenizer, formatted,
                                     max_tokens=max_tokens):
        text += response.text
        token_count += 1
        tg_tps = response.generation_tps

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024**3

    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()

    return {
        "label": "standard",
        "text": text,
        "tokens": token_count,
        "tg_tps": tg_tps,
        "total_ms": total * 1000,
        "peak_gb": peak,
    }


def run_compact_k1(model_path: str, prompt: str, max_tokens: int,
                   pool_size: int) -> dict:
    """Compact pool + K-1 clamping only (no hot-swap)."""
    from mlx_lm.models.expert_offload import patch_model_for_offload

    print(f"\n  Loading model (compact K-1, pool={pool_size})...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, model_path,
        pool_size=pool_size,
        max_workers=4,
        cpu_cache_gb=0.0,
        enable_prefetch=False,
        enable_telemetry=True,
    )
    gc.collect()

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    mx.random.seed(42)
    mx.metal.reset_peak_memory()
    t0 = time.perf_counter()
    text = ""
    token_count = 0
    tg_tps = 0
    compacted = False

    for response in stream_generate(model, tokenizer, formatted,
                                     max_tokens=max_tokens):
        token_count += 1
        if not compacted:
            ctx.compact(pool_size=pool_size)
            gc.collect()
            compacted = True
        text += response.text
        tg_tps = response.generation_tps

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024**3

    ctx.close()
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()

    return {
        "label": f"k1_pool{pool_size}",
        "text": text,
        "tokens": token_count,
        "tg_tps": tg_tps,
        "total_ms": total * 1000,
        "peak_gb": peak,
        "pool_size": pool_size,
    }


def run_hot_swap(model_path: str, prompt: str, max_tokens: int,
                 pool_size: int, swap_at: int, swap_budget: int) -> dict:
    """Compact pool + hot-swap after N tokens."""
    from mlx_lm.models.expert_offload import patch_model_for_offload

    print(f"\n  Loading model (hot-swap, pool={pool_size}, swap@{swap_at}, budget={swap_budget})...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, model_path,
        pool_size=pool_size,
        max_workers=4,
        cpu_cache_gb=0.0,
        enable_prefetch=False,
        enable_telemetry=True,
    )
    gc.collect()

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    mx.random.seed(42)
    mx.metal.reset_peak_memory()
    t0 = time.perf_counter()
    text = ""
    token_count = 0
    tg_tps = 0
    compacted = False
    swapped = False
    swap_stats = None

    for response in stream_generate(model, tokenizer, formatted,
                                     max_tokens=max_tokens):
        token_count += 1

        # Compact after first token (PP done)
        if not compacted:
            ctx.compact(pool_size=pool_size)
            gc.collect()
            compacted = True

        # Hot-swap after N TG tokens
        if compacted and not swapped and token_count >= swap_at:
            swap_stats = ctx.hot_swap(budget=swap_budget)
            swapped = True
            print(f"  Hot-swap @token {token_count}: "
                  f"{swap_stats['total_swapped']} experts across "
                  f"{swap_stats['layers_updated']} layers "
                  f"({swap_stats['elapsed_ms']:.0f}ms)")

        text += response.text
        tg_tps = response.generation_tps

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024**3

    ctx.close()
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()

    return {
        "label": f"hotswap_pool{pool_size}_at{swap_at}_b{swap_budget}",
        "text": text,
        "tokens": token_count,
        "tg_tps": tg_tps,
        "total_ms": total * 1000,
        "peak_gb": peak,
        "pool_size": pool_size,
        "swap_at": swap_at,
        "swap_budget": swap_budget,
        "swap_stats": swap_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="TEP Phase B — Hot-Swap Quality Recovery")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--pool", type=int, default=32)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP Phase B — Hot-Swap Quality Recovery")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.tokens}, Pool: {args.pool}")
    print("=" * 70)

    prompt = (
        "Explain the mathematical proof of the Pythagorean theorem "
        "using the area method, step by step."
    )

    all_results = []

    # 1. Standard reference
    print(f"\n{'=' * 60}")
    print(f"  [1/5] Standard (reference)")
    print(f"{'=' * 60}")
    ref = run_standard(args.model, prompt, args.tokens)
    all_results.append(ref)
    print(f"  TG: {ref['tg_tps']:.1f} tok/s | Tokens: {ref['tokens']}")
    print(f"  Text: {ref['text'][:80]}...")

    # 2. K-1 clamping only (baseline for quality loss)
    print(f"\n{'=' * 60}")
    print(f"  [2/5] Compact + K-1 only (no hot-swap)")
    print(f"{'=' * 60}")
    r_k1 = run_compact_k1(args.model, prompt, args.tokens, args.pool)
    comp_k1 = compare_texts(ref["text"], r_k1["text"])
    r_k1["comparison"] = comp_k1
    all_results.append(r_k1)
    print(f"  TG: {r_k1['tg_tps']:.1f} tok/s")
    print(f"  Match: {comp_k1['match_ratio']:.1%} | Diverge: word {comp_k1['first_diverge_word']}")

    # 3. Hot-swap at token 5, budget 16
    print(f"\n{'=' * 60}")
    print(f"  [3/5] Hot-swap @5 tokens, budget=16")
    print(f"{'=' * 60}")
    r_hs5 = run_hot_swap(args.model, prompt, args.tokens, args.pool,
                         swap_at=5, swap_budget=16)
    comp_hs5 = compare_texts(ref["text"], r_hs5["text"])
    r_hs5["comparison"] = comp_hs5
    all_results.append(r_hs5)
    print(f"  TG: {r_hs5['tg_tps']:.1f} tok/s")
    print(f"  Match: {comp_hs5['match_ratio']:.1%} | Diverge: word {comp_hs5['first_diverge_word']}")

    # 4. Hot-swap at token 3, budget 32
    print(f"\n{'=' * 60}")
    print(f"  [4/5] Hot-swap @3 tokens, budget=32")
    print(f"{'=' * 60}")
    r_hs3 = run_hot_swap(args.model, prompt, args.tokens, args.pool,
                         swap_at=3, swap_budget=32)
    comp_hs3 = compare_texts(ref["text"], r_hs3["text"])
    r_hs3["comparison"] = comp_hs3
    all_results.append(r_hs3)
    print(f"  TG: {r_hs3['tg_tps']:.1f} tok/s")
    print(f"  Match: {comp_hs3['match_ratio']:.1%} | Diverge: word {comp_hs3['first_diverge_word']}")

    # 5. Hot-swap at token 10, budget 32 (more TG data)
    print(f"\n{'=' * 60}")
    print(f"  [5/5] Hot-swap @10 tokens, budget=32")
    print(f"{'=' * 60}")
    r_hs10 = run_hot_swap(args.model, prompt, args.tokens, args.pool,
                          swap_at=10, swap_budget=32)
    comp_hs10 = compare_texts(ref["text"], r_hs10["text"])
    r_hs10["comparison"] = comp_hs10
    all_results.append(r_hs10)
    print(f"  TG: {r_hs10['tg_tps']:.1f} tok/s")
    print(f"  Match: {comp_hs10['match_ratio']:.1%} | Diverge: word {comp_hs10['first_diverge_word']}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  HOT-SWAP QUALITY RECOVERY SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Config':<35} {'TG tok/s':>9} {'Match%':>8} "
          f"{'Diverge@':>10} {'Swaps':>7} {'Verdict':>10}")
    print(f"  {'-'*35} {'-'*9} {'-'*8} {'-'*10} {'-'*7} {'-'*10}")

    for r in all_results:
        comp = r.get("comparison", {})
        match = comp.get("match_ratio", 1.0)
        diverge = comp.get("first_diverge_word", -1)
        div_str = f"word {diverge}" if diverge >= 0 else "never"
        ss = r.get("swap_stats", {})
        swaps = ss.get("total_swapped", 0) if ss else "N/A"

        if match >= 0.95:
            verdict = "EXCELLENT"
        elif match >= 0.80:
            verdict = "GOOD"
        elif match >= 0.50:
            verdict = "MODERATE"
        elif match >= 0.20:
            verdict = "POOR"
        else:
            verdict = "SEVERE"
        if r["label"] == "standard":
            verdict = "REFERENCE"

        print(f"  {r['label']:<35} {r['tg_tps']:>8.1f} "
              f"{match:>7.1%} {div_str:>10} {str(swaps):>7} {verdict:>10}")

    # Recovery analysis
    k1_match = comp_k1["match_ratio"]
    best_hs_match = max(
        comp_hs5["match_ratio"],
        comp_hs3["match_ratio"],
        comp_hs10["match_ratio"],
    )
    recovery = (best_hs_match - k1_match) / (1.0 - k1_match) if k1_match < 1.0 else 0
    print(f"\n  K-1 baseline: {k1_match:.1%} match")
    print(f"  Best hot-swap: {best_hs_match:.1%} match")
    print(f"  Recovery: {recovery:.1%} of lost quality recovered")

    if recovery > 0.5:
        print(f"  → HOT-SWAP HIGH VALUE: recovers >{recovery:.0%} of quality")
    elif recovery > 0.1:
        print(f"  → HOT-SWAP MODERATE VALUE: recovers {recovery:.0%}")
    else:
        print(f"  → HOT-SWAP LOW VALUE: only {recovery:.0%} recovery")
        print(f"    Consider: earlier swap, larger budget, or multi-round swaps")

    # Save
    report = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tokens": args.tokens,
        "pool_size": args.pool,
        "prompt": prompt,
        "results": all_results,
    }
    out_path = args.output
    if not out_path:
        solar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".solar")
        os.makedirs(solar_dir, exist_ok=True)
        out_path = os.path.join(solar_dir, "tep-hot-swap.json")

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
