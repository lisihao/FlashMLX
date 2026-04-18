#!/usr/bin/env python3
"""
TEP Experiment C — Miss Policy Gradient

Tests 4 miss policies on a gradient from "perfectly correct" to "completely wrong":
  1. true_load  — SSD load real expert (correct, slow)
  2. cpu_cache  — CPU cache hit, pre-warmed (correct, fast on UMA)
  3. zero_out   — zero missed expert's output (dropout-like, fast)
  4. k1_clamp   — use wrong expert from K-1 slot (current production, fast, garbage)

Key question: Is there a policy between true_load and k1_clamp that gives
acceptable quality at near-standard speed?

Usage:
    python3 benchmarks/bench_miss_policy.py [--model PATH] [--tokens N]
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


def generate_with_stats(model, tokenizer, prompt_text: str,
                        max_tokens: int = 150, seed: int = 42) -> dict:
    """Generate text with timing stats."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    mx.random.seed(seed)
    mx.metal.reset_peak_memory()
    gc.collect()

    t0 = time.perf_counter()
    text = ""
    token_count = 0
    tg_tps = 0
    got_first = False
    token_times = []

    for response in stream_generate(model, tokenizer, formatted,
                                     max_tokens=max_tokens):
        now = time.perf_counter()
        if not got_first:
            ttof = now - t0
            got_first = True
            prev_time = now
        else:
            dt = now - prev_time
            token_times.append(dt * 1000)
            prev_time = now
        text += response.text
        token_count += 1
        tg_tps = response.generation_tps

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024 / 1024 / 1024

    tpot_arr = np.array(token_times[3:]) if len(token_times) > 3 else np.array(token_times)

    result = {
        "text": text,
        "tokens": token_count,
        "tg_tps": tg_tps,
        "total_ms": total * 1000,
        "peak_gb": peak,
    }

    if len(tpot_arr) > 0:
        result["tpot"] = {
            "mean_ms": float(tpot_arr.mean()),
            "p50_ms": float(np.percentile(tpot_arr, 50)),
            "p95_ms": float(np.percentile(tpot_arr, 95)),
            "std_ms": float(tpot_arr.std()),
        }

    return result


def compare_texts(reference: str, candidate: str) -> dict:
    """Compare two generated texts."""
    max_len = max(len(reference), len(candidate))
    if max_len == 0:
        return {"match_ratio": 1.0}
    matches = sum(1 for a, b in zip(reference, candidate) if a == b)
    return {
        "match_ratio": matches / max_len,
        "ref_len": len(reference),
        "cand_len": len(candidate),
    }


def _get_switch_layers(model):
    """Get all FlashMoeSwitchGLU layers."""
    from mlx_lm.models.expert_offload import FlashMoeSwitchGLU
    inner = model
    for attr in ("model", "model", "language_model", "model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    switches = []
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if isinstance(sw, FlashMoeSwitchGLU):
                switches.append(sw)
    return switches


def run_standard(model_path: str, prompt: str, max_tokens: int) -> dict:
    """Standard inference baseline."""
    print("\n  Loading model (standard)...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    result = generate_with_stats(model, tokenizer, prompt, max_tokens)
    result["label"] = "standard"
    result["miss_policy"] = "none"

    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    return result


def run_with_policy(model_path: str, prompt: str, max_tokens: int,
                    pool_size: int, miss_policy: str, label: str,
                    cpu_cache_gb: float = 0.0) -> dict:
    """Run with a specific miss policy.

    miss_policy:
      "k1_clamp"  — current production: K-1 clamping (wrong expert)
      "zero_out"   — zero missed expert output (dropout-like)
      "true_load"  — force_miss_load=True, no CPU cache (pure SSD)
      "cpu_cache"  — force_miss_load=True, with pre-warmed CPU cache
    """
    from mlx_lm.models.expert_offload import patch_model_for_offload, FlashMoeSwitchGLU

    # cpu_cache policy needs CPU cache for warm hits
    actual_cpu_cache_gb = 8.0 if miss_policy == "cpu_cache" else cpu_cache_gb

    print(f"\n  Loading model (pool={pool_size}, policy={miss_policy})...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, model_path,
        pool_size=pool_size,
        max_workers=4,
        cpu_cache_gb=actual_cpu_cache_gb,
        enable_prefetch=False,
        enable_telemetry=True,
    )
    gc.collect()

    # Dummy generation to populate PP buffer
    dummy_msgs = [{"role": "user", "content": "Hello, what is 1+1?"}]
    dummy_fmt = tokenizer.apply_chat_template(
        dummy_msgs, add_generation_prompt=True, tokenize=False
    )
    for resp in stream_generate(model, tokenizer, dummy_fmt, max_tokens=5):
        pass

    # Compact from PP data
    ctx.compact(pool_size=pool_size, disable_coverage_gate=True)
    gc.collect()

    # Apply miss policy to all switch layers
    switches = _get_switch_layers(model)
    if miss_policy in ("true_load", "cpu_cache"):
        for sw in switches:
            sw._force_miss_load = True
    elif miss_policy == "zero_out":
        for sw in switches:
            sw._miss_policy = "zero_out"
    # k1_clamp is default behavior, no changes needed

    # Clear GPU cache
    mx.metal.set_cache_limit(0)
    gc.collect()
    mx.metal.clear_cache()
    mx.metal.set_cache_limit(int(64 * 1024**3))

    mem_after = mx.metal.get_active_memory() / 1024**3
    print(f"  After compact: {mem_after:.2f} GB")

    # Benchmark generation
    result = generate_with_stats(model, tokenizer, prompt, max_tokens)
    result["label"] = label
    result["miss_policy"] = miss_policy
    result["pool_size"] = pool_size
    result["mem_after_compact_gb"] = mem_after

    # Telemetry
    if ctx.telemetry:
        tel = ctx.telemetry.summary()
        miss_lat = tel.get("miss_latency", {})
        result["miss_count"] = miss_lat.get("count", 0)
        result["miss_mean_ms"] = miss_lat.get("mean_ms", 0)
        result["hit_rate"] = tel.get("overall_pool_hit_rate", 0)
        result["miss_sources"] = miss_lat.get("source_counts", {})

    ctx.close()
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    mx.metal.set_cache_limit(0)
    return result


def main():
    parser = argparse.ArgumentParser(description="TEP Exp C — Miss Policy Gradient")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit",
                        help="Path to MoE model")
    parser.add_argument("--tokens", type=int, default=150,
                        help="Max tokens to generate")
    parser.add_argument("--pool-size", type=int, default=32,
                        help="Pool size for offloaded runs")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP Experiment C — Miss Policy Gradient")
    print(f"  Model: {args.model}")
    print(f"  Pool: {args.pool_size}/256  |  Tokens: {args.tokens}")
    print("=" * 70)

    prompt = (
        "Explain the mathematical foundations of attention mechanisms in "
        "transformers, including scaled dot-product attention and multi-head "
        "attention. Provide the full mathematical derivation."
    )

    all_results = []

    configs = [
        ("standard", "none", {}),
        ("true_load", "true_load", {"pool_size": args.pool_size}),
        ("cpu_cache", "cpu_cache", {"pool_size": args.pool_size}),
        ("zero_out", "zero_out", {"pool_size": args.pool_size}),
        ("k1_clamp", "k1_clamp", {"pool_size": args.pool_size}),
    ]

    for i, (label, policy, cfg) in enumerate(configs):
        print(f"\n{'=' * 60}")
        print(f"  [{i+1}/{len(configs)}] {label} (policy={policy})")
        print(f"{'=' * 60}")

        if label == "standard":
            r = run_standard(args.model, prompt, args.tokens)
        else:
            r = run_with_policy(args.model, prompt, args.tokens,
                               pool_size=cfg["pool_size"],
                               miss_policy=policy, label=label)

        all_results.append(r)
        tpot = r.get("tpot", {})
        print(f"\n  -> TG: {r['tg_tps']:.1f} tok/s | Peak: {r['peak_gb']:.2f} GB")
        if tpot:
            print(f"     TPOT: p50={tpot['p50_ms']:.2f}ms, p95={tpot['p95_ms']:.2f}ms")
        miss_n = r.get("miss_count", 0)
        if miss_n > 0:
            print(f"     Misses: {miss_n} × {r.get('miss_mean_ms', 0):.3f}ms")
        print(f"     Text: {r['text'][:100]}...")

    # Summary table
    ref_text = all_results[0]["text"]
    baseline_tg = all_results[0]["tg_tps"]

    print(f"\n{'=' * 70}")
    print(f"  MISS POLICY GRADIENT SUMMARY (pool={args.pool_size}/256)")
    print(f"{'=' * 70}")
    print(f"  {'Policy':<12} {'TG tok/s':>9} {'vs Std':>7} {'TPOT p50':>10} "
          f"{'Match%':>7} {'Misses':>7} {'Verdict':>12}")
    print(f"  {'-'*12} {'-'*9} {'-'*7} {'-'*10} {'-'*7} {'-'*7} {'-'*12}")

    for r in all_results:
        tpot = r.get("tpot", {})
        comp = compare_texts(ref_text, r["text"]) if r["label"] != "standard" else {}
        match_pct = comp.get("match_ratio", 1.0) * 100
        delta = (r["tg_tps"] / baseline_tg - 1) * 100 if baseline_tg > 0 else 0
        miss_n = r.get("miss_count", 0)

        # Quality verdict
        if match_pct >= 95:
            verdict = "EXCELLENT"
        elif match_pct >= 80:
            verdict = "GOOD"
        elif match_pct >= 50:
            verdict = "MODERATE"
        elif match_pct >= 20:
            verdict = "POOR"
        else:
            verdict = "GARBAGE"
        if r["label"] == "standard":
            verdict = "REFERENCE"

        print(f"  {r['label']:<12} {r['tg_tps']:>8.1f} {delta:>+6.0f}% "
              f"{tpot.get('p50_ms', 0):>9.2f}ms "
              f"{match_pct:>6.1f}% {miss_n:>7} {verdict:>12}")

    # Quality analysis
    print(f"\n  Quality Analysis (first 200 chars):")
    for r in all_results:
        print(f"    {r['label']:<12}: {r['text'][:200]}")
        print()

    # Go/No-Go
    print(f"  Go/No-Go:")
    for r in all_results[1:]:
        comp = compare_texts(ref_text, r["text"])
        degradation = 1.0 - r["tg_tps"] / baseline_tg if baseline_tg > 0 else 0
        match = comp["match_ratio"]
        print(f"    {r['label']:<12}: {degradation:+.0%} speed, {match:.0%} quality "
              f"→ {'GO' if match > 0.8 and degradation < 0.3 else 'NO-GO'}")

    # Save
    report = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tokens": args.tokens,
        "pool_size": args.pool_size,
        "prompt": prompt,
        "results": all_results,
    }

    out_path = args.output
    if not out_path:
        solar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".solar")
        os.makedirs(solar_dir, exist_ok=True)
        out_path = os.path.join(solar_dir, "tep-miss-policy.json")

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
