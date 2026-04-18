#!/usr/bin/env python3
"""
TEP Diagnostic — Phase-Split Timing + Layer-Wise Miss Analysis

Purpose: Identify whether the 42 vs 68 tok/s gap is cold-start or steady-state,
and where exactly the misses happen.

Collects per-token:
  - Wall-clock timestamp (for phase-split TG analysis)
  - Per-layer miss delta (from telemetry snapshots)
  - Cumulative hit/miss state

Reports:
  - Phase A (token 1-16): cold-start behavior
  - Phase B (token 17-64): warming up
  - Phase C (token 65+): steady-state
  - Per-layer miss heatmap (which layers miss most)
  - Miss token position histogram (do misses cluster?)

Usage:
    python3 benchmarks/bench_diagnostic.py [--model PATH] [--tokens N]
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


def _get_switch_layers(model):
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


def run_standard(model_path, prompt, max_tokens):
    """Standard inference with per-token timing for phase comparison."""
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

    per_token = []
    t0 = time.perf_counter()
    got_first = False

    for response in stream_generate(model, tokenizer, formatted, max_tokens=max_tokens):
        now = time.perf_counter()
        idx = len(per_token)
        if not got_first:
            got_first = True
            dt_ms = (now - t0) * 1000  # includes PP
        else:
            dt_ms = (now - per_token[-1]["wall_time"]) * 1000
        per_token.append({
            "idx": idx,
            "wall_time": now,
            "dt_ms": dt_ms,
            "text_chunk": response.text,
            "tg_tps": response.generation_tps,
        })

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024**3
    text = "".join(t["text_chunk"] for t in per_token)

    result = {
        "label": "standard",
        "text": text,
        "tokens": len(per_token),
        "total_ms": total * 1000,
        "peak_gb": peak,
        "per_token": per_token,
    }

    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    return result


def run_diagnostic(model_path, prompt, max_tokens, pool_size, warmup_tokens):
    """decode_recompact + zero_out with full per-token + per-layer diagnostics."""
    from mlx_lm.models.expert_offload import patch_model_for_offload

    print(f"\n  Loading model (diagnostic: pool={pool_size}, warmup={warmup_tokens})...")
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

    # Dummy gen to populate PP buffer
    dummy_msgs = [{"role": "user", "content": "Hello, what is 1+1?"}]
    dummy_fmt = tokenizer.apply_chat_template(
        dummy_msgs, add_generation_prompt=True, tokenize=False
    )
    for resp in stream_generate(model, tokenizer, dummy_fmt, max_tokens=5):
        pass

    # Compact from PP
    ctx.compact(pool_size=pool_size, disable_coverage_gate=True)
    gc.collect()

    mx.metal.set_cache_limit(0)
    gc.collect()
    mx.metal.clear_cache()
    mx.metal.set_cache_limit(int(64 * 1024**3))

    # Warmup with force_miss_load → builds TG data for decode_recompact
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    switches = _get_switch_layers(model)
    for sw in switches:
        sw._force_miss_load = True

    mx.random.seed(42)
    for response in stream_generate(model, tokenizer, formatted, max_tokens=warmup_tokens):
        pass
    print(f"  Warmup: {warmup_tokens} tokens (force_miss_load)")

    # Decode recompact
    recompact_stats = ctx.decode_recompact(pool_size=pool_size)
    gc.collect()
    print(f"  Recompact: {recompact_stats['total_swapped']} swaps, "
          f"HR: {recompact_stats['avg_old_hit_rate']:.1%} → {recompact_stats['avg_new_hit_rate']:.1%}")

    # Set zero_out policy
    for sw in switches:
        sw._force_miss_load = False
        sw._miss_policy = "zero_out"

    # Reset telemetry counters for clean measurement
    tel = ctx.telemetry
    tel._pool_hits[:] = 0
    tel._pool_misses[:] = 0
    tel._miss_latencies = []
    tel._miss_source_counts = {"pool": 0, "cpu_cache": 0, "ssd": 0}

    mem_after = mx.metal.get_active_memory() / 1024**3
    print(f"  After recompact: {mem_after:.2f} GB, policy=zero_out")

    # Benchmark with per-token telemetry snapshots
    mx.random.seed(42)
    mx.metal.reset_peak_memory()

    num_layers = tel.num_layers
    prev_hits = tel._pool_hits.copy()
    prev_misses = tel._pool_misses.copy()

    per_token = []
    t0 = time.perf_counter()
    got_first = False

    for response in stream_generate(model, tokenizer, formatted, max_tokens=max_tokens):
        now = time.perf_counter()
        idx = len(per_token)

        # Snapshot telemetry delta for this token
        curr_hits = tel._pool_hits.copy()
        curr_misses = tel._pool_misses.copy()
        delta_hits = curr_hits - prev_hits
        delta_misses = curr_misses - prev_misses
        prev_hits = curr_hits
        prev_misses = curr_misses

        token_hits = int(delta_hits.sum())
        token_misses = int(delta_misses.sum())

        # Per-layer miss list for this token (only layers with misses)
        layer_misses = {}
        for li in range(num_layers):
            if delta_misses[li] > 0:
                layer_misses[li] = int(delta_misses[li])

        if not got_first:
            got_first = True
            dt_ms = (now - t0) * 1000
        else:
            dt_ms = (now - per_token[-1]["wall_time"]) * 1000

        per_token.append({
            "idx": idx,
            "wall_time": now,
            "dt_ms": dt_ms,
            "text_chunk": response.text,
            "tg_tps": response.generation_tps,
            "hits": token_hits,
            "misses": token_misses,
            "layer_misses": layer_misses,
        })

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024**3
    text = "".join(t["text_chunk"] for t in per_token)

    # Final per-layer miss rates
    final_hits = tel._pool_hits.copy()
    final_misses = tel._pool_misses.copy()
    per_layer_rates = {}
    for li in range(num_layers):
        h, m = int(final_hits[li]), int(final_misses[li])
        total_acts = h + m
        per_layer_rates[li] = {
            "hits": h, "misses": m, "total": total_acts,
            "miss_rate": m / total_acts if total_acts > 0 else 0.0,
        }

    result = {
        "label": f"dr+zero_{pool_size}",
        "text": text,
        "tokens": len(per_token),
        "total_ms": total * 1000,
        "peak_gb": peak,
        "pool_size": pool_size,
        "mem_after_compact_gb": mem_after,
        "recompact_stats": recompact_stats,
        "per_token": per_token,
        "per_layer_miss_rates": per_layer_rates,
    }

    ctx.close()
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    mx.metal.set_cache_limit(0)
    return result


def analyze_phases(per_token, label):
    """Split per-token data into phases and compute stats."""
    phases = [
        ("1-16", 0, 16),
        ("17-64", 16, 64),
        ("65+", 64, len(per_token)),
    ]

    print(f"\n  Phase Analysis [{label}]:")
    print(f"  {'Phase':<8} {'Tokens':>7} {'Mean ms':>9} {'P50 ms':>9} {'P95 ms':>9} "
          f"{'TG tok/s':>9} {'Misses':>7} {'Miss/tok':>9}")
    print(f"  {'-'*8} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*7} {'-'*9}")

    phase_results = []
    for name, start, end in phases:
        tokens = per_token[start:end]
        if not tokens:
            continue

        # Skip first token (includes PP time) for dt stats
        if start == 0 and len(tokens) > 1:
            dt_tokens = tokens[1:]
        else:
            dt_tokens = tokens

        if dt_tokens:
            dts = np.array([t["dt_ms"] for t in dt_tokens])
            mean_ms = float(dts.mean())
            p50_ms = float(np.percentile(dts, 50))
            p95_ms = float(np.percentile(dts, 95))
            tg_tps = 1000.0 / mean_ms if mean_ms > 0 else 0
        else:
            mean_ms = p50_ms = p95_ms = tg_tps = 0

        total_misses = sum(t.get("misses", 0) for t in tokens)
        miss_per_tok = total_misses / len(tokens) if tokens else 0

        print(f"  {name:<8} {len(tokens):>7} {mean_ms:>8.2f} {p50_ms:>8.2f} "
              f"{p95_ms:>8.2f} {tg_tps:>8.1f} {total_misses:>7} {miss_per_tok:>8.1f}")

        phase_results.append({
            "phase": name, "tokens": len(tokens),
            "mean_ms": mean_ms, "p50_ms": p50_ms, "p95_ms": p95_ms,
            "tg_tps": tg_tps, "total_misses": total_misses,
            "miss_per_tok": miss_per_tok,
        })

    return phase_results


def analyze_layer_misses(per_layer_rates):
    """Print per-layer miss rate heatmap."""
    print(f"\n  Per-Layer Miss Rates (top 10 worst + bottom 5 best):")
    print(f"  {'Layer':>6} {'Hits':>7} {'Misses':>7} {'MissRate':>9}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*9}")

    sorted_layers = sorted(per_layer_rates.items(),
                           key=lambda x: x[1]["miss_rate"], reverse=True)

    # Top 10 worst
    for li, stats in sorted_layers[:10]:
        print(f"  {li:>6} {stats['hits']:>7} {stats['misses']:>7} "
              f"{stats['miss_rate']:>8.1%}")

    if len(sorted_layers) > 15:
        print(f"  {'...':>6}")

    # Bottom 5 best
    for li, stats in sorted_layers[-5:]:
        print(f"  {li:>6} {stats['hits']:>7} {stats['misses']:>7} "
              f"{stats['miss_rate']:>8.1%}")

    # Summary stats
    miss_rates = [s["miss_rate"] for _, s in sorted_layers if s["total"] > 0]
    if miss_rates:
        arr = np.array(miss_rates)
        print(f"\n  Miss rate across {len(miss_rates)} layers: "
              f"mean={arr.mean():.1%}, std={arr.std():.1%}, "
              f"min={arr.min():.1%}, max={arr.max():.1%}")


def analyze_miss_position_histogram(per_token):
    """Show miss distribution across token positions."""
    print(f"\n  Miss Position Histogram:")

    miss_counts = [t.get("misses", 0) for t in per_token]
    if not any(miss_counts):
        print("  No misses recorded.")
        return

    # Bin into groups of 10
    bins = list(range(0, len(miss_counts), 10))
    print(f"  {'Tokens':<12} {'Misses':>7} {'Miss/tok':>9} {'Bar'}")
    print(f"  {'-'*12} {'-'*7} {'-'*9} {'-'*30}")

    max_misses_per_bin = 1
    bin_data = []
    for i in range(0, len(miss_counts), 10):
        chunk = miss_counts[i:i+10]
        total = sum(chunk)
        per_tok = total / len(chunk)
        bin_data.append((i, i + len(chunk), total, per_tok))
        if total > max_misses_per_bin:
            max_misses_per_bin = total

    for start, end, total, per_tok in bin_data:
        bar_len = int(30 * total / max_misses_per_bin) if max_misses_per_bin > 0 else 0
        bar = "#" * bar_len
        print(f"  {start:>4}-{end-1:<6} {total:>7} {per_tok:>8.1f} {bar}")


def analyze_miss_clustering(per_token):
    """Check if misses cluster or are evenly distributed."""
    miss_counts = [t.get("misses", 0) for t in per_token]
    if not any(miss_counts):
        return

    arr = np.array(miss_counts)
    zero_miss = int(np.sum(arr == 0))
    low_miss = int(np.sum((arr > 0) & (arr <= 16)))   # ≤2 per layer on average
    med_miss = int(np.sum((arr > 16) & (arr <= 40)))
    high_miss = int(np.sum(arr > 40))

    total = len(arr)
    print(f"\n  Miss Clustering:")
    print(f"    Zero-miss tokens:  {zero_miss:>4} ({zero_miss/total:.0%})")
    print(f"    Low-miss (1-16):   {low_miss:>4} ({low_miss/total:.0%})")
    print(f"    Med-miss (17-40):  {med_miss:>4} ({med_miss/total:.0%})")
    print(f"    High-miss (>40):   {high_miss:>4} ({high_miss/total:.0%})")

    # Check for consecutive high-miss streaks
    streak = 0
    max_streak = 0
    for m in miss_counts:
        if m > 16:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print(f"    Max consecutive high-miss streak: {max_streak}")


def main():
    parser = argparse.ArgumentParser(description="TEP Diagnostic Benchmark")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP Diagnostic — Phase-Split + Layer-Wise Miss Analysis")
    print(f"  Model: {args.model}")
    print(f"  Pool: {args.pool_size}/256  |  Tokens: {args.tokens}")
    print("=" * 70)

    prompt = (
        "Explain the mathematical foundations of attention mechanisms in "
        "transformers, including scaled dot-product attention and multi-head "
        "attention. Provide the full mathematical derivation."
    )

    # Run standard baseline
    print(f"\n{'=' * 60}")
    print(f"  [1/2] Standard Baseline")
    print(f"{'=' * 60}")
    std_result = run_standard(args.model, prompt, args.tokens)
    std_phases = analyze_phases(std_result["per_token"], "standard")
    std_result["phases"] = std_phases

    # Run diagnostic
    print(f"\n{'=' * 60}")
    print(f"  [2/2] Decode Recompact + zero_out (pool={args.pool_size})")
    print(f"{'=' * 60}")
    diag_result = run_diagnostic(args.model, prompt, args.tokens,
                                  pool_size=args.pool_size, warmup_tokens=20)
    diag_phases = analyze_phases(diag_result["per_token"], diag_result["label"])
    diag_result["phases"] = diag_phases

    # Layer-wise analysis
    analyze_layer_misses(diag_result["per_layer_miss_rates"])

    # Miss position histogram
    analyze_miss_position_histogram(diag_result["per_token"])

    # Miss clustering
    analyze_miss_clustering(diag_result["per_token"])

    # Side-by-side phase comparison
    print(f"\n{'=' * 70}")
    print(f"  PHASE COMPARISON: Standard vs dr+zero_{args.pool_size}")
    print(f"{'=' * 70}")
    print(f"  {'Phase':<8} {'Std tok/s':>10} {'Pool tok/s':>11} {'Gap':>7} {'Pool miss/tok':>13}")
    print(f"  {'-'*8} {'-'*10} {'-'*11} {'-'*7} {'-'*13}")

    for sp, dp in zip(std_phases, diag_phases):
        gap = (dp["tg_tps"] / sp["tg_tps"] - 1) * 100 if sp["tg_tps"] > 0 else 0
        print(f"  {sp['phase']:<8} {sp['tg_tps']:>9.1f} {dp['tg_tps']:>10.1f} "
              f"{gap:>+6.0f}% {dp['miss_per_tok']:>12.1f}")

    # Overall summary
    print(f"\n  Key Findings:")
    # Check if gap narrows in steady state
    if len(std_phases) >= 3 and len(diag_phases) >= 3:
        cold_gap = (diag_phases[0]["tg_tps"] / std_phases[0]["tg_tps"] - 1) * 100
        steady_gap = (diag_phases[2]["tg_tps"] / std_phases[2]["tg_tps"] - 1) * 100
        print(f"    Cold-start gap (1-16):  {cold_gap:+.0f}%")
        print(f"    Steady-state gap (65+): {steady_gap:+.0f}%")
        if abs(steady_gap) < abs(cold_gap):
            print(f"    → Gap narrows from cold to steady: {cold_gap:+.0f}% → {steady_gap:+.0f}%")
        else:
            print(f"    → Gap does NOT narrow: wrapper overhead is constant")

    # Text preview
    print(f"\n  Text Samples:")
    print(f"    [standard]: {std_result['text'][:200]}")
    print(f"    [{diag_result['label']}]: {diag_result['text'][:200]}")

    # Save full report
    # Strip wall_time from per_token for JSON serialization
    for t in std_result["per_token"]:
        t.pop("wall_time", None)
    for t in diag_result["per_token"]:
        t.pop("wall_time", None)

    report = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tokens": args.tokens,
        "pool_size": args.pool_size,
        "prompt": prompt,
        "standard": std_result,
        "diagnostic": diag_result,
    }

    out_path = args.output
    if not out_path:
        solar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".solar")
        os.makedirs(solar_dir, exist_ok=True)
        out_path = os.path.join(solar_dir, "tep-diagnostic.json")

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
