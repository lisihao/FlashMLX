#!/usr/bin/env python3
"""
TEP Phase A — Memory Pressure Benchmark

Measures expert offloading performance under memory-constrained conditions
to quantify the value of predictive expert prefetch.

Three regimes:
  1. Regime Comfort:  model << memory (baseline, no pressure)
  2. Regime Mild:     model ≈ memory (some expert eviction needed)
  3. Regime Severe:   model > memory (aggressive SSD swap)

Key metrics:
  - TG tok/s (steady-state)
  - TPOT distribution (p50/p95/p99)
  - Pool miss rate per layer
  - Miss penalty latency (CPU cache vs SSD breakdown)
  - Model residency vs peak memory

Usage:
    python3 benchmarks/bench_memory_pressure.py [--model PATH] [--tokens N]
    python3 benchmarks/bench_memory_pressure.py --pool-size 32  # force small pool
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


# ============================================================================
# Per-token timing
# ============================================================================

def measure_generation_detailed(model, tokenizer, prompt_text: str,
                                max_tokens: int = 200,
                                label: str = "") -> dict:
    """Measure generation with per-token TPOT tracking."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    mx.metal.reset_peak_memory()
    gc.collect()

    t0 = time.perf_counter()
    text = ""
    token_count = 0
    ttof = 0
    got_first = False
    last_response = None
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
            token_times.append(dt * 1000)  # ms
            prev_time = now
        text += response.text
        token_count += 1
        last_response = response

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024 / 1024 / 1024
    active = mx.metal.get_active_memory() / 1024 / 1024 / 1024

    tg_tps = last_response.generation_tps if last_response else 0
    pp_tps = last_response.prompt_tps if last_response else 0

    # TPOT distribution (skip first few tokens for warmup)
    tpot_arr = np.array(token_times[5:]) if len(token_times) > 5 else np.array(token_times)

    result = {
        "label": label,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "ttof_ms": ttof * 1000,
        "total_ms": total * 1000,
        "peak_gb": peak,
        "active_gb": active,
        "tokens": token_count,
        "text": text[:120],
    }

    if len(tpot_arr) > 0:
        result["tpot"] = {
            "mean_ms": float(tpot_arr.mean()),
            "p50_ms": float(np.percentile(tpot_arr, 50)),
            "p95_ms": float(np.percentile(tpot_arr, 95)),
            "p99_ms": float(np.percentile(tpot_arr, 99)),
            "max_ms": float(tpot_arr.max()),
            "std_ms": float(tpot_arr.std()),
            "count": len(tpot_arr),
        }

    return result


# ============================================================================
# Test regimes
# ============================================================================

def run_regime(model_path: str, prompt_text: str, max_tokens: int,
               pool_size: int, cpu_cache_gb: float,
               label: str) -> dict:
    """Run inference under specified memory constraints."""
    from mlx_lm.models.expert_offload import patch_model_for_offload

    print(f"\n  Loading model...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    mem_before = mx.metal.get_active_memory() / 1024 / 1024 / 1024
    print(f"  Model loaded: {mem_before:.2f} GB")

    ctx = patch_model_for_offload(
        model, model_path,
        pool_size=pool_size,
        max_workers=4,
        cpu_cache_gb=cpu_cache_gb,
        enable_prefetch=True,
        enable_telemetry=True,
    )
    gc.collect()
    mx.metal.clear_cache()

    mem_after = mx.metal.get_active_memory() / 1024 / 1024 / 1024
    print(f"  After patch: {mem_after:.2f} GB "
          f"(pool_size={pool_size}, cpu_cache={cpu_cache_gb:.1f} GB)")
    print(f"  Regime: {ctx.regime.regime}")
    print(f"  Param saved: {(mem_before - mem_after) * 1024:.0f} MB")

    # Run benchmark
    result = measure_generation_detailed(
        model, tokenizer, prompt_text,
        max_tokens=max_tokens, label=label,
    )
    result["pool_size"] = pool_size
    result["cpu_cache_gb"] = cpu_cache_gb
    result["mem_before_gb"] = mem_before
    result["mem_after_gb"] = mem_after
    result["param_saved_mb"] = (mem_before - mem_after) * 1024
    result["regime"] = ctx.regime.regime

    # Telemetry
    if ctx.telemetry:
        tel = ctx.telemetry.summary()
        result["telemetry"] = tel
        hit_rate = tel["overall_pool_hit_rate"]
        miss_lat = tel.get("miss_latency", {})
        print(f"  Pool hit rate: {hit_rate:.2%}")
        if miss_lat.get("count", 0) > 0:
            print(f"  Miss latency: mean={miss_lat['mean_ms']:.3f}ms, "
                  f"p95={miss_lat['p95_ms']:.3f}ms, "
                  f"sources={miss_lat['source_counts']}")

    if ctx.cpu_cache:
        cpu = ctx.cpu_cache.summary()
        result["cpu_cache"] = cpu
        print(f"  CPU cache: {cpu['entries']} entries, "
              f"hit rate: {cpu['hit_rate']:.2%}, "
              f"util: {cpu['utilization']:.0%}")

    ctx.close()
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()

    return result


def run_standard(model_path: str, prompt_text: str,
                 max_tokens: int, label: str) -> dict:
    """Run standard inference (no offloading) as baseline."""
    print(f"\n  Loading model (standard, no offload)...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    mem = mx.metal.get_active_memory() / 1024 / 1024 / 1024
    print(f"  Model loaded: {mem:.2f} GB")

    result = measure_generation_detailed(
        model, tokenizer, prompt_text,
        max_tokens=max_tokens, label=label,
    )
    result["pool_size"] = "N/A (standard)"
    result["mem_before_gb"] = mem
    result["regime"] = "standard"

    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TEP Phase A — Memory Pressure Benchmark")
    parser.add_argument("--model", default="/Volumes/toshiba/models/qwen3.5-35b-mlx",
                        help="Path to MoE model")
    parser.add_argument("--tokens", type=int, default=200,
                        help="Max tokens to generate")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="Override pool size (skip auto-detection)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP Phase A — Memory Pressure Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.tokens}")
    print("=" * 70)

    prompt = (
        "Explain the mathematical foundations of attention mechanisms in "
        "transformers, including the derivation of scaled dot-product attention, "
        "multi-head attention, and their computational complexity analysis. "
        "Then discuss how Mixture of Experts models route tokens to different "
        "expert networks, and compare the efficiency of dense vs sparse models "
        "for long-context inference on edge devices."
    )

    all_results = []

    if args.pool_size is not None:
        # Single run with specified pool size
        print(f"\n{'=' * 60}")
        print(f"  Custom pool_size={args.pool_size}")
        print(f"{'=' * 60}")
        r = run_regime(args.model, prompt, args.tokens,
                       pool_size=args.pool_size, cpu_cache_gb=2.0,
                       label=f"pool_{args.pool_size}")
        all_results.append(r)
        _print_result(r)
    else:
        # === Regime 1: Standard (no offloading) ===
        print(f"\n{'=' * 60}")
        print(f"  Regime: Standard (no offloading)")
        print(f"{'=' * 60}")
        r_std = run_standard(args.model, prompt, args.tokens, label="standard")
        all_results.append(r_std)
        _print_result(r_std)

        # === Regime 2: Comfort (auto pool, plenty of memory) ===
        print(f"\n{'=' * 60}")
        print(f"  Regime: Comfort (auto pool, full CPU cache)")
        print(f"{'=' * 60}")
        r_comfort = run_regime(args.model, prompt, args.tokens,
                               pool_size=0, cpu_cache_gb=4.0,
                               label="comfort")
        all_results.append(r_comfort)
        _print_result(r_comfort)

        # === Regime 3: Mild pressure (small pool, limited CPU cache) ===
        print(f"\n{'=' * 60}")
        print(f"  Regime: Mild (pool=48, cpu_cache=1.0 GB)")
        print(f"{'=' * 60}")
        r_mild = run_regime(args.model, prompt, args.tokens,
                            pool_size=48, cpu_cache_gb=1.0,
                            label="mild")
        all_results.append(r_mild)
        _print_result(r_mild)

        # === Regime 4: Severe (tiny pool, minimal CPU cache) ===
        print(f"\n{'=' * 60}")
        print(f"  Regime: Severe (pool=16, cpu_cache=0.5 GB)")
        print(f"{'=' * 60}")
        r_severe = run_regime(args.model, prompt, args.tokens,
                              pool_size=16, cpu_cache_gb=0.5,
                              label="severe")
        all_results.append(r_severe)
        _print_result(r_severe)

        # === Regime 5: Extreme (pool=8, no CPU cache) ===
        print(f"\n{'=' * 60}")
        print(f"  Regime: Extreme (pool=8, cpu_cache=0 GB)")
        print(f"{'=' * 60}")
        r_extreme = run_regime(args.model, prompt, args.tokens,
                               pool_size=8, cpu_cache_gb=0.0,
                               label="extreme")
        all_results.append(r_extreme)
        _print_result(r_extreme)

    # Summary comparison
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Regime':<12} {'TG tok/s':>10} {'TPOT p50':>10} {'TPOT p95':>10} "
          f"{'Peak GB':>8} {'Hit Rate':>9} {'Miss Lat':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*9} {'-'*10}")
    for r in all_results:
        tpot = r.get("tpot", {})
        tel = r.get("telemetry", {})
        miss_lat = tel.get("miss_latency", {}) if tel else {}
        hit_rate = tel.get("overall_pool_hit_rate", 1.0) if tel else "N/A"
        hit_str = f"{hit_rate:.2%}" if isinstance(hit_rate, float) else hit_rate
        miss_str = f"{miss_lat.get('mean_ms', 0):.3f}ms" if miss_lat.get("count", 0) > 0 else "N/A"
        print(f"  {r['label']:<12} {r['tg_tps']:>10.1f} "
              f"{tpot.get('p50_ms', 0):>9.2f}ms "
              f"{tpot.get('p95_ms', 0):>9.2f}ms "
              f"{r['peak_gb']:>7.2f} "
              f"{hit_str:>9} {miss_str:>10}")

    # Go/No-Go analysis
    if len(all_results) >= 4:
        baseline_tg = all_results[0]["tg_tps"]  # standard
        severe_tg = all_results[3]["tg_tps"]     # severe
        if baseline_tg > 0:
            degradation = 1.0 - severe_tg / baseline_tg
            print(f"\n  Severe vs Standard: {degradation:.1%} degradation")
            if degradation > 0.30:
                print(f"  → TEP JUSTIFIED: >{degradation:.0%} degradation under memory pressure")
                print(f"  → Predictive prefetch can recover significant performance")
            elif degradation > 0.10:
                print(f"  → TEP MARGINAL: {degradation:.0%} degradation, moderate value")
            else:
                print(f"  → TEP LOW VALUE: only {degradation:.0%} degradation, "
                      f"memory pressure is not the bottleneck")

    # Save results
    report = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tokens": args.tokens,
        "results": all_results,
    }

    out_path = args.output
    if not out_path:
        solar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".solar")
        os.makedirs(solar_dir, exist_ok=True)
        out_path = os.path.join(solar_dir, "tep-memory-pressure.json")

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")

    return report


def _print_result(r: dict):
    """Print a single regime result."""
    tpot = r.get("tpot", {})
    print(f"\n  Results: {r['label']}")
    print(f"    TG: {r['tg_tps']:.1f} tok/s | Peak: {r['peak_gb']:.2f} GB")
    if tpot:
        print(f"    TPOT: p50={tpot['p50_ms']:.2f}ms, "
              f"p95={tpot['p95_ms']:.2f}ms, "
              f"p99={tpot['p99_ms']:.2f}ms, "
              f"max={tpot['max_ms']:.2f}ms")
    print(f"    Text: {r['text'][:60]}")


if __name__ == "__main__":
    main()
