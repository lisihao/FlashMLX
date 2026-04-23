#!/usr/bin/env python3
"""
TEP Phase A — Forced Memory Pressure Benchmark

Unlike bench_memory_pressure.py, this benchmark FORCES real SSD loading by:
  1. Building pool with only pool_size experts (not full prebuild)
  2. Clearing GPU cache after compact (evict freed weights from UMA)
  3. Routing pool misses to _pool_miss_call (SSD/CPU-cache, not K-1 clamp)
  4. Disabling CPU warm cache to force SSD reads

This simulates what happens on a 16GB device where non-pool experts
physically cannot stay in memory.

Usage:
    python3 benchmarks/bench_forced_pressure.py [--model PATH] [--tokens N]
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


def measure_generation(model, tokenizer, prompt_text: str,
                       max_tokens: int = 100, label: str = "") -> dict:
    """Measure generation with per-token timing."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    mx.metal.reset_peak_memory()
    gc.collect()

    t0 = time.perf_counter()
    text = ""
    token_count = 0
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
            token_times.append(dt * 1000)
            prev_time = now
        text += response.text
        token_count += 1
        last_response = response

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024 / 1024 / 1024

    tg_tps = last_response.generation_tps if last_response else 0
    pp_tps = last_response.prompt_tps if last_response else 0

    tpot_arr = np.array(token_times[3:]) if len(token_times) > 3 else np.array(token_times)

    result = {
        "label": label,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "ttof_ms": ttof * 1000 if got_first else 0,
        "total_ms": total * 1000,
        "peak_gb": peak,
        "tokens": token_count,
        "text": text,
        "text_preview": text[:120],
    }

    if len(tpot_arr) > 0:
        result["tpot"] = {
            "mean_ms": float(tpot_arr.mean()),
            "p50_ms": float(np.percentile(tpot_arr, 50)),
            "p95_ms": float(np.percentile(tpot_arr, 95)),
            "p99_ms": float(np.percentile(tpot_arr, 99)),
            "max_ms": float(np.percentile(tpot_arr, 100)),
            "std_ms": float(tpot_arr.std()),
        }

    return result


def run_standard(model_path: str, prompt: str, max_tokens: int) -> dict:
    """Standard inference baseline."""
    print("\n  Loading model (standard)...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()
    print(f"  Model: {mx.metal.get_active_memory() / 1024**3:.2f} GB")

    result = measure_generation(model, tokenizer, prompt, max_tokens, "standard")

    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    return result


def run_forced_pressure(model_path: str, prompt: str, max_tokens: int,
                        pool_size: int, cpu_cache_gb: float,
                        force_ssd: bool, label: str) -> dict:
    """Run with forced memory pressure.

    force_ssd=True: routes ALL TG misses through _pool_miss_call (SSD/CPU path)
    force_ssd=False: uses K-1 clamping (current production behavior, wrong expert)
    """
    from mlx_lm.models.expert_offload import patch_model_for_offload, FlashMoeSwitchGLU

    print(f"\n  Loading model (pool={pool_size}, force_ssd={force_ssd})...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, model_path,
        pool_size=pool_size,
        max_workers=4,
        cpu_cache_gb=cpu_cache_gb,
        enable_prefetch=False,  # No prefetch — measure raw miss cost
        enable_telemetry=True,
    )
    gc.collect()

    # Step 1: Dummy generation to populate PP indices buffer
    # (compact needs PP data to know which experts are hot)
    dummy_prompt = "Hello, answer briefly: what is 1+1?"
    dummy_msgs = [{"role": "user", "content": dummy_prompt}]
    dummy_fmt = tokenizer.apply_chat_template(
        dummy_msgs, add_generation_prompt=True, tokenize=False
    )
    for resp in stream_generate(model, tokenizer, dummy_fmt, max_tokens=5):
        pass
    print(f"  PP buffer populated via dummy generation")

    # Step 2: Compact to target pool_size (now has PP data)
    # disable_coverage_gate: test actual pool_size, not auto-expanded
    ctx.compact(pool_size=pool_size, disable_coverage_gate=True)
    gc.collect()

    # Step 3: Verify compact actually worked
    inner = model
    for attr in ("model", "model", "language_model", "model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    sw0 = None
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if isinstance(sw, FlashMoeSwitchGLU):
                if sw0 is None:
                    sw0 = sw
    if sw0:
        print(f"  Pool: {len(sw0._pool_expert_ids)}/256 experts, identity={sw0._pool_is_identity}")

    # Step 4: Clear GPU cache to evict freed expert weights
    mx.metal.set_cache_limit(0)
    gc.collect()
    mx.metal.clear_cache()

    mem_after = mx.metal.get_active_memory() / 1024**3
    print(f"  After compact+clear: {mem_after:.2f} GB")

    # Step 5: Enable force_miss_load for real SSD loading on misses
    if force_ssd:
        for layer in inner.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
                sw = layer.mlp.switch_mlp
                if isinstance(sw, FlashMoeSwitchGLU):
                    sw._force_miss_load = True

    # Restore cache limit for generation
    mx.metal.set_cache_limit(int(64 * 1024**3))

    # Step 6: Real benchmark generation
    result = measure_generation(model, tokenizer, prompt, max_tokens, label)
    result["pool_size"] = pool_size
    result["force_ssd"] = force_ssd
    result["mem_after_compact_gb"] = mem_after

    # Telemetry
    if ctx.telemetry:
        tel = ctx.telemetry.summary()
        result["telemetry"] = tel
        hit_rate = tel.get("overall_pool_hit_rate", 0)
        miss_lat = tel.get("miss_latency", {})
        print(f"  Pool hit rate: {hit_rate:.2%}")
        if miss_lat.get("count", 0) > 0:
            print(f"  Miss count: {miss_lat['count']}")
            print(f"  Miss latency: mean={miss_lat['mean_ms']:.3f}ms, "
                  f"p95={miss_lat['p95_ms']:.3f}ms, "
                  f"max={miss_lat['max_ms']:.3f}ms")
            print(f"  Miss sources: {miss_lat['source_counts']}")

    ctx.close()
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    mx.metal.set_cache_limit(0)  # Reset

    return result


def main():
    parser = argparse.ArgumentParser(description="TEP — Forced Memory Pressure Benchmark")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit",
                        help="Path to MoE model")
    parser.add_argument("--tokens", type=int, default=100,
                        help="Max tokens to generate")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP — Forced Memory Pressure Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.tokens}")
    print("=" * 70)

    prompt = (
        "Explain the mathematical foundations of attention mechanisms in "
        "transformers, including scaled dot-product attention and multi-head "
        "attention. Provide the full mathematical derivation."
    )

    all_results = []
    pool_sizes = [64, 32, 16]
    total_runs = 1 + len(pool_sizes)
    run_idx = 0

    # 1. Standard baseline
    run_idx += 1
    print(f"\n{'='*60}")
    print(f"  [{run_idx}/{total_runs}] Standard (no offloading)")
    print(f"{'='*60}")
    r = run_standard(args.model, prompt, args.tokens)
    all_results.append(r)
    _print(r)

    # 2-4. True miss load at each pool size (no CPU cache → pure SSD cost)
    for ps in pool_sizes:
        run_idx += 1
        print(f"\n{'='*60}")
        print(f"  [{run_idx}/{total_runs}] Pool={ps}/256, force_miss_load=True (SSD)")
        print(f"{'='*60}")
        r = run_forced_pressure(args.model, prompt, args.tokens,
                                pool_size=ps, cpu_cache_gb=0.0,
                                force_ssd=True, label=f"pool{ps}_miss_load")
        all_results.append(r)
        _print(r)

    # Summary
    print(f"\n{'='*70}")
    print(f"  FORCED PRESSURE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<20} {'TG tok/s':>9} {'TPOT p50':>10} "
          f"{'TPOT p95':>10} {'Peak GB':>8} {'Hit%':>6} "
          f"{'Misses':>7} {'Miss ms':>8}")
    print(f"  {'-'*20} {'-'*9} {'-'*10} {'-'*10} {'-'*8} "
          f"{'-'*6} {'-'*7} {'-'*8}")

    baseline_tg = all_results[0]["tg_tps"]
    for r in all_results:
        tpot = r.get("tpot", {})
        tel = r.get("telemetry", {})
        miss_lat = tel.get("miss_latency", {}) if tel else {}
        hit = tel.get("overall_pool_hit_rate", 1.0) if tel else "N/A"
        hit_str = f"{hit:.0%}" if isinstance(hit, float) else "N/A"
        miss_n = miss_lat.get("count", 0)
        miss_ms = f"{miss_lat.get('mean_ms', 0):.2f}" if miss_n > 0 else "N/A"
        delta = (r["tg_tps"] / baseline_tg - 1) * 100 if baseline_tg > 0 else 0

        print(f"  {r['label']:<20} {r['tg_tps']:>8.1f} "
              f"{tpot.get('p50_ms', 0):>9.2f}ms "
              f"{tpot.get('p95_ms', 0):>9.2f}ms "
              f"{r['peak_gb']:>7.2f} "
              f"{hit_str:>6} {miss_n:>7} {miss_ms:>8}")

    # Quality comparison vs standard
    ref_text = all_results[0]["text"]
    print(f"\n  Quality vs Standard:")
    for r in all_results[1:]:
        cand = r["text"]
        min_len = min(len(ref_text), len(cand))
        max_len = max(len(ref_text), len(cand))
        if max_len > 0:
            matches = sum(1 for a, b in zip(ref_text, cand) if a == b)
            match_pct = matches / max_len * 100
        else:
            match_pct = 100.0
        print(f"    {r['label']:<20} match={match_pct:.1f}%")

    # Go/No-Go per pool size
    print(f"\n  Go/No-Go Analysis:")
    for r in all_results[1:]:
        degradation = 1.0 - r["tg_tps"] / baseline_tg if baseline_tg > 0 else 0
        tel = r.get("telemetry", {})
        miss_lat = tel.get("miss_latency", {}) if tel else {}
        miss_n = miss_lat.get("count", 0)
        label = r["label"]
        print(f"\n    {label}: {degradation:.0%} TG degradation ({r['tg_tps']:.1f} vs {baseline_tg:.1f})")
        if miss_n > 0:
            print(f"      {miss_n} misses × {miss_lat.get('mean_ms', 0):.3f}ms = "
                  f"{miss_n * miss_lat.get('mean_ms', 0) / 1000:.1f}s total stall")
            sources = miss_lat.get("source_counts", {})
            print(f"      Sources: {sources}")

    # Save
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
        out_path = os.path.join(solar_dir, "tep-forced-pressure.json")

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


def _print(r: dict):
    tpot = r.get("tpot", {})
    print(f"\n  → TG: {r['tg_tps']:.1f} tok/s | Peak: {r['peak_gb']:.2f} GB")
    if tpot:
        print(f"    TPOT: p50={tpot['p50_ms']:.2f}ms, p95={tpot['p95_ms']:.2f}ms")
    print(f"    Text: {r.get('text_preview', r['text'][:60])}...")


if __name__ == "__main__":
    main()
