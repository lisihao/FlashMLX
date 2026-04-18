#!/usr/bin/env python3
"""
TEP Oracle — Hit-Only Wrapper Overhead + Layer-Aware Fix

Two critical experiments:

1. hit_only_256: Full pool (256 experts), force pool_is_identity=False.
   Zero misses, pure wrapper overhead measurement.
   If this is still 35-40% slower → bottleneck is kernel/wrapper, not cache.

2. layer_fix_32: Pool=32, worst 8 layers use force_miss_load (correct expert),
   remaining 32 layers use zero_out.
   Tests whether targeted layer fixes improve quality significantly.

Usage:
    python3 benchmarks/bench_oracle.py [--model PATH] [--tokens N]
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


def generate_timed(model, tokenizer, prompt, max_tokens, seed=42):
    """Generate with per-token timing. Returns (text, per_token_data, final_tps)."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    mx.random.seed(seed)
    mx.metal.reset_peak_memory()

    per_token = []
    t0 = time.perf_counter()
    got_first = False
    tg_tps = 0

    for response in stream_generate(model, tokenizer, formatted, max_tokens=max_tokens):
        now = time.perf_counter()
        idx = len(per_token)
        if not got_first:
            got_first = True
            dt_ms = (now - t0) * 1000
        else:
            dt_ms = (now - per_token[-1]["wall"]) * 1000
        per_token.append({"idx": idx, "wall": now, "dt_ms": dt_ms})
        tg_tps = response.generation_tps

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024**3
    text = ""
    # Re-generate for text (or use a separate capture)
    # Actually stream_generate yields text — let me fix this
    return per_token, tg_tps, total, peak


def generate_full(model, tokenizer, prompt, max_tokens, seed=42):
    """Generate with per-token timing and text capture."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    mx.random.seed(seed)
    mx.metal.reset_peak_memory()

    per_token = []
    text_parts = []
    t0 = time.perf_counter()
    got_first = False
    tg_tps = 0

    for response in stream_generate(model, tokenizer, formatted, max_tokens=max_tokens):
        now = time.perf_counter()
        idx = len(per_token)
        if not got_first:
            got_first = True
            dt_ms = (now - t0) * 1000
        else:
            dt_ms = (now - per_token[-1]["wall"]) * 1000
        per_token.append({"idx": idx, "wall": now, "dt_ms": dt_ms})
        text_parts.append(response.text)
        tg_tps = response.generation_tps

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024**3
    text = "".join(text_parts)
    return text, per_token, tg_tps, total, peak


def phase_stats(per_token):
    """Compute phase-split stats from per_token data."""
    phases = [("1-16", 0, 16), ("17-64", 16, 64), ("65+", 64, len(per_token))]
    results = []
    for name, start, end in phases:
        tokens = per_token[start:end]
        if not tokens:
            continue
        # Skip first token for dt stats (includes PP)
        dt_tokens = tokens[1:] if start == 0 and len(tokens) > 1 else tokens
        if dt_tokens:
            dts = np.array([t["dt_ms"] for t in dt_tokens])
            results.append({
                "phase": name, "tokens": len(tokens),
                "mean_ms": float(dts.mean()), "p50_ms": float(np.percentile(dts, 50)),
                "tg_tps": 1000.0 / float(dts.mean()),
            })
        else:
            results.append({"phase": name, "tokens": len(tokens),
                            "mean_ms": 0, "p50_ms": 0, "tg_tps": 0})
    return results


def run_standard(model_path, prompt, max_tokens):
    print("\n  Loading model (standard)...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    text, per_token, tg_tps, total, peak = generate_full(
        model, tokenizer, prompt, max_tokens)

    phases = phase_stats(per_token)
    # Clean per_token for JSON
    for t in per_token:
        t.pop("wall", None)

    result = {
        "label": "standard",
        "text": text,
        "tokens": len(per_token),
        "tg_tps": tg_tps,
        "total_ms": total * 1000,
        "peak_gb": peak,
        "phases": phases,
    }

    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    return result


def run_hit_only(model_path, prompt, max_tokens):
    """Full pool (256), force pool path (not identity). Pure wrapper overhead test."""
    from mlx_lm.models.expert_offload import patch_model_for_offload

    print("\n  Loading model (hit_only_256: full pool, forced remap path)...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, model_path,
        pool_size=256,  # full pool
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

    # Don't compact — keep full pool. But force pool_is_identity = False
    # to go through the remap + gather_qmm path.
    switches = _get_switch_layers(model)
    for sw in switches:
        sw._pool_is_identity = False
        sw._pool_compacted = True  # prevent PP buffer accumulation
        sw._miss_policy = "zero_out"  # shouldn't matter with 0 misses

    mx.metal.set_cache_limit(0)
    gc.collect()
    mx.metal.clear_cache()
    mx.metal.set_cache_limit(int(64 * 1024**3))

    mem = mx.metal.get_active_memory() / 1024**3
    print(f"  Pool: 256/256 experts, forced remap path, mem: {mem:.2f} GB")

    text, per_token, tg_tps, total, peak = generate_full(
        model, tokenizer, prompt, max_tokens)

    phases = phase_stats(per_token)
    for t in per_token:
        t.pop("wall", None)

    # Check telemetry for misses (should be 0)
    tel = ctx.telemetry
    total_misses = int(tel._pool_misses.sum())
    total_hits = int(tel._pool_hits.sum())

    result = {
        "label": "hit_only_256",
        "text": text,
        "tokens": len(per_token),
        "tg_tps": tg_tps,
        "total_ms": total * 1000,
        "peak_gb": peak,
        "mem_gb": mem,
        "phases": phases,
        "telemetry_hits": total_hits,
        "telemetry_misses": total_misses,
    }

    ctx.close()
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    mx.metal.set_cache_limit(0)
    return result


def run_layer_fix(model_path, prompt, max_tokens, pool_size, warmup_tokens,
                  fix_layers):
    """Pool=32, worst N layers use force_miss_load, rest use zero_out."""
    from mlx_lm.models.expert_offload import patch_model_for_offload

    fix_set = set(fix_layers)
    print(f"\n  Loading model (layer_fix: pool={pool_size}, "
          f"fix layers={sorted(fix_set)})...")
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

    # Dummy gen → compact
    dummy_msgs = [{"role": "user", "content": "Hello, what is 1+1?"}]
    dummy_fmt = tokenizer.apply_chat_template(
        dummy_msgs, add_generation_prompt=True, tokenize=False
    )
    for resp in stream_generate(model, tokenizer, dummy_fmt, max_tokens=5):
        pass

    ctx.compact(pool_size=pool_size, disable_coverage_gate=True)
    gc.collect()

    mx.metal.set_cache_limit(0)
    gc.collect()
    mx.metal.clear_cache()
    mx.metal.set_cache_limit(int(64 * 1024**3))

    # Warmup with force_miss_load
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    switches = _get_switch_layers(model)
    for sw in switches:
        sw._force_miss_load = True

    mx.random.seed(42)
    for response in stream_generate(model, tokenizer, formatted,
                                     max_tokens=warmup_tokens):
        pass

    # Decode recompact
    recompact_stats = ctx.decode_recompact(pool_size=pool_size)
    gc.collect()
    print(f"  Recompact: {recompact_stats['total_swapped']} swaps, "
          f"HR: {recompact_stats['avg_old_hit_rate']:.1%} → "
          f"{recompact_stats['avg_new_hit_rate']:.1%}")

    # Layer-aware policy: fix_layers get force_miss_load, rest get zero_out
    for sw in switches:
        if sw._layer_idx in fix_set:
            sw._force_miss_load = True
            sw._miss_policy = "k1_clamp"  # doesn't matter, force_miss_load overrides
        else:
            sw._force_miss_load = False
            sw._miss_policy = "zero_out"

    mem = mx.metal.get_active_memory() / 1024**3
    print(f"  After recompact: {mem:.2f} GB")
    print(f"  Policy: {len(fix_set)} layers force_miss_load, "
          f"{len(switches) - len(fix_set)} layers zero_out")

    text, per_token, tg_tps, total, peak = generate_full(
        model, tokenizer, prompt, max_tokens)

    phases = phase_stats(per_token)
    for t in per_token:
        t.pop("wall", None)

    result = {
        "label": f"layer_fix_{len(fix_set)}",
        "text": text,
        "tokens": len(per_token),
        "tg_tps": tg_tps,
        "total_ms": total * 1000,
        "peak_gb": peak,
        "pool_size": pool_size,
        "fix_layers": sorted(fix_set),
        "recompact_stats": recompact_stats,
        "phases": phases,
    }

    ctx.close()
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    mx.metal.set_cache_limit(0)
    return result


def main():
    parser = argparse.ArgumentParser(description="TEP Oracle Experiments")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP Oracle — Hit-Only Wrapper + Layer-Aware Fix")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.tokens}")
    print("=" * 70)

    prompt = (
        "Explain the mathematical foundations of attention mechanisms in "
        "transformers, including scaled dot-product attention and multi-head "
        "attention. Provide the full mathematical derivation."
    )

    all_results = []

    # --- Experiment 1: Standard baseline ---
    print(f"\n{'=' * 60}")
    print(f"  [1/4] Standard Baseline")
    print(f"{'=' * 60}")
    r_std = run_standard(args.model, prompt, args.tokens)
    all_results.append(r_std)

    # --- Experiment 2: Hit-only (full pool, forced remap) ---
    print(f"\n{'=' * 60}")
    print(f"  [2/4] Hit-Only 256 (pure wrapper overhead)")
    print(f"{'=' * 60}")
    r_hit = run_hit_only(args.model, prompt, args.tokens)
    all_results.append(r_hit)

    # --- Experiment 3: dr+zero_32 (reference for layer_fix) ---
    print(f"\n{'=' * 60}")
    print(f"  [3/4] dr+zero_32 (reference)")
    print(f"{'=' * 60}")
    from mlx_lm.models.expert_offload import patch_model_for_offload
    # Quick run of dr+zero_32 for comparison
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())
    gc.collect()
    ctx = patch_model_for_offload(model, args.model, pool_size=32,
                                   max_workers=4, cpu_cache_gb=0.0,
                                   enable_prefetch=False, enable_telemetry=True)
    gc.collect()
    dummy_msgs = [{"role": "user", "content": "Hello, what is 1+1?"}]
    dummy_fmt = tokenizer.apply_chat_template(dummy_msgs, add_generation_prompt=True, tokenize=False)
    for resp in stream_generate(model, tokenizer, dummy_fmt, max_tokens=5):
        pass
    ctx.compact(pool_size=32, disable_coverage_gate=True)
    gc.collect()
    mx.metal.set_cache_limit(0); gc.collect(); mx.metal.clear_cache()
    mx.metal.set_cache_limit(int(64 * 1024**3))

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    switches = _get_switch_layers(model)
    for sw in switches:
        sw._force_miss_load = True
    mx.random.seed(42)
    for resp in stream_generate(model, tokenizer, formatted, max_tokens=20):
        pass
    recompact_stats = ctx.decode_recompact(pool_size=32)
    gc.collect()
    print(f"  Recompact: HR {recompact_stats['avg_old_hit_rate']:.1%} → "
          f"{recompact_stats['avg_new_hit_rate']:.1%}")
    for sw in switches:
        sw._force_miss_load = False
        sw._miss_policy = "zero_out"
    text, per_token, tg_tps, total, peak = generate_full(
        model, tokenizer, prompt, args.tokens)
    phases = phase_stats(per_token)
    for t in per_token:
        t.pop("wall", None)
    r_zero = {
        "label": "dr+zero_32",
        "text": text, "tokens": len(per_token), "tg_tps": tg_tps,
        "total_ms": total * 1000, "peak_gb": peak,
        "phases": phases, "recompact_stats": recompact_stats,
    }
    all_results.append(r_zero)
    ctx.close()
    del model, tokenizer
    gc.collect(); mx.metal.clear_cache(); mx.metal.set_cache_limit(0)

    # --- Experiment 4: Layer-fix (worst 8 layers true_load, rest zero_out) ---
    # Worst layers from diagnostic: 0, 2, 32, 1, 33, 34, 24, 4
    print(f"\n{'=' * 60}")
    print(f"  [4/4] Layer-Fix (worst 8 layers → true_load)")
    print(f"{'=' * 60}")
    worst_layers = [0, 1, 2, 3, 4, 24, 32, 33]  # from diagnostic miss rates
    r_fix = run_layer_fix(args.model, prompt, args.tokens,
                           pool_size=32, warmup_tokens=20,
                           fix_layers=worst_layers)
    all_results.append(r_fix)

    # === Summary ===
    baseline_tg = r_std["tg_tps"]

    print(f"\n{'=' * 70}")
    print(f"  ORACLE EXPERIMENT RESULTS")
    print(f"{'=' * 70}")

    # Overall comparison
    print(f"\n  Overall:")
    print(f"  {'Config':<16} {'TG tok/s':>9} {'vs Std':>7} {'Peak GB':>8}")
    print(f"  {'-'*16} {'-'*9} {'-'*7} {'-'*8}")
    for r in all_results:
        delta = (r["tg_tps"] / baseline_tg - 1) * 100
        print(f"  {r['label']:<16} {r['tg_tps']:>8.1f} {delta:>+6.0f}% {r['peak_gb']:>7.2f}")

    # Phase comparison
    print(f"\n  Phase-Split TG tok/s:")
    print(f"  {'Config':<16} {'1-16':>8} {'17-64':>8} {'65+':>8}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        p = r.get("phases", [])
        vals = ["-"] * 3
        for i, ph in enumerate(p[:3]):
            vals[i] = f"{ph['tg_tps']:.1f}"
        print(f"  {r['label']:<16} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")

    # Hit-only verdict
    hit_gap = (r_hit["tg_tps"] / baseline_tg - 1) * 100
    print(f"\n  VERDICT:")
    print(f"    hit_only_256 vs standard: {hit_gap:+.0f}%")
    if abs(hit_gap) < 10:
        print(f"    → Wrapper overhead is SMALL (<10%). Bottleneck is miss handling.")
    elif abs(hit_gap) < 25:
        print(f"    → Wrapper overhead is MODERATE (10-25%). Both wrapper and miss matter.")
    else:
        print(f"    → Wrapper overhead is LARGE (>25%). Fix kernel/wrapper FIRST.")

    # Text quality comparison
    print(f"\n  Text Samples (first 250 chars):")
    for r in all_results:
        print(f"\n  [{r['label']}]:")
        print(f"  {r['text'][:250]}")

    # Save
    report = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tokens": args.tokens,
        "prompt": prompt,
        "results": all_results,
    }

    out_path = args.output
    if not out_path:
        solar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".solar")
        os.makedirs(solar_dir, exist_ok=True)
        out_path = os.path.join(solar_dir, "tep-oracle.json")

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
