#!/usr/bin/env python3
"""
TEP Experiment B+C Combo — Decode Recompact + Miss Policy

Tests the winning combination:
  1. Decode recompact → boost hit rate to 90%+
  2. zero_out for remaining misses → fast, dropout-like
  3. Compare: decode_recompact+zero_out vs decode_recompact+true_load vs standard

Usage:
    python3 benchmarks/bench_combo.py [--model PATH] [--tokens N]
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


def compute_cv_hit_rate(switches):
    total_hits, total_misses = 0, 0
    for sw in switches:
        pool_set_np = np.array(sw._pool_expert_ids, dtype=np.int64)
        for buf in sw._tg_indices_buffer:
            ids = np.array(buf, copy=False).astype(np.int64)
            hits = np.isin(ids, pool_set_np).sum()
            total_hits += int(hits)
            total_misses += len(ids) - int(hits)
    total = total_hits + total_misses
    return {
        "hit_rate": total_hits / total if total > 0 else None,
        "total": total,
        "misses": total_misses,
    }


def run_standard(model_path, prompt, max_tokens):
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
    token_times = []
    got_first = False

    for response in stream_generate(model, tokenizer, formatted, max_tokens=max_tokens):
        now = time.perf_counter()
        if not got_first:
            got_first = True
            prev_time = now
        else:
            token_times.append((now - prev_time) * 1000)
            prev_time = now
        text += response.text
        token_count += 1
        tg_tps = response.generation_tps

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024**3

    tpot_arr = np.array(token_times[3:]) if len(token_times) > 3 else np.array(token_times)

    result = {
        "label": "standard",
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
        }

    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    return result


def run_combo(model_path, prompt, max_tokens, pool_size, warmup_tokens,
              miss_policy, label):
    """Decode recompact + miss policy.

    Flow:
      1. Load + patch + dummy gen → PP buffer → compact
      2. Warmup W tokens (force_miss_load=True for correct warmup)
      3. decode_recompact() → rebuild pool from TG data
      4. Set miss_policy and generate remaining tokens
    """
    from mlx_lm.models.expert_offload import patch_model_for_offload

    print(f"\n  Loading model (combo: pool={pool_size}, warmup={warmup_tokens}, policy={miss_policy})...")
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

    # Warmup: generate W tokens with force_miss_load (correct output, builds TG data)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    switches = _get_switch_layers(model)
    # Use force_miss_load during warmup for correct expert loading
    for sw in switches:
        sw._force_miss_load = True

    mx.random.seed(42)
    warmup_count = 0
    for response in stream_generate(model, tokenizer, formatted, max_tokens=warmup_tokens):
        warmup_count += 1
    print(f"  Warmup: {warmup_count} tokens (force_miss_load)")

    # Decode recompact
    recompact_stats = ctx.decode_recompact(pool_size=pool_size)
    gc.collect()
    print(f"  Recompact: {recompact_stats['total_swapped']} swaps, "
          f"HR: {recompact_stats['avg_old_hit_rate']:.1%} → {recompact_stats['avg_new_hit_rate']:.1%}")

    # Set miss policy for benchmark phase
    for sw in switches:
        sw._force_miss_load = (miss_policy in ("true_load",))
        sw._miss_policy = miss_policy if miss_policy in ("zero_out", "k1_clamp") else "k1_clamp"

    mem_after = mx.metal.get_active_memory() / 1024**3
    print(f"  After recompact: {mem_after:.2f} GB, policy={miss_policy}")

    # Benchmark generation (full prompt, fresh context)
    mx.random.seed(42)
    mx.metal.reset_peak_memory()
    t0 = time.perf_counter()
    text = ""
    token_count = 0
    tg_tps = 0
    token_times = []
    got_first = False

    for response in stream_generate(model, tokenizer, formatted, max_tokens=max_tokens):
        now = time.perf_counter()
        if not got_first:
            got_first = True
            prev_time = now
        else:
            token_times.append((now - prev_time) * 1000)
            prev_time = now
        text += response.text
        token_count += 1
        tg_tps = response.generation_tps

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024**3

    tpot_arr = np.array(token_times[3:]) if len(token_times) > 3 else np.array(token_times)

    # Cross-val hit rate on benchmark data
    cv = compute_cv_hit_rate(switches)

    result = {
        "label": label,
        "text": text,
        "tokens": token_count,
        "tg_tps": tg_tps,
        "total_ms": total * 1000,
        "peak_gb": peak,
        "pool_size": pool_size,
        "warmup_tokens": warmup_tokens,
        "miss_policy": miss_policy,
        "mem_after_compact_gb": mem_after,
        "recompact_stats": recompact_stats,
        "cv_hit_rate": cv["hit_rate"],
        "cv_misses": cv["misses"],
        "cv_total": cv["total"],
    }

    if len(tpot_arr) > 0:
        result["tpot"] = {
            "mean_ms": float(tpot_arr.mean()),
            "p50_ms": float(np.percentile(tpot_arr, 50)),
            "p95_ms": float(np.percentile(tpot_arr, 95)),
        }

    # Telemetry
    if ctx.telemetry:
        tel = ctx.telemetry.summary()
        miss_lat = tel.get("miss_latency", {})
        result["miss_count"] = miss_lat.get("count", 0)
        result["miss_mean_ms"] = miss_lat.get("mean_ms", 0)

    ctx.close()
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    mx.metal.set_cache_limit(0)
    return result


def main():
    parser = argparse.ArgumentParser(description="TEP B+C Combo")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP Experiment B+C — Decode Recompact + Miss Policy Combo")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.tokens}")
    print("=" * 70)

    prompt = (
        "Explain the mathematical foundations of attention mechanisms in "
        "transformers, including scaled dot-product attention and multi-head "
        "attention. Provide the full mathematical derivation."
    )

    all_results = []

    configs = [
        ("standard", {}),
        # Decode recompact + zero_out (the winning combo hypothesis)
        ("dr+zero_64", {"pool_size": 64, "warmup_tokens": 20, "miss_policy": "zero_out"}),
        ("dr+zero_32", {"pool_size": 32, "warmup_tokens": 20, "miss_policy": "zero_out"}),
        # Decode recompact + k1_clamp (compare zero_out improvement)
        ("dr+k1_64", {"pool_size": 64, "warmup_tokens": 20, "miss_policy": "k1_clamp"}),
        ("dr+k1_32", {"pool_size": 32, "warmup_tokens": 20, "miss_policy": "k1_clamp"}),
        # Decode recompact + true_load (quality ceiling, speed floor)
        ("dr+load_64", {"pool_size": 64, "warmup_tokens": 20, "miss_policy": "true_load"}),
    ]

    for i, (label, cfg) in enumerate(configs):
        print(f"\n{'=' * 60}")
        print(f"  [{i+1}/{len(configs)}] {label}")
        print(f"{'=' * 60}")

        if label == "standard":
            r = run_standard(args.model, prompt, args.tokens)
        else:
            r = run_combo(args.model, prompt, args.tokens, **cfg, label=label)

        all_results.append(r)
        tpot = r.get("tpot", {})
        cv = r.get("cv_hit_rate")
        cv_str = f"{cv:.1%}" if cv is not None else "N/A"
        print(f"\n  -> TG: {r['tg_tps']:.1f} tok/s | Peak: {r['peak_gb']:.2f} GB | CV HR: {cv_str}")
        if tpot:
            print(f"     TPOT: p50={tpot['p50_ms']:.2f}ms, p95={tpot['p95_ms']:.2f}ms")
        print(f"     Text: {r['text'][:120]}...")

    # Summary
    ref_text = all_results[0]["text"]
    baseline_tg = all_results[0]["tg_tps"]

    print(f"\n{'=' * 70}")
    print(f"  COMBO RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Config':<14} {'TG tok/s':>9} {'vs Std':>7} {'TPOT p50':>10} "
          f"{'CV HR':>7} {'Miss#':>6} {'Peak GB':>8} {'Coherent?':>10}")
    print(f"  {'-'*14} {'-'*9} {'-'*7} {'-'*10} {'-'*7} {'-'*6} {'-'*8} {'-'*10}")

    for r in all_results:
        tpot = r.get("tpot", {})
        cv = r.get("cv_hit_rate")
        cv_str = f"{cv:.0%}" if cv is not None else "N/A"
        delta = (r["tg_tps"] / baseline_tg - 1) * 100
        miss_n = r.get("miss_count", r.get("cv_misses", 0))

        # Quick coherence check: does text have repetitive patterns?
        text = r["text"]
        words = text.split()
        if len(words) > 10:
            # Check for 3+ consecutive repeated words
            repeats = 0
            for j in range(2, len(words)):
                if words[j] == words[j-1] == words[j-2]:
                    repeats += 1
            coherent = "YES" if repeats < 3 else "NO"
        else:
            coherent = "SHORT"

        print(f"  {r['label']:<14} {r['tg_tps']:>8.1f} {delta:>+6.0f}% "
              f"{tpot.get('p50_ms', 0):>9.2f}ms "
              f"{cv_str:>7} {miss_n:>6} {r['peak_gb']:>7.2f} {coherent:>10}")

    # Text samples
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
        out_path = os.path.join(solar_dir, "tep-combo.json")

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
