#!/usr/bin/env python3
"""
TEP Experiment B — Decode-Local Pool vs PP-Frequency Pool

Compares pool strategies:
  1. Standard (no offloading) — reference baseline
  2. PP-frequency pool — current production behavior (compact from PP data)
  3. Decode-local pool — recompact after W warmup TG tokens using TG activation data

Key question: Does a decode-trajectory-aware pool achieve higher TG hit rate
than a PP-frequency pool, and does that translate to fewer misses and better speed?

Usage:
    python3 benchmarks/bench_decode_pool.py [--model PATH] [--tokens N]
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
    pp_tps = 0
    token_times = []
    got_first = False

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
        pp_tps = response.prompt_tps

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024 / 1024 / 1024

    tpot_arr = np.array(token_times[3:]) if len(token_times) > 3 else np.array(token_times)

    result = {
        "text": text,
        "tokens": token_count,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "total_ms": total * 1000,
        "peak_gb": peak,
    }

    if len(tpot_arr) > 0:
        result["tpot"] = {
            "mean_ms": float(tpot_arr.mean()),
            "p50_ms": float(np.percentile(tpot_arr, 50)),
            "p95_ms": float(np.percentile(tpot_arr, 95)),
            "p99_ms": float(np.percentile(tpot_arr, 99)),
            "std_ms": float(tpot_arr.std()),
        }

    return result


def compare_texts(reference: str, candidate: str) -> dict:
    """Compare two generated texts."""
    max_len = max(len(reference), len(candidate))
    if max_len == 0:
        return {"match_ratio": 1.0, "first_diverge_char": -1}

    matches = sum(1 for a, b in zip(reference, candidate) if a == b)
    match_ratio = matches / max_len

    first_diverge = -1
    for i in range(min(len(reference), len(candidate))):
        if reference[i] != candidate[i]:
            first_diverge = i
            break
    if first_diverge == -1 and len(reference) != len(candidate):
        first_diverge = min(len(reference), len(candidate))

    return {
        "match_ratio": match_ratio,
        "first_diverge_char": first_diverge,
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


def compute_cross_val_hit_rate(switches) -> dict:
    """Compute cross-validation hit rate from TG buffers vs pool."""
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
        "total_activations": total,
        "misses": total_misses,
    }


def run_standard(model_path: str, prompt: str, max_tokens: int) -> dict:
    """Standard inference baseline."""
    print("\n  Loading model (standard)...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()
    print(f"  Model: {mx.metal.get_active_memory() / 1024**3:.2f} GB")

    result = generate_with_stats(model, tokenizer, prompt, max_tokens)
    result["label"] = "standard"
    result["pool_strategy"] = "none"

    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    return result


def run_pp_pool(model_path: str, prompt: str, max_tokens: int,
                pool_size: int, label: str) -> dict:
    """PP-frequency pool (current production behavior).

    Flow: dummy gen → PP buffer → compact from PP → force_miss_load for TG
    """
    from mlx_lm.models.expert_offload import patch_model_for_offload, FlashMoeSwitchGLU

    print(f"\n  Loading model (pp_pool, pool={pool_size})...")
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

    # Dummy generation to populate PP indices buffer
    dummy_msgs = [{"role": "user", "content": "Hello, what is 1+1?"}]
    dummy_fmt = tokenizer.apply_chat_template(
        dummy_msgs, add_generation_prompt=True, tokenize=False
    )
    for resp in stream_generate(model, tokenizer, dummy_fmt, max_tokens=5):
        pass
    print("  PP buffer populated")

    # Compact from PP data
    ctx.compact(pool_size=pool_size, disable_coverage_gate=True)
    gc.collect()

    # Enable force_miss_load for quality-preserving miss handling
    switches = _get_switch_layers(model)
    for sw in switches:
        sw._force_miss_load = True

    # Clear GPU cache
    mx.metal.set_cache_limit(0)
    gc.collect()
    mx.metal.clear_cache()
    mx.metal.set_cache_limit(int(64 * 1024**3))

    mem_after = mx.metal.get_active_memory() / 1024**3
    print(f"  After compact: {mem_after:.2f} GB, pool={pool_size}")

    # Real benchmark
    result = generate_with_stats(model, tokenizer, prompt, max_tokens)
    result["label"] = label
    result["pool_strategy"] = "pp_frequency"
    result["pool_size"] = pool_size
    result["mem_after_compact_gb"] = mem_after

    # Cross-val hit rate on the TG data
    cv = compute_cross_val_hit_rate(switches)
    result["cv_hit_rate"] = cv["hit_rate"]
    result["cv_misses"] = cv["misses"]
    result["cv_total"] = cv["total_activations"]

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


def run_decode_pool(model_path: str, prompt: str, max_tokens: int,
                    pool_size: int, warmup_tokens: int, label: str) -> dict:
    """Decode-local pool: recompact after W warmup TG tokens.

    Flow:
      1. Dummy gen → PP buffer → compact from PP
      2. Generate W warmup tokens (K-1 clamp — may be garbled, that's OK)
      3. decode_recompact() — rebuild pool from TG activation data
      4. force_miss_load for remaining tokens → measure quality + speed
    """
    from mlx_lm.models.expert_offload import patch_model_for_offload, FlashMoeSwitchGLU

    print(f"\n  Loading model (decode_pool, pool={pool_size}, warmup={warmup_tokens})...")
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

    # Dummy generation to populate PP indices buffer
    dummy_msgs = [{"role": "user", "content": "Hello, what is 1+1?"}]
    dummy_fmt = tokenizer.apply_chat_template(
        dummy_msgs, add_generation_prompt=True, tokenize=False
    )
    for resp in stream_generate(model, tokenizer, dummy_fmt, max_tokens=5):
        pass
    print("  PP buffer populated")

    # Initial compact from PP data
    ctx.compact(pool_size=pool_size, disable_coverage_gate=True)
    gc.collect()

    # Clear GPU cache
    mx.metal.set_cache_limit(0)
    gc.collect()
    mx.metal.clear_cache()
    mx.metal.set_cache_limit(int(64 * 1024**3))

    # Phase 1: Warmup — generate W tokens with K-1 clamping
    # (garbled output OK, we just need TG routing data)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    mx.random.seed(42)
    warmup_text = ""
    warmup_count = 0
    for response in stream_generate(model, tokenizer, formatted,
                                     max_tokens=warmup_tokens):
        warmup_text += response.text
        warmup_count += 1
    print(f"  Warmup: {warmup_count} tokens (K-1 clamp)")

    # Phase 2: Decode recompact — rebuild pool from TG data
    recompact_stats = ctx.decode_recompact(pool_size=pool_size)
    gc.collect()
    print(f"  Recompact: {recompact_stats['total_swapped']} swaps, "
          f"HR: {recompact_stats['avg_old_hit_rate']:.1%} → {recompact_stats['avg_new_hit_rate']:.1%}")

    # Phase 3: Enable force_miss_load and generate remaining tokens
    switches = _get_switch_layers(model)
    for sw in switches:
        sw._force_miss_load = True

    mem_after = mx.metal.get_active_memory() / 1024**3
    print(f"  After recompact: {mem_after:.2f} GB")

    # Generate remaining tokens with the decode-local pool
    remaining_tokens = max_tokens - warmup_count
    if remaining_tokens <= 0:
        remaining_tokens = max_tokens  # If warmup used all tokens, generate more

    result = generate_with_stats(model, tokenizer, prompt, remaining_tokens)
    result["label"] = label
    result["pool_strategy"] = "decode_local"
    result["pool_size"] = pool_size
    result["warmup_tokens"] = warmup_tokens
    result["mem_after_compact_gb"] = mem_after
    result["recompact_stats"] = recompact_stats

    # Cross-val hit rate
    cv = compute_cross_val_hit_rate(switches)
    result["cv_hit_rate"] = cv["hit_rate"]
    result["cv_misses"] = cv["misses"]
    result["cv_total"] = cv["total_activations"]

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
    parser = argparse.ArgumentParser(description="TEP Exp B — Decode-Local Pool")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit",
                        help="Path to MoE model")
    parser.add_argument("--tokens", type=int, default=150,
                        help="Max tokens to generate")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP Experiment B — Decode-Local Pool vs PP-Frequency Pool")
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
        ("pp_pool_64", {"pool_size": 64}),
        ("decode_w10_64", {"pool_size": 64, "warmup_tokens": 10}),
        ("decode_w20_64", {"pool_size": 64, "warmup_tokens": 20}),
        ("decode_w10_32", {"pool_size": 32, "warmup_tokens": 10}),
    ]

    for i, (label, cfg) in enumerate(configs):
        print(f"\n{'=' * 60}")
        print(f"  [{i+1}/{len(configs)}] {label}")
        print(f"{'=' * 60}")

        if label == "standard":
            r = run_standard(args.model, prompt, args.tokens)
        elif label.startswith("pp_pool"):
            r = run_pp_pool(args.model, prompt, args.tokens,
                           pool_size=cfg["pool_size"], label=label)
        else:
            r = run_decode_pool(args.model, prompt, args.tokens,
                               pool_size=cfg["pool_size"],
                               warmup_tokens=cfg["warmup_tokens"],
                               label=label)

        all_results.append(r)
        tpot = r.get("tpot", {})
        print(f"\n  -> TG: {r['tg_tps']:.1f} tok/s | Peak: {r['peak_gb']:.2f} GB")
        if tpot:
            print(f"     TPOT: p50={tpot['p50_ms']:.2f}ms, p95={tpot['p95_ms']:.2f}ms")
        print(f"     Text: {r['text'][:100]}...")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"  DECODE-LOCAL POOL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Config':<18} {'TG tok/s':>9} {'TPOT p50':>10} {'CV HR':>8} "
          f"{'Misses':>7} {'Peak GB':>8} {'Match%':>7}")
    print(f"  {'-'*18} {'-'*9} {'-'*10} {'-'*8} {'-'*7} {'-'*8} {'-'*7}")

    ref_text = all_results[0]["text"]
    baseline_tg = all_results[0]["tg_tps"]

    for r in all_results:
        tpot = r.get("tpot", {})
        cv_hr = r.get("cv_hit_rate")
        cv_str = f"{cv_hr:.1%}" if cv_hr is not None else "N/A"
        miss_n = r.get("miss_count", 0) + r.get("cv_misses", 0)
        comp = compare_texts(ref_text, r["text"]) if r["label"] != "standard" else {}
        match_pct = comp.get("match_ratio", 1.0) * 100

        print(f"  {r['label']:<18} {r['tg_tps']:>8.1f} "
              f"{tpot.get('p50_ms', 0):>9.2f}ms "
              f"{cv_str:>8} {miss_n:>7} "
              f"{r['peak_gb']:>7.2f} {match_pct:>6.1f}%")

    # Decode recompact analysis
    print(f"\n  Decode Recompact Analysis:")
    for r in all_results:
        rc = r.get("recompact_stats")
        if rc:
            print(f"    {r['label']}: HR {rc['avg_old_hit_rate']:.1%} → {rc['avg_new_hit_rate']:.1%} "
                  f"({rc['total_swapped']} swaps, {rc['elapsed_ms']:.0f}ms)")

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
        out_path = os.path.join(solar_dir, "tep-decode-pool.json")

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
