#!/usr/bin/env python3
"""
TEP P2: CPU Cache Warming / Prefetch Benchmark

Key hypothesis: pre-warming ALL non-pool experts into CPU cache (~6us/hit)
eliminates SSD latency (~240us/miss), making true_load viable at decent speed.

Experiment A showed: force_miss_load + SSD = 8.7 tok/s (87% degradation).
The bottleneck was 28,304 SSD misses × 0.35ms = 9.9s stall.
If misses hit CPU cache instead: 28,304 × 0.006ms = 0.17s — 58x faster.

But there's a second overhead: .tolist() GPU sync per layer per token = 40
syncs/token, which breaks lazy eval pipelining. This benchmark measures
both effects.

Configs:
  A. standard              — 6-bit, no offloading (baseline)
  B. dr+zero_32            — decode recompact, pool=32, zero_out (fast, divergent)
  C. dr+miss_32+cpu        — pool=32, CPU prewarm, force_miss_load (slow, correct)
  D. dr+miss_32+cpu+rr01   — same + reranking bonus=0.01 (fewer misses)
  E. dr+miss_64+cpu        — pool=64, CPU prewarm, force_miss_load (more hits)
  F. dr+zero_64+rr01       — pool=64, reranking, zero_out (speed+coverage)

Usage:
    python3 benchmarks/bench_prefetch.py [--model PATH] [--tokens N]
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
from mlx_lm.models.expert_offload import FlashMoeSwitchGLU, patch_model_for_offload


def _get_switch_layers(model):
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


def timed_generate(model, tokenizer, prompt, max_tokens, seed=42):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    mx.random.seed(seed)
    mx.metal.reset_peak_memory()

    token_times = []
    text_parts = []
    got_first = False
    tg_tps = 0

    for response in stream_generate(model, tokenizer, formatted, max_tokens=max_tokens):
        now = time.perf_counter()
        if not got_first:
            got_first = True
            prev = now
        else:
            token_times.append((now - prev) * 1000)
            prev = now
        text_parts.append(response.text)
        tg_tps = response.generation_tps

    peak = mx.metal.get_peak_memory() / 1024**3
    text = "".join(text_parts)
    arr = np.array(token_times[3:]) if len(token_times) > 3 else np.array(token_times)
    p50 = float(np.percentile(arr, 50)) if len(arr) > 0 else 0
    return text, tg_tps, peak, p50


def measure_hit_rate(switches):
    """Measure hit rate from TG indices buffer (deferred, single GPU sync)."""
    total_hits = 0
    total_count = 0
    for sw in switches:
        if not sw._tg_indices_buffer or not sw._pool_expert_ids:
            continue
        pool_set = set(sw._pool_expert_ids)
        all_tg = mx.concatenate(sw._tg_indices_buffer)
        ids = np.array(all_tg, copy=False)
        hits = int(np.count_nonzero(np.isin(ids, list(pool_set))))
        total_hits += hits
        total_count += len(ids)
    return total_hits / total_count if total_count > 0 else 0.0


def run_standard(model_path, prompt, max_tokens):
    """A: Standard 6-bit, no offloading."""
    print("\n  [A] standard 6-bit...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()
    text, tps, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)
    del model, tokenizer; gc.collect(); mx.metal.clear_cache()
    return {"label": "A_standard", "tg_tps": tps, "p50_ms": p50, "peak_gb": peak,
            "text": text[:400], "hit_rate": 1.0}


def run_pool_variant(model_path, prompt, max_tokens, label,
                     pool_size, miss_policy, force_miss_load=False,
                     rerank_bonus=0, cpu_cache_gb=0.0, warm_cache=False):
    """Generic config: pool + decode recompact + options."""
    opts = []
    if force_miss_load:
        opts.append("miss_load")
    if warm_cache:
        opts.append("cpu_warm")
    if rerank_bonus:
        opts.append(f"rr{rerank_bonus}")
    opts_str = "+".join(opts) if opts else "base"
    print(f"\n  [{label}] pool={pool_size}, miss={miss_policy}, {opts_str}...")

    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=cpu_cache_gb,
        enable_prefetch=False, enable_telemetry=False,
    )
    gc.collect()

    # Warmup: PP
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}], add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    # Compact (disable_coverage_gate for honest pool size,
    # disable auto_expand_cpu_cache to prevent UMA memory blowup)
    ctx.compact(pool_size=pool_size, disable_coverage_gate=True,
                auto_expand_cpu_cache=False)

    switches = _get_switch_layers(model)
    for sw in switches:
        sw._pool_is_identity = False
        sw._pool_compacted = True
        sw._miss_policy = "k1_clamp"

    # TG warmup for decode recompact
    warmup = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 20."}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, warmup, max_tokens=15):
        pass

    # Decode recompact
    ctx.decode_recompact(pool_size=pool_size)

    # CPU cache warming (P2 key feature)
    if warm_cache and cpu_cache_gb > 0:
        ctx.warm_cpu_cache()

    # Enable reranking AFTER recompact
    if rerank_bonus and rerank_bonus > 0:
        ctx.enable_reranking(bonus=rerank_bonus)

    # Set miss policy and configure
    cpu_hits_before = ctx.cpu_cache._hits if ctx.cpu_cache else 0
    cpu_misses_before = ctx.cpu_cache._misses if ctx.cpu_cache else 0

    for sw in switches:
        sw._miss_policy = miss_policy
        sw._force_miss_load = force_miss_load
        sw._tg_indices_buffer = []
        sw._tg_token_count = 0
        sw._disable_tg_buffer = False

    gc.collect(); mx.metal.clear_cache()

    text, tps, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)

    # Measure hit rate
    hr = measure_hit_rate(switches)

    # CPU cache stats
    cpu_hits = (ctx.cpu_cache._hits - cpu_hits_before) if ctx.cpu_cache else 0
    cpu_misses = (ctx.cpu_cache._misses - cpu_misses_before) if ctx.cpu_cache else 0
    cpu_hr = cpu_hits / (cpu_hits + cpu_misses) if (cpu_hits + cpu_misses) > 0 else 0

    ctx.close()
    del model, tokenizer; gc.collect(); mx.metal.clear_cache()

    return {"label": label, "tg_tps": tps, "p50_ms": p50, "peak_gb": peak,
            "text": text[:400], "hit_rate": hr, "pool_size": pool_size,
            "miss_policy": miss_policy, "force_miss_load": force_miss_load,
            "rerank_bonus": rerank_bonus, "cpu_cache_gb": cpu_cache_gb,
            "warm_cache": warm_cache, "cpu_hits": cpu_hits,
            "cpu_misses": cpu_misses, "cpu_hit_rate": cpu_hr}


def main():
    parser = argparse.ArgumentParser(description="TEP P2: CPU Cache Prefetch")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cpu-cache-gb", type=float, default=6.0)
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP P2: CPU Cache Warming / Prefetch Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.tokens}")
    print(f"  CPU Cache: {args.cpu_cache_gb} GB")
    print("=" * 70)

    prompt = (
        "Explain the mathematical foundations of attention mechanisms in "
        "transformers, including scaled dot-product attention and multi-head "
        "attention. Provide the full mathematical derivation."
    )

    results = []

    # A: Standard (run FIRST for clean memory baseline)
    results.append(run_standard(args.model, prompt, args.tokens))

    # B: dr+zero_32 — current best speed (no CPU cache)
    results.append(run_pool_variant(
        args.model, prompt, args.tokens,
        label="B_zero_32", pool_size=32, miss_policy="zero_out",
        cpu_cache_gb=0.0,
    ))

    # C: dr+miss_32+cpu — force_miss_load with CPU prewarm (THE KEY TEST)
    results.append(run_pool_variant(
        args.model, prompt, args.tokens,
        label="C_miss_32_cpu", pool_size=32, miss_policy="k1_clamp",
        force_miss_load=True, cpu_cache_gb=args.cpu_cache_gb,
        warm_cache=True,
    ))

    # D: dr+miss_32+cpu+rr01 — reranking reduces misses
    results.append(run_pool_variant(
        args.model, prompt, args.tokens,
        label="D_miss_rr01", pool_size=32, miss_policy="k1_clamp",
        force_miss_load=True, cpu_cache_gb=args.cpu_cache_gb,
        warm_cache=True, rerank_bonus=0.01,
    ))

    # E: dr+miss_64+cpu — larger pool, fewer misses
    results.append(run_pool_variant(
        args.model, prompt, args.tokens,
        label="E_miss_64_cpu", pool_size=64, miss_policy="k1_clamp",
        force_miss_load=True, cpu_cache_gb=args.cpu_cache_gb,
        warm_cache=True,
    ))

    # F: dr+zero_64+rr01 — larger pool + reranking with zero_out (speed test)
    results.append(run_pool_variant(
        args.model, prompt, args.tokens,
        label="F_zero_64_rr", pool_size=64, miss_policy="zero_out",
        cpu_cache_gb=0.0, rerank_bonus=0.01,
    ))

    # Summary
    baseline = results[0]["tg_tps"]
    print(f"\n{'=' * 70}")
    print(f"  PREFETCH RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Config':<14} {'TG tok/s':>9} {'vs Std':>7} {'P50 ms':>9} "
          f"{'Peak GB':>8} {'Pool HR':>8} {'CPU HR':>7}")
    print(f"  {'-'*14} {'-'*9} {'-'*7} {'-'*9} {'-'*8} {'-'*8} {'-'*7}")

    for r in results:
        delta = (r["tg_tps"] / baseline - 1) * 100 if baseline > 0 else 0
        cpu_hr_str = f"{r.get('cpu_hit_rate', 0):.0%}" if r.get("cpu_hits", 0) > 0 else "N/A"
        print(f"  {r['label']:<14} {r['tg_tps']:>8.1f} {delta:>+6.0f}% "
              f"{r['p50_ms']:>8.2f} {r['peak_gb']:>7.2f} "
              f"{r['hit_rate']:>7.1%} {cpu_hr_str:>7}")

    # Quality
    print(f"\n  QUALITY CHECK:")
    ref = results[0]["text"][:300]
    for r in results:
        if not r["text"]:
            print(f"    {r['label']}: EMPTY")
            continue
        common = sum(1 for a, b in zip(ref, r["text"][:300]) if a == b)
        match = common / max(len(ref), 1)
        status = "MATCH" if match > 0.90 else ("CLOSE" if match > 0.70 else "DIFFER")
        print(f"    {r['label']}: {match:.0%} → {status}")
        if status == "DIFFER" and r["label"] != "A_standard":
            print(f"      → {r['text'][:120]}")

    # CPU Cache Analysis
    print(f"\n  CPU CACHE ANALYSIS:")
    for r in results:
        if r.get("cpu_hits", 0) > 0 or r.get("cpu_misses", 0) > 0:
            print(f"    {r['label']}: {r['cpu_hits']} hits, {r['cpu_misses']} misses "
                  f"→ {r.get('cpu_hit_rate', 0):.1%} hit rate")

    # Key question
    print(f"\n  KEY ANALYSIS:")
    a_tps = results[0]["tg_tps"]
    for r in results:
        if r.get("force_miss_load"):
            speedup = r["tg_tps"] / 8.7  # vs Exp A SSD-only baseline
            print(f"    {r['label']}: {r['tg_tps']:.1f} tok/s "
                  f"({speedup:.1f}x vs SSD-only 8.7 tok/s, "
                  f"{r['tg_tps']/a_tps:.0%} of standard)")

    # Save
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".solar", "tep-prefetch.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "cpu_cache_gb": args.cpu_cache_gb,
                    "results": results}, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
