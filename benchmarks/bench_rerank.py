#!/usr/bin/env python3
"""
TEP P1: Cache-aware Reranking Benchmark

Tests whether biasing gate scores toward in-pool experts improves hit rate
and quality without hurting speed.

Configs:
  A. standard              — no offloading (baseline)
  B. dr+zero_32            — decode recompact, pool=32, zero_out, NO rerank
  C. dr+zero_32+rr001      — same + reranking bonus=0.001
  D. dr+zero_32+rr005      — same + reranking bonus=0.005
  E. dr+zero_32+rr01       — same + reranking bonus=0.01
  F. dr+zero_32+rr05       — same + reranking bonus=0.05
  G. dr+k1_32+rr01         — k1_clamp + reranking (test if higher HR saves k1)

Usage:
    python3 benchmarks/bench_rerank.py [--model PATH] [--tokens N]
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
    t0 = time.perf_counter()
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
    print("\n  [A] standard...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()
    text, tps, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)
    del model, tokenizer; gc.collect(); mx.metal.clear_cache()
    return {"label": "A_standard", "tg_tps": tps, "p50_ms": p50, "peak_gb": peak,
            "text": text[:400], "hit_rate": 1.0}


def run_rerank_variant(model_path, prompt, max_tokens, label,
                       pool_size, miss_policy, rerank_bonus):
    bonus_str = f"rr{rerank_bonus}" if rerank_bonus else "no_rr"
    print(f"\n  [{label}] pool={pool_size}, miss={miss_policy}, {bonus_str}...")

    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=False,
    )
    gc.collect()

    # Warmup: PP
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}], add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    # Compact
    ctx.compact(pool_size=pool_size)

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

    # Enable reranking AFTER recompact
    if rerank_bonus and rerank_bonus > 0:
        ctx.enable_reranking(bonus=rerank_bonus)

    # Set miss policy and clear TG buffers for clean measurement
    for sw in switches:
        sw._miss_policy = miss_policy
        sw._tg_indices_buffer = []
        sw._tg_token_count = 0
        sw._disable_tg_buffer = False  # ensure TG buffer is ON for HR measurement

    gc.collect(); mx.metal.clear_cache()

    text, tps, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)

    # Measure hit rate
    hr = measure_hit_rate(switches)

    ctx.close()
    del model, tokenizer; gc.collect(); mx.metal.clear_cache()

    return {"label": label, "tg_tps": tps, "p50_ms": p50, "peak_gb": peak,
            "text": text[:400], "hit_rate": hr, "rerank_bonus": rerank_bonus,
            "pool_size": pool_size, "miss_policy": miss_policy}


def main():
    parser = argparse.ArgumentParser(description="TEP P1: Cache-aware Reranking")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP P1: Cache-aware Reranking Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.tokens}")
    print("=" * 70)

    prompt = (
        "Explain the mathematical foundations of attention mechanisms in "
        "transformers, including scaled dot-product attention and multi-head "
        "attention. Provide the full mathematical derivation."
    )

    results = []

    # A: Standard
    results.append(run_standard(args.model, prompt, args.tokens))

    # B: No reranking (baseline for comparison)
    results.append(run_rerank_variant(
        args.model, prompt, args.tokens,
        label="B_no_rr", pool_size=32, miss_policy="zero_out",
        rerank_bonus=0,
    ))

    # C-F: Different bonus levels
    for bonus, lbl in [(0.001, "C_rr001"), (0.005, "D_rr005"),
                        (0.01, "E_rr01"), (0.05, "F_rr05")]:
        results.append(run_rerank_variant(
            args.model, prompt, args.tokens,
            label=lbl, pool_size=32, miss_policy="zero_out",
            rerank_bonus=bonus,
        ))

    # G: k1_clamp + reranking (can higher HR save k1?)
    results.append(run_rerank_variant(
        args.model, prompt, args.tokens,
        label="G_k1_rr01", pool_size=32, miss_policy="k1_clamp",
        rerank_bonus=0.01,
    ))

    # Summary
    baseline = results[0]["tg_tps"]
    print(f"\n{'=' * 70}")
    print(f"  RERANKING RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Config':<12} {'TG tok/s':>9} {'vs Std':>7} {'P50 ms':>9} {'Peak GB':>8} {'Hit Rate':>9}")
    print(f"  {'-'*12} {'-'*9} {'-'*7} {'-'*9} {'-'*8} {'-'*9}")

    for r in results:
        delta = (r["tg_tps"] / baseline - 1) * 100 if baseline > 0 else 0
        print(f"  {r['label']:<12} {r['tg_tps']:>8.1f} {delta:>+6.0f}% "
              f"{r['p50_ms']:>8.2f} {r['peak_gb']:>7.2f} {r['hit_rate']:>8.1%}")

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
        if status == "DIFFER" and r["label"] not in ("A_standard",):
            print(f"      → {r['text'][:120]}")

    # Analysis
    print(f"\n  ANALYSIS:")
    b_hr = results[1]["hit_rate"]
    for r in results[2:]:
        if "rr" in r["label"] and r.get("rerank_bonus"):
            hr_delta = r["hit_rate"] - b_hr
            print(f"    bonus={r['rerank_bonus']}: HR {b_hr:.1%} → {r['hit_rate']:.1%} "
                  f"({hr_delta:+.1%}), speed {r['tg_tps']:.1f} tok/s")

    # Save
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".solar", "tep-rerank.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "results": results}, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
