#!/usr/bin/env python3
"""
TEP P3: Two-tier Pool + Shadow Benchmark

Tests the combination of pool (6-bit hot experts) + shadow (4-bit all experts)
for miss fallback. Unlike zero_out (which drops missed expert output), shadow
miss provides a lower-precision but CORRECT expert output — all within the
MLX lazy graph, zero GPU sync.

Key hypothesis from P0-P2:
  - P0: 4-bit full (256 experts) = 73.4 tok/s, 18.8 GB, 77% quality
  - P2: GPU sync (.tolist()) is the speed bottleneck for true_load
  - Shadow miss avoids GPU sync (pure MLX ops), so should preserve speed

Configs:
  A. standard              — 6-bit, no offloading (baseline)
  B. dr+shadow_32          — pool=32 + shadow@4bit, shadow miss policy
  C. dr+shadow_32+rr01     — same + reranking bonus=0.01
  D. dr+shadow_64+rr01     — pool=64 + shadow@4bit + reranking
  E. dr+zero_32            — pool=32 + zero_out (speed comparison)
  F. dr+zero_64+rr01       — pool=64 + zero_out + reranking

Usage:
    python3 benchmarks/bench_twotier.py [--model PATH] [--tokens N]
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
    print("\n  [A] standard 6-bit...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()
    text, tps, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)
    del model, tokenizer; gc.collect(); mx.metal.clear_cache()
    return {"label": "A_standard", "tg_tps": tps, "p50_ms": p50, "peak_gb": peak,
            "text": text[:400], "hit_rate": 1.0}


def run_twotier(model_path, prompt, max_tokens, label,
                pool_size, miss_policy, shadow_bits=None,
                rerank_bonus=0, tier1_size=0):
    """Run a two-tier pool + optional shadow config."""
    opts = [f"miss={miss_policy}"]
    if shadow_bits:
        opts.append(f"shd{shadow_bits}bit")
    if rerank_bonus:
        opts.append(f"rr{rerank_bonus}")
    if tier1_size:
        opts.append(f"t1={tier1_size}")
    print(f"\n  [{label}] pool={pool_size}, {'+'.join(opts)}...")

    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=False,
    )
    gc.collect()

    # PP warmup
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}], add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    # Compact
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

    # Create shadow AFTER recompact (needs full expert access from SSD)
    shadow_gb = 0
    if shadow_bits and miss_policy == "shadow":
        info = ctx.create_shadow(bits=shadow_bits)
        shadow_gb = info.get("shadow_gb", 0)

    # Enable reranking AFTER recompact
    if rerank_bonus and rerank_bonus > 0:
        ctx.enable_reranking(bonus=rerank_bonus)

    # Set Tier 1 protection
    if tier1_size > 0:
        ctx.set_tier1(tier1_size)

    # Set miss policy and clear TG buffers
    for sw in switches:
        sw._miss_policy = miss_policy
        sw._tg_indices_buffer = []
        sw._tg_token_count = 0
        sw._disable_tg_buffer = False

    gc.collect(); mx.metal.clear_cache()

    text, tps, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)

    hr = measure_hit_rate(switches)

    ctx.close()
    del model, tokenizer; gc.collect(); mx.metal.clear_cache()

    return {"label": label, "tg_tps": tps, "p50_ms": p50, "peak_gb": peak,
            "text": text[:400], "hit_rate": hr, "pool_size": pool_size,
            "miss_policy": miss_policy, "shadow_bits": shadow_bits,
            "shadow_gb": shadow_gb, "rerank_bonus": rerank_bonus,
            "tier1_size": tier1_size}


def main():
    parser = argparse.ArgumentParser(description="TEP P3: Two-tier Pool + Shadow")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP P3: Two-tier Pool + Shadow Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.tokens}")
    print("=" * 70)

    prompt = (
        "Explain the mathematical foundations of attention mechanisms in "
        "transformers, including scaled dot-product attention and multi-head "
        "attention. Provide the full mathematical derivation."
    )

    results = []

    # A: Standard (run FIRST for clean baseline)
    results.append(run_standard(args.model, prompt, args.tokens))

    # B: pool=32 + shadow@4bit (key test: shadow vs zero_out quality)
    results.append(run_twotier(
        args.model, prompt, args.tokens,
        label="B_shd_32", pool_size=32, miss_policy="shadow",
        shadow_bits=4,
    ))

    # C: pool=32 + shadow@4bit + reranking (fewer misses = less shadow overhead)
    results.append(run_twotier(
        args.model, prompt, args.tokens,
        label="C_shd32_rr", pool_size=32, miss_policy="shadow",
        shadow_bits=4, rerank_bonus=0.01,
    ))

    # D: pool=64 + shadow@4bit + reranking (large pool + shadow safety)
    results.append(run_twotier(
        args.model, prompt, args.tokens,
        label="D_shd64_rr", pool_size=64, miss_policy="shadow",
        shadow_bits=4, rerank_bonus=0.01,
    ))

    # E: pool=32 + zero_out (speed comparison, no shadow)
    results.append(run_twotier(
        args.model, prompt, args.tokens,
        label="E_zero_32", pool_size=32, miss_policy="zero_out",
    ))

    # F: pool=64 + zero_out + reranking (speed comparison)
    results.append(run_twotier(
        args.model, prompt, args.tokens,
        label="F_zero64rr", pool_size=64, miss_policy="zero_out",
        rerank_bonus=0.01,
    ))

    # Summary
    baseline = results[0]["tg_tps"]
    print(f"\n{'=' * 70}")
    print(f"  TWO-TIER POOL + SHADOW RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Config':<12} {'TG tok/s':>9} {'vs Std':>7} {'P50 ms':>9} "
          f"{'Peak GB':>8} {'Hit Rate':>9} {'Shadow':>7}")
    print(f"  {'-'*12} {'-'*9} {'-'*7} {'-'*9} {'-'*8} {'-'*9} {'-'*7}")

    for r in results:
        delta = (r["tg_tps"] / baseline - 1) * 100 if baseline > 0 else 0
        shd = f"{r.get('shadow_gb', 0):.1f}GB" if r.get("shadow_bits") else "none"
        print(f"  {r['label']:<12} {r['tg_tps']:>8.1f} {delta:>+6.0f}% "
              f"{r['p50_ms']:>8.2f} {r['peak_gb']:>7.2f} "
              f"{r['hit_rate']:>8.1%} {shd:>7}")

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

    # Shadow vs Zero-out comparison
    print(f"\n  SHADOW vs ZERO-OUT ANALYSIS:")
    shadow_configs = [r for r in results if r.get("shadow_bits")]
    zero_configs = [r for r in results if r.get("miss_policy") == "zero_out"]

    for sr in shadow_configs:
        ps = sr["pool_size"]
        rr = sr.get("rerank_bonus", 0)
        zr = next((z for z in zero_configs
                    if z["pool_size"] == ps and z.get("rerank_bonus", 0) == rr), None)
        if zr:
            speed_delta = (sr["tg_tps"] / zr["tg_tps"] - 1) * 100
            mem_delta = sr["peak_gb"] - zr["peak_gb"]
            print(f"    pool={ps}: shadow {sr['tg_tps']:.1f} vs zero {zr['tg_tps']:.1f} "
                  f"({speed_delta:+.0f}% speed, {mem_delta:+.1f} GB mem)")

    # Save
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".solar", "tep-twotier.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "results": results}, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
