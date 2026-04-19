#!/usr/bin/env python3
"""
FTEC: Fidelity-first TEP Execution Contract Benchmark

Tests the FTEC architecture:
  - Guarded rerank (protect top-J, only bias tail when margin < tau)
  - Decode-hot shadow pool (64 experts at 4-bit, not full 256)
  - Three-way miss dispatch: pool hit → full, shadow hit → same-expert 4-bit, both miss → zero

Configs:
  A. standard              — 6-bit, no offloading (baseline)
  B. dr+zero_32+rr01       — pool=32 + zero_out + legacy rerank (current best speed)
  C. dr+zero_32+guard_rr   — pool=32 + zero_out + guarded rerank (rerank only)
  D. dr+ftec_32_shd64      — pool=32 + guarded rerank + decode_shadow_64 + ftec (MAIN)
  E. dr+zero_64+rr01       — pool=64 + zero_out + legacy rerank (P2 reference)

Usage:
    python3 benchmarks/bench_ftec.py [--model PATH] [--tokens N]
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


def measure_mass_metrics(switches):
    """Flush telemetry and collect FTEC mass metrics."""
    for sw in switches:
        sw.flush_tg_telemetry()

    total_exact = 0.0
    total_shadow = 0.0
    total_zeroed = 0.0
    layers_with_mass = 0

    for sw in switches:
        if sw._telemetry:
            em = sw._telemetry._exact_mass[sw._layer_idx]
            sm = sw._telemetry._shadow_mass[sw._layer_idx]
            zm = sw._telemetry._zeroed_mass[sw._layer_idx]
            total = em + sm + zm
            if total > 0:
                total_exact += em
                total_shadow += sm
                total_zeroed += zm
                layers_with_mass += 1

    grand_total = total_exact + total_shadow + total_zeroed
    if grand_total > 0:
        return {
            "exact_pct": total_exact / grand_total * 100,
            "shadow_pct": total_shadow / grand_total * 100,
            "zeroed_pct": total_zeroed / grand_total * 100,
            "layers": layers_with_mass,
        }
    return {"exact_pct": 0, "shadow_pct": 0, "zeroed_pct": 0, "layers": 0}


def measure_shadow_coverage(switches):
    """Measure combined pool+shadow coverage from TG indices."""
    total_pool_hit = 0
    total_shadow_hit = 0
    total_double_miss = 0
    total_count = 0

    for sw in switches:
        if not sw._tg_indices_buffer or not sw._pool_expert_ids:
            continue
        pool_set = set(sw._pool_expert_ids)
        shadow_set = set(sw._shadow_expert_ids) if sw._shadow_expert_ids else set()
        all_tg = mx.concatenate(sw._tg_indices_buffer)
        ids = np.array(all_tg, copy=False)

        pool_hit = np.isin(ids, list(pool_set))
        shadow_hit = np.isin(ids, list(shadow_set)) & ~pool_hit if shadow_set else np.zeros_like(pool_hit)
        double_miss = ~pool_hit & ~np.isin(ids, list(shadow_set)) if shadow_set else ~pool_hit

        total_pool_hit += int(pool_hit.sum())
        total_shadow_hit += int(shadow_hit.sum())
        total_double_miss += int(double_miss.sum())
        total_count += len(ids)

    if total_count > 0:
        return {
            "pool_pct": total_pool_hit / total_count * 100,
            "shadow_pct": total_shadow_hit / total_count * 100,
            "miss_pct": total_double_miss / total_count * 100,
        }
    return {"pool_pct": 0, "shadow_pct": 0, "miss_pct": 0}


def run_standard(model_path, prompt, max_tokens):
    print("\n  [A] standard 6-bit...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()
    text, tps, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)
    del model, tokenizer; gc.collect(); mx.metal.clear_cache()
    return {"label": "A_standard", "tg_tps": tps, "p50_ms": p50, "peak_gb": peak,
            "text": text[:400], "hit_rate": 1.0,
            "mass": {"exact_pct": 100, "shadow_pct": 0, "zeroed_pct": 0},
            "coverage": {"pool_pct": 100, "shadow_pct": 0, "miss_pct": 0}}


def run_ftec_variant(model_path, prompt, max_tokens, label,
                     pool_size, miss_policy,
                     rerank_bonus=0, guard_j=0, guard_tau=0.02,
                     shadow_size=0, shadow_bits=4):
    """Run an FTEC variant config."""
    opts = [f"miss={miss_policy}"]
    if rerank_bonus:
        opts.append(f"rr{rerank_bonus}")
    if guard_j:
        opts.append(f"guard_j={guard_j}")
    if shadow_size:
        opts.append(f"shd{shadow_size}@{shadow_bits}bit")
    print(f"\n  [{label}] pool={pool_size}, {', '.join(opts)}...")

    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=True,
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

    # Create decode-hot shadow AFTER recompact (uses TG frequency data)
    shadow_gb = 0
    if shadow_size > 0 and miss_policy == "ftec":
        info = ctx.create_decode_shadow(size=shadow_size, bits=shadow_bits)
        shadow_gb = info.get("shadow_gb", 0)

    # Enable reranking AFTER recompact
    if rerank_bonus > 0:
        ctx.enable_reranking(bonus=rerank_bonus, guard_j=guard_j, guard_tau=guard_tau)

    # Set miss policy and clear TG buffers for measurement
    for sw in switches:
        sw._miss_policy = miss_policy
        sw._tg_indices_buffer = []
        sw._tg_scores_buffer = []
        sw._tg_token_count = 0
        sw._disable_tg_buffer = False
        # Reset telemetry mass counters
        if sw._telemetry:
            sw._telemetry._exact_mass[sw._layer_idx] = 0
            sw._telemetry._shadow_mass[sw._layer_idx] = 0
            sw._telemetry._zeroed_mass[sw._layer_idx] = 0

    gc.collect(); mx.metal.clear_cache()

    text, tps, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)

    hr = measure_hit_rate(switches)
    coverage = measure_shadow_coverage(switches)
    mass = measure_mass_metrics(switches)

    ctx.close()
    del model, tokenizer; gc.collect(); mx.metal.clear_cache()

    return {"label": label, "tg_tps": tps, "p50_ms": p50, "peak_gb": peak,
            "text": text[:400], "hit_rate": hr,
            "pool_size": pool_size, "miss_policy": miss_policy,
            "shadow_size": shadow_size, "shadow_bits": shadow_bits,
            "shadow_gb": shadow_gb,
            "rerank_bonus": rerank_bonus, "guard_j": guard_j,
            "mass": mass, "coverage": coverage}


def main():
    parser = argparse.ArgumentParser(description="FTEC Benchmark")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  FTEC: Fidelity-first TEP Execution Contract Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.tokens}")
    print("=" * 70)

    prompt = (
        "Explain the mathematical foundations of attention mechanisms in "
        "transformers, including scaled dot-product attention and multi-head "
        "attention. Provide the full mathematical derivation."
    )

    results = []

    # A: Standard (clean baseline)
    results.append(run_standard(args.model, prompt, args.tokens))

    # B: pool=32 + zero_out + legacy rerank (current best speed)
    results.append(run_ftec_variant(
        args.model, prompt, args.tokens,
        label="B_zero32_rr", pool_size=32, miss_policy="zero_out",
        rerank_bonus=0.01, guard_j=0,
    ))

    # C: pool=32 + zero_out + guarded rerank (test guard alone)
    results.append(run_ftec_variant(
        args.model, prompt, args.tokens,
        label="C_zero32_grr", pool_size=32, miss_policy="zero_out",
        rerank_bonus=0.005, guard_j=2, guard_tau=0.02,
    ))

    # D: FTEC main — pool=32 + guarded rerank + decode_shadow_64
    results.append(run_ftec_variant(
        args.model, prompt, args.tokens,
        label="D_ftec_main", pool_size=32, miss_policy="ftec",
        rerank_bonus=0.005, guard_j=2, guard_tau=0.02,
        shadow_size=64, shadow_bits=4,
    ))

    # E: pool=64 + zero_out + legacy rerank (P2 reference)
    results.append(run_ftec_variant(
        args.model, prompt, args.tokens,
        label="E_zero64_rr", pool_size=64, miss_policy="zero_out",
        rerank_bonus=0.01, guard_j=0,
    ))

    # Summary
    baseline = results[0]["tg_tps"]
    print(f"\n{'=' * 70}")
    print(f"  FTEC RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Config':<14} {'TG tok/s':>9} {'vs Std':>7} {'P50 ms':>9} "
          f"{'Peak GB':>8} {'Pool HR':>8}")
    print(f"  {'-'*14} {'-'*9} {'-'*7} {'-'*9} {'-'*8} {'-'*8}")

    for r in results:
        delta = (r["tg_tps"] / baseline - 1) * 100 if baseline > 0 else 0
        print(f"  {r['label']:<14} {r['tg_tps']:>8.1f} {delta:>+6.0f}% "
              f"{r['p50_ms']:>8.2f} {r['peak_gb']:>7.2f} "
              f"{r['hit_rate']:>7.1%}")

    # Coverage breakdown (FTEC key metric)
    print(f"\n  COVERAGE BREAKDOWN:")
    print(f"  {'Config':<14} {'Pool Hit':>9} {'Shadow Hit':>11} {'Double Miss':>12}")
    print(f"  {'-'*14} {'-'*9} {'-'*11} {'-'*12}")
    for r in results:
        c = r.get("coverage", {})
        print(f"  {r['label']:<14} {c.get('pool_pct', 100):>8.1f}% "
              f"{c.get('shadow_pct', 0):>10.1f}% "
              f"{c.get('miss_pct', 0):>11.1f}%")

    # Mass metrics (FTEC primary quality indicator)
    print(f"\n  MASS METRICS (gate mass distribution):")
    print(f"  {'Config':<14} {'Exact Mass':>11} {'Shadow Mass':>12} {'Zeroed Mass':>12}")
    print(f"  {'-'*14} {'-'*11} {'-'*12} {'-'*12}")
    for r in results:
        m = r.get("mass", {})
        print(f"  {r['label']:<14} {m.get('exact_pct', 100):>10.1f}% "
              f"{m.get('shadow_pct', 0):>11.1f}% "
              f"{m.get('zeroed_pct', 0):>11.1f}%")

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
        if r["label"] != "A_standard":
            print(f"      → {r['text'][:120]}")

    # FTEC vs Zero analysis
    print(f"\n  FTEC vs ZERO-OUT ANALYSIS:")
    ftec = next((r for r in results if r["label"] == "D_ftec_main"), None)
    zero = next((r for r in results if r["label"] == "B_zero32_rr"), None)
    if ftec and zero:
        speed_delta = (ftec["tg_tps"] / zero["tg_tps"] - 1) * 100
        mem_delta = ftec["peak_gb"] - zero["peak_gb"]
        print(f"    Speed: {ftec['tg_tps']:.1f} vs {zero['tg_tps']:.1f} tok/s "
              f"({speed_delta:+.0f}%)")
        print(f"    Memory: {ftec['peak_gb']:.1f} vs {zero['peak_gb']:.1f} GB "
              f"({mem_delta:+.1f} GB)")
        fm = ftec.get("mass", {})
        zm = zero.get("mass", {})
        print(f"    Zeroed mass: {fm.get('zeroed_pct', 0):.1f}% vs "
              f"{zm.get('zeroed_pct', 0):.1f}%")
        print(f"    Shadow mass: {fm.get('shadow_pct', 0):.1f}% (FTEC only)")

    # Save
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".solar", "tep-ftec.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "results": results}, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
