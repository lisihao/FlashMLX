#!/usr/bin/env python3
"""
TEP P0: Low-bit Shadow Expert Benchmark

Tests shadow miss policy vs zero_out vs k1_clamp at various pool sizes.
Shadow provides low-precision but correct expert output for misses,
instead of zeroing (zero_out) or using wrong expert (k1_clamp).

Configs:
  A. standard              — no offloading (baseline)
  B. dr+zero_32            — decode recompact, pool=32, zero_out
  C. dr+shadow4_32         — decode recompact, pool=32, 4-bit shadow
  D. dr+shadow2_32         — decode recompact, pool=32, 2-bit shadow
  E. dr+shadow4_64         — decode recompact, pool=64, 4-bit shadow
  F. shadow4_only          — ALL experts from 4-bit shadow (no pool)

Usage:
    python3 benchmarks/bench_shadow.py [--model PATH] [--tokens N]
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
    """Generate and return (text, tg_tps, peak_gb, p50_ms)."""
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


def run_standard(model_path, prompt, max_tokens):
    print("\n  [A] standard (no offloading)...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    text, tps, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)

    del model, tokenizer; gc.collect(); mx.metal.clear_cache()
    return {"label": "A_standard", "tg_tps": tps, "p50_ms": p50, "peak_gb": peak,
            "text": text[:300]}


def run_shadow_variant(model_path, prompt, max_tokens, label,
                       pool_size, miss_policy, shadow_bits=None,
                       shadow_only=False):
    print(f"\n  [{label}] pool={pool_size}, miss={miss_policy}, "
          f"shadow_bits={shadow_bits}, shadow_only={shadow_only}...")

    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, model_path,
        pool_size=256,
        max_workers=4,
        cpu_cache_gb=0.0,
        enable_prefetch=False,
        enable_telemetry=False,
    )
    gc.collect()

    # Warmup: prefill populates PP indices
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}], add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    # Compact to target pool size
    ctx.compact(pool_size=pool_size)

    # Decode recompact warmup: generate 15 tokens with k1_clamp to build TG indices
    switches = _get_switch_layers(model)
    for sw in switches:
        sw._pool_is_identity = False
        sw._pool_compacted = True
        sw._miss_policy = "k1_clamp"

    warmup_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 20."}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, warmup_prompt, max_tokens=15):
        pass

    # Decode recompact: rebuild pool from TG data
    ctx.decode_recompact(pool_size=pool_size)

    # Create shadow if needed
    if shadow_bits is not None:
        ctx.create_shadow(bits=shadow_bits)

    # Configure miss policy
    for sw in switches:
        sw._miss_policy = miss_policy

        if shadow_only and sw._shadow is not None:
            # Replace pool with shadow — ALL experts from shadow
            sw._pool = sw._shadow
            sw._pool_bits = shadow_bits  # override gather_qmm bits
            sw._pool_expert_ids = list(range(sw.num_experts))
            sw._pool_remap_np = np.arange(sw.num_experts, dtype=np.int32)
            sw._pool_remap = mx.array(sw._pool_remap_np)
            sw._pool_is_identity = True
            sw._miss_policy = "k1_clamp"  # no misses in shadow_only

    # Clear GPU cache
    mx.metal.set_cache_limit(0); gc.collect(); mx.metal.clear_cache()
    mx.metal.set_cache_limit(int(64 * 1024**3))

    text, tps, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)

    ctx.close()
    del model, tokenizer; gc.collect(); mx.metal.clear_cache()
    mx.metal.set_cache_limit(0)

    return {"label": label, "tg_tps": tps, "p50_ms": p50, "peak_gb": peak,
            "text": text[:300], "pool_size": pool_size,
            "miss_policy": miss_policy, "shadow_bits": shadow_bits}


def main():
    parser = argparse.ArgumentParser(description="TEP P0: Low-bit Shadow Benchmark")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP P0: Low-bit Shadow Expert Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.tokens}")
    print("=" * 70)

    prompt = (
        "Explain the mathematical foundations of attention mechanisms in "
        "transformers, including scaled dot-product attention and multi-head "
        "attention. Provide the full mathematical derivation."
    )

    results = []

    # A: Standard baseline
    results.append(run_standard(args.model, prompt, args.tokens))

    # B: dr+zero_32 (known good baseline)
    results.append(run_shadow_variant(
        args.model, prompt, args.tokens,
        label="B_dr+zero_32", pool_size=32,
        miss_policy="zero_out",
    ))

    # C: dr+shadow4_32 (4-bit shadow, pool=32)
    results.append(run_shadow_variant(
        args.model, prompt, args.tokens,
        label="C_dr+shd4_32", pool_size=32,
        miss_policy="shadow", shadow_bits=4,
    ))

    # D: dr+shadow2_32 (2-bit shadow, pool=32)
    results.append(run_shadow_variant(
        args.model, prompt, args.tokens,
        label="D_dr+shd2_32", pool_size=32,
        miss_policy="shadow", shadow_bits=2,
    ))

    # E: dr+shadow4_64 (4-bit shadow, pool=64)
    results.append(run_shadow_variant(
        args.model, prompt, args.tokens,
        label="E_dr+shd4_64", pool_size=64,
        miss_policy="shadow", shadow_bits=4,
    ))

    # F: shadow4_only (all 256 experts from 4-bit shadow, no pool)
    results.append(run_shadow_variant(
        args.model, prompt, args.tokens,
        label="F_shd4_only", pool_size=32,
        miss_policy="k1_clamp", shadow_bits=4,
        shadow_only=True,
    ))

    # Summary
    baseline = results[0]["tg_tps"]
    print(f"\n{'=' * 70}")
    print(f"  LOW-BIT SHADOW RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Config':<16} {'TG tok/s':>9} {'vs Std':>7} {'P50 ms':>9} {'Peak GB':>8}")
    print(f"  {'-'*16} {'-'*9} {'-'*7} {'-'*9} {'-'*8}")

    for r in results:
        delta = (r["tg_tps"] / baseline - 1) * 100 if baseline > 0 else 0
        print(f"  {r['label']:<16} {r['tg_tps']:>8.1f} {delta:>+6.0f}% "
              f"{r['p50_ms']:>8.2f} {r['peak_gb']:>7.2f}")

    # Quality comparison
    print(f"\n  QUALITY CHECK (first 200 chars):")
    ref = results[0]["text"][:200]
    for r in results:
        if len(r["text"]) == 0:
            print(f"    {r['label']}: EMPTY OUTPUT")
            continue
        common = sum(1 for a, b in zip(ref, r["text"][:200]) if a == b)
        match = common / max(len(ref), 1)
        status = "MATCH" if match > 0.90 else ("CLOSE" if match > 0.70 else "DIFFER")
        print(f"    {r['label']}: {match:.0%} match → {status}")
        if match < 0.70:
            print(f"      First 100 chars: {r['text'][:100]}")

    # Analysis
    print(f"\n  ANALYSIS:")
    a_tps = results[0]["tg_tps"]
    b_tps = results[1]["tg_tps"]
    c_tps = results[2]["tg_tps"]
    d_tps = results[3]["tg_tps"]

    print(f"    zero_out→shadow4 (pool=32): {(c_tps/b_tps-1)*100:+.0f}% speed")
    print(f"    zero_out→shadow2 (pool=32): {(d_tps/b_tps-1)*100:+.0f}% speed")
    print(f"    shadow4 memory overhead: {results[2]['peak_gb'] - results[1]['peak_gb']:+.1f} GB")
    print(f"    shadow2 memory overhead: {results[3]['peak_gb'] - results[1]['peak_gb']:+.1f} GB")

    # Save
    out_path = args.output
    if not out_path:
        solar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".solar")
        os.makedirs(solar_dir, exist_ok=True)
        out_path = os.path.join(solar_dir, "tep-shadow.json")

    with open(out_path, "w") as f:
        json.dump({"model": args.model, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "results": results}, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
