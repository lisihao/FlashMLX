#!/usr/bin/env python3
"""
TEP P0 v2: Optimal Shadow Strategy

Key insight from v1: 4-bit full shadow is 100% quality match to 6-bit standard.
This means we can use 4-bit as the POOL format, saving memory without quality loss.

New configs to test:
  A. standard              — 6-bit, no offloading (baseline, run FIRST)
  B. 4bit_full             — 4-bit full model (all 256 experts), identity path
  C. 2bit_full             — 2-bit full model (quality floor)
  D. dr+k1_32_4bit         — 4-bit pool=32 + decode recompact + k1_clamp
  E. dr+zero_32_4bit       — 4-bit pool=32 + decode recompact + zero_out
  F. dr+zero_32_6bit       — 6-bit pool=32 + decode recompact + zero_out (reference)

Usage:
    python3 benchmarks/bench_shadow_v2.py [--model PATH] [--tokens N]
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


def run_standard(model_path, prompt, max_tokens):
    """A: Standard 6-bit, no offloading."""
    print("\n  [A] standard 6-bit...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()
    text, tps, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)
    del model, tokenizer; gc.collect(); mx.metal.clear_cache()
    return {"label": "A_std_6bit", "tg_tps": tps, "p50_ms": p50, "peak_gb": peak,
            "text": text[:400]}


def run_shadow_full(model_path, prompt, max_tokens, bits, label):
    """B/C: All 256 experts re-quantized to `bits`, identity path."""
    print(f"\n  [{label}] full {bits}-bit...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=False,
    )
    gc.collect()

    # Warmup
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}], add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=3):
        pass

    # Create shadow with all experts at target bits
    ctx.create_shadow(bits=bits)

    # Replace pool with shadow (all 256 experts, identity)
    switches = _get_switch_layers(model)
    for sw in switches:
        sw._pool = sw._shadow
        sw._pool_bits = bits
        sw._pool_expert_ids = list(range(sw.num_experts))
        sw._pool_remap_np = np.arange(sw.num_experts, dtype=np.int32)
        sw._pool_remap = mx.array(sw._pool_remap_np)
        sw._pool_is_identity = True
        sw._shadow = None  # free shadow reference (pool IS the shadow now)

    gc.collect(); mx.metal.clear_cache()

    text, tps, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)
    ctx.close()
    del model, tokenizer; gc.collect(); mx.metal.clear_cache()
    return {"label": label, "tg_tps": tps, "p50_ms": p50, "peak_gb": peak,
            "text": text[:400], "bits": bits}


def run_shadow_pool(model_path, prompt, max_tokens, pool_size, bits,
                    miss_policy, label):
    """D/E/F: Pool=N with decode recompact, experts at `bits`."""
    print(f"\n  [{label}] pool={pool_size}, {bits}-bit, miss={miss_policy}...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=False,
    )
    gc.collect()

    # Warmup
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}], add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    # Compact
    ctx.compact(pool_size=pool_size)

    # TG warmup for decode recompact
    switches = _get_switch_layers(model)
    for sw in switches:
        sw._pool_is_identity = False
        sw._pool_compacted = True
        sw._miss_policy = "k1_clamp"

    warmup = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 20."}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, warmup, max_tokens=15):
        pass

    # Decode recompact
    ctx.decode_recompact(pool_size=pool_size)

    # If 4-bit pool: re-quantize pool weights to target bits
    if bits != 6:  # model is 6-bit, re-quantize pool
        for sw in switches:
            new_pool = {}
            for comp_name, arr in sw._pool.items():
                if comp_name.endswith(".weight"):
                    base = comp_name.replace(".weight", "")
                    scales = sw._pool[f"{base}.scales"]
                    biases = sw._pool[f"{base}.biases"]
                    w_float = mx.dequantize(arr, scales, biases,
                                            group_size=sw.group_size, bits=sw.bits)
                    w_q, s_q, b_q = mx.quantize(w_float, group_size=sw.group_size,
                                                 bits=bits)
                    new_pool[comp_name] = w_q
                    new_pool[f"{base}.scales"] = s_q
                    new_pool[f"{base}.biases"] = b_q
            mx.eval(new_pool)
            sw._pool = new_pool
            sw._pool_bits = bits

    # Set miss policy
    for sw in switches:
        sw._miss_policy = miss_policy

    gc.collect(); mx.metal.clear_cache()

    text, tps, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)
    ctx.close()
    del model, tokenizer; gc.collect(); mx.metal.clear_cache()
    return {"label": label, "tg_tps": tps, "p50_ms": p50, "peak_gb": peak,
            "text": text[:400], "pool_size": pool_size, "bits": bits,
            "miss_policy": miss_policy}


def main():
    parser = argparse.ArgumentParser(description="TEP P0 v2: Optimal Shadow Strategy")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP P0 v2: Optimal Shadow Strategy")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.tokens}")
    print("=" * 70)

    prompt = (
        "Explain the mathematical foundations of attention mechanisms in "
        "transformers, including scaled dot-product attention and multi-head "
        "attention. Provide the full mathematical derivation."
    )

    results = []

    # A: Standard (run FIRST to get clean memory baseline)
    results.append(run_standard(args.model, prompt, args.tokens))

    # F: 6-bit pool=32 + zero_out (known good reference, run SECOND)
    results.append(run_shadow_pool(
        args.model, prompt, args.tokens,
        pool_size=32, bits=6, miss_policy="zero_out",
        label="F_6bit_z32",
    ))

    # D: 4-bit pool=32 + k1_clamp
    results.append(run_shadow_pool(
        args.model, prompt, args.tokens,
        pool_size=32, bits=4, miss_policy="k1_clamp",
        label="D_4bit_k32",
    ))

    # E: 4-bit pool=32 + zero_out
    results.append(run_shadow_pool(
        args.model, prompt, args.tokens,
        pool_size=32, bits=4, miss_policy="zero_out",
        label="E_4bit_z32",
    ))

    # B: 4-bit full (all 256 experts)
    results.append(run_shadow_full(
        args.model, prompt, args.tokens, bits=4, label="B_4bit_full",
    ))

    # C: 2-bit full (quality floor)
    results.append(run_shadow_full(
        args.model, prompt, args.tokens, bits=2, label="C_2bit_full",
    ))

    # Summary
    baseline = results[0]["tg_tps"]
    print(f"\n{'=' * 70}")
    print(f"  SHADOW v2 RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Config':<16} {'TG tok/s':>9} {'vs Std':>7} {'P50 ms':>9} {'Peak GB':>8}")
    print(f"  {'-'*16} {'-'*9} {'-'*7} {'-'*9} {'-'*8}")

    for r in results:
        delta = (r["tg_tps"] / baseline - 1) * 100 if baseline > 0 else 0
        print(f"  {r['label']:<16} {r['tg_tps']:>8.1f} {delta:>+6.0f}% "
              f"{r['p50_ms']:>8.2f} {r['peak_gb']:>7.2f}")

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
        if status == "DIFFER":
            print(f"      → {r['text'][:120]}")

    # Save
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".solar", "tep-shadow-v2.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "results": results}, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
