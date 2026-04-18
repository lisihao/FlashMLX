#!/usr/bin/env python3
"""
TEP Wrapper Gap Microbenchmark — Isolate Overhead Sources

Tests pool path variants to pinpoint the 38% gap:
  A. standard:          no offloading (baseline, 68 tok/s)
  B. identity_tel:      pool=256, identity=True,  telemetry=ON  (default offload)
  C. identity_notel:    pool=256, identity=True,  telemetry=OFF
  D. remap_tel:         pool=256, identity=False, telemetry=ON  (= hit_only_256)
  E. remap_notel:       pool=256, identity=False, telemetry=OFF
  F. remap_notel_nobuf: pool=256, identity=False, telemetry=OFF, TG buffer=OFF

If D→E shows big improvement: telemetry GPU sync is the culprit.
If E→A shows big gap: remap + gather path overhead.
If B≈A: identity path is already fine, only partial pool is slow.

Usage:
    python3 benchmarks/bench_wrapper_gap.py [--model PATH] [--tokens N]
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


def timed_generate(model, tokenizer, prompt, max_tokens, seed=42):
    """Generate and return (text, tg_tps, total_ms, peak_gb, per_token_ms)."""
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

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024**3
    text = "".join(text_parts)

    # Skip first 3 tokens for TPOT
    arr = np.array(token_times[3:]) if len(token_times) > 3 else np.array(token_times)
    p50 = float(np.percentile(arr, 50)) if len(arr) > 0 else 0

    return text, tg_tps, total * 1000, peak, p50


def run_standard(model_path, prompt, max_tokens):
    print("\n  [A] standard (no offloading)...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    text, tps, total, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)

    del model, tokenizer; gc.collect(); mx.metal.clear_cache()
    return {"label": "A_standard", "tg_tps": tps, "p50_ms": p50, "peak_gb": peak, "text": text}


def run_offload_variant(model_path, prompt, max_tokens, label,
                        force_identity, enable_telemetry, disable_tg_buffer):
    from mlx_lm.models.expert_offload import patch_model_for_offload

    print(f"\n  [{label}] identity={force_identity}, tel={enable_telemetry}, buf={not disable_tg_buffer}...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, model_path,
        pool_size=256,
        max_workers=4,
        cpu_cache_gb=0.0,
        enable_prefetch=False,
        enable_telemetry=enable_telemetry,
    )
    gc.collect()

    # Dummy gen
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}], add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=3):
        pass

    # Configure switches
    switches = _get_switch_layers(model)
    for sw in switches:
        if not force_identity:
            sw._pool_is_identity = False
        sw._pool_compacted = True

        if not enable_telemetry:
            sw._telemetry = None

        if disable_tg_buffer:
            # Monkey-patch: override the TG buffer append to no-op
            sw._disable_tg_buffer = True

    # Clear GPU cache
    mx.metal.set_cache_limit(0); gc.collect(); mx.metal.clear_cache()
    mx.metal.set_cache_limit(int(64 * 1024**3))

    text, tps, total, peak, p50 = timed_generate(model, tokenizer, prompt, max_tokens)

    ctx.close()
    del model, tokenizer; gc.collect(); mx.metal.clear_cache()
    mx.metal.set_cache_limit(0)
    return {"label": label, "tg_tps": tps, "p50_ms": p50, "peak_gb": peak, "text": text}


def main():
    parser = argparse.ArgumentParser(description="TEP Wrapper Gap Microbenchmark")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP Wrapper Gap — Isolating 38% Overhead")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.tokens}")
    print("=" * 70)

    prompt = (
        "Explain the mathematical foundations of attention mechanisms in "
        "transformers, including scaled dot-product attention and multi-head "
        "attention. Provide the full mathematical derivation."
    )

    results = []

    # A: Standard (no offloading)
    results.append(run_standard(args.model, prompt, args.tokens))

    # B: Identity + telemetry ON (default offload path)
    results.append(run_offload_variant(
        args.model, prompt, args.tokens,
        label="B_id+tel",
        force_identity=True, enable_telemetry=True, disable_tg_buffer=False,
    ))

    # C: Identity + telemetry OFF
    results.append(run_offload_variant(
        args.model, prompt, args.tokens,
        label="C_id-tel",
        force_identity=True, enable_telemetry=False, disable_tg_buffer=False,
    ))

    # D: Remap + telemetry ON (= hit_only_256)
    results.append(run_offload_variant(
        args.model, prompt, args.tokens,
        label="D_remap+tel",
        force_identity=False, enable_telemetry=True, disable_tg_buffer=False,
    ))

    # E: Remap + telemetry OFF
    results.append(run_offload_variant(
        args.model, prompt, args.tokens,
        label="E_remap-tel",
        force_identity=False, enable_telemetry=False, disable_tg_buffer=False,
    ))

    # F: Remap + telemetry OFF + TG buffer OFF
    results.append(run_offload_variant(
        args.model, prompt, args.tokens,
        label="F_remap-all",
        force_identity=False, enable_telemetry=False, disable_tg_buffer=True,
    ))

    # Summary
    baseline = results[0]["tg_tps"]
    print(f"\n{'=' * 70}")
    print(f"  WRAPPER GAP RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Config':<16} {'TG tok/s':>9} {'vs Std':>7} {'P50 ms':>9} {'Peak GB':>8}")
    print(f"  {'-'*16} {'-'*9} {'-'*7} {'-'*9} {'-'*8}")

    for r in results:
        delta = (r["tg_tps"] / baseline - 1) * 100
        print(f"  {r['label']:<16} {r['tg_tps']:>8.1f} {delta:>+6.0f}% "
              f"{r['p50_ms']:>8.2f} {r['peak_gb']:>7.2f}")

    # Diagnosis
    print(f"\n  DIAGNOSIS:")
    a = results[0]["tg_tps"]
    b = results[1]["tg_tps"]
    c = results[2]["tg_tps"]
    d = results[3]["tg_tps"]
    e = results[4]["tg_tps"]
    f = results[5]["tg_tps"]

    print(f"    A→B (identity offload overhead): {(b/a-1)*100:+.0f}%")
    print(f"    B→C (telemetry cost on identity): {(c/b-1)*100:+.0f}%")
    print(f"    A→D (remap+tel, = hit_only): {(d/a-1)*100:+.0f}%")
    print(f"    D→E (telemetry cost on remap): {(e/d-1)*100:+.0f}%")
    print(f"    E→F (TG buffer cost): {(f/e-1)*100:+.0f}%")
    print(f"    A→F (pure remap overhead, no tel/buf): {(f/a-1)*100:+.0f}%")

    if (e/d - 1) > 0.10:
        print(f"\n    → TELEMETRY GPU SYNC is major culprit ({(e/d-1)*100:+.0f}% recovery)")
    if (f/a - 1) < -0.15:
        print(f"    → REMAP PATH has significant kernel overhead ({(f/a-1)*100:+.0f}% remaining)")
    if abs(b/a - 1) < 0.05:
        print(f"    → IDENTITY PATH is already near-standard ({(b/a-1)*100:+.0f}%)")

    # Text check (should all match standard)
    ref = results[0]["text"]
    for r in results[1:]:
        match = sum(1 for a, b in zip(ref, r["text"]) if a == b) / max(len(ref), len(r["text"]))
        status = "MATCH" if match > 0.95 else "DIFFER"
        print(f"    {r['label']}: {match:.0%} match → {status}")

    # Save
    out_path = args.output
    if not out_path:
        solar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".solar")
        os.makedirs(solar_dir, exist_ok=True)
        out_path = os.path.join(solar_dir, "tep-wrapper-gap.json")

    with open(out_path, "w") as f:
        json.dump({"model": args.model, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "results": results}, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
