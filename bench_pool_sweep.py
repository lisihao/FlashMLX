#!/usr/bin/env python3
"""
FlashMLX Pool Size Sweep — Memory↔Performance Tradeoff

Measures TG speed and memory at different compact pool sizes:
  pool=256 (no compact, baseline)
  pool=224 (12.5% evicted)
  pool=192 (25% evicted)
  pool=128 (50% evicted)
  pool=64  (75% evicted)

All configs use same full-pool PP, then compact to target size.
Shows that Expert Offload CAN save 2-13 GB with acceptable TG impact.
"""

import gc
import json
import os
import sys
import time

MLX_LM_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx-lm-source")
sys.path.insert(0, MLX_LM_SOURCE)

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import generate_step

MODEL_PATH = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
PROMPT = "Explain the difference between TCP and UDP in 3 sentences."
MAX_GEN_TOKENS = 50
POOL_SIZES = [256, 224, 192, 128, 64]


def measure_tg(model, tokenizer, prompt_text, max_tokens=MAX_GEN_TOKENS, label=""):
    """Measure TG speed after first token."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_array = mx.array(tokenizer.encode(formatted))
    prompt_tokens = len(tokenizer.encode(formatted))

    mx.metal.reset_peak_memory()
    gc.collect()
    mx.eval(mx.zeros(1))

    # PP: first token
    t0 = time.perf_counter()
    gen = generate_step(prompt_array, model)
    first_token, _ = next(gen)
    t_pp = time.perf_counter() - t0
    pp_tps = prompt_tokens / t_pp

    return gen, first_token, pp_tps, prompt_tokens


def finish_generation(gen, first_token, max_tokens=MAX_GEN_TOKENS):
    """Finish generation after compact (if any) and measure TG."""
    t_gen = time.perf_counter()
    ft = first_token if isinstance(first_token, int) else first_token.item()
    tokens_out = [ft]
    for i, (tok, _) in enumerate(gen):
        t = tok if isinstance(tok, int) else tok.item()
        tokens_out.append(t)
        if i + 1 >= max_tokens - 1:
            break
    gen_time = time.perf_counter() - t_gen
    tg_tps = len(tokens_out) / gen_time if gen_time > 0 else 0
    peak = mx.metal.get_peak_memory() / 1024**3
    return tokens_out, tg_tps, peak


def run_standard(model, tokenizer):
    """Standard baseline (no offloading)."""
    print("\n  --- Standard (no offload) ---")
    gen, first_token, pp_tps, prompt_tokens = measure_tg(model, tokenizer, PROMPT)
    tokens_out, tg_tps, peak = finish_generation(gen, first_token)
    text = tokenizer.decode(tokens_out)
    print(f"    PP: {pp_tps:.0f} tok/s | TG: {tg_tps:.1f} tok/s | Peak: {peak:.2f} GB")
    return {
        "label": "standard",
        "pool_size": 256,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "peak_gb": peak,
        "text": text[:80],
    }


def run_offload_sweep(model, tokenizer, pool_sizes):
    """Run offloaded inference at each pool size."""
    from mlx_lm.models.expert_offload import patch_model_for_offload

    print("\n  Loading model (offload)...")
    model_off, tokenizer_off = load(MODEL_PATH)
    mx.eval(model_off.parameters())
    gc.collect()

    mem_before = mx.metal.get_active_memory() / 1024**3
    ctx = patch_model_for_offload(
        model_off, MODEL_PATH,
        max_workers=4,
        cpu_cache_gb=2.0,  # Will be auto-expanded by compact()
        enable_prefetch=True,
        enable_telemetry=True,
    )
    gc.collect()
    mx.metal.clear_cache()
    mem_after = mx.metal.get_active_memory() / 1024**3
    print(f"  Model: {mem_before:.2f} → {mem_after:.2f} GB, Regime: {ctx.regime.regime}")

    results = []

    for pool_size in pool_sizes:
        print(f"\n  --- Offload pool={pool_size} ---")

        # Need to re-expand pool to FULL for each test
        # Re-load model fresh for clean measurement
        del model_off
        gc.collect()
        mx.metal.clear_cache()

        model_off, tokenizer_off = load(MODEL_PATH)
        mx.eval(model_off.parameters())
        gc.collect()

        ctx_new = patch_model_for_offload(
            model_off, MODEL_PATH,
            max_workers=4,
            cpu_cache_gb=2.0,
            enable_prefetch=False,  # Disable prefetch for clean measurement
            enable_telemetry=True,
        )
        gc.collect()
        mx.metal.clear_cache()

        # PP with full pool
        gen, first_token, pp_tps, prompt_tokens = measure_tg(
            model_off, tokenizer_off, PROMPT, label=f"pool_{pool_size}"
        )

        # Compact to target pool size (unless 256 = no compact)
        compact_info = None
        if pool_size < 256:
            compact_info = ctx_new.compact(pool_size=pool_size)

        mem_after_compact = mx.metal.get_active_memory() / 1024**3

        # TG with compact pool
        tokens_out, tg_tps, peak = finish_generation(gen, first_token)
        text = tokenizer_off.decode(tokens_out)

        # CPU cache stats
        cpu_info = {}
        if ctx_new.cpu_cache:
            cpu_info = ctx_new.cpu_cache.summary()

        r = {
            "label": f"pool_{pool_size}",
            "pool_size": pool_size,
            "pp_tps": pp_tps,
            "tg_tps": tg_tps,
            "peak_gb": peak,
            "mem_after_compact_gb": mem_after_compact,
            "text": text[:80],
        }
        if compact_info:
            r["compact"] = compact_info
        if cpu_info:
            r["cpu_cache"] = cpu_info

        results.append(r)
        print(f"    PP: {pp_tps:.0f} tok/s | TG: {tg_tps:.1f} tok/s | Peak: {peak:.2f} GB")
        print(f"    Mem after compact: {mem_after_compact:.2f} GB")
        if compact_info:
            print(f"    Coverage: {compact_info['pp_coverage']:.1%} | "
                  f"CPU cache: {compact_info.get('cpu_cache_gb', 0):.1f} GB")

        ctx_new.close()

    return results


def main():
    print("=" * 80)
    print("  FlashMLX Pool Size Sweep — Memory↔Performance Tradeoff")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Pool sizes: {POOL_SIZES}")
    print(f"  Gen tokens: {MAX_GEN_TOKENS}")
    print("=" * 80)

    # Standard baseline
    print("\n  Loading model (standard)...")
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    gc.collect()
    mem = mx.metal.get_active_memory() / 1024**3
    print(f"  Model loaded: {mem:.2f} GB")

    std_result = run_standard(model, tokenizer)
    del model
    gc.collect()
    mx.metal.clear_cache()

    # Offloaded sweep
    off_results = run_offload_sweep(None, None, POOL_SIZES)

    all_results = [std_result] + off_results

    # Summary table
    print(f"\n{'=' * 80}")
    print("  SUMMARY — Pool Size vs Performance vs Memory")
    print(f"{'=' * 80}")
    hdr = f"  {'Config':<12} | {'Pool':>5} | {'TG tok/s':>9} | {'TG Δ':>7} | {'Peak GB':>8} | {'Mem Δ':>7} | {'Coverage':>9}"
    print(hdr)
    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*9}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}")

    std_tg = std_result["tg_tps"]
    std_peak = std_result["peak_gb"]

    for r in all_results:
        tg_delta = (r["tg_tps"] - std_tg) / std_tg * 100 if std_tg > 0 else 0
        mem_delta = r["peak_gb"] - std_peak
        cov = ""
        if "compact" in r:
            cov = f"{r['compact']['pp_coverage']:.1%}"
        print(f"  {r['label']:<12} | {r['pool_size']:>5} | {r['tg_tps']:>9.1f} | "
              f"{tg_delta:>+6.1f}% | {r['peak_gb']:>7.2f} | {mem_delta:>+6.1f} | {cov:>9}")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".solar", "bench-pool-sweep-results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
