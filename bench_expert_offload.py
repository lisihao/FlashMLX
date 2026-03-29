#!/usr/bin/env python3
"""
FlashMLX Expert Offload v3 Benchmark — Three-Tier Architecture

Compares:
  1. Standard inference (all experts in GPU memory)
  2. Three-tier offloading (GPU pool + CPU warm + SSD cold)

Measures: TG speed, peak memory, quality, telemetry stats.
"""

import gc
import json
import os
import sys
import time

MLX_LM_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx-lm-source")
sys.path.insert(0, MLX_LM_SOURCE)

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate

MODEL_PATH = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
MAX_TOKENS = 100

PROMPTS = [
    ("short", "What is 2+2? Answer in one word:"),
    ("medium", "Explain the difference between TCP and UDP in 3 sentences."),
]


def measure_generation(model, tokenizer, prompt_text, max_tokens=MAX_TOKENS, label=""):
    """Measure generation performance."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    mx.metal.reset_peak_memory()
    gc.collect()

    t0 = time.perf_counter()
    text = ""
    token_count = 0
    ttof = 0
    got_first = False
    last_response = None

    for response in stream_generate(model, tokenizer, formatted, max_tokens=max_tokens):
        if not got_first:
            ttof = time.perf_counter() - t0
            got_first = True
        text += response.text
        token_count += 1
        last_response = response

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024 / 1024 / 1024

    tg_tps = last_response.generation_tps if last_response else 0
    pp_tps = last_response.prompt_tps if last_response else 0

    return {
        "label": label,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "ttof_ms": ttof * 1000,
        "total_ms": total * 1000,
        "peak_gb": peak,
        "tokens": token_count,
        "text": text[:120],
    }


def run_standard(prompt_name, prompt_text):
    """Run standard inference (all experts in memory)."""
    print(f"\n  Loading model (standard)...")
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    gc.collect()

    mem = mx.metal.get_active_memory() / 1024 / 1024 / 1024
    print(f"  Model loaded: {mem:.2f} GB")

    result = measure_generation(model, tokenizer, prompt_text,
                                 label=f"standard_{prompt_name}")
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    return result


def run_offloaded(prompt_name, prompt_text, cpu_cache_gb=2.0):
    """Run with three-tier expert offloading."""
    from mlx_lm.models.expert_offload import patch_model_for_offload

    print(f"\n  Loading model (three-tier)...")
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    gc.collect()

    mem_before = mx.metal.get_active_memory() / 1024 / 1024 / 1024
    print(f"  Model loaded: {mem_before:.2f} GB")

    ctx = patch_model_for_offload(
        model, MODEL_PATH,
        max_workers=4,
        cpu_cache_gb=cpu_cache_gb,
        enable_prefetch=True,
        enable_telemetry=True,
    )
    gc.collect()
    mx.metal.clear_cache()

    mem_after = mx.metal.get_active_memory() / 1024 / 1024 / 1024
    print(f"  After patch: {mem_after:.2f} GB (saved {mem_before - mem_after:.2f} GB)")
    print(f"  Regime: {ctx.regime.regime}")

    result = measure_generation(model, tokenizer, prompt_text,
                                 label=f"offload_{prompt_name}")
    result["mem_before_gb"] = mem_before
    result["mem_after_gb"] = mem_after
    result["regime"] = ctx.regime.regime

    # Collect telemetry
    if ctx.telemetry:
        tel_summary = ctx.telemetry.summary()
        result["telemetry"] = tel_summary
        print(f"  Telemetry: {tel_summary['total_tokens']} tokens, "
              f"pool hit rate: {tel_summary['overall_pool_hit_rate']:.2%}")

    if ctx.cpu_cache:
        cpu_summary = ctx.cpu_cache.summary()
        result["cpu_cache"] = cpu_summary
        print(f"  CPU cache: {cpu_summary['entries']} entries, "
              f"hit rate: {cpu_summary['hit_rate']:.2%}")

    ctx.close()
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    return result


def main():
    print("=" * 80)
    print("  FlashMLX Expert Offload v3 Benchmark — Three-Tier Architecture")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Platform: Apple M4 Pro")
    print(f"  Max tokens: {MAX_TOKENS}")
    print("=" * 80)

    all_results = []

    for prompt_name, prompt_text in PROMPTS:
        print(f"\n{'=' * 60}")
        print(f"  Prompt: {prompt_name}")
        print(f"{'=' * 60}")

        # Standard
        print(f"\n  --- Standard ---")
        r_std = run_standard(prompt_name, prompt_text)
        all_results.append(r_std)
        print(f"    TG: {r_std['tg_tps']:.1f} tok/s | Peak: {r_std['peak_gb']:.2f} GB")
        print(f"    TTOF: {r_std['ttof_ms']:.0f}ms | Text: {r_std['text'][:60]}")

        # Three-tier offloaded
        print(f"\n  --- Three-Tier Offload ---")
        r_off = run_offloaded(prompt_name, prompt_text)
        all_results.append(r_off)
        print(f"    TG: {r_off['tg_tps']:.1f} tok/s | Peak: {r_off['peak_gb']:.2f} GB")
        print(f"    TTOF: {r_off['ttof_ms']:.0f}ms | Text: {r_off['text'][:60]}")
        print(f"    Memory: {r_off.get('mem_before_gb', 0):.2f} -> {r_off.get('mem_after_gb', 0):.2f} GB")

        # Deltas
        if r_std['tg_tps'] > 0:
            tg_delta = (r_off['tg_tps'] - r_std['tg_tps']) / r_std['tg_tps'] * 100
            mem_delta = (r_off['peak_gb'] - r_std['peak_gb']) / r_std['peak_gb'] * 100
            print(f"    Delta: TG {tg_delta:+.1f}%, Memory {mem_delta:+.1f}%")

    # Summary
    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    hdr = f"  {'Config':<25} | {'TG tok/s':>9} | {'PP tok/s':>9} | {'Peak GB':>8} | Output"
    print(hdr)
    print(f"  {'-'*25}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*30}")
    for r in all_results:
        print(f"  {r['label']:<25} | {r['tg_tps']:>9.1f} | {r['pp_tps']:>9.1f} | "
              f"{r['peak_gb']:>7.2f} | {r['text'][:30]}")

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".solar", "bench-v3-three-tier-results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert non-serializable items
    clean_results = []
    for r in all_results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (int, float, str, bool, list, dict)):
                cr[k] = v
            else:
                cr[k] = str(v)
        clean_results.append(cr)

    with open(output_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
