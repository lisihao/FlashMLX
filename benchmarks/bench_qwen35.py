#!/usr/bin/env python3
"""
FlashMLX Benchmark — Qwen3.5-35B-A3B (Hybrid SSM+Attention)

Runs each configuration in a separate subprocess for accurate memory isolation.
Measures: PP speed, TG speed, TTOF, Peak memory, quality.

Usage:
    python3 bench_qwen35.py                    # Run all benchmarks
    python3 bench_qwen35.py --worker <json>    # Internal: run single benchmark

Configs: standard, scored_bf16, scored_q8, scored_q4
Context: 16K, 32K
"""

import json
import os
import subprocess
import sys
import time
import gc
from datetime import datetime

MODEL_PATH = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
MLX_LM_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx-lm-source")
sys.path.insert(0, MLX_LM_SOURCE)

NEEDLE_FACT = "The annual budget for Project Alpha is exactly $12,500."

CONFIGS = {
    "standard": {"kv_cache": None, "flat_quant": None},
    "scored_bf16": {"kv_cache": "scored_pq", "flat_quant": None},
    "scored_q8": {"kv_cache": "scored_pq", "flat_quant": "q8_0"},
    "scored_q4": {"kv_cache": "scored_pq", "flat_quant": "q4_0"},
}


def build_needle_prompt(target_tokens, tokenizer):
    """Build a needle-in-haystack prompt with ~target_tokens length."""
    padding_block = (
        "The following document discusses various organizational topics. "
        "Section {n}: Performance metrics for department {n} show steady improvement "
        "in Q3 2025, with productivity up 3.2% and employee satisfaction at 78%. "
        "Budget allocations for the fiscal year were reviewed and approved by the "
        "finance committee on March 15th. Infrastructure upgrades are scheduled for "
        "completion by end of Q4. Training programs will be expanded to cover new "
        "compliance requirements. Resource allocation models suggest optimal staffing "
        "levels will be reached by mid-year. Cross-departmental collaboration "
        "initiatives continue to show positive results across all measured KPIs. "
    )

    blocks = []
    needle_inserted = False
    n = 1
    while True:
        block = padding_block.format(n=n)
        blocks.append(block)
        if not needle_inserted and n > 5:
            blocks.append(f"IMPORTANT NOTE: {NEEDLE_FACT} This information is critical. ")
            needle_inserted = True
        text = "".join(blocks)
        toks = tokenizer.encode(text)
        if len(toks) >= target_tokens - 200:
            break
        n += 1

    if not needle_inserted:
        mid = len(blocks) // 3
        blocks.insert(mid, f"IMPORTANT NOTE: {NEEDLE_FACT} This information is critical. ")

    question = "\n\nQuestion: What is the exact annual budget for Project Alpha? Answer concisely."
    return "".join(blocks) + question


def run_worker(config_json):
    """Run a single benchmark (called in subprocess)."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import stream_generate

    config = json.loads(config_json)
    model_path = config["model_path"]
    target_tokens = config["target_tokens"]
    kv_cache = config.get("kv_cache")
    flat_quant = config.get("flat_quant")
    max_tokens = config.get("max_tokens", 50)

    # Load model
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    # Build prompt
    prompt_text = build_needle_prompt(target_tokens, tokenizer)
    actual_tokens = len(tokenizer.encode(prompt_text))

    # Apply chat template
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    # Reset peak memory
    mx.metal.reset_peak_memory()
    gc.collect()

    # Generate
    kwargs = dict(max_tokens=max_tokens)
    if kv_cache:
        kwargs["kv_cache"] = kv_cache
    if flat_quant:
        kwargs["kv_flat_quant"] = flat_quant

    t0 = time.perf_counter()
    text = ""
    last_response = None
    got_first_token = False
    ttof = 0

    for response in stream_generate(model, tokenizer, formatted, **kwargs):
        if not got_first_token:
            ttof = time.perf_counter() - t0
            got_first_token = True
        text += response.text
        last_response = response

    total_time = time.perf_counter() - t0

    result = {
        "config_name": config["config_name"],
        "target_tokens": target_tokens,
        "actual_tokens": actual_tokens,
        "prompt_tokens": last_response.prompt_tokens if last_response else 0,
        "pp_tps": last_response.prompt_tps if last_response else 0,
        "tg_tps": last_response.generation_tps if last_response else 0,
        "ttof_ms": ttof * 1000,
        "total_time_ms": total_time * 1000,
        "peak_memory_gb": last_response.peak_memory if last_response else 0,
        "text_preview": text[:80],
        "quality": "PASS" if "12,500" in text or "12500" in text else "FAIL",
    }
    print("BENCH_RESULT:" + json.dumps(result))


def run_single_benchmark(config_name, target_tokens, kv_cache, flat_quant):
    """Run a single benchmark in a separate subprocess."""
    config_json = json.dumps({
        "model_path": MODEL_PATH,
        "target_tokens": target_tokens,
        "config_name": config_name,
        "kv_cache": kv_cache,
        "flat_quant": flat_quant,
        "max_tokens": 50,
    })

    ctx_label = f"{target_tokens // 1024}K"
    print(f"\n  --- {ctx_label}_{config_name} ---")
    sys.stdout.flush()

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", config_json],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT (600s)")
        return None

    elapsed = time.time() - t0

    # Parse result
    result = None
    for line in (proc.stdout or "").split("\n"):
        if line.startswith("BENCH_RESULT:"):
            result = json.loads(line[len("BENCH_RESULT:"):])
            break

    if result is None:
        print(f"    FAILED (exit={proc.returncode}, {elapsed:.0f}s)")
        if proc.stderr:
            stderr_lines = proc.stderr.strip().split("\n")
            for line in stderr_lines[-15:]:
                print(f"    ERR: {line}")
        return None

    # Display
    print(f"    Tokens: {result['actual_tokens']}")
    print(f"    PP:     {result['pp_tps']:>8.1f} tok/s  ({result['ttof_ms']:.0f}ms)")
    if result['tg_tps'] > 0:
        print(f"    TG:     {result['tg_tps']:>8.1f} tok/s  ({1000/result['tg_tps']:.1f}ms/tok)")
    print(f"    TTOF:   {result['ttof_ms']:>8.0f} ms")
    print(f"    Peak:   {result['peak_memory_gb']:.2f} GB")
    print(f"    Quality: {result['quality']}")
    print(f"    Text:  {result['text_preview']}")

    return result


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    print("=" * 100)
    print(f"  FlashMLX Hybrid Benchmark — Qwen3.5-35B-A3B (30 SSM + 10 Attention)")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Platform: Apple M4 Pro 24GB")
    print("=" * 100)

    context_lengths = [16384, 32768]
    all_results = []

    for ctx_len in context_lengths:
        ctx_label = f"{ctx_len // 1024}K"
        print(f"\n{'=' * 60}")
        print(f"  Context: {ctx_label}")
        print(f"{'=' * 60}")

        for config_name, params in CONFIGS.items():
            result = run_single_benchmark(
                config_name, ctx_len,
                params["kv_cache"], params["flat_quant"],
            )
            if result:
                result["ctx_label"] = ctx_label
                all_results.append(result)

    # Summary table
    print(f"\n{'=' * 110}")
    print(f"  SUMMARY")
    print(f"{'=' * 110}")
    hdr = f"  {'Config':<18} | {'Ctx':>4} | {'PP tok/s':>9} | {'TG tok/s':>9} | {'TTOF':>8} | {'Peak GB':>8} | {'Qual'}"
    print(hdr)
    print(f"  {'-' * 18}-+-{'-' * 4}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 4}")
    for r in all_results:
        ttof_s = r['ttof_ms'] / 1000
        print(f"  {r['config_name']:<18} | {r['ctx_label']:>4} | {r['pp_tps']:>9.1f} | {r['tg_tps']:>9.1f} | {ttof_s:>7.1f}s | {r['peak_memory_gb']:>7.2f} | {r['quality']}")

    # Delta vs standard
    for ctx_len in context_lengths:
        ctx_label = f"{ctx_len // 1024}K"
        std = next((r for r in all_results if r["config_name"] == "standard" and r["ctx_label"] == ctx_label), None)
        if not std:
            continue
        print(f"\n  Delta vs standard @ {ctx_label}:")
        print(f"  {'Config':<18} | {'PP Δ':>8} | {'TG Δ':>8} | {'TTOF Δ':>8} | {'Qual'}")
        print(f"  {'-' * 18}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 4}")
        for r in all_results:
            if r["ctx_label"] != ctx_label or r["config_name"] == "standard":
                continue
            pp_d = (r["pp_tps"] / std["pp_tps"] - 1) * 100 if std["pp_tps"] > 0 else 0
            tg_d = (r["tg_tps"] / std["tg_tps"] - 1) * 100 if std["tg_tps"] > 0 else 0
            ttof_d = (r["ttof_ms"] / std["ttof_ms"] - 1) * 100 if std["ttof_ms"] > 0 else 0
            print(f"  {r['config_name']:<18} | {pp_d:>+7.1f}% | {tg_d:>+7.1f}% | {ttof_d:>+7.1f}% | {r['quality']}")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".solar", "bench-qwen35-results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")
    print(f"\nDone!")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        idx = sys.argv.index("--worker")
        run_worker(sys.argv[idx + 1])
    else:
        main()
