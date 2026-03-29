#!/usr/bin/env python3
"""
FlashMLX PP (Prompt Processing) Benchmark — Long Context

Accurately measures prefill speed at 1K / 4K / 8K / 16K / 32K tokens.
Uses mx.eval on first token to isolate PP from generation overhead.

Compares: Standard vs Expert Offload (prebuild pool, Regime C).
Loads model only twice (once standard, once offloaded) to save time.
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
MAX_GEN_TOKENS = 10

# Target prompt lengths
PROMPT_LENGTHS = [1024, 4096, 8192, 16384, 32768]

# Filler text — repeated to fill prompt
FILLER_PARAGRAPH = (
    "The rapid advancement of artificial intelligence and machine learning has "
    "fundamentally transformed how we approach complex computational problems. "
    "Neural network architectures, particularly transformer-based models, have "
    "demonstrated remarkable capabilities in natural language understanding, "
    "code generation, mathematical reasoning, and multimodal perception. "
    "The key innovation lies in the self-attention mechanism, which allows the "
    "model to weigh the importance of different parts of the input sequence "
    "when producing each element of the output. This has led to breakthroughs "
    "in machine translation, text summarization, question answering, and many "
    "other tasks that were previously considered extremely challenging for "
    "automated systems. Researchers continue to explore ways to make these "
    "models more efficient, including techniques such as mixture-of-experts "
    "architectures, quantization, pruning, knowledge distillation, and novel "
    "attention mechanisms that reduce the quadratic computational complexity "
    "of standard self-attention. The implications of these developments extend "
    "far beyond the field of computer science, touching areas such as healthcare, "
    "education, scientific discovery, creative arts, and economic forecasting. "
    "As these systems become more capable and accessible, it becomes increasingly "
    "important to consider the ethical implications and ensure that the benefits "
    "of AI are distributed equitably across society. "
)

FINAL_QUESTION = "\n\nBased on the above text, what is the main topic discussed? Answer in one sentence."


def build_long_prompt(tokenizer, target_tokens):
    """Build a prompt that reaches approximately target_tokens length."""
    test_msg = [{"role": "user", "content": "test"}]
    overhead = tokenizer.apply_chat_template(test_msg, add_generation_prompt=True, tokenize=False)
    overhead_tokens = len(tokenizer.encode(overhead))

    question_tokens = len(tokenizer.encode(FINAL_QUESTION))
    content_target = target_tokens - overhead_tokens - question_tokens

    filler_tokens = len(tokenizer.encode(FILLER_PARAGRAPH))
    repeats = max(1, content_target // filler_tokens + 2)

    content = ""
    for i in range(repeats):
        content += FILLER_PARAGRAPH
        if len(tokenizer.encode(content)) >= content_target:
            break

    content += FINAL_QUESTION
    messages = [{"role": "user", "content": content}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    actual_tokens = len(tokenizer.encode(formatted))
    return formatted, actual_tokens


def measure_pp(model, tokenizer, formatted_prompt, prompt_tokens, label="",
               compact_fn=None):
    """Measure PP accurately: time from start to first token yield.

    generate_step internally does mx.eval(y) at n==0 before yielding,
    so next(gen) includes the full prefill + Metal sync.

    compact_fn: optional callable to run AFTER PP, BEFORE TG measurement.
    """
    prompt_array = mx.array(tokenizer.encode(formatted_prompt))

    mx.metal.reset_peak_memory()
    gc.collect()
    mx.eval(mx.zeros(1))  # Warm up Metal

    # ---- Prefill: generate_step does mx.eval(y) before first yield ----
    t0 = time.perf_counter()
    gen = generate_step(prompt_array, model)
    first_token, first_logprobs = next(gen)  # includes prefill + eval
    t1 = time.perf_counter()

    pp_time = t1 - t0
    pp_tps = prompt_tokens / pp_time

    # ---- Optional compact between PP and TG ----
    compact_result = None
    if compact_fn is not None:
        compact_result = compact_fn()

    # ---- Short generation for quality check ----
    t_gen = time.perf_counter()
    ft = first_token if isinstance(first_token, int) else first_token.item()
    tokens_out = [ft]
    for i, (tok, _) in enumerate(gen):
        t = tok if isinstance(tok, int) else tok.item()
        tokens_out.append(t)
        if i + 1 >= MAX_GEN_TOKENS - 1:
            break
    gen_time = time.perf_counter() - t_gen
    gen_tps = len(tokens_out) / gen_time if gen_time > 0 else 0

    peak = mx.metal.get_peak_memory() / 1024 / 1024 / 1024
    text = tokenizer.decode(tokens_out)

    result = {
        "label": label,
        "prompt_tokens": prompt_tokens,
        "pp_tps": pp_tps,
        "pp_time_ms": pp_time * 1000,
        "tg_tps": gen_tps,
        "tg_tokens": len(tokens_out),
        "peak_gb": peak,
        "text": text[:120],
    }
    if compact_result:
        result["compact"] = compact_result
    return result


def main():
    print("=" * 80)
    print("  FlashMLX PP Benchmark — Long Context (1K → 32K)")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Platform: Apple M4 Pro 24GB")
    print(f"  Method: mx.eval(first_token) isolates true prefill time")
    print("=" * 80)

    all_results = []

    # ========================================================
    # Phase 1: Standard inference — all prompts with one model load
    # ========================================================
    print(f"\n{'=' * 60}")
    print("  PHASE 1: Standard Inference")
    print(f"{'=' * 60}")

    print("\n  Loading model (standard)...")
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    gc.collect()
    mem = mx.metal.get_active_memory() / 1024 / 1024 / 1024
    print(f"  Model loaded: {mem:.2f} GB")

    # Pre-build all prompts
    prompts = {}
    for target_len in PROMPT_LENGTHS:
        name = f"{target_len // 1024}K"
        formatted, actual = build_long_prompt(tokenizer, target_len)
        prompts[name] = (formatted, actual)
        print(f"  Built prompt {name}: {actual} tokens")

    for name in prompts:
        formatted, actual = prompts[name]
        print(f"\n  --- Standard {name} ({actual} tokens) ---")
        r = measure_pp(model, tokenizer, formatted, actual, label=f"standard_{name}")
        all_results.append(r)
        print(f"    PP: {r['pp_tps']:.1f} tok/s ({r['pp_time_ms']:.0f}ms)")
        print(f"    TG: {r['tg_tps']:.1f} tok/s | Peak: {r['peak_gb']:.2f} GB")

    del model
    gc.collect()
    mx.metal.clear_cache()

    # ========================================================
    # Phase 2: Expert Offload — all prompts with one model load
    # ========================================================
    print(f"\n{'=' * 60}")
    print("  PHASE 2: Expert Offload (Regime C, prebuild)")
    print(f"{'=' * 60}")

    from mlx_lm.models.expert_offload import patch_model_for_offload

    print("\n  Loading model (offload)...")
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    gc.collect()

    mem_before = mx.metal.get_active_memory() / 1024 / 1024 / 1024
    ctx = patch_model_for_offload(
        model, MODEL_PATH,
        max_workers=4,
        cpu_cache_gb=2.0,
        enable_prefetch=True,
        enable_telemetry=True,
    )
    gc.collect()
    mx.metal.clear_cache()
    mem_after = mx.metal.get_active_memory() / 1024 / 1024 / 1024
    print(f"  Model: {mem_before:.2f} → {mem_after:.2f} GB, Regime: {ctx.regime.regime}")

    first_prompt = True
    for name in prompts:
        formatted, actual = prompts[name]
        print(f"\n  --- Offload {name} ({actual} tokens) ---")

        # First prompt: PP with full pool, then compact to hot-K before TG
        # Subsequent prompts: already compacted pool
        cfn = ctx.compact if first_prompt else None
        r = measure_pp(model, tokenizer, formatted, actual,
                       label=f"offload_{name}", compact_fn=cfn)
        r["regime"] = ctx.regime.regime
        all_results.append(r)
        print(f"    PP: {r['pp_tps']:.1f} tok/s ({r['pp_time_ms']:.0f}ms)")
        print(f"    TG: {r['tg_tps']:.1f} tok/s | Peak: {r['peak_gb']:.2f} GB")
        if r.get("compact"):
            c = r["compact"]
            print(f"    Compact: {c['pool_size']} experts, coverage={c['pp_coverage']:.1%}, "
                  f"memory={c['memory_gb']:.2f} GB ({c['elapsed_ms']:.0f}ms)")
        first_prompt = False

    ctx.close()
    del model
    gc.collect()
    mx.metal.clear_cache()

    # ========================================================
    # Summary
    # ========================================================
    print(f"\n{'=' * 80}")
    print("  SUMMARY — PP Performance (Long Context)")
    print(f"{'=' * 80}")
    hdr = f"  {'Config':<20} | {'Tokens':>7} | {'PP tok/s':>9} | {'PP ms':>8} | {'TG tok/s':>9} | {'Peak GB':>8}"
    print(hdr)
    print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*9}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}")
    for r in all_results:
        print(f"  {r['label']:<20} | {r['prompt_tokens']:>7} | {r['pp_tps']:>9.1f} | "
              f"{r['pp_time_ms']:>7.0f} | {r['tg_tps']:>9.1f} | {r['peak_gb']:>7.2f}")

    # Comparison table
    print(f"\n  {'Prompt':<8} | {'Std PP':>10} | {'Off PP':>10} | {'Delta':>8}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    std_results = [r for r in all_results if r['label'].startswith('standard_')]
    off_results = [r for r in all_results if r['label'].startswith('offload_')]
    for s, o in zip(std_results, off_results):
        name = s['label'].replace('standard_', '')
        delta = (o['pp_tps'] - s['pp_tps']) / s['pp_tps'] * 100 if s['pp_tps'] > 0 else 0
        print(f"  {name:<8} | {s['pp_tps']:>8.1f}/s | {o['pp_tps']:>8.1f}/s | {delta:>+7.1f}%")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".solar", "bench-pp-long-results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
