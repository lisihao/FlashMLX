#!/usr/bin/env python3
"""
TEP Phase B.0 — Pool Miss Quality Audit

Measures how pool misses (speculative K-1 clamping) affect output quality.

Approach:
  1. Standard inference → reference output
  2. Offloaded with large pool (e.g. 128/256) → should match reference
  3. Offloaded with small pool (e.g. 32/256) → more misses, may diverge
  4. Compare outputs and track miss rates at each pool size

Key metrics:
  - Text match ratio (character-level)
  - First divergence position (where outputs start differing)
  - Pool hit rate at each pool size
  - TPOT distribution changes

Usage:
    python3 benchmarks/bench_pool_quality.py [--model PATH] [--tokens N]
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


def generate_text(model, tokenizer, prompt_text: str, max_tokens: int = 200,
                  seed: int = 42) -> dict:
    """Generate text with fixed seed for reproducibility."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    mx.random.seed(seed)
    t0 = time.perf_counter()
    text = ""
    token_count = 0
    tg_tps = 0

    for response in stream_generate(model, tokenizer, formatted,
                                     max_tokens=max_tokens):
        text += response.text
        token_count += 1
        tg_tps = response.generation_tps

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024 / 1024 / 1024

    return {
        "text": text,
        "tokens": token_count,
        "tg_tps": tg_tps,
        "total_ms": total * 1000,
        "peak_gb": peak,
    }


def compare_texts(reference: str, candidate: str) -> dict:
    """Compare two generated texts."""
    min_len = min(len(reference), len(candidate))
    max_len = max(len(reference), len(candidate))

    if max_len == 0:
        return {"match_ratio": 1.0, "first_diverge_char": -1, "first_diverge_word": -1}

    # Character-level match
    matches = sum(1 for a, b in zip(reference, candidate) if a == b)
    match_ratio = matches / max_len if max_len > 0 else 1.0

    # First divergence position
    first_diverge = -1
    for i in range(min_len):
        if reference[i] != candidate[i]:
            first_diverge = i
            break
    if first_diverge == -1 and len(reference) != len(candidate):
        first_diverge = min_len

    # Word-level divergence
    ref_words = reference.split()
    cand_words = candidate.split()
    first_diverge_word = -1
    for i in range(min(len(ref_words), len(cand_words))):
        if ref_words[i] != cand_words[i]:
            first_diverge_word = i
            break
    if first_diverge_word == -1 and len(ref_words) != len(cand_words):
        first_diverge_word = min(len(ref_words), len(cand_words))

    return {
        "match_ratio": match_ratio,
        "first_diverge_char": first_diverge,
        "first_diverge_word": first_diverge_word,
        "ref_len": len(reference),
        "cand_len": len(candidate),
    }


def run_standard(model_path: str, prompt: str, max_tokens: int) -> dict:
    """Run standard inference as reference."""
    print("\n  Loading model (standard)...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    mx.metal.reset_peak_memory()
    result = generate_text(model, tokenizer, prompt, max_tokens)
    result["label"] = "standard"
    result["pool_size"] = "N/A"

    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    return result


def _find_inner_model(model):
    """Walk model hierarchy to find the layers container."""
    inner = model
    for attr in ("model", "model", "language_model", "model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    return inner


def _get_switch_layers(model):
    """Get all FlashMoeSwitchGLU layers."""
    from mlx_lm.models.expert_offload import FlashMoeSwitchGLU
    inner = _find_inner_model(model)
    switches = []
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if isinstance(sw, FlashMoeSwitchGLU):
                switches.append(sw)
    return switches


def run_offloaded(model_path: str, prompt: str, max_tokens: int,
                  pool_size: int, cpu_cache_gb: float, label: str) -> dict:
    """Run with expert offloading — production-realistic quality test.

    Simulates real 16GB device behavior:
      - PP phase: full pool (all experts correct) — identity mode
      - Compact after PP: shrink pool to target size
      - TG phase: compact pool with K-1 clamping for misses

    This is done by letting the first token (PP) run with identity pool,
    then compacting before remaining TG tokens.
    """
    from mlx_lm.models.expert_offload import patch_model_for_offload, FlashMoeSwitchGLU

    print(f"\n  Loading model (offload pool={pool_size})...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, model_path,
        pool_size=pool_size,
        max_workers=4,
        cpu_cache_gb=cpu_cache_gb,
        enable_prefetch=False,
        enable_telemetry=True,
    )
    gc.collect()

    # Format prompt
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    mx.random.seed(42)
    mx.metal.reset_peak_memory()
    t0 = time.perf_counter()

    text = ""
    token_count = 0
    tg_tps = 0
    compacted = False

    for response in stream_generate(model, tokenizer, formatted,
                                     max_tokens=max_tokens):
        token_count += 1

        # After first token (PP complete), compact pool for TG
        # disable_coverage_gate=True: test actual pool_size, not auto-expanded
        if not compacted:
            ctx.compact(pool_size=pool_size, disable_coverage_gate=True)
            gc.collect()
            switches = _get_switch_layers(model)
            if switches:
                sw0 = switches[0]
                print(f"  PP done. Compacted → {len(sw0._pool_expert_ids)}/256 experts")
            compacted = True

        text += response.text
        tg_tps = response.generation_tps

    total = time.perf_counter() - t0
    peak = mx.metal.get_peak_memory() / 1024 / 1024 / 1024

    result = {
        "text": text,
        "tokens": token_count,
        "tg_tps": tg_tps,
        "total_ms": total * 1000,
        "peak_gb": peak,
        "label": label,
        "pool_size": pool_size,
    }

    # Telemetry — pool hit/miss stats
    if ctx.telemetry:
        tel = ctx.telemetry.summary()
        result["pool_hit_rate"] = tel["overall_pool_hit_rate"]
        result["per_layer_hit_rate"] = tel.get("per_layer_hit_rate", {})
        miss_lat = tel.get("miss_latency", {})
        result["miss_count"] = miss_lat.get("count", 0)
        result["miss_latency"] = miss_lat

    # Cross-validation: compute hit rate directly from switch layers' TG buffers
    switches = _get_switch_layers(model)
    if switches:
        total_hits, total_misses = 0, 0
        for sw in switches:
            pool_set_np = np.array(sw._pool_expert_ids, dtype=np.int64)
            for buf in sw._tg_indices_buffer:
                ids = np.array(buf, copy=False).astype(np.int64)
                hits = np.isin(ids, pool_set_np).sum()
                total_hits += int(hits)
                total_misses += len(ids) - int(hits)
        total = total_hits + total_misses
        result["cross_val_hit_rate"] = total_hits / total if total > 0 else None
        result["cross_val_total_activations"] = total
        result["cross_val_misses"] = total_misses

    ctx.close()
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    mx.metal.set_cache_limit(0)
    return result


def main():
    parser = argparse.ArgumentParser(description="TEP Phase B.0 — Pool Miss Quality Audit")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit",
                        help="Path to MoE model")
    parser.add_argument("--tokens", type=int, default=150,
                        help="Max tokens to generate")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP Phase B.0 — Pool Miss Quality Audit")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.tokens}")
    print("=" * 70)

    prompt = (
        "Explain the mathematical proof of the Pythagorean theorem "
        "using the area method, step by step."
    )

    pool_sizes = [64, 32, 16, 8]
    all_results = []

    # 1. Standard reference
    print(f"\n{'=' * 60}")
    print(f"  Reference: Standard (no offloading)")
    print(f"{'=' * 60}")
    ref = run_standard(args.model, prompt, args.tokens)
    all_results.append(ref)
    print(f"  TG: {ref['tg_tps']:.1f} tok/s | Peak: {ref['peak_gb']:.2f} GB")
    print(f"  Text: {ref['text'][:80]}...")

    # 2. Offloaded at different pool sizes (no CPU cache to avoid memory bloat)
    for ps in pool_sizes:
        print(f"\n{'=' * 60}")
        print(f"  Pool size: {ps}/256 ({ps*100/256:.0f}%)")
        print(f"{'=' * 60}")
        r = run_offloaded(args.model, prompt, args.tokens,
                          pool_size=ps, cpu_cache_gb=0.0,
                          label=f"pool_{ps}")
        comp = compare_texts(ref["text"], r["text"])
        r["comparison"] = comp
        all_results.append(r)

        hit_rate = r.get("pool_hit_rate", "N/A")
        hit_str = f"{hit_rate:.2%}" if isinstance(hit_rate, float) else hit_rate
        cv_hit = r.get("cross_val_hit_rate")
        cv_str = f"{cv_hit:.2%}" if cv_hit is not None else "N/A"
        cv_miss = r.get("cross_val_misses", "?")
        print(f"  TG: {r['tg_tps']:.1f} tok/s | Peak: {r['peak_gb']:.2f} GB")
        print(f"  Pool hit (telemetry): {hit_str} | Pool hit (cross-val): {cv_str}")
        print(f"  Cross-val misses: {cv_miss} / {r.get('cross_val_total_activations', '?')}")
        print(f"  Match: {comp['match_ratio']:.2%} | "
              f"First diverge: char {comp['first_diverge_char']}, "
              f"word {comp['first_diverge_word']}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  QUALITY IMPACT SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Pool':>6} {'Telem HR':>10} {'CV HR':>10} {'TG tok/s':>10} "
          f"{'Match%':>8} {'Diverge@':>10} {'Verdict':>12}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*12}")

    for r in all_results:
        comp = r.get("comparison", {})
        hit_rate = r.get("pool_hit_rate", 1.0)
        hit_str = f"{hit_rate:.2%}" if isinstance(hit_rate, float) else "N/A"
        cv_hit = r.get("cross_val_hit_rate")
        cv_str = f"{cv_hit:.2%}" if cv_hit is not None else "N/A"
        match = comp.get("match_ratio", 1.0)
        diverge = comp.get("first_diverge_word", -1)
        div_str = f"word {diverge}" if diverge >= 0 else "never"

        if match >= 0.99:
            verdict = "IDENTICAL"
        elif match >= 0.90:
            verdict = "MINOR"
        elif match >= 0.70:
            verdict = "MODERATE"
        else:
            verdict = "SEVERE"

        if r["label"] == "standard":
            verdict = "REFERENCE"

        print(f"  {str(r['pool_size']):>6} {hit_str:>10} {cv_str:>10} {r['tg_tps']:>10.1f} "
              f"{match:>7.1%} {div_str:>10} {verdict:>12}")

    # Verdict
    worst_match = min(r.get("comparison", {}).get("match_ratio", 1.0)
                      for r in all_results[1:])
    if worst_match >= 0.95:
        print(f"\n  VERDICT: Pool misses have NEGLIGIBLE quality impact")
        print(f"  → Speculative K-1 execution is sufficient")
        print(f"  → TEP predictive prefetch has LOW value for quality")
    elif worst_match >= 0.80:
        print(f"\n  VERDICT: Pool misses cause MODERATE quality degradation")
        print(f"  → TEP predictive prefetch can improve quality at small pool sizes")
    else:
        print(f"\n  VERDICT: Pool misses cause SEVERE quality degradation")
        print(f"  → TEP predictive prefetch is HIGH VALUE")
        print(f"  → Reducing miss rate directly improves output quality")

    # Save
    report = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tokens": args.tokens,
        "prompt": prompt,
        "results": all_results,
    }

    out_path = args.output
    if not out_path:
        solar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".solar")
        os.makedirs(solar_dir, exist_ok=True)
        out_path = os.path.join(solar_dir, "tep-pool-quality.json")

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
