#!/usr/bin/env python3
"""
Gemma 4 long-context compression benchmark.

Measures PP/TG throughput and peak memory for FlashMLX compression strategies
(standard / triple_pq / scored_pq) at 8K / 16K / 32K context lengths on a
Gemma 4 hybrid attention model (sliding + full_attention).

Usage:
    export GEMMA4_MODEL_PATH=/path/to/gemma-4-31B
    python examples/bench_gemma4_long_context.py

Architecture recap (gemma-4-31B):
  - 60 layers: 50 sliding_attention + 10 full_attention (pattern [S,S,S,S,S,F] x 10)
  - Sliding layers use RotatingKVCache (capped at sliding_window=1024)
  - Full-attention layers use KVCache (grows with context)
  - Per full_attention layer: 4 KV heads x 512 head_dim = 4 KB/token (bf16)
  - Compressible KV budget at 32K: ~2.5 GB (10 layers x 8 KB/tok x 32K)

Findings (see docs/gemma4-integration-report.md):
  - standard / scored_pq behave identically (scored_pq falls back without calibration)
  - triple_pq regresses severely at 32K (TG -49%, peak +19%); use with caution
  - Gemma 4 is NOT a great target for KV compression because full_attention layers
    already have very narrow KV heads (4) - compression ceiling is small.
"""
import os
import sys
import time
import gc
import traceback
from pathlib import Path

FLASHMLX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FLASHMLX_ROOT / "mlx-vlm-source"))
sys.path.insert(0, str(FLASHMLX_ROOT / "mlx-lm-source"))
sys.path.insert(0, str(FLASHMLX_ROOT / "src"))

# Clear any pre-imported mlx_vlm to force use of the local fork
for m in [k for k in list(sys.modules.keys()) if k.startswith("mlx_vlm")]:
    del sys.modules[m]

import mlx.core as mx
import mlx_vlm
assert "FlashMLX/mlx-vlm-source" in mlx_vlm.__file__, (
    f"Expected local mlx-vlm-source fork, got {mlx_vlm.__file__}"
)
print(f"[INFO] mlx_vlm: {mlx_vlm.__file__}")

from mlx_vlm import load, generate
from flashmlx.vlm_bridge import create_vlm_cache

MODEL_PATH = os.environ.get("GEMMA4_MODEL_PATH")
if not MODEL_PATH:
    print("ERROR: Set GEMMA4_MODEL_PATH to a local Gemma 4 model directory.")
    print("Example: export GEMMA4_MODEL_PATH=/path/to/gemma-4-31B")
    sys.exit(1)

MAX_TOKENS = 30           # small generation window; focus on prefill + KV growth
TARGET_LENGTHS = [8192, 16384, 32768]
STRATEGIES = ["standard", "triple_pq", "scored_pq"]

# Realistic long document base (~250 tokens)
BASE_PARAGRAPH = """
Machine learning on Apple Silicon benefits from MLX's unified memory design, which
eliminates the CPU/GPU copy cost that dominates wall time on many discrete-GPU
frameworks. Because MLX arrays live in a shared address space, operations like
attention and matrix multiplication can dispatch to either the CPU, GPU, or Neural
Engine without synchronous copies. This architectural choice is particularly
important for large language models, where each decoding step reads the full
key-value cache and writes a new key and value vector. Traditional approaches that
rely on explicit device transfers become a bottleneck as context length grows into
the tens of thousands of tokens. The compressed KV cache techniques developed in
FlashMLX further reduce memory pressure, allowing long contexts to fit within the
unified memory budget while preserving numerical fidelity through adaptive
quantization and polar-quant bucketing. In practice, this means a 32K-token
conversation can run on consumer hardware without swapping or OOM crashes.
""".strip()

QUESTION = "\n\nBased on the full discussion above, summarize the three most important reasons why MLX's unified memory matters for long-context inference."


def build_prompt_to_length(tokenizer, target_tokens: int) -> tuple[str, int]:
    """Repeat BASE_PARAGRAPH until the tokenized prompt has >= target_tokens,
    then append the question. Returns (prompt_str, actual_token_count)."""
    base_ids = tokenizer.encode(BASE_PARAGRAPH)
    q_ids = tokenizer.encode(QUESTION)
    repeats = max(1, (target_tokens - len(q_ids)) // len(base_ids))
    body_parts = [f"[Section {i+1}]\n{BASE_PARAGRAPH}" for i in range(repeats)]
    body = "\n\n".join(body_parts)
    prompt = body + QUESTION
    actual = len(tokenizer.encode(prompt))
    while actual < target_tokens - 200:
        body_parts.append(f"[Section {len(body_parts)+1}]\n{BASE_PARAGRAPH}")
        body = "\n\n".join(body_parts)
        prompt = body + QUESTION
        actual = len(tokenizer.encode(prompt))
    return prompt, actual


try:
    mx.clear_cache()
except AttributeError:
    mx.metal.clear_cache()

print(f"\n[LOAD] {MODEL_PATH} ...")
t0 = time.time()
model, processor = load(MODEL_PATH)
print(f"[LOAD] Done in {time.time()-t0:.1f}s")
print(f"[MEM]  Active: {mx.metal.get_active_memory()/1024**3:.2f} GB")

tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

# Validate native cache composition once
native = model.language_model.make_cache()
kvc = sum(1 for c in native if type(c).__name__ == "KVCache")
rot = sum(1 for c in native if type(c).__name__ == "RotatingKVCache")
print(f"[INSPECT] Native cache: {len(native)} total, {kvc} KVCache (compressible), "
      f"{rot} RotatingKVCache (sliding)")


def run_single(strategy: str, prompt: str, actual_tokens: int, target: int) -> dict:
    label = f"{strategy}@{target//1024}K"
    print(f"\n  -> {label}  (actual {actual_tokens} tok)")

    try:
        mx.metal.reset_peak_memory()
    except Exception:
        pass

    try:
        cache = create_vlm_cache(model, strategy=strategy)
    except Exception as e:
        print(f"     Cache creation failed: {e}")
        traceback.print_exc()
        return {"label": label, "strategy": strategy, "target": target, "error": str(e)}

    type_counts = {}
    for c in cache:
        type_counts[type(c).__name__] = type_counts.get(type(c).__name__, 0) + 1

    mem_before = mx.metal.get_active_memory() / 1024**3
    t0 = time.time()
    try:
        out = generate(
            model, processor, prompt,
            max_tokens=MAX_TOKENS, temp=0.0, verbose=False,
            prompt_cache=cache,
        )
    except Exception as e:
        print(f"     Generation failed: {e}")
        traceback.print_exc()
        return {"label": label, "strategy": strategy, "target": target, "error": str(e),
                "type_counts": type_counts}

    elapsed = time.time() - t0
    mem_after = mx.metal.get_active_memory() / 1024**3
    peak = mx.metal.get_peak_memory() / 1024**3

    prompt_tps = float(getattr(out, "prompt_tps", 0))
    gen_tps = float(getattr(out, "generation_tps", MAX_TOKENS / elapsed))
    gen_tokens = int(getattr(out, "generation_tokens", MAX_TOKENS))
    text = getattr(out, "text", str(out))[:160]

    kv_delta_gb = peak - mem_before
    print(f"     PP={prompt_tps:7.1f}  TG={gen_tps:5.2f} tok/s  "
          f"peak={peak:5.2f} GB  KVd={kv_delta_gb:+.2f} GB  elapsed={elapsed:.1f}s")
    print(f"        cache: {type_counts}")
    print(f"        text:  {text!r}")

    del cache
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        mx.metal.clear_cache()

    return {
        "label": label,
        "strategy": strategy,
        "target": target,
        "actual_tokens": actual_tokens,
        "prompt_tps": prompt_tps,
        "gen_tps": gen_tps,
        "gen_tokens": gen_tokens,
        "elapsed": elapsed,
        "peak_mem_gb": peak,
        "mem_before_gb": mem_before,
        "mem_after_gb": mem_after,
        "kv_delta_gb": kv_delta_gb,
        "type_counts": type_counts,
        "text": text,
    }


# ---- Run benchmark matrix ----
results = []
for target in TARGET_LENGTHS:
    print(f"\n{'='*70}\n[CONTEXT {target//1024}K] building prompt ...")
    prompt, actual = build_prompt_to_length(tokenizer, target)
    print(f"[CONTEXT {target//1024}K] actual prompt tokens: {actual}")

    for strategy in STRATEGIES:
        r = run_single(strategy, prompt, actual, target)
        results.append(r)

# ---- Summary ----
print(f"\n{'#'*70}\n# LONG-CONTEXT SUMMARY\n{'#'*70}")
print(f"{'Context':>8} {'Strategy':>12} {'PP tok/s':>10} {'TG tok/s':>10} "
      f"{'Peak GB':>9} {'KVd GB':>9}  Notes")
print("-"*80)

by_context = {}
for r in results:
    by_context.setdefault(r["target"], []).append(r)

for target in sorted(by_context):
    rows = by_context[target]
    base = next((r for r in rows if r["strategy"] == "standard" and "error" not in r), None)
    for r in rows:
        if "error" in r:
            print(f"{r['target']//1024:>6}K {r['strategy']:>12} {'ERROR':>10} {r['error'][:40]}")
            continue
        if base and r is not base:
            dpp = (r["prompt_tps"] / base["prompt_tps"] - 1) * 100 if base["prompt_tps"] else 0
            dtg = (r["gen_tps"] / base["gen_tps"] - 1) * 100 if base["gen_tps"] else 0
            dmem = (r["peak_mem_gb"] / base["peak_mem_gb"] - 1) * 100 if base["peak_mem_gb"] else 0
            note = f"PP{dpp:+.0f}% TG{dtg:+.0f}% peak{dmem:+.0f}%"
        else:
            note = "baseline"
        print(f"{r['target']//1024:>6}K {r['strategy']:>12} "
              f"{r['prompt_tps']:>10.1f} {r['gen_tps']:>10.2f} "
              f"{r['peak_mem_gb']:>9.2f} {r['kv_delta_gb']:>+9.2f}  {note}")

print(f"\n{'#'*70}\n# OUTPUT CONSISTENCY (per context length)\n{'#'*70}")
for target in sorted(by_context):
    texts = [r["text"] for r in by_context[target] if "text" in r]
    unique = set(texts)
    status = "identical" if len(unique) == 1 else f"{len(unique)} variants"
    print(f"  {target//1024}K: {status}")
    if len(unique) > 1:
        for r in by_context[target]:
            if "text" in r:
                print(f"    {r['strategy']:>12}: {r['text'][:80]!r}")

print("\n[DONE]")
