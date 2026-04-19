#!/usr/bin/env python3
"""Quality benchmark: MATH-500 accuracy for expert offload configs.

Tests whether expert offloading actually hurts model *capability*,
not just char-match. Runs N MATH problems, extracts boxed answers,
compares to ground truth.

Configs:
  A: standard (no offload, full 6-bit)
  B: pool=32 + zero_out + rerank (5 GB, fastest)
  C: pool=64 + zero_out + rerank (8 GB, highest HR)
  D: pool=32 + full shadow (4-bit) + rerank (~22 GB, same-expert fallback)
  E: pool=64 + full shadow (4-bit) + rerank (~25 GB, same-expert fallback)
"""

import argparse
import gc
import json
import re
import sys
import time

sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.expert_offload import patch_model_for_offload


def extract_boxed(text: str) -> str:
    """Extract \\boxed{...} answer from model output."""
    # Find last \boxed{...}
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        return matches[-1].strip()
    # Fallback: look for "the answer is X" patterns
    m = re.search(r'(?:answer|result)\s*(?:is|=)\s*[\\$]*([^\s,.$]+)', text, re.I)
    if m:
        return m.group(1).strip()
    return ""


def normalize_answer(ans: str) -> str:
    """Normalize math answer for comparison."""
    ans = ans.strip()
    # Remove LaTeX formatting
    ans = ans.replace('\\$', '').replace('$', '')
    ans = ans.replace('\\text{', '').replace('}', '')
    ans = ans.replace('\\mathrm{', '').replace('\\', '')
    ans = ans.replace(' ', '')
    # Remove trailing period
    ans = ans.rstrip('.')
    # Try to normalize fractions
    m = re.match(r'^(\d+)/(\d+)$', ans)
    if m:
        try:
            from fractions import Fraction
            f = Fraction(int(m.group(1)), int(m.group(2)))
            ans = f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)
        except:
            pass
    return ans


def run_problem(model, tokenizer, problem: dict, max_tokens: int = 1024) -> dict:
    """Run a single MATH problem and check answer."""
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content":
              f"Solve this math problem step by step, then put your final answer "
              f"in \\boxed{{}}.\n\n{problem['problem']}"}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content":
              f"Solve this math problem step by step, then put your final answer "
              f"in \\boxed{{}}.\n\n{problem['problem']}"}],
            add_generation_prompt=True, tokenize=False,
        )

    parts = []
    for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        parts.append(r.text)
    text = "".join(parts)

    predicted = normalize_answer(extract_boxed(text))
    expected = normalize_answer(problem["answer"])
    correct = predicted == expected

    return {
        "problem_id": problem["problem_id"],
        "subject": problem.get("subject", ""),
        "correct": correct,
        "predicted": predicted,
        "expected": expected,
    }


def run_config(model_path, problems, max_tokens, label, setup_fn):
    """Run all problems with a specific config."""
    print(f"\n  [{label}] Loading model...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = setup_fn(model, model_path, tokenizer)

    correct = 0
    total = len(problems)
    results = []

    for i, prob in enumerate(problems):
        t0 = time.perf_counter()
        result = run_problem(model, tokenizer, prob, max_tokens)
        elapsed = time.perf_counter() - t0
        results.append(result)

        if result["correct"]:
            correct += 1
        status = "✓" if result["correct"] else "✗"
        print(f"    [{i+1}/{total}] {status} {result['subject'][:12]:12s} "
              f"pred={result['predicted'][:20]:20s} "
              f"exp={result['expected'][:20]:20s} "
              f"({elapsed:.1f}s)")
        # Prevent metal buffer accumulation across problems
        # flush_tg_telemetry clears _tg_indices_buffer/_tg_scores_buffer
        # which hold mx.array refs that pin metal buffers
        if ctx and hasattr(ctx, 'flush_tg_telemetry'):
            ctx.flush_tg_telemetry()
        gc.collect()
        mx.clear_cache()

    accuracy = correct / total if total > 0 else 0
    print(f"  [{label}] Accuracy: {correct}/{total} = {accuracy:.1%}")

    if ctx:
        ctx.close()
    del model, tokenizer
    gc.collect()
    mx.clear_cache()

    return {"label": label, "correct": correct, "total": total,
            "accuracy": accuracy, "results": results}


def setup_standard(model, model_path, tokenizer):
    """No offload."""
    return None


def _setup_offload(model, model_path, tokenizer, pool_size):
    """Common offload setup for pool=N + zero_out + rerank."""
    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()

    # PP warmup
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    ctx.compact(pool_size=pool_size, disable_coverage_gate=True,
                auto_expand_cpu_cache=False)

    # Set k1_clamp for TG warmup
    inner = model
    for attr in ("model", "model", "language_model", "model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy"):
                sw._pool_is_identity = False
                sw._pool_compacted = True
                sw._miss_policy = "k1_clamp"

    # TG warmup
    warmup = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 20."}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, warmup, max_tokens=15):
        pass

    ctx.decode_recompact(pool_size=pool_size)

    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy"):
                sw._miss_policy = "zero_out"

    ctx.enable_reranking(bonus=0.01)
    return ctx


def setup_pool32_rr(model, model_path, tokenizer):
    """pool=32 + zero_out + rerank."""
    return _setup_offload(model, model_path, tokenizer, pool_size=32)


def setup_pool64_rr(model, model_path, tokenizer):
    """pool=64 + zero_out + rerank."""
    return _setup_offload(model, model_path, tokenizer, pool_size=64)


def _setup_offload_shadow(model, model_path, tokenizer, pool_size,
                           shadow_bits=4, enable_rerank=True):
    """pool=N + full shadow (N-bit, all 256 experts) + miss_policy=shadow.

    Same-expert fallback: pool miss uses the SAME expert at lower precision
    instead of zeroing the output. This is the core FTEC hypothesis.
    """
    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()

    # PP warmup
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    ctx.compact(pool_size=pool_size, disable_coverage_gate=True,
                auto_expand_cpu_cache=False)

    # Create full shadow (all 256 experts) BEFORE setting miss policy
    print(f"  [{pool_size}] Creating full shadow ({shadow_bits}-bit, all experts)...")
    ctx.create_shadow(bits=shadow_bits)

    # Set shadow miss policy (not k1_clamp)
    inner = model
    for attr in ("model", "model", "language_model", "model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy"):
                sw._pool_is_identity = False
                sw._pool_compacted = True
                sw._miss_policy = "shadow"

    # TG warmup with shadow policy
    warmup = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 20."}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, warmup, max_tokens=15):
        pass

    ctx.decode_recompact(pool_size=pool_size)

    # Keep shadow policy after recompact
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy"):
                sw._miss_policy = "shadow"

    if enable_rerank:
        ctx.enable_reranking(bonus=0.01)
    return ctx


def setup_pool32_shadow(model, model_path, tokenizer):
    """pool=32 + full shadow (4-bit) + rerank."""
    return _setup_offload_shadow(model, model_path, tokenizer, pool_size=32)


def setup_pool64_shadow(model, model_path, tokenizer):
    """pool=64 + full shadow (4-bit) + rerank."""
    return _setup_offload_shadow(model, model_path, tokenizer, pool_size=64)


def setup_pool32_shadow6(model, model_path, tokenizer):
    """pool=32 + full shadow (6-bit = same as model) + rerank."""
    return _setup_offload_shadow(model, model_path, tokenizer,
                                 pool_size=32, shadow_bits=6)


def setup_pool32_shadow6_norr(model, model_path, tokenizer):
    """pool=32 + full shadow (6-bit) + NO rerank.

    Isolates rerank effect on quality.
    """
    return _setup_offload_shadow(model, model_path, tokenizer,
                                 pool_size=32, shadow_bits=6,
                                 enable_rerank=False)


def setup_pool32_poolshadow(model, model_path, tokenizer):
    """pool=32 + shadow from POOL tensor (pre-compact, identical data).

    Shadow is created from the same full pool tensor before compact,
    guaranteeing byte-identical data. Tests if loader inconsistency
    is causing the quality gap.
    """
    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()

    # PP warmup
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    # Create shadow FROM THE FULL POOL (before compact)
    print("  Creating shadow from full pool (pre-compact, same data)...")
    inner = model
    for attr in ("model", "model", "language_model", "model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)

    shadow_bytes = 0
    shadow_layers = 0
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_pool") and sw._pool is not None:
                # Copy pool data directly as shadow (same bits, same data)
                sw._shadow = {k: v for k, v in sw._pool.items()}
                sw._shadow_bits = sw.bits
                for v in sw._shadow.values():
                    shadow_bytes += v.nbytes
                shadow_layers += 1
    print(f"  Shadow from pool: {shadow_layers} layers, "
          f"{shadow_bytes / 1024**3:.2f} GB")

    # NOW compact
    ctx.compact(pool_size=32, disable_coverage_gate=True,
                auto_expand_cpu_cache=False)

    # Set shadow miss policy
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy"):
                sw._pool_is_identity = False
                sw._pool_compacted = True
                sw._miss_policy = "shadow"

    # TG warmup
    warmup = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 20."}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, warmup, max_tokens=15):
        pass

    ctx.decode_recompact(pool_size=32)

    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy"):
                sw._miss_policy = "shadow"

    ctx.enable_reranking(bonus=0.01)
    return ctx


def _setup_offload_ftec(model, model_path, tokenizer, pool_size,
                         shadow_size=64, shadow_bits=6,
                         guard_j=2, guard_tau=0.02):
    """pool=N + decode-hot shadow (top-M non-pool experts) + ftec 3-way dispatch.

    FTEC: pool hit → full precision, pool miss + shadow hit → shadow precision,
    pool miss + shadow miss → zero.  Much smaller memory than full shadow.
    """
    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()

    # PP warmup
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    ctx.compact(pool_size=pool_size, disable_coverage_gate=True,
                auto_expand_cpu_cache=False)

    # Set k1_clamp for TG warmup to gather frequency data
    inner = model
    for attr in ("model", "model", "language_model", "model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy"):
                sw._pool_is_identity = False
                sw._pool_compacted = True
                sw._miss_policy = "k1_clamp"

    # TG warmup — generates frequency data for decode_shadow
    warmup = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 20."}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, warmup, max_tokens=15):
        pass

    ctx.decode_recompact(pool_size=pool_size)

    # Create decode-hot shadow from TG frequency (AFTER recompact has freq data)
    print(f"  [{pool_size}] Creating decode shadow ({shadow_size} experts, "
          f"{shadow_bits}-bit)...")
    ctx.create_decode_shadow(size=shadow_size, bits=shadow_bits)

    # Set ftec miss policy
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy"):
                sw._miss_policy = "ftec"

    ctx.enable_reranking(bonus=0.01, guard_j=guard_j, guard_tau=guard_tau)
    return ctx


def setup_pool32_ftec64(model, model_path, tokenizer):
    """pool=32 + decode-hot shadow (64, 6-bit) + ftec + guarded rerank."""
    return _setup_offload_ftec(model, model_path, tokenizer,
                                pool_size=32, shadow_size=64, shadow_bits=6)


def setup_pool32_ftec128(model, model_path, tokenizer):
    """pool=32 + decode-hot shadow (128, 6-bit) + ftec + guarded rerank."""
    return _setup_offload_ftec(model, model_path, tokenizer,
                                pool_size=32, shadow_size=128, shadow_bits=6)


def setup_pool32_ftec64_4bit(model, model_path, tokenizer):
    """pool=32 + decode-hot shadow (64, 4-bit) + ftec + guarded rerank."""
    return _setup_offload_ftec(model, model_path, tokenizer,
                                pool_size=32, shadow_size=64, shadow_bits=4)


def setup_pool32_ftec224(model, model_path, tokenizer):
    """pool=32 + decode-hot shadow (224 = all non-pool, 6-bit) + ftec.

    Near-full coverage test: if this works (~14/20), then FTEC dispatch code
    is correct and partial shadow failure is purely a coverage issue.
    """
    return _setup_offload_ftec(model, model_path, tokenizer,
                                pool_size=32, shadow_size=224, shadow_bits=6)


def setup_pool32_shadow6_ftec(model, model_path, tokenizer):
    """pool=32 + full shadow (6-bit) + FTEC dispatch + unguarded rerank.

    Tests FTEC dispatch code in isolation: uses full shadow data (which works
    with "shadow" policy at 14-15/20) but switches to "ftec" dispatch.
    This isolates whether 0/20 is caused by the dispatch code or the setup.
    """
    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()

    # PP warmup
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    ctx.compact(pool_size=32, disable_coverage_gate=True,
                auto_expand_cpu_cache=False)

    # Create full shadow (all 256 experts)
    print(f"  [32] Creating full shadow (6-bit, all experts)...")
    ctx.create_shadow(bits=6)

    # Navigate to layers
    inner = model
    for attr in ("model", "model", "language_model", "model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)

    # Set up FTEC metadata for full shadow:
    # _shadow_expert_ids = all 256 experts, _shadow_remap = identity
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy") and sw._shadow is not None:
                sw._pool_is_identity = False
                sw._pool_compacted = True
                # Set FTEC dispatch with full shadow (identity remap)
                all_ids = list(range(sw.num_experts))
                sw._shadow_expert_ids = all_ids
                import numpy as np
                remap_np = np.arange(sw.num_experts, dtype=np.int32)
                sw._shadow_remap = mx.array(remap_np)
                sw._shadow_remap_np = remap_np
                sw._miss_policy = "ftec"

    # TG warmup with ftec policy
    warmup = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 20."}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, warmup, max_tokens=15):
        pass

    ctx.decode_recompact(pool_size=32)

    # Re-set ftec policy and metadata after recompact
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy") and sw._shadow is not None:
                all_ids = list(range(sw.num_experts))
                sw._shadow_expert_ids = all_ids
                import numpy as np
                remap_np = np.arange(sw.num_experts, dtype=np.int32)
                sw._shadow_remap = mx.array(remap_np)
                sw._shadow_remap_np = remap_np
                sw._miss_policy = "ftec"

    # Unguarded rerank (same as shadow policy uses)
    ctx.enable_reranking(bonus=0.01)
    return ctx


def setup_pool32_ftec224_noguard(model, model_path, tokenizer):
    """pool=32 + decode-hot shadow (224 = all non-pool, 6-bit) + NO guard.

    Same as N but without guarded rerank. Tests if guard_j=2 is the problem.
    """
    return _setup_offload_ftec(model, model_path, tokenizer,
                                pool_size=32, shadow_size=224, shadow_bits=6,
                                guard_j=0, guard_tau=0.02)


def setup_pool32_k1warm_fullshadow(model, model_path, tokenizer):
    """K1_CLAMP warmup → decode_recompact → FULL shadow + ftec dispatch.

    Tests if k1_clamp warmup corrupts the pool composition via
    decode_recompact. Uses full shadow so coverage is 100%.
    If 0/20: k1_clamp warmup is the problem (bad pool).
    If ~75%: the problem is coverage, not the warmup.
    """
    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()

    # PP warmup
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    ctx.compact(pool_size=32, disable_coverage_gate=True,
                auto_expand_cpu_cache=False)

    # K1_CLAMP warmup (same as _setup_offload_ftec)
    inner = model
    for attr in ("model", "model", "language_model", "model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy"):
                sw._pool_is_identity = False
                sw._pool_compacted = True
                sw._miss_policy = "k1_clamp"

    warmup = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 20."}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, warmup, max_tokens=15):
        pass

    ctx.decode_recompact(pool_size=32)

    # NOW use FULL shadow (all 256 experts) + ftec dispatch
    print(f"  [Q] Creating FULL shadow (6-bit) after k1_clamp warmup...")
    ctx.create_shadow(bits=6)

    # Set ftec metadata with identity remap
    import numpy as np
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy") and sw._shadow is not None:
                all_ids = list(range(sw.num_experts))
                sw._shadow_expert_ids = all_ids
                remap_np = np.arange(sw.num_experts, dtype=np.int32)
                sw._shadow_remap = mx.array(remap_np)
                sw._shadow_remap_np = remap_np
                sw._miss_policy = "ftec"

    ctx.enable_reranking(bonus=0.01)
    return ctx


def setup_pool256_identity(model, model_path, tokenizer):
    """pool=256 (all experts in pool, identity remap, no offloading).

    Tests whether the offload code path itself causes quality loss,
    independent of precision or miss handling.
    """
    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()

    # PP warmup
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    # NO compact — keep all 256 experts in pool (identity remap)
    # NO shadow, NO rerank
    # Just disable TG buffering to save memory
    inner = model
    for attr in ("model", "model", "language_model", "model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy"):
                sw._disable_tg_buffer = True

    return ctx


def setup_pool256_forced_compact(model, model_path, tokenizer):
    """pool=256 + forced compact path (not identity).

    All 256 experts stay in pool, but uses the non-identity remap code
    path. No misses possible. Tests if the compact code path itself
    causes quality loss vs identity path.
    """
    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()

    # PP warmup
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    # Compact to 256 = keep all experts, but switch to non-identity path
    ctx.compact(pool_size=256, disable_coverage_gate=True,
                auto_expand_cpu_cache=False)

    inner = model
    for attr in ("model", "model", "language_model", "model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy"):
                sw._disable_tg_buffer = True

    return ctx


def main():
    parser = argparse.ArgumentParser(description="Quality benchmark")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit")
    parser.add_argument("--data", default="/Users/lisihao/ThunderOMLX/data/memcollab/math500_subset_50.jsonl")
    parser.add_argument("--n", type=int, default=20, help="Number of problems")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--configs", default="A,B,C,D,E", help="Configs to run")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Load problems
    with open(args.data) as f:
        problems = [json.loads(line) for line in f][:args.n]

    print("=" * 70)
    print("  MATH-500 Quality Benchmark for Expert Offload")
    print(f"  Model: {args.model}")
    print(f"  Problems: {len(problems)}")
    print(f"  Max tokens: {args.max_tokens}")
    print("=" * 70)

    configs = {
        "A": ("A_standard", setup_standard),
        "B": ("B_pool32_rr", setup_pool32_rr),
        "C": ("C_pool64_rr", setup_pool64_rr),
        "D": ("D_pool32_shadow", setup_pool32_shadow),
        "E": ("E_pool64_shadow", setup_pool64_shadow),
        "F": ("F_pool256_identity", setup_pool256_identity),
        "G": ("G_pool32_shadow6", setup_pool32_shadow6),
        "H": ("H_pool32_shd6_norr", setup_pool32_shadow6_norr),
        "I": ("I_pool256_compact", setup_pool256_forced_compact),
        "J": ("J_pool32_poolshadow", setup_pool32_poolshadow),
        "K": ("K_pool32_ftec64", setup_pool32_ftec64),
        "L": ("L_pool32_ftec128", setup_pool32_ftec128),
        "M": ("M_pool32_ftec64_4b", setup_pool32_ftec64_4bit),
        "N": ("N_pool32_ftec224", setup_pool32_ftec224),
        "O": ("O_pool32_shd6_ftec", setup_pool32_shadow6_ftec),
        "P": ("P_ftec224_noguard", setup_pool32_ftec224_noguard),
        "Q": ("Q_k1warm_fullshd", setup_pool32_k1warm_fullshadow),
    }

    all_results = []
    for cfg_key in args.configs.split(","):
        cfg_key = cfg_key.strip()
        if cfg_key in configs:
            label, setup_fn = configs[cfg_key]
            result = run_config(args.model, problems, args.max_tokens,
                                label, setup_fn)
            all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("  QUALITY RESULTS")
    print("=" * 70)
    print(f"  {'Config':<16s} {'Correct':>8s} {'Total':>6s} {'Accuracy':>9s}")
    print(f"  {'-'*16} {'-'*8} {'-'*6} {'-'*9}")
    for r in all_results:
        print(f"  {r['label']:<16s} {r['correct']:>8d} {r['total']:>6d} "
              f"{r['accuracy']:>8.1%}")

    # Per-subject breakdown if enough data
    if len(all_results) > 1:
        print(f"\n  PER-SUBJECT BREAKDOWN:")
        subjects = set()
        for r in all_results:
            for p in r["results"]:
                subjects.add(p["subject"])

        for subj in sorted(subjects):
            line = f"    {subj[:20]:<20s}"
            for r in all_results:
                subj_results = [p for p in r["results"] if p["subject"] == subj]
                if subj_results:
                    correct = sum(1 for p in subj_results if p["correct"])
                    line += f"  {correct}/{len(subj_results)}"
                else:
                    line += "  -"
            print(line)

    # Save
    out_path = args.output or ".solar/tep-quality.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
