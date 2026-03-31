#!/usr/bin/env python3
"""Quick test: verify compact + miss handling works. Two pool sizes: 256, 128."""

import gc
import json
import os
import sys
import time

MLX_LM_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx-lm-source")
sys.path.insert(0, MLX_LM_SOURCE)

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

MODEL_PATH = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
PROMPT = "Explain the difference between TCP and UDP in 3 sentences."
MAX_GEN_TOKENS = 200


def run_test(pool_size):
    """Load model, offload, PP, compact, TG."""
    from mlx_lm.models.expert_offload import patch_model_for_offload

    print(f"\n{'='*60}")
    print(f"  Pool size: {pool_size}")
    print(f"{'='*60}")

    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    gc.collect()

    mem_base = mx.metal.get_active_memory() / 1024**3
    ctx = patch_model_for_offload(
        model, MODEL_PATH,
        max_workers=4,
        cpu_cache_gb=2.0,
        enable_prefetch=False,
        enable_telemetry=True,
    )
    gc.collect()
    mx.metal.clear_cache()

    # Build prompt
    messages = [{"role": "user", "content": PROMPT}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_array = mx.array(tokenizer.encode(formatted))
    prompt_tokens = len(tokenizer.encode(formatted))

    mx.metal.reset_peak_memory()

    # PP
    t0 = time.perf_counter()
    gen = generate_step(prompt_array, model)
    first_token, _ = next(gen)
    pp_time = time.perf_counter() - t0
    pp_tps = prompt_tokens / pp_time

    mem_after_pp = mx.metal.get_active_memory() / 1024**3
    print(f"  PP: {pp_tps:.0f} tok/s ({pp_time*1000:.0f}ms) | Mem: {mem_after_pp:.2f} GB")

    # Compact (if not full, or force_remap to test remap overhead)
    force_remap = os.environ.get("FORCE_REMAP") == "1" and pool_size == 256
    if pool_size < 256 or force_remap:
        compact_info = ctx.compact(pool_size=pool_size)
        mem_after_compact = mx.metal.get_active_memory() / 1024**3
        label = " (REMAP FORCED)" if force_remap else ""
        print(f"  Compact{label}: pool={compact_info['pool_size']}, "
              f"coverage={compact_info['pp_coverage']:.1%}, "
              f"CPU cache={compact_info.get('cpu_cache_gb', 0):.1f} GB, "
              f"mem={mem_after_compact:.2f} GB (saved {mem_after_pp - mem_after_compact:.2f} GB)")
    else:
        mem_after_compact = mem_after_pp

    # TG with warmup/steady-state split
    WARMUP_TOKENS = 50
    t_gen = time.perf_counter()
    ft = first_token if isinstance(first_token, int) else first_token.item()
    tokens_out = [ft]
    t_warmup_end = None
    for i, (tok, _) in enumerate(gen):
        t = tok if isinstance(tok, int) else tok.item()
        tokens_out.append(t)
        if len(tokens_out) == WARMUP_TOKENS + 1 and t_warmup_end is None:
            t_warmup_end = time.perf_counter()
        if i + 1 >= MAX_GEN_TOKENS - 1:
            break
    gen_time = time.perf_counter() - t_gen
    tg_tps = len(tokens_out) / gen_time if gen_time > 0 else 0

    # Steady-state TG (after warmup)
    tg_steady = 0
    if t_warmup_end is not None and len(tokens_out) > WARMUP_TOKENS + 1:
        steady_tokens = len(tokens_out) - WARMUP_TOKENS - 1
        steady_time = time.perf_counter() - t_warmup_end
        # Wait, we need the end time. Actually gen already finished.
        steady_time = gen_time - (t_warmup_end - t_gen)
        tg_steady = steady_tokens / steady_time if steady_time > 0 else 0

    peak = mx.metal.get_peak_memory() / 1024**3
    text = tokenizer.decode(tokens_out)

    print(f"  TG: {tg_tps:.1f} tok/s avg ({len(tokens_out)} tokens) | "
          f"Steady: {tg_steady:.1f} tok/s | Peak: {peak:.2f} GB")
    print(f"  Text: {text[:100]}")

    # CPU cache stats
    if ctx.cpu_cache:
        cs = ctx.cpu_cache.summary()
        print(f"  CPU cache: {cs['entries']} entries, "
              f"hit_rate={cs['hit_rate']:.2%}, "
              f"used={cs['used_gb']:.1f}/{cs['capacity_gb']:.1f} GB")

    ctx.close()
    del model
    gc.collect()
    mx.metal.clear_cache()

    return {
        "pool_size": pool_size,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "tg_steady": tg_steady,
        "peak_gb": peak,
        "mem_after_compact_gb": mem_after_compact,
        "text": text[:80],
    }


def main():
    print("=" * 60)
    print("  Quick Compact Test: pool=256 vs pool=128")
    print("=" * 60)

    results = []
    for ps in [256, 192, 128]:
        r = run_test(ps)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  pool={r['pool_size']:>3}: TG={r['tg_tps']:>5.1f} avg | "
              f"Steady={r['tg_steady']:>5.1f} tok/s | "
              f"Peak={r['peak_gb']:.2f} GB | Mem={r['mem_after_compact_gb']:.2f} GB | "
              f"Saved={18.21 - r['mem_after_compact_gb']:.1f} GB")

    if len(results) == 2:
        tg_delta = (results[1]['tg_tps'] - results[0]['tg_tps']) / results[0]['tg_tps'] * 100
        mem_saved = results[0]['peak_gb'] - results[1]['peak_gb']
        print(f"\n  Compact 256→128: TG {tg_delta:+.1f}%, Memory saved: {mem_saved:.1f} GB")


if __name__ == "__main__":
    main()
