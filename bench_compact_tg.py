#!/usr/bin/env python3
"""Focused test: compact pool TG warmup analysis.
Generates 200 tokens and measures TG in 50-token segments.
"""

import gc
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
POOL_SIZE = 128
SEGMENT_SIZE = 50


def main():
    from mlx_lm.models.expert_offload import patch_model_for_offload

    print(f"Compact TG Warmup Test: pool={POOL_SIZE}, {MAX_GEN_TOKENS} tokens")
    print(f"Measuring TG in {SEGMENT_SIZE}-token segments")
    print("=" * 60)

    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, MODEL_PATH,
        max_workers=4, cpu_cache_gb=2.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()
    mx.metal.clear_cache()

    # PP
    messages = [{"role": "user", "content": PROMPT}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_array = mx.array(tokenizer.encode(formatted))
    prompt_tokens = len(tokenizer.encode(formatted))

    mx.metal.reset_peak_memory()

    t0 = time.perf_counter()
    gen = generate_step(prompt_array, model)
    first_token, _ = next(gen)
    pp_time = time.perf_counter() - t0
    print(f"PP: {prompt_tokens / pp_time:.0f} tok/s ({pp_time * 1000:.0f}ms)")

    # Compact
    compact_info = ctx.compact(pool_size=POOL_SIZE)
    mem = mx.metal.get_active_memory() / 1024**3
    print(f"Compact: pool={compact_info['pool_size']}, coverage={compact_info['pp_coverage']:.1%}, "
          f"mem={mem:.2f} GB")

    # TG in segments
    ft = first_token if isinstance(first_token, int) else first_token.item()
    tokens_out = [ft]
    segment_start = time.perf_counter()
    segment_tokens = 0

    for i, (tok, _) in enumerate(gen):
        t = tok if isinstance(tok, int) else tok.item()
        tokens_out.append(t)
        segment_tokens += 1

        if segment_tokens >= SEGMENT_SIZE:
            elapsed = time.perf_counter() - segment_start
            tps = segment_tokens / elapsed
            total = len(tokens_out)
            print(f"  Tokens {total - segment_tokens + 1}-{total}: {tps:.1f} tok/s "
                  f"({elapsed * 1000:.0f}ms)")
            segment_start = time.perf_counter()
            segment_tokens = 0

        if i + 1 >= MAX_GEN_TOKENS - 1:
            break

    # Final partial segment
    if segment_tokens > 0:
        elapsed = time.perf_counter() - segment_start
        tps = segment_tokens / elapsed
        total = len(tokens_out)
        print(f"  Tokens {total - segment_tokens + 1}-{total}: {tps:.1f} tok/s "
              f"({elapsed * 1000:.0f}ms)")

    # Overall
    peak = mx.metal.get_peak_memory() / 1024**3
    text = tokenizer.decode(tokens_out)
    print(f"\nOverall: {len(tokens_out)} tokens | Peak: {peak:.2f} GB")
    print(f"Text: {text[:200]}")

    ctx.close()


if __name__ == "__main__":
    main()
