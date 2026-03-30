#!/usr/bin/env python3
"""A2: FlashBatchGenerator test — BatchGenerator + compact + maintenance + pruning.

Compares:
  1. Raw BatchGenerator (full pool, no compact)
  2. FlashBatchGenerator (compacted pool + maintenance + pruning)

Key: compact cost is amortized across many requests.
We pre-compact, then measure decode-only throughput.
"""

import time
import gc
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.expert_offload import (
    patch_model_for_offload,
    FlashBatchGenerator,
)

MODEL_PATH = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
MAX_GEN = 128

PROMPTS = [
    "Explain quantum computing in simple terms:",
    "Write a short poem about the ocean:",
    "What are the key differences between TCP and UDP?",
    "Describe the water cycle in 3 sentences:",
    "List 5 benefits of regular exercise:",
    "Explain how a neural network learns:",
    "What is the difference between AI and machine learning?",
    "Describe the process of photosynthesis:",
]


def encode_prompts(tokenizer, batch_size):
    encoded = []
    for p in PROMPTS[:batch_size]:
        msgs = [{"role": "user", "content": p}]
        fmt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        encoded.append(tokenizer.encode(fmt))
    return encoded


def run_batch(gen_factory, tokenizer, batch_size):
    """Run a batch test with the given generator factory. Returns throughput."""
    encoded = encode_prompts(tokenizer, batch_size)
    gen = gen_factory(batch_size)
    uids = gen.insert(encoded, max_tokens=[MAX_GEN] * batch_size)

    all_tokens = {uid: [] for uid in uids}
    finished = set()
    total_tokens = 0
    t0 = time.perf_counter()

    while True:
        responses = gen.next()
        if not responses:
            break
        for resp in responses:
            all_tokens[resp.uid].append(resp.token)
            total_tokens += 1
            if resp.finish_reason is not None:
                finished.add(resp.uid)
        if len(finished) == batch_size:
            break

    elapsed = time.perf_counter() - t0
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    gen.close()

    quality_pass = sum(1 for uid in uids if len(tokenizer.decode(all_tokens[uid]).strip()) > 20)
    return throughput, quality_pass, elapsed, total_tokens


def main():
    print("=" * 70)
    print("A2: FlashBatchGenerator vs Raw BatchGenerator")
    print("=" * 70)

    # --- Phase 1: Raw BatchGenerator ---
    print("\n[1] Loading model for raw BatchGenerator...")
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    gc.collect()

    ctx_raw = patch_model_for_offload(
        model, MODEL_PATH, max_workers=4, cpu_cache_gb=2.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()
    mx.clear_cache()

    def raw_factory(bs):
        return BatchGenerator(model, max_tokens=MAX_GEN,
                              completion_batch_size=bs, prefill_batch_size=bs)

    raw_results = []
    print("\n[2] Raw BatchGenerator (full 256-expert pool)...\n")
    for bs in [1, 2, 4]:
        tps, q, elapsed, total = run_batch(raw_factory, tokenizer, bs)
        raw_results.append((bs, tps, q, elapsed, total))
        print(f"  Batch {bs}: {tps:.1f} tok/s, quality {q}/{bs}, "
              f"tokens {total}, time {elapsed:.1f}s")

    del model, ctx_raw
    gc.collect()
    mx.clear_cache()

    # --- Phase 2: FlashBatchGenerator (pre-compacted) ---
    print("\n[3] Loading model for FlashBatchGenerator...")
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    gc.collect()

    ctx = patch_model_for_offload(
        model, MODEL_PATH, max_workers=4, cpu_cache_gb=2.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()
    mx.clear_cache()

    # Warmup: trigger compact with a short generation
    print("\n[4] Warmup: trigger compact...")
    warmup_enc = encode_prompts(tokenizer, 1)
    warmup_gen = FlashBatchGenerator(
        model, ctx, max_tokens=8, completion_batch_size=1, prefill_batch_size=1,
    )
    warmup_uids = warmup_gen.insert(warmup_enc, max_tokens=[8])
    for _ in range(10):
        r = warmup_gen.next()
        if not r:
            break
    warmup_gen.close()
    gc.collect()
    mx.clear_cache()
    print("  Compact done. Pool is now compacted.")

    def flash_factory(bs):
        # Pool is already compacted — FlashBatchGenerator won't re-compact
        # because _compacted starts False but compact() is idempotent on already-compacted pool
        g = FlashBatchGenerator(
            model, ctx, max_tokens=MAX_GEN,
            completion_batch_size=bs, prefill_batch_size=bs,
        )
        g._compacted = True  # Skip re-compact, already done
        return g

    flash_results = []
    print("\n[5] FlashBatchGenerator (compacted pool + maintenance)...\n")
    for bs in [1, 2, 4]:
        tps, q, elapsed, total = run_batch(flash_factory, tokenizer, bs)
        flash_results.append((bs, tps, q, elapsed, total))
        print(f"  Batch {bs}: {tps:.1f} tok/s, quality {q}/{bs}, "
              f"tokens {total}, time {elapsed:.1f}s")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("RESULTS: Raw (full pool) vs Flash (compacted + maintenance)")
    print("=" * 70)
    print(f"\n  {'Batch':>6} {'Raw tok/s':>12} {'Flash tok/s':>12} {'Delta':>8} {'Quality':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")

    for i, (bs, raw_tps, raw_q, _, _) in enumerate(raw_results):
        flash_tps = flash_results[i][1] if i < len(flash_results) else 0
        flash_q = flash_results[i][2] if i < len(flash_results) else 0
        delta = (flash_tps - raw_tps) / raw_tps * 100 if raw_tps > 0 else 0
        print(f"  {bs:>6} {raw_tps:>10.1f}/s {flash_tps:>10.1f}/s {delta:>+7.1f}% "
              f"{raw_q}/{bs} → {flash_q}/{bs}")

    # Scaling
    print(f"\n  Scaling:")
    raw_base = raw_results[0][1] if raw_results[0][1] > 0 else 1
    flash_base = flash_results[0][1] if flash_results[0][1] > 0 else 1
    for i, (bs, _, _, _, _) in enumerate(raw_results):
        raw_s = raw_results[i][1] / raw_base
        flash_s = flash_results[i][1] / flash_base
        print(f"    Batch {bs}: Raw {raw_s:.1f}×, Flash {flash_s:.1f}× (ideal {bs:.1f}×)")


if __name__ == "__main__":
    main()
