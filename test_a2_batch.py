#!/usr/bin/env python3
"""A2: Continuous Batching test — verify BatchGenerator works with expert offloading
and measure throughput scaling.

Tests:
  1. BatchGenerator + offloaded model compatibility
  2. Throughput scaling: batch_size 1, 2, 4, 8
  3. Quality: verify each request generates coherent text
"""

import time
import gc
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.expert_offload import patch_model_for_offload

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


def run_batch_test(model, tokenizer, batch_size, ctx=None):
    """Run BatchGenerator with given batch_size. Returns total throughput."""
    prompts_to_use = PROMPTS[:batch_size]

    # Tokenize (BatchGenerator expects Python lists, not mx.arrays)
    encoded = []
    for p in prompts_to_use:
        msgs = [{"role": "user", "content": p}]
        fmt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        encoded.append(tokenizer.encode(fmt))

    # Create generator
    gen = BatchGenerator(
        model,
        max_tokens=MAX_GEN,
        completion_batch_size=batch_size,
        prefill_batch_size=batch_size,
    )
    uids = gen.insert(encoded, max_tokens=[MAX_GEN] * batch_size)

    # Note: with BatchGenerator, prefill happens inside _process_prompts().
    # The full pool (256 experts) is used — no compact needed for this test.
    # Compact would need to be called after first prefill round for optimal perf.

    # Run generation
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

    # Quality check
    quality_pass = 0
    for uid in uids:
        text = tokenizer.decode(all_tokens[uid])
        if len(text.strip()) > 20:
            quality_pass += 1

    return throughput, quality_pass, batch_size, elapsed, total_tokens


def main():
    print("=" * 65)
    print("A2: Continuous Batching Test — BatchGenerator + Expert Offloading")
    print("=" * 65)

    # Load model
    print("\n[1] Loading model...")
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    gc.collect()

    # Patch for offloading
    print("[2] Patching for expert offloading...")
    ctx = patch_model_for_offload(
        model, MODEL_PATH, max_workers=4, cpu_cache_gb=2.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()
    mx.clear_cache()

    # Test batch sizes
    print("\n[3] Running batch scaling tests...\n")
    results = []

    for bs in [1, 2, 4]:
        print(f"  --- Batch size {bs} ---")
        try:
            throughput, quality, batch_size, elapsed, total_tokens = run_batch_test(
                model, tokenizer, bs, ctx
            )
            results.append((bs, throughput, quality, elapsed, total_tokens))
            print(f"  Throughput: {throughput:.1f} tok/s total, "
                  f"quality: {quality}/{bs} PASS, "
                  f"tokens: {total_tokens}, time: {elapsed:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((bs, 0, 0, 0, 0))
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"\n  {'Batch':>6} {'Throughput':>12} {'Per-seq':>10} {'Quality':>10} {'Scaling':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    baseline_tps = results[0][1] if results[0][1] > 0 else 1
    for bs, tps, quality, elapsed, total in results:
        per_seq = tps / bs if bs > 0 and tps > 0 else 0
        scaling = tps / baseline_tps if baseline_tps > 0 else 0
        print(f"  {bs:>6} {tps:>10.1f}/s {per_seq:>8.1f}/s "
              f"{quality}/{bs:>5} {scaling:>9.1f}×")


if __name__ == "__main__":
    main()
