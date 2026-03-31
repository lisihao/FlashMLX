#!/usr/bin/env python3
"""
P5: End-to-End Interleaved Scheduling Benchmark
=================================================
Full optimization stack on Qwen3.5-35B-A3B.

Validates:
  1. Quality — coherent output across all configurations
  2. TTFT — interleaved should improve time-to-first-token for batch>1
  3. Memory — peak memory should stay bounded regardless of batch size
  4. Throughput — decode speed should be comparable to standard mode

Test matrix:
  - Prompt lengths: ~500 tok (short), ~4K tok (medium)
  - Batch sizes: 1, 2, 4
  - Modes: standard (synchronous prefill) vs interleaved (chunked prefill)
"""

import time, gc, json, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.expert_offload import (
    patch_model_for_offload,
    FlashBatchGenerator,
)

MODEL_PATH = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
MAX_GEN = 60

# ── Prompts ──

SHORT_PROMPTS = [
    "Explain the difference between TCP and UDP in networking.",
    "Write a Python function to check if a string is a palindrome.",
    "What causes the seasons on Earth? Explain briefly.",
    "Compare REST and GraphQL APIs, listing pros and cons of each.",
]

# ~4K tokens each after chat template + repetition
MEDIUM_BASE = (
    "Analyze the following topics in depth: "
    "distributed consensus algorithms including Paxos and Raft, "
    "their correctness proofs, performance characteristics, "
    "and real-world implementations in systems like ZooKeeper, etcd, and CockroachDB. "
    "Discuss the CAP theorem implications and how modern databases handle "
    "network partitions. Compare strong consistency with eventual consistency models. "
    "Also discuss the role of leader election, log replication, and snapshot mechanisms. "
)
MEDIUM_PROMPTS = [
    (MEDIUM_BASE * 10) + " Focus on Paxos correctness proofs.",
    (MEDIUM_BASE * 10) + " Focus on Raft leader election protocol.",
    (MEDIUM_BASE * 10) + " Focus on CockroachDB's multi-region architecture.",
    (MEDIUM_BASE * 10) + " Focus on network partition handling strategies.",
]


def run_benchmark(model, tokenizer, ctx, prompts, max_gen=MAX_GEN,
                  interleaved=True, chunk_size=1024, label=""):
    """Run one benchmark configuration, return metrics dict."""
    batch_size = len(prompts)
    encoded = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        fmt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        encoded.append(tokenizer.encode(fmt))

    prompt_lens = [len(e) for e in encoded]

    gc.collect(); mx.clear_cache()
    mem_before = mx.get_active_memory() / 1024**3

    gen = FlashBatchGenerator(
        model, ctx,
        max_tokens=max_gen,
        completion_batch_size=batch_size,
        prefill_batch_size=min(batch_size, 1 if interleaved else batch_size),
        interleaved=interleaved,
        interleaved_chunk_size=chunk_size,
    )
    uids = gen.insert(encoded, max_tokens=[max_gen] * batch_size)

    all_tokens = {uid: [] for uid in uids}
    first_token_time = {uid: None for uid in uids}
    finished = set()
    step_times = []
    mem_trace = []
    t0 = time.perf_counter()

    while True:
        t_step = time.perf_counter()
        resps = gen.next()
        dt = time.perf_counter() - t_step
        step_times.append(dt)
        mem_trace.append(mx.get_active_memory() / 1024**3)

        if not resps:
            if len(finished) >= batch_size:
                break
            continue

        now = time.perf_counter()
        for r in resps:
            all_tokens[r.uid].append(r.token)
            if first_token_time[r.uid] is None:
                first_token_time[r.uid] = now
            if r.finish_reason is not None:
                finished.add(r.uid)

        if len(finished) >= batch_size:
            break

    t_end = time.perf_counter()

    # Stats from generator
    stats = gen.stats()
    gen.close()

    mem_after = mx.get_active_memory() / 1024**3
    mem_peak = max(mem_trace) if mem_trace else mem_before

    # Per-request TTFT
    ttft_per_uid = {}
    for uid in uids:
        if first_token_time[uid] is not None:
            ttft_per_uid[uid] = first_token_time[uid] - t0
    ttft_min = min(ttft_per_uid.values()) if ttft_per_uid else 0
    ttft_max = max(ttft_per_uid.values()) if ttft_per_uid else 0
    ttft_avg = sum(ttft_per_uid.values()) / len(ttft_per_uid) if ttft_per_uid else 0

    total_tokens = sum(len(t) for t in all_tokens.values())
    ttot = t_end - t0
    # TG: use decode window from first token to end
    decode_start = min(first_token_time.values()) if any(first_token_time.values()) else t0
    decode_time = t_end - decode_start
    tg = total_tokens / decode_time if decode_time > 0 else 0

    # Step time analysis
    import numpy as np
    st = np.array(step_times)
    prefill_steps = st[st > 0.3]  # > 300ms = prefill chunk
    decode_steps = st[st <= 0.3]
    avg_decode_ms = decode_steps.mean() * 1000 if len(decode_steps) > 0 else 0

    # Quality check
    texts = {}
    quality_pass = 0
    for uid, toks in all_tokens.items():
        text = tokenizer.decode(toks)
        texts[uid] = text
        t = text.strip()
        is_ok = (len(t) > 15
                 and not any(t.count(c * 5) > 0 for c in "abcdefghijklmnop")
                 and t.count("the the") < 3)
        if is_ok:
            quality_pass += 1

    # Memory budget skips
    prefill_skipped = gen._gen._prefill_skipped if hasattr(gen._gen, '_prefill_skipped') else 0

    mode = "INT" if interleaved else "STD"
    print(f"\n  [{label}] mode={mode}, batch={batch_size}, "
          f"prompts={prompt_lens} tok")
    print(f"    TTFT: min={ttft_min:.2f}s avg={ttft_avg:.2f}s max={ttft_max:.2f}s")
    print(f"    TTOT: {ttot:.1f}s | TG: {tg:.1f} tok/s "
          f"({total_tokens} tok in {decode_time:.1f}s decode)")
    print(f"    Decode step: {avg_decode_ms:.1f}ms avg "
          f"({len(decode_steps)} steps, {len(prefill_steps)} prefill chunks)")
    print(f"    Memory: before={mem_before:.2f}G peak={mem_peak:.2f}G "
          f"after={mem_after:.2f}G (Δpeak={mem_peak-mem_before:+.2f}G)")
    print(f"    Quality: {quality_pass}/{batch_size} "
          f"{'PASS' if quality_pass == batch_size else 'FAIL'}")
    if prefill_skipped > 0:
        print(f"    P3 budget gate: {prefill_skipped} prefill skips")

    for uid, text in texts.items():
        preview = text.strip()[:80].replace('\n', ' ')
        print(f"    uid={uid}: \"{preview}\"")

    return {
        "label": label,
        "mode": mode,
        "interleaved": interleaved,
        "batch_size": batch_size,
        "prompt_lens": prompt_lens,
        "ttft_min": round(ttft_min, 3),
        "ttft_avg": round(ttft_avg, 3),
        "ttft_max": round(ttft_max, 3),
        "ttot": round(ttot, 2),
        "tg": round(tg, 1),
        "avg_decode_ms": round(avg_decode_ms, 1),
        "mem_before": round(mem_before, 2),
        "mem_peak": round(mem_peak, 2),
        "mem_after": round(mem_after, 2),
        "quality": f"{quality_pass}/{batch_size}",
        "quality_pass": quality_pass == batch_size,
        "total_tokens": total_tokens,
        "prefill_skipped": prefill_skipped,
    }


def main():
    print("=" * 70)
    print("  P5: End-to-End Interleaved Scheduling Benchmark")
    print("  Qwen3.5-35B-A3B — Full Optimization Stack")
    print("=" * 70)

    # Load model
    print("\nLoading Qwen3.5-35B...")
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    print(f"Loaded in {time.perf_counter()-t0:.1f}s")

    # Patch for expert offload
    print("Patching for expert offload...")
    ctx = patch_model_for_offload(model, MODEL_PATH)
    gc.collect(); mx.clear_cache()
    mem_model = mx.get_active_memory() / 1024**3
    print(f"Model ready: {mem_model:.2f} GB")

    results = []

    # ═══════════════════════════════════════════════════════
    # Test 1: Short prompts (~500 tok) — batch 1, 2, 4
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  Section A: Short Prompts (~500 tok)")
    print(f"{'━' * 70}")

    # Batch=1 (baseline — interleaved has no effect)
    r = run_benchmark(model, tokenizer, ctx, SHORT_PROMPTS[:1],
                      interleaved=False, label="short-b1-std")
    results.append(r)

    r = run_benchmark(model, tokenizer, ctx, SHORT_PROMPTS[:1],
                      interleaved=True, label="short-b1-int")
    results.append(r)

    # Batch=2
    r = run_benchmark(model, tokenizer, ctx, SHORT_PROMPTS[:2],
                      interleaved=False, label="short-b2-std")
    results.append(r)

    r = run_benchmark(model, tokenizer, ctx, SHORT_PROMPTS[:2],
                      interleaved=True, label="short-b2-int")
    results.append(r)

    # Batch=4
    r = run_benchmark(model, tokenizer, ctx, SHORT_PROMPTS[:4],
                      interleaved=False, label="short-b4-std")
    results.append(r)

    r = run_benchmark(model, tokenizer, ctx, SHORT_PROMPTS[:4],
                      interleaved=True, label="short-b4-int")
    results.append(r)

    # ═══════════════════════════════════════════════════════
    # Test 2: Medium prompts (~4K tok) — batch 1, 2, 4
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  Section B: Medium Prompts (~4K tok)")
    print(f"{'━' * 70}")

    # Batch=1
    r = run_benchmark(model, tokenizer, ctx, MEDIUM_PROMPTS[:1],
                      interleaved=False, label="med-b1-std")
    results.append(r)

    r = run_benchmark(model, tokenizer, ctx, MEDIUM_PROMPTS[:1],
                      interleaved=True, label="med-b1-int")
    results.append(r)

    # Batch=2
    r = run_benchmark(model, tokenizer, ctx, MEDIUM_PROMPTS[:2],
                      interleaved=False, label="med-b2-std")
    results.append(r)

    r = run_benchmark(model, tokenizer, ctx, MEDIUM_PROMPTS[:2],
                      interleaved=True, label="med-b2-int")
    results.append(r)

    # Batch=4
    r = run_benchmark(model, tokenizer, ctx, MEDIUM_PROMPTS[:4],
                      interleaved=False, label="med-b4-std")
    results.append(r)

    r = run_benchmark(model, tokenizer, ctx, MEDIUM_PROMPTS[:4],
                      interleaved=True, label="med-b4-int")
    results.append(r)

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  P5 SUMMARY — Interleaved Scheduling on Qwen3.5-35B-A3B")
    print(f"{'=' * 80}")

    print(f"\n  {'Label':<18} {'Mode':>4} {'B':>2} {'TTFT-1st':>8} {'TTFT-last':>9} "
          f"{'TTOT':>6} {'TG':>7} {'Peak':>6} {'Qual':>5}")
    print(f"  {'─' * 75}")

    for r in results:
        print(f"  {r['label']:<18} {r['mode']:>4} {r['batch_size']:>2} "
              f"{r['ttft_min']:>7.2f}s {r['ttft_max']:>8.2f}s "
              f"{r['ttot']:>5.1f}s {r['tg']:>6.1f} {r['mem_peak']:>5.2f}G "
              f"{r['quality']:>5}")

    # Paired comparisons
    print(f"\n  ── Interleaved vs Standard ──")
    print(f"  {'Config':<18} {'TTFT Δ':>10} {'TTOT Δ':>10} {'Peak Δ':>10} {'Decode ms':>12}")
    print(f"  {'─' * 60}")

    pairs = []
    for i in range(0, len(results), 2):
        std = results[i]
        intl = results[i + 1]
        ttft_delta = intl["ttft_min"] - std["ttft_min"]
        ttft_pct = (ttft_delta / std["ttft_min"] * 100) if std["ttft_min"] > 0.01 else 0
        ttot_delta = intl["ttot"] - std["ttot"]
        ttot_pct = (ttot_delta / std["ttot"] * 100) if std["ttot"] > 0.01 else 0
        mem_delta = intl["mem_peak"] - std["mem_peak"]

        label = std["label"].replace("-std", "")
        print(f"  {label:<18} {ttft_pct:>+8.1f}% {ttot_pct:>+8.1f}% "
              f"{mem_delta:>+8.2f}G "
              f"{std['avg_decode_ms']:>5.1f}→{intl['avg_decode_ms']:.1f}ms")
        pairs.append({
            "config": label,
            "ttft_pct": round(ttft_pct, 1),
            "ttot_pct": round(ttot_pct, 1),
            "mem_delta_gb": round(mem_delta, 2),
            "decode_std_ms": std["avg_decode_ms"],
            "decode_int_ms": intl["avg_decode_ms"],
        })

    # Quality verdict
    all_pass = all(r["quality_pass"] for r in results)
    print(f"\n  Quality: {'ALL PASS ✅' if all_pass else 'SOME FAIL ❌'}")

    # Memory budget gate stats
    budget_skips = sum(r["prefill_skipped"] for r in results)
    if budget_skips > 0:
        print(f"  P3 Memory Budget: {budget_skips} total prefill skips")
    else:
        print(f"  P3 Memory Budget: no skips (sufficient headroom)")

    # Save results
    out = "/Users/lisihao/FlashMLX/.solar/p5-benchmark-results.json"
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Qwen3.5-35B-A3B",
        "model_memory_gb": round(mem_model, 2),
        "results": results,
        "comparisons": pairs,
        "all_quality_pass": all_pass,
    }
    with open(out, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved: {out}")


if __name__ == "__main__":
    main()
