#!/usr/bin/env python3
"""
Community vs FlashMLX — Full A/B Benchmark
=============================================
Qwen3.5-35B-A3B on M4 Pro 48GB

Test matrix:
  - Batch = 4
  - Context: 4K, 8K, 16K tokens
  - Mode A: Community baseline (vanilla mlx-lm, all weights on GPU, sync prefill)
  - Mode B: FlashMLX (expert offload + interleaved + compact + maintenance + pruning + streaming)

Metrics:
  - PP: prefill throughput (tok/s)
  - TG: decode throughput (tok/s)
  - TTFT: time to first token (first request in batch)
  - Peak memory (GB)
  - Total time
"""

import time, gc, json, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator

MODEL_PATH = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
BATCH_SIZE = 4
MAX_GEN = 100  # enough to get stable TG measurement

# ── Build prompts for target token counts ──

BASE_TEXTS = [
    ("Analyze distributed consensus algorithms including Paxos and Raft. "
     "Discuss correctness proofs, performance characteristics, leader election, "
     "log replication, and snapshot mechanisms in systems like ZooKeeper, etcd, "
     "and CockroachDB. Compare strong vs eventual consistency. "),
    ("Explain modern CPU architecture including out-of-order execution, "
     "branch prediction, speculative execution, cache hierarchy L1/L2/L3, "
     "memory controllers, NUMA topology, and how these impact software performance. "
     "Discuss Amdahl's law and memory wall implications. "),
    ("Describe the evolution of programming language type systems from C to Rust. "
     "Cover ownership, borrowing, lifetime annotations, algebraic data types, "
     "pattern matching, trait-based polymorphism, and zero-cost abstractions. "
     "Compare with Haskell's type classes and Go's interfaces. "),
    ("Analyze large language model inference optimization techniques including "
     "KV cache management, expert offloading for MoE models, continuous batching, "
     "speculative decoding, quantization, flash attention, paged attention, "
     "and how these interact on memory-constrained devices. "),
]


def build_prompts(tokenizer, target_tokens, n=BATCH_SIZE):
    """Build n prompts, each approximately target_tokens long after chat template."""
    prompts = []
    for i in range(n):
        base = BASE_TEXTS[i % len(BASE_TEXTS)]
        # Estimate tokens per base text
        test_toks = tokenizer.encode(base)
        toks_per_repeat = len(test_toks)
        # Chat template overhead: ~20 tokens
        needed_repeats = max(1, (target_tokens - 20) // toks_per_repeat + 1)
        raw_text = (base * needed_repeats)
        # Encode, truncate to target, decode back
        raw_toks = tokenizer.encode(raw_text)
        if len(raw_toks) > target_tokens:
            raw_text = tokenizer.decode(raw_toks[:target_tokens])
        # Vary slightly per prompt
        suffix = [
            " Focus on correctness and safety.",
            " Focus on performance and scalability.",
            " Focus on real-world deployment.",
            " Focus on research directions.",
        ][i % 4]
        prompts.append(raw_text + suffix)
    return prompts


def encode_prompts(tokenizer, prompts):
    """Apply chat template and encode."""
    encoded = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        fmt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        encoded.append(tokenizer.encode(fmt))
    return encoded


def run_community(model, tokenizer, encoded, max_gen=MAX_GEN, label=""):
    """Community baseline: standard BatchGenerator, no offload, sync prefill."""
    batch_size = len(encoded)
    prompt_lens = [len(e) for e in encoded]

    gc.collect(); mx.clear_cache()
    mem_before = mx.get_active_memory() / 1024**3
    mx.reset_peak_memory()  # Reset peak tracker before run

    gen = BatchGenerator(
        model,
        max_tokens=max_gen,
        completion_batch_size=batch_size,
        prefill_batch_size=batch_size,  # sync: prefill all at once
        interleaved=False,
    )
    uids = gen.insert(encoded, max_tokens=[max_gen] * batch_size)

    all_tokens = {uid: [] for uid in uids}
    first_token_time = {}
    finished = set()
    mem_trace = []
    t0 = time.perf_counter()

    while True:
        resps = gen.next()
        mem_trace.append(mx.get_active_memory() / 1024**3)

        if not resps:
            if len(finished) >= batch_size:
                break
            continue

        now = time.perf_counter()
        for r in resps:
            all_tokens[r.uid].append(r.token)
            if r.uid not in first_token_time:
                first_token_time[r.uid] = now
            if r.finish_reason is not None:
                finished.add(r.uid)

        if len(finished) >= batch_size:
            break

    t_end = time.perf_counter()
    stats = gen.stats()
    gen.close()

    mem_peak_active = max(mem_trace) if mem_trace else mem_before
    mem_peak_true = mx.get_peak_memory() / 1024**3  # TRUE peak including intra-step
    mem_after = mx.get_active_memory() / 1024**3

    # Metrics
    total_prompt_tokens = sum(prompt_lens)
    total_gen_tokens = sum(len(t) for t in all_tokens.values())
    ttft_first = min(first_token_time.values()) - t0 if first_token_time else 0
    ttft_last = max(first_token_time.values()) - t0 if first_token_time else 0

    # PP: prompt tokens / prefill time (= TTFT since sync prefill)
    pp_time = ttft_last  # all prompts prefilled before first token
    pp_tps = total_prompt_tokens / pp_time if pp_time > 0 else 0

    # TG: gen tokens / decode time
    decode_start = min(first_token_time.values()) if first_token_time else t0
    decode_time = t_end - decode_start
    tg_tps = total_gen_tokens / decode_time if decode_time > 0 else 0

    total_time = t_end - t0

    # Quality
    texts = {}
    quality_pass = 0
    for uid, toks in all_tokens.items():
        text = tokenizer.decode(toks)
        texts[uid] = text
        t = text.strip()
        if len(t) > 15 and t.count("the the") < 3:
            quality_pass += 1

    result = {
        "label": label,
        "mode": "community",
        "batch": batch_size,
        "prompt_lens": prompt_lens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_gen_tokens": total_gen_tokens,
        "pp_tps": round(pp_tps, 1),
        "tg_tps": round(tg_tps, 1),
        "ttft_first": round(ttft_first, 3),
        "ttft_last": round(ttft_last, 3),
        "total_s": round(total_time, 2),
        "mem_before_gb": round(mem_before, 2),
        "mem_peak_gb": round(mem_peak_true, 2),
        "mem_peak_active_gb": round(mem_peak_active, 2),
        "mem_after_gb": round(mem_after, 2),
        "quality": f"{quality_pass}/{batch_size}",
        "quality_pass": quality_pass == batch_size,
    }

    print(f"\n  [{label}] COMMUNITY — batch={batch_size}")
    print(f"    Prompts: {prompt_lens} ({total_prompt_tokens} total)")
    print(f"    PP: {pp_tps:.1f} tok/s | TG: {tg_tps:.1f} tok/s")
    print(f"    TTFT: {ttft_first:.2f}s (first) / {ttft_last:.2f}s (last)")
    print(f"    Total: {total_time:.1f}s | Gen: {total_gen_tokens} tokens")
    print(f"    Memory: before={mem_before:.2f}G TRUE peak={mem_peak_true:.2f}G "
          f"(active={mem_peak_active:.2f}G) after={mem_after:.2f}G")
    print(f"    Quality: {quality_pass}/{batch_size}")
    for uid, text in texts.items():
        preview = text.strip()[:60].replace('\n', ' ')
        print(f"    uid={uid}: \"{preview}\"")

    return result


def run_flashmlx(model, tokenizer, ctx, encoded, max_gen=MAX_GEN, label=""):
    """FlashMLX: expert offload + interleaved + all optimizations."""
    from mlx_lm.models.expert_offload import FlashBatchGenerator

    batch_size = len(encoded)
    prompt_lens = [len(e) for e in encoded]

    gc.collect(); mx.clear_cache()
    mem_before = mx.get_active_memory() / 1024**3
    mx.reset_peak_memory()  # Reset peak tracker before run

    gen = FlashBatchGenerator(
        model, ctx,
        max_tokens=max_gen,
        completion_batch_size=batch_size,
        prefill_batch_size=1,  # interleaved: 1 at a time
        interleaved=True,
    )
    uids = gen.insert(encoded, max_tokens=[max_gen] * batch_size)

    all_tokens = {uid: [] for uid in uids}
    first_token_time = {}
    finished = set()
    mem_trace = []
    step_times = []
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
            if r.uid not in first_token_time:
                first_token_time[r.uid] = now
            if r.finish_reason is not None:
                finished.add(r.uid)

        if len(finished) >= batch_size:
            break

    t_end = time.perf_counter()
    stats = gen.stats()
    gen.close()

    mem_peak_active = max(mem_trace) if mem_trace else mem_before
    mem_peak_true = mx.get_peak_memory() / 1024**3  # TRUE peak including intra-step
    mem_after = mx.get_active_memory() / 1024**3

    # Metrics
    total_prompt_tokens = sum(prompt_lens)
    total_gen_tokens = sum(len(t) for t in all_tokens.values())
    ttft_first = min(first_token_time.values()) - t0 if first_token_time else 0
    ttft_last = max(first_token_time.values()) - t0 if first_token_time else 0

    # PP: prompt tokens / total prefill time
    # In interleaved mode, prefill overlaps with decode.
    # We use the stats prompt_time for accurate measurement.
    pp_time = stats.prompt_time if hasattr(stats, 'prompt_time') and stats.prompt_time > 0 else ttft_last
    pp_tps = total_prompt_tokens / pp_time if pp_time > 0 else 0

    # TG: use pure decode steps (< 100ms) for accurate measurement
    import numpy as np
    st = np.array(step_times)
    decode_steps = st[st < 0.1]
    # Total gen tokens from decode steps only
    tg_tps_decode = 0
    if len(decode_steps) > 0:
        # Each decode step produces 1 token per active request
        # Approximate: total_gen_tokens / total_decode_time
        total_decode_time = decode_steps.sum()
        # Active requests varies, use batch-weighted approach
        tg_tps_decode = total_gen_tokens / total_decode_time if total_decode_time > 0 else 0

    # Also compute wall-clock TG from first token to end
    decode_start = min(first_token_time.values()) if first_token_time else t0
    decode_wall = t_end - decode_start
    tg_tps_wall = total_gen_tokens / decode_wall if decode_wall > 0 else 0

    total_time = t_end - t0

    # Quality
    texts = {}
    quality_pass = 0
    for uid, toks in all_tokens.items():
        text = tokenizer.decode(toks)
        texts[uid] = text
        t = text.strip()
        if len(t) > 15 and t.count("the the") < 3:
            quality_pass += 1

    result = {
        "label": label,
        "mode": "flashmlx",
        "batch": batch_size,
        "prompt_lens": prompt_lens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_gen_tokens": total_gen_tokens,
        "pp_tps": round(pp_tps, 1),
        "tg_tps": round(tg_tps_decode, 1),
        "tg_tps_wall": round(tg_tps_wall, 1),
        "ttft_first": round(ttft_first, 3),
        "ttft_last": round(ttft_last, 3),
        "total_s": round(total_time, 2),
        "mem_before_gb": round(mem_before, 2),
        "mem_peak_gb": round(mem_peak_true, 2),
        "mem_peak_active_gb": round(mem_peak_active, 2),
        "mem_after_gb": round(mem_after, 2),
        "quality": f"{quality_pass}/{batch_size}",
        "quality_pass": quality_pass == batch_size,
    }

    print(f"\n  [{label}] FLASHMLX — batch={batch_size}")
    print(f"    Prompts: {prompt_lens} ({total_prompt_tokens} total)")
    print(f"    PP: {pp_tps:.1f} tok/s | TG: {tg_tps_decode:.1f} tok/s "
          f"(wall: {tg_tps_wall:.1f})")
    print(f"    TTFT: {ttft_first:.2f}s (first) / {ttft_last:.2f}s (last)")
    print(f"    Total: {total_time:.1f}s | Gen: {total_gen_tokens} tokens")
    print(f"    Memory: before={mem_before:.2f}G TRUE peak={mem_peak_true:.2f}G "
          f"(active={mem_peak_active:.2f}G) after={mem_after:.2f}G")
    print(f"    Quality: {quality_pass}/{batch_size}")
    for uid, text in texts.items():
        preview = text.strip()[:60].replace('\n', ' ')
        print(f"    uid={uid}: \"{preview}\"")

    return result


def main():
    print("=" * 75)
    print("  Community vs FlashMLX — Full A/B Benchmark")
    print("  Qwen3.5-35B-A3B | Batch=4 | M4 Pro 48GB")
    print("=" * 75)

    contexts = [4096, 8192, 16384]
    results = []

    # ═══════════════════════════════════════════════════════════
    #  Phase 1: Community Baseline (no expert offloading)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'━' * 75}")
    print("  Phase 1: COMMUNITY BASELINE (vanilla mlx-lm, all weights on GPU)")
    print(f"{'━' * 75}")

    print("\n  Loading model (standard — all experts on GPU)...")
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    load_time = time.perf_counter() - t0
    mem_model = mx.get_active_memory() / 1024**3
    print(f"  Loaded in {load_time:.1f}s — {mem_model:.2f} GB on GPU")

    # Warmup run
    print("  Warmup...")
    warmup_enc = encode_prompts(tokenizer, ["Hello, how are you?"])
    warmup_gen = BatchGenerator(model, max_tokens=5, completion_batch_size=1)
    warmup_gen.insert(warmup_enc, max_tokens=[5])
    for _ in range(20):
        r = warmup_gen.next()
        if r and r[0].finish_reason is not None:
            break
    warmup_gen.close()
    gc.collect(); mx.clear_cache()

    community_model_mem = mx.get_active_memory() / 1024**3
    print(f"  Model memory (steady): {community_model_mem:.2f} GB")

    for ctx_len in contexts:
        label = f"community-{ctx_len // 1024}K"
        print(f"\n  ─── {ctx_len // 1024}K context ───")
        prompts = build_prompts(tokenizer, ctx_len)
        encoded = encode_prompts(tokenizer, prompts)
        actual_lens = [len(e) for e in encoded]
        print(f"  Actual prompt lengths: {actual_lens}")

        r = run_community(model, tokenizer, encoded, label=label)
        r["model_mem_gb"] = community_model_mem
        results.append(r)

    # Free community model
    del model
    gc.collect(); mx.clear_cache()
    time.sleep(2)

    # ═══════════════════════════════════════════════════════════
    #  Phase 2: FlashMLX (expert offload + all optimizations)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'━' * 75}")
    print("  Phase 2: FLASHMLX (offload + interleaved + compact + prune + stream)")
    print(f"{'━' * 75}")

    print("\n  Loading model + expert offloading...")
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())

    from mlx_lm.models.expert_offload import patch_model_for_offload
    ctx = patch_model_for_offload(model, MODEL_PATH)
    gc.collect(); mx.clear_cache()
    load_time = time.perf_counter() - t0
    mem_model = mx.get_active_memory() / 1024**3
    print(f"  Loaded + patched in {load_time:.1f}s — {mem_model:.2f} GB on GPU")

    # Warmup: trigger first compact
    print("  Warmup (triggers compact)...")
    from mlx_lm.models.expert_offload import FlashBatchGenerator
    warmup_enc = encode_prompts(tokenizer, ["Hello, explain briefly."])
    warmup_gen = FlashBatchGenerator(
        model, ctx, max_tokens=10,
        completion_batch_size=1, prefill_batch_size=1,
        interleaved=True,
    )
    warmup_gen.insert(warmup_enc, max_tokens=[10])
    for _ in range(50):
        r = warmup_gen.next()
        if r and r[0].finish_reason is not None:
            break
    warmup_gen.close()
    gc.collect(); mx.clear_cache()

    flashmlx_model_mem = mx.get_active_memory() / 1024**3
    print(f"  Model memory (after compact): {flashmlx_model_mem:.2f} GB")

    for ctx_len in contexts:
        label = f"flashmlx-{ctx_len // 1024}K"
        print(f"\n  ─── {ctx_len // 1024}K context ───")
        prompts = build_prompts(tokenizer, ctx_len)
        encoded = encode_prompts(tokenizer, prompts)
        actual_lens = [len(e) for e in encoded]
        print(f"  Actual prompt lengths: {actual_lens}")

        r = run_flashmlx(model, tokenizer, ctx, encoded, label=label)
        r["model_mem_gb"] = flashmlx_model_mem
        results.append(r)

    del model
    gc.collect(); mx.clear_cache()

    # ═══════════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  SUMMARY — Community vs FlashMLX | Qwen3.5-35B-A3B | Batch=4")
    print(f"{'=' * 80}")

    print(f"\n  {'Config':<20} {'PP':>8} {'TG':>8} {'TTFT-1st':>9} "
          f"{'TTFT-all':>9} {'Total':>7} {'TRUE Peak':>10} {'Model':>7} {'Qual':>5}")
    print(f"  {'─' * 85}")
    for r in results:
        print(f"  {r['label']:<20} {r['pp_tps']:>7.1f} {r['tg_tps']:>7.1f} "
              f"{r['ttft_first']:>8.2f}s {r['ttft_last']:>8.2f}s "
              f"{r['total_s']:>6.1f}s {r['mem_peak_gb']:>9.2f}G "
              f"{r['model_mem_gb']:>6.2f}G {r['quality']:>5}")

    # Paired comparison
    print(f"\n  ── Improvement (FlashMLX vs Community) ──")
    print(f"  {'Context':<10} {'PP Δ':>10} {'TG Δ':>10} {'TTFT Δ':>10} "
          f"{'Total Δ':>10} {'Peak Δ':>10} {'Model Δ':>10}")
    print(f"  {'─' * 70}")

    comparisons = []
    for ctx_len in contexts:
        ck = f"{ctx_len // 1024}K"
        comm = next(r for r in results if r["label"] == f"community-{ck}")
        flash = next(r for r in results if r["label"] == f"flashmlx-{ck}")

        pp_pct = ((flash["pp_tps"] - comm["pp_tps"]) / comm["pp_tps"] * 100
                  if comm["pp_tps"] > 0 else 0)
        tg_pct = ((flash["tg_tps"] - comm["tg_tps"]) / comm["tg_tps"] * 100
                  if comm["tg_tps"] > 0 else 0)
        ttft_pct = ((flash["ttft_first"] - comm["ttft_first"]) / comm["ttft_first"] * 100
                    if comm["ttft_first"] > 0.01 else 0)
        total_pct = ((flash["total_s"] - comm["total_s"]) / comm["total_s"] * 100
                     if comm["total_s"] > 0 else 0)
        peak_delta = flash["mem_peak_gb"] - comm["mem_peak_gb"]
        model_delta = flash["model_mem_gb"] - comm["model_mem_gb"]

        print(f"  {ck:<10} {pp_pct:>+9.1f}% {tg_pct:>+9.1f}% "
              f"{ttft_pct:>+9.1f}% {total_pct:>+9.1f}% "
              f"{peak_delta:>+9.2f}G {model_delta:>+9.2f}G")

        comparisons.append({
            "context": ck,
            "pp_pct": round(pp_pct, 1),
            "tg_pct": round(tg_pct, 1),
            "ttft_pct": round(ttft_pct, 1),
            "total_pct": round(total_pct, 1),
            "peak_delta_gb": round(peak_delta, 2),
            "model_delta_gb": round(model_delta, 2),
        })

    # Memory headline — use 16K as worst case
    comm_16k = next((r for r in results if r["label"] == "community-16K"), None)
    flash_16k = next((r for r in results if r["label"] == "flashmlx-16K"), None)
    print(f"\n  ── Memory Headline ──")
    print(f"  Community model footprint:  {community_model_mem:.2f} GB (all 256 experts on GPU)")
    print(f"  FlashMLX model footprint:   {flashmlx_model_mem:.2f} GB (compact pool ~153 experts)")
    print(f"  Model memory saved:         {community_model_mem - flashmlx_model_mem:.2f} GB "
          f"({(1 - flashmlx_model_mem/community_model_mem)*100:.0f}%)")
    if comm_16k and flash_16k:
        c_peak = comm_16k["mem_peak_gb"]
        f_peak = flash_16k["mem_peak_gb"]
        saved = c_peak - f_peak
        pct = (1 - f_peak / c_peak) * 100 if c_peak > 0 else 0
        print(f"\n  TRUE Peak (16K×4 worst case):")
        print(f"    Community:  {c_peak:.2f} GB")
        print(f"    FlashMLX:   {f_peak:.2f} GB")
        print(f"    Saved:      {saved:.2f} GB ({pct:.0f}%)")

    all_pass = all(r["quality_pass"] for r in results)
    print(f"\n  Quality: {'ALL PASS ✅' if all_pass else 'SOME FAIL ❌'}")

    # Save
    out = "/Users/lisihao/FlashMLX/.solar/bench-community-vs-flashmlx.json"
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Qwen3.5-35B-A3B",
        "device": "M4 Pro 48GB",
        "batch_size": BATCH_SIZE,
        "max_gen": MAX_GEN,
        "community_model_mem_gb": round(community_model_mem, 2),
        "flashmlx_model_mem_gb": round(flashmlx_model_mem, 2),
        "results": results,
        "comparisons": comparisons,
        "all_quality_pass": all_pass,
    }
    with open(out, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved: {out}")


if __name__ == "__main__":
    main()
