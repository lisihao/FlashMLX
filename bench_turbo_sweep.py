#!/usr/bin/env python3
"""
Turbo Sweep: Find optimal pool/k/maintenance settings
=======================================================
Focus: Batch=4, 16K context — the hardest case.
Sweep pool_size, k_override, maintenance_interval.
Measure TG, TTFT, peak memory, quality.
"""

import time, gc, json
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.expert_offload import (
    patch_model_for_offload,
    FlashBatchGenerator,
    FlashMoeSwitchGLU,
)

MODEL_PATH = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
MAX_GEN = 100

BASE_TEXT = (
    "Analyze distributed consensus algorithms including Paxos and Raft. "
    "Discuss correctness proofs, performance characteristics, leader election, "
    "log replication, and snapshot mechanisms in systems like ZooKeeper, etcd, "
    "and CockroachDB. Compare strong vs eventual consistency. "
)

SUFFIXES = [
    " Focus on correctness and safety.",
    " Focus on performance and scalability.",
    " Focus on real-world deployment.",
    " Focus on research directions.",
]


def build_encoded(tokenizer, target_tokens=16384, n=4):
    """Build n encoded prompts of approximately target_tokens length."""
    encoded = []
    for i in range(n):
        raw_toks = tokenizer.encode(BASE_TEXT)
        repeats = max(1, (target_tokens - 20) // len(raw_toks) + 1)
        raw_text = (BASE_TEXT * repeats)
        toks = tokenizer.encode(raw_text)
        if len(toks) > target_tokens:
            raw_text = tokenizer.decode(toks[:target_tokens])
        raw_text += SUFFIXES[i % 4]
        msgs = [{"role": "user", "content": raw_text}]
        fmt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        encoded.append(tokenizer.encode(fmt))
    return encoded


def override_k(model, k_override):
    """Override effective_top_k on all MoE layers (on layer.mlp, not switch_mlp)."""
    inner = model
    for attr in ["model", "model"]:
        if hasattr(inner, attr):
            inner = getattr(inner, attr)

    count = 0
    for layer in inner.layers:
        mlp = layer.mlp
        if hasattr(mlp, "_effective_top_k"):
            mlp._effective_top_k = min(k_override, mlp.top_k)
            mlp._gate_pruning_enabled = True
            count += 1
    return count


def restore_k(model):
    """Restore original top_k on all MoE layers."""
    inner = model
    for attr in ["model", "model"]:
        if hasattr(inner, attr):
            inner = getattr(inner, attr)

    for layer in inner.layers:
        mlp = layer.mlp
        if hasattr(mlp, "_effective_top_k"):
            mlp._effective_top_k = mlp.top_k


def run_config(model, tokenizer, ctx, encoded, pool_size=None, k_override=None,
               chunk_size=1024, maint_interval=8, k_adj_interval=64, label=""):
    """Run one configuration and return metrics."""
    batch_size = len(encoded)
    prompt_lens = [len(e) for e in encoded]

    gc.collect(); mx.clear_cache()

    # Override pool size if specified
    if pool_size is not None:
        ctx.compact(pool_size=pool_size)
        ctx.enable_dynamic_pruning(model)
        ctx.adjust_effective_k(model)
        gc.collect(); mx.clear_cache()

    # Override k if specified
    if k_override is not None:
        override_k(model, k_override)

    mx.reset_peak_memory()
    mem_before = mx.get_active_memory() / 1024**3

    gen = FlashBatchGenerator(
        model, ctx,
        max_tokens=MAX_GEN,
        completion_batch_size=batch_size,
        prefill_batch_size=1,
        interleaved=True,
        interleaved_chunk_size=chunk_size,
        maintenance_interval=maint_interval,
        k_adjust_interval=k_adj_interval,
    )

    # CRITICAL: prevent FlashBatchGenerator from auto-compacting
    # (which would override our manual pool_size with auto-sizing)
    if pool_size is not None:
        gen._compacted = True

    uids = gen.insert(encoded, max_tokens=[MAX_GEN] * batch_size)

    all_tokens = {uid: [] for uid in uids}
    first_token_time = {}
    finished = set()
    step_times = []
    t0 = time.perf_counter()

    while True:
        t_step = time.perf_counter()
        resps = gen.next()
        dt = time.perf_counter() - t_step
        step_times.append(dt)

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
    gen.close()

    mem_peak = mx.get_peak_memory() / 1024**3
    mem_after = mx.get_active_memory() / 1024**3

    # Restore k if overridden
    if k_override is not None:
        restore_k(model)

    # Metrics
    import numpy as np
    total_gen = sum(len(t) for t in all_tokens.values())
    total_prompt = sum(prompt_lens)
    ttft_first = min(first_token_time.values()) - t0 if first_token_time else 0
    ttft_last = max(first_token_time.values()) - t0 if first_token_time else 0
    total_time = t_end - t0

    st = np.array(step_times)
    decode_steps = st[st < 0.15]  # pure decode < 150ms
    tg_decode = total_gen / decode_steps.sum() if len(decode_steps) > 0 and decode_steps.sum() > 0 else 0
    avg_step_ms = decode_steps.mean() * 1000 if len(decode_steps) > 0 else 0

    # PP throughput
    pp_steps = st[st >= 0.15]
    pp_time = pp_steps.sum() if len(pp_steps) > 0 else ttft_last
    pp_tps = total_prompt / pp_time if pp_time > 0 else 0

    # Quality — check for coherent output (not garbled)
    quality_pass = 0
    for uid, toks in all_tokens.items():
        text = tokenizer.decode(toks).strip()
        # Pass if: has real content, no extreme repetition
        has_content = len(text) > 15
        no_garble = not any(text.count(c * 8) > 0 for c in "abcdefghijklmnopqrstuvwxyz ")
        if has_content and no_garble:
            quality_pass += 1

    # Hit rate
    hit_rate = ctx.telemetry.get_overall_hit_rate() if ctx.telemetry else -1

    result = {
        "label": label,
        "pool_size": pool_size or "auto(153)",
        "k_override": k_override or "auto",
        "chunk_size": chunk_size,
        "maint_interval": maint_interval,
        "pp_tps": round(pp_tps, 1),
        "tg_tps": round(tg_decode, 1),
        "avg_step_ms": round(avg_step_ms, 1),
        "ttft_first": round(ttft_first, 2),
        "ttft_last": round(ttft_last, 2),
        "total_s": round(total_time, 1),
        "mem_peak_gb": round(mem_peak, 2),
        "mem_after_gb": round(mem_after, 2),
        "hit_rate": round(hit_rate, 3) if hit_rate >= 0 else "N/A",
        "quality": f"{quality_pass}/{batch_size}",
        "quality_pass": quality_pass == batch_size,
        "total_gen": total_gen,
    }

    mode_str = (f"pool={pool_size or 153}, k={k_override or 'auto'}, "
                f"chunk={chunk_size}, maint={maint_interval}")
    print(f"\n  [{label}] {mode_str}")
    print(f"    PP: {pp_tps:.0f} tok/s | TG: {tg_decode:.1f} tok/s "
          f"(step={avg_step_ms:.1f}ms)")
    print(f"    TTFT: {ttft_first:.1f}s/{ttft_last:.1f}s | Total: {total_time:.1f}s")
    print(f"    Peak: {mem_peak:.2f}G | Hit rate: "
          f"{hit_rate:.1%}" if isinstance(hit_rate, float) else f"    Peak: {mem_peak:.2f}G")
    print(f"    Quality: {quality_pass}/{batch_size} "
          f"{'PASS' if quality_pass == batch_size else '!! FAIL !!'}")

    return result


def main():
    print("=" * 75)
    print("  Turbo Sweep — Qwen3.5-35B | Batch=4 | 16K Context")
    print("  Finding optimal pool/k/maintenance settings")
    print("=" * 75)

    print("\nLoading model + expert offload...")
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    ctx = patch_model_for_offload(model, MODEL_PATH)
    gc.collect(); mx.clear_cache()
    print(f"Loaded in {time.perf_counter()-t0:.1f}s")

    # Warmup
    print("Warmup...")
    warmup_enc = build_encoded(tokenizer, target_tokens=512, n=1)
    wg = FlashBatchGenerator(model, ctx, max_tokens=5, completion_batch_size=1,
                             interleaved=True)
    wg.insert(warmup_enc, max_tokens=[5])
    for _ in range(20):
        r = wg.next()
        if r and r[0].finish_reason is not None:
            break
    wg.close()
    gc.collect(); mx.clear_cache()
    print(f"Ready: {mx.get_active_memory() / 1024**3:.2f} GB")

    # Build 16K prompts
    encoded = build_encoded(tokenizer, target_tokens=16384, n=4)
    print(f"Prompt lengths: {[len(e) for e in encoded]}")

    results = []

    # ── Sweep 1: Pool size ──
    print(f"\n{'━' * 75}")
    print("  Sweep 1: Pool Size (k=auto, chunk=1024, maint=8)")
    print(f"{'━' * 75}")

    for pool in [153, 128, 100, 80]:
        r = run_config(model, tokenizer, ctx, encoded,
                       pool_size=pool, label=f"pool-{pool}")
        results.append(r)

    # ── Sweep 2: k override (at best pool from sweep 1) ──
    # Find best pool (highest TG with quality PASS)
    passing = [r for r in results if r["quality_pass"]]
    best_pool_r = max(passing, key=lambda r: r["tg_tps"]) if passing else results[0]
    best_pool = best_pool_r["pool_size"]
    if isinstance(best_pool, str) or best_pool is None:
        best_pool = 153
    print(f"\n  >> Best pool from sweep 1: {best_pool} "
          f"(TG={best_pool_r['tg_tps']} tok/s)")

    print(f"\n{'━' * 75}")
    print(f"  Sweep 2: k Override (pool={best_pool}, chunk=1024, maint=8)")
    print(f"{'━' * 75}")

    for k in [8, 6, 5, 4]:
        r = run_config(model, tokenizer, ctx, encoded,
                       pool_size=best_pool, k_override=k,
                       label=f"k-{k}")
        results.append(r)

    # ── Sweep 3: maintenance interval (at best pool + best k) ──
    passing_k = [r for r in results if r["quality_pass"] and "k-" in r["label"]]
    best_k_r = max(passing_k, key=lambda r: r["tg_tps"]) if passing_k else None
    best_k = best_k_r["k_override"] if best_k_r else None

    print(f"\n{'━' * 75}")
    print(f"  Sweep 3: Maintenance Interval (pool={best_pool}, k={best_k or 'auto'})")
    print(f"{'━' * 75}")

    for maint in [4, 8, 16, 32]:
        r = run_config(model, tokenizer, ctx, encoded,
                       pool_size=best_pool, k_override=best_k,
                       maint_interval=maint,
                       label=f"maint-{maint}")
        results.append(r)

    # ── Sweep 4: chunk size (at best combo) ──
    passing_m = [r for r in results if r["quality_pass"] and "maint-" in r["label"]]
    best_m_r = max(passing_m, key=lambda r: r["tg_tps"]) if passing_m else None
    best_maint = best_m_r["maint_interval"] if best_m_r else 8

    print(f"\n{'━' * 75}")
    print(f"  Sweep 4: Chunk Size (pool={best_pool}, k={best_k or 'auto'}, maint={best_maint})")
    print(f"{'━' * 75}")

    for chunk in [512, 1024, 2048, 4096]:
        r = run_config(model, tokenizer, ctx, encoded,
                       pool_size=best_pool, k_override=best_k,
                       maint_interval=best_maint, chunk_size=chunk,
                       label=f"chunk-{chunk}")
        results.append(r)

    # ── Summary ──
    print(f"\n{'=' * 85}")
    print("  TURBO SWEEP SUMMARY")
    print(f"{'=' * 85}")

    print(f"\n  {'Label':<15} {'Pool':>5} {'k':>4} {'Maint':>5} {'Chunk':>5} "
          f"{'PP':>7} {'TG':>7} {'Step':>7} {'TTFT':>6} {'Peak':>7} {'Qual':>5}")
    print(f"  {'─' * 82}")

    for r in results:
        pool_str = str(r["pool_size"])[:5]
        k_str = str(r["k_override"])[:4]
        print(f"  {r['label']:<15} {pool_str:>5} {k_str:>4} {r['maint_interval']:>5} "
              f"{r['chunk_size']:>5} {r['pp_tps']:>6.0f} {r['tg_tps']:>6.1f} "
              f"{r['avg_step_ms']:>6.1f} {r['ttft_first']:>5.1f}s "
              f"{r['mem_peak_gb']:>6.2f}G {r['quality']:>5}")

    # Best config
    all_passing = [r for r in results if r["quality_pass"]]
    if all_passing:
        best = max(all_passing, key=lambda r: r["tg_tps"])
        baseline = results[0]  # pool-153
        tg_gain = (best["tg_tps"] - baseline["tg_tps"]) / baseline["tg_tps"] * 100
        mem_saved = baseline["mem_peak_gb"] - best["mem_peak_gb"]

        print(f"\n  ★ BEST CONFIG: {best['label']}")
        print(f"    Pool={best['pool_size']}, k={best['k_override']}, "
              f"maint={best['maint_interval']}, chunk={best['chunk_size']}")
        print(f"    TG: {baseline['tg_tps']:.1f} → {best['tg_tps']:.1f} tok/s "
              f"({tg_gain:+.1f}%)")
        print(f"    Peak: {baseline['mem_peak_gb']:.2f} → {best['mem_peak_gb']:.2f}G "
              f"({mem_saved:+.2f}G)")
        print(f"    Quality: {best['quality']}")

    # Save
    out = "/Users/lisihao/FlashMLX/.solar/turbo-sweep-results.json"
    with open(out, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "results": results}, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
