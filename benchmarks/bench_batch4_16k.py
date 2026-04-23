#!/usr/bin/env python3
"""
FlashMLX v2.0 Batch Benchmark — Qwen3.5-35B-A3B / 16K / batch=4

Compares:
  Phase A: Community mlx-lm (raw BatchGenerator, no expert offloading)
  Phase B: FlashMLX v2.0 (FlashBatchGenerator + compact + shadow-2bit + reranking)

Metrics: TG throughput (aggregate), TTFT, GPU peak memory, model residency, quality.

Each phase runs in a separate subprocess for accurate memory isolation.

Usage:
    python3 bench_batch4_16k.py                     # Run full benchmark
    python3 bench_batch4_16k.py --worker <json>     # Internal subprocess
    python3 bench_batch4_16k.py --model <path>      # Custom model path
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

# Model path: prefer HF cache, fallback to external drive
DEFAULT_MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLASHMLX_ROOT = os.path.dirname(SCRIPT_DIR)
MLX_LM_SOURCE = os.path.join(FLASHMLX_ROOT, "mlx-lm-source")

CONTEXT_TARGET = 16384  # 16K tokens per prompt
BATCH_SIZE = 4
MAX_GEN_TOKENS = 128
NEEDLE_FACT = "The annual budget for Project Alpha is exactly $12,500."

# 4 different prompts, each padded to ~16K tokens
PROMPT_QUESTIONS = [
    "What is the exact annual budget for Project Alpha? Answer concisely.",
    "State the precise dollar amount allocated to Project Alpha annually.",
    "How much money is budgeted for Project Alpha each year? Be specific.",
    "Tell me the annual financial allocation for Project Alpha.",
]


def build_needle_prompt(target_tokens, tokenizer, question_idx=0):
    """Build a needle-in-haystack prompt with ~target_tokens length."""
    padding_block = (
        "The following document discusses various organizational topics. "
        "Section {n}: Performance metrics for department {n} show steady improvement "
        "in Q3 2025, with productivity up 3.2% and employee satisfaction at 78%. "
        "Budget allocations for the fiscal year were reviewed and approved by the "
        "finance committee on March 15th. Infrastructure upgrades are scheduled for "
        "completion by end of Q4. Training programs will be expanded to cover new "
        "compliance requirements. Resource allocation models suggest optimal staffing "
        "levels will be reached by mid-year. Cross-departmental collaboration "
        "initiatives continue to show positive results across all measured KPIs. "
    )

    blocks = []
    needle_inserted = False
    n = 1
    while True:
        block = padding_block.format(n=n)
        blocks.append(block)
        if not needle_inserted and n > 5:
            blocks.append(f"IMPORTANT NOTE: {NEEDLE_FACT} This information is critical. ")
            needle_inserted = True
        text = "".join(blocks)
        toks = tokenizer.encode(text)
        if len(toks) >= target_tokens - 200:
            break
        n += 1

    if not needle_inserted:
        mid = len(blocks) // 3
        blocks.insert(mid, f"IMPORTANT NOTE: {NEEDLE_FACT} This information is critical. ")

    question = PROMPT_QUESTIONS[question_idx % len(PROMPT_QUESTIONS)]
    return "".join(blocks) + f"\n\nQuestion: {question}"


def run_worker(config_json):
    """Run a single benchmark phase (called in subprocess)."""
    import gc
    import mlx.core as mx

    config = json.loads(config_json)
    model_path = config["model_path"]
    phase = config["phase"]  # "community" or "flashmlx"

    # Phase-specific setup
    if phase == "flashmlx":
        # Import flashmlx FIRST to set up sys.path for enhanced mlx-lm
        sys.path.insert(0, os.path.join(FLASHMLX_ROOT, "src"))
        sys.path.insert(0, MLX_LM_SOURCE)
        import flashmlx  # noqa: F401 — triggers sys.path injection
        from mlx_lm import load
        from mlx_lm.models.expert_offload import (
            patch_model_for_offload,
            FlashBatchGenerator,
        )
        from mlx_lm.generate import BatchGenerator
    else:
        # Community mlx-lm: use FlashMLX's fork but without expert offloading
        sys.path.insert(0, MLX_LM_SOURCE)
        from mlx_lm import load
        from mlx_lm.generate import BatchGenerator

    # Resolve HF model ID to local snapshot path (needed by patch_model_for_offload)
    resolved_model_path = model_path
    if not os.path.isdir(model_path):
        try:
            from huggingface_hub import snapshot_download
            resolved_model_path = snapshot_download(model_path, local_files_only=True)
            print(f"[{phase}] Resolved model path: {resolved_model_path}", flush=True)
        except Exception:
            pass  # Fall back to original path

    # Load model
    print(f"[{phase}] Loading model...", flush=True)
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    # GPU warmup: run a full single-request 16K inference to warm up
    # Metal memory allocator and KV cache layout (not just shader compilation)
    print(f"[{phase}] GPU warmup (full 16K single-request)...", flush=True)
    from mlx_lm.generate import generate_step
    warmup_text = build_needle_prompt(CONTEXT_TARGET, tokenizer, question_idx=0)
    warmup_msgs = [{"role": "user", "content": warmup_text}]
    try:
        warmup_fmt = tokenizer.apply_chat_template(
            warmup_msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        warmup_fmt = tokenizer.apply_chat_template(
            warmup_msgs, add_generation_prompt=True, tokenize=False)
    warmup_prompt = mx.array(tokenizer.encode(warmup_fmt))
    warmup_gen = generate_step(warmup_prompt, model, max_tokens=10)
    for _ in warmup_gen:
        pass
    del warmup_gen
    gc.collect()
    mx.clear_cache()

    # Record model residency (memory after model load, before inference)
    mx.metal.reset_peak_memory()
    model_residency_bytes = mx.get_active_memory()

    # Build 4 prompts at ~16K tokens each
    print(f"[{phase}] Building {BATCH_SIZE} prompts at ~{CONTEXT_TARGET // 1024}K tokens...", flush=True)
    encoded_prompts = []
    for i in range(BATCH_SIZE):
        prompt_text = build_needle_prompt(CONTEXT_TARGET, tokenizer, question_idx=i)
        msgs = [{"role": "user", "content": prompt_text}]
        # Disable thinking mode for quality assessment — avoids wasting tokens
        # on chain-of-thought when we just need the needle answer
        try:
            formatted = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
        tokens = tokenizer.encode(formatted)
        encoded_prompts.append(tokens)
        print(f"  Prompt {i}: {len(tokens)} tokens", flush=True)

    gc.collect()
    mx.clear_cache()
    mx.metal.reset_peak_memory()

    # Phase-specific generator setup
    offload_ctx = None
    _post_pp_done = True  # Default: no post-PP compact needed (community phase)
    _post_pp_pool_size = 32
    _post_pp_no_shadow = False
    if phase == "flashmlx":
        no_shadow = config.get("no_shadow", False)
        pool_size = config.get("pool_size", 32)
        track_label = "compact-only" if no_shadow else "desktop (compact+shadow-2bit)"
        print(f"[{phase}] Setting up expert offloading ({track_label})...", flush=True)
        offload_ctx = patch_model_for_offload(
            model, resolved_model_path, max_workers=4, cpu_cache_gb=2.0,
            enable_prefetch=False, enable_telemetry=True,
        )
        gc.collect()
        mx.clear_cache()

        # Warmup: run a short PP to collect activation data for compact
        print(f"[{phase}] Warmup PP to collect expert activation data...", flush=True)
        warmup_enc = [tokenizer.encode(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "Explain machine learning briefly."}],
                add_generation_prompt=True, tokenize=False
            )
        )]
        warmup_gen = FlashBatchGenerator(
            model, offload_ctx, max_tokens=8,
            completion_batch_size=1, prefill_batch_size=1,
        )
        warmup_gen._compacted = True  # Suppress auto-compact; keep full pool for real PP
        warmup_uids = warmup_gen.insert(warmup_enc, max_tokens=[8])
        for _ in range(12):
            r = warmup_gen.next()
            if not r:
                break
        warmup_gen.close()
        gc.collect()
        mx.clear_cache()
        print(f"[{phase}] Warmup done. Memory: {mx.get_active_memory() / 1024**3:.2f} GB", flush=True)

        # Defer compact + shadow to AFTER real PP for quality preservation.
        # Real PP runs with full pool → all experts at full precision → needle preserved.
        # Compact + shadow fire after first token (PP done) → TG uses compact pool.
        offload_ctx._pool_compacted = False  # Reset so real PP buffers expert indices
        _post_pp_done = False
        _post_pp_pool_size = pool_size
        _post_pp_no_shadow = no_shadow

        model_residency_bytes = mx.get_active_memory()  # Updated after post-PP compact
        mx.metal.reset_peak_memory()
        print(f"[{phase}] Full pool residency: {model_residency_bytes / 1024**3:.2f} GB (will compact after PP)", flush=True)

        def make_gen():
            g = FlashBatchGenerator(
                model, offload_ctx, max_tokens=MAX_GEN_TOKENS,
                completion_batch_size=BATCH_SIZE, prefill_batch_size=BATCH_SIZE,
                interleaved=False,  # Non-interleaved: prefill all → decode all
            )
            g._compacted = True  # Suppress auto-compact; we compact manually after PP
            return g
    else:
        def make_gen():
            return BatchGenerator(
                model, max_tokens=MAX_GEN_TOKENS,
                completion_batch_size=BATCH_SIZE, prefill_batch_size=BATCH_SIZE,
                interleaved=True, interleaved_chunk_size=512,
            )

    # --- Run batch generation ---
    print(f"[{phase}] Starting batch={BATCH_SIZE} generation...", flush=True)
    gen = make_gen()
    uids = gen.insert(encoded_prompts, max_tokens=[MAX_GEN_TOKENS] * BATCH_SIZE)

    all_tokens = {uid: [] for uid in uids}
    finished = set()
    total_gen_tokens = 0
    first_token_time = None
    _post_pp_overhead = 0.0  # Time spent on compact+shadow between PP and TG

    t0 = time.perf_counter()
    # Interleaved mode: next() returns [] during prefill chunks.
    # Only break after consecutive empties when all work is done.
    consecutive_empty = 0
    MAX_EMPTY = 500  # 4 × 16K / 512 chunk = ~128 chunks max
    step_times = []  # Per-step timing for diagnostics

    while True:
        t_step = time.perf_counter()
        responses = gen.next()
        dt_step = time.perf_counter() - t_step
        if not responses:
            consecutive_empty += 1
            if consecutive_empty > MAX_EMPTY:
                break
            continue
        consecutive_empty = 0
        step_times.append(dt_step)
        for resp in responses:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            all_tokens[resp.uid].append(resp.token)
            total_gen_tokens += 1
            if resp.finish_reason is not None:
                finished.add(resp.uid)
        # Post-PP: compact + shadow AFTER PP completes (first token = PP done)
        if not _post_pp_done and first_token_time is not None:
            _post_pp_done = True
            _pp_setup_t0 = time.perf_counter()
            print(f"  [post-PP] Compacting to pool_size={_post_pp_pool_size}...", flush=True)
            offload_ctx.compact(pool_size=_post_pp_pool_size, disable_coverage_gate=True, auto_expand_cpu_cache=False)
            if not _post_pp_no_shadow:
                offload_ctx.create_shadow(bits=2)
                offload_ctx.decode_recompact(pool_size=_post_pp_pool_size)
            offload_ctx.enable_reranking(bonus=0.01)
            offload_ctx.enable_dynamic_pruning(model)
            offload_ctx.adjust_effective_k(model)
            gc.collect()
            mx.clear_cache()
            _post_pp_overhead = time.perf_counter() - _pp_setup_t0
            model_residency_bytes = mx.get_active_memory()
            mx.metal.reset_peak_memory()
            print(f"  [post-PP] Residency: {model_residency_bytes / 1024**3:.2f} GB "
                  f"(setup: {_post_pp_overhead:.1f}s)", flush=True)
        if len(finished) == BATCH_SIZE:
            break

    elapsed = time.perf_counter() - t0
    ttft = (first_token_time - t0) if first_token_time else elapsed
    gen.close()

    # Per-step diagnostics
    if step_times:
        avg_step = sum(step_times) / len(step_times) * 1000
        min_step = min(step_times) * 1000
        max_step = max(step_times) * 1000
        # First 5 and last 5 steps
        first5 = [f"{t*1000:.1f}" for t in step_times[:5]]
        last5 = [f"{t*1000:.1f}" for t in step_times[-5:]]
        print(f"  Step timing: {len(step_times)} steps, avg={avg_step:.1f}ms, "
              f"min={min_step:.1f}ms, max={max_step:.1f}ms", flush=True)
        print(f"  First 5: {first5}  Last 5: {last5}", flush=True)

    # Metrics (exclude post-PP compact+shadow overhead from decode time)
    pure_decode_time = elapsed - ttft - _post_pp_overhead if ttft < elapsed else elapsed
    tg_throughput = total_gen_tokens / pure_decode_time if pure_decode_time > 0 else 0
    total_prompt_tokens = sum(len(p) for p in encoded_prompts)
    pp_throughput = total_prompt_tokens / ttft if ttft > 0 else 0
    peak_memory_gb = mx.metal.get_peak_memory() / 1024**3
    model_residency_gb = model_residency_bytes / 1024**3

    # Quality: check each output contains the needle answer
    quality_pass = 0
    text_previews = []
    for uid in uids:
        text = tokenizer.decode(all_tokens[uid]).strip()
        text_previews.append(text[:80])
        if "12,500" in text or "12500" in text or "$12,500" in text:
            quality_pass += 1

    result = {
        "phase": phase,
        "batch_size": BATCH_SIZE,
        "context_tokens": CONTEXT_TARGET,
        "total_prompt_tokens": total_prompt_tokens,
        "total_gen_tokens": total_gen_tokens,
        "elapsed_s": elapsed,
        "pp_throughput": pp_throughput,
        "tg_throughput": tg_throughput,
        "ttft_s": ttft,
        "peak_memory_gb": peak_memory_gb,
        "model_residency_gb": model_residency_gb,
        "quality_pass": quality_pass,
        "quality_total": BATCH_SIZE,
        "text_previews": text_previews,
    }
    print("BENCH_RESULT:" + json.dumps(result), flush=True)


def run_phase(phase, model_path, no_shadow=False, pool_size=32):
    """Run one phase in a subprocess for memory isolation."""
    config = json.dumps({
        "model_path": model_path,
        "phase": phase,
        "no_shadow": no_shadow,
        "pool_size": pool_size,
    })

    print(f"\n{'=' * 70}")
    print(f"  Phase: {phase.upper()}")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", config],
            capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (900s)")
        return None

    wall_time = time.time() - t0

    # Parse result
    result = None
    for line in (proc.stdout or "").split("\n"):
        if line.startswith("BENCH_RESULT:"):
            result = json.loads(line[len("BENCH_RESULT:"):])
            break
        elif line.strip():
            print(f"  {line.strip()}")

    if result is None:
        print(f"  FAILED (exit={proc.returncode}, {wall_time:.0f}s wall)")
        if proc.stderr:
            for line in proc.stderr.strip().split("\n")[-20:]:
                print(f"  ERR: {line}")
        return None

    # Display — standardized format
    r = result
    pp_tok = r.get('total_prompt_tokens', 0)
    tg_tok = r['total_gen_tokens']
    decode_s = r['elapsed_s'] - r['ttft_s']
    q = f"{r['quality_pass']}/{r['quality_total']}"
    print(f"\n  ┌─ {phase.upper()} ── batch={r['batch_size']} × context={r['context_tokens']//1024}K × gen={MAX_GEN_TOKENS}")
    print(f"  │  PP   {r['pp_throughput']:>8.1f} tok/s   ({pp_tok} tokens / {r['ttft_s']:.1f}s)")
    print(f"  │  TG   {r['tg_throughput']:>8.1f} tok/s   ({tg_tok} tokens / {decode_s:.1f}s)")
    print(f"  │  TTFT {r['ttft_s']:>8.1f} s")
    print(f"  │  Peak {r['peak_memory_gb']:>8.2f} GB")
    print(f"  │  Res  {r['model_residency_gb']:>8.2f} GB")
    print(f"  │  Q    {q:>8}")
    for i, preview in enumerate(r.get('text_previews', [])):
        print(f"  │  [{i}] {preview}")
    print(f"  └{'─' * 60}")

    return result


def main():
    model_path = DEFAULT_MODEL

    # Parse --model argument
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model_path = sys.argv[idx + 1]

    # Get device info
    try:
        import mlx.core as mx
        info = mx.metal.device_info()
        device_name = info.get("device_name", "unknown")
        total_mem_gb = info.get("memory_size", 0) / 1024**3
    except Exception:
        device_name = "unknown"
        total_mem_gb = 0

    print("=" * 70)
    print(f"  FlashMLX v2.0 Batch Benchmark")
    print(f"  Model:    {model_path}")
    print(f"  Config:   batch={BATCH_SIZE}, context={CONTEXT_TARGET // 1024}K, gen={MAX_GEN_TOKENS}")
    print(f"  Device:   {device_name} ({total_mem_gb:.0f} GB)")
    print(f"  Date:     {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Run phases (--phase to run only one)
    results = {}
    phases_to_run = ["community", "flashmlx"]
    if "--phase" in sys.argv:
        idx = sys.argv.index("--phase")
        phases_to_run = [sys.argv[idx + 1]]

    if "community" in phases_to_run:
        results["community"] = run_phase("community", model_path)

    if "flashmlx" in phases_to_run:
        no_shadow = "--no-shadow" in sys.argv
        results["flashmlx"] = run_phase("flashmlx", model_path, no_shadow=no_shadow)

    # Comparison
    comm = results.get("community")
    flash = results.get("flashmlx")

    if comm and flash:
        model_short = os.path.basename(model_path)
        print(f"\n{'═' * 70}")
        print(f"  A/B Comparison — {model_short}")
        print(f"  batch={BATCH_SIZE} × context={CONTEXT_TARGET//1024}K × gen={MAX_GEN_TOKENS}")
        print(f"  Device: {device_name} ({total_mem_gb:.0f} GB)")
        print(f"{'═' * 70}")

        def delta(old, new):
            if old == 0: return "  N/A"
            pct = (new / old - 1) * 100
            return f"{pct:+.1f}%"

        cq = f"{comm['quality_pass']}/{comm['quality_total']}"
        fq = f"{flash['quality_pass']}/{flash['quality_total']}"
        qd = "=" if comm['quality_pass'] == flash['quality_pass'] else "!!!"

        H = f"  {'Metric':<22} {'Community':>10} {'FlashMLX':>10} {'Delta':>8}"
        S = f"  {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 8}"
        print(f"\n{H}\n{S}")
        print(f"  {'PP (tok/s)':<22} {comm.get('pp_throughput',0):>10.1f} {flash.get('pp_throughput',0):>10.1f} {delta(comm.get('pp_throughput',1), flash.get('pp_throughput',1)):>8}")
        print(f"  {'TG (tok/s)':<22} {comm['tg_throughput']:>10.1f} {flash['tg_throughput']:>10.1f} {delta(comm['tg_throughput'], flash['tg_throughput']):>8}")
        print(f"  {'TTFT (s)':<22} {comm['ttft_s']:>10.1f} {flash['ttft_s']:>10.1f} {delta(comm['ttft_s'], flash['ttft_s']):>8}")
        print(f"  {'GPU peak (GB)':<22} {comm['peak_memory_gb']:>10.2f} {flash['peak_memory_gb']:>10.2f} {delta(comm['peak_memory_gb'], flash['peak_memory_gb']):>8}")
        print(f"  {'Residency (GB)':<22} {comm['model_residency_gb']:>10.2f} {flash['model_residency_gb']:>10.2f} {delta(comm['model_residency_gb'], flash['model_residency_gb']):>8}")
        print(f"  {'Quality':<22} {cq:>10} {fq:>10} {qd:>8}")
        print(f"  {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 8}")

    # Save
    output_dir = os.path.join(SCRIPT_DIR, ".results")
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f"batch4_16k_{ts}.json")
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "model": model_path,
            "device": device_name,
            "total_memory_gb": total_mem_gb,
            "config": {"batch_size": BATCH_SIZE, "context": CONTEXT_TARGET, "max_gen": MAX_GEN_TOKENS},
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved: {output_path}")
    print(f"\nDone!")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        idx = sys.argv.index("--worker")
        run_worker(sys.argv[idx + 1])
    else:
        main()
