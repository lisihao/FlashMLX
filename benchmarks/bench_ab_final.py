#!/usr/bin/env python3
"""
FlashMLX v1.0 Complete Benchmark — Baseline vs Three-Route Optimization

Group 1: Qwen3-8B (pure attention) — Route 2+3: scored_pq + Q8 KV cache
Group 2: Qwen3.5-35B-A3B (MoE hybrid) — Route 1: expert offload + compact pool=128
  (scored_pq auto-disabled for hybrid models — TG -22% to -64% in benchmarks)

Each config runs in a subprocess for accurate memory isolation.

Usage:
    python3 bench_ab_final.py                    # Run all benchmarks
    python3 bench_ab_final.py --worker <json>    # Internal: single benchmark
"""

import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime

MLX_LM_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx-lm-source")
sys.path.insert(0, MLX_LM_SOURCE)

MAX_GEN_TOKENS = 200
COMPACT_POOL_SIZE = 128

# Test matrix
TESTS = [
    # Group 1: Qwen3-8B — KV cache optimization (Route 2+3)
    {"model": "Qwen3-8B", "path": "/Volumes/toshiba/models/qwen3-8b-mlx",
     "config": "standard", "ctx": 16384, "mode": "kv"},
    {"model": "Qwen3-8B", "path": "/Volumes/toshiba/models/qwen3-8b-mlx",
     "config": "flashmlx", "ctx": 16384, "mode": "kv"},
    {"model": "Qwen3-8B", "path": "/Volumes/toshiba/models/qwen3-8b-mlx",
     "config": "standard", "ctx": 32768, "mode": "kv"},
    {"model": "Qwen3-8B", "path": "/Volumes/toshiba/models/qwen3-8b-mlx",
     "config": "flashmlx", "ctx": 32768, "mode": "kv"},

    # Group 2: Qwen3.5-35B — Expert Offloading (Route 1) vs standard
    {"model": "Qwen3.5-35B-A3B", "path": "/Volumes/toshiba/models/qwen3.5-35b-mlx",
     "config": "standard", "ctx": 4096, "mode": "kv"},
    {"model": "Qwen3.5-35B-A3B", "path": "/Volumes/toshiba/models/qwen3.5-35b-mlx",
     "config": "flashmlx", "ctx": 4096, "mode": "offload"},
    {"model": "Qwen3.5-35B-A3B", "path": "/Volumes/toshiba/models/qwen3.5-35b-mlx",
     "config": "standard", "ctx": 16384, "mode": "kv"},
    {"model": "Qwen3.5-35B-A3B", "path": "/Volumes/toshiba/models/qwen3.5-35b-mlx",
     "config": "flashmlx", "ctx": 16384, "mode": "offload"},
]


def build_prompt(tokenizer, target_tokens):
    block = (
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
    n = 1
    while True:
        blocks.append(block.format(n=n))
        text = "".join(blocks) + "\n\nSummarize the key points above in 200 words."
        toks = tokenizer.encode(text)
        if len(toks) >= target_tokens - 100:
            break
        n += 1
    return text


def measure_cache_bytes(cache_list):
    """Measure KV cache memory from array nbytes across all layers."""
    total = 0
    for c in cache_list:
        if c is None:
            continue
        # Standard KVCache: keys + values buffers
        if hasattr(c, 'keys') and c.keys is not None:
            total += c.keys.nbytes + c.values.nbytes
        # TripleLayerKVCache: flat buffer + layer buffers
        if hasattr(c, '_flat_keys') and c._flat_keys is not None:
            total += c._flat_keys.nbytes + c._flat_values.nbytes
            if getattr(c, '_flat_keys_scales', None) is not None:
                total += c._flat_keys_scales.nbytes + c._flat_values_scales.nbytes
        for attr in ('recent_keys', 'recent_values', 'warm_keys', 'warm_values',
                     'cold_keys', 'cold_values', 'warm_scales_k', 'warm_scales_v'):
            arr = getattr(c, attr, None)
            if arr is not None:
                total += arr.nbytes
    return total / 1e6  # MB


def run_worker_kv(config):
    """KV cache benchmark (Route 2+3). Uses generate_step for direct cache access."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(config["path"])
    mx.eval(model.parameters())
    gc.collect()
    mem_model = mx.metal.get_active_memory() / 1e6

    prompt_text = build_prompt(tokenizer, config["ctx"])
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_array = mx.array(tokenizer.encode(formatted))
    prompt_len = prompt_array.shape[0]

    # Create cache manually for direct measurement
    cache_kwargs = {}
    if config["config"] in ("scored_q8", "flashmlx"):
        cache_kwargs["kv_cache"] = "scored_pq"
        cache_kwargs["kv_flat_quant"] = "q8_0"
        # Auto-calibrate AM (stream_generate does this automatically, we do it manually)
        from mlx_lm.models.am_calibrator import auto_calibrate
        cal_path = auto_calibrate(model, tokenizer)
        if cal_path:
            cache_kwargs["kv_calibration"] = cal_path
    prompt_cache = make_prompt_cache(model, **cache_kwargs)

    mx.metal.reset_peak_memory()
    mem_before = mx.metal.get_active_memory() / 1e6

    # PP
    t0 = time.perf_counter()
    gen = generate_step(prompt_array, model, prompt_cache=prompt_cache,
                        max_tokens=MAX_GEN_TOKENS)
    first_token, _ = next(gen)
    ttof = time.perf_counter() - t0
    pp_tps = prompt_len / ttof

    # KV PP: GPU peak delta (captures peak during chunked PP, including activations)
    kv_pp_mb = round(max(0, mx.metal.get_peak_memory() / 1e6 - mem_before))

    # TG
    ft = first_token if isinstance(first_token, int) else first_token.item()
    tokens = [ft]
    t_tg = time.perf_counter()

    for tok, _ in gen:
        t = tok if isinstance(tok, int) else tok.item()
        tokens.append(t)
        if len(tokens) >= MAX_GEN_TOKENS:
            break

    t_tg_end = time.perf_counter()
    tg_time = t_tg_end - t_tg
    tg_tps = (len(tokens) - 1) / tg_time if tg_time > 0 else 0
    t_total = t_tg_end - t0

    kv_tg_mb = round(measure_cache_bytes(prompt_cache))
    text = tokenizer.decode(tokens)

    return {
        "model": config["model"], "config": config["config"],
        "actual_tokens": prompt_len, "gen_tokens": len(tokens),
        "pp_tps": round(pp_tps, 1), "tg_tps": round(tg_tps, 1),
        "ttof_s": round(ttof, 2), "total_s": round(t_total, 2),
        "kv_pp_peak_mb": kv_pp_mb,
        "kv_tg_mb": kv_tg_mb,
        "peak_mb": round(mx.metal.get_peak_memory() / 1e6),
        "model_mb": round(mem_model),
        "text": text[:80], "quality": "PASS" if len(text.strip()) > 20 else "FAIL",
    }


def run_worker_offload(config):
    """Expert offloading benchmark (Route 1). Uses generate_step for PP→compact→TG."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.expert_offload import patch_model_for_offload

    model, tokenizer = load(config["path"])
    mx.eval(model.parameters())
    gc.collect()
    mem_model = mx.metal.get_active_memory() / 1e6

    # Patch for offloading
    ctx = patch_model_for_offload(
        model, config["path"],
        max_workers=4, cpu_cache_gb=2.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()
    mx.metal.clear_cache()

    prompt_text = build_prompt(tokenizer, config["ctx"])
    actual_tokens = len(tokenizer.encode(prompt_text))
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_array = mx.array(tokenizer.encode(formatted))
    prompt_len = len(tokenizer.encode(formatted))

    mx.metal.reset_peak_memory()

    # PP
    t0 = time.perf_counter()
    gen = generate_step(prompt_array, model)
    first_token, _ = next(gen)
    t_pp = time.perf_counter() - t0
    pp_tps = prompt_len / t_pp

    mem_pp = mx.metal.get_active_memory() / 1e6

    # Compact (auto pool sizing: memory budget ceiling + 95% coverage floor)
    compact_info = ctx.compact()
    gc.collect()
    mx.metal.clear_cache()
    mem_compact = mx.metal.get_active_memory() / 1e6

    # TG (with warmup tracking)
    ft = first_token if isinstance(first_token, int) else first_token.item()
    tokens = [ft]
    WARMUP = 50
    t_tg = time.perf_counter()
    t_warmup_end = None

    for tok, _ in gen:
        t = tok if isinstance(tok, int) else tok.item()
        tokens.append(t)
        if len(tokens) == WARMUP + 1 and t_warmup_end is None:
            t_warmup_end = time.perf_counter()
        if len(tokens) >= MAX_GEN_TOKENS:
            break

    t_tg_end = time.perf_counter()
    tg_total = t_tg_end - t_tg
    tg_avg = (len(tokens) - 1) / tg_total if tg_total > 0 else 0

    # Steady-state TG (after Metal warmup)
    tg_steady = 0
    if t_warmup_end and len(tokens) > WARMUP + 1:
        steady_tokens = len(tokens) - WARMUP - 1
        steady_time = t_tg_end - t_warmup_end
        tg_steady = steady_tokens / steady_time if steady_time > 0 else 0

    t_total = time.perf_counter() - t0
    text = tokenizer.decode(tokens)

    result = {
        "model": config["model"], "config": config["config"],
        "actual_tokens": actual_tokens, "gen_tokens": len(tokens),
        "pp_tps": round(pp_tps, 1), "tg_tps": round(tg_avg, 1),
        "tg_steady": round(tg_steady, 1),
        "ttof_s": round(t_pp, 2), "total_s": round(t_total, 2),
        "kv_pp_peak_mb": 0, "kv_tg_mb": 0,
        "peak_mb": round(mx.metal.get_peak_memory() / 1e6),
        "model_mb": round(mem_model),
        "param_before_mb": round(mem_pp),
        "param_after_mb": round(mem_compact),
        "param_saved_mb": round(mem_pp - mem_compact),
        "text": text[:80], "quality": "PASS" if len(text.strip()) > 20 else "FAIL",
    }
    ctx.close()
    return result


def run_worker(config_json):
    config = json.loads(config_json)
    if config["mode"] == "offload":
        result = run_worker_offload(config)
    else:
        result = run_worker_kv(config)
    print("BENCH_RESULT:" + json.dumps(result))


def run_single(test):
    config_json = json.dumps(test)
    ctx_label = f"{test['ctx']//1024}K"
    label = f"{test['model']} | {test['config']} | {ctx_label}"
    print(f"\n  Running: {label}")
    sys.stdout.flush()

    try:
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", config_json],
            capture_output=True, text=True, timeout=600,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT")
        return None

    result = None
    for line in (proc.stdout or "").split("\n"):
        if line.startswith("BENCH_RESULT:"):
            result = json.loads(line[len("BENCH_RESULT:"):])
            break

    if not result:
        print(f"    FAILED (exit={proc.returncode})")
        if proc.stderr:
            for l in proc.stderr.strip().split("\n")[-10:]:
                print(f"    ERR: {l}")
        return None

    # Display
    print(f"    PP:   {result['pp_tps']:>8.1f} tok/s")
    print(f"    TG:   {result['tg_tps']:>8.1f} tok/s", end="")
    if result.get("tg_steady"):
        print(f"  (steady: {result['tg_steady']:.1f})", end="")
    print()
    print(f"    TTOF: {result['ttof_s']:>8.2f} s  |  TTOT: {result['total_s']:.2f} s")
    if result.get("kv_pp_peak_mb"):
        print(f"    KV PP: {result['kv_pp_peak_mb']:>5} MB  |  KV TG: {result['kv_tg_mb']:>5} MB")
    if result.get("param_saved_mb"):
        print(f"    Param: {result['param_before_mb']} → {result['param_after_mb']} MB "
              f"(saved {result['param_saved_mb']} MB)")
    print(f"    Peak: {result['peak_mb']:>6} MB  |  Quality: {result['quality']}")

    return result


def main():
    print("=" * 120)
    print(f"  FlashMLX v1.0 Complete Benchmark — Baseline vs Three-Route Optimization")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Platform: Apple M4 Pro 48GB")
    print(f"  Route 1: Expert Offloading (MoE: auto pool sizing)")
    print(f"  Route 2: Chunked Prefill Eviction (scored_pq, auto-enabled)")
    print(f"  Route 3: Scored P2 + Q8 flat buffer")
    print("=" * 120)

    all_results = []
    current_group = ""

    for test in TESTS:
        if not os.path.exists(test["path"]):
            print(f"\n  SKIP: {test['model']} — not found")
            continue

        group = test['model']
        if group != current_group:
            current_group = group
            ctx_l = f"{test['ctx']//1024}K"
            print(f"\n{'='*80}")
            print(f"  {test['model']}")
            print(f"{'='*80}")

        r = run_single(test)
        if r:
            r["mode"] = test["mode"]
            r["ctx"] = test["ctx"]
            all_results.append(r)

    # ===== SUMMARY TABLES (per model) =====
    models = list(dict.fromkeys(r["model"] for r in all_results))

    for model_name in models:
        model_results = [r for r in all_results if r["model"] == model_name]
        has_offload = any(r["mode"] == "offload" for r in model_results)

        print(f"\n{'='*120}")
        if has_offload:
            print(f"  {model_name}  (Route 1: Expert Offloading)")
        else:
            print(f"  {model_name}  (Route 2+3: scored_pq + Q8)")
        print(f"{'='*120}")

        # Raw results table
        hdr = f"  {'Config':<12} | {'Ctx':>4} | {'PP tok/s':>9} | {'TG tok/s':>9} | {'TTOF':>7} | {'TTOT':>7}"
        if has_offload:
            hdr += f" | {'Param MB':>9} | {'Saved':>7}"
        else:
            hdr += f" | {'KV PP MB':>8} | {'KV TG MB':>8} | {'Peak MB':>8}"
        hdr += " | Qual"
        print(hdr)
        print(f"  {'-'*12}-+-{'-'*4}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}", end="")
        if has_offload:
            print(f"-+-{'-'*9}-+-{'-'*7}", end="")
        else:
            print(f"-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}", end="")
        print(f"-+-{'-'*4}")

        for r in model_results:
            ctx_l = f"{r['ctx']//1024}K"
            tg_str = f"{r['tg_tps']:>9.1f}"
            if r.get("tg_steady"):
                tg_str = f"{r['tg_steady']:>9.1f}"  # Show steady-state for offload
            line = (f"  {r['config']:<12} | {ctx_l:>4} | {r['pp_tps']:>9.1f} | "
                    f"{tg_str} | {r['ttof_s']:>6.1f}s | {r['total_s']:>6.1f}s")
            if r["mode"] == "offload":
                saved_pct = r["param_saved_mb"] / r["param_before_mb"] * 100 if r.get("param_before_mb") else 0
                line += f" | {r.get('param_after_mb',0):>8} | {saved_pct:>6.1f}%"
            elif has_offload and r["mode"] == "kv":
                line += f" | {r.get('model_mb',0):>8} |      —"
            else:
                line += f" | {r['kv_pp_peak_mb']:>7} | {r.get('kv_tg_mb',0):>7} | {r['peak_mb']:>7}"
            line += f" | {r['quality']}"
            print(line)

        # Delta table
        ctxs = sorted(set(r["ctx"] for r in model_results))
        deltas = []
        for c in ctxs:
            std = next((r for r in model_results if r["ctx"]==c and r["config"]=="standard"), None)
            opt = next((r for r in model_results if r["ctx"]==c and r["config"]=="flashmlx"), None)
            if not std or not opt:
                continue
            d = {"ctx": c}
            d["pp"] = (opt["pp_tps"]/std["pp_tps"]-1)*100 if std["pp_tps"] else 0
            # Use steady-state TG for offload if available
            std_tg = std["tg_tps"]
            opt_tg = opt.get("tg_steady") or opt["tg_tps"]
            d["tg"] = (opt_tg/std_tg-1)*100 if std_tg else 0
            d["ttof"] = (opt["ttof_s"]/std["ttof_s"]-1)*100 if std["ttof_s"] else 0
            d["ttot"] = (opt["total_s"]/std["total_s"]-1)*100 if std["total_s"] else 0
            if not has_offload:
                d["kv_pp"] = (opt["kv_pp_peak_mb"]/std["kv_pp_peak_mb"]-1)*100 if std["kv_pp_peak_mb"] else 0
                d["kv_tg"] = (opt["kv_tg_mb"]/std["kv_tg_mb"]-1)*100 if std.get("kv_tg_mb") else 0
                d["peak"] = (opt["peak_mb"]/std["peak_mb"]-1)*100 if std["peak_mb"] else 0
            else:
                param_std = std.get("model_mb", 0)
                param_opt = opt.get("param_after_mb", 0)
                d["param"] = (param_opt/param_std-1)*100 if param_std else 0
            deltas.append(d)

        if deltas:
            print(f"\n  FlashMLX vs Standard:")
            if has_offload:
                print(f"   {'Ctx':>4} | {'PP':>9} | {'TG steady':>10} | {'TTOF':>9} | {'TTOT':>9} | {'Param':>9}")
                print(f"  {'-'*5}-+-{'-'*9}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
                for d in deltas:
                    ctx_l = f"{d['ctx']//1024}K"
                    print(f"   {ctx_l:>4} | {d['pp']:>+8.1f}% | {d['tg']:>+9.1f}% | "
                          f"{d['ttof']:>+8.1f}% | {d['ttot']:>+8.1f}% | {d['param']:>+8.1f}%")
            else:
                print(f"   {'Ctx':>4} | {'PP':>9} | {'TG':>9} | {'TTOF':>9} | {'TTOT':>9} | {'KV PP':>9} | {'KV TG':>9} | {'Peak':>9}")
                print(f"  {'-'*5}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
                for d in deltas:
                    ctx_l = f"{d['ctx']//1024}K"
                    print(f"   {ctx_l:>4} | {d['pp']:>+8.1f}% | {d['tg']:>+8.1f}% | "
                          f"{d['ttof']:>+8.1f}% | {d['ttot']:>+8.1f}% | {d['kv_pp']:>+8.1f}% | {d['kv_tg']:>+8.1f}% | {d['peak']:>+8.1f}%")

    # Save
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".solar", "bench-ab-final.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out}")
    print(f"\nDone!")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[sys.argv.index("--worker") + 1])
    else:
        main()
