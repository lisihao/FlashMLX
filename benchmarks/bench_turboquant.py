#!/usr/bin/env python3
"""
TurboQuant Flat Buffer Benchmark — Full Metrics
================================================
Uses generate_step (same methodology as bench_ab_final.py) for accurate
PP/TG/TTOF/memory measurement.

Model: Qwen3-8B (dense, 8 KV heads, head_dim=128, 36 layers)
Contexts: 2K, 8K, 16K
Gen: 200 tokens
Metrics: PP tok/s, TG tok/s, TTOF (s), KV Cache MB (PP peak + TG steady), Memory Savings
"""

import gc, json, os, sys, time

MLX_LM_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx-lm-source")
sys.path.insert(0, MLX_LM_SOURCE)

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"
CAL_PATH = "/Users/lisihao/FlashMLX/calibrations/am_calibration_qwen3-8b_2.0x.pkl"
MAX_GEN = 200
CONTEXTS = [2048, 8192, 16384]

# Test configs: (label, cache_kwargs)
CONFIGS = [
    ("standard", {}),
    ("triple+bf16", {"kv_cache": "triple"}),
    ("triple+q8_0", {"kv_cache": "triple", "kv_flat_quant": "q8_0"}),
    ("triple+q4_0", {"kv_cache": "triple", "kv_flat_quant": "q4_0"}),
    ("triple+tq", {"kv_cache": "triple", "kv_flat_quant": "turboquant"}),
    ("scored+bf16", {"kv_cache": "scored_pq", "kv_calibration": CAL_PATH}),
    ("scored+q8_0", {"kv_cache": "scored_pq", "kv_calibration": CAL_PATH, "kv_flat_quant": "q8_0"}),
    ("scored+q4_0", {"kv_cache": "scored_pq", "kv_calibration": CAL_PATH, "kv_flat_quant": "q4_0"}),
    ("scored+tq", {"kv_cache": "scored_pq", "kv_calibration": CAL_PATH, "kv_flat_quant": "turboquant"}),
]


def measure_cache_bytes(cache_list):
    """Measure KV cache memory from array nbytes across all layers."""
    total = 0
    for c in cache_list:
        if c is None:
            continue
        if hasattr(c, 'keys') and c.keys is not None:
            total += c.keys.nbytes + c.values.nbytes
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


def build_prompt(tokenizer, target_tokens):
    """Build prompt targeting exact token count."""
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


def run_one(model, tokenizer, ctx, label, cache_kwargs):
    """Run single benchmark with generate_step. Returns metrics dict."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    gc.collect()
    mx.metal.clear_cache()

    prompt_text = build_prompt(tokenizer, ctx)
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_array = mx.array(tokenizer.encode(formatted))
    prompt_len = prompt_array.shape[0]

    # Create cache
    prompt_cache = make_prompt_cache(model, **cache_kwargs)

    mx.metal.reset_peak_memory()
    mem_before = mx.metal.get_active_memory() / 1e6

    # PP (prefill) — time to first token
    t0 = time.perf_counter()
    gen = generate_step(prompt_array, model, prompt_cache=prompt_cache, max_tokens=MAX_GEN)
    first_token, _ = next(gen)
    ttof = time.perf_counter() - t0
    pp_tps = prompt_len / ttof

    kv_pp_peak_mb = round(max(0, mx.metal.get_peak_memory() / 1e6 - mem_before))

    # TG (token generation)
    ft = first_token if isinstance(first_token, int) else first_token.item()
    tokens = [ft]
    t_tg = time.perf_counter()

    for tok, _ in gen:
        t = tok if isinstance(tok, int) else tok.item()
        tokens.append(t)
        if len(tokens) >= MAX_GEN:
            break

    t_tg_end = time.perf_counter()
    tg_time = t_tg_end - t_tg
    tg_tps = (len(tokens) - 1) / tg_time if tg_time > 0 else 0

    kv_tg_mb = round(measure_cache_bytes(prompt_cache))
    peak_mb = round(mx.metal.get_peak_memory() / 1e6)

    return {
        "label": label,
        "ctx": ctx,
        "actual_tokens": prompt_len,
        "gen_tokens": len(tokens),
        "pp_tps": round(pp_tps, 1),
        "tg_tps": round(tg_tps, 1),
        "ttof_s": round(ttof, 2),
        "kv_pp_peak_mb": kv_pp_peak_mb,
        "kv_tg_mb": kv_tg_mb,
        "peak_mb": peak_mb,
    }


def main():
    import mlx.core as mx
    from mlx_lm import load

    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    gc.collect()
    model_mb = round(mx.metal.get_active_memory() / 1e6)
    print(f"Model loaded: {model_mb} MB\n")

    results = []

    for ctx in CONTEXTS:
        print(f"{'='*80}")
        print(f"Context: {ctx} tokens | Gen: {MAX_GEN} tokens")
        print(f"{'='*80}")
        print(f"  {'Config':20s} | {'PP':>8s} | {'TG':>8s} | {'TTOF':>7s} | {'KV PP':>8s} | {'KV TG':>8s} | {'KV Save':>8s}")
        print(f"  {'':20s} | {'tok/s':>8s} | {'tok/s':>8s} | {'(s)':>7s} | {'Peak MB':>8s} | {'MB':>8s} | {'vs std':>8s}")
        print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

        std_kv = None
        for label, kwargs in CONFIGS:
            r = run_one(model, tokenizer, ctx, label, kwargs)
            results.append(r)

            if std_kv is None:
                std_kv = r["kv_tg_mb"]

            kv_save = ""
            if std_kv and std_kv > 0 and r["kv_tg_mb"] < std_kv:
                pct = (1 - r["kv_tg_mb"] / std_kv) * 100
                kv_save = f"-{pct:.0f}%"

            print(f"  {label:20s} | {r['pp_tps']:8.1f} | {r['tg_tps']:8.1f} | {r['ttof_s']:7.2f} | {r['kv_pp_peak_mb']:8d} | {r['kv_tg_mb']:8d} | {kv_save:>8s}")

        print()

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), ".solar", "bench-turboquant-results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": "Qwen3-8B", "model_mb": model_mb, "results": results}, f, indent=2)
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
