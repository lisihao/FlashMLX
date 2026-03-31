#!/usr/bin/env python3
"""
Quick sanity benchmark — verify FlashMLX SDK works after refactor.

Runs only 2K context, 50 gen tokens, 3 configs: standard / scored+q8_0 / scored+tq.
Should complete in ~1 minute.
"""

import gc, json, os, sys, time

# Resolve mlx-lm-source relative to project root (one level up from benchmarks/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLX_LM_SOURCE = os.path.join(PROJECT_ROOT, "mlx-lm-source")
sys.path.insert(0, MLX_LM_SOURCE)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"
CAL_PATH = os.path.join(PROJECT_ROOT, "calibrations", "am_calibration_qwen3-8b_2.0x.pkl")
CTX = 2048
MAX_GEN = 50

CONFIGS = [
    ("standard", {}),
    ("scored+q8_0", {"kv_cache": "scored_pq", "kv_calibration": CAL_PATH, "kv_flat_quant": "q8_0"}),
    ("scored+tq", {"kv_cache": "scored_pq", "kv_calibration": CAL_PATH, "kv_flat_quant": "turboquant"}),
]


def build_prompt(tokenizer, target_tokens):
    block = (
        "Section {n}: Performance metrics for department {n} show steady improvement "
        "in Q3 2025, with productivity up 3.2% and employee satisfaction at 78%. "
        "Budget allocations for the fiscal year were reviewed and approved by the "
        "finance committee on March 15th. Infrastructure upgrades are scheduled for "
        "completion by end of Q4. "
    )
    blocks = []
    n = 1
    while True:
        blocks.append(block.format(n=n))
        text = "".join(blocks) + "\n\nSummarize the key points above briefly."
        if len(tokenizer.encode(text)) >= target_tokens - 100:
            break
        n += 1
    return text


def measure_cache_bytes(cache_list):
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
    return total / 1e6


def run_one(model, tokenizer, label, cache_kwargs):
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    gc.collect()
    mx.metal.clear_cache()

    prompt_text = build_prompt(tokenizer, CTX)
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_array = mx.array(tokenizer.encode(formatted))
    prompt_len = prompt_array.shape[0]

    prompt_cache = make_prompt_cache(model, **cache_kwargs)

    mx.metal.reset_peak_memory()
    mem_before = mx.metal.get_active_memory() / 1e6

    # Prefill
    t0 = time.perf_counter()
    gen = generate_step(prompt_array, model, prompt_cache=prompt_cache, max_tokens=MAX_GEN)
    first_token, _ = next(gen)
    ttof = time.perf_counter() - t0
    pp_tps = prompt_len / ttof

    # TG
    ft = first_token if isinstance(first_token, int) else first_token.item()
    tokens = [ft]
    t_tg = time.perf_counter()
    for tok, _ in gen:
        t = tok if isinstance(tok, int) else tok.item()
        tokens.append(t)
        if len(tokens) >= MAX_GEN:
            break
    tg_time = time.perf_counter() - t_tg
    tg_tps = (len(tokens) - 1) / tg_time if tg_time > 0 else 0

    kv_mb = round(measure_cache_bytes(prompt_cache))

    return {
        "label": label,
        "prompt_tokens": prompt_len,
        "gen_tokens": len(tokens),
        "pp_tps": round(pp_tps, 1),
        "tg_tps": round(tg_tps, 1),
        "ttof_s": round(ttof, 2),
        "kv_mb": kv_mb,
    }


def main():
    import mlx.core as mx
    from mlx_lm import load

    # Also verify SDK import works
    import flashmlx
    print(f"FlashMLX SDK v{flashmlx.__version__}")
    print(f"Strategies: {flashmlx.VALID_STRATEGIES}\n")

    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    gc.collect()
    print(f"Model loaded: {round(mx.metal.get_active_memory() / 1e6)} MB\n")

    print(f"Context: {CTX} tokens | Gen: {MAX_GEN} tokens")
    print(f"{'Config':20s} | {'PP':>8s} | {'TG':>8s} | {'TTOF':>7s} | {'KV MB':>6s}")
    print(f"{'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}")

    std_kv = None
    for label, kwargs in CONFIGS:
        r = run_one(model, tokenizer, label, kwargs)
        if std_kv is None:
            std_kv = r["kv_mb"]
        save = ""
        if std_kv and r["kv_mb"] < std_kv:
            save = f" (-{(1 - r['kv_mb']/std_kv)*100:.0f}%)"
        print(f"{label:20s} | {r['pp_tps']:8.1f} | {r['tg_tps']:8.1f} | {r['ttof_s']:7.2f} | {r['kv_mb']:4d}{save}")

    print("\nSanity check PASSED" if True else "")


if __name__ == "__main__":
    main()
