"""
Route 5: Scored KV-Direct — Context Length Scaling Benchmark
Model: Qwen3-8B-MLX-4bit
Context: 4K, 8K, 16K, 32K
Strategies: standard, scored_pq, scored_kv_direct (bf16/Q8/Q4 h^(0))

Measures: PP tok/s, TG tok/s, TTFT ms, Memory (peak & steady)
"""

import sys
import time
import gc

import mlx.core as mx

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.cache_factory import get_cache_info
from mlx_lm.sample_utils import make_sampler

MODEL = "/Volumes/toshiba/models/qwen3-8b-mlx"
CALIB = "/Users/lisihao/FlashMLX/calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl"
GREEDY = make_sampler(temp=0.0)

# Build prompts of target lengths by repeating a base paragraph
BASE_PARA = (
    "The development of artificial intelligence has been one of the most "
    "transformative technological advances of the 21st century. From natural "
    "language processing to computer vision, AI systems are now capable of "
    "performing tasks that were once thought to be exclusively human. The "
    "implications of this technology span across multiple sectors including "
    "healthcare, finance, education, and manufacturing. Researchers continue "
    "to push the boundaries of what is possible, developing new architectures "
    "and training methodologies that improve both the efficiency and capability "
    "of these systems. "
)

TG_TOKENS = 100  # Generate 100 tokens for TG measurement


def build_prompt(tokenizer, target_tokens):
    """Build a prompt of approximately target_tokens length."""
    # Start with a question prefix
    prefix = "Please summarize the following text and provide key insights:\n\n"
    prefix_len = len(tokenizer.encode(prefix))

    # Repeat paragraph until we hit target
    text = prefix
    while True:
        candidate = text + BASE_PARA
        tokens = tokenizer.encode(candidate)
        if len(tokens) >= target_tokens:
            # Trim to exact length
            tokens = tokens[:target_tokens]
            return tokenizer.decode(tokens), target_tokens
        text = candidate


def warmup(model, tokenizer):
    tokens = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(tokens)
    mx.eval(model.parameters())


def get_mem_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except AttributeError:
        return mx.metal.get_active_memory() / (1024 * 1024)


def bench_config(model, tokenizer, prompt_text, prompt_len, strategy,
                 h0_quant=None, flat_quant=None):
    """Benchmark using generate_step (supports chunked prefill for long ctx)."""
    prompt_tokens = mx.array(tokenizer.encode(prompt_text)[:prompt_len])

    gc.collect()
    mx.clear_cache()
    time.sleep(0.5)

    mem_before = get_mem_mb()

    # Create cache
    cache_kwargs = {"kv_cache": strategy}
    if strategy in ("scored_pq", "scored_kv_direct"):
        cache_kwargs["kv_calibration"] = CALIB
    if h0_quant is not None:
        cache_kwargs["h0_quant"] = h0_quant
    if flat_quant is not None:
        cache_kwargs["kv_flat_quant"] = flat_quant
    cache = make_prompt_cache(model, **cache_kwargs)

    # --- Generate using generate_step (handles chunked prefill) ---
    gen_tokens = []
    pp_toks = 0.0
    tg_toks = 0.0
    ttft_ms = 0.0

    t_total_start = time.perf_counter()
    for i, (token_id, logprobs) in enumerate(generate_step(
        prompt_tokens, model, max_tokens=TG_TOKENS, sampler=GREEDY,
        prompt_cache=cache,
    )):
        if i == 0:
            # First token = end of prefill
            t_first = time.perf_counter()
            ttft_ms = (t_first - t_total_start) * 1000
            pp_toks = prompt_len / (t_first - t_total_start)
            mem_after_pp = get_mem_mb()
            t_gen_start = time.perf_counter()

        gen_tokens.append(token_id)

        if token_id in (151643, 151645):
            break

    t_gen_end = time.perf_counter()
    if len(gen_tokens) > 1:
        tg_toks = (len(gen_tokens) - 1) / (t_gen_end - t_gen_start)
    else:
        tg_toks = 0.0
        mem_after_pp = get_mem_mb()

    mem_after_tg = get_mem_mb()
    tg_mem = mem_after_tg - mem_before
    pp_peak = mem_after_pp - mem_before if 'mem_after_pp' in dir() else tg_mem

    # Cache info
    info = get_cache_info(cache)
    try:
        cache_bytes = sum(c.nbytes for c in cache)
    except (NotImplementedError, AttributeError):
        cache_bytes = 0
        for c in cache:
            s = c.state
            if s and len(s) == 2:
                cache_bytes += s[0].nbytes + s[1].nbytes
    cache_mb = cache_bytes / (1024 * 1024)

    h0_mb = info.get("h0_bytes", 0) / (1024 * 1024)

    output_text = tokenizer.decode(gen_tokens)

    return {
        "strategy": strategy,
        "h0_quant": h0_quant or "—",
        "flat_quant": flat_quant or "bf16",
        "prompt_len": prompt_len,
        "pp_toks": pp_toks,
        "tg_toks": tg_toks,
        "ttft_ms": ttft_ms,
        "pp_peak_mb": pp_peak,
        "tg_mem_mb": tg_mem,
        "cache_mb": cache_mb,
        "h0_mb": h0_mb,
        "output": output_text[:60],
        "info": info,
    }


def label(r):
    s = r["strategy"]
    if s == "scored_kv_direct":
        return f"skv_direct(h0={r['h0_quant']})"
    if s == "scored_pq":
        fq = r["flat_quant"]
        return f"scored_pq" + (f"({fq})" if fq != "bf16" else "")
    return s


def print_context_results(results, ctx_len):
    std = results[0]
    print(f"\n{'='*100}")
    print(f"  Context: {ctx_len:,} tokens | Model: Qwen3-8B-MLX-4bit | TG: {TG_TOKENS} tokens")
    print(f"{'='*100}")
    print(f"  {'Strategy':<26} {'PP tok/s':>9} {'TG tok/s':>9} {'TTFT':>9} "
          f"{'PP Peak':>9} {'TG Mem':>9} {'h0 MB':>7} {'KV MB':>8}")
    print(f"  {'-'*88}")

    for r in results:
        pp_d = f"({r['pp_toks']/std['pp_toks']*100-100:+.1f}%)" if r != std else ""
        tg_d = f"({r['tg_toks']/std['tg_toks']*100-100:+.1f}%)" if r != std else ""
        pp_mem_d = f"({r['pp_peak_mb']/std['pp_peak_mb']*100-100:+.0f}%)" if r != std and std['pp_peak_mb'] > 0 else ""
        tg_mem_d = f"({r['tg_mem_mb']/std['tg_mem_mb']*100-100:+.0f}%)" if r != std and std['tg_mem_mb'] > 0 else ""

        lbl = label(r)
        print(f"  {lbl:<26} {r['pp_toks']:>9.1f} {r['tg_toks']:>9.1f} "
              f"{r['ttft_ms']:>8.1f}ms {r['pp_peak_mb']:>8.1f}M {r['tg_mem_mb']:>8.1f}M "
              f"{r['h0_mb']:>7.1f} {r['cache_mb']:>8.1f}")
        if pp_d or tg_d:
            print(f"  {'':26} {pp_d:>9} {tg_d:>9} {'':>9} {pp_mem_d:>9} {tg_mem_d:>9}")

    # Output match check
    outputs = set(r["output"] for r in results)
    if len(outputs) == 1:
        print(f"\n  Output: ALL IDENTICAL")
        print(f"    {results[0]['output']!r}...")
    else:
        print(f"\n  Output comparison:")
        for r in results:
            print(f"    {label(r)}: {r['output']!r}...")


def main():
    print("Loading Qwen3-8B...")
    model, tokenizer = load(MODEL)
    print("Loaded. Warming up...")
    warmup(model, tokenizer)
    print("Ready.\n")

    context_lengths = [4096, 8192, 16384, 32768]

    strategies = [
        ("standard",         {"flat_quant": None, "h0_quant": None}),
        ("scored_pq",        {"flat_quant": None, "h0_quant": None}),
        ("scored_pq",        {"flat_quant": "q8_0", "h0_quant": None}),
        ("scored_kv_direct", {"flat_quant": None, "h0_quant": None}),      # bf16 h0
        ("scored_kv_direct", {"flat_quant": None, "h0_quant": "q8"}),      # Q8 h0
        ("scored_kv_direct", {"flat_quant": None, "h0_quant": "q4"}),      # Q4 h0
        ("scored_kv_direct", {"flat_quant": "q8_0", "h0_quant": "q8"}),   # best combo
    ]

    all_results = {}

    for ctx in context_lengths:
        print(f"\n--- Building {ctx:,}-token prompt ---")
        prompt_text, prompt_len = build_prompt(tokenizer, ctx)
        print(f"  Actual prompt: {prompt_len} tokens")

        results = []
        for strat, kwargs in strategies:
            lbl = strat + (f"(h0={kwargs['h0_quant']})" if kwargs.get('h0_quant') else "")
            print(f"  Benchmarking {lbl}...")
            try:
                r = bench_config(
                    model, tokenizer, prompt_text, prompt_len,
                    strategy=strat, h0_quant=kwargs.get("h0_quant"),
                    flat_quant=kwargs.get("flat_quant"),
                )
                results.append(r)
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()

        all_results[ctx] = results
        print_context_results(results, ctx)

    # Final summary table
    print(f"\n\n{'='*100}")
    print(f"  SUMMARY: Scored KV-Direct vs Standard (all context lengths)")
    print(f"{'='*100}")
    print(f"  {'Context':>8} | {'Strategy':<26} | {'PP tok/s':>9} {'TG tok/s':>9} "
          f"{'TTFT ms':>9} {'PP Peak':>9} {'TG Mem':>9} {'h0 MB':>7}")
    print(f"  {'-'*96}")

    for ctx in context_lengths:
        results = all_results.get(ctx, [])
        for r in results:
            lbl = label(r)
            print(f"  {ctx:>8,} | {lbl:<26} | {r['pp_toks']:>9.1f} {r['tg_toks']:>9.1f} "
                  f"{r['ttft_ms']:>9.1f} {r['pp_peak_mb']:>8.1f}M {r['tg_mem_mb']:>8.1f}M "
                  f"{r['h0_mb']:>7.1f}")
        if results:
            print(f"  {'-'*96}")


if __name__ == "__main__":
    main()
