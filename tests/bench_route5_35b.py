"""
Route 5: Scored KV-Direct Benchmark — Qwen3.5-35B-A3B-MLX (6bit)
MoE hybrid model: 40 layers (30 SSM + 10 full attention)

Context: 4K, 8K, 16K
No AM calibration — scored_pq uses attention-based scoring fallback.
Tests h^(0) capture + reconstruction on hybrid architecture.

Previous best (bench_qwen35.py, 16K):
  standard:    PP 663.7, TG 72.6
  scored_bf16: PP 890.4, TG 83.0 (+34% PP, +14% TG)
  scored_q8:   PP 926.8, TG 82.1 (+40% PP, +13% TG)
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

MODEL = "/Volumes/toshiba/models/Qwen3.5-35B-A3B-MLX"
GREEDY = make_sampler(temp=0.0)
TG_TOKENS = 50

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


def build_prompt(tokenizer, target_tokens):
    prefix = "Please summarize the following text:\n\n"
    text = prefix
    while True:
        candidate = text + BASE_PARA
        tokens = tokenizer.encode(candidate)
        if len(tokens) >= target_tokens:
            tokens = tokens[:target_tokens]
            return tokenizer.decode(tokens), target_tokens
        text = candidate


def get_mem_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except AttributeError:
        return mx.metal.get_active_memory() / (1024 * 1024)


def warmup(model, tokenizer):
    tokens = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(tokens)
    mx.eval(model.parameters())


def bench_config(model, tokenizer, prompt_text, prompt_len, label,
                 kv_cache=None, h0_quant=None, flat_quant=None):
    """Benchmark a single config. kv_cache=None means default (model.make_cache)."""
    prompt_tokens = mx.array(tokenizer.encode(prompt_text)[:prompt_len])

    gc.collect()
    mx.clear_cache()
    time.sleep(0.5)

    mem_before = get_mem_mb()

    # Build cache — kv_cache=None uses model.make_cache() (correct for hybrid)
    cache_kwargs = {}
    if kv_cache is not None:
        cache_kwargs["kv_cache"] = kv_cache
    if h0_quant is not None:
        cache_kwargs["h0_quant"] = h0_quant
    if flat_quant is not None:
        cache_kwargs["kv_flat_quant"] = flat_quant
    cache = make_prompt_cache(model, **cache_kwargs)

    gen_tokens = []
    t_total_start = time.perf_counter()
    t_gen_start = None
    mem_after_pp = mem_before

    for i, (token_id, logprobs) in enumerate(generate_step(
        prompt_tokens, model, max_tokens=TG_TOKENS, sampler=GREEDY,
        prompt_cache=cache,
    )):
        if i == 0:
            t_first = time.perf_counter()
            ttft_ms = (t_first - t_total_start) * 1000
            pp_toks = prompt_len / (t_first - t_total_start)
            mem_after_pp = get_mem_mb()
            t_gen_start = time.perf_counter()
        gen_tokens.append(token_id)

    t_gen_end = time.perf_counter()
    if len(gen_tokens) > 1 and t_gen_start:
        tg_toks = (len(gen_tokens) - 1) / (t_gen_end - t_gen_start)
    else:
        tg_toks = 0.0

    mem_after_tg = get_mem_mb()
    pp_peak = mem_after_pp - mem_before
    tg_mem = mem_after_tg - mem_before

    info = get_cache_info(cache)
    h0_mb = info.get("h0_bytes", 0) / (1024 * 1024)

    output_text = tokenizer.decode(gen_tokens)

    return {
        "label": label,
        "prompt_len": prompt_len,
        "gen_tokens": len(gen_tokens),
        "pp_toks": pp_toks,
        "tg_toks": tg_toks,
        "ttft_ms": ttft_ms,
        "pp_peak_mb": pp_peak,
        "tg_mem_mb": tg_mem,
        "h0_mb": h0_mb,
        "output": output_text[:80],
    }


def print_results(results, title=""):
    std = results[0]
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    print(f"  {'Strategy':<30} {'PP tok/s':>9} {'TG tok/s':>9} {'TTFT':>10} "
          f"{'PP Peak':>9} {'TG Mem':>9} {'h0 MB':>7}")
    print(f"  {'-'*84}")

    for r in results:
        pp_d = f"({r['pp_toks']/std['pp_toks']*100-100:+.1f}%)" if r != std and std['pp_toks'] > 0 else ""
        tg_d = f"({r['tg_toks']/std['tg_toks']*100-100:+.1f}%)" if r != std and std['tg_toks'] > 0 else ""
        mem_d = f"({r['tg_mem_mb']/std['tg_mem_mb']*100-100:+.0f}%)" if r != std and std['tg_mem_mb'] > 0 else ""

        print(f"  {r['label']:<30} {r['pp_toks']:>9.1f} {r['tg_toks']:>9.1f} "
              f"{r['ttft_ms']:>9.0f}ms {r['pp_peak_mb']:>8.0f}M {r['tg_mem_mb']:>8.0f}M "
              f"{r['h0_mb']:>7.1f}")
        if pp_d:
            print(f"  {'':30} {pp_d:>9} {tg_d:>9} {'':>10} {'':>9} {mem_d:>9}")

    outputs = set(r["output"] for r in results)
    if len(outputs) == 1:
        print(f"\n  Output: ALL IDENTICAL — {results[0]['output']!r}...")
    else:
        print(f"\n  Output comparison:")
        for r in results:
            print(f"    {r['label']}: {r['output']!r}...")


def main():
    print("Loading Qwen3.5-35B-A3B-MLX (6bit)...")
    model, tokenizer = load(MODEL)
    print("Loaded. Warming up...")
    warmup(model, tokenizer)
    print("Ready.\n")

    context_lengths = [4096, 8192, 16384]

    # Configs: label, kv_cache, h0_quant, flat_quant
    # - standard: kv_cache=None → model.make_cache() (hybrid-correct)
    # - scored_kv_direct: Route 5 (now NOT auto-disabled for hybrid)
    configs = [
        ("standard",              None,               None,  None),
        ("scored_kv_direct",      "scored_kv_direct",  None,  None),     # bf16 h0
        ("skv_direct(h0=q8)",     "scored_kv_direct", "q8",   None),     # Q8 h0
        ("skv_direct(h0=q4)",     "scored_kv_direct", "q4",   None),     # Q4 h0
        ("skv_direct(q8+q8flat)", "scored_kv_direct", "q8",  "q8_0"),   # Q8 h0 + Q8 flat
    ]

    all_results = {}
    for ctx in context_lengths:
        print(f"\n--- Building {ctx:,}-token prompt ---")
        prompt_text, prompt_len = build_prompt(tokenizer, ctx)
        print(f"  Actual: {prompt_len} tokens")

        results = []
        for label, kv_cache, h0_quant, flat_quant in configs:
            print(f"  Benchmarking {label}...")
            try:
                r = bench_config(model, tokenizer, prompt_text, prompt_len,
                                 label=label, kv_cache=kv_cache,
                                 h0_quant=h0_quant, flat_quant=flat_quant)
                results.append(r)
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()

        all_results[ctx] = results
        print_results(results, f"Qwen3.5-35B-A3B | {ctx:,} tokens | TG={TG_TOKENS}")

    # Summary
    print(f"\n\n{'='*100}")
    print(f"  SUMMARY: Qwen3.5-35B-A3B-MLX (6bit) | Route 5 Scored KV-Direct")
    print(f"{'='*100}")
    print(f"  {'Context':>8} | {'Strategy':<30} | {'PP tok/s':>9} {'TG tok/s':>9} "
          f"{'TTFT ms':>10} {'PP Peak':>9} {'TG Mem':>9} {'h0 MB':>7}")
    print(f"  {'-'*98}")

    for ctx in context_lengths:
        for r in all_results.get(ctx, []):
            print(f"  {ctx:>8,} | {r['label']:<30} | {r['pp_toks']:>9.1f} {r['tg_toks']:>9.1f} "
                  f"{r['ttft_ms']:>10.0f} {r['pp_peak_mb']:>8.0f}M {r['tg_mem_mb']:>8.0f}M "
                  f"{r['h0_mb']:>7.1f}")
        print(f"  {'-'*98}")


if __name__ == "__main__":
    main()
