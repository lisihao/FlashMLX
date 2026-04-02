"""
Route 0 Mode Throughput Benchmark — PP tok/s, TG tok/s, memory for all modes.

Usage:
    python3 benchmarks/bench_modes_throughput.py /path/to/model
    python3 benchmarks/bench_modes_throughput.py /path/to/model --contexts 4096,8192,16384
"""

import argparse
import gc
import sys
import time

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx

from flashmlx.model_cards import load_card_or_detect
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.cache_factory import get_cache_info
from mlx_lm.models.kv_direct_cache import unpatch_model
from mlx_lm.sample_utils import make_sampler

GREEDY = make_sampler(temp=0.0)
TG_TOKENS = 100

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
    prefix = "Please summarize the following text and provide key insights:\n\n"
    text = prefix
    while True:
        candidate = text + BASE_PARA
        tokens = tokenizer.encode(candidate)
        if len(tokens) >= target_tokens:
            return tokenizer.decode(tokens[:target_tokens]), target_tokens
        text = candidate


def get_mem_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except AttributeError:
        return mx.metal.get_active_memory() / (1024 * 1024)


def bench_one(model, tokenizer, prompt_text, prompt_len, label, cache_kwargs,
              tg_tokens=100):
    gc.collect()
    mx.clear_cache()
    time.sleep(0.3)

    unpatch_model(model)

    mem_before = get_mem_mb()
    cache = make_prompt_cache(model, **cache_kwargs)

    prompt_tokens = mx.array(tokenizer.encode(prompt_text)[:prompt_len])

    gen_tokens = []
    t_total_start = time.perf_counter()
    t_gen_start = None
    mem_after_pp = mem_before

    for i, (token_id, logprobs) in enumerate(generate_step(
        prompt_tokens, model, max_tokens=tg_tokens, sampler=GREEDY,
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
    }


def main():
    parser = argparse.ArgumentParser(description="Route 0 Mode Throughput Benchmark")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--contexts", default="8192",
                        help="Comma-separated context lengths (default: 8192)")
    parser.add_argument("--tg-tokens", type=int, default=TG_TOKENS,
                        help="Number of tokens to generate (default: 100)")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load(args.model_path)

    card = load_card_or_detect(model, args.model_path)
    print(f"Card: {card.model_name}")

    # Build all mode configs
    recall_kwargs = card.to_cache_kwargs(mode="recall_first")
    recall_manual = {k: v for k, v in recall_kwargs.items() if k != "auto_reconstruct"}
    configs = [
        ("standard", {}),
        ("scored_pq (baseline)", card.to_cache_kwargs()),
        ("ultra_long (10x)", card.to_cache_kwargs(mode="ultra_long")),
        ("recall_first (10x+h0)", recall_manual),
        ("recall_first+AUTO", recall_kwargs),
    ]

    # Warmup
    print("Warming up...")
    warm_tokens = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(warm_tokens)
    mx.eval(model.parameters())
    print("Ready.\n")

    context_lengths = [int(x.strip()) for x in args.contexts.split(",")]

    for ctx in context_lengths:
        print(f"--- Building {ctx:,}-token prompt ---")
        prompt_text, prompt_len = build_prompt(tokenizer, ctx)
        print(f"  Actual: {prompt_len} tokens\n")

        results = []
        for label, kwargs in configs:
            print(f"  Benchmarking {label}...")
            try:
                r = bench_one(model, tokenizer, prompt_text, prompt_len,
                              label=label, cache_kwargs=kwargs,
                              tg_tokens=args.tg_tokens)
                results.append(r)
                print(f"    PP={r['pp_toks']:.1f} tok/s  TG={r['tg_toks']:.1f} tok/s  "
                      f"TTFT={r['ttft_ms']:.0f}ms  Mem={r['tg_mem_mb']:.0f}MB  "
                      f"h0={r['h0_mb']:.1f}MB")
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()

        # Print summary table
        if not results:
            continue
        std = results[0]
        print(f"\n{'='*110}")
        print(f"  {card.model_name} | {ctx:,} tokens | TG={args.tg_tokens}")
        print(f"{'='*110}")
        print(f"  {'Mode':<28} {'PP tok/s':>9} {'TG tok/s':>9} {'TTFT':>10} "
              f"{'PP Peak':>9} {'TG Mem':>9} {'h0':>7}")
        print(f"  {'-'*90}")

        for r in results:
            print(f"  {r['label']:<28} {r['pp_toks']:>9.1f} {r['tg_toks']:>9.1f} "
                  f"{r['ttft_ms']:>9.0f}ms {r['pp_peak_mb']:>8.0f}M {r['tg_mem_mb']:>8.0f}M "
                  f"{r['h0_mb']:>7.1f}")
            if r != std and std['pp_toks'] > 0:
                pp_d = f"({r['pp_toks']/std['pp_toks']*100-100:+.0f}%)"
                tg_d = f"({r['tg_toks']/std['tg_toks']*100-100:+.0f}%)" if std['tg_toks'] > 0 else ""
                mem_d = f"({r['tg_mem_mb']/std['tg_mem_mb']*100-100:+.0f}%)" if std['tg_mem_mb'] > 0 else ""
                print(f"  {'':28} {pp_d:>9} {tg_d:>9} {'':>10} {'':>9} {mem_d:>9}")
        print()


if __name__ == "__main__":
    main()
