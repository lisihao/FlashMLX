"""
Route 0 Benchmark B+C: Three product modes comparison.

B group: scored_pq baseline vs +Route 0 vs +Route 0+5
C group: balanced vs ultra_long vs recall_first (each should win in its scenario)

Usage:
    python3 benchmarks/bench_density_modes.py /path/to/model
    python3 benchmarks/bench_density_modes.py /path/to/model --contexts 4096,32768
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


def warmup(model, tokenizer):
    tokens = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(tokens)
    mx.eval(model.parameters())


def bench_one(model, tokenizer, prompt_text, prompt_len, label, cache_kwargs, tg_tokens=100):
    prompt_tokens = mx.array(tokenizer.encode(prompt_text)[:prompt_len])

    gc.collect()
    mx.clear_cache()
    time.sleep(0.3)

    # Unpatch h0 capture if active from previous run
    unpatch_model(model)

    mem_before = get_mem_mb()
    cache = make_prompt_cache(model, **cache_kwargs)

    gen_tokens = []
    t_total_start = time.perf_counter()
    t_gen_start = None

    for i, (token_id, logprobs) in enumerate(generate_step(
        prompt_tokens, model, max_tokens=tg_tokens, sampler=GREEDY,
        prompt_cache=cache,
    )):
        if i == 0:
            t_first = time.perf_counter()
            ttft_ms = (t_first - t_total_start) * 1000
            pp_toks = prompt_len / (t_first - t_total_start)
            t_gen_start = time.perf_counter()
        gen_tokens.append(token_id)

    t_gen_end = time.perf_counter()
    tg_toks = ((len(gen_tokens) - 1) / (t_gen_end - t_gen_start)
               if len(gen_tokens) > 1 and t_gen_start else 0.0)

    mem_after_tg = get_mem_mb()
    tg_mem = mem_after_tg - mem_before

    info = get_cache_info(cache)
    h0_mb = info.get("h0_bytes", 0) / (1024 * 1024)
    output_text = tokenizer.decode(gen_tokens)

    return {
        "label": label,
        "prompt_len": prompt_len,
        "pp_toks": pp_toks,
        "tg_toks": tg_toks,
        "ttft_ms": ttft_ms,
        "tg_mem_mb": tg_mem,
        "h0_mb": h0_mb,
        "output": output_text[:80],
        "strategy": info.get("strategy", "?"),
    }


def print_results(results, title):
    if not results:
        return
    baseline = results[0]
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    print(f"  {'Mode':<35} {'PP tok/s':>9} {'TG tok/s':>9} {'TTFT':>10} "
          f"{'TG Mem':>9} {'h0 MB':>7} {'Strategy':>18}")
    print(f"  {'-'*98}")

    for r in results:
        print(f"  {r['label']:<35} {r['pp_toks']:>9.1f} {r['tg_toks']:>9.1f} "
              f"{r['ttft_ms']:>9.0f}ms {r['tg_mem_mb']:>8.0f}M "
              f"{r['h0_mb']:>7.1f} {r['strategy']:>18}")
        if r != baseline and baseline['tg_toks'] > 0:
            tg_d = f"({r['tg_toks']/baseline['tg_toks']*100-100:+.1f}%)"
            mem_d = (f"({r['tg_mem_mb']/baseline['tg_mem_mb']*100-100:+.0f}%)"
                     if baseline['tg_mem_mb'] > 0 else "")
            print(f"  {'':35} {'':>9} {tg_d:>9} {'':>10} {mem_d:>9}")

    outputs = set(r["output"] for r in results)
    if len(outputs) == 1:
        print(f"\n  Output: ALL IDENTICAL")
    else:
        print(f"\n  Output comparison:")
        for r in results:
            print(f"    {r['label']:<35}: {r['output']!r}...")


def main():
    parser = argparse.ArgumentParser(description="Route 0 Benchmark B+C: Product Modes")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--contexts", default="4096,16384",
                        help="Comma-separated context lengths")
    parser.add_argument("--tg-tokens", type=int, default=TG_TOKENS)
    args = parser.parse_args()

    tg_tokens = args.tg_tokens
    context_lengths = [int(x.strip()) for x in args.contexts.split(",")]

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load(args.model_path)

    card = load_card_or_detect(model, args.model_path)
    print(f"Card: {card.model_name}")
    print(f"Available modes: {list(card.modes.keys())}")

    print("Warming up...")
    warmup(model, tokenizer)

    # Build configs: baseline + three modes
    configs = [
        ("baseline (scored_pq)", card.to_cache_kwargs()),
    ]
    for mode_name in ["balanced", "ultra_long", "recall_first"]:
        if mode_name in card.modes:
            mc = card.modes[mode_name]
            label = f"{mode_name} (scale={mc.density_scale:+.1f})"
            configs.append((label, card.to_cache_kwargs(mode=mode_name)))

    for ctx in context_lengths:
        print(f"\n--- Building {ctx:,}-token prompt ---")
        prompt_text, prompt_len = build_prompt(tokenizer, ctx)
        print(f"  Actual: {prompt_len} tokens")

        results = []
        for label, kwargs in configs:
            print(f"  Benchmarking {label}...")
            try:
                r = bench_one(model, tokenizer, prompt_text, prompt_len,
                              label=label, cache_kwargs=kwargs, tg_tokens=tg_tokens)
                results.append(r)
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()

        print_results(results, f"Route 0 Product Modes | {card.model_name} | {ctx:,} tokens")

    # Summary
    print(f"\n\n{'='*110}")
    print(f"  Route 0 Product Mode Definitions ({card.model_name})")
    print(f"{'='*110}")
    for name, mc in card.modes.items():
        print(f"  {name:<15} scale={mc.density_scale:+.1f}  strategy={mc.strategy:<20} {mc.description}")


if __name__ == "__main__":
    main()
