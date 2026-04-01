"""
Universal Model Card Benchmark — reads config from model cards.

Usage:
    python3 benchmarks/bench_card.py /path/to/model
    python3 benchmarks/bench_card.py /path/to/model --contexts 4096,16384
    python3 benchmarks/bench_card.py /path/to/model --update-card
    python3 benchmarks/bench_card.py --list-cards

Compares: standard baseline vs card's optimal config.
All parameters come from the model card — zero hardcoding.
"""

import argparse
import gc
import sys
import time

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx

from flashmlx.model_cards import load_card, load_card_or_detect, save_card
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.cache_factory import get_cache_info
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


def bench_one(model, tokenizer, prompt_text, prompt_len, label, cache_kwargs,
              tg_tokens=100):
    """Benchmark a single configuration."""
    prompt_tokens = mx.array(tokenizer.encode(prompt_text)[:prompt_len])

    gc.collect()
    mx.clear_cache()
    time.sleep(0.5)

    mem_before = get_mem_mb()
    cache = make_prompt_cache(model, **cache_kwargs)

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


def print_results(results, title):
    if not results:
        return
    std = results[0]
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    print(f"  {'Strategy':<30} {'PP tok/s':>9} {'TG tok/s':>9} {'TTFT':>10} "
          f"{'PP Peak':>9} {'TG Mem':>9} {'h0 MB':>7}")
    print(f"  {'-'*84}")

    for r in results:
        print(f"  {r['label']:<30} {r['pp_toks']:>9.1f} {r['tg_toks']:>9.1f} "
              f"{r['ttft_ms']:>9.0f}ms {r['pp_peak_mb']:>8.0f}M {r['tg_mem_mb']:>8.0f}M "
              f"{r['h0_mb']:>7.1f}")
        if r != std and std['pp_toks'] > 0 and std['tg_toks'] > 0:
            pp_d = f"({r['pp_toks']/std['pp_toks']*100-100:+.1f}%)"
            tg_d = f"({r['tg_toks']/std['tg_toks']*100-100:+.1f}%)"
            mem_d = f"({r['tg_mem_mb']/std['tg_mem_mb']*100-100:+.0f}%)" if std['tg_mem_mb'] > 0 else ""
            print(f"  {'':30} {pp_d:>9} {tg_d:>9} {'':>10} {'':>9} {mem_d:>9}")

    outputs = set(r["output"] for r in results)
    if len(outputs) == 1:
        print(f"\n  Output: ALL IDENTICAL — {results[0]['output']!r}...")
    else:
        print(f"\n  Output comparison:")
        for r in results:
            print(f"    {r['label']}: {r['output']!r}...")


def main():
    parser = argparse.ArgumentParser(description="Universal Model Card Benchmark")
    parser.add_argument("model_path", nargs="?", help="Path to model directory")
    parser.add_argument("--contexts", default=None,
                        help="Comma-separated context lengths (e.g. 4096,16384)")
    parser.add_argument("--update-card", action="store_true",
                        help="Write benchmark results back to card file")
    parser.add_argument("--list-cards", action="store_true",
                        help="List all available model cards")
    parser.add_argument("--tg-tokens", type=int, default=TG_TOKENS,
                        help="Number of tokens to generate (default: 100)")
    args = parser.parse_args()

    if args.list_cards:
        from flashmlx.model_cards import list_cards
        cards = list_cards()
        if not cards:
            print("No model cards found.")
            return
        print(f"\n{'='*80}")
        print(f"  Available Model Cards ({len(cards)})")
        print(f"{'='*80}")
        for c in cards:
            print(f"  {c.model_id:<30} {c.model_name:<35} {c.architecture.type}")
            print(f"  {'':30} strategy={c.optimal.strategy}  flat_quant={c.optimal.flat_quant}")
            if c.notes:
                print(f"  {'':30} {c.notes[:70]}")
            print()
        return

    if not args.model_path:
        parser.error("model_path is required (or use --list-cards)")

    tg_tokens = args.tg_tokens

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load(args.model_path)

    # Load or detect card
    card = load_card_or_detect(model, args.model_path)
    print(f"Card: {card.model_name} ({card.model_id})")
    print(f"Architecture: {card.architecture.type} "
          f"({card.architecture.attention_layers}/{card.architecture.num_layers} attn layers)")
    print(f"Optimal: strategy={card.optimal.strategy}, flat_quant={card.optimal.flat_quant}")
    if card.notes:
        print(f"Notes: {card.notes}")

    # Warmup
    print("Warming up...")
    warmup(model, tokenizer)
    print("Ready.\n")

    # Context lengths
    if args.contexts:
        context_lengths = [int(x.strip()) for x in args.contexts.split(",")]
    else:
        context_lengths = [4096, 8192, 16384]

    # Build configs: standard baseline vs card optimal
    configs = [
        ("standard", {}),  # kv_cache=None → model.make_cache()
        (f"optimal ({card.optimal.strategy})", card.to_cache_kwargs()),
    ]

    all_results = {}
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

        all_results[ctx] = results
        print_results(results, f"{card.model_name} | {ctx:,} tokens | TG={tg_tokens}")

    # Summary
    print(f"\n\n{'='*100}")
    print(f"  SUMMARY: {card.model_name}")
    print(f"{'='*100}")
    print(f"  {'Context':>8} | {'Strategy':<30} | {'PP tok/s':>9} {'TG tok/s':>9} "
          f"{'TTFT ms':>10} {'PP Peak':>9} {'TG Mem':>9}")
    print(f"  {'-'*92}")

    for ctx in context_lengths:
        for r in all_results.get(ctx, []):
            print(f"  {ctx:>8,} | {r['label']:<30} | {r['pp_toks']:>9.1f} {r['tg_toks']:>9.1f} "
                  f"{r['ttft_ms']:>10.0f} {r['pp_peak_mb']:>8.0f}M {r['tg_mem_mb']:>8.0f}M")
        print(f"  {'-'*92}")

    # Update card with benchmark results if requested
    if args.update_card:
        from flashmlx.model_cards import BenchmarkResult
        for ctx, results in all_results.items():
            key = f"{ctx // 1024}k"
            # Standard baseline (first entry)
            if len(results) >= 1:
                s = results[0]
                card.standard_baselines[key] = BenchmarkResult(
                    context=ctx,
                    pp_toks=round(s["pp_toks"], 1),
                    tg_toks=round(s["tg_toks"], 1),
                    ttft_ms=round(s["ttft_ms"]),
                    tg_mem_mb=round(s["tg_mem_mb"]),
                    pp_peak_mb=round(s["pp_peak_mb"]),
                )
            # Optimal config result (second entry)
            if len(results) >= 2:
                r = results[1]
            elif results:
                r = results[0]
            else:
                continue
            card.benchmarks[key] = BenchmarkResult(
                context=ctx,
                pp_toks=round(r["pp_toks"], 1),
                tg_toks=round(r["tg_toks"], 1),
                ttft_ms=round(r["ttft_ms"]),
                tg_mem_mb=round(r["tg_mem_mb"]),
                pp_peak_mb=round(r["pp_peak_mb"]),
            )
        saved_path = save_card(card)
        print(f"\nCard updated: {saved_path}")


if __name__ == "__main__":
    main()
