"""
Route 0 Benchmark A: Continuous (adaptive) vs Discrete compression levels.

Tests the paper claim: discrete ratios are more stable than continuous ones.

Compares:
    - adaptive (compression_ratio=0, current default)
    - 5 discrete levels: keep_80(1.25x), keep_50(2x), keep_33(3x), keep_20(5x), keep_10(10x)
    - scale sweep at keep_50 baseline: scale={-1, 0, +1, +2}

Usage:
    python3 benchmarks/bench_route0_discrete.py /path/to/model
    python3 benchmarks/bench_route0_discrete.py /path/to/model --contexts 4096,16384
    python3 benchmarks/bench_route0_discrete.py /path/to/model --scale-sweep
"""

import argparse
import gc
import sys
import time

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx
import numpy as np

from flashmlx.config import DensityLevel, snap_to_nearest
from flashmlx.model_cards import load_card_or_detect
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


def bench_one(model, tokenizer, prompt_text, prompt_len, label, cache_kwargs, tg_tokens=100):
    """Benchmark a single configuration. Returns result dict."""
    prompt_tokens = mx.array(tokenizer.encode(prompt_text)[:prompt_len])

    gc.collect()
    mx.clear_cache()
    time.sleep(0.3)

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

    # Get density signal from layer 0
    density_signal = None
    from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
    for c in cache:
        if isinstance(c, TripleLayerKVCache) and c._density_signal is not None:
            density_signal = c._density_signal
            break

    info = get_cache_info(cache)
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
        "output": output_text[:80],
        "density_signal": density_signal,
    }


def print_results(results, title):
    if not results:
        return
    baseline = results[0]
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    print(f"  {'Config':<35} {'PP tok/s':>9} {'TG tok/s':>9} {'TTFT':>10} "
          f"{'PP Peak':>9} {'TG Mem':>9} {'Keep%':>7}")
    print(f"  {'-'*90}")

    for r in results:
        ds = r.get("density_signal")
        keep_pct = f"{ds['keep_ratio']*100:.0f}%" if ds else "n/a"
        print(f"  {r['label']:<35} {r['pp_toks']:>9.1f} {r['tg_toks']:>9.1f} "
              f"{r['ttft_ms']:>9.0f}ms {r['pp_peak_mb']:>8.0f}M {r['tg_mem_mb']:>8.0f}M "
              f"{keep_pct:>7}")
        if r != baseline and baseline['pp_toks'] > 0 and baseline['tg_toks'] > 0:
            pp_d = f"({r['pp_toks']/baseline['pp_toks']*100-100:+.1f}%)"
            tg_d = f"({r['tg_toks']/baseline['tg_toks']*100-100:+.1f}%)"
            mem_d = (f"({r['tg_mem_mb']/baseline['tg_mem_mb']*100-100:+.0f}%)"
                     if baseline['tg_mem_mb'] > 0 else "")
            print(f"  {'':35} {pp_d:>9} {tg_d:>9} {'':>10} {'':>9} {mem_d:>9}")

    # Output comparison
    outputs = set(r["output"] for r in results)
    if len(outputs) == 1:
        print(f"\n  Output: ALL IDENTICAL — {results[0]['output']!r}...")
    else:
        print(f"\n  Output comparison:")
        for r in results:
            print(f"    {r['label']:<35}: {r['output']!r}...")


def main():
    parser = argparse.ArgumentParser(description="Route 0 Benchmark A: Continuous vs Discrete")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--contexts", default="4096,16384",
                        help="Comma-separated context lengths (default: 4096,16384)")
    parser.add_argument("--tg-tokens", type=int, default=TG_TOKENS,
                        help="Number of tokens to generate (default: 100)")
    parser.add_argument("--scale-sweep", action="store_true",
                        help="Run scale sweep test (scale={-1, 0, +1, +2})")
    parser.add_argument("--levels", default="all",
                        help="Which discrete levels to test: 'all' or comma-separated "
                             "(e.g. 'keep_80,keep_50,keep_33')")
    args = parser.parse_args()

    tg_tokens = args.tg_tokens
    context_lengths = [int(x.strip()) for x in args.contexts.split(",")]

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load(args.model_path)

    # Load card for calibration file
    card = load_card_or_detect(model, args.model_path)
    calib = card.optimal.calibration_file
    flat_quant = card.optimal.flat_quant
    print(f"Card: {card.model_name}, strategy={card.optimal.strategy}")
    print(f"Calibration: {calib}")
    print(f"Flat quant: {flat_quant}")

    # Warmup
    print("Warming up...")
    warmup(model, tokenizer)

    # Resolve levels
    if args.levels == "all":
        test_levels = list(DensityLevel)
    else:
        test_levels = [DensityLevel[name.strip()] for name in args.levels.split(",")]

    # Build configs
    base_kwargs = {}
    if calib:
        base_kwargs["kv_calibration"] = calib
    if flat_quant:
        base_kwargs["kv_flat_quant"] = flat_quant

    configs = [
        # Baseline: adaptive (continuous, current default)
        ("adaptive (continuous)", {**base_kwargs, "kv_cache": "scored_pq", "kv_compression_ratio": 0}),
    ]
    # Discrete levels: explicit compression_ratio for each
    for lvl in test_levels:
        label = f"discrete {lvl.name} ({lvl.compression_ratio:.1f}x)"
        configs.append((label, {
            **base_kwargs,
            "kv_cache": "scored_pq",
            "kv_compression_ratio": lvl.compression_ratio,
        }))

    # Run benchmarks
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

        print_results(results, f"Route 0 Benchmark A | {card.model_name} | {ctx:,} tokens")

    # Scale sweep (optional)
    if args.scale_sweep:
        print(f"\n\n{'='*110}")
        print(f"  SCALE SWEEP (base=keep_50/2.0x)")
        print(f"{'='*110}")

        for ctx in context_lengths:
            prompt_text, prompt_len = build_prompt(tokenizer, ctx)
            results = []

            for scale in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
                lvl = snap_to_nearest(DensityLevel.keep_50.log2_ratio, scale=scale)
                label = f"scale={scale:+.0f} → {lvl.name} ({lvl.compression_ratio:.1f}x)"
                kwargs = {
                    **base_kwargs,
                    "kv_cache": "scored_pq",
                    "kv_compression_ratio": lvl.compression_ratio,
                }
                print(f"  Benchmarking {label}...")
                try:
                    r = bench_one(model, tokenizer, prompt_text, prompt_len,
                                  label=label, cache_kwargs=kwargs, tg_tokens=tg_tokens)
                    results.append(r)
                except Exception as e:
                    print(f"    FAILED: {e}")

            print_results(results, f"Scale Sweep | {ctx:,} tokens")


if __name__ == "__main__":
    main()
