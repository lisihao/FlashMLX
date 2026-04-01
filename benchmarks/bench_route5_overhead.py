"""
Route 5 System Overhead Benchmark — scored_pq vs scored_kv_direct.

Measures the cost of adding h^(0) capture on top of scored_pq:
  1. h^(0) memory overhead (hot tier, per-token)
  2. Prefill overhead (embed_tokens capture)
  3. TG latency (per-step capture + h0_store.append)
  4. Reconstruction cost (chunked vs unchunked)
  5. Recall injection merge overhead

Usage:
    python3 benchmarks/bench_route5_overhead.py
    python3 benchmarks/bench_route5_overhead.py --model /path/to/model
    python3 benchmarks/bench_route5_overhead.py --context 8192
"""

import argparse
import gc
import sys
import time

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx

from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.cache_factory import get_cache_info
from mlx_lm.models.kv_direct_cache import unpatch_model
from mlx_lm.sample_utils import make_sampler

DEFAULT_MODEL = "/Volumes/toshiba/models/qwen3-8b-mlx"
GREEDY = make_sampler(temp=0.0)
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


def bench_strategy(model, tokenizer, prompt_text, prompt_len, label, cache_kwargs,
                   tg_tokens=50):
    """Benchmark a single strategy, returning detailed timing."""
    prompt_tokens = mx.array(tokenizer.encode(prompt_text)[:prompt_len])

    gc.collect()
    mx.clear_cache()
    time.sleep(0.3)

    mem_before = get_mem_mb()
    cache = make_prompt_cache(model, **cache_kwargs)
    mem_after_cache = get_mem_mb()

    gen_tokens = []
    tg_times = []
    t_total_start = time.perf_counter()

    for i, (token_id, logprobs) in enumerate(generate_step(
        prompt_tokens, model, max_tokens=tg_tokens, sampler=GREEDY,
        prompt_cache=cache,
    )):
        t_step = time.perf_counter()
        if i == 0:
            t_first = t_step
            ttft_ms = (t_first - t_total_start) * 1000
            pp_toks = prompt_len / (t_first - t_total_start)
            mem_after_pp = get_mem_mb()
        else:
            tg_times.append(t_step - t_prev)
        t_prev = t_step
        gen_tokens.append(token_id)

    mem_after_tg = get_mem_mb()

    info = get_cache_info(cache)
    h0_mb = info.get("h0_bytes", 0) / (1024 * 1024)

    # TG stats
    if tg_times:
        tg_avg = sum(tg_times) / len(tg_times) * 1000  # ms
        tg_p99 = sorted(tg_times)[int(len(tg_times) * 0.99)] * 1000
        tg_toks = len(tg_times) / sum(tg_times) if sum(tg_times) > 0 else 0
    else:
        tg_avg = tg_p99 = 0
        tg_toks = 0

    return {
        "label": label,
        "prompt_len": prompt_len,
        "pp_toks": pp_toks,
        "tg_toks": tg_toks,
        "ttft_ms": ttft_ms,
        "tg_avg_ms": tg_avg,
        "tg_p99_ms": tg_p99,
        "mem_cache_mb": mem_after_cache - mem_before,
        "mem_pp_mb": mem_after_pp - mem_before,
        "mem_tg_mb": mem_after_tg - mem_before,
        "h0_mb": h0_mb,
        "gen_tokens": len(gen_tokens),
        "output": tokenizer.decode(gen_tokens)[:60],
    }


def main():
    parser = argparse.ArgumentParser(description="Route 5 System Overhead Benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--context", type=int, default=4096, help="Context length")
    parser.add_argument("--tg-tokens", type=int, default=50, help="TG tokens")
    parser.add_argument("--calib", default=None, help="AM calibration file")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model, tokenizer = load(args.model)

    # Warmup
    warmup_tokens = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(warmup_tokens)
    mx.eval(model.parameters())
    print("Model loaded.\n")

    # Build prompt
    prompt_text, prompt_len = build_prompt(tokenizer, args.context)
    print(f"Prompt: {prompt_len} tokens, TG: {args.tg_tokens} tokens\n")

    # Strategies to compare
    strategies = [
        ("scored_pq (baseline)", {
            "kv_cache": "scored_pq",
        }),
        ("scored_kv_direct (bf16)", {
            "kv_cache": "scored_kv_direct",
        }),
        ("scored_kv_direct (q8)", {
            "kv_cache": "scored_kv_direct",
            "h0_quant": "q8",
        }),
    ]

    # Add calibration if provided
    if args.calib:
        for _, kwargs in strategies:
            kwargs["kv_calibration"] = args.calib

    results = []
    for label, kwargs in strategies:
        print(f"  Benchmarking {label}...")
        # Unpatch model before each strategy to avoid double-patch
        unpatch_model(model)
        try:
            r = bench_strategy(model, tokenizer, prompt_text, prompt_len,
                               label=label, cache_kwargs=kwargs, tg_tokens=args.tg_tokens)
            results.append(r)
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Results table
    if not results:
        print("No results.")
        return

    baseline = results[0]
    print(f"\n{'='*100}")
    print(f"  Route 5 System Overhead — {args.model.split('/')[-1]} | {prompt_len:,} tokens")
    print(f"{'='*100}")
    print(f"  {'Strategy':<30} {'PP tok/s':>9} {'TG tok/s':>9} {'TTFT':>10} "
          f"{'TG avg':>9} {'TG p99':>9} {'TG Mem':>8} {'h^(0)':>7}")
    print(f"  {'-'*94}")

    for r in results:
        print(f"  {r['label']:<30} {r['pp_toks']:>9.1f} {r['tg_toks']:>9.1f} "
              f"{r['ttft_ms']:>9.0f}ms {r['tg_avg_ms']:>8.1f}ms {r['tg_p99_ms']:>8.1f}ms "
              f"{r['mem_tg_mb']:>7.0f}M {r['h0_mb']:>6.1f}M")
        if r != baseline and baseline['pp_toks'] > 0:
            pp_d = f"({r['pp_toks']/baseline['pp_toks']*100-100:+.1f}%)"
            tg_d = f"({r['tg_toks']/baseline['tg_toks']*100-100:+.1f}%)" if baseline['tg_toks'] > 0 else ""
            mem_d = f"({r['mem_tg_mb']-baseline['mem_tg_mb']:+.0f}M)"
            print(f"  {'':30} {pp_d:>9} {tg_d:>9} {'':>10} {'':>9} {'':>9} {mem_d:>8}")

    # Output match check
    outputs = set(r["output"] for r in results)
    if len(outputs) == 1:
        print(f"\n  Output: ALL IDENTICAL — {results[0]['output']!r}...")
    else:
        print(f"\n  Output comparison:")
        for r in results:
            print(f"    {r['label']}: {r['output']!r}...")

    # Overhead summary
    print(f"\n{'='*100}")
    print(f"  OVERHEAD SUMMARY (vs scored_pq baseline)")
    print(f"{'='*100}")
    for r in results[1:]:
        pp_pct = (r['pp_toks'] / baseline['pp_toks'] * 100 - 100) if baseline['pp_toks'] > 0 else 0
        tg_pct = (r['tg_toks'] / baseline['tg_toks'] * 100 - 100) if baseline['tg_toks'] > 0 else 0
        mem_delta = r['mem_tg_mb'] - baseline['mem_tg_mb']
        ttft_delta = r['ttft_ms'] - baseline['ttft_ms']
        tg_avg_delta = r['tg_avg_ms'] - baseline['tg_avg_ms']

        print(f"\n  {r['label']}:")
        print(f"    PP throughput: {pp_pct:+.1f}%")
        print(f"    TG throughput: {tg_pct:+.1f}%")
        print(f"    TTFT delta:    {ttft_delta:+.0f}ms")
        print(f"    TG avg delta:  {tg_avg_delta:+.1f}ms/step")
        print(f"    Memory delta:  {mem_delta:+.0f}M (h^(0): {r['h0_mb']:.1f}M)")


if __name__ == "__main__":
    main()
