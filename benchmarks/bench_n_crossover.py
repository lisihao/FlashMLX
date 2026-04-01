"""
N Crossover Benchmark — find the reconstruction break-even point.

For each N in [1, 8, 16, 32, 50, 64, 100, 200, 500]:
  1. Standard prefill: feed N tokens through model, measure time to produce KV
  2. Reconstruction: from pre-captured h^(0), run reconstruct_prefix_kv, measure time

The crossover point is where reconstruction becomes faster than standard prefill.
Paper claims N≈50 for 135M-4B models. We measure on Qwen3-8B.

Usage:
    python3 benchmarks/bench_n_crossover.py
    python3 benchmarks/bench_n_crossover.py --model /path/to/model
"""

import argparse
import gc
import sys
import time

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx

from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.kv_direct_cache import (
    H0Store, apply_h0_capture_only, reconstruct_prefix_kv,
    _find_inner_model, unpatch_model,
)

DEFAULT_MODEL = "/Volumes/toshiba/models/qwen3-8b-mlx"
N_VALUES = [1, 2, 4, 8, 16, 32, 50, 64, 100, 200, 500]
WARMUP_ITERS = 2
MEASURE_ITERS = 5


def get_mem_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except AttributeError:
        return mx.metal.get_active_memory() / (1024 * 1024)


def bench_standard_prefill(model, inner_model, tokens, n):
    """Time a standard prefill of N tokens (produces KV via normal forward)."""
    prompt = tokens[:n].reshape(1, -1)
    num_layers = len(inner_model.layers)
    times = []

    for i in range(WARMUP_ITERS + MEASURE_ITERS):
        cache = [KVCache() for _ in range(num_layers)]
        gc.collect()
        mx.eval(prompt)

        t0 = time.perf_counter()
        out = model(prompt, cache=cache)
        mx.eval(out)
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            times.append(t1 - t0)

    return sum(times) / len(times)


def bench_reconstruction(inner_model, h0_store, n):
    """Time reconstruct_prefix_kv for N tokens from pre-captured h^(0)."""
    if h0_store.count < n:
        return float('inf')

    times = []

    for i in range(WARMUP_ITERS + MEASURE_ITERS):
        gc.collect()

        t0 = time.perf_counter()
        kv_list = reconstruct_prefix_kv(inner_model, h0_store, 0, n)
        # Force evaluation
        for k, v in kv_list:
            mx.eval(k, v)
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            times.append(t1 - t0)

    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser(description="N Crossover Benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model, tokenizer = load(args.model)

    # Warmup
    warmup_tokens = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(warmup_tokens)
    mx.eval(model.parameters())
    print("Model loaded and warmed up.\n")

    # Prepare a long prompt for h^(0) capture
    max_n = max(N_VALUES)
    text = "The development of artificial intelligence " * 200
    all_tokens = mx.array(tokenizer.encode(text))
    if all_tokens.shape[0] < max_n:
        print(f"ERROR: need {max_n} tokens but only got {all_tokens.shape[0]}")
        return
    all_tokens = all_tokens[:max_n]
    print(f"Prepared {max_n} tokens for benchmarking.\n")

    # Capture h^(0) for all tokens via a single prefill with capture patch
    print("Capturing h^(0) for all tokens...")
    h0_store = H0Store(quant=None)  # bf16 exact
    apply_h0_capture_only(model, h0_store)

    # Run prefill to capture h^(0)
    cache_for_capture = make_prompt_cache(model)
    prompt_batch = all_tokens.reshape(1, -1)
    out = model(prompt_batch, cache=cache_for_capture)
    mx.eval(out)
    print(f"Captured h^(0) for {h0_store.count} tokens. H0 size: {h0_store.nbytes / 1024 / 1024:.1f} MB\n")

    # Unpatch model so standard prefill measures clean forward pass
    unpatch_model(model)
    print("Model unpatched for clean prefill measurement.\n")

    inner_model = _find_inner_model(model)

    # Run benchmarks
    print(f"{'N':>6} | {'Prefill (ms)':>12} | {'Recon (ms)':>12} | {'Ratio':>8} | {'Winner':>10}")
    print(f"{'-'*60}")

    results = []
    for n in N_VALUES:
        print(f"  Testing N={n}...", end="", flush=True)

        t_prefill = bench_standard_prefill(model, inner_model, all_tokens, n)
        t_recon = bench_reconstruction(inner_model, h0_store, n)

        ratio = t_recon / t_prefill if t_prefill > 0 else float('inf')
        winner = "RECON" if ratio < 1.0 else "PREFILL"

        results.append({
            "n": n,
            "prefill_ms": t_prefill * 1000,
            "recon_ms": t_recon * 1000,
            "ratio": ratio,
            "winner": winner,
        })

        print(f"\r{n:>6} | {t_prefill*1000:>11.2f}ms | {t_recon*1000:>11.2f}ms | {ratio:>7.2f}x | {winner:>10}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: N Crossover Benchmark")
    print(f"  Model: {args.model.split('/')[-1]}")
    print(f"  h^(0): bf16 exact")
    print(f"{'='*60}")

    crossover_n = None
    for r in results:
        if r["ratio"] < 1.0 and crossover_n is None:
            crossover_n = r["n"]

    if crossover_n:
        print(f"\n  CROSSOVER at N={crossover_n}")
        print(f"  → Reconstruction faster than prefill for N >= {crossover_n}")
        print(f"  → Recommended min_tokens for recall: {crossover_n}")
    else:
        print(f"\n  NO CROSSOVER found in range [1, {max(N_VALUES)}]")
        print(f"  → Prefill always faster (reconstruction not beneficial)")

    # Detailed table
    print(f"\n  {'N':>6} | {'Prefill':>10} | {'Recon':>10} | {'Speedup':>10}")
    print(f"  {'-'*45}")
    for r in results:
        speedup = (1.0 / r["ratio"] - 1) * 100 if r["ratio"] > 0 else 0
        marker = " ★" if r["ratio"] < 1.0 else ""
        print(f"  {r['n']:>6} | {r['prefill_ms']:>9.2f}ms | {r['recon_ms']:>9.2f}ms | {speedup:>+9.1f}%{marker}")


if __name__ == "__main__":
    main()
