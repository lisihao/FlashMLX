"""
Micro-benchmark: Reconstruction chunk_size optimization.

Tests h^(0) → K/V reconstruction latency with different chunk_size values.
Answers: "Is chunk_size=0 (no chunking) faster than chunk_size=256 (default)?"

Usage:
    python3 benchmarks/bench_recon_chunking.py /path/to/model
    python3 benchmarks/bench_recon_chunking.py /path/to/model --prompt-tokens 16384
"""

import argparse
import gc
import sys
import time

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx

from flashmlx import ReconstructionController
from flashmlx.model_cards import load_card_or_detect
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.cache_factory import get_cache_info
from mlx_lm.models.kv_direct_cache import unpatch_model


def get_mem_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except AttributeError:
        return mx.metal.get_active_memory() / (1024 * 1024)


FILLER_PARA = (
    "The development of artificial intelligence has progressed rapidly in recent years. "
    "Machine learning algorithms continue to improve across various benchmarks. Research "
    "teams around the world are exploring new architectures for language understanding. "
    "The computational requirements for training large models have grown exponentially. "
    "Transfer learning has enabled smaller teams to build on pre-trained foundations. "
    "Ethical considerations remain central to AI development discussions globally. "
)


def build_filler_prompt(tokenizer, target_tokens):
    """Build a filler prompt of approximately target_tokens length."""
    filler_tokens = len(tokenizer.encode(FILLER_PARA))
    n_paras = (target_tokens // filler_tokens) + 5

    prefix = "Read the following document carefully.\n\n"
    text = prefix + FILLER_PARA * n_paras
    tokens = tokenizer.encode(text)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        text = tokenizer.decode(tokens)
    return text, len(tokens)


def run_recon_trial(model, tokenizer, prompt, cache_kwargs, chunk_size, eval_every, label):
    """Run a single reconstruction trial and return timing info."""
    gc.collect()
    mx.clear_cache()
    time.sleep(0.3)

    unpatch_model(model)
    mem_before = get_mem_mb()

    cache = make_prompt_cache(model, **cache_kwargs)

    # Phase 1: Prefill (fills h0_store)
    prompt_tokens = mx.array(tokenizer.encode(prompt))
    n_tokens = prompt_tokens.shape[0]

    t0 = time.perf_counter()
    model_out = model(prompt_tokens.reshape(1, -1), cache=cache)
    mx.eval(model_out)
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Check h0_store (TripleLayerKVCache uses "h0_count", KVDirectCache uses "h0_tokens")
    info = get_cache_info(cache)
    h0_tokens = info.get("h0_count", 0) or info.get("h0_tokens", 0)

    if h0_tokens == 0:
        return {
            "label": label,
            "chunk_size": chunk_size,
            "n_tokens": n_tokens,
            "h0_tokens": 0,
            "prefill_ms": prefill_ms,
            "recon_ms": 0,
            "mem_delta_mb": 0,
            "error": f"no h0 tokens (info keys: {list(info.keys())})",
        }

    # Phase 2: Reconstruct with specified chunk_size
    recon = ReconstructionController.from_cache(cache, model)
    if not recon.available:
        return {
            "label": label,
            "chunk_size": chunk_size,
            "n_tokens": n_tokens,
            "h0_tokens": h0_tokens,
            "prefill_ms": prefill_ms,
            "recon_ms": 0,
            "mem_delta_mb": 0,
            "error": "recon not available",
        }

    t1 = time.perf_counter()
    result = recon.reconstruct(
        strategy="full",
        chunk_size=chunk_size,
        eval_every=eval_every,
    )
    recon_ms = (time.perf_counter() - t1) * 1000

    mem_after = get_mem_mb()

    return {
        "label": label,
        "chunk_size": chunk_size,
        "eval_every": eval_every,
        "n_tokens": n_tokens,
        "h0_tokens": h0_tokens,
        "recon_tokens": result.tokens_reconstructed,
        "layers": result.layers_injected,
        "prefill_ms": prefill_ms,
        "recon_ms": recon_ms,
        "recon_ms_api": result.time_ms,
        "mem_delta_mb": mem_after - mem_before,
        "error": result.error,
    }


def main():
    parser = argparse.ArgumentParser(description="Reconstruction chunk_size benchmark")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--prompt-tokens", type=int, default=8192,
                        help="Target number of prompt tokens")
    parser.add_argument("--runs", type=int, default=2,
                        help="Number of runs per config (best of N)")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load(args.model_path)

    card = load_card_or_detect(model, args.model_path)
    print(f"Card: {card.model_name}")

    # Use recall_first mode (enables h0_store)
    cache_kwargs = card.to_cache_kwargs(mode="recall_first")
    # Remove auto_reconstruct to test manual reconstruction
    cache_kwargs = {k: v for k, v in cache_kwargs.items() if k != "auto_reconstruct"}
    print(f"Cache kwargs: {cache_kwargs}")

    # Build prompt
    print(f"\nBuilding {args.prompt_tokens:,}-token filler prompt...")
    prompt, actual_tokens = build_filler_prompt(tokenizer, args.prompt_tokens)
    print(f"  Actual: {actual_tokens:,} tokens")

    # Warmup
    print("Warming up...")
    warm = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(warm)
    mx.eval(model.parameters())

    # Test configs: (chunk_size, eval_every, label)
    configs = [
        (256,  4,  "chunk=256, eval/4 (default)"),
        (256,  8,  "chunk=256, eval/8"),
        (512,  4,  "chunk=512, eval/4"),
        (512,  8,  "chunk=512, eval/8"),
        (1024, 4,  "chunk=1024, eval/4"),
        (1024, 8,  "chunk=1024, eval/8"),
        (2048, 4,  "chunk=2048, eval/4"),
        (0,    1,  "chunk=0 (no chunking)"),
    ]

    print(f"\n{'='*100}")
    print(f"  Reconstruction Chunk Size Benchmark | {card.model_name} | {actual_tokens:,} tokens | {args.runs} runs each")
    print(f"{'='*100}")
    print(f"  {'Config':<30} {'h0 tok':>8} {'Recon ms':>10} {'Prefill ms':>12} "
          f"{'Ratio':>8} {'Mem MB':>8} {'Layers':>7}")
    print(f"  {'-'*90}")

    results = []

    for chunk_size, eval_every, label in configs:
        best = None
        for run in range(args.runs):
            r = run_recon_trial(
                model, tokenizer, prompt, cache_kwargs,
                chunk_size=chunk_size,
                eval_every=eval_every,
                label=label,
            )
            if r.get("error"):
                print(f"  {label:<30} ERROR: {r['error']}")
                break
            if best is None or r["recon_ms"] < best["recon_ms"]:
                best = r

        if best and not best.get("error"):
            ratio = best["recon_ms"] / best["prefill_ms"] if best["prefill_ms"] > 0 else 0
            print(f"  {label:<30} {best['h0_tokens']:>8} {best['recon_ms']:>9.0f}ms "
                  f"{best['prefill_ms']:>11.0f}ms {ratio:>7.2f}x {best['mem_delta_mb']:>7.0f}M "
                  f"{best.get('layers', 0):>7}")
            results.append(best)

    # Summary
    if len(results) >= 2:
        baseline = results[0]  # chunk=256 default
        print(f"\n  --- vs baseline (chunk=256, eval/4) ---")
        for r in results:
            delta = ((r["recon_ms"] - baseline["recon_ms"]) / baseline["recon_ms"]) * 100
            sign = "+" if delta > 0 else ""
            print(f"  {r['label']:<30}  {sign}{delta:.1f}%  ({r['recon_ms']:.0f}ms vs {baseline['recon_ms']:.0f}ms)")

    print(f"\n  Key insight: recon/prefill ratio shows reconstruction cost relative to original prefill.")
    print(f"  Ratio ~1.0x means reconstruction ≈ second prefill (expected for full h^(0) replay).")


if __name__ == "__main__":
    main()
