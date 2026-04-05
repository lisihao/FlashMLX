#!/usr/bin/env python3
"""
TurboAngle Benchmark v2 - With Real Cache Integration

Compares quantization methods with actual cache usage:
- Standard (no compression)
- Triple PolarQuant (FlashMLX default)
- TurboAngle baseline (K128V64, uniform)
- TurboAngle E4 boost (per-layer, from preset)

Metrics:
- Perplexity (WikiText-2 sample)
- Memory usage (peak)
- Inference speed (tokens/sec)
"""

import sys
import os
sys.path.insert(0, "mlx-lm-source")

import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import numpy as np


def load_wikitext2_sample(max_tokens=2048):
    """
    Load WikiText-2 validation set sample.

    Uses a realistic longer text for perplexity evaluation.
    """
    # Use a realistic longer text (placeholder for WikiText-2)
    text = """
    The Tower of London is a historic castle located on the north bank of the River Thames
    in central London. It was founded towards the end of 1066 as part of the Norman Conquest
    of England. The White Tower, which gives the entire castle its name, was built by William
    the Conqueror in 1078 and was a resented symbol of oppression, inflicted upon London by
    the new ruling elite. The castle was used as a prison from 1100 until 1952, although that
    was not its primary purpose. A grand palace early in its history, it served as a royal
    residence. As a whole, the Tower is a complex of several buildings set within two concentric
    rings of defensive walls and a moat. There have been several phases of expansion, mainly
    under Kings Richard I, Henry III, and Edward I in the 12th and 13th centuries. The general
    layout established by the late 13th century remains despite later activity on the site.

    The Tower of London has played a prominent role in English history. It was besieged several
    times, and controlling it has been important to controlling the country. The Tower has served
    variously as an armoury, a treasury, a menagerie, the home of the Royal Mint, a public record
    office, and the home of the Crown Jewels of England. From the early 14th century until the
    reign of Charles II, a procession would be led from the Tower to Westminster Abbey on the
    coronation of a monarch. In the absence of the monarch, the Constable of the Tower is in
    charge of the castle. This was a powerful and trusted position in the medieval period.

    In the late 15th century, the castle was the prison of the Princes in the Tower. Under the
    Tudors, the Tower became used less as a royal residence, and despite attempts to refortify
    and repair the castle, its defences lagged behind developments to deal with artillery. The
    peak period of the castle's use as a prison was the 16th and 17th centuries, when many
    figures who had fallen into disgrace, such as Elizabeth I before she became queen, were held
    within its walls. This use has led to the phrase "sent to the Tower". Despite its enduring
    reputation as a place of torture and death, popularised by 16th-century religious
    propagandists and 19th-century writers, only seven people were executed within the Tower
    before the World Wars of the 20th century.
    """ * 3  # ~1500 tokens

    return text


def compute_perplexity(model, tokenizer, text, cache=None):
    """
    Compute perplexity on text.

    Parameters
    ----------
    model : nn.Module
        Language model
    tokenizer : Tokenizer
        Tokenizer
    text : str
        Input text
    cache : KVCache or None
        Optional KV cache

    Returns
    -------
    ppl : float
        Perplexity
    """
    # Tokenize
    tokens = tokenizer.encode(text)
    tokens_mx = mx.array([tokens])

    # Forward pass
    logits = model(tokens_mx, cache=cache)

    # Compute log-likelihood
    # logits: [1, seq_len, vocab_size]
    # tokens: [seq_len]
    vocab_size = logits.shape[-1]

    # Shift: predict token i+1 from context up to i
    logits_shifted = logits[:, :-1, :]  # [1, seq_len-1, vocab]
    targets = mx.array(tokens[1:])       # [seq_len-1]

    # Compute cross-entropy
    # MLX doesn't have log_softmax, use softmax + log
    probs = mx.softmax(logits_shifted, axis=-1)
    log_probs = mx.log(probs + 1e-10)  # Add epsilon to avoid log(0)

    # Gather log probs for target tokens
    target_log_probs = []
    for i, target in enumerate(targets.tolist()):
        target_log_probs.append(log_probs[0, i, target].item())

    # Perplexity = exp(-mean(log_prob))
    mean_log_prob = np.mean(target_log_probs)
    ppl = np.exp(-mean_log_prob)

    return ppl


def benchmark_cache_strategy(
    model,
    tokenizer,
    text,
    strategy_name,
    cache_kwargs=None,
):
    """
    Benchmark a cache strategy.

    Parameters
    ----------
    model : nn.Module
        Model
    tokenizer : Tokenizer
        Tokenizer
    text : str
        Test text
    strategy_name : str
        Strategy name for display
    cache_kwargs : dict or None
        Arguments for make_prompt_cache()

    Returns
    -------
    results : dict
        {ppl, memory_mb, tok_per_sec, time_sec}
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking: {strategy_name}")
    if cache_kwargs:
        print(f"  Config: {cache_kwargs}")
    print('='*80)

    # Create cache
    cache_kwargs = cache_kwargs or {}
    cache = make_prompt_cache(model, **cache_kwargs)

    # Tokenize
    tokens = tokenizer.encode(text)
    print(f"  Tokens: {len(tokens)}")

    # Measure memory
    mx.clear_cache()
    mx.reset_peak_memory()

    # Warmup
    _ = model(mx.array([tokens[:100]]), cache=cache)
    mx.eval(_)

    # Reset for actual measurement
    mx.clear_cache()
    mx.reset_peak_memory()

    # Actual run
    start = time.perf_counter()
    logits = model(mx.array([tokens]), cache=cache)
    mx.eval(logits)
    elapsed = time.perf_counter() - start

    peak_mem = mx.get_peak_memory() / (1024**2)  # MB

    # Compute perplexity
    # Reset cache for clean perplexity measurement
    cache = make_prompt_cache(model, **cache_kwargs)
    ppl = compute_perplexity(model, tokenizer, text, cache=cache)

    # Estimate tokens/sec (rough approximation)
    tok_per_sec = len(tokens) / elapsed

    print(f"\n  Results:")
    print(f"    Perplexity: {ppl:.4f}")
    print(f"    Peak Memory: {peak_mem:.1f} MB")
    print(f"    Tokens/sec: {tok_per_sec:.1f}")
    print(f"    Total time: {elapsed:.2f}s")

    return {
        'ppl': ppl,
        'memory_mb': peak_mem,
        'tok_per_sec': tok_per_sec,
        'time_sec': elapsed,
    }


def main():
    """Run benchmark."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 22 + "TurboAngle Benchmark v2" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Configuration
    MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"
    MAX_TOKENS = 1500  # Realistic length for initial test

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model not found: {MODEL_PATH}")
        print("Please update MODEL_PATH in the script")
        return

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    num_layers = len(model.model.layers)
    print(f"✅ Model loaded: {num_layers} layers")
    print()

    # Load test text
    text = load_wikitext2_sample(max_tokens=MAX_TOKENS)
    print(f"Test text: ~{len(text)} chars")
    print()

    # Benchmark configurations
    configs = [
        ("Standard (no compression)", {}),

        ("Triple PolarQuant", {
            "kv_cache": "triple_pq",
        }),

        ("TurboAngle Baseline (K128V64, uniform)", {
            "kv_cache": "triple_pq",
            "kv_layer_quantizers": {
                i: None for i in range(num_layers)  # Will use TurboAngle baseline
            },
        }),

        ("TurboAngle E4 (preset: mistral-7b)", {
            "kv_cache": "triple_pq",
            "kv_layer_quantizers": "mistral-7b",  # Per-layer preset
        }),
    ]

    # Create uniform TurboAngle baseline
    from mlx_lm.models.turboangle import TurboAngleQuantizer

    # Update config 3 with actual quantizers
    baseline_quantizers = {}
    for i in range(num_layers):
        baseline_quantizers[i] = TurboAngleQuantizer(n_k=128, n_v=64, head_dim=128)

    configs[2] = (
        "TurboAngle Baseline (K128V64, uniform)",
        {
            "kv_cache": "triple_pq",
            "kv_layer_quantizers": baseline_quantizers,
        }
    )

    results = []
    for name, kwargs in configs:
        try:
            result = benchmark_cache_strategy(
                model, tokenizer, text, name, kwargs
            )
            result['name'] = name
            results.append(result)
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 32 + "Summary" + " " * 40 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    if not results:
        print("No results to display")
        return

    # Print table
    print(f"{'Method':<45} {'PPL':>8} {'Mem(MB)':>10} {'Tok/s':>8} {'Time(s)':>8}")
    print("-" * 80)

    baseline_ppl = results[0]['ppl'] if results else 1.0

    for r in results:
        name = r['name']
        ppl_delta = r['ppl'] - baseline_ppl

        # Format with delta
        if abs(ppl_delta) < 0.0001:
            ppl_str = f"{r['ppl']:.4f}"
        else:
            ppl_str = f"{r['ppl']:.4f} ({ppl_delta:+.4f})"

        print(
            f"{name:<45} "
            f"{ppl_str:>8} "
            f"{r['memory_mb']:>10.1f} "
            f"{r['tok_per_sec']:>8.1f} "
            f"{r['time_sec']:>8.2f}"
        )

    print()
    print("Notes:")
    print("  - PPL = Perplexity (lower is better)")
    print("  - ΔPPL = Change vs standard baseline")
    print("  - Mem = Peak memory usage")
    print("  - Tok/s = Throughput (higher is better)")
    print()

    # Analysis
    print("Expected from paper (Table 3):")
    print("  - Mistral-7B E4: ΔPPL ≈ +0.0002 (near-lossless)")
    print("  - TurboAngle should achieve |ΔPPL| < 0.01")
    print()


if __name__ == "__main__":
    main()
