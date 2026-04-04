#!/usr/bin/env python3
"""
KV Quantization Diagnostic: Native QuantizedKVCache vs Manual Dequantize.

Tests whether MLX's native KV quantization (QuantizedKVCache) produces
correct generation, and compares with our manual quantize→dequantize approach.

This isolates the root cause of the quality issue in edge_cloud_progressive.py.
"""

from __future__ import annotations

import sys
import argparse

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache
from mlx_lm.sample_utils import make_sampler

GREEDY = make_sampler(temp=0.0)


def generate_tokens(model, tokenizer, prompt, max_tokens=50, cache=None):
    """Generate tokens with optional pre-populated cache."""
    tokens = mx.array(tokenizer.encode(prompt))
    result = []
    for tok, _ in generate_step(tokens, model, max_tokens=max_tokens,
                                 sampler=GREEDY, prompt_cache=cache):
        result.append(tok)
    return result, tokenizer.decode(result)


def make_quantized_cache(model, bits=8, group_size=64):
    """Create a QuantizedKVCache for each layer."""
    # Count layers
    kv_heads = getattr(model, 'args', None)
    n_layers = len(model.model.layers) if hasattr(model, 'model') else len(model.layers)
    return [QuantizedKVCache(group_size=group_size, bits=bits) for _ in range(n_layers)]


def test_native_quantized_cache(model, tokenizer, prompt, max_tokens=50,
                                 bits=8, group_size=64):
    """Test generation with native QuantizedKVCache."""
    cache = make_quantized_cache(model, bits=bits, group_size=group_size)
    tokens = mx.array(tokenizer.encode(prompt))
    result = []
    for tok, _ in generate_step(tokens, model, max_tokens=max_tokens,
                                 sampler=GREEDY, prompt_cache=cache):
        result.append(tok)
    return result, tokenizer.decode(result)


def test_manual_dequantize(model, tokenizer, prompt, max_tokens=50,
                            bits=8, group_size=64):
    """Test: exact prefill → quantize KV → dequantize → inject into regular cache → generate."""
    # Step 1: Exact prefill with regular cache
    exact_cache = make_prompt_cache(model)
    tokens = mx.array(tokenizer.encode(prompt))

    # Run prefill to populate exact_cache
    # We use generate_step with max_tokens=0 to just do prefill
    # Actually, generate_step always generates at least 1 token, so let's do it manually
    prefill_result = model(tokens.reshape(1, -1), cache=exact_cache)
    mx.eval(prefill_result)

    # Step 2: Extract, quantize, dequantize, inject into new regular cache
    n_layers = len(exact_cache)
    test_cache = make_prompt_cache(model)

    for i in range(n_layers):
        k_exact, v_exact = exact_cache[i].state
        mx.eval(k_exact, v_exact)

        if bits == 16:
            test_cache[i].state = (k_exact, v_exact)
        else:
            # Quantize
            q_k = mx.quantize(k_exact, group_size=group_size, bits=bits)
            q_v = mx.quantize(v_exact, group_size=group_size, bits=bits)
            mx.eval(*q_k, *q_v)

            # Dequantize
            k_deq = mx.dequantize(*q_k, group_size=group_size, bits=bits)
            v_deq = mx.dequantize(*q_v, group_size=group_size, bits=bits)
            mx.eval(k_deq, v_deq)

            test_cache[i].state = (k_deq, v_deq)

    mx.eval([c.keys for c in test_cache] + [c.values for c in test_cache])

    # Step 3: Generate from the dequantized cache
    question = "\nQ: What is the main topic?\nA:"
    q_tokens = mx.array(tokenizer.encode(question))
    result = []
    for tok, _ in generate_step(q_tokens, model, max_tokens=max_tokens,
                                 sampler=GREEDY, prompt_cache=test_cache):
        result.append(tok)

    return result, tokenizer.decode(result), exact_cache


def test_inject_into_quantized_cache(model, tokenizer, prompt, exact_cache,
                                      max_tokens=50, bits=8, group_size=64):
    """Test: take exact KV → inject directly into QuantizedKVCache (keep quantized for attention)."""
    n_layers = len(exact_cache)
    qcache = make_quantized_cache(model, bits=bits, group_size=group_size)

    for i in range(n_layers):
        k_exact, v_exact = exact_cache[i].state
        mx.eval(k_exact, v_exact)

        # Quantize the exact KV
        q_k = mx.quantize(k_exact, group_size=group_size, bits=bits)
        q_v = mx.quantize(v_exact, group_size=group_size, bits=bits)
        mx.eval(*q_k, *q_v)

        # Inject quantized tuples directly into QuantizedKVCache
        qcache[i].keys = q_k
        qcache[i].values = q_v
        qcache[i].offset = k_exact.shape[-2]

    # Generate using quantized attention path
    question = "\nQ: What is the main topic?\nA:"
    q_tokens = mx.array(tokenizer.encode(question))
    result = []
    for tok, _ in generate_step(q_tokens, model, max_tokens=max_tokens,
                                 sampler=GREEDY, prompt_cache=qcache):
        result.append(tok)
    return result, tokenizer.decode(result)


def measure_kv_error(exact_cache, bits, group_size, n_layers):
    """Measure quantization error across all layers."""
    k_errs, v_errs = [], []
    for i in range(n_layers):
        k_exact, v_exact = exact_cache[i].state
        mx.eval(k_exact, v_exact)

        q_k = mx.quantize(k_exact, group_size=group_size, bits=bits)
        q_v = mx.quantize(v_exact, group_size=group_size, bits=bits)
        k_deq = mx.dequantize(*q_k, group_size=group_size, bits=bits)
        v_deq = mx.dequantize(*q_v, group_size=group_size, bits=bits)
        mx.eval(k_deq, v_deq)

        k_rms = float(mx.sqrt(mx.mean((k_deq - k_exact) ** 2)).item())
        v_rms = float(mx.sqrt(mx.mean((v_deq - v_exact) ** 2)).item())
        k_norm = float(mx.sqrt(mx.mean(k_exact ** 2)).item())
        v_norm = float(mx.sqrt(mx.mean(v_exact ** 2)).item())

        k_errs.append(k_rms / max(k_norm, 1e-8))
        v_errs.append(v_rms / max(v_norm, 1e-8))

    return k_errs, v_errs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    args = parser.parse_args()

    print("=" * 70)
    print("KV Quantization Diagnostic")
    print("=" * 70)

    model, tokenizer = load(args.model)
    inner = model.model if hasattr(model, 'model') else model
    n_layers = len(inner.layers)
    d_model = inner.layers[0].hidden_size
    print(f"Model: {args.model.split('/')[-1]}")
    print(f"Layers: {n_layers}, d_model: {d_model}")

    # Build a prompt of target length
    base = "The quick brown fox jumps over the lazy dog. " * 100
    tokens = tokenizer.encode(base)[:args.prompt_tokens]
    prompt = tokenizer.decode(tokens)
    print(f"Prompt tokens: {len(tokens)}")

    # ── Test 0: Exact baseline ──
    print(f"\n{'─' * 70}")
    print("Test 0: Exact baseline (regular KVCache)")
    print(f"{'─' * 70}")

    question = prompt + "\nQ: What is the main topic?\nA:"
    baseline_tok, baseline_text = generate_tokens(model, tokenizer, question)
    print(f"  Tokens: {len(baseline_tok)}")
    print(f"  Text: {baseline_text[:120]}")

    # ── Test 1: Native QuantizedKVCache ──
    for bits in [8, 4]:
        for gs in [64, 32, 128]:
            print(f"\n{'─' * 70}")
            print(f"Test 1: Native QuantizedKVCache (bits={bits}, group_size={gs})")
            print(f"{'─' * 70}")

            native_tok, native_text = test_native_quantized_cache(
                model, tokenizer, question, bits=bits, group_size=gs)

            match = sum(1 for a, b in zip(baseline_tok, native_tok) if a == b)
            total = min(len(baseline_tok), len(native_tok))
            exact = baseline_tok == native_tok
            label = "EXACT ✓" if exact else f"{match}/{total} DIVERGED"
            print(f"  Match: {label}")
            if not exact:
                print(f"  Text: {native_text[:120]}")

    # ── Test 2: Manual dequantize → regular cache ──
    for bits in [8, 4]:
        for gs in [64, 32, 128]:
            print(f"\n{'─' * 70}")
            print(f"Test 2: Manual deq → regular cache (bits={bits}, group_size={gs})")
            print(f"{'─' * 70}")

            deq_tok, deq_text, exact_cache = test_manual_dequantize(
                model, tokenizer, prompt, bits=bits, group_size=gs)

            match = sum(1 for a, b in zip(baseline_tok, deq_tok) if a == b)
            total = min(len(baseline_tok), len(deq_tok))
            exact = baseline_tok == deq_tok
            label = "EXACT ✓" if exact else f"{match}/{total} DIVERGED"
            print(f"  Match: {label}")
            if not exact:
                print(f"  Text: {deq_text[:120]}")

    # ── Test 3: Inject quantized into QuantizedKVCache (use native attention) ──
    # First get exact cache from test 2
    _, _, exact_cache = test_manual_dequantize(model, tokenizer, prompt, bits=16)
    # Generate baseline from this exact cache
    q_text = "\nQ: What is the main topic?\nA:"
    q_tokens = mx.array(tokenizer.encode(q_text))
    base2_tok = []
    exact_c2 = make_prompt_cache(model)
    for i in range(len(exact_cache)):
        exact_c2[i].state = exact_cache[i].state
    mx.eval([c.keys for c in exact_c2] + [c.values for c in exact_c2])
    for tok, _ in generate_step(q_tokens, model, max_tokens=50,
                                 sampler=GREEDY, prompt_cache=exact_c2):
        base2_tok.append(tok)
    print(f"\n  Prefill-then-gen baseline: {len(base2_tok)} tokens")

    for bits in [8, 4]:
        for gs in [64, 32]:
            print(f"\n{'─' * 70}")
            print(f"Test 3: Inject quant → QuantizedKVCache native attn (bits={bits}, gs={gs})")
            print(f"{'─' * 70}")

            inject_tok, inject_text = test_inject_into_quantized_cache(
                model, tokenizer, prompt, exact_cache,
                bits=bits, group_size=gs)

            match = sum(1 for a, b in zip(base2_tok, inject_tok) if a == b)
            total = min(len(base2_tok), len(inject_tok))
            exact = base2_tok == inject_tok
            label = "EXACT ✓" if exact else f"{match}/{total} DIVERGED"
            print(f"  Match: {label}")
            if not exact:
                print(f"  Text: {inject_text[:120]}")

    # ── Test 4: Per-layer error analysis ──
    print(f"\n{'─' * 70}")
    print("Test 4: Per-layer quantization error")
    print(f"{'─' * 70}")

    # Use fresh exact cache
    _, _, exact_cache_4 = test_manual_dequantize(model, tokenizer, prompt, bits=16)

    for bits in [8, 4]:
        k_errs, v_errs = measure_kv_error(exact_cache_4, bits, 64, n_layers)
        print(f"\n  int{bits} (group_size=64):")
        print(f"    K err: mean={np.mean(k_errs):.6f}, max={np.max(k_errs):.6f} (layer {np.argmax(k_errs)})")
        print(f"    V err: mean={np.mean(v_errs):.6f}, max={np.max(v_errs):.6f} (layer {np.argmax(v_errs)})")

        # Show per-layer detail for worst 5
        combined = [(i, k_errs[i], v_errs[i]) for i in range(n_layers)]
        combined.sort(key=lambda x: x[1] + x[2], reverse=True)
        print(f"    Worst 5 layers:")
        for i, ke, ve in combined[:5]:
            print(f"      Layer {i:2d}: K={ke:.6f}, V={ve:.6f}")


if __name__ == "__main__":
    main()
