#!/usr/bin/env python3
"""
KV Quantization: Head-aligned group_size hypothesis.

Key finding from diagnostic: int8 + group_size=128 (= head_dim) gives EXACT MATCH
on 1.7B, while group_size=64 diverges. Hypothesis: aligning quantization groups
with head boundaries preserves attention-critical structure.

This test validates:
1. group_size=head_dim specifically (vs sub-head and super-head)
2. Prefill-only quant (decode KV stays exact) vs all-quant
3. Longer generation (100 tokens)
4. Partial layer quantization (first N layers only)
5. 1.7B and 8B models
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


def prefill_and_get_cache(model, tokenizer, prompt):
    """Do a full prefill and return the exact KVCache."""
    cache = make_prompt_cache(model)
    tokens = mx.array(tokenizer.encode(prompt))
    out = model(tokens.reshape(1, -1), cache=cache)
    mx.eval(out)
    return cache, tokens


def generate_from_cache(model, tokenizer, cache, question, max_tokens=50):
    """Generate from a pre-populated cache."""
    q_tokens = mx.array(tokenizer.encode(question))
    result = []
    for tok, _ in generate_step(q_tokens, model, max_tokens=max_tokens,
                                 sampler=GREEDY, prompt_cache=cache):
        result.append(tok)
    return result, tokenizer.decode(result)


def clone_cache_quantized(exact_cache, n_layers, bits, group_size,
                           quant_layers=None):
    """Clone exact cache with quantization applied to specified layers.

    Args:
        quant_layers: set of layer indices to quantize. None = all layers.
    """
    from mlx_lm.models.cache import KVCache

    new_cache = [KVCache() for _ in range(n_layers)]
    for i in range(n_layers):
        k_exact, v_exact = exact_cache[i].state
        mx.eval(k_exact, v_exact)

        should_quant = (quant_layers is None or i in quant_layers) and bits < 16
        if not should_quant:
            # Keep exact
            new_cache[i].state = (k_exact, v_exact)
        else:
            q_k = mx.quantize(k_exact, group_size=group_size, bits=bits)
            q_v = mx.quantize(v_exact, group_size=group_size, bits=bits)
            k_deq = mx.dequantize(*q_k, group_size=group_size, bits=bits)
            v_deq = mx.dequantize(*q_v, group_size=group_size, bits=bits)
            mx.eval(k_deq, v_deq)
            new_cache[i].state = (k_deq, v_deq)

    mx.eval([c.keys for c in new_cache] + [c.values for c in new_cache])
    return new_cache


def compare(baseline, generated, label):
    """Compare generated tokens vs baseline."""
    match = sum(1 for a, b in zip(baseline, generated) if a == b)
    total = min(len(baseline), len(generated))
    exact = baseline == generated
    status = "EXACT ✓" if exact else f"{match}/{total}"
    print(f"  {label:45s} │ {status}")
    return exact, match, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=50)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    inner = model.model if hasattr(model, 'model') else model
    n_layers = len(inner.layers)
    head_dim = inner.layers[0].self_attn.rope.dims  # RoPE dims = head_dim
    n_kv_heads = inner.layers[0].self_attn.n_kv_heads
    n_heads = inner.layers[0].self_attn.n_heads
    d_model = inner.layers[0].hidden_size

    print("=" * 70)
    print("KV Quantization: Head-aligned Group Size")
    print("=" * 70)
    print(f"Model: {args.model.split('/')[-1]}")
    print(f"Layers={n_layers}, d_model={d_model}, head_dim={head_dim}")
    print(f"n_heads={n_heads}, n_kv_heads={n_kv_heads}")
    print(f"Prompt tokens: {args.prompt_tokens}, Gen tokens: {args.gen_tokens}")

    # Build prompt
    base = "The quick brown fox jumps over the lazy dog. " * 100
    tokens = tokenizer.encode(base)[:args.prompt_tokens]
    prompt = tokenizer.decode(tokens)

    # Prefill
    print(f"\nPrefilling {len(tokens)} tokens...")
    exact_cache, _ = prefill_and_get_cache(model, tokenizer, prompt)

    question = "\nQ: What is the main topic?\nA:"

    # Baseline
    baseline_cache = clone_cache_quantized(exact_cache, n_layers, 16, 64)
    baseline_tok, baseline_text = generate_from_cache(
        model, tokenizer, baseline_cache, question, args.gen_tokens)
    print(f"Baseline: {baseline_text[:120]}")

    # ── Part 1: Group size sweep (all layers, int8) ──
    print(f"\n{'═' * 70}")
    print("Part 1: int8 Group Size Sweep (all layers quantized)")
    print(f"{'═' * 70}")
    print(f"  {'Config':45s} │ {'Quality':>10}")
    print(f"  {'─' * 45}─┼{'─' * 10}")

    gs_candidates = sorted(set([32, 64, head_dim]) & {32, 64, 128})
    for gs in gs_candidates:
        label = f"int8, gs={gs}" + (" (= head_dim)" if gs == head_dim else "")
        qcache = clone_cache_quantized(exact_cache, n_layers, 8, gs)
        gen_tok, _ = generate_from_cache(model, tokenizer, qcache, question, args.gen_tokens)
        compare(baseline_tok, gen_tok, label)

    # ── Part 2: Group size sweep (all layers, int4) ──
    print(f"\n{'═' * 70}")
    print("Part 2: int4 Group Size Sweep (all layers quantized)")
    print(f"{'═' * 70}")
    print(f"  {'Config':45s} │ {'Quality':>10}")
    print(f"  {'─' * 45}─┼{'─' * 10}")

    for gs in gs_candidates:
        label = f"int4, gs={gs}" + (" (= head_dim)" if gs == head_dim else "")
        qcache = clone_cache_quantized(exact_cache, n_layers, 4, gs)
        gen_tok, _ = generate_from_cache(model, tokenizer, qcache, question, args.gen_tokens)
        compare(baseline_tok, gen_tok, label)

    # ── Part 3: Partial layer quantization with int8 + head_dim ──
    print(f"\n{'═' * 70}")
    print(f"Part 3: Partial Layer Quantization (int8, gs={head_dim})")
    print(f"{'═' * 70}")
    print(f"  {'Config':45s} │ {'Quality':>10}")
    print(f"  {'─' * 45}─┼{'─' * 10}")

    # Quantize first N layers (cloud→edge transfer scenario)
    for n_quant in [2, 4, 8, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]:
        n_quant = min(n_quant, n_layers)
        quant_set = set(range(n_quant))
        label = f"quant layers 0..{n_quant-1} ({n_quant}/{n_layers})"
        qcache = clone_cache_quantized(exact_cache, n_layers, 8, head_dim, quant_set)
        gen_tok, _ = generate_from_cache(model, tokenizer, qcache, question, args.gen_tokens)
        compare(baseline_tok, gen_tok, label)

    # ── Part 4: Partial layer quantization with int4 + head_dim ──
    print(f"\n{'═' * 70}")
    print(f"Part 4: Partial Layer Quantization (int4, gs={head_dim})")
    print(f"{'═' * 70}")
    print(f"  {'Config':45s} │ {'Quality':>10}")
    print(f"  {'─' * 45}─┼{'─' * 10}")

    for n_quant in [2, 4, 8, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]:
        n_quant = min(n_quant, n_layers)
        quant_set = set(range(n_quant))
        label = f"quant layers 0..{n_quant-1} ({n_quant}/{n_layers})"
        qcache = clone_cache_quantized(exact_cache, n_layers, 4, head_dim, quant_set)
        gen_tok, _ = generate_from_cache(model, tokenizer, qcache, question, args.gen_tokens)
        compare(baseline_tok, gen_tok, label)

    # ── Part 5: Transfer size comparison ──
    print(f"\n{'═' * 70}")
    print("Part 5: Transfer Size Analysis")
    print(f"{'═' * 70}")

    # Exact KV size for all layers
    exact_kv_bytes = 0
    for i in range(n_layers):
        k, v = exact_cache[i].state
        exact_kv_bytes += k.nbytes + v.nbytes

    print(f"  Exact KV (bf16):  {exact_kv_bytes / 1024 / 1024:8.1f} MB")

    for bits in [8, 4]:
        total = 0
        for i in range(n_layers):
            k, v = exact_cache[i].state
            q_k = mx.quantize(k, group_size=head_dim, bits=bits)
            q_v = mx.quantize(v, group_size=head_dim, bits=bits)
            total += sum(x.nbytes for x in q_k) + sum(x.nbytes for x in q_v)
        ratio = exact_kv_bytes / total
        print(f"  int{bits} (gs={head_dim}): {total / 1024 / 1024:8.1f} MB  ({ratio:.1f}× compression)")

    # h^(0) for reference
    h0 = exact_cache[0].keys  # Not h(0), but gives size reference
    h0_size = args.prompt_tokens * d_model * 2  # bf16
    print(f"  h^(0) reference:  {h0_size / 1024 / 1024:8.1f} MB  ({exact_kv_bytes / h0_size:.1f}× vs exact KV)")


if __name__ == "__main__":
    main()
