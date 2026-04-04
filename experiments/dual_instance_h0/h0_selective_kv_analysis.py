#!/usr/bin/env python3
"""
Selective KV Transfer: h^(0) for unimportant tokens + exact KV for attention sinks.

Core insight from the delta analysis:
  - Low-rank approximation of h^(l) fails because attention is sensitive to K/V errors
  - BUT: not all tokens matter equally for attention
  - Research shows: "attention sinks" (first few tokens) + "heavy hitters" (semantically key)
    receive ~80-90% of attention weight
  - Idea: send exact KV ONLY for important tokens, use h^(0) approximation for the rest

This script measures:
  1. Attention weight distribution: how concentrated is attention?
  2. If we keep exact KV for top-K% tokens + approximate the rest, what's the quality?
  3. Transfer size at various selectivity levels
  4. End-to-end generation quality comparison
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.kv_direct_cache import (
    _find_inner_model,
    reconstruct_prefix_kv,
    H0Store,
)
from mlx_lm.sample_utils import make_sampler

GREEDY = make_sampler(temp=0.0)


def capture_attention_weights(inner_model, tokens):
    """Run forward pass capturing attention weights at every layer.

    Returns:
        residuals: list of h^(0)..h^(L)
        attn_weights: list of (n_heads, N, N) attention weight matrices per layer
        kv_caches: list of KVCache objects
    """
    x = inner_model.embed_tokens(tokens.reshape(1, -1))
    mx.eval(x)

    N = tokens.shape[0]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(x.dtype)

    num_layers = len(inner_model.layers)
    cache = [KVCache() for _ in range(num_layers)]

    residuals = [x]
    all_attn_weights = []

    for i, layer in enumerate(inner_model.layers):
        # Run layer normally
        x_out = layer(x, mask=mask, cache=cache[i])
        mx.eval(x_out)

        # Also compute attention weights explicitly for analysis
        attn = layer.self_attn
        norm = layer.input_layernorm
        h_normed = norm(x.astype(mx.float32))

        B, L, D = h_normed.shape
        q = attn.q_proj(h_normed)
        k = attn.k_proj(h_normed)

        n_heads = attn.n_heads
        n_kv = attn.n_kv_heads
        head_dim = D // n_heads

        q = q.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, n_kv, head_dim).transpose(0, 2, 1, 3)

        # Expand KV heads for GQA
        if n_kv < n_heads:
            repeats = n_heads // n_kv
            k = mx.repeat(k, repeats, axis=1)

        scale = head_dim ** -0.5
        scores = (q * scale) @ k.transpose(0, 1, 3, 2)  # (B, n_heads, N, N)
        scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        mx.eval(weights)

        # Average across heads and batch, keep as (N, N) — row i = attention from token i
        w_avg = weights[0].mean(axis=0)  # (N, N)
        mx.eval(w_avg)
        all_attn_weights.append(np.array(w_avg.astype(mx.float32)))

        x = x_out
        residuals.append(x)

    return residuals, all_attn_weights, cache


def compute_token_importance(attn_weights_list, method="received"):
    """Compute per-token importance scores.

    method="received": importance = total attention received across all layers and positions
                       (how much do other tokens attend to this token?)
    """
    N = attn_weights_list[0].shape[0]
    importance = np.zeros(N)

    for w in attn_weights_list:
        # w: (N, N) — w[i, j] = how much token i attends to token j
        # Column sum = how much attention token j receives
        importance += w.sum(axis=0)

    return importance


def build_selective_kv(
    inner_model, residuals, important_mask, n_kv_heads, head_dim, d_model
):
    """Build KV cache using exact residuals for important tokens,
    h^(0)-projected for unimportant tokens.

    Returns list of (K, V) pairs for all layers.
    """
    h0_np = np.array(residuals[0][0].astype(mx.float32))  # (N, d)
    N = h0_np.shape[0]
    n_layers = len(inner_model.layers)

    kv_pairs = []
    for l in range(n_layers):
        # Layer l's KV uses h^(l) = residuals[l] (INPUT to the layer, not output)
        hl_exact = np.array(residuals[l][0].astype(mx.float32))  # (N, d)

        # Mix: exact for important, h^(0) for unimportant
        hl_mixed = np.where(important_mask[:, None], hl_exact, h0_np)
        hl_mx = mx.array(hl_mixed.astype(np.float32)).reshape(1, N, d_model)

        layer_obj = inner_model.layers[l]
        attn = layer_obj.self_attn
        norm = layer_obj.input_layernorm

        hl_normed = norm(hl_mx)
        k = attn.k_proj(hl_normed)
        v = attn.v_proj(hl_normed)

        k = k.reshape(1, N, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(1, N, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        mx.eval(k, v)

        kv_pairs.append((k, v))

    return kv_pairs


def generate_text(model, tokenizer, cache, question_text, max_tokens=50):
    """Generate text with a given cache."""
    q = mx.array(tokenizer.encode(question_text))
    tokens = []
    for tok, _ in generate_step(q, model, max_tokens=max_tokens, sampler=GREEDY, prompt_cache=cache):
        tokens.append(tok)
    return tokens, tokenizer.decode(tokens)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    args = parser.parse_args()

    print("Loading model...", file=sys.stderr)
    model, tokenizer = load(args.model)
    cfg = model.args
    n_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    n_kv_heads = cfg.num_key_value_heads
    n_q_heads = cfg.num_attention_heads
    head_dim = d_model // n_q_heads

    inner_model = _find_inner_model(model)

    # Warmup
    model(mx.array(tokenizer.encode("Hello")).reshape(1, -1))
    mx.eval(model.parameters())

    # Build tokens
    FILLER = ("The development of artificial intelligence has progressed rapidly in recent years. "
              "Machine learning models continue to grow in capability and efficiency. "
              "Large language models can now understand and generate human-like text. "
              "Researchers explore new architectures to push boundaries further. ") * 50
    tokens = mx.array(tokenizer.encode(FILLER)[:args.prompt_tokens])
    N = tokens.shape[0]
    print(f"Model: {n_layers}L, d={d_model} | Tokens: {N}", file=sys.stderr)

    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
    h0_per_token = d_model * 2

    # ════════════════════════════════════════════════════════
    # Step 1: Capture attention weights
    # ════════════════════════════════════════════════════════
    print("\nCapturing attention weights...", file=sys.stderr)
    t0 = time.perf_counter()
    residuals, attn_weights, _ = capture_attention_weights(inner_model, tokens)
    t_cap = (time.perf_counter() - t0) * 1000
    print(f"Captured in {t_cap:.0f}ms", file=sys.stderr)

    # ════════════════════════════════════════════════════════
    # Step 2: Attention distribution analysis
    # ════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print(f"  ATTENTION WEIGHT DISTRIBUTION")
    print(f"{'=' * 90}")

    importance = compute_token_importance(attn_weights)
    importance_sorted = np.sort(importance)[::-1]
    total_importance = importance.sum()

    print(f"\n  Total attention mass: {total_importance:.1f}")
    print(f"\n  Cumulative attention received by top-K% tokens:")

    for pct in [1, 2, 5, 10, 20, 30, 50]:
        k = max(1, int(N * pct / 100))
        top_k_mass = importance_sorted[:k].sum() / total_importance * 100
        print(f"    Top {pct:>2}% ({k:>4} tokens): {top_k_mass:>5.1f}% of attention")

    # Show top-20 tokens
    top_indices = np.argsort(importance)[::-1][:20]
    print(f"\n  Top 20 most-attended tokens:")
    token_ids = tokens.tolist()
    for i, idx in enumerate(top_indices):
        tok_text = tokenizer.decode([token_ids[idx]])
        print(f"    #{i+1:>2}: pos={idx:>4}, importance={importance[idx]:>8.1f}, "
              f"token='{tok_text}'")

    # ════════════════════════════════════════════════════════
    # Step 3: Exact KV baseline
    # ════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print(f"  GENERATION QUALITY: exact KV vs selective KV")
    print(f"{'=' * 90}")

    h0_store = H0Store()
    h0_store.append(residuals[0])
    kv_exact = reconstruct_prefix_kv(inner_model, h0_store, 0, N, chunk_size=512, eval_every=8)
    mx.eval(*[k for k, v in kv_exact] + [v for k, v in kv_exact])

    cache_exact = make_prompt_cache(model)
    for i, (k, v) in enumerate(kv_exact):
        cache_exact[i].state = (k, v)
    mx.eval([c.keys for c in cache_exact] + [c.values for c in cache_exact])

    question = "What is artificial intelligence?"
    gen_exact, text_exact = generate_text(model, tokenizer, cache_exact, question)
    print(f"\n  Exact KV ({kv_per_token * N / 1024 / 1024:.0f} MB): {text_exact[:200]}")

    # ════════════════════════════════════════════════════════
    # Step 4: Selective KV at various thresholds
    # ════════════════════════════════════════════════════════
    results = []
    for pct in [5, 10, 20, 30, 50, 70, 100]:
        k = max(1, int(N * pct / 100))
        top_indices_k = np.argsort(importance)[::-1][:k]
        important_mask = np.zeros(N, dtype=bool)
        important_mask[top_indices_k] = True

        # Build selective KV
        kv_selective = build_selective_kv(
            inner_model, residuals, important_mask, n_kv_heads, head_dim, d_model
        )

        cache_sel = make_prompt_cache(model)
        for i, (kk, vv) in enumerate(kv_selective):
            cache_sel[i].state = (kk, vv)
        mx.eval([c.keys for c in cache_sel] + [c.values for c in cache_sel])

        gen_sel, text_sel = generate_text(model, tokenizer, cache_sel, question)

        # Match analysis
        match_count = sum(1 for a, b in zip(gen_exact, gen_sel) if a == b)
        exact_20 = gen_exact[:20] == gen_sel[:20]

        # Transfer size: h^(0) for all + exact h^(l) for important tokens
        # h^(0): N × d_model × 2 bytes
        # Per important token: n_layers × d_model × 2 bytes (need h^(l) at each layer)
        # OR: just send exact KV for important tokens
        exact_kv_size = k * kv_per_token  # exact KV for important tokens
        h0_size = N * h0_per_token        # h^(0) for all (used for unimportant)
        total_transfer = exact_kv_size + h0_size
        # With int4 KV for important tokens:
        total_int4 = exact_kv_size // 4 + h0_size
        full_kv = N * kv_per_token
        compression = full_kv / total_transfer
        compression_int4 = full_kv / total_int4

        print(f"\n  Top {pct:>2}% exact ({k:>4} tokens) + h^(0) rest:")
        print(f"    Transfer: {total_transfer / 1024 / 1024:.1f} MB "
              f"(int4: {total_int4 / 1024 / 1024:.1f} MB) "
              f"│ {compression:.1f}× ({compression_int4:.1f}× int4) vs full KV")
        print(f"    Output: {text_sel[:200]}")
        print(f"    Match: {match_count}/{min(len(gen_exact), len(gen_sel))} "
              f"({'EXACT 20' if exact_20 else 'DIVERGED'})")

        results.append({
            'pct': pct, 'k': k, 'transfer_mb': total_transfer / 1024 / 1024,
            'int4_mb': total_int4 / 1024 / 1024,
            'compression': compression, 'match': match_count,
            'exact_20': exact_20, 'text': text_sel,
        })

    # ════════════════════════════════════════════════════════
    # Step 5: Summary table
    # ════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY: Selective KV Transfer")
    print(f"{'=' * 90}")

    full_kv_mb = N * kv_per_token / 1024 / 1024
    print(f"\n  Full KV: {full_kv_mb:.0f} MB | h^(0): {N * h0_per_token / 1024 / 1024:.1f} MB | N={N}\n")

    print(f"  {'%exact':>6} │ {'#tokens':>7} │ {'Transfer':>10} │ {'int4':>10} │ {'Compress':>8} │ {'Match':>6} │ {'First20':>7}")
    print(f"  {'─' * 6}─┼{'─' * 7}─┼{'─' * 10}─┼{'─' * 10}─┼{'─' * 8}─┼{'─' * 6}─┼{'─' * 7}")
    for r in results:
        print(f"  {r['pct']:>5}% │ {r['k']:>7} │ {r['transfer_mb']:>8.1f} MB│ {r['int4_mb']:>8.1f} MB│ "
              f"{r['compression']:>7.1f}× │ {r['match']:>4}/50 │ {'  YES' if r['exact_20'] else '   NO'}")


if __name__ == "__main__":
    main()
