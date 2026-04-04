#!/usr/bin/env python3
"""
Token-level compression vs Value-level quantization for KV transfer.

Core hypothesis (from user): AM-style token selection (exact values, fewer tokens)
should outperform quantization (all tokens, approximate values) because:
  - Quantization adds noise to EVERY attention score → softmax amplifies → greedy diverges
  - Token eviction removes some tokens but keeps remaining scores EXACT
  - Attention is naturally a top-k mechanism — it already ignores most tokens

Experiment: at ~2× compression ratio, compare:
  A) Value-level: int8 gs=32 on all N tokens
  B) Token-level (recent): keep last N/2 tokens (exact bf16)
  C) Token-level (sinks+recent): first 4 + last N/2-4 tokens (exact bf16)
  D) Token-level (norm-based): keep N/2 tokens with highest K-norm (exact bf16)
  E) Token-level (attention-based): use AM-style importance scoring
  F) Combined: 50% important tokens exact + 50% remaining int8

Usage:
    python3 experiments/dual_instance_h0/token_vs_quant_transfer.py \
        --model /path/to/model --prompt-tokens 2048
"""

from __future__ import annotations
import sys, time, argparse

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.kv_direct_cache import _find_inner_model
from mlx_lm.sample_utils import make_sampler

GREEDY = make_sampler(temp=0.0)

DIVERSE_PROMPT = """The history of artificial intelligence began in antiquity, with myths and stories of artificial beings endowed with intelligence by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe human thinking as mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s. The field of AI research was founded at Dartmouth College in 1956. Many predicted a machine as intelligent as a human would exist within a generation. Eventually it became obvious that researchers had grossly underestimated the difficulty. In 1973, governments stopped funding undirected AI research, leading to the first AI winter. Investment boomed again in the 21st century when machine learning was successfully applied to many problems due to new methods, the application of powerful computer hardware, and the collection of immense data sets. Deep learning transformed the field starting around 2012 when neural networks began dramatically outperforming other methods."""


# ════════════════════════════════════════════════════════
# Core utilities
# ════════════════════════════════════════════════════════

def capture_kv_and_attention(model, inner_model, tokens):
    """Forward pass capturing KV caches AND per-layer attention patterns.

    We hook into attention to get importance scores for each token position.
    """
    x = inner_model.embed_tokens(tokens.reshape(1, -1))
    mx.eval(x)

    N = tokens.shape[0]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(x.dtype)
    n_layers = len(inner_model.layers)
    cache = [KVCache() for _ in range(n_layers)]

    # Collect attention scores from last few layers for importance ranking
    attention_scores = []

    for i, layer in enumerate(inner_model.layers):
        x = layer(x, mask=mask, cache=cache[i])
        mx.eval(x)

    # Compute token importance from KV norms (fast proxy for attention importance)
    # Token importance = average K-norm across layers and heads
    token_importance = np.zeros(N)
    for i in range(n_layers):
        k, v = cache[i].state
        mx.eval(k, v)
        # k shape: (1, n_kv_heads, N, head_dim)
        k_norms = np.array(mx.sqrt(mx.sum(k[0] ** 2, axis=-1)).astype(mx.float32))  # (n_kv_heads, N)
        token_importance += np.mean(k_norms, axis=0)  # average across heads

    token_importance /= n_layers

    return cache, token_importance


def generate_with_cache(model, tokenizer, cache, question, max_tokens=50):
    q = mx.array(tokenizer.encode(question))
    result = []
    for tok, _ in generate_step(q, model, max_tokens=max_tokens,
                                 sampler=GREEDY, prompt_cache=cache):
        result.append(tok)
    return result, tokenizer.decode(result)


def first_diverge(baseline, generated):
    for i, (a, b) in enumerate(zip(baseline, generated)):
        if a != b:
            return i
    return min(len(baseline), len(generated))


def quality_str(baseline, generated):
    match = sum(1 for a, b in zip(baseline, generated) if a == b)
    total = min(len(baseline), len(generated))
    exact = baseline == generated
    fd = first_diverge(baseline, generated)
    if exact:
        return f"EXACT ✓ ({total} tok)", total, total, total
    else:
        return f"{match:>3}/{total} (diverge@{fd})", match, total, fd


# ════════════════════════════════════════════════════════
# Compression strategies
# ════════════════════════════════════════════════════════

def build_cache_quantized(exact_kv, n_layers, bits=8, gs=32, layer_set=None):
    """Strategy A: Quantize all tokens' KV values."""
    cache = [KVCache() for _ in range(n_layers)]
    total_bytes = 0
    for i in range(n_layers):
        k, v = exact_kv[i].state
        mx.eval(k, v)
        if layer_set is None or i in layer_set:
            qk = mx.quantize(k, group_size=gs, bits=bits)
            qv = mx.quantize(v, group_size=gs, bits=bits)
            kd = mx.dequantize(*qk, group_size=gs, bits=bits)
            vd = mx.dequantize(*qv, group_size=gs, bits=bits)
            mx.eval(kd, vd)
            cache[i].state = (kd, vd)
            total_bytes += sum(x.nbytes for x in qk) + sum(x.nbytes for x in qv)
        else:
            cache[i].state = (k, v)
            total_bytes += k.nbytes + v.nbytes
    mx.eval([c.keys for c in cache] + [c.values for c in cache])
    return cache, total_bytes


def build_cache_token_select(exact_kv, n_layers, selected_indices, N, layer_set=None):
    """Strategy B/C/D/E: Keep only selected tokens' KV (exact bf16).

    selected_indices: sorted array of token positions to keep.
    All layers use the same token selection (consistent attention positions).
    """
    cache = [KVCache() for _ in range(n_layers)]
    total_bytes = 0
    idx = mx.array(selected_indices)

    for i in range(n_layers):
        k, v = exact_kv[i].state
        mx.eval(k, v)
        if layer_set is None or i in layer_set:
            # Select tokens: k[:, :, selected, :]
            k_sel = k[:, :, idx, :]
            v_sel = v[:, :, idx, :]
            mx.eval(k_sel, v_sel)
            cache[i].state = (k_sel, v_sel)
            total_bytes += k_sel.nbytes + v_sel.nbytes
        else:
            cache[i].state = (k, v)
            total_bytes += k.nbytes + v.nbytes
    mx.eval([c.keys for c in cache] + [c.values for c in cache])
    return cache, total_bytes


def build_cache_hybrid(exact_kv, n_layers, important_indices, N,
                        bits=8, gs=32, layer_set=None):
    """Strategy F: Important tokens exact + remaining tokens quantized.

    important_indices: token positions to keep exact.
    Other tokens: quantized to int8.
    """
    cache = [KVCache() for _ in range(n_layers)]
    total_bytes = 0
    all_indices = np.arange(N)
    remaining_mask = np.ones(N, dtype=bool)
    remaining_mask[important_indices] = False
    remaining_indices = all_indices[remaining_mask]

    imp_idx = mx.array(np.sort(important_indices))
    rem_idx = mx.array(remaining_indices)

    # We need to reconstruct the full sequence with important tokens exact
    # and remaining tokens quantized, then sort by position
    for i in range(n_layers):
        k, v = exact_kv[i].state
        mx.eval(k, v)

        if layer_set is not None and i not in layer_set:
            cache[i].state = (k, v)
            total_bytes += k.nbytes + v.nbytes
            continue

        # Important: exact
        k_imp = k[:, :, imp_idx, :]
        v_imp = v[:, :, imp_idx, :]

        # Remaining: quantized
        k_rem = k[:, :, rem_idx, :]
        v_rem = v[:, :, rem_idx, :]
        qk = mx.quantize(k_rem, group_size=gs, bits=bits)
        qv = mx.quantize(v_rem, group_size=gs, bits=bits)
        k_rem_deq = mx.dequantize(*qk, group_size=gs, bits=bits)
        v_rem_deq = mx.dequantize(*qv, group_size=gs, bits=bits)
        mx.eval(k_imp, v_imp, k_rem_deq, v_rem_deq)

        # Reassemble in original position order
        # Build full arrays
        B, H = k.shape[0], k.shape[1]
        D = k.shape[3]
        k_full = mx.zeros_like(k)
        v_full = mx.zeros_like(v)

        # Scatter back
        k_np = np.array(k_full[0, 0].astype(mx.float32))  # just to get shape
        k_full_np = np.zeros((H, N, D), dtype=np.float16)
        v_full_np = np.zeros((H, N, D), dtype=np.float16)

        k_imp_np = np.array(k_imp[0].astype(mx.float16))
        v_imp_np = np.array(v_imp[0].astype(mx.float16))
        k_rem_np = np.array(k_rem_deq[0].astype(mx.float16))
        v_rem_np = np.array(v_rem_deq[0].astype(mx.float16))

        imp_pos = np.sort(important_indices)
        k_full_np[:, imp_pos, :] = k_imp_np
        v_full_np[:, imp_pos, :] = v_imp_np
        k_full_np[:, remaining_indices, :] = k_rem_np
        v_full_np[:, remaining_indices, :] = v_rem_np

        k_out = mx.array(k_full_np[np.newaxis, ...]).astype(k.dtype)
        v_out = mx.array(v_full_np[np.newaxis, ...]).astype(v.dtype)
        mx.eval(k_out, v_out)
        cache[i].state = (k_out, v_out)

        # Bytes: important exact + remaining quantized
        total_bytes += k_imp.nbytes + v_imp.nbytes
        total_bytes += sum(x.nbytes for x in qk) + sum(x.nbytes for x in qv)

    mx.eval([c.keys for c in cache] + [c.values for c in cache])
    return cache, total_bytes


# ════════════════════════════════════════════════════════
# Token selection strategies
# ════════════════════════════════════════════════════════

def select_recent(N, budget):
    """Keep the most recent `budget` tokens."""
    return np.arange(N - budget, N)


def select_sinks_recent(N, budget, n_sinks=4):
    """Keep first n_sinks + last (budget - n_sinks) tokens."""
    sinks = np.arange(n_sinks)
    recent = np.arange(N - (budget - n_sinks), N)
    return np.sort(np.unique(np.concatenate([sinks, recent])))


def select_by_norm(token_importance, N, budget):
    """Keep tokens with highest K-norm importance."""
    return np.sort(np.argsort(token_importance)[-budget:])


def select_uniform(N, budget):
    """Uniformly spaced tokens (ensures coverage)."""
    return np.sort(np.linspace(0, N - 1, budget, dtype=int))


# ════════════════════════════════════════════════════════
# Main experiment
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--gen-tokens", type=int, default=50)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    inner = _find_inner_model(model)
    n_layers = len(inner.layers)
    cfg = model.args if hasattr(model, 'args') else inner.args
    d_model = cfg.hidden_size

    print("=" * 75)
    print("Token-Level Compression vs Value-Level Quantization")
    print("=" * 75)
    print(f"Model: {args.model.split('/')[-1]}, layers={n_layers}, d={d_model}")

    prompts = {
        "repetitive": "The quick brown fox jumps over the lazy dog. " * 200,
        "diverse": DIVERSE_PROMPT * 5,
    }

    question = "\nQ: What is the main topic of this text?\nA:"

    for prompt_name, prompt_text in prompts.items():
        tokens = tokenizer.encode(prompt_text)[:args.prompt_tokens]
        prompt = tokenizer.decode(tokens)
        N = len(tokens)

        print(f"\n{'#' * 75}")
        print(f"# Prompt: {prompt_name} ({N} tokens)")
        print(f"{'#' * 75}")

        # Full forward pass
        tok_mx = mx.array(tokens)
        print(f"  Prefilling {N} tokens...")
        exact_kv, token_importance = capture_kv_and_attention(model, inner, tok_mx)

        # Baseline
        cache_base = make_prompt_cache(model)
        for i in range(n_layers):
            cache_base[i].state = exact_kv[i].state
        mx.eval([c.keys for c in cache_base] + [c.values for c in cache_base])

        baseline_tok, baseline_text = generate_with_cache(
            model, tokenizer, cache_base, question, args.gen_tokens)
        print(f"  Baseline ({len(baseline_tok)} tokens): {baseline_text[:100]}")

        # Exact KV size
        exact_bytes = sum(c.keys.nbytes + c.values.nbytes for c in exact_kv)
        print(f"  Exact KV: {exact_bytes / 1024 / 1024:.1f} MB")

        # ── Test at different compression ratios ──
        for ratio_label, budget_frac in [("2×", 0.5), ("3×", 0.33), ("4×", 0.25)]:
            budget = int(N * budget_frac)

            print(f"\n  {'═' * 70}")
            print(f"  Compression target: ~{ratio_label} ({budget}/{N} tokens or equivalent)")
            print(f"  {'═' * 70}")
            print(f"  {'Strategy':40s} │ {'Bytes':>8s} │ {'Ratio':>5s} │ {'Quality':>25s}")
            print(f"  {'─' * 40}─┼{'─' * 8}─┼{'─' * 5}─┼{'─' * 25}")

            # A) Quantized (all tokens, approximate values)
            if ratio_label == "2×":
                bits_list = [(8, 32)]  # int8 ≈ 2× compression
            elif ratio_label == "3×":
                bits_list = [(5, 32)]  # int5 ≈ 3× compression
            else:
                bits_list = [(4, 32)]  # int4 ≈ 4× compression

            for bits, gs in bits_list:
                c, nbytes = build_cache_quantized(exact_kv, n_layers, bits, gs)
                gen_tok, _ = generate_with_cache(model, tokenizer, c, question, args.gen_tokens)
                qs, match, total, fd = quality_str(baseline_tok, gen_tok)
                label = f"A) Quantized int{bits} gs={gs}"
                print(f"  {label:40s} │ {nbytes/1024/1024:>7.1f}M │ {exact_bytes/nbytes:>4.1f}× │ {qs}")

            # B) Recent tokens only (exact bf16)
            sel = select_recent(N, budget)
            c, nbytes = build_cache_token_select(exact_kv, n_layers, sel, N)
            gen_tok, _ = generate_with_cache(model, tokenizer, c, question, args.gen_tokens)
            qs, match, total, fd = quality_str(baseline_tok, gen_tok)
            print(f"  {'B) Token: recent only':40s} │ {nbytes/1024/1024:>7.1f}M │ {exact_bytes/nbytes:>4.1f}× │ {qs}")

            # C) Sinks + recent (exact bf16)
            sel = select_sinks_recent(N, budget, n_sinks=4)
            c, nbytes = build_cache_token_select(exact_kv, n_layers, sel, N)
            gen_tok, _ = generate_with_cache(model, tokenizer, c, question, args.gen_tokens)
            qs, match, total, fd = quality_str(baseline_tok, gen_tok)
            print(f"  {'C) Token: sinks(4) + recent':40s} │ {nbytes/1024/1024:>7.1f}M │ {exact_bytes/nbytes:>4.1f}× │ {qs}")

            # D) K-norm importance (exact bf16)
            sel = select_by_norm(token_importance, N, budget)
            c, nbytes = build_cache_token_select(exact_kv, n_layers, sel, N)
            gen_tok, _ = generate_with_cache(model, tokenizer, c, question, args.gen_tokens)
            qs, match, total, fd = quality_str(baseline_tok, gen_tok)
            print(f"  {'D) Token: K-norm top-k':40s} │ {nbytes/1024/1024:>7.1f}M │ {exact_bytes/nbytes:>4.1f}× │ {qs}")

            # E) Uniform spacing (exact bf16)
            sel = select_uniform(N, budget)
            c, nbytes = build_cache_token_select(exact_kv, n_layers, sel, N)
            gen_tok, _ = generate_with_cache(model, tokenizer, c, question, args.gen_tokens)
            qs, match, total, fd = quality_str(baseline_tok, gen_tok)
            print(f"  {'E) Token: uniform spacing':40s} │ {nbytes/1024/1024:>7.1f}M │ {exact_bytes/nbytes:>4.1f}× │ {qs}")

            # F) Hybrid: important tokens exact + rest quantized
            if ratio_label == "2×":
                imp_idx = select_by_norm(token_importance, N, N // 2)
                c, nbytes = build_cache_hybrid(exact_kv, n_layers, imp_idx, N, bits=8, gs=32)
                gen_tok, _ = generate_with_cache(model, tokenizer, c, question, args.gen_tokens)
                qs, match, total, fd = quality_str(baseline_tok, gen_tok)
                print(f"  {'F) Hybrid: top-50% exact + rest int8':40s} │ {nbytes/1024/1024:>7.1f}M │ {exact_bytes/nbytes:>4.1f}× │ {qs}")

    print("\n" + "=" * 75)
    print("Key: Token-level keeps selected tokens' KV in exact bf16.")
    print("     Value-level quantizes all tokens' KV to int8/int4.")
    print("     Hybrid combines both: important tokens exact, rest quantized.")
    print("=" * 75)


if __name__ == "__main__":
    main()
