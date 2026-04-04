#!/usr/bin/env python3
"""
Correct edge-cloud transfer compression experiment.

Key correction: only compress KV[0:cut] (cloud layers that need transfer).
KV[cut:L] stays exact (edge reconstructs from h^(cut)).

Architecture:
  Cloud: layers 0..cut-1 → computes KV[0:cut-1], sends compressed to edge
  Edge:  layers cut..L-1 → reconstructs KV[cut:L-1] from h^(cut) (exact)

Compression strategies (applied ONLY to layers 0..cut-1):
  A) int8 quantization (all tokens, approximate values)
  B) Token eviction (fewer tokens, exact values)
  C) Hybrid (important tokens exact + rest quantized)
  D) AM-style with h^(0) backup (evict + store h^(0) for reconstruction)

Usage:
    python3 experiments/dual_instance_h0/transfer_compress_correct.py \
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


def capture_residuals_and_kv(inner_model, tokens):
    """Forward pass → residuals + KV + token importance."""
    x = inner_model.embed_tokens(tokens.reshape(1, -1))
    mx.eval(x)
    residuals = [x]

    N = tokens.shape[0]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(x.dtype)
    n_layers = len(inner_model.layers)
    cache = [KVCache() for _ in range(n_layers)]

    for i, layer in enumerate(inner_model.layers):
        x = layer(x, mask=mask, cache=cache[i])
        mx.eval(x)
        residuals.append(x)

    # Token importance from K-norms (proxy for attention importance)
    token_importance = np.zeros(N)
    for i in range(n_layers):
        k, _ = cache[i].state
        mx.eval(k)
        k_norms = np.array(mx.sqrt(mx.sum(k[0] ** 2, axis=-1)).astype(mx.float32))
        token_importance += np.mean(k_norms, axis=0)
    token_importance /= n_layers

    return residuals, cache, token_importance


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
        return f"EXACT ✓ ({total} tok)", match, total, fd
    else:
        return f"{match:>3}/{total} (div@{fd})", match, total, fd


# ═══════════════════════════════════════════════════════════
# Build compressed caches — ONLY cloud layers compressed
# ═══════════════════════════════════════════════════════════

def build_baseline(exact_kv, n_layers):
    """Exact KV for all layers."""
    cache = [KVCache() for _ in range(n_layers)]
    for i in range(n_layers):
        k, v = exact_kv[i].state
        cache[i].state = (k, v)
    mx.eval([c.keys for c in cache] + [c.values for c in cache])
    return cache


def build_pipeline_exact(exact_kv, edge_caches, n_layers, cut_l):
    """Phase 1: cloud KV exact + edge KV reconstructed. (pipeline baseline)"""
    cache = make_prompt_cache_from_layers(n_layers)
    for i in range(n_layers):
        if i < cut_l:
            cache[i].state = exact_kv[i].state
        else:
            cache[i].state = edge_caches[i - cut_l].state
    mx.eval([c.keys for c in cache] + [c.values for c in cache])
    return cache


def make_prompt_cache_from_layers(n_layers):
    return [KVCache() for _ in range(n_layers)]


def build_quantized_cloud(exact_kv, edge_caches, n_layers, cut_l,
                           bits=8, gs=32):
    """Cloud layers quantized, edge layers exact."""
    cache = [KVCache() for _ in range(n_layers)]
    cloud_bytes = 0
    for i in range(n_layers):
        if i >= cut_l:
            cache[i].state = edge_caches[i - cut_l].state
        else:
            k, v = exact_kv[i].state
            mx.eval(k, v)
            qk = mx.quantize(k, group_size=gs, bits=bits)
            qv = mx.quantize(v, group_size=gs, bits=bits)
            kd = mx.dequantize(*qk, group_size=gs, bits=bits)
            vd = mx.dequantize(*qv, group_size=gs, bits=bits)
            mx.eval(kd, vd)
            cache[i].state = (kd, vd)
            cloud_bytes += sum(x.nbytes for x in qk) + sum(x.nbytes for x in qv)
    mx.eval([c.keys for c in cache] + [c.values for c in cache])
    return cache, cloud_bytes


def build_evicted_cloud(exact_kv, edge_caches, n_layers, cut_l,
                         selected_indices, N):
    """Cloud layers: only selected tokens (exact bf16). Edge layers: full exact."""
    cache = [KVCache() for _ in range(n_layers)]
    idx = mx.array(selected_indices)
    cloud_bytes = 0
    for i in range(n_layers):
        if i >= cut_l:
            cache[i].state = edge_caches[i - cut_l].state
        else:
            k, v = exact_kv[i].state
            mx.eval(k, v)
            k_sel = k[:, :, idx, :]
            v_sel = v[:, :, idx, :]
            mx.eval(k_sel, v_sel)
            cache[i].state = (k_sel, v_sel)
            cloud_bytes += k_sel.nbytes + v_sel.nbytes
    mx.eval([c.keys for c in cache] + [c.values for c in cache])
    return cache, cloud_bytes


def build_hybrid_cloud(exact_kv, edge_caches, n_layers, cut_l,
                        important_indices, N, bits=8, gs=32):
    """Cloud layers: important tokens exact + rest quantized. Edge: full exact."""
    cache = [KVCache() for _ in range(n_layers)]
    all_idx = np.arange(N)
    remaining_mask = np.ones(N, dtype=bool)
    remaining_mask[important_indices] = False
    remaining_idx = all_idx[remaining_mask]

    imp_mx = mx.array(np.sort(important_indices))
    rem_mx = mx.array(remaining_idx)
    cloud_bytes = 0

    for i in range(n_layers):
        if i >= cut_l:
            cache[i].state = edge_caches[i - cut_l].state
            continue

        k, v = exact_kv[i].state
        mx.eval(k, v)

        # Important: exact
        k_imp = k[:, :, imp_mx, :]
        v_imp = v[:, :, imp_mx, :]

        # Remaining: quantized
        k_rem = k[:, :, rem_mx, :]
        v_rem = v[:, :, rem_mx, :]
        qk = mx.quantize(k_rem, group_size=gs, bits=bits)
        qv = mx.quantize(v_rem, group_size=gs, bits=bits)
        k_rem_d = mx.dequantize(*qk, group_size=gs, bits=bits)
        v_rem_d = mx.dequantize(*qv, group_size=gs, bits=bits)
        mx.eval(k_imp, v_imp, k_rem_d, v_rem_d)

        # Reassemble
        B, H, _, D = k.shape
        k_np = np.zeros((H, N, D), dtype=np.float16)
        v_np = np.zeros((H, N, D), dtype=np.float16)

        imp_pos = np.sort(important_indices)
        k_np[:, imp_pos, :] = np.array(k_imp[0].astype(mx.float16))
        v_np[:, imp_pos, :] = np.array(v_imp[0].astype(mx.float16))
        k_np[:, remaining_idx, :] = np.array(k_rem_d[0].astype(mx.float16))
        v_np[:, remaining_idx, :] = np.array(v_rem_d[0].astype(mx.float16))

        k_out = mx.array(k_np[np.newaxis, ...]).astype(k.dtype)
        v_out = mx.array(v_np[np.newaxis, ...]).astype(v.dtype)
        mx.eval(k_out, v_out)
        cache[i].state = (k_out, v_out)

        cloud_bytes += k_imp.nbytes + v_imp.nbytes
        cloud_bytes += sum(x.nbytes for x in qk) + sum(x.nbytes for x in qv)

    mx.eval([c.keys for c in cache] + [c.values for c in cache])
    return cache, cloud_bytes


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

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
    n_kv_heads = cfg.num_key_value_heads
    head_dim = d_model // cfg.num_attention_heads

    print("=" * 80)
    print("Edge-Cloud Transfer Compression (Correct: only cloud layers compressed)")
    print("=" * 80)
    print(f"Model: {args.model.split('/')[-1]}, layers={n_layers}, d={d_model}")

    question = "\nQ: What is the main topic of this text?\nA:"

    prompts = {
        "repetitive": "The quick brown fox jumps over the lazy dog. " * 200,
        "diverse": DIVERSE_PROMPT * 5,
    }

    for prompt_name, prompt_text in prompts.items():
        tokens = tokenizer.encode(prompt_text)[:args.prompt_tokens]
        N = len(tokens)

        print(f"\n{'#' * 80}")
        print(f"# Prompt: {prompt_name} ({N} tokens)")
        print(f"{'#' * 80}")

        tok_mx = mx.array(tokens)
        print(f"  Capturing residuals + KV...")
        residuals, exact_kv, token_importance = capture_residuals_and_kv(inner, tok_mx)

        # Detect mode shift
        norms = []
        for l in range(len(residuals)):
            hl = np.array(residuals[l][0].astype(mx.float32))
            norms.append(np.linalg.norm(hl) / np.sqrt(N * d_model))
        mode_shift = 1
        for l in range(1, len(norms)):
            if norms[l] / norms[l-1] > 3.0:
                mode_shift = l
                break

        for cut_l in sorted(set([mode_shift, n_layers // 2])):
            print(f"\n  {'═' * 75}")
            print(f"  Cut@{cut_l}: Cloud layers 0..{cut_l-1}, Edge layers {cut_l}..{n_layers-1}")
            print(f"  {'═' * 75}")

            # Edge reconstruction from h^(cut)
            h_cut = residuals[cut_l]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(h_cut.dtype)
            edge_caches = [KVCache() for _ in range(n_layers - cut_l)]
            x = h_cut
            for i in range(n_layers - cut_l):
                x = inner.layers[cut_l + i](x, mask=mask, cache=edge_caches[i])
                mx.eval(x)

            # Exact cloud KV size
            exact_cloud_bytes = 0
            for i in range(cut_l):
                k, v = exact_kv[i].state
                exact_cloud_bytes += k.nbytes + v.nbytes

            # Baseline: exact everything
            c_base = build_pipeline_exact(exact_kv, edge_caches, n_layers, cut_l)
            base_tok, base_text = generate_with_cache(
                model, tokenizer, c_base, question, args.gen_tokens)
            print(f"  Baseline: {base_text[:90]}")
            print(f"  Cloud KV[0:{cut_l}] exact: {exact_cloud_bytes/1024/1024:.1f} MB")

            print(f"\n  {'Strategy':45s} │ {'CloudKV':>8s} │ {'Ratio':>5s} │ {'Quality':>20s}")
            print(f"  {'─' * 45}─┼{'─' * 8}─┼{'─' * 5}─┼{'─' * 20}")

            # ── A) Quantized cloud layers ──
            for bits, gs in [(8, 32), (8, 64), (4, 32)]:
                c, cb = build_quantized_cloud(exact_kv, edge_caches, n_layers, cut_l, bits, gs)
                gen, _ = generate_with_cache(model, tokenizer, c, question, args.gen_tokens)
                qs, m, t, fd = quality_str(base_tok, gen)
                label = f"A) int{bits} gs={gs} (all {N} tok)"
                print(f"  {label:45s} │ {cb/1024/1024:>7.1f}M │ {exact_cloud_bytes/cb:>4.1f}× │ {qs}")

            # ── B) Token eviction on cloud layers ──
            for frac_label, frac in [("50%", 0.5), ("75%", 0.75), ("90%", 0.9)]:
                budget = int(N * frac)
                # Recent tokens
                sel = np.arange(N - budget, N)
                c, cb = build_evicted_cloud(exact_kv, edge_caches, n_layers, cut_l, sel, N)
                gen, _ = generate_with_cache(model, tokenizer, c, question, args.gen_tokens)
                qs, m, t, fd = quality_str(base_tok, gen)
                label = f"B) Evict: keep recent {frac_label} ({budget} tok)"
                print(f"  {label:45s} │ {cb/1024/1024:>7.1f}M │ {exact_cloud_bytes/cb:>4.1f}× │ {qs}")

            # ── B2) Token eviction: K-norm top-k ──
            for frac_label, frac in [("50%", 0.5), ("75%", 0.75)]:
                budget = int(N * frac)
                sel = np.sort(np.argsort(token_importance)[-budget:])
                c, cb = build_evicted_cloud(exact_kv, edge_caches, n_layers, cut_l, sel, N)
                gen, _ = generate_with_cache(model, tokenizer, c, question, args.gen_tokens)
                qs, m, t, fd = quality_str(base_tok, gen)
                label = f"B2) Evict: K-norm top {frac_label} ({budget} tok)"
                print(f"  {label:45s} │ {cb/1024/1024:>7.1f}M │ {exact_cloud_bytes/cb:>4.1f}× │ {qs}")

            # ── C) Hybrid: important exact + rest int8 ──
            imp_50 = np.sort(np.argsort(token_importance)[-N//2:])
            c, cb = build_hybrid_cloud(exact_kv, edge_caches, n_layers, cut_l, imp_50, N, 8, 32)
            gen, _ = generate_with_cache(model, tokenizer, c, question, args.gen_tokens)
            qs, m, t, fd = quality_str(base_tok, gen)
            label = f"C) Hybrid: top-50% exact + rest int8"
            print(f"  {label:45s} │ {cb/1024/1024:>7.1f}M │ {exact_cloud_bytes/cb:>4.1f}× │ {qs}")

            # ── D) Eviction on cloud + exact on edge = different N per layer group ──
            # This is the key test: cloud sends 50% tokens, edge has 100%
            # The attention for cloud layers sees 50%, edge layers see 100%
            for frac_label, frac in [("50%", 0.5), ("75%", 0.75), ("90%", 0.9)]:
                budget = int(N * frac)
                # Sinks(4) + recent
                n_sinks = min(4, budget)
                sinks = np.arange(n_sinks)
                recent = np.arange(N - (budget - n_sinks), N)
                sel = np.sort(np.unique(np.concatenate([sinks, recent])))
                c, cb = build_evicted_cloud(exact_kv, edge_caches, n_layers, cut_l, sel, N)
                gen, _ = generate_with_cache(model, tokenizer, c, question, args.gen_tokens)
                qs, m, t, fd = quality_str(base_tok, gen)
                label = f"D) Sinks+Recent {frac_label} (cloud only)"
                print(f"  {label:45s} │ {cb/1024/1024:>7.1f}M │ {exact_cloud_bytes/cb:>4.1f}× │ {qs}")

    print()


if __name__ == "__main__":
    main()
