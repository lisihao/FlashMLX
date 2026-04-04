#!/usr/bin/env python3
"""
Edge-side AM compression experiment.

Idea: AM compression happens ON THE EDGE after KV reconstruction,
not for transfer. This reduces edge decode-time memory + speeds up attention.

Flow:
  1. Cloud sends h^(cut) + compressed KV[0:cut]
  2. Edge reconstructs KV[cut:L-1] from h^(cut) — full N tokens
  3. Edge applies AM eviction to ALL layers — keep top-K tokens
  4. Edge decodes with compressed KV — less memory, faster attention

Test matrix:
  - Cut@18 + int8 cloud KV + edge AM eviction at 50%, 25%
  - Cut@1 + exact cloud KV + edge AM eviction at 50%, 25%
  - Quality: token match vs exact baseline
  - Memory: KV bytes after eviction
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
    """Forward pass → residuals + exact KV."""
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

    return residuals, cache


def generate_with_cache(model, tokenizer, cache, question, max_tokens=50):
    q = mx.array(tokenizer.encode(question))
    result = []
    for tok, _ in generate_step(q, model, max_tokens=max_tokens,
                                 sampler=GREEDY, prompt_cache=cache):
        result.append(tok)
    return result, tokenizer.decode(result)


def quality_str(baseline, generated):
    match = sum(1 for a, b in zip(baseline, generated) if a == b)
    total = min(len(baseline), len(generated))
    exact = baseline == generated
    fd = next((i for i, (a, b) in enumerate(zip(baseline, generated)) if a != b),
              min(len(baseline), len(generated)))
    if exact:
        return f"EXACT ({total} tok)"
    else:
        return f"{match:>3}/{total} (div@{fd})"


def compute_k_norm_importance(cache, n_layers, N):
    """Global K-norm importance across all layers."""
    importance = np.zeros(N)
    for i in range(n_layers):
        k, _ = cache[i].state
        mx.eval(k)
        k_norms = np.array(mx.sqrt(mx.sum(k[0] ** 2, axis=-1)).astype(mx.float32))
        importance += np.mean(k_norms, axis=0)  # avg over heads
    importance /= n_layers
    return importance


def evict_cache(cache, selected_indices, n_layers):
    """Evict tokens from ALL layers, keeping only selected_indices."""
    idx = mx.array(selected_indices)
    evicted = [KVCache() for _ in range(n_layers)]
    for i in range(n_layers):
        k, v = cache[i].state
        mx.eval(k, v)
        k_sel = k[:, :, idx, :]
        v_sel = v[:, :, idx, :]
        mx.eval(k_sel, v_sel)
        evicted[i].state = (k_sel, v_sel)
    mx.eval([c.keys for c in evicted] + [c.values for c in evicted])
    return evicted


def build_pipeline_cache(exact_kv, edge_caches, n_layers, cut_l,
                          cloud_bits=None, cloud_gs=32):
    """Build full cache: cloud layers (optionally quantized) + edge layers (exact)."""
    cache = [KVCache() for _ in range(n_layers)]
    for i in range(n_layers):
        if i < cut_l:
            k, v = exact_kv[i].state
            mx.eval(k, v)
            if cloud_bits and cloud_bits < 16:
                qk = mx.quantize(k, group_size=cloud_gs, bits=cloud_bits)
                qv = mx.quantize(v, group_size=cloud_gs, bits=cloud_bits)
                k = mx.dequantize(*qk, group_size=cloud_gs, bits=cloud_bits)
                v = mx.dequantize(*qv, group_size=cloud_gs, bits=cloud_bits)
                mx.eval(k, v)
            cache[i].state = (k, v)
        else:
            cache[i].state = edge_caches[i - cut_l].state
    mx.eval([c.keys for c in cache] + [c.values for c in cache])
    return cache


def kv_bytes(cache):
    return sum(c.keys.nbytes + c.values.nbytes for c in cache)


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

    model_name = args.model.split('/')[-1]
    print("=" * 80)
    print(f"Edge-side AM Experiment: {model_name}")
    print(f"layers={n_layers}, d_model={d_model}")
    print("=" * 80)

    question = "\nQ: What is the main topic of this text?\nA:"

    prompts = {
        "repetitive": "The quick brown fox jumps over the lazy dog. " * 200,
        "diverse": DIVERSE_PROMPT * 5,
    }

    for prompt_name, prompt_text in prompts.items():
        tokens = tokenizer.encode(prompt_text)[:args.prompt_tokens]
        N = len(tokens)
        tok_mx = mx.array(tokens)

        print(f"\n{'#' * 80}")
        print(f"# {prompt_name} ({N} tokens)")
        print(f"{'#' * 80}")

        residuals, exact_kv = capture_residuals_and_kv(inner, tok_mx)

        # Baseline: exact full KV, generate
        baseline_cache = [KVCache() for _ in range(n_layers)]
        for i in range(n_layers):
            baseline_cache[i].state = exact_kv[i].state
        mx.eval([c.keys for c in baseline_cache] + [c.values for c in baseline_cache])
        base_tok, base_text = generate_with_cache(
            model, tokenizer, baseline_cache, question, args.gen_tokens)
        print(f"  Baseline: {base_text[:90]}")
        print(f"  Full KV:  {kv_bytes(exact_kv)/1024/1024:.1f} MB")

        for cut_l in [1, n_layers // 2]:
            print(f"\n  {'═' * 75}")
            print(f"  Cut@{cut_l}: Cloud 0..{cut_l-1}, Edge {cut_l}..{n_layers-1}")
            print(f"  {'═' * 75}")

            # Edge reconstruction
            h_cut = residuals[cut_l]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(h_cut.dtype)
            edge_caches = [KVCache() for _ in range(n_layers - cut_l)]
            x = h_cut
            for i in range(n_layers - cut_l):
                x = inner.layers[cut_l + i](x, mask=mask, cache=edge_caches[i])
                mx.eval(x)

            # Cloud KV transfer options
            cloud_configs = [
                ("exact bf16", None, 32),
                ("int8 gs=32", 8, 32),
            ]

            for cloud_label, cloud_bits, cloud_gs in cloud_configs:
                # Build fresh cache for baseline test
                c0 = build_pipeline_cache(
                    exact_kv, edge_caches, n_layers, cut_l, cloud_bits, cloud_gs)

                # Compute importance BEFORE generation (N tokens only)
                importance = compute_k_norm_importance(c0, n_layers, N)
                full_kv_mb = kv_bytes(c0) / 1024 / 1024

                # Pipeline baseline (no AM on edge) — rebuild to avoid mutation
                gen, _ = generate_with_cache(
                    model, tokenizer, c0, question, args.gen_tokens)
                qs0 = quality_str(base_tok, gen)

                print(f"\n    Cloud: {cloud_label}")
                print(f"    Pipeline baseline (no AM): {qs0}, KV={full_kv_mb:.1f} MB")

                # AM eviction at various rates
                print(f"    {'Edge AM strategy':40s} │ {'KV MB':>7s} │ {'Saving':>6s} │ {'Quality':>22s}")
                print(f"    {'─' * 40}─┼{'─' * 7}─┼{'─' * 6}─┼{'─' * 22}")

                for keep_label, keep_frac in [("75%", 0.75), ("50%", 0.5), ("25%", 0.25)]:
                    budget = int(N * keep_frac)

                    # K-norm top-K — rebuild cache each time
                    fresh = build_pipeline_cache(
                        exact_kv, edge_caches, n_layers, cut_l, cloud_bits, cloud_gs)
                    sel_knorm = np.sort(np.argsort(importance)[-budget:])
                    cache_k = evict_cache(fresh, sel_knorm, n_layers)
                    gen_k, _ = generate_with_cache(
                        model, tokenizer, cache_k, question, args.gen_tokens)
                    qs_k = quality_str(base_tok, gen_k)
                    kv_k = kv_bytes(cache_k) / 1024 / 1024
                    saving_k = (1 - kv_k / full_kv_mb) * 100

                    label = f"K-norm top {keep_label} ({budget} tok)"
                    print(f"    {label:40s} │ {kv_k:>6.1f}M │ {saving_k:>5.0f}% │ {qs_k}")

                    # Sinks + recent
                    fresh = build_pipeline_cache(
                        exact_kv, edge_caches, n_layers, cut_l, cloud_bits, cloud_gs)
                    n_sinks = min(4, budget)
                    sinks = np.arange(n_sinks)
                    recent = np.arange(N - (budget - n_sinks), N)
                    sel_sr = np.sort(np.unique(np.concatenate([sinks, recent])))
                    cache_sr = evict_cache(fresh, sel_sr, n_layers)
                    gen_sr, _ = generate_with_cache(
                        model, tokenizer, cache_sr, question, args.gen_tokens)
                    qs_sr = quality_str(base_tok, gen_sr)
                    kv_sr = kv_bytes(cache_sr) / 1024 / 1024
                    saving_sr = (1 - kv_sr / full_kv_mb) * 100

                    label = f"Sinks+Recent {keep_label} ({len(sel_sr)} tok)"
                    print(f"    {label:40s} │ {kv_sr:>6.1f}M │ {saving_sr:>5.0f}% │ {qs_sr}")

                    # K-norm + sinks
                    fresh = build_pipeline_cache(
                        exact_kv, edge_caches, n_layers, cut_l, cloud_bits, cloud_gs)
                    sel_ks = np.sort(np.unique(np.concatenate([
                        sinks, np.argsort(importance)[-(budget - n_sinks):]
                    ])))
                    cache_ks = evict_cache(fresh, sel_ks, n_layers)
                    gen_ks, _ = generate_with_cache(
                        model, tokenizer, cache_ks, question, args.gen_tokens)
                    qs_ks = quality_str(base_tok, gen_ks)
                    kv_ks = kv_bytes(cache_ks) / 1024 / 1024
                    saving_ks = (1 - kv_ks / full_kv_mb) * 100

                    label = f"Sinks+K-norm {keep_label} ({len(sel_ks)} tok)"
                    print(f"    {label:40s} │ {kv_ks:>6.1f}M │ {saving_ks:>5.0f}% │ {qs_ks}")

    print()


if __name__ == "__main__":
    main()
