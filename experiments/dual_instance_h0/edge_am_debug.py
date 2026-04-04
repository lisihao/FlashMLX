#!/usr/bin/env python3
"""
Debug: why scored_pq pipeline quality degraded + why memory_stats returns 0.
"""

from __future__ import annotations
import sys, time, argparse, gc

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache
from mlx_lm.models.cache_factory import make_optimized_cache
from mlx_lm.models.kv_direct_cache import _find_inner_model
from mlx_lm.sample_utils import make_sampler

GREEDY = make_sampler(temp=0.0)


def generate_text(model, tokenizer, cache, question, max_tokens=50):
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


def dump_cache_state(cache, label, layer_range=None):
    """Print internals of a cache layer."""
    if layer_range is None:
        layer_range = range(len(cache))
    for i in layer_range:
        c = cache[i]
        ctype = type(c).__name__
        if ctype == 'KVCache':
            kn = c.keys.shape if c.keys is not None else None
            print(f"  L{i:>2} [{ctype}] keys={kn}, offset={c.offset}")
        else:
            # TripleLayerKVCache
            attrs = {}
            for attr in ['_flat_mode', '_flat_offset', '_true_offset', '_scored_active',
                         'scored_mode', 'offset', 'sequence_position']:
                if hasattr(c, attr):
                    attrs[attr] = getattr(c, attr)
            fk = c._flat_keys.shape if hasattr(c, '_flat_keys') and c._flat_keys is not None else None
            rk = c.recent_keys.shape if hasattr(c, 'recent_keys') and c.recent_keys is not None else None
            wk = c.warm_keys.shape if hasattr(c, 'warm_keys') and c.warm_keys is not None else None
            ms = c.memory_stats() if hasattr(c, 'memory_stats') else -1
            print(f"  L{i:>2} [{ctype}] flat_keys={fk}, recent={rk}, warm={wk}, "
                  f"mem_stats={ms}, attrs={attrs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    inner = _find_inner_model(model)
    n_layers = len(inner.layers)

    import os, glob
    cal_file = glob.glob(os.path.expanduser(
        "~/.cache/flashmlx/calibrations/qwen3_h*_l{}_*.pkl".format(n_layers)))[0]

    tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 200)[:args.prompt_tokens]
    N = len(tokens)
    tok_mx = mx.array(tokens)
    question = "\nQ: What is the main topic?\nA:"
    mask = None

    print(f"Model: {args.model.split('/')[-1]}, N={N}")

    # ── 1) Standard baseline ──
    print("\n=== Standard KVCache baseline ===")
    std_cache = [KVCache() for _ in range(n_layers)]
    x = inner.embed_tokens(tok_mx.reshape(1, -1))
    mx.eval(x)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(x.dtype)
    residuals = [x]
    for i, layer in enumerate(inner.layers):
        x = layer(x, mask=mask, cache=std_cache[i])
        mx.eval(x)
        residuals.append(x)

    base_tok, base_text = generate_text(model, tokenizer, std_cache, question, 50)
    print(f"  Output: {base_text[:80]}")
    dump_cache_state(std_cache, "standard", range(3))

    # ── 2) scored_pq full local ──
    print("\n=== scored_pq full local ===")
    spq_cache = make_optimized_cache(
        inner, strategy="scored_pq", calibration_file=cal_file, scored_max_cache=2048)
    x = inner.embed_tokens(tok_mx.reshape(1, -1))
    mx.eval(x)
    for i, layer in enumerate(inner.layers):
        x = layer(x, mask=mask, cache=spq_cache[i])
        mx.eval(x)

    print("  After prefill:")
    dump_cache_state(spq_cache, "scored_pq full", range(3))
    dump_cache_state(spq_cache, "scored_pq full", [n_layers-1])

    gen_spq, _ = generate_text(model, tokenizer, spq_cache, question, 50)
    qs = quality_str(base_tok, gen_spq)
    print(f"  Quality: {qs}")
    print("  After gen:")
    dump_cache_state(spq_cache, "scored_pq full after gen", range(2))

    # ── 3) Pipeline@18 + scored_pq ──
    cut_l = n_layers // 2
    print(f"\n=== Pipeline@{cut_l} + scored_pq ===")

    pipe_cache = make_optimized_cache(
        inner, strategy="scored_pq", calibration_file=cal_file, scored_max_cache=2048)

    # Inject cloud KV for layers 0..cut
    for i in range(cut_l):
        pipe_cache[i] = KVCache()
        pipe_cache[i].state = std_cache[i].state
    mx.eval([pipe_cache[i].keys for i in range(cut_l)])

    # Edge reconstruction
    h_cut = residuals[cut_l]
    for i in range(cut_l, n_layers):
        h_cut = inner.layers[i](h_cut, mask=mask, cache=pipe_cache[i])
        mx.eval(h_cut)

    print("  After recon:")
    dump_cache_state(pipe_cache, "pipeline", [0, 1, cut_l, cut_l+1, n_layers-1])

    gen_pipe, _ = generate_text(model, tokenizer, pipe_cache, question, 50)
    qs_pipe = quality_str(base_tok, gen_pipe)
    print(f"  Quality: {qs_pipe}")

    # ── 4) Pipeline@18 + standard KVCache (control) ──
    print(f"\n=== Pipeline@{cut_l} + standard KVCache (control) ===")
    ctrl_cache = [KVCache() for _ in range(n_layers)]
    for i in range(cut_l):
        ctrl_cache[i].state = std_cache[i].state
    mx.eval([ctrl_cache[i].keys for i in range(cut_l)])

    h_cut2 = residuals[cut_l]
    for i in range(cut_l, n_layers):
        h_cut2 = inner.layers[i](h_cut2, mask=mask, cache=ctrl_cache[i])
        mx.eval(h_cut2)

    gen_ctrl, _ = generate_text(model, tokenizer, ctrl_cache, question, 50)
    qs_ctrl = quality_str(base_tok, gen_ctrl)
    print(f"  Quality: {qs_ctrl}")

    # ── 5) scored_pq full local, then replace cloud with standard ──
    print(f"\n=== scored_pq full, then replace cloud layers with standard ===")
    spq2 = make_optimized_cache(
        inner, strategy="scored_pq", calibration_file=cal_file, scored_max_cache=2048)
    x = inner.embed_tokens(tok_mx.reshape(1, -1))
    mx.eval(x)
    for i, layer in enumerate(inner.layers):
        x = layer(x, mask=mask, cache=spq2[i])
        mx.eval(x)

    # NOW replace cloud layers with standard
    for i in range(cut_l):
        spq2[i] = KVCache()
        spq2[i].state = std_cache[i].state
    mx.eval([spq2[i].keys for i in range(cut_l)])

    gen_mix, _ = generate_text(model, tokenizer, spq2, question, 50)
    qs_mix = quality_str(base_tok, gen_mix)
    print(f"  Quality: {qs_mix}")
    print("  Cache state:")
    dump_cache_state(spq2, "mixed", [0, cut_l, n_layers-1])

    # ── 6) scored_pq full local, do NOT replace cloud ──
    print(f"\n=== scored_pq full, keep scored_pq for cloud too ===")
    spq3 = make_optimized_cache(
        inner, strategy="scored_pq", calibration_file=cal_file, scored_max_cache=2048)
    x = inner.embed_tokens(tok_mx.reshape(1, -1))
    mx.eval(x)
    for i, layer in enumerate(inner.layers):
        x = layer(x, mask=mask, cache=spq3[i])
        mx.eval(x)

    gen_pure, _ = generate_text(model, tokenizer, spq3, question, 50)
    qs_pure = quality_str(base_tok, gen_pure)
    print(f"  Quality: {qs_pure}")

    print("\n=== Summary ===")
    print(f"  Standard baseline:         {quality_str(base_tok, base_tok)}")
    print(f"  scored_pq full local:      {qs}")
    print(f"  scored_pq full (keep all): {qs_pure}")
    print(f"  scored_pq full → replace:  {qs_mix}")
    print(f"  Pipeline + scored_pq:      {qs_pipe}")
    print(f"  Pipeline + standard:       {qs_ctrl}")


if __name__ == "__main__":
    main()
