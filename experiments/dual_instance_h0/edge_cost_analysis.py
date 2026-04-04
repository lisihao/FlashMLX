#!/usr/bin/env python3
"""
Edge-side compute & memory cost analysis for two transfer strategies:

  Strategy 1: Cut@1 + AM evict 50% (K-norm)
    - Cloud sends: h^(1) + 50% of layer 0 KV (exact bf16)
    - Edge: reconstructs KV[1..L-1] from h^(1) = full prefill of 35 layers
    - Edge memory: full model weights + KV[0] (50% tokens) + KV[1..L-1] (full)

  Strategy 2: Cut@18 + int8 gs=32
    - Cloud sends: h^(18) + int8 KV[0..17] (all tokens)
    - Edge: reconstructs KV[18..L-1] from h^(18) = prefill of 18 layers
    - Edge memory: full model weights + KV[0..17] (int8→bf16) + KV[18..L-1] (full)

Measures:
  - Edge reconstruction time (real measurement)
  - Edge KV memory footprint
  - Edge model memory (weights)
  - Transfer size
  - Decode speed impact (pipeline RTT vs local)
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
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.kv_direct_cache import _find_inner_model
from mlx_lm.sample_utils import make_sampler
from mlx.utils import tree_flatten

GREEDY = make_sampler(temp=0.0)


def mem_mb():
    """Current MLX memory usage in MB."""
    return mx.metal.get_active_memory() / 1024 / 1024


def peak_mb():
    """Peak MLX memory usage in MB."""
    return mx.metal.get_peak_memory() / 1024 / 1024


def reset_peak():
    mx.metal.reset_peak_memory()


def measure_reconstruction(inner_model, h_cut, cut_l, n_layers, N):
    """Measure edge-side KV reconstruction from h^(cut).

    Returns: edge_caches, recon_time_ms, recon_mem_delta_mb, peak_mem_mb
    """
    gc.collect()
    mx.eval(h_cut)

    mem_before = mem_mb()
    reset_peak()

    mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(h_cut.dtype)

    t0 = time.perf_counter()
    edge_caches = [KVCache() for _ in range(n_layers - cut_l)]
    x = h_cut
    for i in range(n_layers - cut_l):
        x = inner_model.layers[cut_l + i](x, mask=mask, cache=edge_caches[i])
        mx.eval(x)
    recon_ms = (time.perf_counter() - t0) * 1000

    mem_after = mem_mb()
    peak = peak_mb()

    return edge_caches, recon_ms, mem_after - mem_before, peak


def measure_decode_speed(model, tokenizer, cache, question, n_tokens=20):
    """Measure per-token decode latency."""
    q = mx.array(tokenizer.encode(question))
    times = []
    count = 0
    for tok, _ in generate_step(q, model, max_tokens=n_tokens,
                                 sampler=GREEDY, prompt_cache=cache):
        t0 = time.perf_counter()
        mx.eval(mx.array([tok]))  # force sync
        times.append((time.perf_counter() - t0) * 1000)
        count += 1
    # First token includes question processing, skip it
    if len(times) > 2:
        avg_ms = np.mean(times[2:])
    else:
        avg_ms = np.mean(times)
    return avg_ms, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    inner = _find_inner_model(model)
    n_layers = len(inner.layers)
    cfg = model.args if hasattr(model, 'args') else inner.args
    d_model = cfg.hidden_size
    n_kv_heads = cfg.num_key_value_heads
    head_dim = d_model // cfg.num_attention_heads
    kv_per_token_per_layer = 2 * n_kv_heads * head_dim * 2  # K+V, bf16 bytes

    model_name = args.model.split('/')[-1]
    print("=" * 80)
    print(f"Edge-Side Cost Analysis: {model_name}")
    print("=" * 80)
    print(f"Layers={n_layers}, d_model={d_model}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    print(f"KV per token per layer: {kv_per_token_per_layer} bytes")

    # Model weights size
    model_bytes = sum(p.nbytes for _, p in tree_flatten(model.parameters()))
    print(f"Model weights: {model_bytes / 1024 / 1024:.0f} MB")

    # Build prompt
    base = "The quick brown fox jumps over the lazy dog. " * 200
    tokens = tokenizer.encode(base)[:args.prompt_tokens]
    N = len(tokens)
    print(f"Prompt tokens: {N}")

    question = "\nQ: What is the main topic?\nA:"

    # ── Full prefill baseline ──
    print(f"\n{'═' * 80}")
    print("Baseline: Full Prefill (all layers on edge)")
    print(f"{'═' * 80}")

    tok_mx = mx.array(tokens)
    x = inner.embed_tokens(tok_mx.reshape(1, -1))
    mx.eval(x)
    residuals = [x]

    reset_peak()
    mem_before_prefill = mem_mb()
    t0 = time.perf_counter()
    mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(x.dtype)
    full_cache = [KVCache() for _ in range(n_layers)]
    xp = x
    for i, layer in enumerate(inner.layers):
        xp = layer(xp, mask=mask, cache=full_cache[i])
        mx.eval(xp)
        residuals.append(xp)
    full_prefill_ms = (time.perf_counter() - t0) * 1000
    full_prefill_peak = peak_mb()
    mem_after_prefill = mem_mb()

    full_kv_bytes = sum(c.keys.nbytes + c.values.nbytes for c in full_cache)

    print(f"  Prefill time:     {full_prefill_ms:.0f} ms ({n_layers} layers)")
    print(f"  KV memory:        {full_kv_bytes / 1024 / 1024:.1f} MB")
    print(f"  Memory delta:     {mem_after_prefill - mem_before_prefill:.1f} MB")
    print(f"  Peak memory:      {full_prefill_peak:.1f} MB")

    # Decode speed baseline
    cache_base = make_prompt_cache(model)
    for i in range(n_layers):
        cache_base[i].state = full_cache[i].state
    mx.eval([c.keys for c in cache_base] + [c.values for c in cache_base])
    tg_ms, _ = measure_decode_speed(model, tokenizer, cache_base, question, 20)
    print(f"  Decode speed:     {tg_ms:.1f} ms/token")

    # ══════════════════════════════════════════════════════════════════
    # Strategy 1: Cut@1 + AM evict 50%
    # ══════════════════════════════════════════════════════════════════
    cut_1 = 1
    print(f"\n{'═' * 80}")
    print(f"Strategy 1: Cut@{cut_1} + AM evict 50% (K-norm)")
    print(f"  Cloud: layer 0 → sends 50% tokens KV (exact bf16) + h^(1)")
    print(f"  Edge:  layers 1..{n_layers-1} → reconstructs from h^(1)")
    print(f"{'═' * 80}")

    h_1 = residuals[cut_1]

    # Edge reconstruction
    edge_caches_1, recon_ms_1, recon_mem_1, recon_peak_1 = measure_reconstruction(
        inner, h_1, cut_1, n_layers, N)

    # Cloud KV[0] — 50% tokens (K-norm selected)
    k0, v0 = full_cache[0].state
    mx.eval(k0, v0)
    k0_np_norms = np.array(mx.sqrt(mx.sum(k0[0] ** 2, axis=-1)).astype(mx.float32))
    tok_importance = np.mean(k0_np_norms, axis=0)
    budget = N // 2
    selected = np.sort(np.argsort(tok_importance)[-budget:])
    idx = mx.array(selected)
    k0_sel = k0[:, :, idx, :]
    v0_sel = v0[:, :, idx, :]
    mx.eval(k0_sel, v0_sel)
    cloud_kv_bytes_1 = k0_sel.nbytes + v0_sel.nbytes

    # h^(1) transfer size
    h1_bytes = N * d_model * 2

    # Edge KV memory: layer 0 (50% tokens) + layers 1..L-1 (full)
    edge_kv_layer0 = k0_sel.nbytes + v0_sel.nbytes
    edge_kv_rest = sum(c.keys.nbytes + c.values.nbytes for c in edge_caches_1)
    edge_kv_total_1 = edge_kv_layer0 + edge_kv_rest

    # Half-model weights (theoretical: edge only loads layers 1..L-1 + embed + lm_head)
    layers_1_to_L = sum(
        p.nbytes for name, p in tree_flatten(model.parameters())
        if 'layers.' in name and int(name.split('.')[2]) >= cut_1
    )
    embed_lmhead = model_bytes - sum(
        p.nbytes for name, p in tree_flatten(model.parameters())
        if 'layers.' in name
    )
    edge_model_bytes_1 = layers_1_to_L + embed_lmhead

    print(f"\n  Transfer:")
    print(f"    h^(1):              {h1_bytes / 1024 / 1024:>8.1f} MB")
    print(f"    KV[0] (50% tok):    {cloud_kv_bytes_1 / 1024 / 1024:>8.1f} MB")
    print(f"    Total transfer:     {(h1_bytes + cloud_kv_bytes_1) / 1024 / 1024:>8.1f} MB")
    print(f"    vs Full KV:         {full_kv_bytes / 1024 / 1024:>8.1f} MB "
          f"({full_kv_bytes / (h1_bytes + cloud_kv_bytes_1):.0f}× larger)")

    print(f"\n  Edge Compute:")
    print(f"    Reconstruction:     {recon_ms_1:>8.0f} ms ({n_layers - cut_1} layers)")
    print(f"    vs Full prefill:    {full_prefill_ms:>8.0f} ms ({n_layers} layers)")
    print(f"    Compute saving:     {(1 - recon_ms_1 / full_prefill_ms) * 100:>7.1f}%")

    print(f"\n  Edge Memory:")
    print(f"    Model weights:      {edge_model_bytes_1 / 1024 / 1024:>8.0f} MB "
          f"(layers {cut_1}..{n_layers-1} + embed + lm_head)")
    print(f"    KV cache total:     {edge_kv_total_1 / 1024 / 1024:>8.1f} MB")
    print(f"      Layer 0 (50%):    {edge_kv_layer0 / 1024 / 1024:>8.1f} MB")
    print(f"      Layers 1-{n_layers-1}:     {edge_kv_rest / 1024 / 1024:>8.1f} MB")
    print(f"    Total edge memory:  {(edge_model_bytes_1 + edge_kv_total_1) / 1024 / 1024:>8.0f} MB")
    print(f"    vs Full (model+KV): {(model_bytes + full_kv_bytes) / 1024 / 1024:>8.0f} MB "
          f"({(1 - (edge_model_bytes_1 + edge_kv_total_1) / (model_bytes + full_kv_bytes)) * 100:.0f}% saving)")

    print(f"\n  Edge Decode:")
    print(f"    Pipeline mode:      {tg_ms:.1f} ms/tok + RTT")
    print(f"    After KV received:  {tg_ms:.1f} ms/tok (no RTT, local)")

    # ══════════════════════════════════════════════════════════════════
    # Strategy 2: Cut@18 + int8 gs=32
    # ══════════════════════════════════════════════════════════════════
    cut_18 = n_layers // 2
    print(f"\n{'═' * 80}")
    print(f"Strategy 2: Cut@{cut_18} + int8 gs=32")
    print(f"  Cloud: layers 0..{cut_18-1} → sends int8 KV (all tokens) + h^({cut_18})")
    print(f"  Edge:  layers {cut_18}..{n_layers-1} → reconstructs from h^({cut_18})")
    print(f"{'═' * 80}")

    h_18 = residuals[cut_18]

    # Edge reconstruction
    edge_caches_18, recon_ms_18, recon_mem_18, recon_peak_18 = measure_reconstruction(
        inner, h_18, cut_18, n_layers, N)

    # Cloud KV[0:cut] int8 transfer size
    cloud_kv_bytes_18 = 0
    for i in range(cut_18):
        k, v = full_cache[i].state
        mx.eval(k, v)
        qk = mx.quantize(k, group_size=32, bits=8)
        qv = mx.quantize(v, group_size=32, bits=8)
        cloud_kv_bytes_18 += sum(x.nbytes for x in qk) + sum(x.nbytes for x in qv)

    # h^(18) transfer size
    h18_bytes = N * d_model * 2

    # Edge KV memory: layers 0..17 (dequantized to bf16) + layers 18..35 (exact)
    edge_kv_cloud = sum(full_cache[i].keys.nbytes + full_cache[i].values.nbytes
                         for i in range(cut_18))
    edge_kv_rest_18 = sum(c.keys.nbytes + c.values.nbytes for c in edge_caches_18)
    edge_kv_total_18 = edge_kv_cloud + edge_kv_rest_18

    # Half-model weights
    layers_18_to_L = sum(
        p.nbytes for name, p in tree_flatten(model.parameters())
        if 'layers.' in name and int(name.split('.')[2]) >= cut_18
    )
    edge_model_bytes_18 = layers_18_to_L + embed_lmhead

    print(f"\n  Transfer:")
    print(f"    h^({cut_18}):             {h18_bytes / 1024 / 1024:>8.1f} MB")
    print(f"    int8 KV[0:{cut_18}]:      {cloud_kv_bytes_18 / 1024 / 1024:>8.1f} MB")
    print(f"    Total transfer:     {(h18_bytes + cloud_kv_bytes_18) / 1024 / 1024:>8.1f} MB")
    print(f"    vs Full KV:         {full_kv_bytes / 1024 / 1024:>8.1f} MB "
          f"({full_kv_bytes / (h18_bytes + cloud_kv_bytes_18):.0f}× larger)")

    print(f"\n  Edge Compute:")
    print(f"    Reconstruction:     {recon_ms_18:>8.0f} ms ({n_layers - cut_18} layers)")
    print(f"    vs Full prefill:    {full_prefill_ms:>8.0f} ms ({n_layers} layers)")
    print(f"    Compute saving:     {(1 - recon_ms_18 / full_prefill_ms) * 100:>7.1f}%")

    print(f"\n  Edge Memory:")
    print(f"    Model weights:      {edge_model_bytes_18 / 1024 / 1024:>8.0f} MB "
          f"(layers {cut_18}..{n_layers-1} + embed + lm_head)")
    print(f"    KV cache total:     {edge_kv_total_18 / 1024 / 1024:>8.1f} MB")
    print(f"      Cloud KV (bf16):  {edge_kv_cloud / 1024 / 1024:>8.1f} MB (received int8→deq)")
    print(f"      Edge KV:          {edge_kv_rest_18 / 1024 / 1024:>8.1f} MB")
    print(f"    Total edge memory:  {(edge_model_bytes_18 + edge_kv_total_18) / 1024 / 1024:>8.0f} MB")
    print(f"    vs Full (model+KV): {(model_bytes + full_kv_bytes) / 1024 / 1024:>8.0f} MB "
          f"({(1 - (edge_model_bytes_18 + edge_kv_total_18) / (model_bytes + full_kv_bytes)) * 100:.0f}% saving)")

    print(f"\n  Edge Decode:")
    print(f"    Pipeline mode:      {tg_ms:.1f} ms/tok + RTT")
    print(f"    After KV received:  {tg_ms:.1f} ms/tok (no RTT, local)")
    print(f"    NOTE: Pipeline decode only runs {n_layers - cut_18} layers per token on edge")
    print(f"          but needs cloud round-trip for layers 0..{cut_18-1}")

    # ══════════════════════════════════════════════════════════════════
    # Side by side
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"COMPARISON (N={N} tokens, {model_name})")
    print(f"{'═' * 80}")

    full_mem = model_bytes + full_kv_bytes

    print(f"\n  {'Metric':30s} │ {'S1: Cut@1+AM50%':>18s} │ {'S2: Cut@{0}+int8'.format(cut_18):>18s} │ {'Full local':>12s}")
    print(f"  {'─' * 30}─┼{'─' * 18}─┼{'─' * 18}─┼{'─' * 12}")

    s1_transfer = h1_bytes + cloud_kv_bytes_1
    s2_transfer = h18_bytes + cloud_kv_bytes_18

    s1_edge_mem = edge_model_bytes_1 + edge_kv_total_1
    s2_edge_mem = edge_model_bytes_18 + edge_kv_total_18

    print(f"  {'Transfer size':30s} │ {s1_transfer/1024/1024:>15.1f} MB │ {s2_transfer/1024/1024:>15.1f} MB │ {full_kv_bytes/1024/1024:>9.1f} MB")
    print(f"  {'Edge recon time':30s} │ {recon_ms_1:>14.0f} ms │ {recon_ms_18:>14.0f} ms │ {full_prefill_ms:>8.0f} ms")
    print(f"  {'Edge recon layers':30s} │ {n_layers - cut_1:>15d} L │ {n_layers - cut_18:>15d} L │ {n_layers:>9d} L")
    print(f"  {'Edge model weights':30s} │ {edge_model_bytes_1/1024/1024:>13.0f} MB │ {edge_model_bytes_18/1024/1024:>13.0f} MB │ {model_bytes/1024/1024:>7.0f} MB")
    print(f"  {'Edge KV memory':30s} │ {edge_kv_total_1/1024/1024:>13.1f} MB │ {edge_kv_total_18/1024/1024:>13.1f} MB │ {full_kv_bytes/1024/1024:>7.1f} MB")
    print(f"  {'Edge total memory':30s} │ {s1_edge_mem/1024/1024:>13.0f} MB │ {s2_edge_mem/1024/1024:>13.0f} MB │ {full_mem/1024/1024:>7.0f} MB")
    print(f"  {'Memory saving vs full':30s} │ {(1-s1_edge_mem/full_mem)*100:>14.0f}% │ {(1-s2_edge_mem/full_mem)*100:>14.0f}% │ {'0%':>12s}")
    print(f"  {'Pipeline decode layers':30s} │ {n_layers - cut_1:>15d} L │ {n_layers - cut_18:>15d} L │ {'N/A':>12s}")
    print(f"  {'Pipeline RTT overhead':30s} │ {'per token':>18s} │ {'per token':>18s} │ {'none':>12s}")
    print(f"  {'After KV: local decode':30s} │ {'full {0}L'.format(n_layers):>18s} │ {'full {0}L'.format(n_layers):>18s} │ {'full {0}L'.format(n_layers):>12s}")

    # Compute savings detail
    print(f"\n  Note on compute:")
    print(f"    S1 edge recon = {n_layers - cut_1}L forward on {N} tokens = "
          f"{(n_layers - cut_1)/n_layers*100:.0f}% of full prefill compute")
    print(f"    S2 edge recon = {n_layers - cut_18}L forward on {N} tokens = "
          f"{(n_layers - cut_18)/n_layers*100:.0f}% of full prefill compute")
    print(f"    S1 decode: ALL {n_layers} layers per token (cloud 0..{cut_1-1} via RTT + edge {cut_1}..{n_layers-1})")
    print(f"    S2 decode: ALL {n_layers} layers per token (cloud 0..{cut_18-1} via RTT + edge {cut_18}..{n_layers-1})")
    print(f"    After int8 KV received: both switch to full local decode (no RTT)")

    # Theoretical: if edge only loads half model
    print(f"\n  Theoretical half-model deployment:")
    print(f"    S1 edge weights (layers {cut_1}..{n_layers-1}+embed+lm_head): "
          f"{edge_model_bytes_1/1024/1024:.0f} MB "
          f"(save {(1-edge_model_bytes_1/model_bytes)*100:.0f}% vs full model)")
    print(f"    S2 edge weights (layers {cut_18}..{n_layers-1}+embed+lm_head): "
          f"{edge_model_bytes_18/1024/1024:.0f} MB "
          f"(save {(1-edge_model_bytes_18/model_bytes)*100:.0f}% vs full model)")


if __name__ == "__main__":
    main()
