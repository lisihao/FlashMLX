#!/usr/bin/env python3
"""
Edge-side scored_pq: full cost analysis.

Uses FlashMLX's existing scored_pq system on the edge after pipeline split.
Measures: KV memory, compute time, TTFT, transfer size, quality.
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
from mlx_lm.models.cache_factory import make_optimized_cache
from mlx_lm.models.kv_direct_cache import _find_inner_model
from mlx_lm.sample_utils import make_sampler
from mlx.utils import tree_flatten

GREEDY = make_sampler(temp=0.0)

DIVERSE_PROMPT = """The history of artificial intelligence began in antiquity, with myths and stories of artificial beings endowed with intelligence by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe human thinking as mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s. The field of AI research was founded at Dartmouth College in 1956. Many predicted a machine as intelligent as a human would exist within a generation. Eventually it became obvious that researchers had grossly underestimated the difficulty. In 1973, governments stopped funding undirected AI research, leading to the first AI winter. Investment boomed again in the 21st century when machine learning was successfully applied to many problems due to new methods, the application of powerful computer hardware, and the collection of immense data sets. Deep learning transformed the field starting around 2012 when neural networks began dramatically outperforming other methods."""


def mem_mb():
    return mx.get_active_memory() / 1024 / 1024


def peak_mb():
    return mx.get_peak_memory() / 1024 / 1024


def reset_peak():
    mx.reset_peak_memory()


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


def _single_cache_bytes(c):
    """KV bytes for one cache layer."""
    ctype = type(c).__name__
    if ctype == 'TripleLayerKVCache':
        total = 0
        # Flat buffer
        if hasattr(c, '_flat_keys') and c._flat_keys is not None:
            # Only count up to _flat_offset tokens
            B, H, _, D = c._flat_keys.shape
            off = getattr(c, '_flat_offset', c._flat_keys.shape[2])
            total += off * H * D * 2 * 2  # K+V, bf16 = 2 bytes
            return total
        # Recent
        if hasattr(c, 'recent_keys') and c.recent_keys is not None:
            total += c.recent_keys.nbytes + c.recent_values.nbytes
        # Warm
        if hasattr(c, 'warm_keys') and c.warm_keys is not None:
            total += c.warm_keys.nbytes + c.warm_values.nbytes
        # Cold
        if hasattr(c, 'cold_compressed_keys') and c.cold_compressed_keys is not None:
            total += c.cold_compressed_keys.nbytes + c.cold_compressed_values.nbytes
        if hasattr(c, 'cold_pending_keys') and c.cold_pending_keys is not None:
            total += c.cold_pending_keys.nbytes + c.cold_pending_values.nbytes
        return total
    elif hasattr(c, 'keys') and c.keys is not None:
        return c.keys.nbytes + c.values.nbytes
    return 0


def cache_kv_bytes(cache_list):
    """Total KV memory across all cache layers."""
    return sum(_single_cache_bytes(c) for c in cache_list)


def cache_kv_bytes_range(cache_list, start, end):
    """KV memory for layers [start, end)."""
    return sum(_single_cache_bytes(cache_list[i])
               for i in range(start, min(end, len(cache_list))))


def timed_decode(model, tokenizer, cache, question, n_tokens=20):
    """Measure decode: TTFT (first token) + TG (subsequent tokens)."""
    q = mx.array(tokenizer.encode(question))
    times = []
    t_start = time.perf_counter()
    for tok, _ in generate_step(q, model, max_tokens=n_tokens,
                                 sampler=GREEDY, prompt_cache=cache):
        mx.eval(mx.array([tok]))
        t_now = time.perf_counter()
        times.append((t_now - t_start) * 1000)
        t_start = t_now
    # times[0] = TTFT (includes question encoding), times[1:] = per-token TG
    ttft = times[0] if times else 0
    tg_avg = np.mean(times[2:]) if len(times) > 2 else (np.mean(times[1:]) if len(times) > 1 else 0)
    return ttft, tg_avg, len(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--gen-tokens", type=int, default=50)
    parser.add_argument("--calibration", default=None)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    inner = _find_inner_model(model)
    n_layers = len(inner.layers)
    cfg = model.args if hasattr(model, 'args') else inner.args
    d_model = cfg.hidden_size
    n_kv_heads = cfg.num_key_value_heads
    head_dim = d_model // cfg.num_attention_heads
    kv_per_token_per_layer = 2 * n_kv_heads * head_dim * 2  # K+V bf16

    model_name = args.model.split('/')[-1]
    model_bytes = sum(p.nbytes for _, p in tree_flatten(model.parameters()))

    # Auto-detect calibration
    cal_file = args.calibration
    if cal_file is None:
        import os, glob
        candidates = glob.glob(os.path.expanduser(
            "~/.cache/flashmlx/calibrations/qwen3_h*_l{}_*.pkl".format(n_layers)))
        if candidates:
            cal_file = candidates[0]

    print("=" * 80)
    print(f"Edge-side scored_pq Cost Analysis: {model_name}")
    print("=" * 80)
    print(f"layers={n_layers}, d={d_model}, kv_heads={n_kv_heads}, head_dim={head_dim}")
    print(f"model weights: {model_bytes/1024/1024:.0f} MB")
    print(f"KV per token per layer: {kv_per_token_per_layer} B")
    print(f"calibration: {cal_file}")

    question = "\nQ: What is the main topic of this text?\nA:"

    # Build prompts long enough for target token count
    target = args.prompt_tokens
    rep_base = "The quick brown fox jumps over the lazy dog. "
    rep_text = rep_base * (target // 5)  # ~10 tokens per repeat, overshoot
    div_text = DIVERSE_PROMPT * (target // 100 + 1)  # ~200 tokens per copy

    prompts = {
        "repetitive": rep_text,
        "diverse": div_text,
    }

    for prompt_name, prompt_text in prompts.items():
        tokens = tokenizer.encode(prompt_text)[:args.prompt_tokens]
        N = len(tokens)
        tok_mx = mx.array(tokens)

        print(f"\n{'#' * 80}")
        print(f"# {prompt_name} ({N} tokens)")
        print(f"{'#' * 80}")

        # Theoretical full KV size
        full_kv_theory = N * n_layers * kv_per_token_per_layer
        print(f"  Theoretical full KV: {full_kv_theory/1024/1024:.1f} MB")

        # ═════════════════════════════════════════════════════════
        # Baseline: standard KVCache, full local
        # ═════════════════════════════════════════════════════════
        print(f"\n  {'═' * 75}")
        print(f"  A) Baseline: standard KVCache, full local prefill")
        print(f"  {'═' * 75}")

        gc.collect()
        cache_std = [KVCache() for _ in range(n_layers)]
        x = inner.embed_tokens(tok_mx.reshape(1, -1))
        mx.eval(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(x.dtype)
        residuals = [x]

        reset_peak()
        mem_b = mem_mb()
        t0 = time.perf_counter()
        for i, layer in enumerate(inner.layers):
            x = layer(x, mask=mask, cache=cache_std[i])
            mx.eval(x)
            residuals.append(x)
        prefill_std_ms = (time.perf_counter() - t0) * 1000
        mem_a = mem_mb()
        peak_a = peak_mb()

        kv_std = cache_kv_bytes(cache_std)

        # Save clean prefill-only KV (before generation pollutes cache)
        clean_cloud_kv = []
        for i in range(n_layers):
            k, v = cache_std[i].state
            mx.eval(k, v)
            clean_cloud_kv.append((k[:, :, :N, :], v[:, :, :N, :]))
            mx.eval(clean_cloud_kv[-1][0], clean_cloud_kv[-1][1])

        baseline_tok, baseline_text = generate_text(
            model, tokenizer, cache_std, question, args.gen_tokens)
        print(f"  Output: {baseline_text[:90]}")
        print(f"  Prefill:    {prefill_std_ms:>7.0f} ms ({n_layers}L)")
        print(f"  KV memory:  {kv_std/1024/1024:>7.1f} MB")
        print(f"  Mem delta:  {mem_a - mem_b:>7.0f} MB")
        print(f"  Peak:       {peak_a:>7.0f} MB")

        # ═════════════════════════════════════════════════════════
        # For each cut point, test scored_pq edge strategy
        # ═════════════════════════════════════════════════════════
        for cut_l in [1, n_layers // 2]:
            print(f"\n  {'═' * 75}")
            print(f"  B) Pipeline@{cut_l} + scored_pq on edge")
            print(f"     Cloud: layers 0..{cut_l-1} → sends h^({cut_l}) + KV[0:{cut_l}]")
            print(f"     Edge:  layers {cut_l}..{n_layers-1} → scored_pq cache")
            print(f"  {'═' * 75}")

            # ── Cloud side (simulated) ──
            h_cut = residuals[cut_l]
            h_cut_bytes = h_cut.nbytes

            # Cloud KV: exact bf16 for layers 0..cut-1 (prefill only)
            cloud_kv_exact = sum(
                clean_cloud_kv[i][0].nbytes + clean_cloud_kv[i][1].nbytes
                for i in range(cut_l))

            # Cloud KV: int8 gs=32 (compressed transfer)
            cloud_kv_int8 = 0
            for i in range(cut_l):
                k, v = clean_cloud_kv[i]
                qk = mx.quantize(k, group_size=32, bits=8)
                qv = mx.quantize(v, group_size=32, bits=8)
                cloud_kv_int8 += sum(x.nbytes for x in qk) + sum(x.nbytes for x in qv)

            transfer_exact = h_cut_bytes + cloud_kv_exact
            transfer_int8 = h_cut_bytes + cloud_kv_int8

            print(f"\n  Transfer:")
            print(f"    h^({cut_l}):              {h_cut_bytes/1024/1024:>8.2f} MB")
            print(f"    Cloud KV (bf16):    {cloud_kv_exact/1024/1024:>8.2f} MB")
            print(f"    Cloud KV (int8):    {cloud_kv_int8/1024/1024:>8.2f} MB")
            print(f"    Total (bf16):       {transfer_exact/1024/1024:>8.2f} MB")
            print(f"    Total (int8):       {transfer_int8/1024/1024:>8.2f} MB")
            print(f"    vs Full KV:         {kv_std/1024/1024:>8.1f} MB "
                  f"({kv_std/transfer_int8:.1f}× larger)")

            # ── Edge side: standard KVCache for reconstruction ──
            # (scored_pq chunk eviction conflicts with static mask at long context,
            #  so reconstruct with standard KVCache, scored_pq tested separately in C)
            gc.collect()
            edge_caches = [KVCache() for _ in range(n_layers)]

            # Cloud layers: inject CLEAN prefill-only KV (N tokens, no generation)
            for i in range(cut_l):
                edge_caches[i].state = clean_cloud_kv[i]

            # Edge reconstruction: layers cut..L-1
            reset_peak()
            mem_b = mem_mb()
            t0 = time.perf_counter()
            xr = h_cut
            mask_r = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(h_cut.dtype)
            for i in range(cut_l, n_layers):
                xr = inner.layers[i](xr, mask=mask_r, cache=edge_caches[i])
                mx.eval(xr)
            recon_ms = (time.perf_counter() - t0) * 1000
            mem_a = mem_mb()
            peak_r = peak_mb()

            # KV memory measurement
            cloud_kv_mem = cache_kv_bytes_range(edge_caches, 0, cut_l)
            edge_kv_mem = cache_kv_bytes_range(edge_caches, cut_l, n_layers)
            total_kv_mem = cloud_kv_mem + edge_kv_mem

            print(f"\n  Edge Compute:")
            print(f"    Reconstruction:     {recon_ms:>8.0f} ms ({n_layers - cut_l} layers)")
            print(f"    vs Full prefill:    {prefill_std_ms:>8.0f} ms ({n_layers} layers)")
            print(f"    Compute saving:     {(1 - recon_ms/prefill_std_ms)*100:>7.1f}%")

            print(f"\n  Edge KV Memory (bf16, pre-compression):")
            print(f"    Cloud KV[0:{cut_l}] (injected):     {cloud_kv_mem/1024/1024:>7.1f} MB")
            print(f"    Edge KV[{cut_l}:{n_layers}] (reconstructed):{edge_kv_mem/1024/1024:>7.1f} MB")
            print(f"    Total KV:           {total_kv_mem/1024/1024:>8.1f} MB")
            print(f"    vs Standard full:   {kv_std/1024/1024:>8.1f} MB "
                  f"({(1 - total_kv_mem/kv_std)*100:.0f}% saving)")
            print(f"    Mem delta (real):   {mem_a - mem_b:>8.0f} MB")
            print(f"    Peak:               {peak_r:>8.0f} MB")

            # Theoretical half-model weights
            edge_layer_bytes = sum(
                p.nbytes for name, p in tree_flatten(model.parameters())
                if 'layers.' in name and int(name.split('.')[2]) >= cut_l)
            embed_lmhead = model_bytes - sum(
                p.nbytes for name, p in tree_flatten(model.parameters())
                if 'layers.' in name)
            edge_model_bytes = edge_layer_bytes + embed_lmhead

            print(f"\n  Edge Model (theoretical half-model):")
            print(f"    Weights:            {edge_model_bytes/1024/1024:>8.0f} MB "
                  f"(layers {cut_l}..{n_layers-1} + embed + lm_head)")
            print(f"    + KV:               {total_kv_mem/1024/1024:>8.1f} MB")
            total_edge = edge_model_bytes + total_kv_mem
            total_full = model_bytes + kv_std
            print(f"    Total edge mem:     {total_edge/1024/1024:>8.0f} MB")
            print(f"    vs Full local:      {total_full/1024/1024:>8.0f} MB "
                  f"({(1 - total_edge/total_full)*100:.0f}% saving)")

            # Quality
            gen, _ = generate_text(
                model, tokenizer, edge_caches, question, args.gen_tokens)
            qs = quality_str(baseline_tok, gen)
            print(f"\n  Quality: {qs}")

            # Decode timing — rebuild fresh standard cache
            gc.collect()
            edge_caches2 = [KVCache() for _ in range(n_layers)]
            for i in range(cut_l):
                edge_caches2[i].state = clean_cloud_kv[i]
            xr2 = h_cut
            for i in range(cut_l, n_layers):
                xr2 = inner.layers[i](xr2, mask=mask_r, cache=edge_caches2[i])
                mx.eval(xr2)
            ttft, tg_avg, n_tok = timed_decode(
                model, tokenizer, edge_caches2, question, 20)
            print(f"\n  Decode Speed:")
            print(f"    TTFT (question):    {ttft:>8.1f} ms")
            print(f"    TG avg:             {tg_avg:>8.1f} ms/tok ({n_tok} tokens)")

        # ═════════════════════════════════════════════════════════
        # Also test: scored_pq full local (no pipeline, just compression)
        # NOTE: Use model's own forward (not manual layer iteration) because
        # scored_pq evicts tokens inside update_and_fetch, changing KV size.
        # model() uses "causal" string mask → MLX handles dynamic KV natively.
        # Manual layer iteration with static (N,N) array mask crashes.
        # ═════════════════════════════════════════════════════════
        print(f"\n  {'═' * 75}")
        print(f"  C) scored_pq full local (no pipeline split)")
        print(f"  {'═' * 75}")

        gc.collect()
        cache_spq = make_optimized_cache(
            inner, strategy="scored_pq",
            calibration_file=cal_file,
            scored_max_cache=2048)

        reset_peak()
        mem_b = mem_mb()
        t0 = time.perf_counter()
        logits_spq = model(tok_mx.reshape(1, -1), cache=cache_spq)
        mx.eval(logits_spq)
        prefill_spq_ms = (time.perf_counter() - t0) * 1000
        mem_a = mem_mb()
        peak_s = peak_mb()

        kv_spq = cache_kv_bytes(cache_spq)
        gen_spq, _ = generate_text(
            model, tokenizer, cache_spq, question, args.gen_tokens)
        qs_spq = quality_str(baseline_tok, gen_spq)

        print(f"  Prefill:    {prefill_spq_ms:>7.0f} ms")
        print(f"  KV memory:  {kv_spq/1024/1024:>7.1f} MB "
              f"({(1 - kv_spq/kv_std)*100:.0f}% saving vs standard)")
        print(f"  Mem delta:  {mem_a - mem_b:>7.0f} MB")
        print(f"  Peak:       {peak_s:>7.0f} MB")
        print(f"  Quality:    {qs_spq}")

        # Decode timing — build fresh scored_pq cache
        gc.collect()
        cache_spq2 = make_optimized_cache(
            inner, strategy="scored_pq",
            calibration_file=cal_file,
            scored_max_cache=2048)
        logits2 = model(tok_mx.reshape(1, -1), cache=cache_spq2)
        mx.eval(logits2)
        ttft_spq, tg_spq, n_spq = timed_decode(
            model, tokenizer, cache_spq2, question, 20)
        print(f"  TTFT:       {ttft_spq:>7.1f} ms")
        print(f"  TG avg:     {tg_spq:>7.1f} ms/tok")

        # ═════════════════════════════════════════════════════════
        # D) Pipeline@cut + scored_pq on edge (the real target scenario)
        # Cloud layers: standard KVCache (injected)
        # Edge layers: scored_pq cache (compressed after reconstruction)
        # ═════════════════════════════════════════════════════════
        for cut_l in [n_layers // 2]:
            print(f"\n  {'═' * 75}")
            print(f"  D) Pipeline@{cut_l} + scored_pq on edge layers")
            print(f"     Cloud: KVCache 0..{cut_l-1} (injected, bf16)")
            print(f"     Edge:  scored_pq {cut_l}..{n_layers-1} (reconstruct → compress)")
            print(f"  {'═' * 75}")

            # Step 1: Create scored_pq cache for ALL layers
            gc.collect()
            cache_d = make_optimized_cache(
                inner, strategy="scored_pq",
                calibration_file=cal_file,
                scored_max_cache=2048)

            # Step 2: Replace cloud layers with standard KVCache (exact, injected)
            for i in range(cut_l):
                cache_d[i] = KVCache()
                cache_d[i].state = clean_cloud_kv[i]

            # Step 3: Reconstruct edge layers using model forward from h^(cut)
            # Use "causal" string mask for scored_pq compatibility
            h_cut_d = residuals[cut_l]
            reset_peak()
            mem_b = mem_mb()
            t0 = time.perf_counter()
            xd = h_cut_d
            for i in range(cut_l, n_layers):
                xd = inner.layers[i](xd, mask="causal", cache=cache_d[i])
                mx.eval(xd)
            recon_d_ms = (time.perf_counter() - t0) * 1000
            mem_a = mem_mb()

            cloud_kv_d = cache_kv_bytes_range(cache_d, 0, cut_l)
            edge_kv_d = cache_kv_bytes_range(cache_d, cut_l, n_layers)
            total_kv_d = cloud_kv_d + edge_kv_d

            gen_d, _ = generate_text(
                model, tokenizer, cache_d, question, args.gen_tokens)
            qs_d = quality_str(baseline_tok, gen_d)

            print(f"  Reconstruction:   {recon_d_ms:>8.0f} ms ({n_layers - cut_l} layers)")
            print(f"  Cloud KV (bf16):  {cloud_kv_d/1024/1024:>8.1f} MB")
            print(f"  Edge KV (scored): {edge_kv_d/1024/1024:>8.1f} MB")
            print(f"  Total KV:         {total_kv_d/1024/1024:>8.1f} MB "
                  f"({(1 - total_kv_d/kv_std)*100:.0f}% saving)")
            print(f"  vs Pipeline bf16: {cache_kv_bytes_range(edge_caches, cut_l, n_layers)/1024/1024:>8.1f} MB edge "
                  f"→ {edge_kv_d/1024/1024:.1f} MB scored_pq "
                  f"({(1 - edge_kv_d/cache_kv_bytes_range(edge_caches, cut_l, n_layers))*100:.0f}% saving)")
            print(f"  Quality:          {qs_d}")

    # ═════════════════════════════════════════════════════════
    # Summary comparison table
    # ═════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print("NETWORK LATENCY MODEL")
    print(f"{'═' * 80}")

    PROFILES = {
        'WiFi  (50Mbps, 5ms)':  {'bw_mbps': 50,   'rtt_ms': 5},
        '5G   (100Mbps, 20ms)': {'bw_mbps': 100,  'rtt_ms': 20},
        '4G    (20Mbps, 50ms)': {'bw_mbps': 20,   'rtt_ms': 50},
        'LAN   (1Gbps, 0.5ms)': {'bw_mbps': 1000, 'rtt_ms': 0.5},
    }

    print(f"\n  Full KV transfer: {kv_std/1024/1024:.1f} MB")
    print(f"  Pipeline@1 int8:  {transfer_int8/1024/1024:.1f} MB (last prompt's values)")
    print(f"  Pipeline@18 int8: (compute from 18-layer cut)")

    for net_name, prof in PROFILES.items():
        full_xfer_ms = kv_std * 8 / (prof['bw_mbps'] * 1e6) * 1000
        pipe_xfer_ms = transfer_int8 * 8 / (prof['bw_mbps'] * 1e6) * 1000

        print(f"\n  {net_name}:")
        print(f"    Full KV transfer:   {full_xfer_ms:>8.0f} ms")
        print(f"    Pipeline int8:      {pipe_xfer_ms:>8.0f} ms")
        print(f"    TTFT (full xfer):   {full_xfer_ms + prof['rtt_ms']:>8.0f} ms")
        print(f"    TTFT (pipe+recon):  {pipe_xfer_ms + prof['rtt_ms'] + recon_ms:>8.0f} ms")

    print()


if __name__ == "__main__":
    main()
