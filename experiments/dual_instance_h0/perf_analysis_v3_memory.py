#!/usr/bin/env python3
"""
v3: Memory-focused analysis.

Q1: Reconstruction chunked → PP peak memory vs traditional unchunked prefill?
Q2: TG with h^(0) backup → can KV cache be much smaller (eviction/window)?
Q3: True minimum memory budget for each node?
"""

from __future__ import annotations

import gc
import sys
import time

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.kv_direct_cache import (
    H0Store,
    _find_inner_model,
    reconstruct_prefix_kv,
    create_attention_mask,
)


def mem_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except:
        return mx.metal.get_active_memory() / (1024 * 1024)

def peak_mb():
    try:
        return mx.get_peak_memory() / (1024 * 1024)
    except:
        return mx.metal.get_peak_memory() / (1024 * 1024)

def reset_peak():
    try:
        mx.reset_peak_memory()
    except:
        mx.metal.reset_peak_memory()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    cfg = model.args
    n_layers = cfg.num_hidden_layers
    n_kv_heads = cfg.num_key_value_heads
    n_q_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_q_heads
    d_hidden = cfg.hidden_size

    print(f"Model: {n_layers}L, {n_q_heads}Q/{n_kv_heads}KV heads, d_h={head_dim}, d={d_hidden}")

    # Build tokens
    FILLER = "The development of artificial intelligence has progressed rapidly in recent years. " * 5
    text = FILLER * 100
    tokens_list = tokenizer.encode(text)[:args.prompt_tokens]
    tokens = mx.array(tokens_list)
    N = len(tokens_list)
    print(f"Prompt: {N} tokens\n")

    # Warmup
    model(mx.array(tokenizer.encode("Hi")).reshape(1, -1))
    mx.eval(model.parameters())
    inner_model = _find_inner_model(model)

    gc.collect(); mx.clear_cache()
    mem_model = mem_mb()
    print(f"Model memory baseline: {mem_model:.0f} MB")

    # Theoretical sizes
    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * 2  # bytes
    kv_total = kv_per_token * N
    h0_per_token = d_hidden * 2
    h0_total = h0_per_token * N
    # Attention score peak (per layer, unchunked): n_q_heads × N × N × sizeof(float32)
    attn_peak_unchunked = n_q_heads * N * N * 4  # per layer

    print(f"\nTheoretical sizes for {N} tokens:")
    print(f"  KV cache total:          {kv_total / 1024 / 1024:>8.1f} MB  ({kv_per_token} bytes/token)")
    print(f"  h^(0) total:             {h0_total / 1024 / 1024:>8.1f} MB  ({h0_per_token} bytes/token)")
    print(f"  Attn scores (unchunked): {attn_peak_unchunked / 1024 / 1024:>8.1f} MB  (per layer, {n_q_heads}H × {N} × {N} × 4B)")

    # ================================================================
    # TEST 1: Traditional unchunked prefill — peak memory
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  TEST 1: TRADITIONAL PREFILL (unchunked)")
    print(f"{'='*80}")

    gc.collect(); mx.clear_cache()
    reset_peak()
    m0 = mem_mb()

    cache = make_prompt_cache(model)
    m_cache = mem_mb()

    out = model(tokens.reshape(1, -1), cache=cache)
    mx.eval(out)
    m_after = mem_mb()
    p_after = peak_mb()

    print(f"  Before:          {m0:>8.0f} MB")
    print(f"  After cache:     {m_cache:>8.0f} MB  (+{m_cache - m0:.0f} MB cache alloc)")
    print(f"  After prefill:   {m_after:>8.0f} MB  (+{m_after - m0:.0f} MB total)")
    print(f"  Peak:            {p_after:>8.0f} MB  (+{p_after - m0:.0f} MB peak overhead)")
    print(f"  KV in cache:     {sum(c.keys.nbytes + c.values.nbytes for c in cache) / 1024 / 1024:.1f} MB")
    print(f"  Activation peak: {p_after - m_after:>8.0f} MB  (peak − steady)")

    del cache, out
    gc.collect(); mx.clear_cache()

    # ================================================================
    # TEST 2: Chunked reconstruction — peak memory for various chunk sizes
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  TEST 2: CHUNKED RECONSTRUCTION — peak memory by chunk_size")
    print(f"{'='*80}")

    # First get h^(0)
    h0 = inner_model.embed_tokens(tokens.reshape(1, -1))
    mx.eval(h0)
    h0_store = H0Store()
    h0_store.append(h0)
    del h0
    gc.collect(); mx.clear_cache()

    print(f"\n  {'chunk_size':>10} {'time':>8} {'peak':>10} {'steady':>10} {'act_peak':>10} {'attn_theory':>12}")
    print(f"  {'─'*62}")

    for chunk_size in [0, 2048, 1024, 512, 256, 128]:
        gc.collect(); mx.clear_cache()
        reset_peak()
        m0 = mem_mb()

        t0 = time.perf_counter()
        kv_pairs = reconstruct_prefix_kv(
            inner_model, h0_store, 0, N,
            chunk_size=chunk_size, eval_every=8,
        )
        mx.eval(*[k for k, v in kv_pairs] + [v for k, v in kv_pairs])
        t_ms = (time.perf_counter() - t0) * 1000

        m_after = mem_mb()
        p_after = peak_mb()
        act_peak = p_after - m_after

        # Theoretical attention peak for this chunk size
        cs = chunk_size if chunk_size > 0 else N
        # Worst case: last chunk attends to full N tokens
        attn_theory = n_q_heads * cs * N * 4 / 1024 / 1024

        label = "unchunked" if chunk_size == 0 else str(chunk_size)
        print(f"  {label:>10} {t_ms:>7.0f}ms {p_after - mem_model:>8.0f} MB  "
              f"{m_after - mem_model:>8.0f} MB  {act_peak:>8.0f} MB  {attn_theory:>10.0f} MB")

        del kv_pairs
        gc.collect(); mx.clear_cache()

    # ================================================================
    # TEST 3: TG memory — full KV vs windowed KV
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  TEST 3: TG MEMORY — full KV vs windowed KV + h^(0) backup")
    print(f"{'='*80}")

    # 3a: Full KV TG (reconstruct all, keep all)
    gc.collect(); mx.clear_cache()

    h0 = inner_model.embed_tokens(tokens.reshape(1, -1))
    mx.eval(h0)
    h0_store = H0Store()
    h0_store.append(h0)

    kv_pairs = reconstruct_prefix_kv(inner_model, h0_store, 0, N, chunk_size=512, eval_every=8)
    mx.eval(*[k for k, v in kv_pairs] + [v for k, v in kv_pairs])

    cache_full = make_prompt_cache(model)
    for i, (k, v) in enumerate(kv_pairs):
        cache_full[i].state = (k, v)
    mx.eval([c.keys for c in cache_full] + [c.values for c in cache_full])
    del kv_pairs

    gc.collect(); mx.clear_cache()
    reset_peak()
    m_full_kv = mem_mb()
    full_kv_mb = sum(c.keys.nbytes + c.values.nbytes for c in cache_full) / 1024 / 1024

    # Do 10 TG steps
    q = mx.array(tokenizer.encode("What is"))
    out = model(q.reshape(1, -1), cache=cache_full)
    mx.eval(out)
    y = mx.argmax(out[:, -1, :], axis=-1)
    for _ in range(9):
        out = model(y.reshape(1, 1), cache=cache_full)
        mx.eval(out)
        y = mx.argmax(out[:, -1, :], axis=-1)
        mx.eval(y)
    m_full_tg = mem_mb()
    p_full_tg = peak_mb()

    del cache_full, out
    gc.collect(); mx.clear_cache()

    # 3b: Windowed KV — only keep last W tokens, h^(0) as backup
    # Simulate: reconstruct only last W tokens into cache
    windows = [256, 512, 1024, 2048]
    print(f"\n  {'Config':<30} {'KV cache':>10} {'h^(0)':>8} {'Total state':>12} {'vs Full KV':>10}")
    print(f"  {'─'*70}")
    print(f"  {'Full KV (all ' + str(N) + ' tokens)':<30} {full_kv_mb:>8.1f} MB {'—':>8} {full_kv_mb:>10.1f} MB {'baseline':>10}")

    for W in windows:
        gc.collect(); mx.clear_cache()

        # Reconstruct only last W tokens
        start = max(0, N - W)
        h0_new = H0Store()
        h0_range = h0_store.get_range(0, N)
        h0_new.append(h0_range)

        kv_w = reconstruct_prefix_kv(inner_model, h0_new, 0, N, chunk_size=512, eval_every=8)
        mx.eval(*[k for k, v in kv_w] + [v for k, v in kv_w])

        # Only keep last W tokens in cache
        cache_w = make_prompt_cache(model)
        for i, (k, v) in enumerate(kv_w):
            # Slice to last W tokens
            cache_w[i].state = (k[:, :, -W:, :], v[:, :, -W:, :])
        mx.eval([c.keys for c in cache_w] + [c.values for c in cache_w])

        w_kv_mb = sum(c.keys.nbytes + c.values.nbytes for c in cache_w) / 1024 / 1024
        h0_mb = h0_store.get_range(0, N).nbytes / 1024 / 1024
        total_mb = w_kv_mb + h0_mb
        vs_full = (1 - total_mb / full_kv_mb) * 100

        del kv_w, cache_w, h0_new
        gc.collect(); mx.clear_cache()

        print(f"  {'Window=' + str(W) + ' + h^(0) backup':<30} {w_kv_mb:>8.1f} MB {h0_mb:>6.1f} MB {total_mb:>10.1f} MB {vs_full:>+9.0f}%")

    # 3c: h^(0) only — zero KV, reconstruct on demand per TG step
    h0_mb = h0_store.get_range(0, N).nbytes / 1024 / 1024
    print(f"  {'h^(0) only (on-demand RC)':<30} {'0.0':>8} MB {h0_mb:>6.1f} MB {h0_mb:>10.1f} MB {(1-h0_mb/full_kv_mb)*100:>+9.0f}%")

    # ================================================================
    # TEST 4: Reconstruction with immediate eviction — layer-by-layer memory
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  TEST 4: MINIMUM MEMORY RECONSTRUCTION")
    print(f"{'='*80}")

    # Can we reconstruct layer-by-layer and immediately compress/evict?
    # reconstruct_prefix_kv returns ALL layers at once.
    # But we could modify it to process one layer at a time and compress immediately.
    # Let's measure what layer-streaming would save.

    gc.collect(); mx.clear_cache()
    reset_peak()
    m0 = mem_mb()

    # Full reconstruction — all layers in memory at once
    h0_fresh = H0Store()
    h0_fresh.append(inner_model.embed_tokens(tokens.reshape(1, -1)))
    kv_all = reconstruct_prefix_kv(inner_model, h0_fresh, 0, N, chunk_size=512, eval_every=8)
    mx.eval(*[k for k, v in kv_all] + [v for k, v in kv_all])
    m_all = mem_mb()
    p_all = peak_mb()

    # Per-layer KV size
    per_layer_kv = (kv_all[0][0].nbytes + kv_all[0][1].nbytes) / 1024 / 1024
    all_layers_kv = per_layer_kv * n_layers

    del kv_all, h0_fresh
    gc.collect(); mx.clear_cache()

    print(f"\n  Full reconstruction (all {n_layers} layers in memory):")
    print(f"    KV total:      {all_layers_kv:>8.1f} MB  ({per_layer_kv:.1f} MB × {n_layers})")
    print(f"    Peak:          {p_all - m0:>8.0f} MB")

    # Simulate layer-streaming: reconstruct 1 layer at a time
    # In theory: only need 1 layer's KV + h^(0) + activations at peak
    min_kv = per_layer_kv  # only 1 layer in memory at a time (if we could stream to SSD/compress)
    print(f"\n  Layer-streaming reconstruction (compress/evict per layer):")
    print(f"    KV peak:       {min_kv:>8.1f} MB  (1 layer)")
    print(f"    h^(0):         {h0_total / 1024 / 1024:>8.1f} MB")
    print(f"    Activations:   ~{n_q_heads * 512 * N * 4 / 1024 / 1024:.0f} MB  (chunk=512)")
    print(f"    vs full:       {min_kv / all_layers_kv * 100:>8.1f}%  (KV memory)")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  MINIMUM MEMORY BUDGETS (Qwen3-1.7B, {N} tokens)")
    print(f"{'='*80}")

    def param_bytes(module):
        leaves = nn.utils.tree_flatten(module.parameters() if hasattr(module, 'parameters') else {})
        return sum(v.nbytes for _, v in leaves if hasattr(v, 'nbytes'))

    embed_mb_val = param_bytes(inner_model.embed_tokens) / 1024 / 1024
    layers_mb = sum(param_bytes(l) for l in inner_model.layers) / 1024 / 1024
    model_mb = embed_mb_val + layers_mb + param_bytes(inner_model.norm) / 1024 / 1024

    print(f"\n  Node A (Embedder):")
    print(f"    embed_tokens:  {embed_mb_val:>8.1f} MB")
    print(f"    h^(0) output:  {h0_total / 1024 / 1024:>8.1f} MB  (transient)")
    print(f"    Total:         {embed_mb_val + h0_total / 1024 / 1024:>8.1f} MB")

    print(f"\n  Node B (Decoder) — three possible configurations:")
    print(f"")
    # Config 1: Full KV
    b1_model = model_mb
    b1_kv = all_layers_kv
    b1_h0 = 0
    b1_total = b1_model + b1_kv
    print(f"    B1: Full KV (traditional)")
    print(f"      Model:       {b1_model:>8.1f} MB")
    print(f"      KV cache:    {b1_kv:>8.1f} MB  (all {N} tokens × {n_layers} layers)")
    print(f"      Total:       {b1_total:>8.1f} MB")

    # Config 2: Window KV + h^(0) backup
    W = 512
    b2_kv = per_layer_kv * n_layers * W / N
    b2_h0 = h0_total / 1024 / 1024
    b2_total = model_mb + b2_kv + b2_h0
    print(f"")
    print(f"    B2: Window={W} + h^(0) backup (on-demand reconstruct)")
    print(f"      Model:       {model_mb:>8.1f} MB")
    print(f"      KV cache:    {b2_kv:>8.1f} MB  (last {W} tokens × {n_layers} layers)")
    print(f"      h^(0):       {b2_h0:>8.1f} MB  (backup for cold reconstruct)")
    print(f"      Total:       {b2_total:>8.1f} MB  (−{(1 - b2_total / b1_total) * 100:.0f}% vs B1)")

    # Config 3: Scored_pq (Q8 flat) + h^(0) backup
    # scored_pq: 50% keep, Q8 flat = 50% of bf16 size
    b3_recent = per_layer_kv * n_layers * 512 / N  # 512 recent, full precision
    b3_flat = all_layers_kv * 0.5 * 0.5  # 50% tokens, Q8 (50% of bf16)
    b3_h0 = h0_total / 1024 / 1024
    b3_total = model_mb + b3_recent + b3_flat + b3_h0
    print(f"")
    print(f"    B3: scored_pq (Q8 flat, 50% keep) + h^(0)")
    print(f"      Model:       {model_mb:>8.1f} MB")
    print(f"      Recent KV:   {b3_recent:>8.1f} MB  (512 tokens, bf16)")
    print(f"      Flat KV:     {b3_flat:>8.1f} MB  ({int(N*0.5)} tokens, Q8)")
    print(f"      h^(0):       {b3_h0:>8.1f} MB")
    print(f"      Total:       {b3_total:>8.1f} MB  (−{(1 - b3_total / b1_total) * 100:.0f}% vs B1)")

    # Config 4: h^(0) only — minimal KV (just for current TG window)
    b4_kv = per_layer_kv * n_layers * 64 / N  # tiny 64-token window
    b4_h0 = h0_total / 1024 / 1024
    b4_total = model_mb + b4_kv + b4_h0
    print(f"")
    print(f"    B4: h^(0) only + 64-token KV window (aggressive)")
    print(f"      Model:       {model_mb:>8.1f} MB")
    print(f"      KV cache:    {b4_kv:>8.1f} MB  (64-token window)")
    print(f"      h^(0):       {b4_h0:>8.1f} MB")
    print(f"      Total:       {b4_total:>8.1f} MB  (−{(1 - b4_total / b1_total) * 100:.0f}% vs B1)")

    # Scaling to longer sequences
    print(f"\n{'='*80}")
    print(f"  SCALING: B2 (Window=512 + h^(0)) vs B1 (Full KV)")
    print(f"{'='*80}")
    print(f"\n  {'Context':>10} {'Full KV':>10} {'Window+h0':>10} {'Savings':>10} {'KV ratio':>10}")
    print(f"  {'─'*50}")
    for ctx in [1024, 4096, 16384, 32768, 65536, 131072]:
        fkv = kv_per_token * ctx / 1024 / 1024
        wkv = kv_per_token * 512 / 1024 / 1024  # fixed window
        wh0 = h0_per_token * ctx / 1024 / 1024
        savings = (1 - (wkv + wh0) / fkv) * 100
        print(f"  {ctx:>10,} {fkv:>8.0f} MB {wkv + wh0:>8.0f} MB {savings:>+9.0f}% {fkv/(wkv+wh0):>9.1f}×")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
