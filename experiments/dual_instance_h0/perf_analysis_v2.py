#!/usr/bin/env python3
"""
v2 Performance Analysis: Minimal Embedder A + Full Decoder B.

Key insight: Instance A only needs embed_tokens to produce h^(0).
No transformer layers needed. This changes the entire cost model.
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
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.kv_direct_cache import (
    H0Store,
    _find_inner_model,
    apply_h0_capture_only,
    reconstruct_prefix_kv,
    unpatch_model,
)


def get_mem_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except Exception:
        try:
            return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            return 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--tg-tokens", type=int, default=150)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    cfg = model.args
    n_layers = cfg.num_hidden_layers
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    d_hidden = cfg.hidden_size

    print(f"Model: {n_layers}L, {n_kv_heads} KV heads, d_h={head_dim}, d_model={d_hidden}")

    # Build prompt
    FILLER = ("The development of artificial intelligence has progressed rapidly. "
              "Machine learning algorithms continue to improve across various benchmarks. "
              "Research teams around the world are exploring new architectures. " * 3)
    needle = "The secret project code name is 'AURORA-7732' and it was started on March 15th, 2024."
    prompt_parts = []
    while len(tokenizer.encode("".join(prompt_parts))) < args.prompt_tokens:
        prompt_parts.append(FILLER)
        if len(prompt_parts) == len(prompt_parts) // 3 + 1:
            prompt_parts.append(f"\n[Note] {needle}\n")
    prompt_text = "".join(prompt_parts)
    tokens_list = tokenizer.encode(prompt_text)[:args.prompt_tokens]
    tokens = mx.array(tokens_list)
    n_tokens = len(tokens_list)
    question = "\n\nQuestion: What is the secret project code name?\nAnswer:"
    q_tokens = mx.array(tokenizer.encode(question))

    # Warmup
    warm = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(warm)
    mx.eval(model.parameters())

    inner_model = _find_inner_model(model)

    print(f"Prompt: {n_tokens} tokens, Question: {len(q_tokens)} tokens\n")

    # ================================================================
    # VERIFICATION: bare embed_tokens == h^(0) from full prefill
    # ================================================================
    print("=" * 80)
    print("  VERIFICATION: bare embed_tokens vs captured h^(0)")
    print("=" * 80)

    # Method 1: Bare embed_tokens (no layers, no cache)
    t0 = time.perf_counter()
    h0_bare = inner_model.embed_tokens(tokens.reshape(1, -1))
    mx.eval(h0_bare)
    t_bare = (time.perf_counter() - t0) * 1000

    # Method 2: Full prefill with h^(0) capture
    unpatch_model(model)
    cache = make_prompt_cache(model)
    h0_store = H0Store()
    apply_h0_capture_only(model, h0_store)

    t0 = time.perf_counter()
    out = model(tokens.reshape(1, -1), cache=cache)
    mx.eval(out)
    t_full = (time.perf_counter() - t0) * 1000

    h0_captured = h0_store.get_range(0, h0_store.count)
    mx.eval(h0_captured)

    # Compare
    diff = mx.max(mx.abs(h0_bare.astype(mx.float32) - h0_captured.astype(mx.float32))).item()
    print(f"\n  bare embed_tokens:    {t_bare:.2f} ms  shape={h0_bare.shape}")
    print(f"  full prefill+capture: {t_full:.0f} ms  shape={h0_captured.shape}")
    print(f"  max |diff|:           {diff}")
    print(f"  BIT-EXACT:            {'YES' if diff == 0.0 else 'NO'}")
    print(f"  Speedup (A):          {t_full / t_bare:.0f}×")

    unpatch_model(model)
    del cache, out, h0_store
    gc.collect(); mx.clear_cache()

    # ================================================================
    # COMPONENT MEMORY ANALYSIS
    # ================================================================
    print(f"\n{'='*80}")
    print("  MODEL COMPONENT SIZES")
    print(f"{'='*80}")

    # Measure embed_tokens size using tree_flatten on parameters
    def param_bytes(module):
        """Sum nbytes of all parameter arrays in a module."""
        params = module.parameters() if hasattr(module, 'parameters') else {}
        leaves = nn.utils.tree_flatten(params)
        return sum(v.nbytes for _, v in leaves if hasattr(v, 'nbytes'))

    embed_params = param_bytes(inner_model.embed_tokens)
    layers_params = sum(param_bytes(l) for l in inner_model.layers)
    norm_params = param_bytes(inner_model.norm)
    # lm_head: might be tied with embed_tokens
    lm_head = getattr(model, 'lm_head', None) or getattr(model, 'head', None)
    lm_head_params = param_bytes(lm_head) if lm_head else 0

    total_params = embed_params + layers_params + norm_params + lm_head_params
    print(f"\n  embed_tokens:  {embed_params / 1024 / 1024:>8.1f} MB  ({embed_params / total_params * 100:>5.1f}%)")
    print(f"  layers (×{n_layers}):  {layers_params / 1024 / 1024:>8.1f} MB  ({layers_params / total_params * 100:>5.1f}%)")
    print(f"  norm:          {norm_params / 1024 / 1024:>8.1f} MB  ({norm_params / total_params * 100:>5.1f}%)")
    print(f"  lm_head:       {lm_head_params / 1024 / 1024:>8.1f} MB  ({lm_head_params / total_params * 100:>5.1f}%)")
    print(f"  {'─'*50}")
    print(f"  Total:         {total_params / 1024 / 1024:>8.1f} MB")

    print(f"\n  Instance A (Embedder) needs:  embed_tokens = {embed_params / 1024 / 1024:.1f} MB")
    print(f"  Instance B (Decoder)  needs:  全部 = {total_params / 1024 / 1024:.1f} MB")
    # B needs embed_tokens for TG (embedding new tokens), layers for recon+TG, norm+lm_head for TG output
    b_recon_only = layers_params
    b_tg_needs = total_params  # full model for TG
    print(f"  Instance B (Recon only):      layers = {b_recon_only / 1024 / 1024:.1f} MB")
    print(f"  Instance B (Recon + TG):      全部 = {b_tg_needs / 1024 / 1024:.1f} MB")

    # ================================================================
    # TIMING BREAKDOWN: Optimized A vs Original A
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  TIMING: ORIGINAL vs OPTIMIZED")
    print(f"{'='*80}")

    gc.collect(); mx.clear_cache()

    # Original A: full prefill
    unpatch_model(model)
    cache = make_prompt_cache(model)
    h0_store = H0Store()
    apply_h0_capture_only(model, h0_store)

    t0 = time.perf_counter()
    out = model(tokens.reshape(1, -1), cache=cache)
    mx.eval(out)
    t_original_a = (time.perf_counter() - t0) * 1000

    unpatch_model(model)
    del cache, out, h0_store
    gc.collect(); mx.clear_cache()

    # Optimized A: bare embed only
    t0 = time.perf_counter()
    h0 = inner_model.embed_tokens(tokens.reshape(1, -1))
    mx.eval(h0)
    t_optimized_a = (time.perf_counter() - t0) * 1000

    # Serialization
    t0 = time.perf_counter()
    h0_bytes = bytes(h0.reshape(-1))
    t_serialize = (time.perf_counter() - t0) * 1000
    h0_mb = len(h0_bytes) / 1024 / 1024

    # B: reconstruct
    h0_store_b = H0Store()
    h0_store_b.append(h0)

    gc.collect(); mx.clear_cache()
    mem_before_recon = get_mem_mb()

    t0 = time.perf_counter()
    kv_pairs = reconstruct_prefix_kv(
        inner_model, h0_store_b, 0, h0_store_b.count,
        chunk_size=512, eval_every=8,
    )
    mx.eval(*[k for k, v in kv_pairs] + [v for k, v in kv_pairs])
    t_recon = (time.perf_counter() - t0) * 1000
    mem_after_recon = get_mem_mb()

    # B: inject + question prefill + TG
    cache_b = make_prompt_cache(model)
    for i, (keys, values) in enumerate(kv_pairs):
        cache_b[i].state = (keys, values)
    mx.eval([c.keys for c in cache_b] + [c.values for c in cache_b])
    del kv_pairs
    gc.collect()

    t0 = time.perf_counter()
    q_out = model(q_tokens.reshape(1, -1), cache=cache_b)
    mx.eval(q_out)
    t_q_pp = (time.perf_counter() - t0) * 1000

    y = mx.argmax(q_out[:, -1, :], axis=-1)
    mx.eval(y)
    gen_ids = [y.item()]
    t0 = time.perf_counter()
    for _ in range(args.tg_tokens - 1):
        out = model(y.reshape(1, 1), cache=cache_b)
        mx.eval(out)
        y = mx.argmax(out[:, -1, :], axis=-1)
        mx.eval(y)
        gen_ids.append(y.item())
        if y.item() == tokenizer.eos_token_id:
            break
    t_tg = (time.perf_counter() - t0) * 1000
    n_gen = len(gen_ids)

    answer = tokenizer.decode(gen_ids).strip()

    print(f"\n  Instance A (Prefill node):")
    print(f"    Original (full prefill):  {t_original_a:>8.0f} ms  ({n_tokens / t_original_a * 1000:.0f} tok/s)")
    print(f"    Optimized (embed only):   {t_optimized_a:>8.2f} ms  ({n_tokens / t_optimized_a * 1000:.0f} tok/s)")
    print(f"    Speedup:                  {t_original_a / t_optimized_a:>8.0f}×")
    print(f"    Serialize h^(0):          {t_serialize:>8.2f} ms  ({h0_mb:.1f} MB)")
    print(f"    A total (optimized):      {t_optimized_a + t_serialize:>8.2f} ms")

    print(f"\n  Instance B (Decode node):")
    print(f"    Reconstruct KV:           {t_recon:>8.0f} ms  ({n_tokens / t_recon * 1000:.0f} tok/s)")
    print(f"    Question prefill:         {t_q_pp:>8.0f} ms")
    print(f"    TG ({n_gen} tokens):       {t_tg:>8.0f} ms  ({(n_gen-1) / t_tg * 1000:.1f} tok/s)")
    print(f"    B total:                  {t_recon + t_q_pp + t_tg:>8.0f} ms")

    kv_bytes = 2 * n_layers * n_kv_heads * head_dim * 2 * n_tokens
    kv_mb = kv_bytes / 1024 / 1024
    h0_bytes_val = n_tokens * d_hidden * 2

    # ================================================================
    # REVISED OPTIMAL RATIO
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  REVISED OPTIMAL PREFILL:DECODE RATIO")
    print(f"{'='*80}")

    pp_speed = n_tokens / t_optimized_a * 1000  # tok/s for embed only
    rc_speed = n_tokens / t_recon * 1000
    tg_speed = (n_gen - 1) / t_tg * 1000

    print(f"\n  Measured speeds:")
    print(f"    A (embed only):   {pp_speed:>12,.0f} tok/s")
    print(f"    B (reconstruct):  {rc_speed:>12,.0f} tok/s")
    print(f"    B (TG decode):    {tg_speed:>12,.1f} tok/s")

    print(f"\n  For request with N prompt tokens, M gen tokens:")
    print(f"    T_A = N / {pp_speed:,.0f}")
    print(f"    T_B = N / {rc_speed:,.0f} + M / {tg_speed:.1f}")
    print(f"    P:D = T_A / T_B")

    print(f"\n  {'Scenario':<28} {'N':>6} {'M':>6} {'T_A':>10} {'T_B':>10} {'P:D ratio':>12} {'1P feeds':>10}")
    print(f"  {'─'*82}")

    scenarios = [
        ("Long ctx, short reply", 8192, 128),
        ("Typical chat", 4096, 256),
        ("Standard QA", 4096, 512),
        ("RAG retrieve+gen", 2048, 512),
        ("Long generation", 2048, 2048),
        ("Code generation", 4096, 4096),
        ("Short prompt, long gen", 512, 2048),
        ("128K context", 131072, 256),
    ]

    for name, N, M in scenarios:
        t_a = N / pp_speed
        t_b = N / rc_speed + M / tg_speed
        ratio = t_a / t_b
        feeds = int(1 / ratio) if ratio > 0 else 999999
        print(f"  {name:<28} {N:>6} {M:>6} {t_a*1000:>8.1f}ms {t_b*1000:>8.0f}ms   1 : {1/ratio:>6.0f}     {feeds:>6}D")

    # ================================================================
    # MEMORY BUDGET COMPARISON
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  MEMORY BUDGET: 3 ARCHITECTURES")
    print(f"{'='*80}")

    model_mb = total_params / 1024 / 1024
    embed_mb = embed_params / 1024 / 1024

    print(f"\n  Architecture 1: Traditional single-instance")
    print(f"    Model:     {model_mb:>8.1f} MB")
    print(f"    KV cache:  {kv_mb:>8.1f} MB")
    print(f"    Total:     {model_mb + kv_mb:>8.1f} MB")

    print(f"\n  Architecture 2: h^(0) disaggregated (original — full model on A)")
    a2_a = model_mb + h0_mb
    a2_b = model_mb + kv_mb
    print(f"    A (full model + h0):  {a2_a:>8.1f} MB")
    print(f"    B (full model + KV):  {a2_b:>8.1f} MB")
    print(f"    Transfer:             {h0_mb:>8.1f} MB (vs KV {kv_mb:.1f} MB)")
    print(f"    System total:         {a2_a + a2_b:>8.1f} MB")

    print(f"\n  Architecture 3: h^(0) disaggregated (optimized — embed-only A)")
    a3_a = embed_mb + h0_mb
    a3_b = model_mb + kv_mb
    print(f"    A (embed + h0):       {a3_a:>8.1f} MB  (−{(1-a3_a/a2_a)*100:.0f}% vs Arch 2)")
    print(f"    B (full model + KV):  {a3_b:>8.1f} MB")
    print(f"    Transfer:             {h0_mb:>8.1f} MB")
    print(f"    System total:         {a3_a + a3_b:>8.1f} MB  (−{(1-(a3_a+a3_b)/(a2_a+a2_b))*100:.0f}% vs Arch 2)")

    # Multi-B scenario
    print(f"\n  Architecture 3 with multiple Decoders:")
    for n_b in [4, 8, 16, 32]:
        total = a3_a + n_b * a3_b
        trad = n_b * (model_mb + kv_mb)
        print(f"    1A + {n_b:>2}B:  {total/1024:>6.1f} GB  (traditional {n_b} instances: {trad/1024:.1f} GB)")

    # ================================================================
    # THROUGHPUT PROJECTION
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  THROUGHPUT PROJECTION: 1 Embedder + N Decoders")
    print(f"{'='*80}")

    # For typical chat (N=4096, M=256)
    N, M = 4096, 256
    t_a = N / pp_speed  # seconds per request on A
    t_b = N / rc_speed + M / tg_speed  # seconds per request on B
    a_throughput = 1 / t_a  # requests/s from A
    b_throughput = 1 / t_b  # requests/s per B

    print(f"\n  Scenario: Typical chat (N={N}, M={M})")
    print(f"  A throughput:  {a_throughput:>8.1f} req/s")
    print(f"  B throughput:  {b_throughput:>8.3f} req/s per instance")
    print(f"  B needed to match A: {a_throughput / b_throughput:.0f}")

    print(f"\n  {'#Decoders':>10} {'System req/s':>14} {'Bottleneck':>12} {'A util':>8} {'B util':>8}")
    print(f"  {'─'*52}")
    for n_d in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        sys_throughput = min(a_throughput, n_d * b_throughput)
        bottleneck = "A" if a_throughput < n_d * b_throughput else "B"
        a_util = sys_throughput / a_throughput * 100
        b_util = sys_throughput / (n_d * b_throughput) * 100
        print(f"  {n_d:>10} {sys_throughput:>14.1f} {bottleneck:>12} {a_util:>7.1f}% {b_util:>7.1f}%")

    print(f"\n  Answer: {answer[:100]}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
