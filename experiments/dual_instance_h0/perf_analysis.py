#!/usr/bin/env python3
"""
Detailed performance analysis: PP/TG/TTFT/Memory for single vs dual-instance.
"""

from __future__ import annotations

import gc
import json
import sys
import time

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.kv_direct_cache import (
    H0Store,
    _find_inner_model,
    apply_h0_capture_only,
    reconstruct_prefix_kv,
    unpatch_model,
)
from mlx_lm.sample_utils import make_sampler

GREEDY = make_sampler(temp=0.0)

FILLER_PARA = (
    "The development of artificial intelligence has progressed rapidly in recent years. "
    "Machine learning algorithms continue to improve across various benchmarks. Research "
    "teams around the world are exploring new architectures for language understanding. "
    "The computational requirements for training large models have grown exponentially. "
    "Transfer learning has enabled smaller teams to build on pre-trained foundations. "
    "Ethical considerations remain central to AI development discussions globally. "
)


def get_mem_mb():
    try:
        return mx.metal.get_active_memory() / (1024 * 1024)
    except Exception:
        return 0.0


def get_peak_mb():
    try:
        return mx.metal.get_peak_memory() / (1024 * 1024)
    except Exception:
        return 0.0


def reset_peak():
    try:
        mx.metal.reset_peak_memory()
    except Exception:
        pass


def build_haystack(tokenizer, target_tokens):
    needle_fact = "The secret project code name is 'AURORA-7732' and it was started on March 15th, 2024."
    filler_tokens = len(tokenizer.encode(FILLER_PARA))
    n_paras = (target_tokens // filler_tokens) + 5
    needle_pos = max(1, int(n_paras * 0.30))
    parts = ["Read the following document carefully.\n\n"]
    for i in range(n_paras):
        if i == needle_pos:
            parts.append(f"\n[Important Note] {needle_fact}\n\n")
        parts.append(FILLER_PARA)
    text = "".join(parts)
    tokens = tokenizer.encode(text)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        text = tokenizer.decode(tokens)
    return text, tokens


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--tg-tokens", type=int, default=150)
    args = parser.parse_args()

    model_path = args.model
    target_tokens = args.prompt_tokens
    tg_tokens = args.tg_tokens

    # Load model
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    model_config = model.args
    n_layers = model_config.num_hidden_layers
    n_kv_heads = model_config.num_key_value_heads
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    d_hidden = model_config.hidden_size
    print(f"  Config: {n_layers}L, {n_kv_heads} KV heads, d_h={head_dim}, d_model={d_hidden}")

    # Build prompt
    haystack_text, haystack_tokens_list = build_haystack(tokenizer, target_tokens)
    n_prompt = len(haystack_tokens_list)
    question = "\n\nQuestion: What is the secret project code name and when was it started?\nAnswer:"
    question_tokens_list = tokenizer.encode(question)
    n_question = len(question_tokens_list)

    full_tokens = mx.array(haystack_tokens_list + question_tokens_list)
    haystack_tokens = mx.array(haystack_tokens_list)
    question_tokens = mx.array(question_tokens_list)

    print(f"  Prompt: {n_prompt} tokens, Question: {n_question} tokens")

    # Warmup
    warm = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(warm)
    mx.eval(model.parameters())
    gc.collect(); mx.clear_cache()

    # ===================================================================
    # TEST 1: BASELINE — single instance, standard prefill+generate
    # ===================================================================
    print(f"\n{'='*80}")
    print("  TEST 1: BASELINE (single instance)")
    print(f"{'='*80}")

    gc.collect(); mx.clear_cache()
    reset_peak()
    mem_before = get_mem_mb()

    cache = make_prompt_cache(model)
    mem_after_cache = get_mem_mb()

    # Prefill (haystack + question as one sequence)
    t_pp_start = time.perf_counter()
    model_out = model(full_tokens.reshape(1, -1), cache=cache)
    mx.eval(model_out)
    t_pp = (time.perf_counter() - t_pp_start) * 1000
    mem_after_pp = get_mem_mb()

    # Extract first token (TTFT = prefill time)
    logits = model_out[:, -1, :]
    first_token = mx.argmax(logits, axis=-1)
    mx.eval(first_token)
    t_ttft = t_pp

    # TG loop — manual loop for accurate timing
    gen_token_ids = [first_token.item()]
    y = first_token
    t_tg_start = time.perf_counter()
    for i in range(tg_tokens - 1):
        out = model(y.reshape(1, 1), cache=cache)
        mx.eval(out)
        y = mx.argmax(out[:, -1, :], axis=-1)
        mx.eval(y)
        gen_token_ids.append(y.item())
        if y.item() == tokenizer.eos_token_id:
            break
    t_tg = (time.perf_counter() - t_tg_start) * 1000
    mem_after_tg = get_mem_mb()
    peak_baseline = get_peak_mb()

    baseline_answer = tokenizer.decode(gen_token_ids).strip()

    # KV cache size calculation
    kv_total_tokens = n_prompt + n_question
    kv_bytes = 2 * n_layers * n_kv_heads * head_dim * 2 * kv_total_tokens
    kv_mb = kv_bytes / (1024 * 1024)

    print(f"\n  Prefill:")
    print(f"    Tokens:     {n_prompt + n_question}")
    print(f"    Time:       {t_pp:.0f} ms")
    print(f"    PP speed:   {(n_prompt + n_question) / t_pp * 1000:.0f} tok/s")
    print(f"\n  Token Generation:")
    print(f"    Tokens:     {len(gen_token_ids)}")
    print(f"    Time:       {t_tg:.0f} ms")
    print(f"    TG speed:   {(len(gen_token_ids) - 1) / t_tg * 1000:.1f} tok/s")
    print(f"\n  TTFT:         {t_ttft:.0f} ms")
    print(f"\n  Memory:")
    print(f"    Before:     {mem_before:.0f} MB")
    print(f"    After cache alloc: {mem_after_cache:.0f} MB (+{mem_after_cache - mem_before:.0f} MB)")
    print(f"    After PP:   {mem_after_pp:.0f} MB (+{mem_after_pp - mem_before:.0f} MB)")
    print(f"    After TG:   {mem_after_tg:.0f} MB (+{mem_after_tg - mem_before:.0f} MB)")
    print(f"    Peak:       {peak_baseline:.0f} MB")
    print(f"    KV cache:   {kv_mb:.1f} MB (theoretical)")
    print(f"\n  Answer: {baseline_answer[:120]}")

    # Cleanup
    del cache, model_out, logits
    gc.collect(); mx.clear_cache()

    # ===================================================================
    # TEST 2: DUAL-INSTANCE SIMULATION — h^(0) capture + reconstruct
    # (In single process for fair memory comparison)
    # ===================================================================
    print(f"\n{'='*80}")
    print("  TEST 2: DUAL-INSTANCE (h^(0) capture + reconstruct)")
    print(f"{'='*80}")

    gc.collect(); mx.clear_cache()
    reset_peak()
    mem_before = get_mem_mb()

    # --- Phase A: Prefill with h^(0) capture ---
    print("\n  Phase A: Prefill + h^(0) capture")
    unpatch_model(model)
    cache_a = make_prompt_cache(model)
    h0_store = H0Store()
    apply_h0_capture_only(model, h0_store)

    t_pp_a_start = time.perf_counter()
    model_out = model(haystack_tokens.reshape(1, -1), cache=cache_a)
    mx.eval(model_out)
    t_pp_a = (time.perf_counter() - t_pp_a_start) * 1000

    h0 = h0_store.get_range(0, h0_store.count)
    mx.eval(h0)
    mem_after_capture = get_mem_mb()
    h0_size_mb = h0.nbytes / (1024 * 1024)

    print(f"    PP time:     {t_pp_a:.0f} ms ({n_prompt / t_pp_a * 1000:.0f} tok/s)")
    print(f"    h^(0) shape: {h0.shape}")
    print(f"    h^(0) size:  {h0_size_mb:.2f} MB")
    print(f"    Memory:      {mem_after_capture:.0f} MB")

    # Simulate "transfer": measure serialization cost
    t_serialize_start = time.perf_counter()
    h0_bytes = bytes(h0.reshape(-1))
    t_serialize = (time.perf_counter() - t_serialize_start) * 1000
    transfer_mb = len(h0_bytes) / (1024 * 1024)
    print(f"    Serialize:   {t_serialize:.1f} ms ({transfer_mb:.2f} MB)")

    # --- Phase A cleanup (simulate A freeing resources) ---
    unpatch_model(model)
    del cache_a, model_out
    gc.collect(); mx.clear_cache()
    mem_after_a_cleanup = get_mem_mb()
    print(f"    After A cleanup: {mem_after_a_cleanup:.0f} MB")

    # --- Phase B: Reconstruct + Generate ---
    print("\n  Phase B: Reconstruct + Generate")

    # Simulate "receive": deserialize h^(0)
    import numpy as np
    t_deserialize_start = time.perf_counter()
    np_arr = np.frombuffer(h0_bytes, dtype=np.uint16).reshape(h0.shape)
    h0_received = mx.array(np_arr).view(mx.bfloat16)
    mx.eval(h0_received)
    t_deserialize = (time.perf_counter() - t_deserialize_start) * 1000
    print(f"    Deserialize: {t_deserialize:.1f} ms")

    # Populate H0Store
    h0_store_b = H0Store()
    h0_store_b.append(h0_received)
    mem_after_h0_load = get_mem_mb()

    # Reconstruct KV
    inner_model = _find_inner_model(model)
    reset_peak()
    t_recon_start = time.perf_counter()
    kv_pairs = reconstruct_prefix_kv(
        inner_model, h0_store_b, 0, h0_store_b.count,
        chunk_size=512, eval_every=8,
    )
    mx.eval(*[k for k, v in kv_pairs] + [v for k, v in kv_pairs])
    t_recon = (time.perf_counter() - t_recon_start) * 1000
    mem_after_recon = get_mem_mb()
    peak_recon = get_peak_mb()

    # Compute reconstructed KV size
    recon_kv_bytes = sum(k.nbytes + v.nbytes for k, v in kv_pairs)
    recon_kv_mb = recon_kv_bytes / (1024 * 1024)

    print(f"    Recon time:  {t_recon:.0f} ms ({n_prompt / t_recon * 1000:.0f} tok/s)")
    print(f"    KV rebuilt:  {recon_kv_mb:.1f} MB ({len(kv_pairs)} layers)")
    print(f"    Memory:      {mem_after_recon:.0f} MB (peak: {peak_recon:.0f} MB)")

    # Inject KV into standard cache
    cache_b = make_prompt_cache(model)
    for i, (keys, values) in enumerate(kv_pairs):
        cache_b[i].state = (keys, values)
    mx.eval([c.keys for c in cache_b] + [c.values for c in cache_b])
    del kv_pairs  # free duplicate
    gc.collect()
    mem_after_inject = get_mem_mb()
    print(f"    After inject: {mem_after_inject:.0f} MB")

    # TTFT: everything up to first token
    t_ttft_dual = t_pp_a + t_serialize + t_deserialize + t_recon

    # Process question (additional prefill for question tokens)
    t_q_start = time.perf_counter()
    q_out = model(question_tokens.reshape(1, -1), cache=cache_b)
    mx.eval(q_out)
    t_q_prefill = (time.perf_counter() - t_q_start) * 1000

    logits = q_out[:, -1, :]
    first_token = mx.argmax(logits, axis=-1)
    mx.eval(first_token)
    t_ttft_dual += t_q_prefill

    # TG loop — manual loop
    gen_token_ids_dual = [first_token.item()]
    y = first_token
    t_tg_start = time.perf_counter()
    for i in range(tg_tokens - 1):
        out = model(y.reshape(1, 1), cache=cache_b)
        mx.eval(out)
        y = mx.argmax(out[:, -1, :], axis=-1)
        mx.eval(y)
        gen_token_ids_dual.append(y.item())
        if y.item() == tokenizer.eos_token_id:
            break
    t_tg_dual = (time.perf_counter() - t_tg_start) * 1000
    mem_after_tg_dual = get_mem_mb()
    peak_dual = get_peak_mb()

    dual_answer = tokenizer.decode(gen_token_ids_dual).strip()

    print(f"\n    Question PP: {t_q_prefill:.0f} ms ({n_question} tokens)")
    print(f"    TG time:     {t_tg_dual:.0f} ms")
    print(f"    TG speed:    {(len(gen_token_ids) - 1) / t_tg_dual * 1000:.1f} tok/s")
    print(f"    TTFT total:  {t_ttft_dual:.0f} ms")
    print(f"    Memory:      {mem_after_tg_dual:.0f} MB (peak: {peak_dual:.0f} MB)")
    print(f"\n    Answer: {dual_answer[:120]}")

    # ===================================================================
    # COMPARISON
    # ===================================================================
    print(f"\n{'='*80}")
    print("  PERFORMANCE COMPARISON")
    print(f"{'='*80}")

    exact_match = baseline_answer.strip() == dual_answer.strip()

    print(f"\n  {'Metric':<30} {'Baseline':>15} {'Dual h^(0)':>15} {'Delta':>12}")
    print(f"  {'-'*72}")

    # PP
    pp_baseline = (n_prompt + n_question) / t_pp * 1000
    pp_dual_a = n_prompt / t_pp_a * 1000
    print(f"  {'PP speed (tok/s)':<30} {pp_baseline:>15.0f} {pp_dual_a:>15.0f} {'':>12}")

    # TG
    tg_baseline = (len(gen_token_ids) - 1) / t_tg * 1000 if t_tg > 0 else 0
    tg_dual = (len(gen_token_ids_dual) - 1) / t_tg_dual * 1000 if t_tg_dual > 0 else 0
    tg_delta = (tg_dual - tg_baseline) / tg_baseline * 100 if tg_baseline > 0 else 0
    print(f"  {'TG speed (tok/s)':<30} {tg_baseline:>15.1f} {tg_dual:>15.1f} {tg_delta:>+11.1f}%")

    # TTFT
    ttft_delta = (t_ttft_dual - t_ttft) / t_ttft * 100
    print(f"  {'TTFT (ms)':<30} {t_ttft:>15.0f} {t_ttft_dual:>15.0f} {ttft_delta:>+11.1f}%")

    # State transfer
    print(f"  {'':>0}")
    print(f"  {'State size (MB)':<30} {kv_mb:>15.1f} {transfer_mb:>15.2f} {-(1 - transfer_mb/kv_mb)*100:>+11.1f}%")
    print(f"  {'Compression ratio':<30} {'1.0×':>15} {kv_mb/transfer_mb:>14.1f}× {'':>12}")

    # Memory
    print(f"  {'':>0}")
    print(f"  {'KV cache (MB)':<30} {kv_mb:>15.1f} {recon_kv_mb:>15.1f} {'':>12}")
    print(f"  {'Peak memory (MB)':<30} {peak_baseline:>15.0f} {peak_dual:>15.0f} {'':>12}")

    # Timing breakdown
    print(f"\n  Dual-instance timing breakdown:")
    total_dual = t_pp_a + t_serialize + t_deserialize + t_recon + t_q_prefill + t_tg_dual
    print(f"    A: Prefill          {t_pp_a:>8.0f} ms  ({t_pp_a/total_dual*100:>5.1f}%)")
    print(f"    Transfer serialize  {t_serialize:>8.1f} ms  ({t_serialize/total_dual*100:>5.1f}%)")
    print(f"    Transfer deserial.  {t_deserialize:>8.1f} ms  ({t_deserialize/total_dual*100:>5.1f}%)")
    print(f"    B: Reconstruct      {t_recon:>8.0f} ms  ({t_recon/total_dual*100:>5.1f}%)")
    print(f"    B: Question PP      {t_q_prefill:>8.0f} ms  ({t_q_prefill/total_dual*100:>5.1f}%)")
    print(f"    B: TG decode        {t_tg_dual:>8.0f} ms  ({t_tg_dual/total_dual*100:>5.1f}%)")
    print(f"    {'─'*45}")
    print(f"    Total               {total_dual:>8.0f} ms")
    baseline_total = t_pp + t_tg
    print(f"    Baseline total      {baseline_total:>8.0f} ms")

    # Disaggregated analysis
    print(f"\n  Disaggregated serving analysis (A and B in parallel):")
    # If A and B are pipelined: B starts reconstruct as soon as A finishes transfer
    pipeline_ttft = max(t_pp_a + t_serialize, 0) + t_deserialize + t_recon + t_q_prefill
    pipeline_total = pipeline_ttft + t_tg_dual
    print(f"    Pipeline TTFT:      {pipeline_ttft:>8.0f} ms  (A prefill hidden by B model load)")
    print(f"    Pipeline total:     {pipeline_total:>8.0f} ms")
    print(f"    vs Baseline:        {baseline_total:>8.0f} ms  ({(pipeline_total-baseline_total)/baseline_total*100:>+.1f}%)")

    # Network transfer analysis
    print(f"\n  Network transfer analysis (if A/B on different nodes):")
    for bw_gbps in [1, 10, 25, 100]:
        kv_transfer_ms = kv_mb / (bw_gbps * 1000 / 8) * 1000
        h0_transfer_ms = transfer_mb / (bw_gbps * 1000 / 8) * 1000
        print(f"    {bw_gbps:>3} Gbps:  KV={kv_transfer_ms:>8.1f} ms  h^(0)={h0_transfer_ms:>8.1f} ms  "
              f"saving={kv_transfer_ms - h0_transfer_ms:>8.1f} ms ({(1-h0_transfer_ms/kv_transfer_ms)*100:.0f}%)")

    # Scaling analysis
    print(f"\n  Scaling projection:")
    print(f"  {'Model':<25} {'Layers':>6} {'d_model':>8} {'KV/tok':>10} {'RS/tok':>10} {'Ratio':>8}")
    print(f"  {'-'*67}")
    models = [
        ("Qwen3-1.7B", n_layers, d_hidden, n_kv_heads, head_dim),
        ("Qwen3-8B", 36, 4096, 4, 128),
        ("Qwen3-32B", 64, 5120, 8, 128),
        ("Qwen3-235B (MoE)", 94, 4096, 4, 128),
        ("Hypothetical 1T", 96, 16384, 8, 128),
    ]
    for name, L, d, nkv, dh in models:
        kv_per_tok = 2 * L * nkv * dh * 2
        rs_per_tok = d * 2
        ratio = kv_per_tok / rs_per_tok
        print(f"  {name:<25} {L:>6} {d:>8} {kv_per_tok:>9,} {rs_per_tok:>9,} {ratio:>7.0f}×")

    print(f"\n  Quality: {'EXACT MATCH' if exact_match else 'MISMATCH'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
