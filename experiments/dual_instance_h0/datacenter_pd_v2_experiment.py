#!/usr/bin/env python3
"""
Datacenter PD Disaggregation v2 — Advanced Experiments on Qwen3-1.7B.

Extends v1 (Qwen3-8B, D1-D5) with throughput-focused analysis:
  D6:  Batch throughput analysis (h^(0) → larger batch sizes)
  D7:  D-node compute idle utilization during decode
  D8:  Sparse checkpoint parallel reconstruction
  D9:  Long context KV explosion (512→16K sweep)
  D10: Roofline characterization + GQA comparison
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
from mlx_lm.models.kv_direct_cache import (
    _find_inner_model,
    ResidualCheckpointStore,
    reconstruct_from_checkpoint,
)
from mlx_lm.sample_utils import make_sampler
from mlx.utils import tree_flatten

GREEDY = make_sampler(temp=0.0)

DIVERSE_PROMPT = """The history of artificial intelligence began in antiquity, with myths and stories of artificial beings endowed with intelligence by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe human thinking as mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s. The field of AI research was founded at Dartmouth College in 1956. Many predicted a machine as intelligent as a human would exist within a generation. Eventually it became obvious that researchers had grossly underestimated the difficulty. In 1973, governments stopped funding undirected AI research, leading to the first AI winter. Investment boomed again in the 21st century when machine learning was successfully applied to many problems due to new methods, the application of powerful computer hardware, and the collection of immense data sets. Deep learning transformed the field starting around 2012 when neural networks began dramatically outperforming other methods."""

# Hardware constants (Mac mini M4 Pro)
PEAK_TFLOPS = 27.0    # bf16
PEAK_BW_GBS = 273.0   # GB/s unified memory


# ═══════════════════════════════════════════════════════════
# Theory helpers (self-contained, copied from v1)
# ═══════════════════════════════════════════════════════════

def per_layer_prefill_flops(cfg, seq_len):
    d = cfg['hidden_size']
    n_q = cfg['num_attention_heads']
    n_kv = cfg['num_key_value_heads']
    hd = d // n_q
    inter = cfg.get('intermediate_size', d * 4)
    N = seq_len
    qkv = 2 * N * d * (d + 2 * n_kv * hd)
    o_proj = 2 * N * d * d
    attn = 2 * n_q * N * N * hd * 2
    ffn = 2 * 3 * N * d * inter
    return qkv + o_proj + attn + ffn


def per_layer_decode_flops(cfg, seq_len):
    d = cfg['hidden_size']
    n_q = cfg['num_attention_heads']
    n_kv = cfg['num_key_value_heads']
    hd = d // n_q
    inter = cfg.get('intermediate_size', d * 4)
    qkv = 2 * d * (d + 2 * n_kv * hd)
    o_proj = 2 * d * d
    attn = 2 * n_q * seq_len * hd * 2
    ffn = 2 * 3 * d * inter
    return qkv + o_proj + attn + ffn


def per_layer_weight_bytes(cfg):
    d = cfg['hidden_size']
    n_kv = cfg['num_key_value_heads']
    hd = d // cfg['num_attention_heads']
    inter = cfg.get('intermediate_size', d * 4)
    bpp = 0.6  # Q4 effective bytes/param
    qkv = d * (d + 2 * n_kv * hd) * bpp
    o_proj = d * d * bpp
    ffn = 3 * d * inter * bpp
    norms = 2 * d * 2
    return qkv + o_proj + ffn + norms


def kv_read_bytes_per_layer(cfg, seq_len):
    n_kv = cfg['num_key_value_heads']
    hd = cfg['hidden_size'] // cfg['num_attention_heads']
    return 2 * n_kv * seq_len * hd * 2


def kv_per_token_per_layer_bytes(cfg):
    n_kv = cfg['num_key_value_heads']
    hd = cfg['hidden_size'] // cfg['num_attention_heads']
    return 2 * n_kv * hd * 2


# ═══════════════════════════════════════════════════════════
# Measurement helpers
# ═══════════════════════════════════════════════════════════

def measure_per_layer_prefill(inner, tokens_mx, n_layers):
    N = tokens_mx.shape[0]
    x = inner.embed_tokens(tokens_mx.reshape(1, -1))
    mx.eval(x)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(x.dtype)
    residuals = [x]
    cache = [KVCache() for _ in range(n_layers)]
    per_layer_ms = []
    for i, layer in enumerate(inner.layers):
        t0 = time.perf_counter()
        x = layer(x, mask=mask, cache=cache[i])
        mx.eval(x)
        per_layer_ms.append((time.perf_counter() - t0) * 1000)
        residuals.append(x)
    return residuals, cache, per_layer_ms


def measure_decode_speed(model, tokenizer, cache, question, n_tokens=100):
    q = mx.array(tokenizer.encode(question))
    times = []
    t_start = time.perf_counter()
    for tok, _ in generate_step(q, model, max_tokens=n_tokens,
                                 sampler=GREEDY, prompt_cache=cache):
        mx.eval(mx.array([tok]))
        t_now = time.perf_counter()
        times.append((t_now - t_start) * 1000)
        t_start = t_now
    return times


def extract_clean_kv(cache, n_layers, N):
    clean_kv = []
    for i in range(n_layers):
        k, v = cache[i].state
        mx.eval(k, v)
        clean_kv.append((k[:, :, :N, :], v[:, :, :N, :]))
        mx.eval(clean_kv[-1][0], clean_kv[-1][1])
    return clean_kv


def make_cache_from_kv(clean_kv, n_layers):
    cache = [KVCache() for _ in range(n_layers)]
    for i in range(n_layers):
        cache[i].state = clean_kv[i]
    mx.eval([cache[i].keys for i in range(n_layers)])
    return cache


def quality_str(baseline, generated):
    match = sum(1 for a, b in zip(baseline, generated) if a == b)
    total = min(len(baseline), len(generated))
    if baseline == generated:
        return f"EXACT ({total} tok)"
    fd = next((i for i, (a, b) in enumerate(zip(baseline, generated)) if a != b), total)
    return f"{match}/{total} (div@{fd})"


# ═══════════════════════════════════════════════════════════
# D6: Batch Throughput Analysis
# ═══════════════════════════════════════════════════════════

def run_d6(cfg, n_layers, model_bytes):
    print(f"\n{'═' * 78}")
    print(f"  D6: Batch Throughput 分析 — h^(0) 释放 HBM 容量")
    print(f"{'═' * 78}")

    d = cfg['hidden_size']
    kv_ptpl = kv_per_token_per_layer_bytes(cfg)
    h0_per_token = d * 2  # bf16

    HBM_SIZES = [
        ("Mac mini 48GB", 48e9),
        ("A100 80GB", 80e9),
        ("H100 80GB", 80e9),
    ]
    CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072]

    # bf16 model weights for datacenter
    bf16_weight_bytes = model_bytes * 4  # Q4→bf16 roughly 4×
    q4_weight_bytes = model_bytes

    for hbm_name, hbm in HBM_SIZES[:2]:  # Mac mini + A100
        wb = q4_weight_bytes if "Mac" in hbm_name else bf16_weight_bytes
        available = hbm - wb
        wb_label = "Q4" if "Mac" in hbm_name else "bf16"

        print(f"\n  {hbm_name} (weights={wb/1e9:.1f}GB {wb_label}, "
              f"available={available/1e9:.1f}GB)")
        print(f"\n  {'Context':>8s} │ {'Full KV':>10s} │ {'batch':>6s} │ "
              f"{'h^(0)+512':>10s} │ {'batch':>6s} │ "
              f"{'Pipe@14':>10s} │ {'batch':>6s} │ {'h0/Full':>7s}")
        print(f"  {'─'*8}─┼{'─'*10}─┼{'─'*6}─┼{'─'*10}─┼{'─'*6}─┼{'─'*10}─┼{'─'*6}─┼{'─'*7}")

        for ctx in CONTEXTS:
            # Full KV (DistServe): all layers stored
            full_kv = kv_ptpl * n_layers * ctx
            batch_full = max(1, int(available / full_kv))

            # h^(0) only: store h^(0) archive + 512-token recent KV for decode
            h0_archive = h0_per_token * ctx
            recent_kv = kv_ptpl * n_layers * min(512, ctx)
            h0_total = h0_archive + recent_kv
            batch_h0 = max(1, int(available / h0_total))

            # Pipeline@14: D-node stores 14 layers of KV
            pipe_kv = kv_ptpl * (n_layers // 2) * ctx
            batch_pipe = max(1, int(available / pipe_kv))

            ratio = batch_h0 / batch_full if batch_full > 0 else float('inf')

            print(f"  {ctx:>8,d} │ {full_kv/1e6:>8.0f} MB │ {batch_full:>6,d} │ "
                  f"{h0_total/1e6:>8.0f} MB │ {batch_h0:>6,d} │ "
                  f"{pipe_kv/1e6:>8.0f} MB │ {batch_pipe:>6,d} │ {ratio:>6.1f}×")

    # Key insight
    print(f"\n  注意: h^(0) 方案中 D 节点只存 h^(0) 归档 + 512-token 滑动窗口 KV。")
    print(f"  Decode 时按需从 h^(0) 重建被 evict 的 KV。")
    print(f"  这利用了 D 节点 decode 阶段的闲置 compute (见 D7)。")


# ═══════════════════════════════════════════════════════════
# D7: D-Node Compute Idle Utilization
# ═══════════════════════════════════════════════════════════

def run_d7(inner, model, tokenizer, cfg, tokens_mx, N, n_layers, question,
           residuals=None, cache=None):
    print(f"\n{'═' * 78}")
    print(f"  D7: D 节点 Compute 闲置利用分析")
    print(f"{'═' * 78}")

    # ── Part A: Measure actual utilization ──
    if residuals is None or cache is None:
        gc.collect()
        residuals, cache, prefill_layer_ms = measure_per_layer_prefill(
            inner, tokens_mx, n_layers)
    else:
        # Quick re-measure prefill for timing
        gc.collect()
        _, _, prefill_layer_ms = measure_per_layer_prefill(inner, tokens_mx, n_layers)

    prefill_total_ms = sum(prefill_layer_ms)

    # Prefill compute
    theory_prefill_flops = per_layer_prefill_flops(cfg, N) * n_layers
    achieved_prefill_tflops = theory_prefill_flops / (prefill_total_ms / 1000) / 1e12
    prefill_compute_util = achieved_prefill_tflops / PEAK_TFLOPS * 100

    # Prefill bandwidth
    prefill_weight_bytes = per_layer_weight_bytes(cfg) * n_layers
    prefill_bw = prefill_weight_bytes / (prefill_total_ms / 1000) / 1e9
    prefill_bw_util = prefill_bw / PEAK_BW_GBS * 100

    print(f"\n  Part A: 实测利用率 (context={N})")
    print(f"\n  {'Phase':12s} │ {'TFLOPS':>9s} │ {'Compute%':>9s} │ "
          f"{'BW GB/s':>9s} │ {'BW%':>6s} │ {'瓶颈':>12s}")
    print(f"  {'─'*12}─┼{'─'*9}─┼{'─'*9}─┼{'─'*9}─┼{'─'*6}─┼{'─'*12}")

    print(f"  {'Prefill':12s} │ {achieved_prefill_tflops:>8.2f}T │ "
          f"{prefill_compute_util:>8.1f}% │ {prefill_bw:>8.1f}  │ "
          f"{prefill_bw_util:>5.1f}% │ {'计算密集':>12s}")

    # Decode
    gc.collect()
    clean_kv = extract_clean_kv(cache, n_layers, N)
    decode_cache = make_cache_from_kv(clean_kv, n_layers)
    decode_times = measure_decode_speed(model, tokenizer, decode_cache, question, 80)
    steady = decode_times[3:] if len(decode_times) > 3 else decode_times[1:]
    decode_avg_ms = np.mean(steady)

    seq_ctx = N + 40
    decode_flops_total = per_layer_decode_flops(cfg, seq_ctx) * n_layers
    decode_weight = per_layer_weight_bytes(cfg) * n_layers
    decode_kv = kv_read_bytes_per_layer(cfg, seq_ctx) * n_layers
    decode_bytes_total = decode_weight + decode_kv

    achieved_decode_tflops = decode_flops_total / (decode_avg_ms / 1000) / 1e12
    achieved_decode_bw = decode_bytes_total / (decode_avg_ms / 1000) / 1e9
    decode_compute_util = achieved_decode_tflops / PEAK_TFLOPS * 100
    decode_bw_util = achieved_decode_bw / PEAK_BW_GBS * 100

    print(f"  {'Decode':12s} │ {achieved_decode_tflops:>7.4f}T │ "
          f"{decode_compute_util:>8.3f}% │ {achieved_decode_bw:>8.1f}  │ "
          f"{decode_bw_util:>5.1f}% │ {'访存密集':>12s}")

    # AI gap
    prefill_ai = theory_prefill_flops / prefill_weight_bytes
    decode_ai = decode_flops_total / decode_bytes_total

    print(f"\n  Arithmetic Intensity:")
    print(f"    Prefill:  {prefill_ai:>10,.0f} FLOPs/Byte")
    print(f"    Decode:   {decode_ai:>10.1f} FLOPs/Byte")
    print(f"    Gap:      {prefill_ai/decode_ai:>10,.0f}×")

    # ── Part B: Idle compute budget ──
    idle_compute_pct = 100.0 - decode_compute_util
    idle_tflops = PEAK_TFLOPS * (idle_compute_pct / 100)

    # How much reconstruction can fit in one decode step?
    chunk_size = 512
    recon_flops_chunk = per_layer_prefill_flops(cfg, chunk_size) * n_layers
    recon_time_dedicated_ms = recon_flops_chunk / (PEAK_TFLOPS * 1e12) * 1000
    recon_time_idle_ms = recon_flops_chunk / (idle_tflops * 1e12) * 1000

    chunks_per_step = decode_avg_ms / recon_time_idle_ms
    tokens_per_step = chunks_per_step * chunk_size

    print(f"\n  Part B: 闲置 Compute 预算")
    print(f"    Decode 每步: {decode_avg_ms:.2f} ms")
    print(f"    闲置 compute: {idle_tflops:.3f} TFLOPS ({idle_compute_pct:.2f}%)")
    print(f"    重建 {chunk_size} tokens (全 {n_layers}L): "
          f"{recon_flops_chunk/1e9:.1f} GFLOPs")
    print(f"    专用重建时间: {recon_time_dedicated_ms:.2f} ms")
    print(f"    用闲置 compute: {recon_time_idle_ms:.2f} ms")
    print(f"    每个 decode step 可重建: {chunks_per_step:.0f} chunks = "
          f"{tokens_per_step:,.0f} tokens")
    print(f"\n    结论: D 节点每个 decode step 理论上可用闲置 compute")
    print(f"    重建 ~{tokens_per_step:,.0f} tokens 的 KV (from h^(0))。")
    print(f"    实际受限于 MLX eager 模式的调度能力。")

    return prefill_total_ms, decode_avg_ms, residuals, clean_kv


# ═══════════════════════════════════════════════════════════
# D8: Sparse Checkpoint Parallel Reconstruction
# ═══════════════════════════════════════════════════════════

def run_d8(inner, model, tokenizer, cfg, n_layers, question, gen_tokens):
    print(f"\n{'═' * 78}")
    print(f"  D8: Sparse Checkpoint 并行重建")
    print(f"{'═' * 78}")

    d = cfg['hidden_size']
    kv_ptpl = kv_per_token_per_layer_bytes(cfg)
    h0_per_tok = d * 2

    # Auto-generate sparse configs based on layer count
    mid = n_layers // 2
    quarter = n_layers // 4
    CONFIGS = [
        ("h^(0) only", [0]),
        (f"[0,{mid}]", [0, mid]),
        (f"[0,{quarter},{mid},{mid+quarter}]", [0, quarter, mid, mid + quarter]),
    ]

    for ctx in [2048, 4096]:
        prompt_text = DIVERSE_PROMPT * (ctx // 100 + 1)
        tokens = tokenizer.encode(prompt_text)[:ctx]
        N = len(tokens)
        tok_mx = mx.array(tokens)

        # Full prefill to capture residuals
        gc.collect()
        residuals, cache, _ = measure_per_layer_prefill(inner, tok_mx, n_layers)
        clean_kv = extract_clean_kv(cache, n_layers, N)

        # Baseline tokens
        baseline_cache = make_cache_from_kv(clean_kv, n_layers)
        baseline_tok = []
        for tok, _ in generate_step(mx.array(tokenizer.encode(question)), model,
                                     max_tokens=gen_tokens, sampler=GREEDY,
                                     prompt_cache=baseline_cache):
            baseline_tok.append(tok)

        print(f"\n  Context = {N} tokens")
        print(f"  {'Config':20s} │ {'Groups':25s} │ {'Per-Group ms':25s} │ "
              f"{'Seq ms':>7s} │ {'Par ms':>7s} │ {'Speedup':>7s} │ {'Quality':>10s}")
        print(f"  {'─'*20}─┼{'─'*25}─┼{'─'*25}─┼{'─'*7}─┼{'─'*7}─┼{'─'*7}─┼{'─'*10}")

        for name, ckpt_layers in CONFIGS:
            gc.collect()
            store = ResidualCheckpointStore(checkpoint_layers=ckpt_layers)
            for l in ckpt_layers:
                store.append(l, residuals[l])

            groups = store.reconstruction_groups(n_layers)
            group_times = []

            for start_l, end_l in groups:
                gc.collect()
                # Build temp cache for this group
                temp_cache = [KVCache() for _ in range(end_l - start_l)]
                ckpt_layer = store.nearest_checkpoint(start_l)
                h = store.get_range(ckpt_layer, 0, N)
                mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(h.dtype)

                t0 = time.perf_counter()
                x = h
                for idx, layer_idx in enumerate(range(ckpt_layer, end_l)):
                    x = inner.layers[layer_idx](x, mask=mask, cache=temp_cache[idx])
                    if (idx + 1) % 4 == 0 or idx == (end_l - ckpt_layer - 1):
                        mx.eval(x)
                mx.eval(*[tc.keys for tc in temp_cache], *[tc.values for tc in temp_cache])
                elapsed = (time.perf_counter() - t0) * 1000

                group_times.append((start_l, end_l, elapsed))

            seq_total = sum(t for _, _, t in group_times)
            par_total = max(t for _, _, t in group_times)
            speedup = seq_total / par_total

            # Quality check: reconstruct full cache from h^(0)
            gc.collect()
            full_recon_cache = [KVCache() for _ in range(n_layers)]
            h0 = residuals[0]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(h0.dtype)
            xr = h0
            for i in range(n_layers):
                xr = inner.layers[i](xr, mask=mask, cache=full_recon_cache[i])
                if (i + 1) % 4 == 0 or i == n_layers - 1:
                    mx.eval(xr)

            recon_tok = []
            for tok, _ in generate_step(mx.array(tokenizer.encode(question)), model,
                                         max_tokens=gen_tokens, sampler=GREEDY,
                                         prompt_cache=full_recon_cache):
                recon_tok.append(tok)
            qs = quality_str(baseline_tok, recon_tok)

            groups_str = " ".join(f"[{s}→{e})" for s, e, _ in group_times)
            times_str = " ".join(f"{t:.0f}" for _, _, t in group_times)

            print(f"  {name:20s} │ {groups_str:25s} │ {times_str:25s} │ "
                  f"{seq_total:>6.0f}  │ {par_total:>6.0f}  │ {speedup:>6.2f}× │ {qs:>10s}")

        # Transfer overhead for parallel deployment
        print(f"\n  跨设备传输开销 (每个 h checkpoint = {N*d*2/1e6:.1f} MB):")
        h_bytes = N * d * 2
        for bw_name, bw in [("NVLink 900GB/s", 900e9), ("IB NDR 50GB/s", 50e9),
                             ("IB HDR 25GB/s", 25e9)]:
            xfer_ms = h_bytes / bw * 1000
            print(f"    {bw_name:20s}: {xfer_ms:.3f} ms per checkpoint")


# ═══════════════════════════════════════════════════════════
# D9: Long Context KV Explosion
# ═══════════════════════════════════════════════════════════

def run_d9(inner, model, tokenizer, cfg, n_layers, model_bytes, question, gen_tokens):
    print(f"\n{'═' * 78}")
    print(f"  D9: Long Context KV 爆炸")
    print(f"{'═' * 78}")

    d = cfg['hidden_size']
    n_q = cfg['num_attention_heads']
    hd = d // n_q
    kv_ptpl = kv_per_token_per_layer_bytes(cfg)
    h0_per_tok = d * 2

    CONTEXTS = [512, 1024, 2048, 4096, 8192]
    try_16k = False  # skip 16K by default

    print(f"\n  {'Ctx':>6s} │ {'Prefill':>8s} │ {'Decode':>8s} │ {'Recon':>8s} │ "
          f"{'Full KV':>8s} │ {'h^(0)':>7s} │ {'Ratio':>6s} │ {'Attn%':>6s} │ "
          f"{'KV>Mdl?':>7s} │ {'Quality':>12s}")
    print(f"  {'─'*6}─┼{'─'*8}─┼{'─'*8}─┼{'─'*8}─┼{'─'*8}─┼{'─'*7}─┼{'─'*6}─┼"
          f"{'─'*6}─┼{'─'*7}─┼{'─'*12}")

    for ctx in CONTEXTS + ([16384] if try_16k else []):
        prompt_text = DIVERSE_PROMPT * (ctx // 100 + 1)
        tokens = tokenizer.encode(prompt_text)[:ctx]
        N = len(tokens)
        tok_mx = mx.array(tokens)

        gc.collect()

        try:
            # Prefill
            residuals, cache, layer_ms = measure_per_layer_prefill(inner, tok_mx, n_layers)
            prefill_ms = sum(layer_ms)
            clean_kv = extract_clean_kv(cache, n_layers, N)

            # Baseline generation
            base_cache = make_cache_from_kv(clean_kv, n_layers)
            baseline_tok = []
            for tok, _ in generate_step(mx.array(tokenizer.encode(question)), model,
                                         max_tokens=gen_tokens, sampler=GREEDY,
                                         prompt_cache=base_cache):
                baseline_tok.append(tok)

            # Decode speed
            gc.collect()
            dec_cache = make_cache_from_kv(clean_kv, n_layers)
            dec_times = measure_decode_speed(model, tokenizer, dec_cache, question, 30)
            steady = dec_times[3:] if len(dec_times) > 3 else dec_times[1:]
            decode_ms = np.mean(steady)

            # h^(0) reconstruction
            gc.collect()
            recon_cache = [KVCache() for _ in range(n_layers)]
            h0 = residuals[0]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(h0.dtype)
            t0 = time.perf_counter()
            xr = h0
            for i in range(n_layers):
                xr = inner.layers[i](xr, mask=mask, cache=recon_cache[i])
                if (i + 1) % 4 == 0 or i == n_layers - 1:
                    mx.eval(xr)
            recon_ms = (time.perf_counter() - t0) * 1000

            # Quality
            recon_tok = []
            for tok, _ in generate_step(mx.array(tokenizer.encode(question)), model,
                                         max_tokens=gen_tokens, sampler=GREEDY,
                                         prompt_cache=recon_cache):
                recon_tok.append(tok)
            qs = quality_str(baseline_tok, recon_tok)

            # Memory
            full_kv_mb = (kv_ptpl * n_layers * N) / 1e6
            h0_mb = (h0_per_tok * N) / 1e6
            ratio = full_kv_mb / h0_mb
            model_mb = model_bytes / 1e6
            kv_exceeds = "YES" if full_kv_mb > model_mb else "no"

            # Attention quadratic share
            attn_flops = 2 * n_q * N * N * hd * 2 * n_layers
            total_flops = per_layer_prefill_flops(cfg, N) * n_layers
            attn_pct = attn_flops / total_flops * 100

            print(f"  {N:>6,d} │ {prefill_ms:>7.0f}ms │ {decode_ms:>6.1f}ms │ "
                  f"{recon_ms:>7.0f}ms │ {full_kv_mb:>6.0f}MB │ {h0_mb:>5.0f}MB │ "
                  f"{ratio:>5.0f}× │ {attn_pct:>5.1f}% │ {kv_exceeds:>7s} │ {qs:>12s}")

        except Exception as e:
            print(f"  {ctx:>6,d} │ {'OOM/ERR':>8s} │ {str(e)[:50]}")
            try_16k = False
            continue

    # Summary
    print(f"\n  模型权重: {model_bytes/1e6:.0f} MB")
    crossover = model_bytes / (kv_ptpl * n_layers)
    print(f"  KV 超过模型权重的 context 长度: ~{crossover:,.0f} tokens")
    print(f"  h^(0) 压缩比: {n_layers * kv_ptpl / (d * 2):.0f}× (与 context 无关)")


# ═══════════════════════════════════════════════════════════
# D10: Roofline Characterization
# ═══════════════════════════════════════════════════════════

def run_d10(inner, model, tokenizer, cfg, tokens_mx, N, n_layers, question,
            prefill_total_ms=None, decode_avg_ms=None, clean_kv=None):
    print(f"\n{'═' * 78}")
    print(f"  D10: Roofline 特征 + GQA 对比")
    print(f"{'═' * 78}")

    ridge = PEAK_TFLOPS * 1e12 / (PEAK_BW_GBS * 1e9)

    # ── Measure if not provided ──
    if prefill_total_ms is None:
        gc.collect()
        _, cache, layer_ms = measure_per_layer_prefill(inner, tokens_mx, n_layers)
        prefill_total_ms = sum(layer_ms)
        clean_kv = extract_clean_kv(cache, n_layers, N)

    if decode_avg_ms is None:
        gc.collect()
        dec_cache = make_cache_from_kv(clean_kv, n_layers)
        dec_times = measure_decode_speed(model, tokenizer, dec_cache, question, 80)
        steady = dec_times[3:] if len(dec_times) > 3 else dec_times[1:]
        decode_avg_ms = np.mean(steady)

    # ── Qwen3-1.7B metrics ──
    theory_prefill = per_layer_prefill_flops(cfg, N) * n_layers
    prefill_weight = per_layer_weight_bytes(cfg) * n_layers
    prefill_ai = theory_prefill / prefill_weight

    seq_ctx = N + 40
    decode_flops = per_layer_decode_flops(cfg, seq_ctx) * n_layers
    decode_weight = per_layer_weight_bytes(cfg) * n_layers
    decode_kv_bytes = kv_read_bytes_per_layer(cfg, seq_ctx) * n_layers
    decode_total_bytes = decode_weight + decode_kv_bytes
    decode_ai = decode_flops / decode_total_bytes

    achieved_p_tflops = theory_prefill / (prefill_total_ms / 1000) / 1e12
    achieved_p_bw = prefill_weight / (prefill_total_ms / 1000) / 1e9
    achieved_d_tflops = decode_flops / (decode_avg_ms / 1000) / 1e12
    achieved_d_bw = decode_total_bytes / (decode_avg_ms / 1000) / 1e9

    # ── Qwen3-8B theoretical (from v1 report) ──
    cfg_8b = {'hidden_size': 4096, 'num_attention_heads': 32,
              'num_key_value_heads': 8, 'intermediate_size': 12288}
    n_8b = 36
    prefill_8b = per_layer_prefill_flops(cfg_8b, N) * n_8b
    weight_8b = per_layer_weight_bytes(cfg_8b) * n_8b
    prefill_ai_8b = prefill_8b / weight_8b

    decode_flops_8b = per_layer_decode_flops(cfg_8b, seq_ctx) * n_8b
    decode_w_8b = per_layer_weight_bytes(cfg_8b) * n_8b
    decode_kv_8b = kv_read_bytes_per_layer(cfg_8b, seq_ctx) * n_8b
    decode_ai_8b = decode_flops_8b / (decode_w_8b + decode_kv_8b)

    print(f"\n  Hardware: M4 Pro ({PEAK_TFLOPS} TFLOPS bf16, {PEAK_BW_GBS} GB/s)")
    print(f"  Ridge point: {ridge:.1f} FLOPs/Byte")
    print(f"  Context: {N} tokens")

    print(f"\n  {'Operation':25s} │ {'AI FLOPs/B':>12s} │ {'Zone':>14s} │ "
          f"{'TFLOPS':>9s} │ {'BW GB/s':>9s}")
    print(f"  {'─'*25}─┼{'─'*12}─┼{'─'*14}─┼{'─'*9}─┼{'─'*9}")

    def zone(ai):
        return "Compute-bound" if ai > ridge else "BW-bound"

    print(f"  {'1.7B Prefill (meas.)':25s} │ {prefill_ai:>11,.0f}  │ {zone(prefill_ai):>14s} │ "
          f"{achieved_p_tflops:>8.2f}  │ {achieved_p_bw:>8.1f} ")
    print(f"  {'1.7B Decode (meas.)':25s} │ {decode_ai:>11.1f}  │ {zone(decode_ai):>14s} │ "
          f"{achieved_d_tflops:>8.4f}  │ {achieved_d_bw:>8.1f} ")
    print(f"  {'8B Prefill (calc.)':25s} │ {prefill_ai_8b:>11,.0f}  │ {zone(prefill_ai_8b):>14s} │ "
          f"{'—':>9s} │ {'—':>9s}")
    print(f"  {'8B Decode (calc.)':25s} │ {decode_ai_8b:>11.1f}  │ {zone(decode_ai_8b):>14s} │ "
          f"{'—':>9s} │ {'—':>9s}")

    # AI gap
    gap_17b = prefill_ai / decode_ai
    gap_8b = prefill_ai_8b / decode_ai_8b

    print(f"\n  AI Gap (Prefill / Decode):")
    print(f"    Qwen3-1.7B:  {gap_17b:>8,.0f}×")
    print(f"    Qwen3-8B:    {gap_8b:>8,.0f}×")

    # GQA impact
    d_17b = cfg['hidden_size']
    d_8b = 4096
    n_kv_17b = cfg['num_key_value_heads']
    n_kv_8b = 8
    n_q_17b = cfg['num_attention_heads']
    n_q_8b = 32
    hd_17b = d_17b // n_q_17b
    hd_8b = d_8b // n_q_8b

    kv_per_tok_17b = 2 * n_kv_17b * hd_17b * 2
    kv_per_tok_8b = 2 * n_kv_8b * hd_8b * 2
    kv_read_17b = kv_per_tok_17b * n_layers * seq_ctx
    kv_read_8b = kv_per_tok_8b * n_8b * seq_ctx

    print(f"\n  GQA 影响:")
    print(f"    1.7B (GQA 2:1): n_kv={n_kv_17b}, KV/tok/layer={kv_per_tok_17b}B, "
          f"decode KV read={kv_read_17b/1e6:.1f}MB")
    print(f"    8B   (GQA 4:1): n_kv={n_kv_8b}, KV/tok/layer={kv_per_tok_8b}B, "
          f"decode KV read={kv_read_8b/1e6:.1f}MB")
    print(f"    1.7B KV/weight = {kv_read_17b/(per_layer_weight_bytes(cfg)*n_layers):.2f}")
    print(f"    8B   KV/weight = {kv_read_8b/(per_layer_weight_bytes(cfg_8b)*n_8b):.2f}")

    # h^(0) compression comparison
    comp_17b = 2 * n_layers * (n_kv_17b / n_q_17b)
    comp_8b = 2 * n_8b * (n_kv_8b / n_q_8b)
    print(f"\n  h^(0) 压缩比:")
    print(f"    1.7B: 2×{n_layers}×({n_kv_17b}/{n_q_17b}) = {comp_17b:.0f}×")
    print(f"    8B:   2×{n_8b}×({n_kv_8b}/{n_q_8b}) = {comp_8b:.0f}×")
    print(f"    GQA 2:1 比 4:1 压缩比高 {comp_17b/comp_8b:.1f}× "
          f"(在相似深度下)")

    # ASCII Roofline
    print(f"\n  Roofline Diagram:")
    print(f"             ┌{'─'*44}┐")
    print(f"  TFLOPS     │         ╱  {PEAK_TFLOPS} TFLOPS ceiling{' '*10}│")
    print(f"   {PEAK_TFLOPS:.0f} ─ ─ ─│─ ─ ─ ─ ╱── ── ── ── ── ── ── ── ─│")
    print(f"             │       ╱  ★P_1.7B (AI={prefill_ai:,.0f}){' '*5}│")
    print(f"             │      ╱   ★P_8B   (AI={prefill_ai_8b:,.0f}){' '*4}│")
    print(f"             │     ╱                            │")
    print(f"             │    ╱  ridge={ridge:.0f}{' '*22}│")
    print(f"             │   ╱                              │")
    print(f"             │  ╱                               │")
    print(f"             │ ╱ ★D_1.7B (AI={decode_ai:.1f}){' '*16}│")
    print(f"             │╱  ★D_8B   (AI={decode_ai_8b:.1f}){' '*16}│")
    print(f"             └{'─'*44}┘")
    print(f"                        AI (FLOPs/Byte)")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Datacenter PD v2 — Advanced Experiments")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3-1.7B-MLX-4bit")
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--gen-tokens", type=int, default=20)
    parser.add_argument("--experiments", default="all",
                        help="Comma-separated: d6,d7,d8,d9,d10 or 'all'")
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    inner = _find_inner_model(model)
    n_layers = len(inner.layers)
    cfg = model.args if hasattr(model, 'args') else inner.args
    if not isinstance(cfg, dict):
        cfg_dict = {
            'hidden_size': cfg.hidden_size,
            'num_attention_heads': cfg.num_attention_heads,
            'num_key_value_heads': cfg.num_key_value_heads,
            'intermediate_size': getattr(cfg, 'intermediate_size', cfg.hidden_size * 4),
        }
    else:
        cfg_dict = cfg

    d_model = cfg_dict['hidden_size']
    model_name = args.model.split('/')[-1]
    model_bytes = sum(p.nbytes for _, p in tree_flatten(model.parameters()))

    exps = (args.experiments.lower().split(',')
            if args.experiments != 'all' else ['d6', 'd7', 'd8', 'd9', 'd10'])

    # Build prompt
    target = args.prompt_tokens
    prompt_text = DIVERSE_PROMPT * (target // 100 + 1)
    tokens = tokenizer.encode(prompt_text)[:target]
    N = len(tokens)
    tok_mx = mx.array(tokens)
    question = "\nQ: What is the main topic of this text?\nA:"

    print("=" * 78)
    print(f"  DATACENTER PD v2 — ADVANCED EXPERIMENTS")
    print(f"  Model: {model_name} ({n_layers}L, d={d_model}, "
          f"{cfg_dict['num_key_value_heads']}KV, "
          f"GQA {cfg_dict['num_attention_heads']//cfg_dict['num_key_value_heads']}:1)")
    print(f"  Weights: {model_bytes/1024/1024:.0f} MB | "
          f"KV/tok/layer: {kv_per_token_per_layer_bytes(cfg_dict)} B | "
          f"h^(0)/tok: {d_model*2} B")
    print(f"  h^(0) compression: "
          f"{2*n_layers*(cfg_dict['num_key_value_heads']/cfg_dict['num_attention_heads']):.0f}×")
    print(f"  Context: {N} tokens | Gen: {args.gen_tokens} tokens")
    print(f"  Hardware: M4 Pro ({PEAK_TFLOPS} TFLOPS, {PEAK_BW_GBS} GB/s)")
    print("=" * 78)

    # Shared state
    prefill_ms = decode_ms = None
    residuals = clean_kv = None

    if 'd6' in exps:
        run_d6(cfg_dict, n_layers, model_bytes)

    if 'd7' in exps:
        prefill_ms, decode_ms, residuals, clean_kv = run_d7(
            inner, model, tokenizer, cfg_dict, tok_mx, N, n_layers, question)

    if 'd8' in exps:
        run_d8(inner, model, tokenizer, cfg_dict, n_layers, question, args.gen_tokens)

    if 'd9' in exps:
        run_d9(inner, model, tokenizer, cfg_dict, n_layers, model_bytes,
               question, args.gen_tokens)

    if 'd10' in exps:
        run_d10(inner, model, tokenizer, cfg_dict, tok_mx, N, n_layers, question,
                prefill_ms, decode_ms, clean_kv)

    print(f"\n{'=' * 78}")
    print(f"  Done.")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
