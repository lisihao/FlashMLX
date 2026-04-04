#!/usr/bin/env python3
"""
Datacenter PD Disaggregation Experiment.

Validates that pipeline split via residual checkpoint h^(cut) enables
efficient Prefill-Decode disaggregation:
  - Prefill is compute-bound (FLOPs-limited)
  - Decode is memory-bandwidth-bound (bytes/s-limited)
  - D node reconstructs KV from h^(cut) with zero decode speed loss

Sub-experiments:
  D1: Prefill vs Decode bottleneck characterization
  D2: Residual checkpoint recovery + decode efficiency
  D3: P:D node ratio analysis
  D4: h^(cut) transfer vs full KV transfer
  D5: KV memory per-node at different context lengths
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
from mlx_lm.models.kv_direct_cache import _find_inner_model
from mlx_lm.sample_utils import make_sampler
from mlx.utils import tree_flatten

GREEDY = make_sampler(temp=0.0)

DIVERSE_PROMPT = """The history of artificial intelligence began in antiquity, with myths and stories of artificial beings endowed with intelligence by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe human thinking as mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s. The field of AI research was founded at Dartmouth College in 1956. Many predicted a machine as intelligent as a human would exist within a generation. Eventually it became obvious that researchers had grossly underestimated the difficulty. In 1973, governments stopped funding undirected AI research, leading to the first AI winter. Investment boomed again in the 21st century when machine learning was successfully applied to many problems due to new methods, the application of powerful computer hardware, and the collection of immense data sets. Deep learning transformed the field starting around 2012 when neural networks began dramatically outperforming other methods."""


# ═══════════════════════════════════════════════════════════
# Theoretical FLOPs / Bytes calculations
# ═══════════════════════════════════════════════════════════

def per_layer_prefill_flops(cfg, seq_len):
    """Theoretical FLOPs for one transformer layer processing N tokens (prefill)."""
    d = cfg['hidden_size']
    n_q = cfg['num_attention_heads']
    n_kv = cfg['num_key_value_heads']
    hd = d // n_q
    inter = cfg.get('intermediate_size', d * 4)
    N = seq_len

    # QKV projection: N * (d → d + 2*n_kv*hd) * 2 (multiply-add)
    qkv = 2 * N * d * (d + 2 * n_kv * hd)
    # Output projection: N * d * d * 2
    o_proj = 2 * N * d * d
    # Attention: Q@K^T (N*N*hd per head) + attn@V (N*N*hd per head)
    attn = 2 * n_q * N * N * hd * 2
    # FFN SwiGLU: gate_proj + up_proj + down_proj
    ffn = 2 * 3 * N * d * inter

    return qkv + o_proj + attn + ffn


def per_layer_decode_flops(cfg, seq_len):
    """Theoretical FLOPs for one transformer layer processing 1 token (decode)."""
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
    """Weight bytes for one transformer layer (Q4 quantized)."""
    d = cfg['hidden_size']
    n_kv = cfg['num_key_value_heads']
    hd = d // cfg['num_attention_heads']
    inter = cfg.get('intermediate_size', d * 4)
    # Q4: ~0.5 bytes/param + scales overhead (~0.6 bytes/param effective)
    bpp = 0.6
    qkv = d * (d + 2 * n_kv * hd) * bpp
    o_proj = d * d * bpp
    ffn = 3 * d * inter * bpp
    norms = 2 * d * 2  # RMSNorm in bf16
    return qkv + o_proj + ffn + norms


def kv_read_bytes_per_layer(cfg, seq_len):
    """KV cache read bytes for one decode step (one layer)."""
    n_kv = cfg['num_key_value_heads']
    hd = cfg['hidden_size'] // cfg['num_attention_heads']
    return 2 * n_kv * seq_len * hd * 2  # K+V, bf16


def kv_per_token_per_layer_bytes(cfg):
    """KV bytes stored per token per layer."""
    n_kv = cfg['num_key_value_heads']
    hd = cfg['hidden_size'] // cfg['num_attention_heads']
    return 2 * n_kv * hd * 2  # K+V, bf16


# ═══════════════════════════════════════════════════════════
# Measurement helpers
# ═══════════════════════════════════════════════════════════

def measure_per_layer_prefill(inner, tokens_mx, n_layers):
    """Measure per-layer prefill time. Returns residuals, cache, per_layer_ms."""
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
    """Measure per-token decode speed. Returns list of per-token times (ms)."""
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


# ═══════════════════════════════════════════════════════════
# D1: Prefill vs Decode bottleneck characterization
# ═══════════════════════════════════════════════════════════

def run_d1(inner, model, tokenizer, cfg, tokens_mx, N, n_layers, question):
    print(f"\n{'═' * 78}")
    print(f"  D1: Prefill vs Decode 瓶颈特征")
    print(f"{'═' * 78}")

    # ── Prefill: measure per-layer ──
    gc.collect()
    residuals, cache, prefill_layer_ms = measure_per_layer_prefill(
        inner, tokens_mx, n_layers)
    prefill_total_ms = sum(prefill_layer_ms)

    # Theoretical FLOPs per layer
    theory_flops = per_layer_prefill_flops(cfg, N)
    theory_total = theory_flops * n_layers

    print(f"\n  Prefill ({N} tokens, {n_layers} layers):")
    print(f"    Total time:         {prefill_total_ms:>8.1f} ms")
    print(f"    Per-layer avg:      {np.mean(prefill_layer_ms):>8.2f} ms")
    print(f"    Per-layer std:      {np.std(prefill_layer_ms):>8.2f} ms")
    print(f"    Theory FLOPs/layer: {theory_flops/1e9:>8.2f} GFLOPs")
    print(f"    Theory FLOPs total: {theory_total/1e12:>8.3f} TFLOPs")

    # Compute utilization: achieved_flops = theory_flops / actual_time
    achieved_tflops = theory_total / (prefill_total_ms / 1000) / 1e12
    print(f"    Achieved throughput: {achieved_tflops:>7.2f} TFLOPS")

    # Prefill bytes (weight reads) — less dominant
    weight_bytes_layer = per_layer_weight_bytes(cfg)
    prefill_weight_bytes = weight_bytes_layer * n_layers
    prefill_bw = prefill_weight_bytes / (prefill_total_ms / 1000) / 1e9
    print(f"    Weight bytes read:  {prefill_weight_bytes/1e6:>8.1f} MB")
    print(f"    Effective BW:       {prefill_bw:>7.1f} GB/s")

    # ── Decode: measure per-token ──
    gc.collect()
    # Save clean KV before decode
    clean_kv = []
    for i in range(n_layers):
        k, v = cache[i].state
        mx.eval(k, v)
        clean_kv.append((k[:, :, :N, :], v[:, :, :N, :]))
        mx.eval(clean_kv[-1][0], clean_kv[-1][1])

    decode_times = measure_decode_speed(model, tokenizer, cache, question, 100)
    # Skip first 2 tokens (warmup/question encoding)
    steady_times = decode_times[3:] if len(decode_times) > 3 else decode_times[1:]
    decode_avg_ms = np.mean(steady_times)

    # Theoretical bytes per decode step
    seq_after_prefill = N + 50  # approximate midpoint
    decode_weight_bytes = weight_bytes_layer * n_layers
    decode_kv_bytes = sum(kv_read_bytes_per_layer(cfg, seq_after_prefill)
                          for _ in range(n_layers))
    decode_total_bytes = decode_weight_bytes + decode_kv_bytes
    decode_theory_flops = sum(per_layer_decode_flops(cfg, seq_after_prefill)
                               for _ in range(n_layers))

    decode_achieved_bw = decode_total_bytes / (decode_avg_ms / 1000) / 1e9
    decode_achieved_tflops = decode_theory_flops / (decode_avg_ms / 1000) / 1e12

    print(f"\n  Decode ({len(decode_times)} tokens, context≈{seq_after_prefill}):")
    print(f"    Avg per-token:      {decode_avg_ms:>8.2f} ms")
    print(f"    Tokens/sec:         {1000/decode_avg_ms:>8.1f}")
    print(f"    Theory bytes/step:  {decode_total_bytes/1e6:>8.1f} MB "
          f"(weights={decode_weight_bytes/1e6:.1f} + KV={decode_kv_bytes/1e6:.1f})")
    print(f"    Achieved BW:        {decode_achieved_bw:>7.1f} GB/s")
    print(f"    Theory FLOPs/step:  {decode_theory_flops/1e9:>8.2f} GFLOPs")
    print(f"    Achieved compute:   {decode_achieved_tflops:>7.4f} TFLOPS")

    # ── Bottleneck characterization ──
    # Arithmetic intensity = FLOPs / Bytes
    prefill_ai = theory_total / prefill_weight_bytes
    decode_ai = decode_theory_flops / decode_total_bytes

    print(f"\n  Arithmetic Intensity (FLOPs/Byte):")
    print(f"    Prefill:  {prefill_ai:>8.1f}  ← 高 → 计算密集")
    print(f"    Decode:   {decode_ai:>8.1f}  ← 低 → 访存密集")
    print(f"    比值:     {prefill_ai/decode_ai:>8.1f}×")

    return residuals, clean_kv, prefill_total_ms, decode_avg_ms, prefill_layer_ms


# ═══════════════════════════════════════════════════════════
# D2: Residual checkpoint recovery + decode efficiency
# ═══════════════════════════════════════════════════════════

def run_d2(inner, model, tokenizer, cfg, tokens_mx, N, n_layers,
           residuals, clean_kv, question, gen_tokens):
    print(f"\n{'═' * 78}")
    print(f"  D2: Residual Checkpoint 恢复 + Decode 效率验证")
    print(f"{'═' * 78}")

    # Get baseline generation
    gc.collect()
    baseline_cache = [KVCache() for _ in range(n_layers)]
    for i in range(n_layers):
        baseline_cache[i].state = clean_kv[i]
    mx.eval([c.keys for c in baseline_cache])

    baseline_tok = []
    for tok, _ in generate_step(mx.array(tokenizer.encode(question)), model,
                                 max_tokens=gen_tokens, sampler=GREEDY,
                                 prompt_cache=baseline_cache):
        baseline_tok.append(tok)

    # Baseline decode speed
    gc.collect()
    base_cache2 = [KVCache() for _ in range(n_layers)]
    for i in range(n_layers):
        base_cache2[i].state = clean_kv[i]
    mx.eval([c.keys for c in base_cache2])
    base_times = measure_decode_speed(model, tokenizer, base_cache2, question, 80)
    base_steady = base_times[3:] if len(base_times) > 3 else base_times[1:]
    base_avg = np.mean(base_steady)

    print(f"\n  A) Full prefill baseline:")
    print(f"    Decode: {base_avg:.2f} ms/tok ({1000/base_avg:.1f} tok/s)")

    for cut_l in [1, n_layers // 4, n_layers // 2, 3 * n_layers // 4]:
        if cut_l >= n_layers:
            continue

        gc.collect()
        pipe_cache = [KVCache() for _ in range(n_layers)]
        # Inject cloud KV for layers 0..cut-1
        for i in range(cut_l):
            pipe_cache[i].state = clean_kv[i]
        mx.eval([pipe_cache[i].keys for i in range(cut_l)])

        # Reconstruct layers cut..L-1 from h^(cut)
        h_cut = residuals[cut_l]
        mask_r = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(h_cut.dtype)
        t0 = time.perf_counter()
        xr = h_cut
        for i in range(cut_l, n_layers):
            xr = inner.layers[i](xr, mask=mask_r, cache=pipe_cache[i])
            mx.eval(xr)
        recon_ms = (time.perf_counter() - t0) * 1000

        # Quality check
        pipe_tok = []
        for tok, _ in generate_step(mx.array(tokenizer.encode(question)), model,
                                     max_tokens=gen_tokens, sampler=GREEDY,
                                     prompt_cache=pipe_cache):
            pipe_tok.append(tok)
        qs = quality_str(baseline_tok, pipe_tok)

        # Decode speed after reconstruction
        gc.collect()
        pipe_cache2 = [KVCache() for _ in range(n_layers)]
        for i in range(cut_l):
            pipe_cache2[i].state = clean_kv[i]
        mx.eval([pipe_cache2[i].keys for i in range(cut_l)])
        xr2 = h_cut
        for i in range(cut_l, n_layers):
            xr2 = inner.layers[i](xr2, mask=mask_r, cache=pipe_cache2[i])
            mx.eval(xr2)
        pipe_times = measure_decode_speed(model, tokenizer, pipe_cache2, question, 80)
        pipe_steady = pipe_times[3:] if len(pipe_times) > 3 else pipe_times[1:]
        pipe_avg = np.mean(pipe_steady)
        diff_pct = (pipe_avg - base_avg) / base_avg * 100

        print(f"\n  B) Pipeline@{cut_l} (D reconstructs {n_layers-cut_l} layers from h^({cut_l})):")
        print(f"    Reconstruction: {recon_ms:.0f} ms ({n_layers-cut_l} layers)")
        print(f"    Quality:        {qs}")
        print(f"    Decode:         {pipe_avg:.2f} ms/tok ({1000/pipe_avg:.1f} tok/s)")
        print(f"    vs Baseline:    {diff_pct:+.1f}%  {'← 无损' if abs(diff_pct) < 2 else '← 差异!'}")


# ═══════════════════════════════════════════════════════════
# D3: P:D ratio analysis
# ═══════════════════════════════════════════════════════════

def run_d3(prefill_total_ms, decode_avg_ms, N, n_layers):
    print(f"\n{'═' * 78}")
    print(f"  D3: P:D 节点配比分析")
    print(f"{'═' * 78}")

    print(f"\n  基础数据 (context={N}):")
    print(f"    Prefill time:     {prefill_total_ms:>8.1f} ms")
    print(f"    Decode per-token: {decode_avg_ms:>8.2f} ms")

    print(f"\n  {'Gen tokens':>12s} │ {'Decode total':>12s} │ {'P:D ratio':>10s} │ {'含义':>30s}")
    print(f"  {'─'*12}─┼{'─'*12}─┼{'─'*10}─┼{'─'*30}")

    for gen_tokens in [32, 64, 128, 256, 512, 1024, 2048]:
        decode_total = decode_avg_ms * gen_tokens
        ratio = decode_total / prefill_total_ms
        if ratio >= 1:
            meaning = f"1 P → {ratio:.0f} D (D 是瓶颈)"
        else:
            meaning = f"{1/ratio:.0f} P → 1 D (P 是瓶颈)"
        print(f"  {gen_tokens:>12d} │ {decode_total:>10.0f} ms │ {ratio:>10.1f} │ {meaning}")


# ═══════════════════════════════════════════════════════════
# D4: Transfer comparison
# ═══════════════════════════════════════════════════════════

def run_d4(cfg, n_layers, N):
    print(f"\n{'═' * 78}")
    print(f"  D4: 传输量对比 — Residual Checkpoint vs 全量 KV")
    print(f"{'═' * 78}")

    d = cfg['hidden_size']
    kv_ptpl = kv_per_token_per_layer_bytes(cfg)

    print(f"\n  {'方案':20s} │ {'传输量':>10s} │ {'IB 50GB/s':>10s} │ {'NVLink 900GB/s':>14s}")
    print(f"  {'─'*20}─┼{'─'*10}─┼{'─'*10}─┼{'─'*14}")

    configs = []
    # Full KV (DistServe)
    full_kv = kv_ptpl * n_layers * N
    configs.append(("Full KV (DistServe)", full_kv))

    # h^(cut) only (no cloud KV)
    for cut_l in [1, n_layers // 4, n_layers // 2]:
        h_bytes = N * d * 2
        kv_prefix = kv_ptpl * cut_l * N
        total = h_bytes + kv_prefix
        label = f"h^({cut_l}) + KV[0:{cut_l}]"
        configs.append((label, total))

    # h^(cut) only, no KV
    h_only = N * d * 2
    configs.append(("h^(18) only", h_only))

    for label, nbytes in configs:
        ib_ms = nbytes / (50e9) * 1000
        nvl_ms = nbytes / (900e9) * 1000
        print(f"  {label:20s} │ {nbytes/1e6:>8.1f} MB │ {ib_ms:>8.2f} ms │ {nvl_ms:>12.3f} ms")

    # Show ratio
    print(f"\n  压缩比 vs 全量 KV ({full_kv/1e6:.1f} MB):")
    for label, nbytes in configs[1:]:
        print(f"    {label:25s}: {full_kv/nbytes:>5.1f}×")


# ═══════════════════════════════════════════════════════════
# D5: KV memory per-node
# ═══════════════════════════════════════════════════════════

def run_d5(cfg, n_layers):
    print(f"\n{'═' * 78}")
    print(f"  D5: KV 内存 per-node — Pipeline Split 扩展 Context 容量")
    print(f"{'═' * 78}")

    kv_ptpl = kv_per_token_per_layer_bytes(cfg)
    cut_l = n_layers // 2

    print(f"\n  Pipeline@{cut_l} (50-50 切分), bf16 KV")
    print(f"\n  {'Context':>8s} │ {'全量 KV':>10s} │ {'P-node KV':>10s} │ {'D-node KV':>10s} │ {'节省':>6s} │ {'GPU 80GB fit':>12s}")
    print(f"  {'─'*8}─┼{'─'*10}─┼{'─'*10}─┼{'─'*10}─┼{'─'*6}─┼{'─'*12}")

    for ctx in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        full = kv_ptpl * n_layers * ctx
        p_kv = kv_ptpl * cut_l * ctx
        d_kv = kv_ptpl * (n_layers - cut_l) * ctx
        max_node = max(p_kv, d_kv)
        saving = (1 - max_node / full) * 100

        # Check fit with bf16 model weights (~16GB for 8B)
        model_bf16 = 16e9  # approximate
        full_fits = "Y" if (full + model_bf16) <= 80e9 else "N"
        pipe_fits = "Y" if (max_node + model_bf16 / 2) <= 80e9 else "N"

        print(f"  {ctx:>8,d} │ {full/1e9:>8.1f} GB │ {p_kv/1e9:>8.1f} GB │ {d_kv/1e9:>8.1f} GB │ "
              f"{saving:>5.0f}% │ {full_fits:>3s} → {pipe_fits:>3s}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--gen-tokens", type=int, default=50)
    parser.add_argument("--experiments", default="all",
                        help="Comma-separated: d1,d2,d3,d4,d5 or 'all'")
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    inner = _find_inner_model(model)
    n_layers = len(inner.layers)
    cfg = model.args if hasattr(model, 'args') else inner.args
    # Convert cfg to dict if needed
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
    n_kv_heads = cfg_dict['num_key_value_heads']
    head_dim = d_model // cfg_dict['num_attention_heads']
    model_name = args.model.split('/')[-1]
    model_bytes = sum(p.nbytes for _, p in tree_flatten(model.parameters()))

    exps = args.experiments.lower().split(',') if args.experiments != 'all' else ['d1','d2','d3','d4','d5']

    # Build prompt
    target = args.prompt_tokens
    prompt_text = DIVERSE_PROMPT * (target // 100 + 1)
    tokens = tokenizer.encode(prompt_text)[:target]
    N = len(tokens)
    tok_mx = mx.array(tokens)
    question = "\nQ: What is the main topic of this text?\nA:"

    print("=" * 78)
    print(f"  DATACENTER PD DISAGGREGATION EXPERIMENT")
    print(f"  Model: {model_name} ({n_layers}L, d={d_model}, "
          f"{n_kv_heads}KV, hd={head_dim})")
    print(f"  Weights: {model_bytes/1024/1024:.0f} MB | "
          f"KV/tok/layer: {kv_per_token_per_layer_bytes(cfg_dict)} B")
    print(f"  Context: {N} tokens | Gen: {args.gen_tokens} tokens")
    print("=" * 78)

    residuals = clean_kv = None
    prefill_total_ms = decode_avg_ms = 0

    if 'd1' in exps:
        residuals, clean_kv, prefill_total_ms, decode_avg_ms, _ = run_d1(
            inner, model, tokenizer, cfg_dict, tok_mx, N, n_layers, question)

    if 'd2' in exps:
        if residuals is None:
            residuals, cache_tmp, _ = measure_per_layer_prefill(inner, tok_mx, n_layers)
            clean_kv = []
            for i in range(n_layers):
                k, v = cache_tmp[i].state
                mx.eval(k, v)
                clean_kv.append((k[:,:,:N,:], v[:,:,:N,:]))
                mx.eval(clean_kv[-1][0], clean_kv[-1][1])
        run_d2(inner, model, tokenizer, cfg_dict, tok_mx, N, n_layers,
               residuals, clean_kv, question, args.gen_tokens)

    if 'd3' in exps:
        if prefill_total_ms == 0:
            # Quick measurement
            _, _, layer_ms = measure_per_layer_prefill(inner, tok_mx, n_layers)
            prefill_total_ms = sum(layer_ms)
        if decode_avg_ms == 0:
            decode_avg_ms = 22.0  # fallback from previous measurements
        run_d3(prefill_total_ms, decode_avg_ms, N, n_layers)

    if 'd4' in exps:
        run_d4(cfg_dict, n_layers, N)

    if 'd5' in exps:
        run_d5(cfg_dict, n_layers)

    print()


if __name__ == "__main__":
    main()
