#!/usr/bin/env python3
"""
Progressive Transfer Protocol v2 — incorporating KV quantization findings.

Key learnings from kv_quant_diagnostic/sweep:
  1. int8 gs=32 is the sweet spot: EXACT on 1.7B repetitive, ~40/50 on 8B
  2. int4 is unreliable for diverse prompts
  3. "First divergence" position matters more than total match count
  4. Greedy decoding is hypersensitive to quantization noise
  5. Phase 1 (pipeline) is always EXACT

Updated protocol:
  Phase 1: h^(cut) → pipeline decode (exact, has RTT penalty)
  Phase 2: int8 KV[0:cut] gs=32 → local decode (near-exact, no RTT)
  Phase 3: exact KV[0:cut] bf16 → local decode (exact, no RTT)

Usage:
    python3 experiments/dual_instance_h0/progressive_v2.py \
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

DIVERSE_PROMPT = """The history of artificial intelligence began in antiquity, with myths and stories of artificial beings endowed with intelligence by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe human thinking as mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s. The field of AI research was founded at Dartmouth College in 1956. Many predicted a machine as intelligent as a human would exist within a generation. Eventually it became obvious that researchers had grossly underestimated the difficulty. In 1973, governments stopped funding undirected AI research, leading to the first "AI winter." Investment boomed again in the 21st century when machine learning was successfully applied to many problems due to new methods and powerful hardware."""


# ════════════════════════════════════════════════════════
# Core utilities
# ════════════════════════════════════════════════════════

def capture_residuals_and_kv(inner_model, tokens):
    """Forward pass capturing all layer residuals and KV caches."""
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
    """Generate from pre-populated cache."""
    q = mx.array(tokenizer.encode(question))
    result = []
    for tok, _ in generate_step(q, model, max_tokens=max_tokens,
                                 sampler=GREEDY, prompt_cache=cache):
        result.append(tok)
    return result, tokenizer.decode(result)


def detect_mode_shift(residuals, n_tokens, d_model, threshold=3.0):
    """Find first major norm jump."""
    norms = []
    for l in range(len(residuals)):
        hl = np.array(residuals[l][0].astype(mx.float32))
        norms.append(np.linalg.norm(hl) / np.sqrt(n_tokens * d_model))
    for l in range(1, len(norms)):
        if norms[l] / norms[l-1] > threshold:
            return l, norms
    return len(residuals) // 2, norms


def first_diverge(baseline, generated):
    """Find the first token position where generation diverges."""
    for i, (a, b) in enumerate(zip(baseline, generated)):
        if a != b:
            return i
    return min(len(baseline), len(generated))


def quality_report(baseline, generated, label):
    """Print quality comparison."""
    match = sum(1 for a, b in zip(baseline, generated) if a == b)
    total = min(len(baseline), len(generated))
    exact = baseline == generated
    fd = first_diverge(baseline, generated)
    if exact:
        return f"{label}: EXACT ✓ ({total} tokens)"
    else:
        return f"{label}: {match}/{total} match, first diverge @ token {fd}"


def clone_with_quant(exact_kv, n_layers, quant_layers, bits=8, gs=32):
    """Clone KV cache with quantization on specified layers."""
    cache = [KVCache() for _ in range(n_layers)]
    qsize = 0
    for i in range(n_layers):
        k, v = exact_kv[i].state
        mx.eval(k, v)
        if i in quant_layers and bits < 16:
            qk = mx.quantize(k, group_size=gs, bits=bits)
            qv = mx.quantize(v, group_size=gs, bits=bits)
            kd = mx.dequantize(*qk, group_size=gs, bits=bits)
            vd = mx.dequantize(*qv, group_size=gs, bits=bits)
            mx.eval(kd, vd)
            cache[i].state = (kd, vd)
            qsize += sum(x.nbytes for x in qk) + sum(x.nbytes for x in qv)
        else:
            cache[i].state = (k, v)
            qsize += k.nbytes + v.nbytes
    mx.eval([c.keys for c in cache] + [c.values for c in cache])
    return cache, qsize


# ════════════════════════════════════════════════════════
# Progressive protocol simulation
# ════════════════════════════════════════════════════════

def run_progressive(model, inner_model, tokenizer, residuals, exact_kv,
                     n_layers, cut_l, baseline_tok, question, cfg, N):
    """Simulate the 3-phase progressive transfer."""

    d_model = cfg.hidden_size
    head_dim = d_model // cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    kv_per_layer_token = 2 * n_kv_heads * head_dim * 2  # K+V, bf16

    h_cut = residuals[cut_l]
    h_bytes = N * d_model * 2

    print(f"\n  {'═' * 65}")
    print(f"  Progressive Protocol: cut@{cut_l} "
          f"(Cloud: 0..{cut_l-1}, Edge: {cut_l}..{n_layers-1})")
    print(f"  {'═' * 65}")

    # ── Phase 1: Pipeline decode ──
    t0 = time.perf_counter()
    mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(h_cut.dtype)
    edge_caches = [KVCache() for _ in range(n_layers - cut_l)]
    x = h_cut
    for i in range(n_layers - cut_l):
        x = inner_model.layers[cut_l + i](x, mask=mask, cache=edge_caches[i])
        mx.eval(x)
    recon_ms = (time.perf_counter() - t0) * 1000

    # Phase 1 cache: exact KV[0:cut] (cloud side) + reconstructed KV[cut:L]
    cache_p1 = make_prompt_cache(model)
    for i in range(cut_l):
        cache_p1[i].state = exact_kv[i].state
    for i in range(n_layers - cut_l):
        cache_p1[cut_l + i].state = edge_caches[i].state
    mx.eval([c.keys for c in cache_p1] + [c.values for c in cache_p1])

    gen_p1, text_p1 = generate_with_cache(model, tokenizer, cache_p1, question)
    print(f"  Phase 1 (h^({cut_l})={h_bytes/1024/1024:.1f}MB, pipeline)")
    print(f"    {quality_report(baseline_tok, gen_p1, 'Quality')}")
    print(f"    Edge recon: {recon_ms:.0f}ms | RTT penalty per token")

    # ── Phase 2: int8 gs=32 KV[0:cut] ──
    quant_set = set(range(cut_l))
    for bits, gs, label in [(8, 32, "int8 gs=32"), (8, 64, "int8 gs=64"), (4, 32, "int4 gs=32")]:
        cache_p2 = [KVCache() for _ in range(n_layers)]
        kv_qbytes = 0
        for i in range(n_layers):
            if i >= cut_l:
                cache_p2[i].state = edge_caches[i - cut_l].state
            else:
                k, v = exact_kv[i].state
                mx.eval(k, v)
                qk = mx.quantize(k, group_size=gs, bits=bits)
                qv = mx.quantize(v, group_size=gs, bits=bits)
                kd = mx.dequantize(*qk, group_size=gs, bits=bits)
                vd = mx.dequantize(*qv, group_size=gs, bits=bits)
                mx.eval(kd, vd)
                cache_p2[i].state = (kd, vd)
                kv_qbytes += sum(x.nbytes for x in qk) + sum(x.nbytes for x in qv)
        mx.eval([c.keys for c in cache_p2] + [c.values for c in cache_p2])

        gen_p2, _ = generate_with_cache(model, tokenizer, cache_p2, question)
        exact_kv_bytes = cut_l * kv_per_layer_token * N
        print(f"  Phase 2 ({label}, {kv_qbytes/1024/1024:.1f}MB = {exact_kv_bytes/kv_qbytes:.1f}× compress)")
        print(f"    {quality_report(baseline_tok, gen_p2, 'Quality')}")
        print(f"    No RTT | Local decode")

    # ── Phase 3: exact KV[0:cut] bf16 ──
    cache_p3 = make_prompt_cache(model)
    exact_bytes = 0
    for i in range(n_layers):
        if i >= cut_l:
            cache_p3[i].state = edge_caches[i - cut_l].state
        else:
            k, v = exact_kv[i].state
            mx.eval(k, v)
            cache_p3[i].state = (k, v)
            exact_bytes += k.nbytes + v.nbytes
    mx.eval([c.keys for c in cache_p3] + [c.values for c in cache_p3])

    gen_p3, _ = generate_with_cache(model, tokenizer, cache_p3, question)
    print(f"  Phase 3 (exact bf16, {exact_bytes/1024/1024:.1f}MB)")
    print(f"    {quality_report(baseline_tok, gen_p3, 'Quality')}")
    print(f"    No RTT | Full fidelity")

    # Summary
    full_kv = kv_per_layer_token * n_layers * N
    print(f"\n  Transfer budget comparison:")
    print(f"    Full KV bf16:     {full_kv/1024/1024:>7.1f} MB")
    print(f"    Phase 1 h^({cut_l}):  {h_bytes/1024/1024:>7.1f} MB  ({full_kv/h_bytes:.0f}× smaller)")
    print(f"    Phase 2 int8:     {kv_qbytes/1024/1024:>7.1f} MB  ({full_kv/kv_qbytes:.1f}× smaller)")
    print(f"    Phase 3 exact:    {exact_bytes/1024/1024:>7.1f} MB  ({full_kv/exact_bytes:.1f}× smaller)")
    print(f"    Progressive total:{(h_bytes+exact_bytes)/1024/1024:>7.1f} MB  "
          f"(h+KV[0:{cut_l}], cloud still sends for cut layers only)")

    return {
        'cut_l': cut_l, 'h_bytes': h_bytes, 'recon_ms': recon_ms,
        'kv_int8_bytes': kv_qbytes, 'kv_exact_bytes': exact_bytes,
        'full_kv_bytes': full_kv,
    }


# ════════════════════════════════════════════════════════
# Latency model
# ════════════════════════════════════════════════════════

def latency_model(phase_data, cfg, N, n_layers, prefill_ms, tg_ms):
    """Model end-to-end latency on WiFi/5G/4G."""

    print(f"\n  {'═' * 65}")
    print(f"  LATENCY MODEL (cut@{phase_data['cut_l']})")
    print(f"  {'═' * 65}")

    profiles = [
        ("WiFi",  50,   5),
        ("5G",   100,  20),
        ("4G",    20,  50),
    ]

    cut_l = phase_data['cut_l']
    h_bytes = phase_data['h_bytes']
    kv_int8 = phase_data['kv_int8_bytes']
    kv_exact = phase_data['kv_exact_bytes']
    full_kv = phase_data['full_kv_bytes']
    recon_ms = phase_data['recon_ms']

    print(f"\n  {'Network':8s} │ {'TTFT':>8s} │ {'50tok':>8s} │ {'100tok':>8s} │ vs Full KV │ vs h(0) recon")
    print(f"  {'─'*8}─┼{'─'*8}─┼{'─'*8}─┼{'─'*8}─┼{'─'*10}─┼{'─'*13}")

    for name, bw, rtt in profiles:
        def xfer(b):
            return b * 8 / (bw * 1e6) * 1000

        # Progressive: h^(cut) → pipeline → int8 KV → local
        ttft = prefill_ms + xfer(h_bytes) + rtt + recon_ms
        t_switch = prefill_ms + xfer(h_bytes) + rtt + max(recon_ms, xfer(kv_int8))
        tg_rtt = tg_ms + rtt
        tg_local = tg_ms

        def total_progressive(n_tok):
            rtt_tok = min(n_tok, max(0, int((t_switch - ttft) / tg_rtt)))
            local_tok = n_tok - rtt_tok
            return ttft + rtt_tok * tg_rtt + local_tok * tg_local

        # Full KV
        ttft_full = prefill_ms + xfer(full_kv) + rtt
        def total_full(n_tok):
            return ttft_full + n_tok * tg_ms

        # h^(0) full recon
        h0_bytes = cfg.hidden_size * 2 * N
        ttft_h0 = prefill_ms + xfer(h0_bytes) + rtt + prefill_ms  # full recon ≈ prefill
        def total_h0(n_tok):
            return ttft_h0 + n_tok * tg_ms

        t50 = total_progressive(50)
        t100 = total_progressive(100)
        f50 = total_full(50)
        f100 = total_full(100)

        speedup_50 = f50 / t50
        speedup_h0 = total_h0(50) / t50

        print(f"  {name:8s} │ {ttft/1000:>7.1f}s │ {t50/1000:>7.1f}s │ {t100/1000:>7.1f}s │"
              f"  {speedup_50:>5.1f}× win │   {speedup_h0:>5.1f}× win")

    # Detailed breakdown for WiFi
    bw, rtt = 50, 5
    def xfer(b):
        return b * 8 / (bw * 1e6) * 1000

    print(f"\n  WiFi detailed timeline:")
    t = 0
    print(f"    t={t:>7.0f}ms  Cloud prefill starts")
    t += prefill_ms
    print(f"    t={t:>7.0f}ms  Cloud done, sends h^({cut_l}) ({h_bytes/1024/1024:.1f}MB)")
    t += xfer(h_bytes)
    print(f"    t={t:>7.0f}ms  h^({cut_l}) arrives, edge starts KV recon")
    t += rtt
    print(f"    t={t:>7.0f}ms  RTT (for pipeline decode setup)")
    t2 = t + recon_ms
    print(f"    t={t2:>7.0f}ms  Edge KV[{cut_l}..{n_layers-1}] ready → FIRST TOKEN")
    print(f"    ...pipeline decode at {tg_ms:.0f}+{rtt}={tg_ms+rtt:.0f}ms/tok...")
    t3 = t + xfer(kv_int8)
    print(f"    t={t3:>7.0f}ms  int8 KV[0..{cut_l-1}] received → switch to local decode")
    print(f"    ...local decode at {tg_ms:.0f}ms/tok (no RTT)...")
    t4 = t + xfer(kv_exact)
    print(f"    t={t4:>7.0f}ms  (Optional) exact KV[0..{cut_l-1}] received → full fidelity")
    print(f"    Compare: Full KV would arrive at t={prefill_ms + xfer(full_kv):.0f}ms")


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

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
    head_dim = d_model // cfg.num_attention_heads

    print("=" * 70)
    print("Progressive Transfer Protocol v2")
    print("=" * 70)
    print(f"Model: {args.model.split('/')[-1]}")
    print(f"Layers={n_layers}, d_model={d_model}, head_dim={head_dim}")

    # Build prompts
    prompts = {
        "repetitive": "The quick brown fox jumps over the lazy dog. " * 200,
        "diverse": DIVERSE_PROMPT * 3,
    }

    question = "\nQ: What is the main topic of this text?\nA:"

    for prompt_name, prompt_text in prompts.items():
        tokens = tokenizer.encode(prompt_text)[:args.prompt_tokens]
        prompt = tokenizer.decode(tokens)
        N = len(tokens)

        print(f"\n{'#' * 70}")
        print(f"# Prompt: {prompt_name} ({N} tokens)")
        print(f"{'#' * 70}")

        # Forward pass
        print(f"  Capturing residuals and KV for {N} tokens...")
        tok_mx = mx.array(tokens)
        t0 = time.perf_counter()
        residuals, exact_kv = capture_residuals_and_kv(inner, tok_mx)
        prefill_ms = (time.perf_counter() - t0) * 1000
        print(f"  Prefill: {prefill_ms:.0f}ms")

        # Detect mode shift
        cut_l, norms = detect_mode_shift(residuals, N, d_model)
        print(f"  Mode shift: layer {cut_l}")

        # Baseline generation
        cache_base = make_prompt_cache(model)
        for i in range(n_layers):
            cache_base[i].state = exact_kv[i].state
        mx.eval([c.keys for c in cache_base] + [c.values for c in cache_base])

        baseline_tok, baseline_text = generate_with_cache(
            model, tokenizer, cache_base, question, args.gen_tokens)
        print(f"  Baseline ({len(baseline_tok)} tokens): {baseline_text[:100]}")

        # Estimate TG speed (rough: prefill_ms / n_layers gives layer time)
        # For M3 Pro: ~22ms/tok for 8B, ~7ms/tok for 1.7B
        tg_ms = 22 if d_model >= 4096 else 7

        # Run progressive protocol at mode shift and 50-50
        for cut in sorted(set([cut_l, n_layers // 2])):
            phase_data = run_progressive(
                model, inner, tokenizer, residuals, exact_kv,
                n_layers, cut, baseline_tok, question, cfg, N)

            latency_model(phase_data, cfg, N, n_layers, prefill_ms, tg_ms)


if __name__ == "__main__":
    main()
