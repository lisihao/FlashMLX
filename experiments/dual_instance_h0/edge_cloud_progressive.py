#!/usr/bin/env python3
"""
Edge-Cloud Progressive Transfer Protocol Experiment.

Three-phase progressive transfer for pipeline-split edge-cloud inference:

  Phase 1: "Quick Start" — send h^(cut) only (16 MB)
    Edge reconstructs KV[cut:L-1], starts pipeline decode (RTT per token)

  Phase 2: "Background KV" — stream int4 KV[0:cut-1] (36 MB)
    Edge receives compressed KV, switches to local-only decode (no RTT)

  Phase 3: "Precision Upgrade" — send int8 KV patches on demand
    Edge requests higher precision for quality-critical layers

Key insight: Quantize KV (no cascade), NOT h^(l) (cascades through L-cut layers).

Usage:
    python3 experiments/dual_instance_h0/edge_cloud_progressive.py \
        --model /path/to/model --prompt-tokens 2048
"""

from __future__ import annotations

import sys
import time

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


# ════════════════════════════════════════════════════════
# KV Quantization Utilities
# ════════════════════════════════════════════════════════

def quantize_kv(keys, values, bits=4, group_size=64):
    """Quantize KV cache tensors using mx.quantize.

    Args:
        keys: (B, n_kv_heads, N, head_dim) post-RoPE keys
        values: (B, n_kv_heads, N, head_dim) values
        bits: 4 or 8
        group_size: quantization group size

    Returns:
        (q_keys, q_values, nbytes) where q_keys/q_values are (data, scales, biases) tuples
    """
    q_keys = mx.quantize(keys, group_size=group_size, bits=bits)
    q_values = mx.quantize(values, group_size=group_size, bits=bits)
    mx.eval(*q_keys, *q_values)

    # Calculate transfer bytes
    nbytes = sum(x.nbytes for x in q_keys) + sum(x.nbytes for x in q_values)
    return q_keys, q_values, nbytes


def dequantize_kv(q_keys, q_values, bits=4, group_size=64):
    """Dequantize KV cache tensors."""
    keys = mx.dequantize(*q_keys, group_size=group_size, bits=bits)
    values = mx.dequantize(*q_values, group_size=group_size, bits=bits)
    mx.eval(keys, values)
    return keys, values


# ════════════════════════════════════════════════════════
# Core Utilities
# ════════════════════════════════════════════════════════

def capture_all_residuals(inner_model, tokens):
    """Run forward pass capturing residual stream and KV caches."""
    x = inner_model.embed_tokens(tokens.reshape(1, -1))
    mx.eval(x)
    residuals = [x]

    N = tokens.shape[0]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(x.dtype)

    num_layers = len(inner_model.layers)
    cache = [KVCache() for _ in range(num_layers)]

    for i, layer in enumerate(inner_model.layers):
        x = layer(x, mask=mask, cache=cache[i])
        mx.eval(x)
        residuals.append(x)

    return residuals, cache


def generate_with_cache(model, tokenizer, cache, question, max_tokens=50):
    """Generate tokens from a pre-populated cache."""
    q = mx.array(tokenizer.encode(question))
    tokens = []
    for tok, _ in generate_step(q, model, max_tokens=max_tokens, sampler=GREEDY, prompt_cache=cache):
        tokens.append(tok)
    return tokens, tokenizer.decode(tokens)


def detect_mode_shift(residuals, n_tokens, d_model, threshold=3.0):
    """Find the first layer with a major norm jump."""
    norms = []
    for l in range(len(residuals)):
        hl_np = np.array(residuals[l][0].astype(mx.float32))
        norm = np.linalg.norm(hl_np) / np.sqrt(n_tokens * d_model)
        norms.append(norm)

    for l in range(1, len(norms)):
        if norms[l] / norms[l - 1] > threshold:
            return l, norms
    return len(residuals) // 2, norms


# ════════════════════════════════════════════════════════
# P1: KV Quantization Quality (the foundation)
# ════════════════════════════════════════════════════════

def test_kv_quantization(model, tokenizer, exact_kv, n_layers, cut_l,
                          gen_baseline, question, bits_list=[16, 8, 4]):
    """Test: exact KV[cut:L] + quantized KV[0:cut] → generation quality."""

    print(f"\n  {'Bits':>4} │ {'GrpSz':>5} │ {'KV[0:{0}] size'.format(cut_l):>14} │ "
          f"{'K err':>10} │ {'V err':>10} │ {'Match':>8} │ {'Quality':>8}")
    print(f"  {'─' * 4}─┼{'─' * 5}─┼{'─' * 14}─┼{'─' * 10}─┼{'─' * 10}─┼{'─' * 8}─┼{'─' * 8}")

    results = []
    for bits in bits_list:
        for group_size in [64]:
            cache_test = make_prompt_cache(model)

            total_kv_bytes = 0
            k_errors = []
            v_errors = []

            for i in range(n_layers):
                if i >= cut_l:
                    # Layers cut..L-1: exact KV (edge reconstructed these)
                    cache_test[i].state = exact_kv[i].state
                else:
                    # Layers 0..cut-1: quantized KV (transferred from cloud)
                    k_exact, v_exact = exact_kv[i].state
                    mx.eval(k_exact, v_exact)

                    if bits == 16:
                        cache_test[i].state = (k_exact, v_exact)
                        total_kv_bytes += k_exact.nbytes + v_exact.nbytes
                    else:
                        q_k, q_v, nbytes = quantize_kv(k_exact, v_exact, bits=bits,
                                                        group_size=group_size)
                        k_deq, v_deq = dequantize_kv(q_k, q_v, bits=bits,
                                                      group_size=group_size)
                        cache_test[i].state = (k_deq, v_deq)
                        total_kv_bytes += nbytes

                        # Error measurement
                        k_err = float(mx.sqrt(mx.mean((k_deq - k_exact) ** 2)).item())
                        v_err = float(mx.sqrt(mx.mean((v_deq - v_exact) ** 2)).item())
                        k_norm = float(mx.sqrt(mx.mean(k_exact ** 2)).item())
                        v_norm = float(mx.sqrt(mx.mean(v_exact ** 2)).item())
                        k_errors.append(k_err / max(k_norm, 1e-8))
                        v_errors.append(v_err / max(v_norm, 1e-8))

            mx.eval([c.keys for c in cache_test] + [c.values for c in cache_test])

            gen_tok, text = generate_with_cache(model, tokenizer, cache_test, question)

            token_match = sum(1 for a, b in zip(gen_baseline, gen_tok) if a == b)
            total = min(len(gen_baseline), len(gen_tok))
            exact = gen_baseline == gen_tok

            avg_k_err = np.mean(k_errors) if k_errors else 0
            avg_v_err = np.mean(v_errors) if v_errors else 0

            label = "bf16" if bits == 16 else f"int{bits}"
            quality = "EXACT ✓" if exact else f"{token_match}/{total}"

            print(f"  {label:>4} │ {group_size:>5} │ "
                  f"{total_kv_bytes / 1024 / 1024:>12.1f} MB│ "
                  f"{avg_k_err:>9.6f} │ {avg_v_err:>9.6f} │ "
                  f"{token_match:>4}/{total:<3} │ "
                  f"{'EXACT ✓' if exact else 'DIVERGED'}")

            if not exact:
                print(f"         → {text[:120]}")

            results.append({
                'bits': bits, 'group_size': group_size,
                'kv_bytes': total_kv_bytes, 'kv_mb': total_kv_bytes / 1024 / 1024,
                'k_err': avg_k_err, 'v_err': avg_v_err,
                'match': token_match, 'total': total, 'exact': exact,
                'text': text,
            })

    return results


# ════════════════════════════════════════════════════════
# P2: Phase Simulation — Progressive Transfer
# ════════════════════════════════════════════════════════

def simulate_progressive_phases(model, inner_model, tokenizer, residuals, exact_kv,
                                 n_layers, cut_l, gen_baseline, question, cfg):
    """Simulate the 3-phase progressive transfer protocol."""

    d_model = cfg.hidden_size
    n_kv_heads = cfg.num_key_value_heads
    head_dim = d_model // cfg.num_attention_heads
    N = residuals[0].shape[1]

    h_cut = residuals[cut_l]
    h_bytes = N * d_model * 2

    print(f"\n  ─── Simulating 3-Phase Protocol: cut@{cut_l} ───")
    print(f"  Cloud: layers 0..{cut_l - 1} | Edge: layers {cut_l}..{n_layers - 1}")

    # ── Phase 1: h^(cut) only → edge reconstructs KV[cut:L-1] ──
    print(f"\n  Phase 1: Send h^({cut_l}) = {h_bytes / 1024 / 1024:.1f} MB")

    t0 = time.perf_counter()
    mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(h_cut.dtype)
    edge_caches = [KVCache() for _ in range(n_layers - cut_l)]
    x = h_cut
    for i in range(n_layers - cut_l):
        layer_obj = inner_model.layers[cut_l + i]
        x = layer_obj(x, mask=mask, cache=edge_caches[i])
        mx.eval(x)
    recon_ms = (time.perf_counter() - t0) * 1000

    # Phase 1 generation: pipeline mode (no KV[0:cut], must round-trip)
    # Simulate: use exact KV[0:cut] as proxy for "cloud handles these layers"
    cache_p1 = make_prompt_cache(model)
    for i in range(cut_l):
        cache_p1[i].state = exact_kv[i].state  # Cloud's KV (pipeline mode)
    for i in range(n_layers - cut_l):
        cache_p1[cut_l + i].state = edge_caches[i].state
    mx.eval([c.keys for c in cache_p1] + [c.values for c in cache_p1])

    gen_p1, text_p1 = generate_with_cache(model, tokenizer, cache_p1, question)
    match_p1 = sum(1 for a, b in zip(gen_baseline, gen_p1) if a == b)
    exact_p1 = gen_baseline == gen_p1

    print(f"    Edge recon: {recon_ms:.0f}ms (layers {cut_l}..{n_layers - 1})")
    print(f"    Pipeline decode: {'EXACT ✓' if exact_p1 else f'{match_p1}/50 DIVERGED'}")
    print(f"    (RTT per token during this phase)")

    # ── Phase 2: int4 KV[0:cut] arrives → switch to local decode ──
    print(f"\n  Phase 2: Receive int4 KV[0:{cut_l}] in background")

    kv_prefix_bytes_exact = 0
    kv_prefix_bytes_int4 = 0

    cache_p2 = make_prompt_cache(model)
    for i in range(n_layers):
        if i >= cut_l:
            cache_p2[i].state = edge_caches[i - cut_l].state
        else:
            k_exact, v_exact = exact_kv[i].state
            mx.eval(k_exact, v_exact)
            kv_prefix_bytes_exact += k_exact.nbytes + v_exact.nbytes

            # Quantize to int4
            q_k, q_v, nbytes = quantize_kv(k_exact, v_exact, bits=4, group_size=64)
            k_deq, v_deq = dequantize_kv(q_k, q_v, bits=4, group_size=64)
            cache_p2[i].state = (k_deq, v_deq)
            kv_prefix_bytes_int4 += nbytes

    mx.eval([c.keys for c in cache_p2] + [c.values for c in cache_p2])

    gen_p2, text_p2 = generate_with_cache(model, tokenizer, cache_p2, question)
    match_p2 = sum(1 for a, b in zip(gen_baseline, gen_p2) if a == b)
    exact_p2 = gen_baseline == gen_p2

    print(f"    int4 KV[0:{cut_l}]: {kv_prefix_bytes_int4 / 1024 / 1024:.1f} MB "
          f"(vs exact {kv_prefix_bytes_exact / 1024 / 1024:.1f} MB)")
    print(f"    Local decode quality: {match_p2}/50 "
          f"{'EXACT ✓' if exact_p2 else 'DIVERGED'}")
    if not exact_p2:
        print(f"    → {text_p2[:150]}")
    print(f"    (No RTT after this phase!)")

    # ── Phase 3: int8 KV patches → upgrade precision ──
    print(f"\n  Phase 3: Upgrade to int8 KV[0:{cut_l}]")

    kv_prefix_bytes_int8 = 0
    cache_p3 = make_prompt_cache(model)
    for i in range(n_layers):
        if i >= cut_l:
            cache_p3[i].state = edge_caches[i - cut_l].state
        else:
            k_exact, v_exact = exact_kv[i].state
            mx.eval(k_exact, v_exact)

            q_k, q_v, nbytes = quantize_kv(k_exact, v_exact, bits=8, group_size=64)
            k_deq, v_deq = dequantize_kv(q_k, q_v, bits=8, group_size=64)
            cache_p3[i].state = (k_deq, v_deq)
            kv_prefix_bytes_int8 += nbytes

    mx.eval([c.keys for c in cache_p3] + [c.values for c in cache_p3])

    gen_p3, text_p3 = generate_with_cache(model, tokenizer, cache_p3, question)
    match_p3 = sum(1 for a, b in zip(gen_baseline, gen_p3) if a == b)
    exact_p3 = gen_baseline == gen_p3

    # Phase 3 delta: how much extra to send for int8 upgrade
    delta_bytes = kv_prefix_bytes_int8 - kv_prefix_bytes_int4

    print(f"    int8 KV[0:{cut_l}]: {kv_prefix_bytes_int8 / 1024 / 1024:.1f} MB "
          f"(delta from int4: +{delta_bytes / 1024 / 1024:.1f} MB)")
    print(f"    Local decode quality: {match_p3}/50 "
          f"{'EXACT ✓' if exact_p3 else 'DIVERGED'}")
    if not exact_p3:
        print(f"    → {text_p3[:150]}")

    # ── Phase 3b: Selective upgrade — only upgrade worst layers ──
    print(f"\n  Phase 3b: Selective int8 upgrade (worst layers only)")

    # Find which layers have highest quantization error at int4
    layer_errors = []
    for i in range(cut_l):
        k_exact, v_exact = exact_kv[i].state
        mx.eval(k_exact, v_exact)

        q_k, q_v, _ = quantize_kv(k_exact, v_exact, bits=4, group_size=64)
        k_deq, v_deq = dequantize_kv(q_k, q_v, bits=4, group_size=64)

        k_err = float(mx.sqrt(mx.mean((k_deq - k_exact) ** 2)).item())
        v_err = float(mx.sqrt(mx.mean((v_deq - v_exact) ** 2)).item())
        layer_errors.append((i, k_err + v_err))

    layer_errors.sort(key=lambda x: x[1], reverse=True)
    top_half = set(idx for idx, _ in layer_errors[:cut_l // 2])

    cache_p3b = make_prompt_cache(model)
    selective_bytes = 0
    for i in range(n_layers):
        if i >= cut_l:
            cache_p3b[i].state = edge_caches[i - cut_l].state
        else:
            k_exact, v_exact = exact_kv[i].state
            mx.eval(k_exact, v_exact)

            if i in top_half:
                # Upgrade these to int8
                q_k, q_v, nbytes = quantize_kv(k_exact, v_exact, bits=8, group_size=64)
                k_deq, v_deq = dequantize_kv(q_k, q_v, bits=8, group_size=64)
                selective_bytes += nbytes
            else:
                # Keep int4
                q_k, q_v, nbytes = quantize_kv(k_exact, v_exact, bits=4, group_size=64)
                k_deq, v_deq = dequantize_kv(q_k, q_v, bits=4, group_size=64)
                selective_bytes += nbytes

            cache_p3b[i].state = (k_deq, v_deq)

    mx.eval([c.keys for c in cache_p3b] + [c.values for c in cache_p3b])

    gen_p3b, text_p3b = generate_with_cache(model, tokenizer, cache_p3b, question)
    match_p3b = sum(1 for a, b in zip(gen_baseline, gen_p3b) if a == b)
    exact_p3b = gen_baseline == gen_p3b

    print(f"    Top {cut_l // 2} worst layers → int8, rest → int4")
    print(f"    Selective KV size: {selective_bytes / 1024 / 1024:.1f} MB")
    print(f"    Quality: {match_p3b}/50 "
          f"{'EXACT ✓' if exact_p3b else 'DIVERGED'}")

    return {
        'p1_exact': exact_p1, 'p1_match': match_p1,
        'p2_exact': exact_p2, 'p2_match': match_p2,
        'p3_exact': exact_p3, 'p3_match': match_p3,
        'p3b_exact': exact_p3b, 'p3b_match': match_p3b,
        'h_bytes': h_bytes,
        'kv_int4_bytes': kv_prefix_bytes_int4,
        'kv_int8_bytes': kv_prefix_bytes_int8,
        'kv_exact_bytes': kv_prefix_bytes_exact,
        'kv_selective_bytes': selective_bytes,
        'recon_ms': recon_ms,
    }


# ════════════════════════════════════════════════════════
# P3: Latency Timeline
# ════════════════════════════════════════════════════════

def latency_timeline(phase_data, cfg, N, cut_l, n_layers, server_prefill_ms, tg_ms):
    """Model the progressive transfer timeline on different networks."""

    print(f"\n{'=' * 110}")
    print(f"  PROGRESSIVE TRANSFER TIMELINE")
    print(f"{'=' * 110}")

    profiles = {
        'WiFi (50M)': {'bw_mbps': 50, 'rtt_ms': 5},
        '5G (100M)':  {'bw_mbps': 100, 'rtt_ms': 20},
        '4G (20M)':   {'bw_mbps': 20, 'rtt_ms': 50},
    }

    h_bytes = phase_data['h_bytes']
    kv_int4_bytes = phase_data['kv_int4_bytes']
    kv_int8_bytes = phase_data['kv_int8_bytes']
    kv_exact_bytes = phase_data['kv_exact_bytes']
    recon_ms = phase_data['recon_ms']
    full_kv_bytes = kv_exact_bytes + sum(
        2 * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2 * N
        for _ in range(n_layers - cut_l)
    )

    for net_name, p in profiles.items():
        bw = p['bw_mbps']
        rtt = p['rtt_ms']

        def xfer_ms(nbytes):
            return nbytes * 8 / (bw * 1e6) * 1000

        t_h = xfer_ms(h_bytes)           # Transfer h^(cut)
        t_kv4 = xfer_ms(kv_int4_bytes)   # Transfer int4 KV
        t_kv8 = xfer_ms(kv_int8_bytes)   # Transfer int8 KV
        t_full = xfer_ms(full_kv_bytes)   # Transfer full KV (baseline)

        print(f"\n  ─── {net_name} (BW={bw}Mbps, RTT={rtt}ms) ───")
        print()

        # Progressive timeline
        t_cloud_prefill = server_prefill_ms
        t_phase1_start = t_cloud_prefill + t_h + rtt      # h^(cut) arrives
        t_phase1_recon = t_phase1_start + recon_ms          # KV[cut:L] ready
        t_first_token = t_phase1_recon                      # First token generated!
        t_phase2_done = t_phase1_start + t_kv4              # int4 KV received
        # Edge switches to local after max(recon, kv4 received)
        t_switch = max(t_phase1_recon, t_phase2_done)

        # How many tokens generated in RTT mode before switch?
        tg_rtt = tg_ms + rtt  # per token in pipeline mode
        tokens_before_switch = max(0, (t_switch - t_first_token) / tg_rtt)

        # Comparison: full KV transfer
        t_fullkv_first = t_cloud_prefill + t_full + rtt  # All KV arrives
        t_fullkv_tg = tg_ms  # No RTT penalty

        # Comparison: h^(0) full recon
        h0_bytes = cfg.hidden_size * 2 * N
        t_h0_first = t_cloud_prefill + xfer_ms(h0_bytes) + rtt + server_prefill_ms  # Full recon ≈ prefill

        print(f"    Progressive Protocol:")
        print(f"      t=0:          Cloud prefill starts")
        print(f"      t={t_cloud_prefill:.0f}ms:    Cloud done, sends h^({cut_l})")
        print(f"      t={t_phase1_start:.0f}ms:   h^({cut_l}) arrives, edge starts recon")
        print(f"      t={t_first_token:.0f}ms:   FIRST TOKEN (pipeline mode, +{rtt}ms/tok)")

        if t_phase2_done > t_first_token:
            print(f"      t={t_phase2_done:.0f}ms:  int4 KV arrives (background transfer)")
        print(f"      t={t_switch:.0f}ms:   Switch to LOCAL decode ({tg_ms:.1f}ms/tok)")
        print(f"      Tokens in RTT mode: ~{tokens_before_switch:.0f}")
        print()

        # N tokens generation time comparison (e.g. 100 tokens)
        for n_gen in [50, 100]:
            # Progressive: some tokens at RTT speed, rest at local speed
            rtt_tokens = min(n_gen, int(tokens_before_switch))
            local_tokens = n_gen - rtt_tokens
            total_progressive = t_first_token + rtt_tokens * tg_rtt + local_tokens * tg_ms

            total_fullkv = t_fullkv_first + n_gen * tg_ms
            total_h0 = t_h0_first + n_gen * tg_ms

            print(f"    {n_gen} tokens total time:")
            print(f"      Progressive:    {total_progressive / 1000:.1f}s "
                  f"(TTFT {t_first_token / 1000:.1f}s)")
            print(f"      Full KV xfer:   {total_fullkv / 1000:.1f}s "
                  f"(TTFT {t_fullkv_first / 1000:.1f}s)")
            print(f"      h^(0) full recon:{total_h0 / 1000:.1f}s "
                  f"(TTFT {t_h0_first / 1000:.1f}s)")
            print(f"      Winner: {'Progressive' if total_progressive < min(total_fullkv, total_h0) else 'Full KV' if total_fullkv < total_h0 else 'h^(0)'}")
            print()

        # Transfer budget summary
        total_progressive_xfer = h_bytes + kv_int4_bytes
        print(f"    Transfer budget:")
        print(f"      Full KV (bf16):    {full_kv_bytes / 1024 / 1024:>8.1f} MB")
        print(f"      Progressive total: {total_progressive_xfer / 1024 / 1024:>8.1f} MB "
              f"(h^({cut_l}) {h_bytes / 1024 / 1024:.1f} + int4 KV {kv_int4_bytes / 1024 / 1024:.1f})")
        print(f"      Savings:           {(1 - total_progressive_xfer / full_kv_bytes) * 100:.0f}%")


# ════════════════════════════════════════════════════════
# P4: Multiple Cut Points Comparison
# ════════════════════════════════════════════════════════

def compare_cut_points(model, inner_model, tokenizer, residuals, exact_kv,
                        gen_baseline, question, n_layers, cfg, N):
    """Compare progressive protocol at different cut points."""

    print(f"\n{'=' * 110}")
    print(f"  CUT POINT COMPARISON — Progressive Transfer Quality")
    print(f"{'=' * 110}")

    d_model = cfg.hidden_size
    n_kv_heads = cfg.num_key_value_heads
    head_dim = d_model // cfg.num_attention_heads
    kv_per_layer_token = 2 * n_kv_heads * head_dim * 2
    kv_full = kv_per_layer_token * n_layers * N
    h_bytes = d_model * 2 * N

    cuts = [1, 3, 6, n_layers // 4, n_layers // 3, n_layers // 2, 2 * n_layers // 3]
    cuts = sorted(set(c for c in cuts if 1 <= c < n_layers))

    print(f"\n  {'Cut':>4} │ {'Edge%':>5} │ {'h^(l)':>7} │ {'int4 KV':>8} │ {'Total':>8} │ "
          f"{'Compress':>8} │ {'P1 (pipe)':>9} │ {'P2 (int4)':>9} │ {'P3 (int8)':>9}")
    print(f"  {'─' * 4}─┼{'─' * 5}─┼{'─' * 7}─┼{'─' * 8}─┼{'─' * 8}─┼"
          f"{'─' * 8}─┼{'─' * 9}─┼{'─' * 9}─┼{'─' * 9}")

    for cut_l in cuts:
        h_cut = residuals[cut_l]

        # Edge reconstruction
        mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(h_cut.dtype)
        edge_caches = [KVCache() for _ in range(n_layers - cut_l)]
        x = h_cut
        for i in range(n_layers - cut_l):
            x = inner_model.layers[cut_l + i](x, mask=mask, cache=edge_caches[i])
            mx.eval(x)

        # Test each phase quality
        results_per_phase = {}
        for bits_label, bits in [("P1", 16), ("P2", 4), ("P3", 8)]:
            cache_test = make_prompt_cache(model)
            kv_prefix_bytes = 0

            for i in range(n_layers):
                if i >= cut_l:
                    cache_test[i].state = edge_caches[i - cut_l].state
                else:
                    k_exact, v_exact = exact_kv[i].state
                    mx.eval(k_exact, v_exact)

                    if bits_label == "P1":
                        # Pipeline mode: use exact (cloud handles)
                        cache_test[i].state = (k_exact, v_exact)
                        kv_prefix_bytes += k_exact.nbytes + v_exact.nbytes
                    else:
                        q_k, q_v, nb = quantize_kv(k_exact, v_exact, bits=bits, group_size=64)
                        k_d, v_d = dequantize_kv(q_k, q_v, bits=bits, group_size=64)
                        cache_test[i].state = (k_d, v_d)
                        kv_prefix_bytes += nb

            mx.eval([c.keys for c in cache_test] + [c.values for c in cache_test])
            gen_tok, _ = generate_with_cache(model, tokenizer, cache_test, question)
            match = sum(1 for a, b in zip(gen_baseline, gen_tok) if a == b)
            total = min(len(gen_baseline), len(gen_tok))
            results_per_phase[bits_label] = (match, total, kv_prefix_bytes)

        # Transfer sizes
        _, _, p1_bytes = results_per_phase["P1"]  # exact (pipeline mode, not actually transferred)
        _, _, p2_bytes = results_per_phase["P2"]  # int4
        total_xfer = h_bytes + p2_bytes
        compress = kv_full / total_xfer

        edge_pct = (n_layers - cut_l) / n_layers * 100

        p1_m, p1_t, _ = results_per_phase["P1"]
        p2_m, p2_t, _ = results_per_phase["P2"]
        p3_m, p3_t, _ = results_per_phase["P3"]

        def fmt_match(m, t):
            if m == t:
                return "EXACT ✓"
            return f"{m}/{t}"

        print(f"  {cut_l:>4} │ {edge_pct:>4.0f}% │ "
              f"{h_bytes / 1024 / 1024:>5.1f} MB│ {p2_bytes / 1024 / 1024:>6.1f} MB│ "
              f"{total_xfer / 1024 / 1024:>6.1f} MB│ {compress:>7.1f}× │ "
              f"{fmt_match(p1_m, p1_t):>9} │ {fmt_match(p2_m, p2_t):>9} │ "
              f"{fmt_match(p3_m, p3_t):>9}")


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--cut-layer", type=int, default=-1)
    args = parser.parse_args()

    print("Loading model...", file=sys.stderr)
    model, tokenizer = load(args.model)
    cfg = model.args
    n_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    n_kv_heads = cfg.num_key_value_heads
    head_dim = d_model // cfg.num_attention_heads

    inner_model = _find_inner_model(model)

    model(mx.array(tokenizer.encode("Hello")).reshape(1, -1))
    mx.eval(model.parameters())

    print(f"Model: {n_layers}L, d={d_model}, {cfg.num_attention_heads}Q/{n_kv_heads}KV",
          file=sys.stderr)

    FILLER = ("The development of artificial intelligence has progressed rapidly in recent years. "
              "Machine learning models continue to grow in capability and efficiency. "
              "Large language models can now understand and generate human-like text. "
              "Researchers explore new architectures to push boundaries further. ") * 100
    tokens = mx.array(tokenizer.encode(FILLER)[:args.prompt_tokens])
    N = tokens.shape[0]

    # Baseline
    print(f"\nCapturing baseline ({N} tokens)...", file=sys.stderr)
    t0 = time.perf_counter()
    residuals, exact_kv = capture_all_residuals(inner_model, tokens)
    server_prefill_ms = (time.perf_counter() - t0) * 1000

    cache_bl = make_prompt_cache(model)
    for i in range(n_layers):
        cache_bl[i].state = exact_kv[i].state
    mx.eval([c.keys for c in cache_bl] + [c.values for c in cache_bl])

    question = "What is artificial intelligence?"
    gen_baseline = []
    t0 = time.perf_counter()
    for tok, _ in generate_step(mx.array(tokenizer.encode(question)), model,
                                 max_tokens=50, sampler=GREEDY, prompt_cache=cache_bl):
        gen_baseline.append(tok)
    tg_total_ms = (time.perf_counter() - t0) * 1000
    tg_ms = tg_total_ms / max(1, len(gen_baseline))
    text_baseline = tokenizer.decode(gen_baseline)

    # Cut point
    mode_shift, norms = detect_mode_shift(residuals, N, d_model)
    cut_l = args.cut_layer if args.cut_layer >= 0 else n_layers // 2

    print(f"\n{'=' * 110}")
    print(f"  EDGE-CLOUD PROGRESSIVE TRANSFER EXPERIMENT")
    print(f"  Model: {n_layers}L, d={d_model} | Tokens: {N}")
    print(f"  Mode shift: layer {mode_shift} | Cut point: layer {cut_l}")
    print(f"  Prefill: {server_prefill_ms:.0f}ms | TG: {tg_ms:.1f}ms/tok")
    print(f"  Baseline: {text_baseline[:120]}")
    print(f"{'=' * 110}")

    # ── Test 1: KV quantization quality sweep ──
    print(f"\n{'=' * 110}")
    print(f"  TEST 1: KV QUANTIZATION QUALITY — int4/int8 KV[0:{cut_l}] + exact KV[{cut_l}:{n_layers}]")
    print(f"{'=' * 110}")

    kv_results = test_kv_quantization(model, tokenizer, exact_kv, n_layers, cut_l,
                                       gen_baseline, question, bits_list=[16, 8, 4])

    # ── Test 2: Full progressive simulation ──
    print(f"\n{'=' * 110}")
    print(f"  TEST 2: THREE-PHASE PROGRESSIVE PROTOCOL SIMULATION")
    print(f"{'=' * 110}")

    phase_data = simulate_progressive_phases(model, inner_model, tokenizer, residuals,
                                              exact_kv, n_layers, cut_l, gen_baseline,
                                              question, cfg)

    # ── Test 3: Latency timeline ──
    latency_timeline(phase_data, cfg, N, cut_l, n_layers, server_prefill_ms, tg_ms)

    # ── Test 4: Cut point sweep ──
    compare_cut_points(model, inner_model, tokenizer, residuals, exact_kv,
                        gen_baseline, question, n_layers, cfg, N)

    # ── Final verdict ──
    print(f"\n{'=' * 110}")
    print(f"  VERDICT")
    print(f"{'=' * 110}")

    kv_full_bytes = sum(exact_kv[i].keys.nbytes + exact_kv[i].values.nbytes for i in range(n_layers))
    total_progressive = phase_data['h_bytes'] + phase_data['kv_int4_bytes']

    p1_q = "EXACT" if phase_data['p1_exact'] else f"{phase_data['p1_match']}/50"
    p2_q = "EXACT" if phase_data['p2_exact'] else f"{phase_data['p2_match']}/50"
    p3_q = "EXACT" if phase_data['p3_exact'] else f"{phase_data['p3_match']}/50"

    print(f"\n  Progressive Transfer @ cut={cut_l}:")
    print(f"    Phase 1 (h^({cut_l})): {phase_data['h_bytes'] / 1024 / 1024:.1f} MB → {p1_q}")
    print(f"    Phase 2 (+int4 KV):  +{phase_data['kv_int4_bytes'] / 1024 / 1024:.1f} MB → {p2_q}")
    delta_mb = (phase_data['kv_int8_bytes'] - phase_data['kv_int4_bytes']) / 1024 / 1024
    print(f"    Phase 3 (+int8 KV):  +{delta_mb:.1f} MB → {p3_q}")
    print(f"\n    Total transfer: {total_progressive / 1024 / 1024:.1f} MB "
          f"(vs {kv_full_bytes / 1024 / 1024:.1f} MB full KV = "
          f"{kv_full_bytes / total_progressive:.1f}× compression)")
    print(f"    Edge model: {(n_layers - cut_l)}/{n_layers} layers ({(n_layers - cut_l) / n_layers * 100:.0f}%)")
    print(f"    Edge recon: {phase_data['recon_ms']:.0f}ms ({phase_data['recon_ms'] / server_prefill_ms * 100:.0f}% of full prefill)")


if __name__ == "__main__":
    main()
