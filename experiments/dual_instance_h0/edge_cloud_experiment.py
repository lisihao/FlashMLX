#!/usr/bin/env python3
"""
Edge-Cloud Collaboration Experiment: Pipeline split for edge deployment.

Architecture:
  Cloud (GPU):  layers 0..cut-1  → sends h^(cut) to Edge
  Edge (device): layers cut..L-1 → reconstruct KV + decode

  Prefill: Cloud sends h^(cut) once (compressible)
  Decode:  Each token: Edge→Cloud(8KB)→Cloud layers→h^(cut)(8KB)→Edge layers→output

5 Sub-experiments:
  E1: h^(l) quantization (int8/int4) — does LayerNorm absorb quant noise?
  E2: h^(l) SVD low-rank compression
  E3: End-to-end latency model (WiFi/5G/4G/LAN)
  E4: Multi-turn incremental h^(l)
  E5: Edge memory footprint

Usage:
    python3 experiments/dual_instance_h0/edge_cloud_experiment.py \
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
from mlx_lm.models.kv_direct_cache import _find_inner_model, H0Store
from mlx_lm.sample_utils import make_sampler

GREEDY = make_sampler(temp=0.0)


# ════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════

def mem_mb():
    try:
        return mx.metal.get_active_memory() / (1024 * 1024)
    except Exception:
        return 0.0

def peak_mb():
    try:
        return mx.metal.get_peak_memory() / (1024 * 1024)
    except Exception:
        return 0.0

def reset_peak():
    try:
        mx.metal.reset_peak_memory()
    except Exception:
        pass


def capture_all_residuals(inner_model, tokens):
    """Run forward pass capturing residual stream at every layer boundary.
    Returns [h^(0), ..., h^(L)] and KV caches."""
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


def reconstruct_and_generate(model, inner_model, tokenizer, exact_kv, residuals,
                             cut_l, h_cut_override, question, n_layers, max_tokens=50):
    """Reconstruct KV from h^(cut) via sequential forward, then generate.

    Args:
        h_cut_override: the (possibly quantized/compressed) h^(cut) to use
        cut_l: layer to cut at
        exact_kv: list of KVCache from full forward (for layers 0..cut-1)
    Returns:
        gen_tokens, text, recon_ms
    """
    N = h_cut_override.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(h_cut_override.dtype)

    # Client: run layers cut..L-1 from h_cut_override
    t0 = time.perf_counter()
    client_caches = [KVCache() for _ in range(n_layers - cut_l)]
    x = h_cut_override
    for i in range(n_layers - cut_l):
        layer_obj = inner_model.layers[cut_l + i]
        x = layer_obj(x, mask=mask, cache=client_caches[i])
        mx.eval(x)
    recon_ms = (time.perf_counter() - t0) * 1000

    # Assemble full cache: server KV[0:cut] + client KV[cut:L]
    cache_full = make_prompt_cache(model)
    for i in range(cut_l):
        cache_full[i].state = exact_kv[i].state
    for i in range(n_layers - cut_l):
        cache_full[cut_l + i].state = client_caches[i].state
    mx.eval([c.keys for c in cache_full] + [c.values for c in cache_full])

    # Generate
    q = mx.array(tokenizer.encode(question))
    tokens = []
    for tok, _ in generate_step(q, model, max_tokens=max_tokens, sampler=GREEDY, prompt_cache=cache_full):
        tokens.append(tok)
    text = tokenizer.decode(tokens)

    return tokens, text, recon_ms


def detect_mode_shift(residuals, n_tokens, d_model, threshold=3.0):
    """Find the first layer with a major norm jump (mode shift)."""
    norms = []
    for l in range(len(residuals)):
        hl_np = np.array(residuals[l][0].astype(mx.float32))
        norm = np.linalg.norm(hl_np) / np.sqrt(n_tokens * d_model)
        norms.append(norm)

    for l in range(1, len(norms)):
        jump = norms[l] / norms[l - 1] if norms[l - 1] > 0 else 0
        if jump > threshold:
            return l, norms
    # Fallback to 50-50
    return len(residuals) // 2, norms


def quantize_hl(h_l, bits):
    """Quantize h^(l) for transfer simulation. Returns (dequantized, transfer_bytes)."""
    if bits == 16:
        nbytes = h_l.shape[1] * h_l.shape[2] * 2  # bf16
        return h_l, nbytes

    if bits == 8:
        qdata, scales = H0Store._q8_encode(h_l)
        mx.eval(qdata, scales)
        h_deq = H0Store._q8_decode(qdata, scales)
        mx.eval(h_deq)
        # Transfer: int8 data + float16 scales
        nbytes = qdata.nbytes + scales.nbytes
        return h_deq, nbytes

    if bits == 4:
        qdata, scales = H0Store._q4_encode(h_l)
        mx.eval(qdata, scales)
        h_deq = H0Store._q4_decode(qdata, scales)
        mx.eval(h_deq)
        nbytes = qdata.nbytes + scales.nbytes
        return h_deq, nbytes

    raise ValueError(f"Unsupported bits: {bits}")


def svd_compress_hl(h_l_np, rank):
    """SVD compress h^(l) directly. Returns (approximation, transfer_bytes)."""
    # h_l_np: (N, d_model) float32
    U, S, Vt = np.linalg.svd(h_l_np, full_matrices=False)
    r = min(rank, len(S))

    h_approx = (U[:, :r] * S[:r]) @ Vt[:r, :]

    N, d = h_l_np.shape
    # Transfer: coefficients (N × r × 4B) + basis (r × d × 4B)
    coeff_bytes = N * r * 4
    basis_bytes = r * d * 4
    total_bytes = coeff_bytes + basis_bytes

    # Reconstruction error
    err = np.linalg.norm(h_approx - h_l_np) / np.linalg.norm(h_l_np)

    return h_approx, total_bytes, err


# ════════════════════════════════════════════════════════
# Experiment E1: h^(l) Quantization
# ════════════════════════════════════════════════════════

def run_e1_quantization(model, inner_model, tokenizer, residuals, exact_kv,
                        gen_baseline, text_baseline, cut_layers, n_layers, cfg, N):
    print(f"\n{'=' * 100}")
    print(f"  E1: h^(l) QUANTIZATION — does LayerNorm absorb quant noise?")
    print(f"{'=' * 100}")

    d_model = cfg.hidden_size
    results = []

    for cut_l in cut_layers:
        h_cut = residuals[cut_l]  # (1, N, d)

        print(f"\n  ─── Cut at layer {cut_l} ({cut_l}/{n_layers} server | "
              f"{n_layers - cut_l}/{n_layers} edge) ───")

        for bits in [16, 8, 4]:
            h_quantized, xfer_bytes = quantize_hl(h_cut, bits)

            # Measure quantization error
            h_exact_np = np.array(h_cut[0].astype(mx.float32))
            h_quant_np = np.array(h_quantized[0].astype(mx.float32))
            quant_err = np.linalg.norm(h_quant_np - h_exact_np) / np.linalg.norm(h_exact_np)

            # Reconstruct + generate
            gen_tok, text, recon_ms = reconstruct_and_generate(
                model, inner_model, tokenizer, exact_kv, residuals,
                cut_l, h_quantized, "What is artificial intelligence?", n_layers
            )

            token_match = sum(1 for a, b in zip(gen_baseline, gen_tok) if a == b)
            total = min(len(gen_baseline), len(gen_tok))
            exact = gen_baseline == gen_tok

            original_bytes = N * d_model * 2
            compression = original_bytes / xfer_bytes if xfer_bytes > 0 else 1.0

            label = "bf16" if bits == 16 else f"int{bits}"
            print(f"    {label:>5}: transfer {xfer_bytes / 1024 / 1024:.1f} MB "
                  f"({compression:.1f}× vs bf16) | "
                  f"quant err {quant_err:.6f} | "
                  f"match {token_match}/{total} "
                  f"{'EXACT ✓' if exact else 'DIVERGED'} | "
                  f"recon {recon_ms:.0f}ms")

            if not exact and bits == 16:
                # bf16 should be exact — something is wrong
                print(f"    WARNING: bf16 baseline diverged! Check cut point.")

            if not exact:
                print(f"    → {text[:150]}")

            results.append({
                'cut': cut_l, 'bits': bits, 'xfer_mb': xfer_bytes / 1024 / 1024,
                'compression': compression, 'quant_err': quant_err,
                'match': token_match, 'total': total, 'exact': exact,
                'recon_ms': recon_ms,
            })

    return results


# ════════════════════════════════════════════════════════
# Experiment E2: h^(l) SVD Compression
# ════════════════════════════════════════════════════════

def run_e2_svd(model, inner_model, tokenizer, residuals, exact_kv,
               gen_baseline, cut_layers, n_layers, cfg, N):
    print(f"\n{'=' * 100}")
    print(f"  E2: h^(l) SVD LOW-RANK COMPRESSION")
    print(f"{'=' * 100}")

    d_model = cfg.hidden_size
    results = []

    for cut_l in cut_layers:
        h_cut_np = np.array(residuals[cut_l][0].astype(mx.float32))  # (N, d)
        original_bytes = N * d_model * 2

        print(f"\n  ─── Cut at layer {cut_l} (h^({cut_l}) size: {original_bytes / 1024 / 1024:.1f} MB) ───")

        for rank in [64, 128, 256, 512, 1024]:
            if rank > min(N, d_model):
                continue

            h_approx_np, xfer_bytes, svd_err = svd_compress_hl(h_cut_np, rank)

            h_approx_mx = mx.array(h_approx_np.astype(np.float32)).reshape(1, N, d_model)
            # Cast back to model dtype
            h_approx_mx = h_approx_mx.astype(residuals[cut_l].dtype)
            mx.eval(h_approx_mx)

            gen_tok, text, recon_ms = reconstruct_and_generate(
                model, inner_model, tokenizer, exact_kv, residuals,
                cut_l, h_approx_mx, "What is artificial intelligence?", n_layers
            )

            token_match = sum(1 for a, b in zip(gen_baseline, gen_tok) if a == b)
            total = min(len(gen_baseline), len(gen_tok))
            exact = gen_baseline == gen_tok
            compression = original_bytes / xfer_bytes

            print(f"    rank={rank:>4}: transfer {xfer_bytes / 1024 / 1024:.1f} MB "
                  f"({compression:.2f}× vs bf16) | "
                  f"SVD err {svd_err:.6f} | "
                  f"match {token_match}/{total} "
                  f"{'EXACT ✓' if exact else 'DIVERGED'}")

            results.append({
                'cut': cut_l, 'rank': rank, 'xfer_mb': xfer_bytes / 1024 / 1024,
                'compression': compression, 'svd_err': svd_err,
                'match': token_match, 'total': total, 'exact': exact,
            })

    return results


# ════════════════════════════════════════════════════════
# Experiment E3: Latency Model
# ════════════════════════════════════════════════════════

NETWORK_PROFILES = {
    'LAN (1Gbps)':   {'bw_mbps': 1000, 'rtt_ms': 0.5},
    'WiFi (50Mbps)':  {'bw_mbps': 50,   'rtt_ms': 5},
    '5G (100Mbps)':   {'bw_mbps': 100,  'rtt_ms': 20},
    '4G (20Mbps)':    {'bw_mbps': 20,   'rtt_ms': 50},
}

def run_e3_latency(cfg, N, cut_layers, n_layers, e1_results, server_prefill_ms,
                   client_recon_ms_map, tg_ms_per_token):
    print(f"\n{'=' * 100}")
    print(f"  E3: END-TO-END LATENCY MODEL")
    print(f"{'=' * 100}")

    d_model = cfg.hidden_size
    n_kv_heads = cfg.num_key_value_heads
    head_dim = d_model // cfg.num_attention_heads
    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
    full_kv_bytes = kv_per_token * N
    h_per_token_bytes = d_model * 2  # per-token decode transfer

    results = []

    for net_name, profile in NETWORK_PROFILES.items():
        bw = profile['bw_mbps']
        rtt = profile['rtt_ms']

        print(f"\n  ─── {net_name} (RTT={rtt}ms) ───")

        header = f"  {'Scheme':>35} │ {'TTFT':>10} │ {'TG ms/tok':>10} │ {'TG tok/s':>9} │ {'Transfer':>10}"
        print(header)
        print(f"  {'─' * 35}─┼{'─' * 10}─┼{'─' * 10}─┼{'─' * 9}─┼{'─' * 10}")

        def xfer_ms(nbytes):
            return nbytes * 8 / (bw * 1e6) * 1000

        # Scheme A: Full KV transfer (no decode round-trip)
        ttft_a = server_prefill_ms + xfer_ms(full_kv_bytes) + rtt
        tg_a = tg_ms_per_token  # No round-trip — all KV local
        print(f"  {'Full KV transfer':>35} │ {ttft_a:>8.0f}ms │ {tg_a:>8.1f}ms │ "
              f"{1000 / tg_a:>7.1f}  │ {full_kv_bytes / 1024 / 1024:>8.1f} MB")

        # Scheme B: h^(0) + full reconstruction (no decode round-trip possible?)
        # Actually with h^(0), edge does full forward → no round-trip needed for decode
        h0_bytes = d_model * 2 * N
        ttft_b = server_prefill_ms + xfer_ms(h0_bytes) + rtt + client_recon_ms_map.get(0, 0)
        tg_b = tg_ms_per_token  # All layers on edge
        print(f"  {'h^(0) + full recon (edge all)':>35} │ {ttft_b:>8.0f}ms │ {tg_b:>8.1f}ms │ "
              f"{1000 / tg_b:>7.1f}  │ {h0_bytes / 1024 / 1024:>8.1f} MB")

        # Scheme C/D: Pipeline split at various cut points
        for cut_l in cut_layers:
            # Transfer: h^(cut) bf16
            h_cut_bytes = d_model * 2 * N
            kv_prefix_bytes = 2 * cut_l * n_kv_heads * head_dim * 2 * N
            total_xfer = h_cut_bytes + kv_prefix_bytes

            # TTFT: server prefill + transfer + RTT + client recon
            client_recon = client_recon_ms_map.get(cut_l, 0)
            ttft_c = server_prefill_ms + xfer_ms(total_xfer) + rtt + client_recon

            # TG: each token round-trips for cloud layers
            tg_c = tg_ms_per_token + rtt  # RTT per token

            label = f"Split@{cut_l} (bf16 h^({cut_l})+KV[0:{cut_l}])"
            print(f"  {label:>35} │ {ttft_c:>8.0f}ms │ {tg_c:>8.1f}ms │ "
                  f"{1000 / tg_c:>7.1f}  │ {total_xfer / 1024 / 1024:>8.1f} MB")

            # Scheme D: Pipeline split + int4 h^(cut)
            # Find int4 result from E1
            int4_bytes = h_cut_bytes // 4 + N * 2  # packed + scales
            total_xfer_int4 = int4_bytes + kv_prefix_bytes

            ttft_d = server_prefill_ms + xfer_ms(total_xfer_int4) + rtt + client_recon
            tg_d = tg_ms_per_token + rtt

            label_d = f"Split@{cut_l} (int4 h^({cut_l})+KV[0:{cut_l}])"
            print(f"  {label_d:>35} │ {ttft_d:>8.0f}ms │ {tg_d:>8.1f}ms │ "
                  f"{1000 / tg_d:>7.1f}  │ {total_xfer_int4 / 1024 / 1024:>8.1f} MB")

            # Scheme E: Split WITHOUT sending KV prefix (edge ignores layers 0..cut-1)
            # Decode: edge only runs layers cut..L-1 → no round-trip!
            # But needs full model on edge for token embedding... and cloud KV during decode
            # This is the "pure edge after split" model
            # For TTFT: only h^(cut) transfer
            total_xfer_no_kv = h_cut_bytes  # Only h^(cut)
            ttft_e = server_prefill_ms + xfer_ms(total_xfer_no_kv) + rtt + client_recon
            # For TG: still need round-trip for layers 0..cut-1
            tg_e = tg_ms_per_token + rtt

            results.append({
                'net': net_name, 'cut': cut_l,
                'ttft_bf16': ttft_c, 'ttft_int4': ttft_d,
                'tg_ms': tg_c, 'tg_tok_s': 1000 / tg_c,
            })

    return results


# ════════════════════════════════════════════════════════
# Experiment E4: Multi-Turn Incremental h^(l)
# ════════════════════════════════════════════════════════

def run_e4_incremental(model, inner_model, tokenizer, n_layers, cfg, prompt_tokens, cut_layers):
    print(f"\n{'=' * 100}")
    print(f"  E4: MULTI-TURN INCREMENTAL h^(l)")
    print(f"{'=' * 100}")

    d_model = cfg.hidden_size

    # Build two prompts: T1 (base) and T2 (base + question)
    FILLER = ("The development of artificial intelligence has progressed rapidly in recent years. "
              "Machine learning models continue to grow in capability and efficiency. "
              "Large language models can now understand and generate human-like text. "
              "Researchers explore new architectures to push boundaries further. ") * 100

    tokens_base = tokenizer.encode(FILLER)[:prompt_tokens]
    question_text = " What is artificial intelligence?"
    tokens_question = tokenizer.encode(question_text)
    tokens_full = tokens_base + tokens_question

    t1 = mx.array(tokens_base)
    t2 = mx.array(tokens_full)
    N1 = len(tokens_base)
    N2 = len(tokens_full)
    N_delta = N2 - N1

    print(f"\n  Turn 1: {N1} tokens | Turn 2: {N2} tokens (delta: {N_delta} tokens)")

    # Full forward for T1 and T2
    print("  Computing T1 residuals...", file=sys.stderr)
    res_t1, kv_t1 = capture_all_residuals(inner_model, t1)
    print("  Computing T2 residuals...", file=sys.stderr)
    res_t2, kv_t2 = capture_all_residuals(inner_model, t2)

    # Baseline: generate from full T2
    cache_bl = make_prompt_cache(model)
    for i in range(n_layers):
        cache_bl[i].state = kv_t2[i].state
    mx.eval([c.keys for c in cache_bl] + [c.values for c in cache_bl])

    gen_bl, text_bl = [], ""
    q_empty = mx.array(tokenizer.encode(""))  # Empty — context already in cache
    # Generate from the full cache (question is already in the prompt)
    # Actually, we need to generate a continuation
    gen_q = mx.array(tokenizer.encode(" The answer is:"))
    for tok, _ in generate_step(gen_q, model, max_tokens=50, sampler=GREEDY, prompt_cache=cache_bl):
        gen_bl.append(tok)
    text_bl = tokenizer.decode(gen_bl)

    for cut_l in cut_layers:
        h_t1 = res_t1[cut_l]  # (1, N1, d)
        h_t2 = res_t2[cut_l]  # (1, N2, d)

        # Incremental: T1's h^(cut)[0:N1] + T2's h^(cut)[N1:N2]
        h_delta = h_t2[:, N1:, :]  # (1, N_delta, d) — only new tokens
        h_combined = mx.concatenate([h_t1, h_delta], axis=1)  # (1, N2, d)
        mx.eval(h_combined)

        # Check if incremental == full T2's h^(cut)
        h_full = h_t2
        diff = mx.abs(h_combined - h_full).max().item()
        bit_exact = diff == 0.0

        # Verify generation
        gen_inc, text_inc, recon_ms = reconstruct_and_generate(
            model, inner_model, tokenizer, kv_t2, res_t2,
            cut_l, h_combined, " The answer is:", n_layers
        )

        token_match = sum(1 for a, b in zip(gen_bl, gen_inc) if a == b)
        total = min(len(gen_bl), len(gen_inc))
        gen_exact = gen_bl == gen_inc

        xfer_delta = N_delta * d_model * 2
        xfer_full = N2 * d_model * 2
        savings = 1.0 - xfer_delta / xfer_full

        print(f"\n  Cut@{cut_l}: h^({cut_l}) incremental")
        print(f"    h^(cut) bit-exact: {'YES ✓' if bit_exact else f'NO (max diff={diff:.2e})'}")
        print(f"    Delta transfer: {xfer_delta / 1024:.1f} KB "
              f"(vs full {xfer_full / 1024 / 1024:.1f} MB = {savings * 100:.0f}% saved)")
        print(f"    Generation: {token_match}/{total} "
              f"{'EXACT ✓' if gen_exact else 'DIVERGED'}")


# ════════════════════════════════════════════════════════
# Experiment E5: Edge Memory Footprint
# ════════════════════════════════════════════════════════

def run_e5_memory(model, inner_model, cfg, N, cut_layers, n_layers):
    print(f"\n{'=' * 100}")
    print(f"  E5: EDGE MEMORY FOOTPRINT")
    print(f"{'=' * 100}")

    d_model = cfg.hidden_size
    n_kv_heads = cfg.num_key_value_heads
    head_dim = d_model // cfg.num_attention_heads

    # Count model parameters by layer group
    params = dict(model.parameters())

    # Flatten nested params
    def flatten_params(d, prefix=""):
        flat = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_params(v, key))
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        flat.update(flatten_params(item, f"{key}.{i}"))
                    elif hasattr(item, 'nbytes'):
                        flat[f"{key}.{i}"] = item
            elif hasattr(v, 'nbytes'):
                flat[key] = v
        return flat

    flat = flatten_params(params)

    total_bytes = sum(v.nbytes for v in flat.values())
    embed_bytes = sum(v.nbytes for k, v in flat.items()
                      if 'embed' in k)
    lm_head_bytes = sum(v.nbytes for k, v in flat.items()
                        if 'lm_head' in k or 'head' in k.split('.')[-1])
    norm_bytes = sum(v.nbytes for k, v in flat.items()
                     if 'norm' in k and 'layers' not in k)

    print(f"\n  Full model: {total_bytes / 1024 / 1024:.0f} MB")
    print(f"  embed_tokens: {embed_bytes / 1024 / 1024:.0f} MB")
    print(f"  lm_head + final norm: {(lm_head_bytes + norm_bytes) / 1024 / 1024:.0f} MB")

    kv_per_layer = 2 * n_kv_heads * head_dim * 2 * N  # bytes

    print(f"\n  {'Cut':>4} │ {'Edge layers':>12} │ {'Edge weights':>12} │ {'Edge KV':>10} │ "
          f"{'Edge total':>10} │ {'vs Full':>8}")
    print(f"  {'─' * 4}─┼{'─' * 12}─┼{'─' * 12}─┼{'─' * 10}─┼{'─' * 10}─┼{'─' * 8}")

    for cut_l in cut_layers:
        # Edge needs: layers cut..L-1 + embed + lm_head + norm
        edge_layer_bytes = 0
        for k, v in flat.items():
            if 'layers' in k:
                # Extract layer number
                parts = k.split('.')
                for i, p in enumerate(parts):
                    if p == 'layers' and i + 1 < len(parts):
                        try:
                            layer_num = int(parts[i + 1])
                            if layer_num >= cut_l:
                                edge_layer_bytes += v.nbytes
                        except ValueError:
                            pass
                        break

        edge_weights = edge_layer_bytes + embed_bytes + lm_head_bytes + norm_bytes
        edge_kv = kv_per_layer * (n_layers - cut_l)
        edge_total = edge_weights + edge_kv

        full_total = total_bytes + kv_per_layer * n_layers
        ratio = edge_total / full_total

        print(f"  {cut_l:>4} │ {n_layers - cut_l:>10}L  │ "
              f"{edge_weights / 1024 / 1024:>10.0f} MB│ {edge_kv / 1024 / 1024:>8.1f} MB│ "
              f"{edge_total / 1024 / 1024:>8.0f} MB│ {ratio:>6.0%}")


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--cut-layer", type=int, default=-1,
                        help="Layer to cut at (-1 = auto-detect mode shift)")
    parser.add_argument("--experiments", default="all",
                        help="Comma-separated: e1,e2,e3,e4,e5 or 'all'")
    args = parser.parse_args()

    exps = set(args.experiments.split(",")) if args.experiments != "all" else {"e1", "e2", "e3", "e4", "e5"}

    print("Loading model...", file=sys.stderr)
    model, tokenizer = load(args.model)
    cfg = model.args
    n_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    n_kv_heads = cfg.num_key_value_heads
    n_q_heads = cfg.num_attention_heads
    head_dim = d_model // n_q_heads

    inner_model = _find_inner_model(model)

    # Warmup
    model(mx.array(tokenizer.encode("Hello")).reshape(1, -1))
    mx.eval(model.parameters())

    print(f"Model: {n_layers}L, d={d_model}, {n_q_heads}Q/{n_kv_heads}KV", file=sys.stderr)

    # ═══ Baseline capture ═══
    FILLER = ("The development of artificial intelligence has progressed rapidly in recent years. "
              "Machine learning models continue to grow in capability and efficiency. "
              "Large language models can now understand and generate human-like text. "
              "Researchers explore new architectures to push boundaries further. ") * 100
    tokens = mx.array(tokenizer.encode(FILLER)[:args.prompt_tokens])
    N = tokens.shape[0]
    print(f"Tokens: {N}", file=sys.stderr)

    print("\nCapturing full forward pass...", file=sys.stderr)
    t0 = time.perf_counter()
    residuals, exact_kv = capture_all_residuals(inner_model, tokens)
    server_prefill_ms = (time.perf_counter() - t0) * 1000

    # Baseline generation
    cache_bl = make_prompt_cache(model)
    for i in range(n_layers):
        cache_bl[i].state = exact_kv[i].state
    mx.eval([c.keys for c in cache_bl] + [c.values for c in cache_bl])

    question = "What is artificial intelligence?"
    q_mx = mx.array(tokenizer.encode(question))
    gen_baseline = []
    t0 = time.perf_counter()
    for tok, _ in generate_step(q_mx, model, max_tokens=50, sampler=GREEDY, prompt_cache=cache_bl):
        gen_baseline.append(tok)
    tg_total_ms = (time.perf_counter() - t0) * 1000
    text_baseline = tokenizer.decode(gen_baseline)
    tg_ms_per_token = tg_total_ms / max(1, len(gen_baseline))

    print(f"Server prefill: {server_prefill_ms:.0f}ms | TG: {tg_ms_per_token:.1f}ms/tok",
          file=sys.stderr)
    print(f"Baseline: {text_baseline[:150]}", file=sys.stderr)

    # ═══ Detect mode shift ═══
    mode_shift_l, norms = detect_mode_shift(residuals, N, d_model)
    half_l = n_layers // 2

    if args.cut_layer >= 0:
        cut_layers = [args.cut_layer]
    else:
        cut_layers = sorted(set([mode_shift_l, half_l]))

    print(f"\n{'=' * 100}")
    print(f"  EDGE-CLOUD EXPERIMENT SUITE")
    print(f"  Model: {n_layers}L, d={d_model} | Tokens: {N}")
    print(f"  Mode shift detected at layer {mode_shift_l} (RMS jump {norms[mode_shift_l] / norms[mode_shift_l - 1]:.1f}×)")
    print(f"  Cut points to test: {cut_layers}")
    print(f"  Server prefill: {server_prefill_ms:.0f}ms | TG: {tg_ms_per_token:.1f}ms/tok")
    print(f"{'=' * 100}")

    # Measure client recon time for each cut point
    client_recon_ms_map = {}
    for cut_l in cut_layers:
        h_cut = residuals[cut_l]
        _, _, recon_ms = reconstruct_and_generate(
            model, inner_model, tokenizer, exact_kv, residuals,
            cut_l, h_cut, question, n_layers
        )
        client_recon_ms_map[cut_l] = recon_ms
    # Also for h^(0) full recon
    h0 = residuals[0]
    _, _, recon_0 = reconstruct_and_generate(
        model, inner_model, tokenizer, exact_kv, residuals,
        0, h0, question, n_layers
    )
    client_recon_ms_map[0] = recon_0

    # ═══ Run experiments ═══
    e1_results = []
    if "e1" in exps:
        e1_results = run_e1_quantization(model, inner_model, tokenizer, residuals, exact_kv,
                                         gen_baseline, text_baseline, cut_layers, n_layers, cfg, N)

    if "e2" in exps:
        run_e2_svd(model, inner_model, tokenizer, residuals, exact_kv,
                   gen_baseline, cut_layers, n_layers, cfg, N)

    if "e3" in exps:
        run_e3_latency(cfg, N, cut_layers, n_layers, e1_results, server_prefill_ms,
                       client_recon_ms_map, tg_ms_per_token)

    if "e4" in exps:
        run_e4_incremental(model, inner_model, tokenizer, n_layers, cfg,
                           args.prompt_tokens, cut_layers)

    if "e5" in exps:
        run_e5_memory(model, inner_model, cfg, N, cut_layers, n_layers)

    # ═══ Final summary ═══
    print(f"\n{'=' * 100}")
    print(f"  FINAL SUMMARY — EDGE-CLOUD VIABILITY")
    print(f"{'=' * 100}")

    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
    full_kv_mb = kv_per_token * N / 1024 / 1024

    for cut_l in cut_layers:
        h_bytes = d_model * 2 * N
        kv_prefix = 2 * cut_l * n_kv_heads * head_dim * 2 * N

        print(f"\n  ─── Cut at layer {cut_l}: Cloud {cut_l}L / Edge {n_layers - cut_l}L ───")
        print(f"    Full KV:     {full_kv_mb:.1f} MB")
        print(f"    h^({cut_l}) bf16:  {h_bytes / 1024 / 1024:.1f} MB")
        print(f"    h^({cut_l}) int8:  {h_bytes / 2 / 1024 / 1024:.1f} MB")
        print(f"    h^({cut_l}) int4:  {h_bytes / 4 / 1024 / 1024:.1f} MB")
        print(f"    KV[0:{cut_l}]:    {kv_prefix / 1024 / 1024:.1f} MB")

        # Best viable scheme from E1
        if e1_results:
            viable = [r for r in e1_results if r['cut'] == cut_l and r['match'] >= 40]
            if viable:
                best = min(viable, key=lambda r: r['xfer_mb'])
                label = f"int{best['bits']}" if best['bits'] < 16 else "bf16"
                print(f"    Best viable: {label} ({best['match']}/{best['total']} match, "
                      f"{best['xfer_mb']:.1f} MB)")

        print(f"    Client recon: {client_recon_ms_map.get(cut_l, 0):.0f}ms")
        print(f"    Decode overhead: +RTT per token (WiFi ~5ms, 5G ~20ms)")


if __name__ == "__main__":
    main()
