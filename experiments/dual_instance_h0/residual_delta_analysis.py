#!/usr/bin/env python3
"""
Residual Delta Analysis: Can we compress h^(l) - h^(0) ?

Core hypothesis:
  h^(l) = h^(0) + δ^(l)
  If δ^(l) is low-rank, we can send h^(0) + compressed deltas
  instead of full KV cache OR full reconstruction.

This script measures:
  1. ||δ^(l)|| / ||h^(l)|| for each layer (how much does residual change?)
  2. SVD spectrum of δ^(l) (is it low-rank?)
  3. Reconstruction quality at various ranks
  4. KV error when using approximate h^(l) from h^(0) + low-rank delta
  5. Transfer size comparison at quality targets

Usage:
    python3 experiments/dual_instance_h0/residual_delta_analysis.py \
        --model /path/to/Qwen3-8B-MLX-4bit --prompt-tokens 2048
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
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.kv_direct_cache import (
    _find_inner_model,
    reconstruct_prefix_kv,
    H0Store,
)


def capture_all_residuals(inner_model, tokens):
    """Run forward pass capturing residual stream at every layer boundary.

    Returns list of h^(0), h^(1), ..., h^(L) as mx arrays, each (1, N, d).
    """
    x = inner_model.embed_tokens(tokens.reshape(1, -1))
    mx.eval(x)
    residuals = [x]  # h^(0)

    N = tokens.shape[0]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(x.dtype)

    # Create temp KV caches
    num_layers = len(inner_model.layers)
    cache = [KVCache() for _ in range(num_layers)]

    for i, layer in enumerate(inner_model.layers):
        x = layer(x, mask=mask, cache=cache[i])
        mx.eval(x)
        residuals.append(x)  # h^(i+1)

    return residuals, cache


def svd_analysis(delta_np, max_rank=None):
    """Compute SVD of delta matrix and return singular values + cumulative energy."""
    # delta_np: (N, d_model)
    U, S, Vt = np.linalg.svd(delta_np, full_matrices=False)
    total_energy = np.sum(S ** 2)
    cumulative = np.cumsum(S ** 2) / total_energy
    return S, cumulative, U, Vt


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    args = parser.parse_args()

    print("Loading model...", file=sys.stderr)
    model, tokenizer = load(args.model)
    cfg = model.args
    n_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    n_kv_heads = cfg.num_key_value_heads
    head_dim = d_model // cfg.num_attention_heads

    print(f"Model: {n_layers}L, d={d_model}, {cfg.num_attention_heads}Q/{n_kv_heads}KV", file=sys.stderr)

    inner_model = _find_inner_model(model)

    # Warmup
    model(mx.array(tokenizer.encode("Hello")).reshape(1, -1))
    mx.eval(model.parameters())

    # Build tokens
    FILLER = ("The development of artificial intelligence has progressed rapidly. "
              "Machine learning models grow in capability. "
              "Large language models understand and generate text fluently. "
              "Researchers explore new architectures and training paradigms. ") * 50
    tokens = mx.array(tokenizer.encode(FILLER)[:args.prompt_tokens])
    N = tokens.shape[0]
    print(f"Tokens: {N}", file=sys.stderr)

    # ════════════════════════════════════════════════════════
    # Step 1: Capture all layer residuals
    # ════════════════════════════════════════════════════════
    print("\nCapturing residuals at every layer...", file=sys.stderr)
    t0 = time.perf_counter()
    residuals, kv_caches = capture_all_residuals(inner_model, tokens)
    t_capture = (time.perf_counter() - t0) * 1000
    print(f"Captured {len(residuals)} residuals in {t_capture:.0f}ms", file=sys.stderr)

    h0 = residuals[0]  # (1, N, d_model)
    h0_np = np.array(h0[0].astype(mx.float32))  # (N, d_model)

    # ════════════════════════════════════════════════════════
    # Step 2: Delta analysis per layer
    # ════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  RESIDUAL DELTA ANALYSIS: h^(l) = h^(0) + δ^(l)")
    print(f"{'=' * 100}")

    print(f"\n  {'Layer':>5} │ {'||h^(l)||':>10} │ {'||δ^(l)||':>10} │ {'δ/h ratio':>9} │ "
          f"{'rank@95%':>8} │ {'rank@99%':>8} │ {'rank@99.9%':>10} │ {'rank@99.99%':>11}")
    print(f"  {'─' * 5}─┼{'─' * 10}─┼{'─' * 10}─┼{'─' * 9}─┼{'─' * 8}─┼{'─' * 8}─┼{'─' * 10}─┼{'─' * 11}")

    layer_stats = []
    for l in range(1, len(residuals)):
        hl_np = np.array(residuals[l][0].astype(mx.float32))  # (N, d_model)
        delta_np = hl_np - h0_np  # (N, d_model)

        h_norm = np.linalg.norm(hl_np)
        d_norm = np.linalg.norm(delta_np)
        ratio = d_norm / h_norm if h_norm > 0 else 0

        S, cumulative, U, Vt = svd_analysis(delta_np)

        # Find ranks for various quality targets
        rank_95 = int(np.searchsorted(cumulative, 0.95)) + 1
        rank_99 = int(np.searchsorted(cumulative, 0.99)) + 1
        rank_999 = int(np.searchsorted(cumulative, 0.999)) + 1
        rank_9999 = int(np.searchsorted(cumulative, 0.9999)) + 1

        print(f"  {l:>5} │ {h_norm:>10.1f} │ {d_norm:>10.1f} │ {ratio:>8.3f}  │ "
              f"{rank_95:>8} │ {rank_99:>8} │ {rank_999:>10} │ {rank_9999:>11}")

        layer_stats.append({
            'layer': l,
            'h_norm': h_norm,
            'd_norm': d_norm,
            'ratio': ratio,
            'rank_95': rank_95,
            'rank_99': rank_99,
            'rank_999': rank_999,
            'rank_9999': rank_9999,
            'S': S,
            'cumulative': cumulative,
            'U': U,
            'Vt': Vt,
        })

    # ════════════════════════════════════════════════════════
    # Step 3: Summary statistics
    # ════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  SUMMARY")
    print(f"{'=' * 100}")

    avg_ratio = np.mean([s['ratio'] for s in layer_stats])
    max_ratio = max(s['ratio'] for s in layer_stats)
    avg_r99 = np.mean([s['rank_99'] for s in layer_stats])
    max_r99 = max(s['rank_99'] for s in layer_stats)
    avg_r999 = np.mean([s['rank_999'] for s in layer_stats])
    max_r999 = max(s['rank_999'] for s in layer_stats)

    print(f"\n  Delta magnitude: avg={avg_ratio:.3f}, max={max_ratio:.3f} of ||h^(l)||")
    print(f"  Rank@99%:  avg={avg_r99:.0f}, max={max_r99}")
    print(f"  Rank@99.9%: avg={avg_r999:.0f}, max={max_r999}")

    # ════════════════════════════════════════════════════════
    # Step 4: Transfer size comparison at various ranks
    # ════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  TRANSFER SIZE COMPARISON (N={N} tokens, d={d_model})")
    print(f"{'=' * 100}")

    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
    kv_total = kv_per_token * N
    h0_total = d_model * 2 * N

    print(f"\n  Full KV cache:        {kv_total / 1024 / 1024:>8.1f} MB  (baseline)")
    print(f"  int4 KV:              {kv_total / 4 / 1024 / 1024:>8.1f} MB  (4× compression)")
    print(f"  h^(0) only:           {h0_total / 1024 / 1024:>8.1f} MB  (needs full recon)")

    print(f"\n  h^(0) + SVD deltas:")
    for rank in [8, 16, 32, 64, 128, 256]:
        # Per layer: send U×S coefficients = N × rank × 4 bytes (float32)
        # V^T basis: rank × d_model × 4 bytes (precomputed, cached on decode side)
        delta_coeff = n_layers * N * rank * 4  # U×S for all layers
        basis_size = n_layers * rank * d_model * 4  # V^T (one-time, cached)
        total_transfer = h0_total + delta_coeff
        total_with_basis = total_transfer + basis_size

        # Quality: use max rank needed across layers
        quality_layers_99 = sum(1 for s in layer_stats if s['rank_99'] <= rank)
        quality_layers_999 = sum(1 for s in layer_stats if s['rank_999'] <= rank)

        compression = kv_total / total_transfer

        print(f"    rank={rank:>3}: transfer={total_transfer / 1024 / 1024:>6.1f} MB "
              f"(+basis {basis_size / 1024 / 1024:.1f} MB cached) "
              f"│ {compression:>5.1f}× vs KV "
              f"│ layers@99%: {quality_layers_99}/{n_layers} "
              f"│ layers@99.9%: {quality_layers_999}/{n_layers}")

    # ════════════════════════════════════════════════════════
    # Step 5: KV reconstruction quality test
    # ════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  KV RECONSTRUCTION QUALITY: exact vs h^(0) + low-rank delta")
    print(f"{'=' * 100}")

    # Get exact KV from the captured caches
    print(f"\n  Testing KV quality at different ranks...")

    for rank in [16, 32, 64, 128, 256]:
        kv_errors = []
        for l in range(n_layers):
            stats = layer_stats[l]
            U = stats['U']
            S = stats['S']
            Vt = stats['Vt']

            # Low-rank approximation of delta
            r = min(rank, len(S))
            delta_approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
            hl_approx = h0_np + delta_approx

            # True h^(l)
            hl_true = np.array(residuals[l + 1][0].astype(mx.float32))

            # Relative error
            err = np.linalg.norm(hl_approx - hl_true) / np.linalg.norm(hl_true)
            kv_errors.append(err)

            # Also compute actual KV error
            # K_true = norm(h^(l)) @ W_K
            # K_approx = norm(h^(l)_approx) @ W_K
            # But we'd need the actual weight matrices...

        avg_err = np.mean(kv_errors)
        max_err = max(kv_errors)
        worst_layer = np.argmax(kv_errors) + 1

        print(f"    rank={rank:>3}: h^(l) relative error avg={avg_err:.6f}, "
              f"max={max_err:.6f} (layer {worst_layer})")

    # ════════════════════════════════════════════════════════
    # Step 6: End-to-end quality test — generate with approximate KV
    # ════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  END-TO-END GENERATION TEST: exact KV vs approximate KV")
    print(f"{'=' * 100}")

    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    GREEDY = make_sampler(temp=0.0)

    # Baseline: exact KV from full forward
    h0_store = H0Store()
    h0_store.append(h0)
    kv_exact = reconstruct_prefix_kv(inner_model, h0_store, 0, N, chunk_size=512, eval_every=8)
    mx.eval(*[k for k, v in kv_exact] + [v for k, v in kv_exact])

    cache_exact = make_prompt_cache(model)
    for i, (k, v) in enumerate(kv_exact):
        cache_exact[i].state = (k, v)
    mx.eval([c.keys for c in cache_exact] + [c.values for c in cache_exact])

    question = mx.array(tokenizer.encode("What is artificial intelligence?"))
    gen_exact = []
    for tok, _ in generate_step(question, model, max_tokens=50, sampler=GREEDY, prompt_cache=cache_exact):
        gen_exact.append(tok)
    text_exact = tokenizer.decode(gen_exact)

    print(f"\n  Exact KV: {text_exact[:200]}")

    # Test at various ranks
    for rank in [32, 64, 128, 256]:
        # Reconstruct approximate h^(l) from h^(0) + low-rank deltas
        # Layer l uses h^(l) = residuals[l] as INPUT
        # delta for layer l = residuals[l] - h^(0) = layer_stats[l-1] (for l>0)
        kv_pairs_approx = []
        for l in range(n_layers):
            if l == 0:
                # Layer 0 uses h^(0) directly — EXACT, no approximation needed
                hl_approx_np = h0_np
            else:
                stats = layer_stats[l - 1]  # SVD of h^(l) - h^(0)
                U = stats['U']
                S = stats['S']
                Vt = stats['Vt']

                r = min(rank, len(S))
                delta_approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
                hl_approx_np = h0_np + delta_approx

            # Convert back to mx and project to KV
            hl_approx = mx.array(hl_approx_np.astype(np.float32)).reshape(1, N, d_model)

            # Get the actual KV projection weights from the layer
            layer_obj = inner_model.layers[l]
            attn = layer_obj.self_attn

            # Apply RMSNorm (input_layernorm)
            norm = layer_obj.input_layernorm
            hl_normed = norm(hl_approx.astype(mx.float32))

            # Project to K, V
            k_proj = attn.k_proj(hl_normed)
            v_proj = attn.v_proj(hl_normed)

            # Reshape to (1, n_kv_heads, N, head_dim)
            k_proj = k_proj.reshape(1, N, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
            v_proj = v_proj.reshape(1, N, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

            kv_pairs_approx.append((k_proj, v_proj))

        mx.eval(*[k for k, v in kv_pairs_approx] + [v for k, v in kv_pairs_approx])

        cache_approx = make_prompt_cache(model)
        for i, (k, v) in enumerate(kv_pairs_approx):
            cache_approx[i].state = (k, v)
        mx.eval([c.keys for c in cache_approx] + [c.values for c in cache_approx])

        gen_approx = []
        for tok, _ in generate_step(question, model, max_tokens=50, sampler=GREEDY, prompt_cache=cache_approx):
            gen_approx.append(tok)
        text_approx = tokenizer.decode(gen_approx)

        match = gen_exact[:20] == gen_approx[:20]
        token_match = sum(1 for a, b in zip(gen_exact, gen_approx) if a == b)

        # Transfer size
        delta_coeff = n_layers * N * rank * 4
        total_transfer = d_model * 2 * N + delta_coeff
        compression = kv_per_token * N / total_transfer

        print(f"\n  rank={rank:>3} ({total_transfer / 1024 / 1024:.1f} MB, {compression:.1f}× vs KV):")
        print(f"    Output: {text_approx[:200]}")
        print(f"    Token match: {token_match}/{min(len(gen_exact), len(gen_approx))} "
              f"({'EXACT' if match else 'DIVERGED'})")

    # ════════════════════════════════════════════════════════
    # Step 7: Compute-time comparison
    # ════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  RECONSTRUCTION SPEED: full forward vs h^(0) + delta projection")
    print(f"{'=' * 100}")

    # Full reconstruction
    h0_store2 = H0Store()
    h0_store2.append(h0)
    t0 = time.perf_counter()
    kv_full = reconstruct_prefix_kv(inner_model, h0_store2, 0, N, chunk_size=512, eval_every=8)
    mx.eval(*[k for k, v in kv_full] + [v for k, v in kv_full])
    t_full = (time.perf_counter() - t0) * 1000

    # Delta projection reconstruction (rank 64)
    rank = 64
    t0 = time.perf_counter()
    for l in range(n_layers):
        if l == 0:
            hl_approx_np = h0_np
        else:
            stats = layer_stats[l - 1]
            r = min(rank, len(stats['S']))
            delta_approx = (stats['U'][:, :r] * stats['S'][:r]) @ stats['Vt'][:r, :]
            hl_approx_np = h0_np + delta_approx
        hl_approx = mx.array(hl_approx_np.astype(np.float32)).reshape(1, N, d_model)

        layer_obj = inner_model.layers[l]
        hl_normed = layer_obj.input_layernorm(hl_approx)
        k = layer_obj.self_attn.k_proj(hl_normed)
        v = layer_obj.self_attn.v_proj(hl_normed)
        mx.eval(k, v)
    t_delta = (time.perf_counter() - t0) * 1000

    print(f"\n  Full reconstruction ({n_layers} layers): {t_full:.0f}ms")
    print(f"  Delta projection (rank={rank}):          {t_delta:.0f}ms")
    print(f"  Speedup: {t_full / t_delta:.1f}×")


if __name__ == "__main__":
    main()
