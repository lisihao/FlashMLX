#!/usr/bin/env python3
"""
Pipeline Split Experiment: h^(l) as the transfer cut point.

Core hypothesis:
  Layers 0-2 perform a "mode shift" (14× norm jump at layer 2→3).
  If we cut at layer 3:
    - Server: embed + layers 0-2 → sends h^(3) + KV[0-2]
    - Client: layers 3-35 → exact KV[3-35]
  Transfer: h^(3) (32 MB) + KV[0-2] (48 MB) = 80 MB vs 576 MB full KV = 7.2× compression

This script tests:
  1. Exact pipeline split at different layers — verify generation EXACT MATCH
  2. Delta compressibility: δ^(j) = h^(j) - h^(base) for various base layers
     → Is δ from h^(3) more compressible than δ from h^(0)?
  3. Transfer size / compute tradeoff at each cut point
  4. End-to-end generation quality with approximate KV from different bases

Usage:
    python3 experiments/dual_instance_h0/pipeline_split_experiment.py \
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


def capture_all_residuals(inner_model, tokens):
    """Run forward pass capturing residual stream at every layer boundary.

    Returns:
        residuals: [h^(0), h^(1), ..., h^(L)] each (1, N, d)
        kv_caches: list of KVCache objects (post-RoPE K/V stored)
    """
    x = inner_model.embed_tokens(tokens.reshape(1, -1))
    mx.eval(x)
    residuals = [x]  # h^(0)

    N = tokens.shape[0]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(x.dtype)

    num_layers = len(inner_model.layers)
    cache = [KVCache() for _ in range(num_layers)]

    for i, layer in enumerate(inner_model.layers):
        x = layer(x, mask=mask, cache=cache[i])
        mx.eval(x)
        residuals.append(x)  # h^(i+1)

    return residuals, cache


def reconstruct_from_layer(inner_model, h_start, start_layer, kv_caches_prefix):
    """Run forward pass from layer start_layer to last layer, starting with h_start.

    Simulates the "client" side of a pipeline split:
    - layers 0..start_layer-1 are done by server (KV in kv_caches_prefix)
    - layers start_layer..L-1 are done here from h_start

    Returns:
        kv_caches: fresh caches for layers start_layer..L-1
    """
    N = h_start.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(h_start.dtype)

    num_layers = len(inner_model.layers)
    client_caches = [KVCache() for _ in range(num_layers - start_layer)]

    x = h_start
    for i, layer_idx in enumerate(range(start_layer, num_layers)):
        layer = inner_model.layers[layer_idx]
        x = layer(x, mask=mask, cache=client_caches[i])
        mx.eval(x)

    return client_caches


def generate_text(model, tokenizer, cache, question_text, max_tokens=50):
    """Generate text with a given cache."""
    q = mx.array(tokenizer.encode(question_text))
    tokens = []
    for tok, _ in generate_step(q, model, max_tokens=max_tokens, sampler=GREEDY, prompt_cache=cache):
        tokens.append(tok)
    return tokens, tokenizer.decode(tokens)


def svd_analysis(delta_np):
    """Compute SVD of delta matrix and return singular values + cumulative energy."""
    U, S, Vt = np.linalg.svd(delta_np, full_matrices=False)
    total_energy = np.sum(S ** 2)
    cumulative = np.cumsum(S ** 2) / total_energy if total_energy > 0 else np.ones_like(S)
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
    n_q_heads = cfg.num_attention_heads
    head_dim = d_model // n_q_heads

    print(f"Model: {n_layers}L, d={d_model}, {n_q_heads}Q/{n_kv_heads}KV, head_dim={head_dim}",
          file=sys.stderr)

    inner_model = _find_inner_model(model)

    # Warmup
    model(mx.array(tokenizer.encode("Hello")).reshape(1, -1))
    mx.eval(model.parameters())

    # Build tokens
    FILLER = ("The development of artificial intelligence has progressed rapidly in recent years. "
              "Machine learning models continue to grow in capability and efficiency. "
              "Large language models can now understand and generate human-like text. "
              "Researchers explore new architectures to push boundaries further. ") * 100
    tokens = mx.array(tokenizer.encode(FILLER)[:args.prompt_tokens])
    N = tokens.shape[0]
    print(f"Tokens: {N}", file=sys.stderr)

    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
    kv_per_layer_token = 2 * n_kv_heads * head_dim * 2
    h_per_token = d_model * 2

    # ════════════════════════════════════════════════════════
    # Part 1: Capture baseline — full forward with exact KV
    # ════════════════════════════════════════════════════════
    print("\nCapturing full forward pass...", file=sys.stderr)
    t0 = time.perf_counter()
    residuals, exact_kv = capture_all_residuals(inner_model, tokens)
    t_full = (time.perf_counter() - t0) * 1000
    print(f"Full forward: {t_full:.0f}ms", file=sys.stderr)

    # Baseline generation
    cache_baseline = make_prompt_cache(model)
    for i in range(n_layers):
        cache_baseline[i].state = exact_kv[i].state
    mx.eval([c.keys for c in cache_baseline] + [c.values for c in cache_baseline])

    question = "What is artificial intelligence?"
    gen_baseline, text_baseline = generate_text(model, tokenizer, cache_baseline, question)
    print(f"\nBaseline: {text_baseline[:200]}", file=sys.stderr)

    # ════════════════════════════════════════════════════════════
    # Part 2: Residual norm profile — identify the "mode shift"
    # ════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  RESIDUAL NORM PROFILE — identifying the mode shift")
    print(f"{'=' * 100}")

    norms = []
    for l in range(len(residuals)):
        hl_np = np.array(residuals[l][0].astype(mx.float32))
        norm = np.linalg.norm(hl_np) / np.sqrt(N * d_model)  # per-element RMS
        norms.append(norm)
        delta_from_prev = ""
        if l > 0:
            jump = norms[l] / norms[l - 1] if norms[l - 1] > 0 else 0
            delta_from_prev = f"  ×{jump:.1f}" if jump > 1.5 else ""
        tag = " ← MODE SHIFT" if l > 0 and norms[l] / norms[l-1] > 5 else ""
        print(f"  h^({l:>2}): RMS = {norm:>10.3f}{delta_from_prev}{tag}")

    # ════════════════════════════════════════════════════════════════
    # Part 3: Exact pipeline split at different cut points
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  EXACT PIPELINE SPLIT — server does layers 0..l-1, client does l..L-1")
    print(f"{'=' * 100}")

    full_kv_mb = kv_per_token * N / 1024 / 1024
    print(f"\n  Full KV: {full_kv_mb:.1f} MB")

    cut_points = [1, 2, 3, 4, 5, 6, 10, n_layers // 2]
    cut_points = [c for c in cut_points if c < n_layers]

    print(f"\n  {'Cut':>4} │ {'Server':>8} │ {'Client':>8} │ {'h^(l)':>8} │ {'KV[0:l]':>8} │ "
          f"{'Transfer':>10} │ {'Compress':>8} │ {'Client ms':>10} │ {'Match':>6}")
    print(f"  {'─' * 4}─┼{'─' * 8}─┼{'─' * 8}─┼{'─' * 8}─┼{'─' * 8}─┼"
          f"{'─' * 10}─┼{'─' * 8}─┼{'─' * 10}─┼{'─' * 6}")

    split_results = []
    for cut_l in cut_points:
        # Simulate: server ran layers 0..cut_l-1
        # Transfer h^(cut_l) + KV for layers 0..cut_l-1
        h_cut = residuals[cut_l]  # Server's output = input to layer cut_l

        # Client reconstructs layers cut_l..L-1 from h^(cut_l)
        t0 = time.perf_counter()
        client_caches = reconstruct_from_layer(inner_model, h_cut, cut_l, exact_kv)
        t_client = (time.perf_counter() - t0) * 1000

        # Build complete cache: server's KV[0:cut_l] + client's KV[cut_l:L]
        cache_split = make_prompt_cache(model)
        for i in range(cut_l):
            cache_split[i].state = exact_kv[i].state  # Server's exact KV
        for i in range(n_layers - cut_l):
            cache_split[cut_l + i].state = client_caches[i].state  # Client's reconstructed
        mx.eval([c.keys for c in cache_split] + [c.values for c in cache_split])

        # Generate and compare
        gen_split, text_split = generate_text(model, tokenizer, cache_split, question)

        token_match = sum(1 for a, b in zip(gen_baseline, gen_split) if a == b)
        exact_match = gen_baseline == gen_split

        # Transfer size
        h_size = h_per_token * N  # h^(l)
        kv_prefix_size = kv_per_layer_token * cut_l * N  # KV for layers 0..cut_l-1
        total_transfer = h_size + kv_prefix_size
        compression = kv_per_token * N / total_transfer

        n_server = cut_l
        n_client = n_layers - cut_l

        print(f"  {cut_l:>4} │ {n_server:>6}L  │ {n_client:>6}L  │ "
              f"{h_size / 1024 / 1024:>6.1f} MB│ {kv_prefix_size / 1024 / 1024:>6.1f} MB│ "
              f"{total_transfer / 1024 / 1024:>8.1f} MB│ {compression:>7.1f}× │ "
              f"{t_client:>8.0f}ms │ "
              f"{'EXACT' if exact_match else f'{token_match}/50'}")

        split_results.append({
            'cut': cut_l, 'transfer_mb': total_transfer / 1024 / 1024,
            'compression': compression, 'client_ms': t_client,
            'exact': exact_match, 'text': text_split,
        })

    # ════════════════════════════════════════════════════════════════
    # Part 4: Delta compressibility from different base layers
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  DELTA COMPRESSIBILITY: δ^(j) = h^(j) - h^(base) for various base layers")
    print(f"{'=' * 100}")

    base_layers = [0, 1, 2, 3, 5]
    base_layers = [b for b in base_layers if b < n_layers - 1]

    for base_l in base_layers:
        h_base_np = np.array(residuals[base_l][0].astype(mx.float32))

        print(f"\n  ─── Base = h^({base_l}) ───")
        print(f"  {'Target':>6} │ {'||δ||':>10} │ {'δ/h ratio':>9} │ "
              f"{'rank@95%':>8} │ {'rank@99%':>8} │ {'rank@99.9%':>10}")
        print(f"  {'─' * 6}─┼{'─' * 10}─┼{'─' * 9}─┼{'─' * 8}─┼{'─' * 8}─┼{'─' * 10}")

        ranks_99 = []
        ranks_999 = []
        for target_l in range(base_l + 1, min(n_layers + 1, len(residuals))):
            ht_np = np.array(residuals[target_l][0].astype(mx.float32))
            delta_np = ht_np - h_base_np

            h_norm = np.linalg.norm(ht_np)
            d_norm = np.linalg.norm(delta_np)
            ratio = d_norm / h_norm if h_norm > 0 else 0

            S, cumulative, _, _ = svd_analysis(delta_np)

            rank_95 = int(np.searchsorted(cumulative, 0.95)) + 1
            rank_99 = int(np.searchsorted(cumulative, 0.99)) + 1
            rank_999 = int(np.searchsorted(cumulative, 0.999)) + 1
            ranks_99.append(rank_99)
            ranks_999.append(rank_999)

            # Only print subset of layers for readability
            if (target_l <= base_l + 5 or target_l == n_layers // 2
                    or target_l >= n_layers - 2 or target_l % 5 == 0):
                print(f"  h^({target_l:>2}) │ {d_norm:>10.1f} │ {ratio:>8.4f}  │ "
                      f"{rank_95:>8} │ {rank_99:>8} │ {rank_999:>10}")

        avg_r99 = np.mean(ranks_99)
        avg_r999 = np.mean(ranks_999)
        print(f"  avg rank@99%={avg_r99:.0f}, rank@99.9%={avg_r999:.0f} "
              f"({len(ranks_99)} layers from base {base_l})")

    # ════════════════════════════════════════════════════════════════
    # Part 5: Transfer savings comparison table
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  TRANSFER SAVINGS COMPARISON")
    print(f"{'=' * 100}")

    print(f"\n  Full KV cache: {full_kv_mb:.1f} MB")
    print(f"  h^(0) only:    {h_per_token * N / 1024 / 1024:.1f} MB (needs full {n_layers}L recon)")
    print()

    print(f"  {'Scheme':>35} │ {'Transfer':>10} │ {'Compress':>8} │ {'Client work':>12} │ {'Quality':>8}")
    print(f"  {'─' * 35}─┼{'─' * 10}─┼{'─' * 8}─┼{'─' * 12}─┼{'─' * 8}")

    # Full KV
    print(f"  {'Full KV transfer':>35} │ {full_kv_mb:>8.1f} MB│ {1.0:>7.1f}× │ {'decode only':>12} │ {'EXACT':>8}")

    # int4 KV
    print(f"  {'int4 KV transfer':>35} │ {full_kv_mb / 4:>8.1f} MB│ {4.0:>7.1f}× │ {'decode only':>12} │ {'~exact':>8}")

    # h^(0) full reconstruction
    h0_mb = h_per_token * N / 1024 / 1024
    print(f"  {'h^(0) + full recon':>35} │ {h0_mb:>8.1f} MB│ {full_kv_mb / h0_mb:>7.1f}× │ {f'{n_layers}L forward':>12} │ {'EXACT':>8}")

    # Pipeline splits
    for r in split_results:
        cut = r['cut']
        label = f"h^({cut}) + KV[0:{cut}] + {n_layers - cut}L recon"
        client_work = f"{n_layers - cut}L forward"
        quality = "EXACT" if r['exact'] else "FAIL"
        print(f"  {label:>35} │ {r['transfer_mb']:>8.1f} MB│ {r['compression']:>7.1f}× │ "
              f"{client_work:>12} │ {quality:>8}")

    # ════════════════════════════════════════════════════════════════
    # Part 6: Approximate reconstruction from h^(3) — the real test
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  APPROXIMATE RECONSTRUCTION: exact layers 0-2 KV + approximate layers 3+ from h^(3)")
    print(f"{'=' * 100}")

    # Best base layer candidates based on mode shift
    # Try: server sends h^(base) + KV[0:base], client reconstructs layers base..L
    # using h^(base) directly as input to each layer (no delta)
    # This tests: "how much information does h^(base) carry for later layers?"

    for base_l in [0, 3]:
        h_base = residuals[base_l]

        print(f"\n  ─── Base = h^({base_l}): using h^({base_l}) as input to ALL layers {base_l}..{n_layers-1} ───")
        print(f"  (No delta correction — testing raw h^({base_l}) as universal proxy)")

        # Build KV cache:
        # layers 0..base_l-1: exact KV from server
        # layers base_l..L-1: project KV from h^(base) through each layer's KV projections
        #   BUT: must go through full layer forward for correct RoPE!

        # Method: run layers base_l..L-1 with h^(base) as input to EVERY layer
        # This is wrong architecturally (each layer normally gets h^(l) not h^(base))
        # but tests whether h^(base) has enough info
        N_tok = N
        mask_for_recon = nn.MultiHeadAttention.create_additive_causal_mask(N_tok).astype(h_base.dtype)

        approx_caches = []
        for layer_idx in range(base_l, n_layers):
            tc = KVCache()
            layer_obj = inner_model.layers[layer_idx]
            # Feed h^(base) through this layer — only to capture KV, ignore output
            _ = layer_obj(h_base, mask=mask_for_recon, cache=tc)
            mx.eval(tc.keys, tc.values)
            approx_caches.append(tc)

        # Build complete cache
        cache_approx = make_prompt_cache(model)
        for i in range(base_l):
            cache_approx[i].state = exact_kv[i].state
        for i in range(n_layers - base_l):
            cache_approx[base_l + i].state = approx_caches[i].state
        mx.eval([c.keys for c in cache_approx] + [c.values for c in cache_approx])

        gen_approx, text_approx = generate_text(model, tokenizer, cache_approx, question)
        token_match = sum(1 for a, b in zip(gen_baseline, gen_approx) if a == b)
        exact = gen_baseline == gen_approx

        h_size = h_per_token * N
        kv_prefix = kv_per_layer_token * base_l * N
        total_xfer = h_size + kv_prefix
        compress = kv_per_token * N / total_xfer

        print(f"    Transfer: {total_xfer / 1024 / 1024:.1f} MB ({compress:.1f}×)")
        print(f"    Output: {text_approx[:200]}")
        print(f"    Match: {token_match}/{min(len(gen_baseline), len(gen_approx))} "
              f"({'EXACT' if exact else 'DIVERGED'})")

    # ════════════════════════════════════════════════════════════════
    # Part 7: The "correct" h^(l) experiment — sequential reconstruction
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  SEQUENTIAL RECONSTRUCTION: server layers 0..l-1, client layers l..L-1")
    print(f"  (Client runs forward sequentially from h^(l) — must be EXACT)")
    print(f"{'=' * 100}")

    for cut_l in [3, 6, n_layers // 2]:
        if cut_l >= n_layers:
            continue

        h_cut = residuals[cut_l]

        # Client runs layers cut_l..L-1 sequentially
        t0 = time.perf_counter()
        seq_caches = [KVCache() for _ in range(n_layers - cut_l)]
        x = h_cut
        mask_seq = nn.MultiHeadAttention.create_additive_causal_mask(N).astype(x.dtype)
        for i in range(n_layers - cut_l):
            layer_obj = inner_model.layers[cut_l + i]
            x = layer_obj(x, mask=mask_seq, cache=seq_caches[i])
            mx.eval(x)
        t_recon = (time.perf_counter() - t0) * 1000

        # Also verify we get the correct final hidden state
        final_h_exact = residuals[n_layers]
        final_h_recon = x
        final_err = float(mx.mean(mx.abs(final_h_exact - final_h_recon)).item())

        # Build complete cache and generate
        cache_seq = make_prompt_cache(model)
        for i in range(cut_l):
            cache_seq[i].state = exact_kv[i].state
        for i in range(n_layers - cut_l):
            cache_seq[cut_l + i].state = seq_caches[i].state
        mx.eval([c.keys for c in cache_seq] + [c.values for c in cache_seq])

        gen_seq, text_seq = generate_text(model, tokenizer, cache_seq, question)
        token_match = sum(1 for a, b in zip(gen_baseline, gen_seq) if a == b)
        exact = gen_baseline == gen_seq

        h_size = h_per_token * N
        kv_prefix = kv_per_layer_token * cut_l * N
        total_xfer = h_size + kv_prefix
        compress = kv_per_token * N / total_xfer

        server_pct = cut_l / n_layers * 100
        client_pct = (n_layers - cut_l) / n_layers * 100

        print(f"\n  Cut at layer {cut_l}: server {cut_l}L ({server_pct:.0f}%) | "
              f"client {n_layers - cut_l}L ({client_pct:.0f}%)")
        print(f"    Transfer:    {total_xfer / 1024 / 1024:.1f} MB ({compress:.1f}×)")
        print(f"    Client time: {t_recon:.0f}ms (vs full {t_full:.0f}ms = "
              f"{t_recon / t_full * 100:.0f}%)")
        print(f"    Final h err: {final_err:.2e}")
        print(f"    Output: {text_seq[:200]}")
        print(f"    Match: {token_match}/{min(len(gen_baseline), len(gen_seq))} "
              f"{'EXACT ✓' if exact else 'DIVERGED ✗'}")


if __name__ == "__main__":
    main()
