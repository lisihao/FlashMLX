#!/usr/bin/env python3
"""
Ultra-fast hit rate test — 3 configs, minimal warmup, key drifts only.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx


def sync():
    mx.eval(mx.array(0))


def test_config(cfg, drift, S=8192, n_test=10, seed=42):
    from flashmlx.mac import MACDecodeWrapper
    from flashmlx.mac.attention import mac_partial_attention

    N, Hq, Hkv, D = 1, 32, 8, 128

    mac = MACDecodeWrapper(**cfg)
    mx.random.seed(seed)
    k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    req_ids = mx.array([0], dtype=mx.int32)
    mx.eval(k, v)

    M = cfg.get("capacity", 512)
    wi = cfg.get("write_interval", 1)
    # Fast warmup: fixed 100 iterations (like minimal test)
    n_warm = 100

    base_q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(base_q)

    # Warmup with batched eval
    for step in range(n_warm):
        noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * drift
        q = base_q + noise
        mac(q, k, v, req_ids)
        base_q = q
        if step % 50 == 49:
            mx.eval(q)
    mx.eval(base_q)
    sync()

    mac.ring_cache.request_length = mx.array([S], dtype=mx.int32)
    mx.eval(mac.ring_cache.request_length)

    # Anchor
    noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * drift
    q_anchor = base_q + noise
    mx.eval(q_anchor)
    mac(q_anchor, k, v, req_ids)
    sync()
    base_q = q_anchor

    hits = 0
    total = 0
    cos_sims = []
    start_zero = mx.zeros((N, Hq), dtype=mx.int32)

    for i in range(n_test):
        noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * drift
        q_t = base_q + noise
        mx.eval(q_t)

        full_o, _ = mac_partial_attention(q_t, k, v, start_zero)
        mac_o = mac(q_t, k, v, req_ids)
        mx.eval(full_o, mac_o)

        stats = mac.last_stats
        if stats:
            hits += stats.num_hits
            total += stats.num_total

        f = full_o.astype(mx.float32)
        m = mac_o.astype(mx.float32)
        fn = mx.sqrt(mx.sum(f * f)).item()
        mn = mx.sqrt(mx.sum(m * m)).item()
        cos = mx.sum(f * m).item() / max(fn * mn, 1e-10)
        cos_sims.append(cos)
        base_q = q_t

    hr = hits / max(total, 1)
    avg_cos = sum(cos_sims) / len(cos_sims)
    return hr, avg_cos


def main():
    print(f"Device: {mx.default_device()}")
    print(f"MLX version: {mx.__version__}")
    print()

    N, Hq, Hkv, D = 1, 32, 8, 128
    base = dict(
        max_requests=4, num_heads=Hq, num_kv_heads=Hkv, head_dim=D,
        band_r=256, window_left=256,
    )

    # Only 3 key configs
    configs = [
        ("baseline",         dict(**base, capacity=512,  threshold=0.5)),
        ("normalize",        dict(**base, capacity=512,  threshold=0.5, normalize_queries=True)),
        ("norm+t0.2+M1024",  dict(**base, capacity=1024, threshold=0.2, normalize_queries=True)),
    ]

    # Key drift points: 0.5 (baseline), 0.7 (critical), 1.0 (extreme)
    drifts = [0.5, 0.7, 1.0]

    print("=" * 78)
    print("Quick Hit Rate Test (3 configs × 3 drifts)")
    print("=" * 78)
    print()
    print(f"  {'config':>20} {'d=0.5':>12} {'d=0.7':>12} {'d=1.0':>12}")
    print("  " + "-" * 62)

    for i, (name, cfg) in enumerate(configs, 1):
        print(f"  [{i}/{len(configs)}] Testing {name}...", end=" ", flush=True)
        results = []
        for drift in drifts:
            hr, cos = test_config(cfg, drift)
            results.append((hr, cos))

        line = f"\r  {name:>20}"
        for hr, cos in results:
            line += f"  {hr:>4.0%} ({cos:>5.3f})"
        print(line, flush=True)

    print()
    print("=" * 78)
    print("Legend: hit% (cosine_similarity)")
    print("=" * 78)
    print()
    print("  - d=0.5: All configs should hit 100%")
    print("  - d=0.7: Critical transition point")
    print("  - d=1.0: Extreme drift — normalize should win")
    print()


if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")
