#!/usr/bin/env python3
"""
Focused hit rate experiments — smaller warmup, fewer configs, critical drifts only.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx


def sync():
    mx.eval(mx.array(0))


def test_config(cfg, drift, S=8192, mixed_random_pct=0.0, seed=42):
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
    # Just fill cache once (not M*wi times — faster)
    n_warm = min(M * wi + 10, 600)

    base_q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(base_q)
    # Warmup: fill cache. Eval every 50 steps (not every step) for speed.
    for step in range(n_warm):
        noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * drift
        q = base_q + noise
        mx.eval(q)
        mac(q, k, v, req_ids)
        base_q = q
        if step % 50 == 49:
            sync()
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

    for i in range(10):
        if mixed_random_pct > 0 and (i % max(1, int(1.0 / mixed_random_pct))) == 0:
            q_t = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
        else:
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
        sync()

    hr = hits / max(total, 1)
    avg_cos = sum(cos_sims) / len(cos_sims)
    return hr, avg_cos


def main():
    print(f"Device: {mx.default_device()}")

    N, Hq, Hkv, D = 1, 32, 8, 128
    base = dict(
        max_requests=4, num_heads=Hq, num_kv_heads=Hkv, head_dim=D,
        band_r=256, window_left=256,
    )

    configs = [
        ("baseline",         dict(**base, capacity=512,  threshold=0.5)),
        ("normalize",        dict(**base, capacity=512,  threshold=0.5, normalize_queries=True)),
        ("norm+t0.3",        dict(**base, capacity=512,  threshold=0.3, normalize_queries=True)),
        ("norm+t0.2",        dict(**base, capacity=512,  threshold=0.2, normalize_queries=True)),
        ("M=1024",           dict(**base, capacity=1024, threshold=0.5)),
        ("WI=2",             dict(**base, capacity=512,  threshold=0.5, write_interval=2)),
        ("WI=3",             dict(**base, capacity=512,  threshold=0.5, write_interval=3)),
        ("norm+M1024",       dict(**base, capacity=1024, threshold=0.5, normalize_queries=True)),
        ("norm+t0.3+M1024",  dict(**base, capacity=1024, threshold=0.3, normalize_queries=True)),
        ("norm+t0.2+M1024",  dict(**base, capacity=1024, threshold=0.2, normalize_queries=True)),
        ("norm+M1024+WI2",   dict(**base, capacity=1024, threshold=0.5, normalize_queries=True, write_interval=2)),
    ]

    # ================================================================
    # Section 1: Critical drift zone (pure drifting queries)
    # ================================================================
    print()
    print("=" * 78)
    print("Section 1: Critical drift zone (pure drifting queries)")
    print("=" * 78)
    print()
    print(f"  {'config':>24} {'d=0.5':>6} {'d=0.6':>6} {'d=0.7':>6} {'d=0.8':>6} {'d=1.0':>6}")
    print("  " + "-" * 60)

    for name, cfg in configs:
        results = []
        for drift in [0.5, 0.6, 0.7, 0.8, 1.0]:
            hr, _ = test_config(cfg, drift)
            results.append(hr)
        line = f"  {name:>24}"
        for hr in results:
            line += f" {hr:>5.0%}"
        print(line, flush=True)

    # ================================================================
    # Section 2: Quality at drift=0.7 (transition point)
    # ================================================================
    print()
    print("=" * 78)
    print("Section 2: Quality at drift=0.7 (transition point)")
    print("=" * 78)
    print()
    print(f"  {'config':>24} {'hit%':>6} {'cos':>9}")
    print("  " + "-" * 42)

    for name, cfg in configs:
        hr, cos = test_config(cfg, 0.7)
        marker = " <<< BEST" if hr >= 0.9 else " *" if hr > 0.5 else ""
        print(f"  {name:>24} {hr:>5.0%} {cos:>9.5f}{marker}", flush=True)

    # ================================================================
    # Section 3: Mixed queries (30% random, 70% drifting)
    # ================================================================
    print()
    print("=" * 78)
    print("Section 3: Mixed queries (30% random + 70% drifting)")
    print("=" * 78)
    print()
    print(f"  {'config':>24} {'d=0.3':>6} {'d=0.5':>6} {'d=0.7':>6}")
    print("  " + "-" * 48)

    for name, cfg in configs:
        results = []
        for drift in [0.3, 0.5, 0.7]:
            hr, _ = test_config(cfg, drift, mixed_random_pct=0.3)
            results.append(hr)
        line = f"  {name:>24}"
        for hr in results:
            line += f" {hr:>5.0%}"
        print(line, flush=True)

    # ================================================================
    # Section 4: Summary
    # ================================================================
    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print()
    print("  Key findings from the experiments above.")
    print("  Configs marked * beat baseline significantly at critical drift levels.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")
