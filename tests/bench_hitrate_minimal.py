#!/usr/bin/env python3
"""
Minimal hit rate test — 1 config, 1 drift, verify code correctness.
Estimated runtime: 5-10 seconds.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx


def sync():
    mx.eval(mx.array(0))


def test_config(cfg, drift, S=8192, n_test=5, seed=42):
    """Test one config at one drift level."""
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
    # MINIMAL warmup: just 100 iterations to test code path
    n_warm = 100

    base_q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(base_q)

    print(f"  Warmup: {n_warm} iterations...", flush=True)
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

    print(f"  Testing: {n_test} queries...", flush=True)
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

    # Only 1 config for minimal test
    cfg = dict(**base, capacity=512, threshold=0.5, normalize_queries=True)
    drift = 0.5

    print("=" * 60)
    print("Minimal Hit Rate Test (1 config × 1 drift)")
    print("=" * 60)
    print()

    hr, cos = test_config(cfg, drift)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Config: normalize, M=512, threshold=0.5")
    print(f"  Drift:  {drift}")
    print(f"  Hit Rate: {hr:.0%}")
    print(f"  Cosine Similarity: {cos:.5f}")
    print()
    print("✅ Test completed successfully!")


if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s")
