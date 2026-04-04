#!/usr/bin/env python3
"""
Hit rate optimization experiments.

Tests 5 optimization levers:
  1. Query normalization (normalize_queries)
  2. Ring cache capacity (M)
  3. Write interval (write_interval)
  4. Threshold (cosine threshold when normalized, raw otherwise)
  5. Combinations

Metrics per configuration:
  - Hit rate: fraction of (N, H) entries that match
  - Quality: cosine similarity vs full attention output
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx


def sync():
    mx.eval(mx.array(0))


def run_experiment(
    config: dict,
    drift: float = 0.1,
    S: int = 8192,
    n_test: int = 20,
    seed: int = 42,
) -> dict:
    """Run one experiment and return metrics.

    Args:
        config: kwargs for MACDecodeWrapper
        drift: noise scale for query drift (smaller = more similar)
        S: sequence length
        n_test: number of test queries
        seed: random seed

    Returns:
        dict with hit_rate, avg_cos, avg_skip_pct
    """
    from flashmlx.mac import MACDecodeWrapper
    from flashmlx.mac.attention import mac_partial_attention

    N, Hq, Hkv, D = 1, 32, 8, 128

    mac = MACDecodeWrapper(**config)

    mx.random.seed(seed)
    k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    req_ids = mx.array([0], dtype=mx.int32)
    mx.eval(k, v)

    M = config.get("capacity", 1024)
    wi = config.get("write_interval", 1)
    # Need enough warmup steps to fill M slots (accounting for write_interval)
    n_warmup = M * wi + 10

    # Fill cache with drifting queries
    base_q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(base_q)
    for _ in range(n_warmup):
        noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * drift
        q = base_q + noise
        mx.eval(q)
        mac(q, k, v, req_ids)
        base_q = q
        sync()

    # Set realistic request_length for meaningful left_start (skip prefix)
    mac.ring_cache.request_length = mx.array([S], dtype=mx.int32)
    mx.eval(mac.ring_cache.request_length)

    # Write one more query at the new request_length to anchor the cache
    noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * drift
    q_anchor = base_q + noise
    mx.eval(q_anchor)
    mac(q_anchor, k, v, req_ids)
    sync()
    base_q = q_anchor

    # Test phase: measure hit rate and quality
    hits = 0
    total = 0
    cos_sims = []
    skip_pcts = []
    start_zero = mx.zeros((N, Hq), dtype=mx.int32)

    for _ in range(n_test):
        noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * drift
        q_t = base_q + noise
        mx.eval(q_t)

        # Full attention (reference)
        full_o, _ = mac_partial_attention(q_t, k, v, start_zero)
        # MAC attention
        mac_o = mac(q_t, k, v, req_ids)
        mx.eval(full_o, mac_o)

        stats = mac.last_stats
        if stats:
            hits += stats.num_hits
            total += stats.num_total
            skip_pcts.append(stats.avg_left_start / S if S > 0 else 0)

        # Quality: cosine similarity between full and MAC output
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
    avg_skip = sum(skip_pcts) / len(skip_pcts) if skip_pcts else 0

    return {"hit_rate": hr, "avg_cos": avg_cos, "avg_skip": avg_skip}


def main():
    print(f"Device: {mx.default_device()}")
    print()

    N, Hq, Hkv, D = 1, 32, 8, 128
    S = 8192

    # Base config (shared across all experiments)
    base = dict(
        max_requests=4,
        num_heads=Hq,
        num_kv_heads=Hkv,
        head_dim=D,
        band_r=256,
        window_left=256,
    )

    # ================================================================
    # Experiment 1: Normalization (with different thresholds)
    # ================================================================
    print("=" * 78)
    print("Exp 1: Query Normalization Effect")
    print("=" * 78)
    print()
    print(f"  {'config':>22} {'drift':>6} {'hit%':>6} {'skip%':>6} {'cos':>9}")
    print("  " + "-" * 55)

    for drift in [0.05, 0.1, 0.2, 0.3]:
        # Baseline (no normalize)
        cfg = {**base, "capacity": 512, "threshold": 0.5}
        r = run_experiment(cfg, drift=drift, S=S)
        print(
            f"  {'baseline τ=0.5':>22} {drift:>6.2f} {r['hit_rate']:>5.0%}"
            f" {r['avg_skip']:>5.0%} {r['avg_cos']:>9.5f}"
        )

        # Normalized (same cosine threshold)
        cfg_n = {**base, "capacity": 512, "threshold": 0.5, "normalize_queries": True}
        r = run_experiment(cfg_n, drift=drift, S=S)
        print(
            f"  {'normalize τ=0.5':>22} {drift:>6.2f} {r['hit_rate']:>5.0%}"
            f" {r['avg_skip']:>5.0%} {r['avg_cos']:>9.5f}"
        )

        # Normalized with looser threshold
        cfg_n3 = {**base, "capacity": 512, "threshold": 0.3, "normalize_queries": True}
        r = run_experiment(cfg_n3, drift=drift, S=S)
        print(
            f"  {'normalize τ=0.3':>22} {drift:>6.2f} {r['hit_rate']:>5.0%}"
            f" {r['avg_skip']:>5.0%} {r['avg_cos']:>9.5f}"
        )

        print()

    # ================================================================
    # Experiment 2: Capacity sweep
    # ================================================================
    print("=" * 78)
    print("Exp 2: Ring Cache Capacity (M)")
    print("=" * 78)
    print()
    print(f"  {'config':>22} {'drift':>6} {'hit%':>6} {'skip%':>6} {'cos':>9}")
    print("  " + "-" * 55)

    for drift in [0.05, 0.1, 0.2]:
        for M in [256, 512, 1024, 2048]:
            cfg = {**base, "capacity": M, "threshold": 0.5}
            r = run_experiment(cfg, drift=drift, S=S)
            print(
                f"  {'M=' + str(M):>22} {drift:>6.2f} {r['hit_rate']:>5.0%}"
                f" {r['avg_skip']:>5.0%} {r['avg_cos']:>9.5f}"
            )
        print()

    # ================================================================
    # Experiment 3: Write interval
    # ================================================================
    print("=" * 78)
    print("Exp 3: Write Interval (skip writes for diversity)")
    print("=" * 78)
    print()
    print(f"  {'config':>22} {'drift':>6} {'hit%':>6} {'skip%':>6} {'cos':>9}")
    print("  " + "-" * 55)

    for drift in [0.05, 0.1, 0.2]:
        for wi in [1, 2, 3, 4]:
            cfg = {**base, "capacity": 512, "threshold": 0.5, "write_interval": wi}
            r = run_experiment(cfg, drift=drift, S=S)
            print(
                f"  {'WI=' + str(wi):>22} {drift:>6.2f} {r['hit_rate']:>5.0%}"
                f" {r['avg_skip']:>5.0%} {r['avg_cos']:>9.5f}"
            )
        print()

    # ================================================================
    # Experiment 4: Normalize + Capacity combos
    # ================================================================
    print("=" * 78)
    print("Exp 4: Normalize + Capacity Combos")
    print("=" * 78)
    print()
    print(f"  {'config':>30} {'drift':>6} {'hit%':>6} {'skip%':>6} {'cos':>9}")
    print("  " + "-" * 63)

    drift = 0.1  # moderate drift
    combos = [
        ("baseline", dict(capacity=512, threshold=0.5)),
        ("norm", dict(capacity=512, threshold=0.5, normalize_queries=True)),
        ("norm+M1024", dict(capacity=1024, threshold=0.5, normalize_queries=True)),
        ("norm+M2048", dict(capacity=2048, threshold=0.5, normalize_queries=True)),
        ("norm+WI2", dict(capacity=512, threshold=0.5, normalize_queries=True, write_interval=2)),
        ("norm+WI3", dict(capacity=512, threshold=0.5, normalize_queries=True, write_interval=3)),
        ("norm+M1024+WI2", dict(capacity=1024, threshold=0.5, normalize_queries=True, write_interval=2)),
        ("norm+τ0.3", dict(capacity=512, threshold=0.3, normalize_queries=True)),
        ("norm+τ0.3+M1024", dict(capacity=1024, threshold=0.3, normalize_queries=True)),
        ("norm+τ0.3+WI2", dict(capacity=512, threshold=0.3, normalize_queries=True, write_interval=2)),
        ("norm+τ0.3+M1024+WI2", dict(capacity=1024, threshold=0.3, normalize_queries=True, write_interval=2)),
    ]

    for name, overrides in combos:
        for d in [0.05, 0.1, 0.2, 0.3]:
            cfg = {**base, **overrides}
            r = run_experiment(cfg, drift=d, S=S)
            print(
                f"  {name:>30} {d:>6.2f} {r['hit_rate']:>5.0%}"
                f" {r['avg_skip']:>5.0%} {r['avg_cos']:>9.5f}"
            )
        print()

    # ================================================================
    # Experiment 5: Best configs at different sequence lengths
    # ================================================================
    print("=" * 78)
    print("Exp 5: Best Configs at Different Seq Lengths")
    print("=" * 78)
    print()
    print(f"  {'config':>30} {'S':>6} {'drift':>6} {'hit%':>6} {'skip%':>6} {'cos':>9}")
    print("  " + "-" * 71)

    drift = 0.1
    best_configs = [
        ("baseline", dict(capacity=512, threshold=0.5)),
        ("norm+τ0.3+M1024", dict(capacity=1024, threshold=0.3, normalize_queries=True)),
    ]

    for seq_len in [4096, 8192, 16384, 32768]:
        for name, overrides in best_configs:
            cfg = {**base, **overrides}
            r = run_experiment(cfg, drift=drift, S=seq_len)
            print(
                f"  {name:>30} {seq_len:>6} {drift:>6.2f} {r['hit_rate']:>5.0%}"
                f" {r['avg_skip']:>5.0%} {r['avg_cos']:>9.5f}"
            )
        print()


if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
