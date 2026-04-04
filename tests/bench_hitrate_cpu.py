#!/usr/bin/env python3
"""
Hit rate optimization experiments — CPU-compatible version.

Uses pure MLX reference implementations (no Metal kernel) so it can run on
CPU when GPU is unavailable. The hit rate results are identical to Metal.

Tests at CRITICAL drift zone (0.5–1.0) where optimizations actually matter.

Configurations tested:
  1. baseline:          M=512, τ=0.5
  2. norm:              M=512, τ=0.5, normalize
  3. norm+τ0.3:         M=512, τ=0.3, normalize
  4. norm+τ0.7:         M=512, τ=0.7, normalize
  5. M1024:             M=1024, τ=0.5
  6. WI2:               M=512, τ=0.5, write_interval=2
  7. WI3:               M=512, τ=0.5, write_interval=3
  8. norm+M1024:        M=1024, τ=0.5, normalize
  9. norm+τ0.3+M1024:   M=1024, τ=0.3, normalize
  10. norm+WI2:          M=512, τ=0.5, normalize, WI=2
  11. norm+τ0.3+M1024+WI2: best combo candidate
"""

from __future__ import annotations

import math
import sys
import time

sys.path.insert(0, "src")

import mlx.core as mx

# Force CPU to bypass GPU deadlock
mx.set_default_device(mx.cpu)


class SimpleMACSimulator:
    """Lightweight MAC simulator for hit rate testing (CPU compatible).

    Reimplements ring cache + match + rectify in pure MLX.
    No Metal kernels needed.
    """

    def __init__(
        self,
        capacity: int = 512,
        num_heads: int = 32,
        head_dim: int = 128,
        threshold: float = 0.5,
        band_r: int = 256,
        normalize: bool = False,
        write_interval: int = 1,
    ):
        self.M = capacity
        self.H = num_heads
        self.D = head_dim
        self.threshold = threshold
        self.band_r = band_r
        self.normalize = normalize
        self.write_interval = max(1, write_interval)
        self._write_count = 0

        # Ring buffer: [M, H, D]
        self.query_cache = mx.zeros((capacity, num_heads, head_dim), dtype=mx.float32)
        self.length = 0  # cumulative writes

        # Precompute threshold squared for match
        if normalize:
            # For normalized queries: L2 dist = 2(1 - cos_sim)
            # We want: 2(1 - cos_sim) < T_sq
            # With kernel formula: T_sq = 2*D*(1-τ')²
            # After threshold transform: τ' = 1 - sqrt((1-τ)/D)
            # So T_sq = 2*D * (1-τ)/D = 2*(1-τ)
            self.T_sq = 2.0 * (1.0 - threshold)
        else:
            one_minus = 1.0 - threshold
            self.T_sq = 2.0 * head_dim * one_minus * one_minus

    def _normalize(self, q: mx.array) -> mx.array:
        """L2 normalize along last axis."""
        norms = mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True))
        norms = mx.maximum(norms, 1e-8)
        return q / norms

    def write(self, query: mx.array):
        """Write a query to ring cache. query: [H, D]"""
        self._write_count += 1
        if self.write_interval > 1 and self._write_count % self.write_interval != 0:
            return

        q = query.astype(mx.float32)
        if self.normalize:
            q = self._normalize(q)

        slot = self.length % self.M
        self.query_cache = self.query_cache.at[slot].multiply(0)
        self.query_cache = self.query_cache.at[slot].add(q)
        self.length += 1

    def match(self, query: mx.array) -> tuple[bool, int]:
        """Match query against cache. Returns (is_hit, best_slot).
        query: [H, D]
        Returns aggregated: (any_head_hit_rate, avg_across_heads)
        """
        V = min(self.length, self.M)
        if V <= 0:
            return 0.0, 0

        q = query.astype(mx.float32)
        if self.normalize:
            q = self._normalize(q)

        # q: [H, D], cache: [V, H, D]
        cache = self.query_cache[:V]  # [V, H, D]
        diffs = q[None, :, :] - cache  # [V, H, D]
        dists = mx.sum(diffs * diffs, axis=-1)  # [V, H]

        # Per-head argmin
        best_dists = mx.min(dists, axis=0)  # [H]
        mx.eval(best_dists)

        # Per-head hit check
        hits_per_head = best_dists < self.T_sq  # [H]
        mx.eval(hits_per_head)
        hit_rate = hits_per_head.astype(mx.float32).mean().item()

        return hit_rate, 0


def run_experiment(
    config: dict,
    drift: float,
    n_warmup: int = 200,
    n_test: int = 50,
    seed: int = 42,
) -> dict:
    """Run a single config at a given drift level.

    Simulates decode: each step generates query = base + noise*drift,
    writes to cache, then next step tries to match.
    """
    H, D = config.get("num_heads", 32), config.get("head_dim", 128)

    sim = SimpleMACSimulator(
        capacity=config.get("capacity", 512),
        num_heads=H,
        head_dim=D,
        threshold=config.get("threshold", 0.5),
        normalize=config.get("normalize", False),
        write_interval=config.get("write_interval", 1),
    )

    mx.random.seed(seed)

    # Generate base query
    base_q = mx.random.normal((H, D))
    mx.eval(base_q)

    # Warmup: fill cache with drifting queries
    wi = config.get("write_interval", 1)
    actual_warmup = max(n_warmup, sim.M * wi + 10)
    for _ in range(actual_warmup):
        noise = mx.random.normal((H, D)) * drift
        q = base_q + noise
        mx.eval(q)
        sim.write(q)
        base_q = q

    # Test: measure hit rate
    hit_rates = []
    for _ in range(n_test):
        noise = mx.random.normal((H, D)) * drift
        q = base_q + noise
        mx.eval(q)

        hr, _ = sim.match(q)
        hit_rates.append(hr)

        # Also write this query (simulating ongoing decode)
        sim.write(q)
        base_q = q

    avg_hr = sum(hit_rates) / len(hit_rates)
    return {"hit_rate": avg_hr}


def run_mixed_experiment(
    config: dict,
    drift: float,
    topic_switch_pct: float = 0.3,
    n_warmup: int = 200,
    n_test: int = 50,
    seed: int = 42,
) -> dict:
    """Mixed scenario: some queries drift smoothly, others jump to new topic.

    topic_switch_pct: fraction of test queries that jump to a random new topic
    """
    H, D = config.get("num_heads", 32), config.get("head_dim", 128)

    sim = SimpleMACSimulator(
        capacity=config.get("capacity", 512),
        num_heads=H,
        head_dim=D,
        threshold=config.get("threshold", 0.5),
        normalize=config.get("normalize", False),
        write_interval=config.get("write_interval", 1),
    )

    mx.random.seed(seed)
    base_q = mx.random.normal((H, D))
    mx.eval(base_q)

    wi = config.get("write_interval", 1)
    actual_warmup = max(n_warmup, sim.M * wi + 10)
    for _ in range(actual_warmup):
        noise = mx.random.normal((H, D)) * drift
        q = base_q + noise
        mx.eval(q)
        sim.write(q)
        base_q = q

    hit_rates = []
    for i in range(n_test):
        if mx.random.uniform(()).item() < topic_switch_pct:
            # Topic switch: random new query (unlikely to match)
            q = mx.random.normal((H, D)) * 3.0
        else:
            noise = mx.random.normal((H, D)) * drift
            q = base_q + noise
        mx.eval(q)

        hr, _ = sim.match(q)
        hit_rates.append(hr)
        sim.write(q)
        base_q = q

    avg_hr = sum(hit_rates) / len(hit_rates)
    return {"hit_rate": avg_hr}


# ============================================================================
# Configurations
# ============================================================================

BASE = dict(num_heads=32, head_dim=128)

CONFIGS = {
    "baseline":             {**BASE, "capacity": 512, "threshold": 0.5},
    "norm":                 {**BASE, "capacity": 512, "threshold": 0.5, "normalize": True},
    "norm+τ0.3":            {**BASE, "capacity": 512, "threshold": 0.3, "normalize": True},
    "norm+τ0.7":            {**BASE, "capacity": 512, "threshold": 0.7, "normalize": True},
    "M1024":                {**BASE, "capacity": 1024, "threshold": 0.5},
    "M2048":                {**BASE, "capacity": 2048, "threshold": 0.5},
    "WI2":                  {**BASE, "capacity": 512, "threshold": 0.5, "write_interval": 2},
    "WI3":                  {**BASE, "capacity": 512, "threshold": 0.5, "write_interval": 3},
    "norm+M1024":           {**BASE, "capacity": 1024, "threshold": 0.5, "normalize": True},
    "norm+τ0.3+M1024":      {**BASE, "capacity": 1024, "threshold": 0.3, "normalize": True},
    "norm+WI2":             {**BASE, "capacity": 512, "threshold": 0.5, "normalize": True, "write_interval": 2},
    "norm+τ0.3+WI2":        {**BASE, "capacity": 512, "threshold": 0.3, "normalize": True, "write_interval": 2},
    "norm+τ0.3+M1024+WI2":  {**BASE, "capacity": 1024, "threshold": 0.3, "normalize": True, "write_interval": 2},
}


def main():
    print(f"Device: {mx.default_device()}")
    print()
    t0 = time.time()

    # ================================================================
    # Experiment 1: All configs at critical drift levels
    # ================================================================
    print("=" * 80)
    print("Exp 1: Critical Zone Sweep (drift 0.4 - 1.2)")
    print("=" * 80)
    print()

    drifts = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]

    # Header
    header = f"  {'config':>24}"
    for d in drifts:
        header += f" {d:>5.1f}"
    print(header)
    print("  " + "-" * (24 + len(drifts) * 6))

    for name, cfg in CONFIGS.items():
        row = f"  {name:>24}"
        for d in drifts:
            r = run_experiment(cfg, drift=d, n_warmup=300, n_test=50)
            row += f" {r['hit_rate']:>4.0%} "
        print(row, flush=True)
    print()

    # ================================================================
    # Experiment 2: Mixed workload (30% topic switch)
    # ================================================================
    print("=" * 80)
    print("Exp 2: Mixed Workload (30% topic switch, 70% drift)")
    print("=" * 80)
    print()

    drifts_mixed = [0.3, 0.5, 0.7, 1.0]

    header = f"  {'config':>24}"
    for d in drifts_mixed:
        header += f" {d:>5.1f}"
    print(header)
    print("  " + "-" * (24 + len(drifts_mixed) * 6))

    for name, cfg in CONFIGS.items():
        row = f"  {name:>24}"
        for d in drifts_mixed:
            r = run_mixed_experiment(cfg, drift=d, topic_switch_pct=0.3, n_warmup=300, n_test=50)
            row += f" {r['hit_rate']:>4.0%} "
        print(row, flush=True)
    print()

    # ================================================================
    # Experiment 3: Focus on best candidates with more seeds
    # ================================================================
    print("=" * 80)
    print("Exp 3: Stability Test (3 seeds, drift=0.7 — the critical point)")
    print("=" * 80)
    print()

    focus_configs = ["baseline", "norm", "norm+τ0.3", "M1024", "norm+τ0.3+M1024",
                     "WI2", "norm+τ0.3+M1024+WI2"]
    seeds = [42, 123, 7777]

    print(f"  {'config':>24}  {'seed42':>7} {'seed123':>7} {'seed7777':>7}  {'avg':>7}")
    print("  " + "-" * 60)

    for name in focus_configs:
        cfg = CONFIGS[name]
        results = []
        row = f"  {name:>24}"
        for s in seeds:
            r = run_experiment(cfg, drift=0.7, n_warmup=300, n_test=100, seed=s)
            results.append(r["hit_rate"])
            row += f"  {r['hit_rate']:>5.1%} "
        avg = sum(results) / len(results)
        row += f"  {avg:>5.1%}"
        print(row, flush=True)
    print()

    # ================================================================
    # Experiment 4: Write interval deep dive at drift=0.7
    # ================================================================
    print("=" * 80)
    print("Exp 4: Write Interval Sweep (drift=0.7)")
    print("=" * 80)
    print()

    print(f"  {'wi':>4} {'M=256':>8} {'M=512':>8} {'M=1024':>8} {'M=2048':>8}")
    print("  " + "-" * 40)

    for wi in [1, 2, 3, 4, 5, 8]:
        row = f"  {wi:>4}"
        for m in [256, 512, 1024, 2048]:
            cfg = {**BASE, "capacity": m, "threshold": 0.5, "write_interval": wi}
            r = run_experiment(cfg, drift=0.7, n_warmup=max(300, m * wi + 10), n_test=50)
            row += f"  {r['hit_rate']:>5.1%} "
        print(row, flush=True)
    print()

    # Normalized version
    print("  (with normalize + τ=0.3)")
    print(f"  {'wi':>4} {'M=256':>8} {'M=512':>8} {'M=1024':>8} {'M=2048':>8}")
    print("  " + "-" * 40)

    for wi in [1, 2, 3, 4, 5, 8]:
        row = f"  {wi:>4}"
        for m in [256, 512, 1024, 2048]:
            cfg = {**BASE, "capacity": m, "threshold": 0.3, "normalize": True, "write_interval": wi}
            r = run_experiment(cfg, drift=0.7, n_warmup=max(300, m * wi + 10), n_test=50)
            row += f"  {r['hit_rate']:>5.1%} "
        print(row, flush=True)
    print()

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
