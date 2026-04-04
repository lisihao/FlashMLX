#!/usr/bin/env python3
"""
Hit rate optimization experiments v2 — proper cosine-controlled drift model.

Key fix: Instead of additive noise (q = base + noise*drift) which inflates
query norms over time, we use spherical interpolation so that the cosine
similarity between consecutive queries is precisely controlled.

  q_new = cos(angle) * base_normed + sin(angle) * ortho_noise
  where angle = arccos(target_cosine)

This gives exact cosine_similarity(q_new, q_old) = target_cosine.

CPU-compatible (uses pure MLX, no Metal).
"""

from __future__ import annotations

import math
import sys
import time

sys.path.insert(0, "src")

import mlx.core as mx

mx.set_default_device(mx.cpu)


def gen_drifted_query(base: mx.array, cos_sim: float) -> mx.array:
    """Generate a query with exact cosine similarity to base.

    Uses Gram-Schmidt to create orthogonal noise, then spherical interp.

    Args:
        base: [H, D] float32 query
        cos_sim: target cosine similarity (0..1)

    Returns:
        [H, D] float32 query with cos(result, base) ≈ cos_sim per head
    """
    H, D = base.shape

    # Normalize base
    base_norm = mx.sqrt(mx.sum(base * base, axis=-1, keepdims=True))
    base_hat = base / mx.maximum(base_norm, 1e-8)

    # Random direction
    noise = mx.random.normal(base.shape)

    # Gram-Schmidt: make noise orthogonal to base per head
    proj = mx.sum(noise * base_hat, axis=-1, keepdims=True)
    ortho = noise - proj * base_hat
    ortho_norm = mx.sqrt(mx.sum(ortho * ortho, axis=-1, keepdims=True))
    ortho_hat = ortho / mx.maximum(ortho_norm, 1e-8)

    # Spherical interpolation: cos(angle)*base + sin(angle)*ortho
    # gives exactly cos_sim cosine similarity
    angle = math.acos(max(-1, min(1, cos_sim)))
    result = math.cos(angle) * base_hat + math.sin(angle) * ortho_hat

    # Re-scale to original norm (not unit norm) to simulate realistic queries
    result = result * base_norm

    mx.eval(result)
    return result


class MACSimulator:
    """MAC ring cache + match simulator for hit rate testing."""

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
        self.normalize = normalize
        self.write_interval = max(1, write_interval)
        self._write_count = 0

        # Ring buffer: [M, H, D]
        self.query_cache = mx.zeros((capacity, num_heads, head_dim), dtype=mx.float32)
        self.length = 0

        # Threshold for L2 distance
        if normalize:
            # Normalized: dist = 2(1-cos), hit when dist < T_sq
            # T_sq = 2*(1-tau) where tau is cosine threshold
            self.T_sq = 2.0 * (1.0 - threshold)
        else:
            one_minus = 1.0 - threshold
            self.T_sq = 2.0 * head_dim * one_minus * one_minus

    def _l2_normalize(self, q: mx.array) -> mx.array:
        norms = mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True))
        return q / mx.maximum(norms, 1e-8)

    def write(self, query: mx.array):
        """Write query [H, D] to ring cache."""
        self._write_count += 1
        if self.write_interval > 1 and self._write_count % self.write_interval != 0:
            return

        q = query.astype(mx.float32)
        if self.normalize:
            q = self._l2_normalize(q)

        slot = self.length % self.M
        self.query_cache = self.query_cache.at[slot].multiply(0)
        self.query_cache = self.query_cache.at[slot].add(q)
        self.length += 1

    def match(self, query: mx.array) -> float:
        """Match query [H, D] against cache. Returns per-head hit rate."""
        V = min(self.length, self.M)
        if V <= 0:
            return 0.0

        q = query.astype(mx.float32)
        if self.normalize:
            q = self._l2_normalize(q)

        # q: [H, D], cache: [V, H, D]
        cache = self.query_cache[:V]
        diffs = q[None, :, :] - cache  # [V, H, D]
        dists = mx.sum(diffs * diffs, axis=-1)  # [V, H]

        best_dists = mx.min(dists, axis=0)  # [H]
        mx.eval(best_dists)

        hits = best_dists < self.T_sq
        mx.eval(hits)
        return hits.astype(mx.float32).mean().item()


def run_exp(config: dict, cos_sim: float, n_warmup: int = 300, n_test: int = 80, seed: int = 42) -> float:
    """Run experiment with controlled cosine similarity between consecutive queries.

    cos_sim: cosine similarity between each query and its predecessor.
    cos_sim=0.95 means queries drift slowly; cos_sim=0.5 means fast drift.
    """
    H, D = config.get("num_heads", 32), config.get("head_dim", 128)

    sim = MACSimulator(
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

    # Warmup
    wi = config.get("write_interval", 1)
    actual_warmup = max(n_warmup, sim.M * wi + 10)
    for _ in range(actual_warmup):
        q = gen_drifted_query(base_q, cos_sim)
        sim.write(q)
        base_q = q

    # Test
    hits = []
    for _ in range(n_test):
        q = gen_drifted_query(base_q, cos_sim)
        hr = sim.match(q)
        hits.append(hr)
        sim.write(q)
        base_q = q

    return sum(hits) / len(hits)


def run_mixed(config: dict, cos_sim: float, switch_pct: float = 0.3,
              n_warmup: int = 300, n_test: int = 80, seed: int = 42) -> float:
    """Mixed workload: switch_pct of queries are completely random (cos~0)."""
    H, D = config.get("num_heads", 32), config.get("head_dim", 128)

    sim = MACSimulator(
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
        q = gen_drifted_query(base_q, cos_sim)
        sim.write(q)
        base_q = q

    hits = []
    mx.random.seed(seed + 10000)
    for i in range(n_test):
        # Deterministic switch pattern based on seed
        u = mx.random.uniform(shape=())
        mx.eval(u)
        if u.item() < switch_pct:
            # Topic switch: completely new random query
            q = mx.random.normal((H, D))
            mx.eval(q)
        else:
            q = gen_drifted_query(base_q, cos_sim)

        hr = sim.match(q)
        hits.append(hr)
        sim.write(q)
        base_q = q

    return sum(hits) / len(hits)


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
    # Exp 1: Critical cosine sweep
    # cos_sim 0.95=very similar, 0.7=moderate, 0.3=very different
    # ================================================================
    print("=" * 90)
    print("Exp 1: Cosine Similarity Sweep (controlled angle between consecutive queries)")
    print("  cos=0.95: nearly identical queries (easy)")
    print("  cos=0.50: moderate diversity (realistic for LLM decode)")
    print("  cos=0.20: very different queries (hard)")
    print("=" * 90)
    print()

    cos_levels = [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.30, 0.20]

    header = f"  {'config':>24}"
    for c in cos_levels:
        header += f" {c:>5.2f}"
    print(header)
    print("  " + "-" * (24 + len(cos_levels) * 6))

    for name, cfg in CONFIGS.items():
        row = f"  {name:>24}"
        for c in cos_levels:
            hr = run_exp(cfg, cos_sim=c, n_warmup=300, n_test=80)
            row += f" {hr:>4.0%} "
        print(row, flush=True)
    print()

    # ================================================================
    # Exp 2: Mixed workload (30% random topic switch)
    # ================================================================
    print("=" * 90)
    print("Exp 2: Mixed Workload (30% completely random queries, 70% drifted)")
    print("=" * 90)
    print()

    cos_mixed = [0.95, 0.85, 0.70, 0.50]
    header = f"  {'config':>24}"
    for c in cos_mixed:
        header += f" {c:>5.2f}"
    print(header)
    print("  " + "-" * (24 + len(cos_mixed) * 6))

    for name, cfg in CONFIGS.items():
        row = f"  {name:>24}"
        for c in cos_mixed:
            hr = run_mixed(cfg, cos_sim=c, switch_pct=0.3, n_warmup=300, n_test=80)
            row += f" {hr:>4.0%} "
        print(row, flush=True)
    print()

    # ================================================================
    # Exp 3: Stability (3 seeds at critical point cos=0.7)
    # ================================================================
    print("=" * 90)
    print("Exp 3: Stability Test (3 seeds, cos=0.70)")
    print("=" * 90)
    print()

    focus = ["baseline", "norm", "norm+τ0.3", "norm+τ0.7",
             "M1024", "WI2", "norm+M1024", "norm+τ0.3+M1024",
             "norm+WI2", "norm+τ0.3+M1024+WI2"]
    seeds = [42, 123, 7777]

    print(f"  {'config':>24}  {'s=42':>6} {'s=123':>6} {'s=7777':>6}  {'avg':>6}")
    print("  " + "-" * 58)

    for name in focus:
        cfg = CONFIGS[name]
        rs = []
        row = f"  {name:>24}"
        for s in seeds:
            hr = run_exp(cfg, cos_sim=0.70, n_warmup=300, n_test=100, seed=s)
            rs.append(hr)
            row += f"  {hr:>4.1%}"
        avg = sum(rs) / len(rs)
        row += f"  {avg:>4.1%}"
        print(row, flush=True)
    print()

    # ================================================================
    # Exp 4: Write interval × capacity grid at cos=0.7
    # ================================================================
    print("=" * 90)
    print("Exp 4: Write Interval x Capacity Grid (cos=0.70, baseline τ=0.5)")
    print("=" * 90)
    print()

    print(f"  {'WI':>4} {'M=256':>8} {'M=512':>8} {'M=1024':>8} {'M=2048':>8}")
    print("  " + "-" * 40)
    for wi in [1, 2, 3, 4, 8]:
        row = f"  {wi:>4}"
        for m in [256, 512, 1024, 2048]:
            cfg = {**BASE, "capacity": m, "threshold": 0.5, "write_interval": wi}
            hr = run_exp(cfg, cos_sim=0.70, n_warmup=max(300, m * wi + 10), n_test=80)
            row += f"  {hr:>5.1%} "
        print(row, flush=True)
    print()

    print("  (with normalize + τ=0.3)")
    print(f"  {'WI':>4} {'M=256':>8} {'M=512':>8} {'M=1024':>8} {'M=2048':>8}")
    print("  " + "-" * 40)
    for wi in [1, 2, 3, 4, 8]:
        row = f"  {wi:>4}"
        for m in [256, 512, 1024, 2048]:
            cfg = {**BASE, "capacity": m, "threshold": 0.3, "normalize": True, "write_interval": wi}
            hr = run_exp(cfg, cos_sim=0.70, n_warmup=max(300, m * wi + 10), n_test=80)
            row += f"  {hr:>5.1%} "
        print(row, flush=True)
    print()

    # ================================================================
    # Exp 5: Threshold sensitivity (normalize mode)
    # ================================================================
    print("=" * 90)
    print("Exp 5: Threshold Sensitivity (normalized, M=512)")
    print("=" * 90)
    print()

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cos_test = [0.95, 0.80, 0.70, 0.50, 0.30]

    header = f"  {'τ':>6}"
    for c in cos_test:
        header += f" {'c=' + str(c):>7}"
    print(header)
    print("  " + "-" * (6 + len(cos_test) * 8))

    for tau in thresholds:
        cfg = {**BASE, "capacity": 512, "threshold": tau, "normalize": True}
        row = f"  {tau:>6.1f}"
        for c in cos_test:
            hr = run_exp(cfg, cos_sim=c, n_warmup=300, n_test=80)
            row += f"  {hr:>5.1%} "
        print(row, flush=True)
    print()

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
