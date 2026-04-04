"""
MACDecodeWrapper — Full MAC-Attention decode workflow on Metal/MLX.

Assembles the three-stage MAC pipeline per decode step:
  1. Match:  L2 distance search in ring cache → hit/miss per head
  2. Amend:  Partial attention (hit: skip prefix, miss: full) + merge cached
  3. Complete: Rectify (downdate full - window) + update ring cache

Ported from e2e_mac_workflow_example.py (CUDA/FlashInfer → pure MLX).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from .attention import (
    mac_partial_attention,
    mac_rectify_and_update,
    merge_attention_states,
)
from .match import mac_ring_match
from .ring_cache import MACRingCache


@dataclass
class MACDecodeStats:
    """Per-step statistics from MAC decode."""

    hit_rate: float  # fraction of (N, H) entries that hit
    avg_left_start: float  # average attention start position
    num_hits: int
    num_total: int


class MACDecodeWrapper:
    """Complete MAC-Attention decode wrapper for MLX.

    Usage:
        mac = MACDecodeWrapper(
            max_requests=64, capacity=1024,
            num_heads=32, num_kv_heads=8, head_dim=128,
        )
        # Per decode step:
        output = mac(queries, keys, values, req_ids)
    """

    def __init__(
        self,
        max_requests: int = 64,
        capacity: int = 1024,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        threshold: float = 0.6,
        band_r: int = 256,
        window_left: int = 256,
        rows_per_tile: int = 64,
        normalize_queries: bool = False,
    ):
        self.ring_cache = MACRingCache(
            max_requests=max_requests,
            capacity=capacity,
            num_heads=num_heads,
            head_dim=head_dim,
            normalize=normalize_queries,
        )
        self.num_kv_heads = num_kv_heads
        self.threshold = threshold
        self.band_r = band_r
        self.window_left = window_left
        self.rows_per_tile = rows_per_tile
        self.normalize_queries = normalize_queries
        self._hit_raw: mx.array | None = None
        self._left_raw: mx.array | None = None
        self._nh_raw: tuple[int, int] = (0, 0)

        # When queries are L2-normalized, L2 distance = 2(1-cos).
        # The Metal kernel threshold formula is: T_sq = 2*D*(1-τ')²
        # We need: 2*D*(1-τ')² = 2*(1-τ)  where τ = cosine threshold
        # Solving: τ' = 1 - sqrt((1-τ)/D)
        if normalize_queries:
            one_minus_tau = max(1.0 - threshold, 0.0)
            self._match_threshold = 1.0 - math.sqrt(one_minus_tau / head_dim)
        else:
            self._match_threshold = threshold

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        req_ids: mx.array,
        scale: float | None = None,
    ) -> mx.array:
        """Run one MAC decode step.

        Args:
            queries: [N, H, D] bf16 — pre-RoPE query vectors
            keys:    [N, S, Hkv, D] bf16 — full KV cache keys
            values:  [N, S, Hkv, D] bf16 — full KV cache values
            req_ids: [N] int32 — request indices
            scale:   attention scale (default: 1/sqrt(D))

        Returns:
            output: [N, H, D] — attention output
        """
        N, H, D = queries.shape

        # --- Step 1: Match ---
        # When normalize_queries is enabled, L2 match operates on unit-norm
        # queries so that distance ↔ cosine similarity.
        if self.normalize_queries:
            q_f32 = queries.astype(mx.float32)
            norms = mx.sqrt(mx.sum(q_f32 * q_f32, axis=-1, keepdims=True))
            norms = mx.maximum(norms, 1e-8)
            match_q = (q_f32 / norms).astype(queries.dtype)
        else:
            match_q = queries

        hit, left_start, indices = mac_ring_match(
            self.ring_cache,
            match_q,
            req_ids,
            threshold=self._match_threshold,
            band_r=self.band_r,
            rows_per_tile=self.rows_per_tile,
        )

        # --- Step 2: Attention ---
        # For hits: compute partial attention from left_start
        # For misses: left_start=0, so this is full attention
        # We compute everything with the per-head start positions
        fresh_o, fresh_lse = mac_partial_attention(
            queries, keys, values, left_start, scale
        )

        # Merge with cached for hit heads
        cached_o, cached_lse = self.ring_cache.fetch(req_ids, indices)

        merged_o, merged_lse = merge_attention_states(
            cached_o.astype(fresh_o.dtype),
            cached_lse,
            fresh_o,
            fresh_lse,
        )

        # Select: hit → merged, miss → fresh (full attention)
        output = mx.where(hit[..., None], merged_o, fresh_o)
        output_lse = mx.where(hit, merged_lse, fresh_lse)

        # --- Step 3: Rectify + update cache ---
        mac_rectify_and_update(
            queries,
            keys,
            values,
            output,
            output_lse,
            self.ring_cache,
            req_ids,
            window_left=self.window_left,
            scale=scale,
        )

        # Store raw tensors for lazy stats (no sync in hot path)
        self._hit_raw = hit
        self._left_raw = left_start
        self._nh_raw = (N, H)

        return output

    @property
    def last_stats(self) -> MACDecodeStats | None:
        """Statistics from the most recent decode step (computed lazily)."""
        if self._hit_raw is None:
            return None
        hit = self._hit_raw
        left = self._left_raw
        N, H = self._nh_raw
        mx.eval(hit, left)
        num_hits = int(hit.sum().item())
        num_total = N * H
        return MACDecodeStats(
            hit_rate=num_hits / max(num_total, 1),
            avg_left_start=float(left.astype(mx.float32).mean().item()),
            num_hits=num_hits,
            num_total=num_total,
        )

    def reset(self, req_ids: mx.array | None = None) -> None:
        """Reset ring cache for given requests (or all)."""
        self.ring_cache.reset(req_ids)
