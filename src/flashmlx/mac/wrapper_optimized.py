"""
MACDecodeWrapper OPTIMIZED VERSION - 只替换merge kernel

完全复制原始wrapper逻辑，仅把merge_attention_states替换为merge_attention_states_fused
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from .attention import mac_fused_partial_attention, mac_rectify_and_update
from .match import mac_ring_match
from .merge_fused import merge_attention_states_fused
from .ring_cache import MACRingCache


@dataclass
class MACDecodeStats:
    hit_rate: float
    avg_left_start: float
    num_hits: int
    num_total: int


class MACDecodeWrapperOptimized:
    """完全复制原始wrapper，只替换merge kernel"""

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

        # CRITICAL: L2 distance threshold conversion for normalized queries
        # When queries are L2-normalized, L2 distance = 2(1-cos)
        # Formula: τ' = 1 - sqrt((1-τ)/D) where τ = cosine threshold
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
    ) -> mx.array:
        """完全复制原始逻辑，只改merge kernel"""
        N, Hq, D = queries.shape
        _, S, Hkv, _ = keys.shape
        scale = D**-0.5

        if self.normalize_queries:
            q_f32 = queries.astype(mx.float32)
            norms = mx.sqrt(mx.sum(q_f32 * q_f32, axis=-1, keepdims=True))
            norms = mx.maximum(norms, 1e-8)
            queries_normalized = (q_f32 / norms).astype(queries.dtype)
        else:
            queries_normalized = queries

        # Match
        hit, left_start, indices = mac_ring_match(
            self.ring_cache,
            queries_normalized,
            req_ids,
            threshold=self._match_threshold,  # ← 修复：使用转换后的threshold
            band_r=self.band_r,
            rows_per_tile=self.rows_per_tile,
        )

        # Partial attention (fused Metal kernel)
        fresh_o, fresh_lse = mac_fused_partial_attention(
            queries, keys, values, left_start, scale
        )

        # Merge (← 唯一改动：使用fused版本)
        cached_o, cached_lse = self.ring_cache.fetch(req_ids, indices)

        merged_o, merged_lse = merge_attention_states_fused(  # ← 这里改了
            cached_o.astype(fresh_o.dtype),
            cached_lse,
            fresh_o,
            fresh_lse,
        )

        # Select hit/miss (完全复制原始逻辑)
        output = mx.where(hit[..., None], merged_o, fresh_o)
        output_lse = mx.where(hit, merged_lse, fresh_lse)

        # Rectify + update (完全复制原始逻辑)
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
        self._nh_raw = (N, Hq)

        return output.astype(mx.float32)

    @property
    def last_stats(self) -> MACDecodeStats | None:
        """Statistics from the most recent decode step (computed lazily)."""
        if self._hit_raw is None:
            return None

        hit = self._hit_raw
        left_start = self._left_raw
        N, H = self._nh_raw

        # Compute stats (triggers sync, only when accessed)
        num_hits = mx.sum(hit.astype(mx.int32)).item()
        num_total = N * H
        hit_rate = num_hits / max(num_total, 1)
        avg_left = mx.mean(left_start.astype(mx.float32)).item()

        return MACDecodeStats(
            hit_rate=hit_rate,
            avg_left_start=avg_left,
            num_hits=num_hits,
            num_total=num_total,
        )
