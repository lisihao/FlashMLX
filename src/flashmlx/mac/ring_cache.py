"""
MACRingCache — Per-request ring buffer for MAC-Attention decode acceleration.

Ported from MAC-Attention CUDA (mac_prefill_update_cache.cu).
Stores pre-RoPE queries, rectified attention outputs, and log-sum-exp values
in circular buffers for computation reuse during decode.

Data layout: [R, M, H, D] where
  R = max concurrent requests
  M = ring capacity (slots per request)
  H = number of query heads
  D = head dimension
"""

from __future__ import annotations

import mlx.core as mx


class MACRingCache:
    """Per-request ring buffer for MAC-Attention.

    Each request has M slots arranged in a circular buffer.
    New entries are written at position (request_length[req] % M),
    overwriting the oldest entry when full.

    Buffers:
        query_cache:  [R, M, H, D] bf16 — pre-RoPE queries for L2 matching
        attn_cache:   [R, M, H, D] bf16 — rectified attention outputs (rest = full - window)
        lse_cache:    [R, M, H]    f32  — log-sum-exp of rectified attention
        request_length: [R] int32       — cumulative token count per request
    """

    def __init__(
        self,
        max_requests: int,
        capacity: int,
        num_heads: int,
        head_dim: int,
        normalize: bool = False,
    ):
        self.max_requests = max_requests
        self.capacity = capacity
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.normalize = normalize

        self.query_cache = mx.zeros(
            (max_requests, capacity, num_heads, head_dim), dtype=mx.bfloat16
        )
        self.attn_cache = mx.zeros(
            (max_requests, capacity, num_heads, head_dim), dtype=mx.bfloat16
        )
        self.lse_cache = mx.full(
            (max_requests, capacity, num_heads), -1e9, dtype=mx.float32
        )
        self.request_length = mx.zeros((max_requests,), dtype=mx.int32)

    def reset(self, req_ids: mx.array | None = None) -> None:
        """Reset ring buffers for given requests (or all if None)."""
        if req_ids is None:
            self.query_cache = mx.zeros_like(self.query_cache)
            self.attn_cache = mx.zeros_like(self.attn_cache)
            self.lse_cache = mx.full_like(self.lse_cache, -1e9)
            self.request_length = mx.zeros_like(self.request_length)
        else:
            R, M, H, D = (
                self.max_requests,
                self.capacity,
                self.num_heads,
                self.head_dim,
            )
            zeros_qk = mx.zeros((M, H, D), dtype=mx.bfloat16)
            neg_inf_lse = mx.full((M, H), -1e9, dtype=mx.float32)
            for req_id in req_ids.tolist():
                self.query_cache[req_id] = zeros_qk
                self.attn_cache[req_id] = zeros_qk
                self.lse_cache[req_id] = neg_inf_lse
            self.request_length = self.request_length.at[req_ids].add(-self.request_length[req_ids])

    def update(
        self,
        req_ids: mx.array,
        new_queries: mx.array,
        new_attn: mx.array,
        new_lse: mx.array,
    ) -> None:
        """Write new entries into ring buffers and bump request_length.

        Mirrors mac_prefill_update_cache_kernel: for each request, writes at
        slot = request_length[req] % M, then increments request_length.

        When ``normalize`` is True, queries are L2-normalized before storage
        so that L2 distance in the match kernel corresponds to cosine distance.

        Args:
            req_ids:     [N] int32 — request indices
            new_queries: [N, H, D] bf16 — pre-RoPE queries
            new_attn:    [N, H, D] bf16 — rectified attention outputs
            new_lse:     [N, H] f32 — log-sum-exp values
        """
        M = self.capacity

        # Optional L2 normalization for matching
        if self.normalize:
            q_f32 = new_queries.astype(mx.float32)
            norms = mx.sqrt(mx.sum(q_f32 * q_f32, axis=-1, keepdims=True))
            norms = mx.maximum(norms, 1e-8)
            new_queries = (q_f32 / norms).astype(new_queries.dtype)

        lengths = self.request_length[req_ids]  # [N]
        slots = lengths % M  # [N] — circular position

        # Scatter into [R, M, H, D] using zero-then-add pattern
        # (avoids floating-point precision loss from large-delta add)
        req_idx = req_ids  # [N]
        slot_idx = slots  # [N]

        self.query_cache = self.query_cache.at[req_idx, slot_idx].multiply(0)
        self.query_cache = self.query_cache.at[req_idx, slot_idx].add(new_queries)

        self.attn_cache = self.attn_cache.at[req_idx, slot_idx].multiply(0)
        self.attn_cache = self.attn_cache.at[req_idx, slot_idx].add(new_attn)

        self.lse_cache = self.lse_cache.at[req_idx, slot_idx].multiply(0)
        self.lse_cache = self.lse_cache.at[req_idx, slot_idx].add(new_lse)

        # Bump request_length
        ones = mx.ones_like(req_ids)
        self.request_length = self.request_length.at[req_ids].add(ones)

    def fetch(
        self,
        req_ids: mx.array,
        indices: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Fetch cached attention output and LSE at given ring indices.

        Args:
            req_ids: [N] int32 — request indices
            indices: [N, H] int32 — ring buffer slot indices (from match kernel)

        Returns:
            (attn_out [N, H, D] bf16, lse [N, H] f32)
        """
        N = req_ids.shape[0]
        H = self.num_heads

        # Vectorized gather using advanced indexing (no Python loops)
        # attn_cache[R, M, H, D] → index dims 0,1,2 with [N,H] arrays
        req_idx = mx.broadcast_to(req_ids[:, None], (N, H))  # [N, H]
        h_idx = mx.broadcast_to(mx.arange(H)[None, :], (N, H))  # [N, H]

        attn_out = self.attn_cache[req_idx, indices, h_idx]  # [N, H, D]
        lse_out = self.lse_cache[req_idx, indices, h_idx]  # [N, H]

        return attn_out, lse_out

    @property
    def valid_lengths(self) -> mx.array:
        """Per-request valid entry count: min(request_length, capacity)."""
        return mx.minimum(self.request_length, self.capacity)
