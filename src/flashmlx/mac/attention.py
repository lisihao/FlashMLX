"""
MAC-Attention merge, rectification, and partial attention.

Two implementations of partial attention:
  - Fused Metal kernel (online softmax, single launch): used when Metal available
  - Reference MLX implementation (unfused, for testing / non-Metal fallback)

Other operations (merge, downdate, rectify) remain pure MLX.

Ported from:
  - mac_decode.py (attention + merge)
  - mac_rectification_cache.py (rectify + downdate)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from .ring_cache import MACRingCache


# ============================================================================
# Fused Metal kernel: partial attention with online softmax
# ============================================================================
# Single kernel launch replaces: GQA expand + matmul + mask + softmax + matmul
# Each threadgroup handles one (batch, query_head) pair.
# Multi-warp: each SIMD group processes KV positions in stride, maintaining
# per-warp online softmax state (running max, sum-exp, output accumulator).
# Cross-warp merge at end via threadgroup memory.

_FUSED_ATTN_SOURCE = """
    uint tg_id = threadgroup_position_in_grid.x;
    uint n = tg_id / HQ;
    uint h_q = tg_id % HQ;
    if (n >= N_QUERIES) return;

    uint warp = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    // GQA: map query head to KV head
    uint h_kv = h_q * HKV / HQ;

    // Per-head start position and scale
    int start = start_pos[n * HQ + h_q];
    float sc = scale_arr[0];

    // Load query to registers (strided layout for coalesced K/V access)
    float q_reg[ELEMS_PER_LANE];
    uint q_off = (n * HQ + h_q) * HEAD_DIM;
    for (uint e = 0; e < ELEMS_PER_LANE; e++) {
        uint d = lane + e * threads_per_simdgroup;
        q_reg[e] = (d < HEAD_DIM) ? static_cast<float>(queries[q_off + d]) : 0.0f;
    }

    // Online softmax state per warp
    float m_w = -1e38f;
    float d_w = 0.0f;
    float o_acc[ELEMS_PER_LANE];
    for (uint e = 0; e < ELEMS_PER_LANE; e++) o_acc[e] = 0.0f;

    // Main loop: each warp processes KV positions in stride
    for (int t = start + (int)warp; t < (int)SEQ_LEN; t += (int)NUM_WARPS) {
        // Q·K dot product
        float dot = 0.0f;
        uint kv_off = ((uint(n) * SEQ_LEN + uint(t)) * HKV + h_kv) * HEAD_DIM;
        for (uint e = 0; e < ELEMS_PER_LANE; e++) {
            uint d = lane + e * threads_per_simdgroup;
            if (d < HEAD_DIM) {
                dot = fma(q_reg[e], static_cast<float>(keys[kv_off + d]), dot);
            }
        }
        float score = simd_sum(dot) * sc;

        // Online softmax update
        float m_new = max(score, m_w);
        float exp_old = exp(m_w - m_new);
        float exp_cur = exp(score - m_new);
        d_w = fma(d_w, exp_old, exp_cur);

        // Accumulate weighted V
        for (uint e = 0; e < ELEMS_PER_LANE; e++) {
            uint d = lane + e * threads_per_simdgroup;
            if (d < HEAD_DIM) {
                float vv = static_cast<float>(values[kv_off + d]);
                o_acc[e] = fma(o_acc[e], exp_old, exp_cur * vv);
            }
        }
        m_w = m_new;
    }

    // Cross-warp merge via threadgroup memory
    threadgroup float tg_m[NUM_WARPS];
    threadgroup float tg_d[NUM_WARPS];
    threadgroup float tg_o[NUM_WARPS * HEAD_DIM];

    tg_m[warp] = m_w;
    tg_d[warp] = d_w;
    for (uint e = 0; e < ELEMS_PER_LANE; e++) {
        uint d = lane + e * threads_per_simdgroup;
        if (d < HEAD_DIM) tg_o[warp * HEAD_DIM + d] = o_acc[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Warp 0: merge all warp states and write final output
    if (warp == 0) {
        float fm = tg_m[0];
        float fd = tg_d[0];
        float fo[ELEMS_PER_LANE];
        for (uint e = 0; e < ELEMS_PER_LANE; e++) {
            uint d = lane + e * threads_per_simdgroup;
            fo[e] = (d < HEAD_DIM) ? tg_o[d] : 0.0f;
        }

        for (uint w = 1; w < NUM_WARPS; w++) {
            float mo = tg_m[w];
            float dw = tg_d[w];
            float mn = max(mo, fm);
            float es = exp(fm - mn);
            float eo = exp(mo - mn);
            fd = fma(fd, es, dw * eo);
            for (uint e = 0; e < ELEMS_PER_LANE; e++) {
                uint d = lane + e * threads_per_simdgroup;
                if (d < HEAD_DIM)
                    fo[e] = fma(fo[e], es, tg_o[w * HEAD_DIM + d] * eo);
            }
            fm = mn;
        }

        // Write attention output: o / sum_exp
        uint o_off = (n * HQ + h_q) * HEAD_DIM;
        float inv = (fd > 0.0f) ? (1.0f / fd) : 0.0f;
        for (uint e = 0; e < ELEMS_PER_LANE; e++) {
            uint d = lane + e * threads_per_simdgroup;
            if (d < HEAD_DIM) output[o_off + d] = fo[e] * inv;
        }

        // Write LSE = max + log(sum_exp)
        if (lane == 0) {
            lse_out[n * HQ + h_q] = (fd > 0.0f) ? (fm + log(fd)) : -1e38f;
        }
    }
"""

_fused_attn_kernel = None


def _get_fused_attn_kernel():
    global _fused_attn_kernel
    if _fused_attn_kernel is None:
        _fused_attn_kernel = mx.fast.metal_kernel(
            name="mac_fused_partial_attn",
            input_names=["queries", "keys", "values", "start_pos", "scale_arr"],
            output_names=["output", "lse_out"],
            source=_FUSED_ATTN_SOURCE,
        )
    return _fused_attn_kernel


def mac_fused_partial_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    start_pos: mx.array,
    scale: float | None = None,
) -> tuple[mx.array, mx.array]:
    """Fused partial attention with per-head start positions (Metal kernel).

    Single kernel launch using online softmax — faster than MLX's built-in
    SDPA at S >= 4K thanks to 8-warp parallelism and fused computation.

    Args:
        queries:   [N, H, D] bf16 — query vectors
        keys:      [N, S, Hkv, D] bf16 — full KV cache keys
        values:    [N, S, Hkv, D] bf16 — full KV cache values
        start_pos: [N, H] int32 — per-head attention start position
        scale:     attention scale (default: 1/sqrt(D))

    Returns:
        (output [N, H, D] f32, lse [N, H] f32)
    """
    N, H, D = queries.shape
    S = keys.shape[1]
    Hkv = keys.shape[2]

    if scale is None:
        scale = D ** (-0.5)

    # 8 warps per head: empirically optimal on Apple Silicon
    # More warps = fewer iterations per warp + better memory latency hiding
    NUM_WARPS = 8
    BLOCK_THREADS = NUM_WARPS * 32
    ELEMS_PER_LANE = (D + 31) // 32

    kernel = _get_fused_attn_kernel()
    scale_arr = mx.array([scale], dtype=mx.float32)
    start_flat = start_pos.reshape(-1).astype(mx.int32)

    outputs = kernel(
        inputs=[queries, keys, values, start_flat, scale_arr],
        output_shapes=[(N * H * D,), (N * H,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(N * H * BLOCK_THREADS, 1, 1),
        threadgroup=(BLOCK_THREADS, 1, 1),
        template=[
            ("HEAD_DIM", D),
            ("HQ", H),
            ("HKV", Hkv),
            ("SEQ_LEN", S),
            ("N_QUERIES", N),
            ("NUM_WARPS", NUM_WARPS),
            ("BLOCK_THREADS", BLOCK_THREADS),
            ("ELEMS_PER_LANE", ELEMS_PER_LANE),
        ],
        stream=mx.gpu,
    )

    output = outputs[0].reshape(N, H, D)
    lse = outputs[1].reshape(N, H)
    return output, lse


def merge_attention_states(
    o_cached: mx.array,
    lse_cached: mx.array,
    o_fresh: mx.array,
    lse_fresh: mx.array,
) -> tuple[mx.array, mx.array]:
    """Merge two attention outputs using log-sum-exp algebra.

    Implements Eq.7 from MAC-Attention paper:
        o_merged = w_cached * o_cached + w_fresh * o_fresh
    where weights are derived from LSE values in log domain.

    Args:
        o_cached:   [N, H, D] — cached attention output
        lse_cached: [N, H]    — cached log-sum-exp
        o_fresh:    [N, H, D] — fresh partial attention output
        lse_fresh:  [N, H]    — fresh log-sum-exp

    Returns:
        (o_merged [N, H, D], lse_merged [N, H])
    """
    # Numerically stable log-sum-exp merge
    lse_max = mx.maximum(lse_cached, lse_fresh)  # [N, H]
    lse_merged = lse_max + mx.log(
        mx.exp(lse_cached - lse_max) + mx.exp(lse_fresh - lse_max)
    )  # [N, H]

    # Weights in [0, 1] range
    w_cached = mx.exp(lse_cached - lse_merged)[..., None]  # [N, H, 1]
    w_fresh = mx.exp(lse_fresh - lse_merged)[..., None]  # [N, H, 1]

    o_merged = w_cached * o_cached + w_fresh * o_fresh  # [N, H, D]
    return o_merged, lse_merged


def downdate_attention(
    full_o: mx.array,
    full_lse: mx.array,
    window_o: mx.array,
    window_lse: mx.array,
    eps: float = 1e-10,
) -> tuple[mx.array, mx.array]:
    """Compute rest = full - window via log-sum-exp subtraction.

    rest represents the attention contribution from tokens BEFORE the window,
    i.e., the prefix portion that MAC caches for reuse.

    Args:
        full_o:     [N, H, D] — full attention output (over all KV)
        full_lse:   [N, H]    — full log-sum-exp
        window_o:   [N, H, D] — windowed attention output (last window_left tokens)
        window_lse: [N, H]    — windowed log-sum-exp
        eps:        numerical safety floor for z_rest

    Returns:
        (rest_o [N, H, D], rest_lse [N, H])
    """
    z_full = mx.exp(full_lse)  # [N, H]
    z_window = mx.exp(window_lse)  # [N, H]
    z_rest = mx.maximum(z_full - z_window, eps)  # [N, H]

    rest_o = (
        full_o * z_full[..., None] - window_o * z_window[..., None]
    ) / z_rest[..., None]  # [N, H, D]
    rest_lse = mx.log(z_rest)  # [N, H]

    return rest_o, rest_lse


def mac_partial_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    start_pos: mx.array,
    scale: float | None = None,
) -> tuple[mx.array, mx.array]:
    """Compute attention only over [start_pos:] for each (batch, head).

    Auto-dispatches to fused Metal kernel when available,
    falls back to unfused MLX reference otherwise.

    Args:
        queries:   [N, H, D] bf16 — query vectors
        keys:      [N, S, Hkv, D] bf16 — full KV cache keys
        values:    [N, S, Hkv, D] bf16 — full KV cache values
        start_pos: [N, H] int32 — per-head attention start position
        scale:     attention scale (default: 1/sqrt(D))

    Returns:
        (output [N, H, D], lse [N, H])
    """
    if mx.metal.is_available():
        return mac_fused_partial_attention(
            queries, keys, values, start_pos, scale
        )
    return _mac_partial_attention_reference(
        queries, keys, values, start_pos, scale
    )


def _mac_partial_attention_reference(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    start_pos: mx.array,
    scale: float | None = None,
) -> tuple[mx.array, mx.array]:
    """Reference implementation: unfused mask + softmax + matmul (pure MLX).

    Used for correctness testing and non-Metal fallback.
    """
    N, H, D = queries.shape
    S = keys.shape[1]
    Hkv = keys.shape[2]

    if scale is None:
        scale = D ** (-0.5)

    q = queries[:, :, None, :]  # [N, H, 1, D]

    groups = H // Hkv
    if groups > 1:
        k = mx.repeat(keys, groups, axis=2)
        v = mx.repeat(values, groups, axis=2)
    else:
        k = keys
        v = values

    k = mx.transpose(k, (0, 2, 1, 3))  # [N, H, S, D]
    v = mx.transpose(v, (0, 2, 1, 3))

    scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * scale  # [N, H, 1, S]
    positions = mx.arange(S)
    mask = positions[None, None, None, :] >= start_pos[:, :, None, None]
    scores = mx.where(mask, scores, mx.array(-1e9, dtype=scores.dtype))

    lse = mx.logsumexp(scores, axis=-1).squeeze(2)  # [N, H]
    weights = mx.softmax(scores, axis=-1)
    output = (weights @ v).squeeze(2)  # [N, H, D]

    return output, lse


def mac_windowed_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    window_left: int,
    scale: float | None = None,
) -> tuple[mx.array, mx.array]:
    """Compute attention over only the last window_left KV tokens.

    Used in the rectification step to separate window vs rest contributions.

    Args:
        queries:     [N, H, D] bf16
        keys:        [N, S, Hkv, D] bf16
        values:      [N, S, Hkv, D] bf16
        window_left: number of recent tokens to attend to
        scale:       attention scale (default: 1/sqrt(D))

    Returns:
        (output [N, H, D], lse [N, H])
    """
    S = keys.shape[1]
    # Window starts at max(0, S - window_left)
    start = max(0, S - window_left)
    start_pos = mx.full(
        (queries.shape[0], queries.shape[1]), start, dtype=mx.int32
    )
    return mac_partial_attention(queries, keys, values, start_pos, scale)


def mac_rectify_and_update(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    full_o: mx.array,
    full_lse: mx.array,
    ring_cache: MACRingCache,
    req_ids: mx.array,
    window_left: int = 256,
    scale: float | None = None,
) -> None:
    """Rectification step: downdate full attention and update ring cache.

    1. Windowed attention: compute attention over last window_left tokens
    2. Downdate: rest = full - window (log-sum-exp subtraction)
    3. Update ring cache: write (query, rest_o, rest_lse) for next match

    Args:
        queries:     [N, H, D] — pre-RoPE queries (for matching in future steps)
        keys:        [N, S, Hkv, D] — full KV cache keys
        values:      [N, S, Hkv, D] — full KV cache values
        full_o:      [N, H, D] — full attention output (this step)
        full_lse:    [N, H] — full attention LSE
        ring_cache:  MACRingCache instance
        req_ids:     [N] int32 — request indices
        window_left: rectification window size
        scale:       attention scale
    """
    # 1. Windowed attention
    window_o, window_lse = mac_windowed_attention(
        queries, keys, values, window_left, scale
    )

    # 2. Downdate: rest = full - window
    rest_o, rest_lse = downdate_attention(full_o, full_lse, window_o, window_lse)

    # 3. Update ring cache
    ring_cache.update(req_ids, queries, rest_o.astype(mx.bfloat16), rest_lse)
