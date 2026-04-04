"""
Partial Attention - 块状访问优化版本

优化点：
1. 改为 block-stride 访问模式（连续块，缓存友好）
2. 保持其他优化（online softmax, 8 warps）

性能预期：
- 当前：跨度访问导致缓存命中率低
- 优化后：连续访问提升缓存命中率，预期 1.5-2× 提升
"""

from __future__ import annotations

import mlx.core as mx

_BLOCK_STRIDE_ATTN_SOURCE = """
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

    // Load query to registers
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

    // OPTIMIZATION: Block-stride instead of interleaved
    // Each warp processes a contiguous block of KV positions
    int total_tokens = int(SEQ_LEN) - start;
    int block_size = (total_tokens + int(NUM_WARPS) - 1) / int(NUM_WARPS);
    int t_start = start + int(warp) * block_size;
    int t_end = min(t_start + block_size, int(SEQ_LEN));

    // Main loop: contiguous access for better cache locality
    for (int t = t_start; t < t_end; t++) {
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

    // Cross-warp merge via threadgroup memory (same as original)
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

_block_stride_kernel = None


def _get_block_stride_kernel():
    global _block_stride_kernel
    if _block_stride_kernel is None:
        _block_stride_kernel = mx.fast.metal_kernel(
            name="mac_partial_attention_block_stride",
            input_names=["queries", "keys", "values", "start_pos", "scale_arr"],
            output_names=["output", "lse_out"],
            source=_BLOCK_STRIDE_ATTN_SOURCE,
        )
    return _block_stride_kernel


def mac_partial_attention_block_stride(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    start_pos: mx.array,
    scale: float | None = None,
) -> tuple[mx.array, mx.array]:
    """Block-stride optimized partial attention.

    Args:
        queries:   [N, H, D] bf16
        keys:      [N, S, Hkv, D] bf16
        values:    [N, S, Hkv, D] bf16
        start_pos: [N, H] int32
        scale:     attention scale (default: 1/sqrt(D))

    Returns:
        (output [N, H, D] f32, lse [N, H] f32)
    """
    N, H, D = queries.shape
    S = keys.shape[1]
    Hkv = keys.shape[2]

    if scale is None:
        scale = D ** (-0.5)

    NUM_WARPS = 8
    BLOCK_THREADS = NUM_WARPS * 32
    ELEMS_PER_LANE = (D + 31) // 32

    kernel = _get_block_stride_kernel()
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
