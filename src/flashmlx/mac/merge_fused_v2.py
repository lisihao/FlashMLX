"""
融合 Merge Kernel - V2 (最简单最安全的版本)

核心改动：每个threadgroup处理一个(n,h)对的所有D维度
- 避免跨threadgroup的同步问题
- 清晰的并行模型
- 安全的shared memory使用
"""

from __future__ import annotations

import mlx.core as mx

_MERGE_V2_SOURCE = """
    // Grid: (N * H, 1, 1) — one threadgroup per (n, h)
    // Threadgroup: (BLOCK_SIZE, 1, 1) — typically 128 or 256

    uint nh = threadgroup_position_in_grid.x;
    if (nh >= NUM_QUERIES * NUM_HEADS) return;

    uint tid = thread_index_in_threadgroup;
    uint n = nh / NUM_HEADS;
    uint h = nh % NUM_HEADS;

    // Shared memory for LSE (computed once, shared by all threads)
    threadgroup float shared_lse_merged;
    threadgroup float shared_w_cached;
    threadgroup float shared_w_fresh;

    // First thread computes LSE and weights
    if (tid == 0) {
        float lse_c = lse_cached[nh];
        float lse_f = lse_fresh[nh];

        // Numerically stable log-sum-exp
        float lse_max = max(lse_c, lse_f);
        float lse_merged_val = lse_max + log(exp(lse_c - lse_max) + exp(lse_f - lse_max));

        // Compute weights
        float w_c = exp(lse_c - lse_merged_val);
        float w_f = exp(lse_f - lse_merged_val);

        // Store in shared memory
        shared_lse_merged = lse_merged_val;
        shared_w_cached = w_c;
        shared_w_fresh = w_f;

        // Write LSE to global memory
        lse_merged[nh] = lse_merged_val;
    }

    // Wait for thread 0 to finish
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All threads read shared weights
    float w_c = shared_w_cached;
    float w_f = shared_w_fresh;

    // Each thread processes HEAD_DIM / BLOCK_SIZE elements
    uint base_idx = nh * HEAD_DIM;
    for (uint d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        uint idx = base_idx + d;
        float o_c = o_cached[idx];
        float o_f = o_fresh[idx];
        o_merged[idx] = w_c * o_c + w_f * o_f;
    }
"""


def merge_attention_states_fused_v2(
    o_cached: mx.array,
    lse_cached: mx.array,
    o_fresh: mx.array,
    lse_fresh: mx.array,
) -> tuple[mx.array, mx.array]:
    """V2: 每个threadgroup处理一个(n,h)，最安全最清晰。

    Args:
        o_cached:   [N, H, D] — cached attention output
        lse_cached: [N, H]    — cached log-sum-exp
        o_fresh:    [N, H, D] — fresh partial attention output
        lse_fresh:  [N, H]    — fresh log-sum-exp

    Returns:
        (o_merged [N, H, D], lse_merged [N, H])
    """
    N, H, D = o_cached.shape

    # Flatten for kernel
    o_cached_flat = o_cached.reshape(-1)
    o_fresh_flat = o_fresh.reshape(-1)
    lse_cached_flat = lse_cached.reshape(-1)
    lse_fresh_flat = lse_fresh.reshape(-1)

    BLOCK_SIZE = min(256, D)  # Use 256 threads per threadgroup (standard)

    kernel = mx.fast.metal_kernel(
        name="merge_attention_v2",
        input_names=["o_cached", "lse_cached", "o_fresh", "lse_fresh"],
        output_names=["o_merged", "lse_merged"],
        source=_MERGE_V2_SOURCE,
    )

    outputs = kernel(
        inputs=[o_cached_flat, lse_cached_flat, o_fresh_flat, lse_fresh_flat],
        output_shapes=[(N * H * D,), (N * H,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(N * H, 1, 1),  # One threadgroup per (n, h)
        threadgroup=(BLOCK_SIZE, 1, 1),
        template=[
            ("NUM_QUERIES", N),
            ("NUM_HEADS", H),
            ("HEAD_DIM", D),
            ("BLOCK_SIZE", BLOCK_SIZE),
        ],
        stream=mx.gpu,
    )

    o_merged = outputs[0].reshape(N, H, D)
    lse_merged = outputs[1].reshape(N, H)

    return o_merged, lse_merged
