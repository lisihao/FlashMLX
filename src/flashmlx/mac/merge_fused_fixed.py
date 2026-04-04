"""
融合 Merge Kernel - FIXED VERSION

修复原版本的bug：
1. ❌ 原版：每个线程重复计算lse_merged（浪费+可能不一致）
2. ❌ 原版：并发写lse_merged可能有race condition
3. ✅ 修复：使用threadgroup shared memory + barrier同步
"""

from __future__ import annotations

import mlx.core as mx

_MERGE_FUSED_SOURCE_FIXED = """
    // Strategy: Use threadgroup memory for LSE computation
    // - First thread of each (n,h) block computes LSE
    // - Barrier to sync
    // - All threads use the shared LSE value

    uint global_idx = thread_position_in_grid.x;
    uint total_elements = NUM_QUERIES * NUM_HEADS * HEAD_DIM;
    if (global_idx >= total_elements) return;

    uint nh = global_idx / HEAD_DIM;
    uint d = global_idx % HEAD_DIM;

    // Threadgroup index for synchronization
    uint tg_idx = thread_index_in_threadgroup;
    uint threads_per_nh = HEAD_DIM;  // All threads processing same (n,h)

    // Shared memory for LSE (one per threadgroup)
    // Each threadgroup handles multiple (n,h) pairs
    threadgroup float shared_lse_merged[256 / HEAD_DIM];  // Assuming BLOCK_SIZE=256
    uint local_nh = tg_idx / HEAD_DIM;  // Which (n,h) within this threadgroup

    // First thread of each (n,h) computes merged LSE
    if (d == 0) {
        float lse_c = lse_cached[nh];
        float lse_f = lse_fresh[nh];

        float lse_max = max(lse_c, lse_f);
        float lse_merged_val = lse_max + log(exp(lse_c - lse_max) + exp(lse_f - lse_max));

        shared_lse_merged[local_nh] = lse_merged_val;
        lse_merged[nh] = lse_merged_val;  // Write to global
    }

    // Barrier: wait for LSE computation
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All threads now use the shared LSE
    float lse_merged_val = shared_lse_merged[local_nh];
    float lse_c = lse_cached[nh];
    float lse_f = lse_fresh[nh];

    // Compute weights
    float w_c = exp(lse_c - lse_merged_val);
    float w_f = exp(lse_f - lse_merged_val);

    // Read and merge outputs
    float o_c = o_cached[global_idx];
    float o_f = o_fresh[global_idx];
    float o_m = w_c * o_c + w_f * o_f;

    // Write merged output
    o_merged[global_idx] = o_m;
"""


def merge_attention_states_fused(
    o_cached: mx.array,
    lse_cached: mx.array,
    o_fresh: mx.array,
    lse_fresh: mx.array,
) -> tuple[mx.array, mx.array]:
    """FIXED fused merge - uses shared memory + barriers for correctness.

    Args:
        o_cached:   [N, H, D] — cached attention output
        lse_cached: [N, H]    — cached log-sum-exp
        o_fresh:    [N, H, D] — fresh partial attention output
        lse_fresh:  [N, H]    — fresh log-sum-exp

    Returns:
        (o_merged [N, H, D], lse_merged [N, H])
    """
    N, H, D = o_cached.shape

    # Flatten for kernel processing
    o_cached_flat = o_cached.reshape(-1)
    o_fresh_flat = o_fresh.reshape(-1)
    lse_cached_flat = lse_cached.reshape(-1)
    lse_fresh_flat = lse_fresh.reshape(-1)

    # Use HEAD_DIM as threadgroup size for proper synchronization
    # Each threadgroup processes HEAD_DIM threads (one per dimension of one (n,h))
    BLOCK_SIZE = D  # Critical: must match HEAD_DIM for barrier sync
    total_elements = N * H * D

    kernel = mx.fast.metal_kernel(
        name="merge_attention_fused_fixed",
        input_names=["o_cached", "lse_cached", "o_fresh", "lse_fresh"],
        output_names=["o_merged", "lse_merged"],
        source=_MERGE_FUSED_SOURCE_FIXED,
    )

    # Grid: one thread per element, but grouped by HEAD_DIM
    num_threadgroups = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    outputs = kernel(
        inputs=[o_cached_flat, lse_cached_flat, o_fresh_flat, lse_fresh_flat],
        output_shapes=[(N * H * D,), (N * H,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(num_threadgroups * BLOCK_SIZE, 1, 1),
        threadgroup=(BLOCK_SIZE, 1, 1),
        template=[
            ("NUM_QUERIES", N),
            ("NUM_HEADS", H),
            ("HEAD_DIM", D),
        ],
        stream=mx.gpu,
    )

    o_merged = outputs[0].reshape(N, H, D)
    lse_merged = outputs[1].reshape(N, H)

    return o_merged, lse_merged
