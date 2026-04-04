"""
MAC-Attention L2 match kernel — Metal port via mx.fast.metal_kernel.

Ported from macMatch.cu (CUDA). Two-phase L2 distance matching:
  Phase 1 (tile): Each threadgroup handles one (query, tile) pair.
    - Load query to threadgroup memory
    - Each SIMD group processes rows in stride, computing L2 distance
    - Intra-SIMD reduction via simd_sum, cross-SIMD reduction via threadgroup memory
  Phase 2 (reduce): Each SIMD group handles one (n, h) pair.
    - Scans partial_min/partial_idx across tiles
    - SIMD reduction to find global min
    - Threshold check + ring index → global index → left_start

CUDA→Metal translations:
  __shfl_xor_sync → simd_sum (for L2 accumulation)
  __shfl_down_sync → simd_shuffle_down
  __syncthreads → threadgroup_barrier(mem_flags::mem_threadgroup)
  __shared__ → threadgroup
  blockIdx → threadgroup_position_in_grid
  threadIdx → thread_index_in_threadgroup
  warp (32 threads) → SIMD group (32 threads on Apple GPU)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from .ring_cache import MACRingCache

# ---------------------------------------------------------------------------
# Metal kernel source: Phase 1 — Tile L2 minimum
# ---------------------------------------------------------------------------
# mx.fast.metal_kernel auto-generates the function signature from input_names
# and output_names. We only write the body. Available attributes:
#   thread_position_in_grid, threadgroup_position_in_grid,
#   thread_index_in_threadgroup, threads_per_threadgroup,
#   simdgroup_index_in_threadgroup, thread_index_in_simdgroup,
#   threads_per_simdgroup

_TILE_HEADER = """
inline bool better_pair(float val, int idx, float bval, int bidx) {
    return (val < bval) || ((val == bval) && (idx < bidx));
}
"""

_TILE_SOURCE = """
    // Grid layout: grid.x = N * num_tiles (one threadgroup per (n, tile) pair)
    //              grid.y = H (one row of threadgroups per head)
    // Threadgroup: (BLOCK_THREADS, 1, 1) — typically 256 = 8 SIMD groups
    //
    // Template constants: HEAD_DIM, M_CAP, NUM_HEADS, NUM_QUERIES,
    //                     NUM_TILES, RPT (rows per tile), NUM_WARPS

    uint tg_x = threadgroup_position_in_grid.x;  // n * NUM_TILES + tile
    uint h = threadgroup_position_in_grid.y;
    if (h >= NUM_HEADS) return;

    uint n = tg_x / NUM_TILES;
    uint tile = tg_x % NUM_TILES;
    if (n >= NUM_QUERIES) return;

    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint warp = simdgroup_index_in_threadgroup;

    // Output index into partial arrays [N * H * NUM_TILES]
    uint idxT = (n * NUM_HEADS + h) * NUM_TILES + tile;

    // Read ring buffer valid length
    int L = request_length[n];  // implicit: req = n
    int V = (L < (int)M_CAP) ? L : (int)M_CAP;

    int tile_start = (int)tile * (int)RPT;

    if (V <= 0 || tile_start >= V) {
        if (tid == 0) {
            partial_min[idxT] = INFINITY;
            partial_idx[idxT] = 0;
        }
        return;
    }

    int tile_end = tile_start + (int)RPT;
    if (tile_end > V) tile_end = V;
    int rows_in_tile = tile_end - tile_start;

    // Load query[n, h, :] into threadgroup memory (bf16 → float)
    threadgroup float q_shared[HEAD_DIM];
    uint q_base = (n * NUM_HEADS + h) * HEAD_DIM;
    for (uint d = tid; d < HEAD_DIM; d += BLOCK_THREADS) {
        q_shared[d] = static_cast<float>(queries[q_base + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-SIMD-group best (val, idx)
    float best_val = INFINITY;
    int best_idx = 0;

    // Each SIMD group processes rows in stride
    for (int rr = (int)warp; rr < rows_in_tile; rr += (int)NUM_WARPS) {
        int local_slot = tile_start + rr;

        // L2 distance: sum((q[d] - cache[d])^2)
        float acc = 0.0f;
        uint c_base = ((uint(n) * M_CAP + uint(local_slot)) * NUM_HEADS + h) * HEAD_DIM;

        for (uint d = lane; d < HEAD_DIM; d += threads_per_simdgroup) {
            float qv = q_shared[d];
            float cv = static_cast<float>(q_cache[c_base + d]);
            float diff = qv - cv;
            acc = fma(diff, diff, acc);
        }

        // Warp-level sum reduction
        acc = simd_sum(acc);

        if (lane == 0 && better_pair(acc, local_slot, best_val, best_idx)) {
            best_val = acc;
            best_idx = local_slot;
        }
    }

    // Cross-SIMD-group reduction via threadgroup memory
    threadgroup float warp_vals[NUM_WARPS];
    threadgroup int warp_idxs[NUM_WARPS];
    if (lane == 0) {
        warp_vals[warp] = best_val;
        warp_idxs[warp] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First SIMD group reduces all warp results
    if (warp == 0) {
        float v = INFINITY;
        int idx = 0;
        if (lane < NUM_WARPS) {
            v = warp_vals[lane];
            idx = warp_idxs[lane];
        }
        // Shuffle-down reduction within SIMD group
        for (uint off = 16; off > 0; off >>= 1) {
            float v2 = simd_shuffle_down(v, off);
            int i2 = simd_shuffle_down(idx, off);
            if (better_pair(v2, i2, v, idx)) {
                v = v2;
                idx = i2;
            }
        }
        if (lane == 0) {
            partial_min[idxT] = v;
            partial_idx[idxT] = idx;
        }
    }
"""

# ---------------------------------------------------------------------------
# Metal kernel source: Phase 2 — Final reduce + threshold + emit
# ---------------------------------------------------------------------------

_REDUCE_HEADER = """
inline bool better_pair(float val, int idx, float bval, int bidx) {
    return (val < bval) || ((val == bval) && (idx < bidx));
}
"""

_REDUCE_SOURCE = """
    // Grid: (N * H * 32, 1, 1) — one SIMD group per (n, h)
    // Threadgroup: (32, 1, 1) — single SIMD group
    //
    // threshold_val is a scalar float input (not template — templates only support int/bool)

    uint nh = thread_position_in_grid.x / threads_per_simdgroup;
    uint lane = thread_index_in_simdgroup;
    if (nh >= NUM_QUERIES * NUM_HEADS) return;

    uint n = nh / NUM_HEADS;
    uint h = nh % NUM_HEADS;

    // Each lane scans part of the tiles
    float best_v = INFINITY;
    int best_i = 0;
    uint base = nh * NUM_TILES;

    for (uint t = lane; t < NUM_TILES; t += threads_per_simdgroup) {
        float v = partial_min[base + t];
        int i = partial_idx[base + t];
        if (better_pair(v, i, best_v, best_i)) {
            best_v = v;
            best_i = i;
        }
    }

    // SIMD reduction
    for (uint off = 16; off > 0; off >>= 1) {
        float v2 = simd_shuffle_down(best_v, off);
        int i2 = simd_shuffle_down(best_i, off);
        if (better_pair(v2, i2, best_v, best_i)) {
            best_v = v2;
            best_i = i2;
        }
    }

    // Lane 0 emits results
    if (lane == 0) {
        int p = best_i;  // local ring slot 0..M-1
        indices_out[nh] = p;

        // Threshold: sqrt(dist) < sqrt(2*D) * (1 - threshold)
        // Squared: dist < 2*D * (1-threshold)^2
        float thresh = threshold_val[0];
        float one_minus = 1.0f - thresh;
        bool allow = (one_minus > 0.0f);
        float T_sq = 2.0f * float(HEAD_DIM) * one_minus * one_minus;
        bool is_hit = allow && (best_v < T_sq) && (best_v < INFINITY);

        // Map local ring slot → global index
        int L = request_length[n];
        int g = 0;
        if (L > 0) {
            if (L < (int)M_CAP) {
                g = p;
            } else {
                int base_g = L - (int)M_CAP;
                int tail = L % (int)M_CAP;
                int order = p - tail;
                if (order < 0) order += (int)M_CAP;
                g = base_g + order;
            }
        }

        // left_start = hit ? max(g + 1 - band_r, 0) : 0
        int left_val = 0;
        if (is_hit) {
            int tmp = g + 1 - (int)BAND_R;
            left_val = (tmp > 0) ? tmp : 0;
        }

        hit_out[nh] = is_hit ? 1 : 0;
        left_out[nh] = left_val;
    }
"""


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

# Cached kernel objects (mx.fast.metal_kernel is JIT-compiled and cached)
_tile_kernel = None
_reduce_kernel = None


def _get_tile_kernel():
    global _tile_kernel
    if _tile_kernel is None:
        _tile_kernel = mx.fast.metal_kernel(
            name="mac_match_tile",
            input_names=["q_cache", "request_length", "queries"],
            output_names=["partial_min", "partial_idx"],
            header=_TILE_HEADER,
            source=_TILE_SOURCE,
        )
    return _tile_kernel


def _get_reduce_kernel():
    global _reduce_kernel
    if _reduce_kernel is None:
        _reduce_kernel = mx.fast.metal_kernel(
            name="mac_match_reduce",
            input_names=["partial_min", "partial_idx", "request_length", "threshold_val"],
            output_names=["hit_out", "left_out", "indices_out"],
            header=_REDUCE_HEADER,
            source=_REDUCE_SOURCE,
        )
    return _reduce_kernel


def mac_ring_match(
    ring_cache: MACRingCache,
    queries: mx.array,
    req_ids: mx.array,
    threshold: float = 0.6,
    band_r: int = 256,
    rows_per_tile: int = 64,
) -> tuple[mx.array, mx.array, mx.array]:
    """L2 distance match: find nearest cached query for each (batch, head).

    Two-phase Metal kernel:
      Phase 1: Tile pass — parallel L2 over ring buffer rows
      Phase 2: Reduce — cross-tile argmin + threshold check

    Args:
        ring_cache:    MACRingCache instance with query_cache [R, M, H, D]
        queries:       [N, H, D] bf16 — current decode queries (pre-RoPE)
        req_ids:       [N] int32 — request indices (currently N == batch, req=n)
        threshold:     L2 match threshold tau (0..1, higher = stricter)
        band_r:        rectification band width (for left_start computation)
        rows_per_tile: rows per tile (fixed, simplifying CUDA adaptive logic)

    Returns:
        hit:   [N, H] bool  — whether a match was found
        left:  [N, H] int32 — attention start position (0 for misses)
        idx:   [N, H] int32 — matched ring buffer slot index
    """
    N, H, D = queries.shape
    M = ring_cache.capacity
    RPT = rows_per_tile
    num_tiles = (M + RPT - 1) // RPT
    BLOCK_THREADS = 256  # 8 SIMD groups of 32
    NUM_WARPS = BLOCK_THREADS // 32

    # --- Phase 1: Tile pass ---
    tile_kernel = _get_tile_kernel()

    # grid specifies total threads (not threadgroups) in mx.fast.metal_kernel
    # x: N * num_tiles threadgroups, each of BLOCK_THREADS threads
    # y: H threadgroups, each of 1 thread (head dimension)
    partial_results = tile_kernel(
        inputs=[ring_cache.query_cache, ring_cache.request_length, queries],
        output_shapes=[(N * H * num_tiles,), (N * H * num_tiles,)],
        output_dtypes=[mx.float32, mx.int32],
        grid=(N * num_tiles * BLOCK_THREADS, H, 1),
        threadgroup=(BLOCK_THREADS, 1, 1),
        template=[
            ("HEAD_DIM", D),
            ("M_CAP", M),
            ("NUM_HEADS", H),
            ("NUM_QUERIES", N),
            ("NUM_TILES", num_tiles),
            ("RPT", RPT),
            ("NUM_WARPS", NUM_WARPS),
            ("BLOCK_THREADS", BLOCK_THREADS),
        ],
        stream=mx.gpu,
    )
    partial_min, partial_idx = partial_results

    # --- Phase 2: Reduce ---
    reduce_kernel = _get_reduce_kernel()

    # threshold is a float — pass as scalar input (templates only support int/bool)
    threshold_arr = mx.array([threshold], dtype=mx.float32)

    reduce_results = reduce_kernel(
        inputs=[partial_min, partial_idx, ring_cache.request_length, threshold_arr],
        output_shapes=[(N * H,), (N * H,), (N * H,)],
        output_dtypes=[mx.int32, mx.int32, mx.int32],
        grid=(N * H * 32, 1, 1),
        threadgroup=(32, 1, 1),
        template=[
            ("HEAD_DIM", D),
            ("M_CAP", M),
            ("NUM_HEADS", H),
            ("NUM_QUERIES", N),
            ("NUM_TILES", num_tiles),
            ("BAND_R", band_r),
        ],
        stream=mx.gpu,
    )
    hit_flat, left_flat, idx_flat = reduce_results

    # Reshape to [N, H]
    hit = hit_flat.reshape(N, H).astype(mx.bool_)
    left = left_flat.reshape(N, H)
    idx = idx_flat.reshape(N, H)

    return hit, left, idx


def mac_ring_match_reference(
    ring_cache: MACRingCache,
    queries: mx.array,
    req_ids: mx.array,
    threshold: float = 0.6,
    band_r: int = 256,
) -> tuple[mx.array, mx.array, mx.array]:
    """Pure MLX reference implementation for testing (no Metal kernel).

    Computes exactly the same result as mac_ring_match but using
    standard MLX ops. Used for correctness verification.
    """
    N, H, D = queries.shape
    M = ring_cache.capacity

    hit_list = []
    left_list = []
    idx_list = []

    for n in range(N):
        req = n  # implicit mapping
        L = ring_cache.request_length[req].item()
        V = min(L, M)

        for h in range(H):
            if V <= 0:
                hit_list.append(False)
                left_list.append(0)
                idx_list.append(0)
                continue

            # Query vector: [D]
            q = queries[n, h].astype(mx.float32)  # [D]

            # Cache vectors: [V, D]
            cache_vecs = ring_cache.query_cache[req, :V, h].astype(mx.float32)  # [V, D]

            # L2 distances
            diffs = q[None, :] - cache_vecs  # [V, D]
            dists = mx.sum(diffs * diffs, axis=-1)  # [V]

            # Argmin with tie-breaking (lower index wins)
            best_idx = mx.argmin(dists).item()
            best_val = dists[best_idx].item()

            # Threshold check
            one_minus = 1.0 - threshold
            T_sq = 2.0 * D * one_minus * one_minus
            is_hit = (one_minus > 0) and (best_val < T_sq) and (best_val < float("inf"))

            # Ring slot → global index
            p = best_idx
            g = 0
            if L > 0:
                if L < M:
                    g = p
                else:
                    base_g = L - M
                    tail = L % M
                    order = p - tail
                    if order < 0:
                        order += M
                    g = base_g + order

            # left_start
            left_val = 0
            if is_hit:
                tmp = g + 1 - band_r
                left_val = max(tmp, 0)

            hit_list.append(is_hit)
            left_list.append(left_val)
            idx_list.append(p)

    hit = mx.array(hit_list, dtype=mx.bool_).reshape(N, H)
    left = mx.array(left_list, dtype=mx.int32).reshape(N, H)
    idx = mx.array(idx_list, dtype=mx.int32).reshape(N, H)

    return hit, left, idx
