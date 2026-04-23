"""
Fused MoE kernels for Apple Silicon via mx.fast.metal_kernel().

Kernel 1: fused_gate_up_swiglu
  - Fuses gate_proj + up_proj + SwiGLU for K experts in ONE dispatch
  - Input x cached in threadgroup shared memory (read once from global)
  - 8 output rows per threadgroup (1 per SIMD group of 32 threads)
  - FMA-optimized 4-bit affine dequantization
  - Inspired by flash-moe/shaders.metal (Dan Woods)

Why this matters:
  - Batch=1 decode: each expert gets 1 token → matvec not matmul
  - gather_qmm handles groups internally but median group_size=1 → GPU underutilized
  - This kernel shares x across all K experts (saves K-1 global reads)
  - gate + up computed simultaneously from shared x (saves another read)
"""

import mlx.core as mx


# ============================================================================
# Metal Kernel: Fused Gate + Up + SwiGLU (4-bit quantized, K experts)
# ============================================================================

_FUSED_GATE_UP_SWIGLU_HEADER = """
// Helper: extract 4-bit nibble from packed uint32
inline float extract_nibble(uint32_t packed, uint n) {
    return float((packed >> (n * 4)) & 0xF);
}
"""

_FUSED_GATE_UP_SWIGLU_SOURCE = """
    // Template constants: IN_DIM, OUT_DIM, GROUP_SIZE
    // Auto-detected attributes: threadgroup_position_in_grid, etc.

    const uint PACKED_PER_GROUP = GROUP_SIZE / 8;
    const uint PACKED_COLS = IN_DIM / 8;
    const uint NUM_GROUPS = IN_DIM / GROUP_SIZE;
    const uint ROWS_PER_TG = 8;

    uint tgid_x = threadgroup_position_in_grid.x;   // row tile index
    uint tgid_y = threadgroup_position_in_grid.y;   // expert index in active set
    uint lid = thread_position_in_threadgroup.x;     // 0..255
    uint simd_group = simdgroup_index_in_threadgroup; // 0..7
    uint simd_lane = thread_index_in_simdgroup;       // 0..31

    // ---- Step 1: Cache x in threadgroup shared memory ----
    threadgroup float x_shared[IN_DIM];
    for (uint i = lid; i < IN_DIM; i += 256) {
        x_shared[i] = float(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Step 2: Which expert, which output row ----
    uint expert_id = uint(expert_ids[tgid_y]);
    uint row = tgid_x * ROWS_PER_TG + simd_group;
    if (row >= OUT_DIM) return;

    // Weight layout: [num_experts, OUT_DIM, PACKED_COLS]
    // Scale layout:  [num_experts, OUT_DIM, NUM_GROUPS]
    uint w_base = (expert_id * OUT_DIM + row) * PACKED_COLS;
    uint s_base = (expert_id * OUT_DIM + row) * NUM_GROUPS;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    // ---- Step 3: FMA-optimized dequant matvec ----
    // Each SIMD lane processes NUM_GROUPS/32 groups
    for (uint g = simd_lane; g < NUM_GROUPS; g += 32) {
        float g_scale = float(gate_scales[s_base + g]);
        float g_bias  = float(gate_biases[s_base + g]);
        float u_scale = float(up_scales[s_base + g]);
        float u_bias  = float(up_biases[s_base + g]);

        uint bp = g * PACKED_PER_GROUP;
        uint bx = g * GROUP_SIZE;

        for (uint p = 0; p < PACKED_PER_GROUP; p++) {
            uint32_t gp = gate_W[w_base + bp + p];
            uint32_t up = up_W[w_base + bp + p];

            for (uint n = 0; n < 8; n++) {
                float xv = x_shared[bx + p * 8 + n];

                // FMA: val = nibble * scale * x + bias * x
                // = fma(nibble, scale*x, acc + bias*x)
                float g_sx = g_scale * xv;
                float g_bx = g_bias * xv;
                float u_sx = u_scale * xv;
                float u_bx = u_bias * xv;

                gate_acc = fma(extract_nibble(gp, n), g_sx, gate_acc + g_bx);
                up_acc   = fma(extract_nibble(up, n), u_sx, up_acc + u_bx);
            }
        }
    }

    // ---- Step 4: SIMD reduction ----
    float gate_sum = simd_sum(gate_acc);
    float up_sum   = simd_sum(up_acc);

    // ---- Step 5: SwiGLU + write ----
    if (simd_lane == 0) {
        float silu = gate_sum / (1.0f + exp(-gate_sum));
        out[tgid_y * OUT_DIM + row] = silu * up_sum;
    }
"""


# Compiled kernel cache (keyed by dimensions)
_kernel_cache = {}


def _get_kernel(in_dim: int, out_dim: int, group_size: int):
    """Get or compile a fused kernel for specific dimensions."""
    key = (in_dim, out_dim, group_size)
    if key not in _kernel_cache:
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"fused_moe_gate_up_swiglu_{in_dim}_{out_dim}_{group_size}",
            input_names=[
                "x", "expert_ids",
                "gate_W", "gate_scales", "gate_biases",
                "up_W", "up_scales", "up_biases",
            ],
            output_names=["out"],
            source=_FUSED_GATE_UP_SWIGLU_SOURCE,
            header=_FUSED_GATE_UP_SWIGLU_HEADER,
        )
    return _kernel_cache[key]


def fused_gate_up_swiglu(
    x: mx.array,
    expert_ids: mx.array,
    gate_W: mx.array,
    gate_scales: mx.array,
    gate_biases: mx.array,
    up_W: mx.array,
    up_scales: mx.array,
    up_biases: mx.array,
    group_size: int = 64,
) -> mx.array:
    """Fused gate+up+SwiGLU for K experts with shared input caching.

    Replaces 2 separate gather_qmm calls (gate_proj + up_proj) plus a
    SwiGLU activation with a single Metal kernel dispatch. Input x is
    read from global memory once and cached in threadgroup shared memory.

    Args:
        x: Input vector, shape [in_dim]. Will be cast to float32.
        expert_ids: Selected expert indices, shape [K] (int32 or uint32).
        gate_W: Gate proj quantized weights [num_experts, out_dim, in_dim/8].
        gate_scales: Gate proj scales [num_experts, out_dim, in_dim/group_size].
        gate_biases: Gate proj biases [num_experts, out_dim, in_dim/group_size].
        up_W: Up proj quantized weights (same layout as gate).
        up_scales, up_biases: Up proj scales/biases.
        group_size: Quantization group size (default 64).

    Returns:
        mx.array of shape [K, out_dim]: SwiGLU(gate(x), up(x)) per expert.
    """
    K = expert_ids.shape[0]
    out_dim = gate_W.shape[1]
    in_dim = gate_scales.shape[2] * group_size

    x_flat = x.reshape(-1).astype(mx.float32)
    eid = expert_ids.reshape(-1).astype(mx.uint32)

    kernel = _get_kernel(in_dim, out_dim, group_size)

    THREADS_PER_TG = 256
    ROWS_PER_TG = 8  # = THREADS_PER_TG / 32 (one row per SIMD group)
    tg_count_x = (out_dim + ROWS_PER_TG - 1) // ROWS_PER_TG

    # NOTE: mx.fast.metal_kernel uses dispatch_threads (total threads),
    # NOT dispatch_threadgroups. Grid = total threads, not threadgroup count.
    result = kernel(
        inputs=[
            x_flat, eid,
            gate_W, gate_scales, gate_biases,
            up_W, up_scales, up_biases,
        ],
        output_shapes=[(K * out_dim,)],
        output_dtypes=[mx.float32],
        grid=(tg_count_x * THREADS_PER_TG, K, 1),
        threadgroup=(THREADS_PER_TG, 1, 1),
        template=[
            ("IN_DIM", in_dim),
            ("OUT_DIM", out_dim),
            ("GROUP_SIZE", group_size),
        ],
    )

    return result[0].reshape(K, out_dim)


def reference_gate_up_swiglu(
    x: mx.array,
    expert_ids: mx.array,
    gate_W: mx.array,
    gate_scales: mx.array,
    gate_biases: mx.array,
    up_W: mx.array,
    up_scales: mx.array,
    up_biases: mx.array,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """Reference implementation using mx.dequantize for correctness checking."""
    K = expert_ids.shape[0]
    out_dim = gate_W.shape[1]
    x_flat = x.reshape(1, -1).astype(mx.float32)

    results = []
    for i in range(K):
        eid = expert_ids[i].item()
        gate_dq = mx.dequantize(
            gate_W[eid], gate_scales[eid], gate_biases[eid],
            group_size=group_size, bits=bits,
        ).astype(mx.float32)
        up_dq = mx.dequantize(
            up_W[eid], up_scales[eid], up_biases[eid],
            group_size=group_size, bits=bits,
        ).astype(mx.float32)

        gate_out = (x_flat @ gate_dq.T).squeeze(0)
        up_out = (x_flat @ up_dq.T).squeeze(0)

        silu_gate = gate_out * mx.sigmoid(gate_out)
        results.append(silu_gate * up_out)

    return mx.stack(results)
