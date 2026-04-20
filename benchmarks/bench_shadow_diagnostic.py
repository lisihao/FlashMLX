#!/usr/bin/env python3
"""Diagnostic: compare pool vs shadow _switchglu output for SAME expert.

This script isolates the exact source of the 10pp shadow quality gap.
It loads the model with pool32 + 6-bit shadow, then for each decode token:
1. Computes pool output (via normal _pool_call)
2. For pool HITS, also computes shadow output for the same expert
3. Reports max absolute difference

If pool output != shadow output for the SAME expert, the bug is in weight
format or gather_qmm behavior. If they match, the bug is elsewhere.
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import gc
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.expert_offload import patch_model_for_offload

MODEL_PATH = "/Users/lisihao/models/Qwen3.5-35B-A3B-6bit"


def run_diagnostic():
    print("=" * 70)
    print("  Shadow Dispatch Diagnostic")
    print("  Comparing pool vs shadow output for same expert")
    print("=" * 70)

    # Load model
    model, tokenizer = load(MODEL_PATH)
    ctx = patch_model_for_offload(
        model, MODEL_PATH, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()

    # PP warmup
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    # Compact to 32
    ctx.compact(pool_size=32, disable_coverage_gate=True,
                auto_expand_cpu_cache=False)

    # Create 6-bit shadow (same precision as model)
    print("\n  Creating 6-bit shadow...")
    ctx.create_shadow(bits=6)

    # Navigate to layers
    inner = model
    for attr in ("model", "model", "language_model", "model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)

    # Inject diagnostic hook into first MoE layer
    target_layer = None
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy"):
                target_layer = sw
                break

    if target_layer is None:
        print("ERROR: No MoE layer found!")
        return

    print(f"\n  Target layer: {target_layer._layer_idx}")
    print(f"  Pool experts: {len(target_layer._pool_expert_ids)}")
    print(f"  Pool bits: {target_layer._pool_bits} (None = model default {target_layer.bits})")
    print(f"  Shadow bits: {target_layer._shadow_bits}")
    print(f"  Model bits: {target_layer.bits}")
    print(f"  Group size: {target_layer.group_size}")

    # Check weight shapes
    pool = target_layer._pool
    shadow = target_layer._shadow
    print(f"\n  Weight shapes:")
    for comp in sorted(pool.keys())[:3]:
        print(f"    Pool  {comp}: {pool[comp].shape}")
    for comp in sorted(shadow.keys())[:3]:
        print(f"    Shadow {comp}: {shadow[comp].shape}")

    # Direct comparison: pick a pool expert, compute through pool vs shadow
    pool_eid = target_layer._pool_expert_ids[0]  # first pool expert
    pool_slot = 0  # it's at slot 0 in pool
    print(f"\n  Testing expert {pool_eid} (pool slot {pool_slot})...")

    # Create a fake input
    D = pool[list(pool.keys())[0]].shape[-1]  # input dim from weight shape
    # Actually get the real input dim from the weight shape
    # gate_proj.weight shape: [N_experts, intermediate_size / pack_factor, input_size?]
    # Let me figure this out from the model
    gate_shape = pool["gate_proj.weight"].shape
    print(f"    gate_proj.weight shape (pool): {gate_shape}")
    print(f"    gate_proj.scales shape (pool): {pool['gate_proj.scales'].shape}")

    # The input dim for gather_qmm with transpose=True is the last dim of weight
    # For quantized: weight shape is [N, out_features, in_features / pack]
    # scales shape is [N, out_features, in_features / group_size]
    # With transpose=True, it does x @ W.T, so input_dim = in_features

    # Let's just use a random input at the right dimension
    # Input dim = scales.shape[-1] * group_size
    scales_shape = pool["gate_proj.scales"].shape
    input_dim = scales_shape[-1] * target_layer.group_size
    print(f"    Inferred input dim: {input_dim}")

    # Create test input [1, 1, 1, 1, input_dim]
    x_test = mx.random.normal((1, 1, 1, 1, input_dim))
    mx.eval(x_test)

    # Pool index for expert pool_eid at slot 0
    pool_idx = mx.array([[[pool_slot]]])  # [1, 1, 1]

    # Shadow index for expert pool_eid at its global ID
    shadow_idx = mx.array([[[pool_eid]]])  # [1, 1, 1]

    # Compute via pool
    bits_pool = target_layer._pool_bits or target_layer.bits
    y_pool = target_layer._switchglu(
        x_test, pool, pool_idx,
        sorted_indices=False, bits=bits_pool
    )
    mx.eval(y_pool)

    # Compute via shadow
    y_shadow = target_layer._switchglu(
        x_test, shadow, shadow_idx,
        sorted_indices=False, bits=target_layer._shadow_bits
    )
    mx.eval(y_shadow)

    # Compare
    diff = mx.abs(y_pool - y_shadow)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    y_pool_norm = float(mx.abs(y_pool).mean())

    print(f"\n  === COMPARISON (same expert {pool_eid}, same input) ===")
    print(f"    Max  |pool - shadow|: {max_diff:.8f}")
    print(f"    Mean |pool - shadow|: {mean_diff:.8f}")
    print(f"    Mean |pool output|:   {y_pool_norm:.8f}")
    print(f"    Relative error:       {mean_diff / (y_pool_norm + 1e-10):.2e}")

    if max_diff < 1e-6:
        print(f"\n  ✓ MATCH: Pool and shadow produce identical outputs!")
        print(f"    Bug is NOT in weight data or _switchglu computation.")
        print(f"    Must be in miss detection, index mapping, or shape broadcasting.")
    else:
        print(f"\n  ✗ MISMATCH: Pool and shadow differ for same expert!")
        print(f"    Bug is in weight format/layout or gather_qmm interpretation.")

    # Test multiple experts
    print(f"\n  Testing all pool experts...")
    max_diffs = []
    for i, eid in enumerate(target_layer._pool_expert_ids[:10]):
        p_idx = mx.array([[[i]]])
        s_idx = mx.array([[[eid]]])
        yp = target_layer._switchglu(x_test, pool, p_idx, sorted_indices=False, bits=bits_pool)
        ys = target_layer._switchglu(x_test, shadow, s_idx, sorted_indices=False, bits=target_layer._shadow_bits)
        mx.eval(yp, ys)
        d = float(mx.abs(yp - ys).max())
        max_diffs.append(d)
        status = "✓" if d < 1e-6 else "✗"
        print(f"    Expert {eid:3d} (slot {i:2d}): max_diff = {d:.2e} {status}")

    # Now test the ACTUAL shadow dispatch path
    print(f"\n\n  === Testing actual _pool_call dispatch ===")
    print(f"  Comparing: full pool (no miss) vs pool32+shadow")

    # Set up for comparison: run same input through identity path vs shadow path
    # Save current state
    target_layer._miss_policy = "shadow"

    # Create input that hits pool and misses pool
    # Pick experts: first 4 from pool (hits), 4 not in pool (misses)
    pool_set = set(target_layer._pool_expert_ids)
    non_pool = [e for e in range(target_layer.num_experts) if e not in pool_set]

    hit_experts = target_layer._pool_expert_ids[:4]
    miss_experts = non_pool[:4]
    test_indices = mx.array([[[*hit_experts, *miss_experts]]])  # [1, 1, 8]
    print(f"    Test indices: hits={hit_experts[:4]}, misses={miss_experts[:4]}")

    # Compute via pool32 + shadow dispatch
    x_input = mx.random.normal((1, 1, input_dim))
    mx.eval(x_input)

    # Method 1: through _pool_call (pool + shadow fallback)
    target_layer._miss_policy = "shadow"
    y_dispatch = target_layer._pool_call(x_input, test_indices)
    mx.eval(y_dispatch)

    # Method 2: through shadow directly for ALL experts (ground truth)
    x_e = mx.expand_dims(x_input, (-2, -3))
    y_truth = target_layer._switchglu(
        x_e, shadow, test_indices,
        sorted_indices=False, bits=target_layer._shadow_bits
    ).squeeze(-2)
    mx.eval(y_truth)

    # Compare per-expert
    print(f"\n    Per-expert comparison (dispatch vs shadow-only):")
    for i in range(8):
        d = float(mx.abs(y_dispatch[0, 0, i] - y_truth[0, 0, i]).max())
        expert_id = int(test_indices[0, 0, i])
        source = "POOL" if expert_id in pool_set else "SHADOW"
        status = "✓" if d < 1e-5 else "✗"
        print(f"      Expert {expert_id:3d} [{source:6s}]: max_diff = {d:.2e} {status}")

    # Overall
    overall_diff = float(mx.abs(y_dispatch - y_truth).max())
    print(f"\n    Overall max diff: {overall_diff:.2e}")
    if overall_diff < 1e-5:
        print(f"    ✓ Pool hits match shadow → weights are identical")
        print(f"    Bug must be in a different interaction (scores, token accumulation, etc.)")
    else:
        print(f"    ✗ MISMATCH in dispatch path!")
        print(f"    The pool-vs-shadow output differs even for same expert/input")


if __name__ == "__main__":
    run_diagnostic()
