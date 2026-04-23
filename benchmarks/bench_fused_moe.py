#!/usr/bin/env python3
"""
Benchmark: Fused MoE Gate+Up+SwiGLU kernel vs gather_qmm baseline.

Tests correctness and performance on Qwen3.5-35B-A3B dimensions:
  - 256 experts, K=8 active per token
  - gate/up: [256, 1024, 4096] (quantized 4-bit, group_size=64)
  - input: [4096]
"""

import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "mlx-lm-source"))

import mlx.core as mx
import numpy as np


def create_test_data(
    num_experts: int = 256,
    in_dim: int = 4096,
    out_dim: int = 1024,
    K: int = 8,
    group_size: int = 64,
    bits: int = 4,
):
    """Create synthetic quantized weight data matching Qwen3.5-35B-A3B."""
    # Random input
    x = mx.random.normal(shape=(in_dim,)).astype(mx.float32)

    # Random expert selection (K out of num_experts)
    expert_ids = mx.array(
        np.random.choice(num_experts, size=K, replace=False).astype(np.uint32)
    )

    # Create random weights and quantize them
    scale = (1.0 / in_dim) ** 0.5
    gate_full = mx.random.uniform(
        low=-scale, high=scale,
        shape=(num_experts, out_dim, in_dim),
    )
    up_full = mx.random.uniform(
        low=-scale, high=scale,
        shape=(num_experts, out_dim, in_dim),
    )

    gate_W, gate_scales, gate_biases = mx.quantize(
        gate_full, group_size=group_size, bits=bits
    )
    up_W, up_scales, up_biases = mx.quantize(
        up_full, group_size=group_size, bits=bits
    )

    mx.eval(x, expert_ids, gate_W, gate_scales, gate_biases, up_W, up_scales, up_biases)

    return {
        "x": x,
        "expert_ids": expert_ids,
        "gate_W": gate_W,
        "gate_scales": gate_scales,
        "gate_biases": gate_biases,
        "up_W": up_W,
        "up_scales": up_scales,
        "up_biases": up_biases,
        "group_size": group_size,
        "bits": bits,
    }


def run_baseline(data, warmup=3, iters=20):
    """Baseline: 2x gather_qmm + SwiGLU (current SwitchGLU path)."""
    x = data["x"].reshape(1, 1, -1)  # [1, 1, in_dim]
    expert_ids = data["expert_ids"].reshape(1, -1)  # [1, K]
    K = expert_ids.shape[1]

    gate_W = data["gate_W"]
    gate_scales = data["gate_scales"]
    gate_biases = data["gate_biases"]
    up_W = data["up_W"]
    up_scales = data["up_scales"]
    up_biases = data["up_biases"]
    gs = data["group_size"]
    bits = data["bits"]

    def step():
        x_exp = mx.expand_dims(x, (-2, -3))  # [1, 1, 1, 1, in_dim]
        gate_out = mx.gather_qmm(
            x_exp, gate_W, gate_scales, gate_biases,
            rhs_indices=expert_ids, transpose=True,
            group_size=gs, bits=bits,
        )
        up_out = mx.gather_qmm(
            x_exp, up_W, up_scales, up_biases,
            rhs_indices=expert_ids, transpose=True,
            group_size=gs, bits=bits,
        )
        silu_gate = gate_out * mx.sigmoid(gate_out)
        result = silu_gate * up_out
        return result

    # Warmup
    for _ in range(warmup):
        r = step()
        mx.eval(r)

    # Benchmark
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = step()
        mx.eval(r)
        times.append((time.perf_counter() - t0) * 1000)

    return r, times


def run_fused(data, warmup=3, iters=20):
    """Fused kernel: 1 dispatch for gate+up+SwiGLU."""
    from flashmlx.kernels.fused_moe import fused_gate_up_swiglu

    x = data["x"]
    expert_ids = data["expert_ids"]
    gs = data["group_size"]

    def step():
        return fused_gate_up_swiglu(
            x, expert_ids,
            data["gate_W"], data["gate_scales"], data["gate_biases"],
            data["up_W"], data["up_scales"], data["up_biases"],
            group_size=gs,
        )

    # Warmup (includes kernel compilation)
    for _ in range(warmup):
        r = step()
        mx.eval(r)

    # Benchmark
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = step()
        mx.eval(r)
        times.append((time.perf_counter() - t0) * 1000)

    return r, times


def run_reference(data):
    """Reference: per-expert dequantize + matmul (for correctness only)."""
    from flashmlx.kernels.fused_moe import reference_gate_up_swiglu

    result = reference_gate_up_swiglu(
        data["x"], data["expert_ids"],
        data["gate_W"], data["gate_scales"], data["gate_biases"],
        data["up_W"], data["up_scales"], data["up_biases"],
        group_size=data["group_size"],
        bits=data["bits"],
    )
    mx.eval(result)
    return result


def check_correctness(ref, fused, baseline, atol=0.5, rtol=0.05):
    """Check fused kernel output matches reference."""
    ref_np = np.array(ref.astype(mx.float32))
    fused_np = np.array(fused.astype(mx.float32))

    # Reshape baseline to match: [1, 1, K, 1, out_dim] → [K, out_dim]
    baseline_np = np.array(baseline.astype(mx.float32)).reshape(ref_np.shape)

    max_diff_fused = np.max(np.abs(ref_np - fused_np))
    max_diff_baseline = np.max(np.abs(ref_np - baseline_np))
    mean_diff_fused = np.mean(np.abs(ref_np - fused_np))
    mean_diff_baseline = np.mean(np.abs(ref_np - baseline_np))

    rel_err_fused = np.mean(np.abs(ref_np - fused_np) / (np.abs(ref_np) + 1e-8))
    rel_err_baseline = np.mean(np.abs(ref_np - baseline_np) / (np.abs(ref_np) + 1e-8))

    print(f"\n  Correctness (vs reference dequantize+matmul):")
    print(f"    Fused kernel:  max_diff={max_diff_fused:.6f}  mean_diff={mean_diff_fused:.6f}  rel_err={rel_err_fused:.6f}")
    print(f"    gather_qmm:    max_diff={max_diff_baseline:.6f}  mean_diff={mean_diff_baseline:.6f}  rel_err={rel_err_baseline:.6f}")

    # Fused should be close to the reference (both do same dequant)
    ok = max_diff_fused < atol
    print(f"    Status: {'PASS' if ok else 'FAIL'} (atol={atol})")
    return ok


def main():
    print("=" * 64)
    print(" Fused MoE Gate+Up+SwiGLU Benchmark")
    print("=" * 64)
    print()
    print("  Config: 256 experts, K=8, in=4096, out=1024, 4-bit q64")
    print(f"  Metal memory: {mx.get_active_memory() / 1e6:.0f} MB")
    print()

    # Create test data
    print("Creating test data...")
    data = create_test_data()
    print(f"  gate_W shape: {data['gate_W'].shape}")
    print(f"  gate_scales shape: {data['gate_scales'].shape}")
    print(f"  x shape: {data['x'].shape}")
    print(f"  expert_ids: {data['expert_ids'].tolist()}")
    print()

    # Reference (correctness ground truth)
    print("Running reference (dequantize + matmul)...")
    ref_result = run_reference(data)
    print(f"  Output shape: {ref_result.shape}")
    print()

    # Baseline (gather_qmm)
    print("Running baseline (2x gather_qmm + SwiGLU)...")
    baseline_result, baseline_times = run_baseline(data)
    baseline_ms = np.median(baseline_times)
    print(f"  Median: {baseline_ms:.3f} ms")
    print(f"  Min/Max: {min(baseline_times):.3f} / {max(baseline_times):.3f} ms")
    print()

    # Fused kernel
    print("Running fused kernel (1 dispatch)...")
    try:
        fused_result, fused_times = run_fused(data)
        fused_ms = np.median(fused_times)
        print(f"  Median: {fused_ms:.3f} ms")
        print(f"  Min/Max: {min(fused_times):.3f} / {max(fused_times):.3f} ms")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # Correctness
    check_correctness(ref_result, fused_result, baseline_result)

    # Summary
    print()
    print("  Summary:")
    print(f"    Baseline (gather_qmm): {baseline_ms:.3f} ms")
    print(f"    Fused kernel:          {fused_ms:.3f} ms")
    if fused_ms > 0:
        speedup = baseline_ms / fused_ms
        pct = (1 - fused_ms / baseline_ms) * 100
        print(f"    Speedup:               {speedup:.2f}x ({pct:+.1f}%)")
    print()


if __name__ == "__main__":
    main()
