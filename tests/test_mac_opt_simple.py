#!/usr/bin/env python3
"""
MAC 优化简化测试 - 直接测试核心kernel

不依赖复杂的ring cache初始化，直接测试：
1. Match kernel: 向量化L2距离
2. Merge kernel: 融合操作
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx


def sync():
    mx.eval(mx.array(0))


def measure_ms(fn, warmup=10, iters=50):
    """精确测量"""
    for _ in range(warmup):
        fn()
        sync()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        result = fn()
        if isinstance(result, (tuple, list)):
            mx.eval(*result)
        elif result is not None:
            mx.eval(result)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


print("=" * 80)
print("MAC 优化验证 - 简化测试")
print("=" * 80)
print()

# ============================================================================
# Test 1: Merge Kernel Fusion
# ============================================================================

print("Test 1: Merge Kernel 融合优化")
print("-" * 80)

N, H, D = 1, 32, 128

mx.random.seed(42)
o_cached = mx.random.normal((N, H, D))
lse_cached = mx.random.normal((N, H))
o_fresh = mx.random.normal((N, H, D))
lse_fresh = mx.random.normal((N, H))
mx.eval(o_cached, lse_cached, o_fresh, lse_fresh)

# Original: 5 separate ops
from flashmlx.mac.attention import merge_attention_states


def run_original():
    return merge_attention_states(o_cached, lse_cached, o_fresh, lse_fresh)


print("  [1/3] 测试原始实现...", flush=True)
t_orig = measure_ms(run_original)
print(f"        原始 Merge (5 ops): {t_orig:.3f} ms")

# Test optimized version
print("  [2/3] 测试优化实现...", flush=True)
try:
    from flashmlx.mac.merge_fused import merge_attention_states_fused

    def run_fused():
        return merge_attention_states_fused(o_cached, lse_cached, o_fresh, lse_fresh)

    t_fused = measure_ms(run_fused)
    print(f"        优化 Merge (fused): {t_fused:.3f} ms")
    print(f"        加速比: {t_orig / t_fused:.2f}×")

    # Correctness check
    print("  [3/3] 验证正确性...", flush=True)
    o_o, lse_o = run_original()
    o_f, lse_f = run_fused()
    mx.eval(o_o, lse_o, o_f, lse_f)

    o_diff = mx.abs(o_o - o_f).max().item()
    lse_diff = mx.abs(lse_o - lse_f).max().item()

    print(f"        Max diff - o: {o_diff:.2e}, lse: {lse_diff:.2e}")

    if o_diff < 1e-4 and lse_diff < 1e-4:
        print("        ✅ 输出数值一致")
        merge_success = True
        merge_speedup = t_orig / t_fused
    else:
        print("        ❌ 输出差异过大")
        merge_success = False
        merge_speedup = 0

except Exception as e:
    print(f"        ❌ 优化版本失败: {e}")
    import traceback

    traceback.print_exc()
    merge_success = False
    merge_speedup = 0

print()

# ============================================================================
# Test 2: Match Kernel (simplified - test L2 distance computation)
# ============================================================================

print("Test 2: Match Kernel L2距离计算优化")
print("-" * 80)

# Test the core L2 distance computation pattern
N, M, H, D = 1, 512, 32, 128

mx.random.seed(42)
queries = mx.random.normal((N, H, D)).astype(mx.bfloat16)
cache = mx.random.normal((N, M, H, D)).astype(mx.bfloat16)
mx.eval(queries, cache)

print("  [1/2] 测试当前L2距离实现...", flush=True)


# Original scalar implementation
def compute_l2_scalar():
    """Current implementation pattern"""
    q = queries[0]  # [H, D]
    c = cache[0]  # [M, H, D]
    dists = []
    for m in range(M):
        diff = q - c[m]  # [H, D]
        sq = diff * diff
        dist = mx.sum(sq, axis=-1)  # [H]
        dists.append(dist)
    result = mx.stack(dists)  # [M, H]
    return result


t_scalar = measure_ms(compute_l2_scalar, warmup=5, iters=20)
print(f"        Scalar L2: {t_scalar:.3f} ms")

print("  [2/2] 测试向量化L2距离...", flush=True)


# Vectorized implementation
def compute_l2_vectorized():
    """Optimized vectorized version"""
    q = queries[0]  # [H, D]
    c = cache[0]  # [M, H, D]
    # Broadcast: q[None, :, :] - c[:, :, :] → [M, H, D]
    diff = q[None, :, :] - c
    sq = diff * diff
    dist = mx.sum(sq, axis=-1)  # [M, H]
    return dist


t_vec = measure_ms(compute_l2_vectorized, warmup=5, iters=20)
print(f"        Vectorized L2: {t_vec:.3f} ms")
print(f"        加速比: {t_scalar / t_vec:.2f}×")

# Correctness
r_scalar = compute_l2_scalar()
r_vec = compute_l2_vectorized()
mx.eval(r_scalar, r_vec)
diff = mx.abs(r_scalar - r_vec).max().item()
print(f"        Max diff: {diff:.2e}")
if diff < 1e-4:
    print("        ✅ 结果一致")
    match_success = True
    match_speedup = t_scalar / t_vec
else:
    print("        ❌ 结果不一致")
    match_success = False
    match_speedup = 0

print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("测试总结")
print("=" * 80)
print()

if merge_success:
    print(f"✅ Merge 优化成功: {merge_speedup:.2f}× 加速")
    print(f"   原始: {t_orig:.3f} ms → 优化: {t_fused:.3f} ms")
else:
    print("❌ Merge 优化失败")

print()

if match_success:
    print(f"✅ Match 优化方向验证: {match_speedup:.2f}× 加速")
    print(f"   Scalar: {t_scalar:.3f} ms → Vectorized: {t_vec:.3f} ms")
    print("   注: 实际Metal kernel优化需进一步测试")
else:
    print("❌ Match 优化方向验证失败")

print()

if merge_success and match_success:
    print("🎉 优化验证通过！")
    print()
    print("预期端到端收益:")
    print(f"  - Merge优化: -{t_orig - t_fused:.3f} ms/step")
    print(
        f"  - Match优化(预估): ~{t_scalar / t_vec:.1f}× 加速（需Metal kernel验证）"
    )
    print()
    print("下一步:")
    print("  1. 集成优化版本到 MACDecodeWrapper")
    print("  2. 端到端测试 (bench_mac_practical.py)")
    print("  3. 验证 32K context 下的实际加速比")
else:
    print("⚠️  部分优化需要修复")

print()
print("=" * 80)
