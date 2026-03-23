#!/usr/bin/env python3
"""
大规模测试 NNLS beta 方法
验证方法是否依赖数据规模
"""
import mlx.core as mx
import sys
import time
sys.path.insert(0, 'src')

from flashmlx.cache.compaction_algorithm import HighestAttentionKeysCompaction

def test_scale(T, d, n, t, nnls_iters=1000):
    """测试指定规模的数据"""
    print(f"\n{'='*60}")
    print(f"规模测试: T={T}, t={t}, d={d}, n={n}")
    print(f"{'='*60}")

    # 创建测试数据
    mx.random.seed(42)
    K = mx.random.normal((T, d))
    V = mx.random.normal((T, d))
    queries = mx.random.normal((n, d))

    # 创建压缩算法实例
    compactor = HighestAttentionKeysCompaction(
        beta_method='nnls',
        c2_method='lsq',
        c2_ridge_lambda=0.0,
        score_method='mean',
        nnls_iters=nnls_iters
    )

    # 运行压缩
    start = time.time()
    C1, beta, C2, indices = compactor.compute_compacted_cache(
        K, V, queries, t
    )
    elapsed = time.time() - start

    # 验证质量
    scale = 1.0 / (d ** 0.5)

    # 原始输出
    attn_orig = mx.softmax(queries @ K.T * scale, axis=-1)
    out_orig = attn_orig @ V

    # 压缩输出
    attn_comp = mx.softmax(queries @ C1.T * scale + beta, axis=-1)
    out_comp = attn_comp @ C2

    # Cosine similarity
    def cosine_sim(a, b):
        a_flat = mx.reshape(a, (-1,))
        b_flat = mx.reshape(b, (-1,))
        dot = mx.sum(a_flat * b_flat)
        norm_a = mx.sqrt(mx.sum(a_flat * a_flat))
        norm_b = mx.sqrt(mx.sum(b_flat * b_flat))
        return dot / (norm_a * norm_b)

    cos_sim = float(cosine_sim(out_orig, out_comp))
    mse = float(mx.mean((out_orig - out_comp) ** 2))

    # Beta 统计
    beta_mean = float(mx.mean(beta))
    beta_std = float(mx.std(beta))
    beta_min = float(mx.min(beta))
    beta_max = float(mx.max(beta))
    beta_clipped = int(mx.sum(beta < -20))  # 计算被 clamp 到 log(1e-12) ≈ -27.6 的数量

    print(f"\nBeta 统计:")
    print(f"  均值: {beta_mean:8.4f}")
    print(f"  标准差: {beta_std:8.4f}")
    print(f"  范围: [{beta_min:8.4f}, {beta_max:8.4f}]")
    print(f"  被 clamp 数量: {beta_clipped}/{t}")

    print(f"\n质量指标:")
    print(f"  MSE:              {mse:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  状态: {'✅ PASS' if cos_sim >= 0.99 else '❌ FAIL'}")

    print(f"\n性能:")
    print(f"  压缩时间: {elapsed:.3f}s")
    print(f"  压缩比: {t}/{T} = {t/T:.1%}")

    return {
        'T': T, 't': t, 'd': d, 'n': n,
        'cos_sim': cos_sim,
        'mse': mse,
        'elapsed': elapsed,
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'beta_clipped': beta_clipped,
    }

# 测试不同规模
print("=" * 60)
print("大规模测试 - NNLS Beta 方法")
print("=" * 60)

results = []

# Test 1: 小规模（基线）
print("\n[Test 1] 小规模基线")
r1 = test_scale(T=20, d=16, n=3, t=5)
results.append(r1)

# Test 2: 增加 queries 数量
print("\n[Test 2] 增加 queries (n=3 → n=50)")
r2 = test_scale(T=20, d=16, n=50, t=5)
results.append(r2)

# Test 3: 增加 queries 数量更多
print("\n[Test 3] 增加 queries (n=3 → n=100)")
r3 = test_scale(T=20, d=16, n=100, t=5)
results.append(r3)

# Test 4: 增大规模（中等）
print("\n[Test 4] 中等规模 (T=100, t=20, n=50)")
r4 = test_scale(T=100, d=64, n=50, t=20, nnls_iters=500)
results.append(r4)

# Test 5: 大规模
print("\n[Test 5] 大规模 (T=1000, t=100, n=100)")
r5 = test_scale(T=1000, d=128, n=100, t=100, nnls_iters=500)
results.append(r5)

# 总结
print("\n" + "=" * 60)
print("总结")
print("=" * 60)

print("\n规模 vs. 质量:")
print(f"{'配置':<30} {'Cosine':<12} {'状态'}")
print("-" * 60)
for r in results:
    config = f"T={r['T']}, t={r['t']}, n={r['n']}"
    status = '✅ PASS' if r['cos_sim'] >= 0.99 else '❌ FAIL'
    print(f"{config:<30} {r['cos_sim']:<12.6f} {status}")

print("\n关键发现:")
if results[0]['cos_sim'] < 0.99 and results[-1]['cos_sim'] >= 0.99:
    print("  ✅ 增大规模后质量显著改善")
    print("  → NNLS 方法依赖大规模数据")
elif all(r['cos_sim'] < 0.99 for r in results):
    print("  ❌ 即使大规模数据，质量仍不达标")
    print("  → 实现可能存在问题，需要方案 A（端到端测试作者代码）")
elif results[0]['cos_sim'] >= 0.99:
    print("  ✅ 小规模数据已达标")
    print("  → 可能是之前的修复生效了")

# 分析 queries 数量的影响
if len(results) >= 3:
    print(f"\nQueries 数量影响 (T={results[0]['T']}, t={results[0]['t']}):")
    print(f"  n=3:   cos={results[0]['cos_sim']:.6f}")
    print(f"  n=50:  cos={results[1]['cos_sim']:.6f}")
    print(f"  n=100: cos={results[2]['cos_sim']:.6f}")

    if results[2]['cos_sim'] > results[0]['cos_sim'] + 0.1:
        print("  → 增加 queries 显著改善质量")
    else:
        print("  → Queries 数量影响不大")
