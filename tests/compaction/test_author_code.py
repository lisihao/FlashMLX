#!/usr/bin/env python3
"""
端到端测试作者的 Attention Matching 实现
使用相同的测试数据，对比质量
"""
import sys
import torch
import numpy as np

# 添加作者代码路径
sys.path.insert(0, 'src/flashmlx/compaction/reference/compaction')

from compaction.algorithms.highest_attention_keys import HighestAttentionKeysCompaction as AuthorCompaction

def test_author_implementation(T, d, n, t):
    """测试作者的实现"""
    print(f"\n{'='*60}")
    print(f"作者实现测试: T={T}, t={t}, d={d}, n={n}")
    print(f"{'='*60}")

    # 创建测试数据（使用 PyTorch）
    torch.manual_seed(42)
    K = torch.randn(T, d, dtype=torch.float32)
    V = torch.randn(T, d, dtype=torch.float32)
    queries = torch.randn(n, d, dtype=torch.float32)

    # 创建作者的压缩算法实例
    # 参数参考: compaction/algorithms/highest_attention_keys.py
    compactor = AuthorCompaction(
        beta_method='nnls',  # 使用 NNLS 方法
        nnls_iters=1000,  # 迭代次数
        score_method='mean',  # 选择策略
        # 其他默认参数
    )

    # 运行压缩
    # 参考: compute_compacted_cache(self, K, V, queries, t)
    C1, beta, C2, selected_indices = compactor.compute_compacted_cache(
        K, V, queries, t
    )

    print(f"\n作者实现结果:")
    print(f"  C1 shape: {C1.shape}")
    print(f"  beta shape: {beta.shape}")
    print(f"  C2 shape: {C2.shape}")
    print(f"  selected_indices: {selected_indices}")

    # Beta 统计
    beta_np = beta.cpu().numpy()
    print(f"\nBeta 统计:")
    print(f"  均值: {np.mean(beta_np):8.4f}")
    print(f"  标准差: {np.std(beta_np):8.4f}")
    print(f"  范围: [{np.min(beta_np):8.4f}, {np.max(beta_np):8.4f}]")
    print(f"  被 clamp 数量: {np.sum(beta_np < -20)}/{t}")

    # 计算质量
    # 注意：作者的 compact_kv_cache 只返回 C1, beta, indices
    # 需要手动计算 C2
    scale = 1.0 / (d ** 0.5)

    # 原始输出
    with torch.no_grad():
        attn_scores_orig = queries @ K.T * scale
        attn_weights_orig = torch.softmax(attn_scores_orig, dim=-1)
        out_orig = attn_weights_orig @ V

        # 压缩 attention (使用作者返回的 C2)
        attn_scores_comp = queries @ C1.T * scale + beta
        attn_weights_comp = torch.softmax(attn_scores_comp, dim=-1)
        out_comp = attn_weights_comp @ C2

        # Cosine similarity
        out_orig_flat = out_orig.flatten()
        out_comp_flat = out_comp.flatten()
        cos_sim = (out_orig_flat @ out_comp_flat) / (
            torch.linalg.norm(out_orig_flat) * torch.linalg.norm(out_comp_flat)
        )
        mse = torch.mean((out_orig - out_comp) ** 2)

    print(f"\n质量指标:")
    print(f"  MSE:              {mse.item():.6f}")
    print(f"  Cosine similarity: {cos_sim.item():.6f}")
    print(f"  状态: {'✅ PASS' if cos_sim.item() >= 0.99 else '❌ FAIL'}")

    return {
        'T': T, 't': t, 'd': d, 'n': n,
        'cos_sim': cos_sim.item(),
        'mse': mse.item(),
        'beta_mean': np.mean(beta_np),
        'beta_std': np.std(beta_np),
        'beta_clipped': int(np.sum(beta_np < -20)),
    }

# 运行测试
print("=" * 60)
print("端到端测试 - 作者的 Attention Matching 实现")
print("=" * 60)

results = []

# Test 1: 小规模（与我们的测试一致）
print("\n[Test 1] 小规模")
r1 = test_author_implementation(T=20, d=16, n=3, t=5)
results.append(r1)

# Test 2: 增加 queries
print("\n[Test 2] 增加 queries")
r2 = test_author_implementation(T=20, d=16, n=50, t=5)
results.append(r2)

# Test 3: 中等规模
print("\n[Test 3] 中等规模")
r3 = test_author_implementation(T=100, d=64, n=50, t=20)
results.append(r3)

# 总结
print("\n" + "=" * 60)
print("作者实现 vs. 我们的实现 对比")
print("=" * 60)

print("\n作者实现质量:")
print(f"{'配置':<30} {'Cosine':<12} {'状态'}")
print("-" * 60)
for r in results:
    config = f"T={r['T']}, t={r['t']}, n={r['n']}"
    status = '✅ PASS' if r['cos_sim'] >= 0.99 else '❌ FAIL'
    print(f"{config:<30} {r['cos_sim']:<12.6f} {status}")

print("\n对比我们的实现 (T=20, t=5, n=3):")
print(f"  作者实现: cos={results[0]['cos_sim']:.6f}")
print(f"  我们实现: cos=0.640")

if results[0]['cos_sim'] >= 0.99:
    print("\n结论:")
    print("  ✅ 作者实现质量达标")
    print("  ❌ 我们的实现有差异，需要深入对比")
elif results[0]['cos_sim'] < 0.80:
    print("\n结论:")
    print("  ❌ 作者实现质量也不达标")
    print("  → NNLS 方法可能本身就不适用于小规模测试")
    print("  → 建议使用 log-ratio 方法（已验证 cos=1.000）")
else:
    print("\n结论:")
    print("  ⚠️ 作者实现质量一般")
    print("  → 需要进一步调查")
