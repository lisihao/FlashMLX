#!/usr/bin/env python3
"""
详细对比作者实现和我们的实现
逐步追踪差异点
"""
import sys
import torch
import mlx.core as mx
import numpy as np

# 作者代码
sys.path.insert(0, 'src/flashmlx/compaction/reference/compaction')
from compaction.algorithms.highest_attention_keys import HighestAttentionKeysCompaction as AuthorCompaction

# 我们的代码
sys.path.insert(0, 'src')
from flashmlx.cache.compaction_algorithm import HighestAttentionKeysCompaction as OurCompaction

print("=" * 60)
print("详细对比：作者 vs. 我们的实现")
print("=" * 60)

# 参数
T, d, n, t = 20, 16, 3, 5
seed = 42

# ===================================================================
# Step 1: 创建相同的测试数据
# ===================================================================
print("\n[Step 1] 创建测试数据")

# PyTorch (作者)
torch.manual_seed(seed)
K_torch = torch.randn(T, d, dtype=torch.float32)
V_torch = torch.randn(T, d, dtype=torch.float32)
queries_torch = torch.randn(n, d, dtype=torch.float32)

# MLX (我们) - 转换自 PyTorch 保证数据一致
K_mlx = mx.array(K_torch.numpy())
V_mlx = mx.array(V_torch.numpy())
queries_mlx = mx.array(queries_torch.numpy())

print(f"  T={T}, d={d}, n={n}, t={t}")
print(f"  K 一致: {np.allclose(K_torch.numpy(), np.array(K_mlx))}")
print(f"  V 一致: {np.allclose(V_torch.numpy(), np.array(V_mlx))}")
print(f"  Q 一致: {np.allclose(queries_torch.numpy(), np.array(queries_mlx))}")

# ===================================================================
# Step 2: 计算 attention scores 和 weights
# ===================================================================
print("\n[Step 2] 计算 attention scores")

scale = 1.0 / (d ** 0.5)

# 作者
with torch.no_grad():
    attn_scores_torch = queries_torch @ K_torch.T * scale
    attn_weights_torch = torch.softmax(attn_scores_torch, dim=-1)

# 我们
attn_scores_mlx = queries_mlx @ K_mlx.T * scale
attn_weights_mlx = mx.softmax(attn_scores_mlx, axis=-1)

print(f"  attn_scores 一致: {np.allclose(attn_scores_torch.numpy(), np.array(attn_scores_mlx), rtol=1e-5)}")
print(f"  attn_weights 一致: {np.allclose(attn_weights_torch.numpy(), np.array(attn_weights_mlx), rtol=1e-5)}")

# ===================================================================
# Step 3: 计算 key scores (选择策略)
# ===================================================================
print("\n[Step 3] 计算 key scores")

# 作者默认使用 'max' 方法
key_scores_torch_max = torch.max(attn_weights_torch, dim=0).values
key_scores_torch_mean = torch.mean(attn_weights_torch, dim=0)

# 我们使用 'mean' 方法
key_scores_mlx_mean = mx.mean(attn_weights_mlx, axis=0)
key_scores_mlx_max = mx.max(attn_weights_mlx, axis=0)

print(f"  作者 (max):")
print(f"    值: {key_scores_torch_max.numpy()}")
print(f"  作者 (mean):")
print(f"    值: {key_scores_torch_mean.numpy()}")
print(f"  我们 (mean):")
print(f"    值: {np.array(key_scores_mlx_mean)}")
print(f"  我们 (max):")
print(f"    值: {np.array(key_scores_mlx_max)}")

# ===================================================================
# Step 4: 选择 top-t keys
# ===================================================================
print("\n[Step 4] 选择 top-t keys")

# 作者 (max)
_, indices_torch_max = torch.topk(key_scores_torch_max, t, largest=True)
indices_torch_max = indices_torch_max.cpu().tolist()

# 作者 (mean)
_, indices_torch_mean = torch.topk(key_scores_torch_mean, t, largest=True)
indices_torch_mean = indices_torch_mean.cpu().tolist()

# 我们 (mean)
indices_mlx_mean = mx.argsort(-key_scores_mlx_mean)[:t]
indices_mlx_mean = sorted([int(i) for i in indices_mlx_mean])

# 我们 (max)
indices_mlx_max = mx.argsort(-key_scores_mlx_max)[:t]
indices_mlx_max = sorted([int(i) for i in indices_mlx_max])

print(f"  作者 (max):  {indices_torch_max}")
print(f"  作者 (mean): {indices_torch_mean}")
print(f"  我们 (mean): {indices_mlx_mean}")
print(f"  我们 (max):  {indices_mlx_max}")

# ===================================================================
# Step 5: 运行完整压缩（使用相同的 score_method）
# ===================================================================
print("\n[Step 5] 运行完整压缩 (score_method='mean')")

# 作者实现
author_compactor = AuthorCompaction(
    beta_method='nnls',
    nnls_iters=1000,
    score_method='mean',  # 强制使用 mean
)
C1_author, beta_author, C2_author, indices_author = author_compactor.compute_compacted_cache(
    K_torch, V_torch, queries_torch, t
)

print(f"\n作者实现 (mean):")
print(f"  indices: {indices_author}")
print(f"  beta: {beta_author.numpy()}")
print(f"  beta 范围: [{beta_author.min().item():.4f}, {beta_author.max().item():.4f}]")

# 我们的实现
our_compactor = OurCompaction(
    beta_method='nnls',
    score_method='mean',
    nnls_iters=1000,
)
C1_ours, beta_ours, C2_ours, indices_ours = our_compactor.compute_compacted_cache(
    K_mlx, V_mlx, queries_mlx, t
)

print(f"\n我们的实现 (mean):")
print(f"  indices: {indices_ours}")
print(f"  beta: {np.array(beta_ours)}")
print(f"  beta 范围: [{float(mx.min(beta_ours)):.4f}, {float(mx.max(beta_ours)):.4f}]")

# ===================================================================
# Step 6: 计算质量
# ===================================================================
print("\n[Step 6] 质量对比")

# 作者质量
with torch.no_grad():
    out_orig_torch = attn_weights_torch @ V_torch
    attn_comp_torch = torch.softmax(queries_torch @ C1_author.T * scale + beta_author, dim=-1)
    out_comp_torch = attn_comp_torch @ C2_author

    out_orig_flat = out_orig_torch.flatten()
    out_comp_flat = out_comp_torch.flatten()
    cos_author = (out_orig_flat @ out_comp_flat) / (
        torch.linalg.norm(out_orig_flat) * torch.linalg.norm(out_comp_flat)
    )

# 我们的质量
out_orig_mlx = attn_weights_mlx @ V_mlx
attn_comp_mlx = mx.softmax(queries_mlx @ C1_ours.T * scale + beta_ours, axis=-1)
out_comp_mlx = attn_comp_mlx @ C2_ours

def cosine_sim(a, b):
    a_flat = mx.reshape(a, (-1,))
    b_flat = mx.reshape(b, (-1,))
    dot = mx.sum(a_flat * b_flat)
    norm_a = mx.sqrt(mx.sum(a_flat * a_flat))
    norm_b = mx.sqrt(mx.sum(b_flat * b_flat))
    return dot / (norm_a * norm_b)

cos_ours = float(cosine_sim(out_orig_mlx, out_comp_mlx))

print(f"  作者实现: cos={cos_author.item():.6f}")
print(f"  我们实现: cos={cos_ours:.6f}")

# ===================================================================
# Step 7: 检查 indices 是否一致
# ===================================================================
print("\n[Step 7] Indices 对比")

if set(indices_author) == set(indices_ours):
    print("  ✅ Indices 一致")
else:
    print("  ❌ Indices 不一致")
    print(f"    作者: {indices_author}")
    print(f"    我们: {indices_ours}")
    print(f"    差异: 作者有但我们没有 {set(indices_author) - set(indices_ours)}")
    print(f"    差异: 我们有但作者没有 {set(indices_ours) - set(indices_author)}")

# ===================================================================
# 结论
# ===================================================================
print("\n" + "=" * 60)
print("结论")
print("=" * 60)

if cos_author.item() >= 0.99 and cos_ours < 0.90:
    print("\n作者实现质量高，我们的低：")
    if set(indices_author) != set(indices_ours):
        print("  → 问题可能在 key 选择策略")
    else:
        print("  → 问题可能在 beta 计算或 C2 计算")
