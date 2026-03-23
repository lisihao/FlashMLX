#!/usr/bin/env python3
"""
详细追踪 NNLS 求解过程
对比作者和我们的实现
"""
import sys
import torch
import mlx.core as mx
import numpy as np

# 作者代码
sys.path.insert(0, 'src/flashmlx/compaction/reference/compaction')

# 我们的代码
sys.path.insert(0, 'src')
from flashmlx.compaction.nnls_author import nnls_pg_author

print("=" * 60)
print("NNLS 详细追踪")
print("=" * 60)

# 使用相同的测试数据
T, d, n, t = 20, 16, 3, 5
seed = 42

torch.manual_seed(seed)
K_torch = torch.randn(T, d, dtype=torch.float32)
queries_torch = torch.randn(n, d, dtype=torch.float32)

K_mlx = mx.array(K_torch.numpy())
queries_mlx = mx.array(queries_torch.numpy())

scale = 1.0 / (d ** 0.5)

# 计算 attention
attn_scores_torch = queries_torch @ K_torch.T * scale
attn_weights_torch = torch.softmax(attn_scores_torch, dim=-1)

attn_scores_mlx = queries_mlx @ K_mlx.T * scale

# 选择相同的 indices
indices = [9, 19, 0, 16, 4]

print(f"\n[Step 1] 构造 NNLS 问题")
print(f"  indices: {indices}")

# 作者的方法
with torch.no_grad():
    # 作者使用 fp32
    scores32_torch = attn_scores_torch
    max_scores_torch = torch.max(scores32_torch, dim=1, keepdim=True).values
    exp_scores_torch = torch.exp(scores32_torch - max_scores_torch)

    target_torch = exp_scores_torch.sum(dim=1)
    M_torch = exp_scores_torch[:, indices]

print(f"\n作者 (PyTorch):")
print(f"  target: {target_torch.numpy()}")
print(f"  M shape: {M_torch.shape}")
print(f"  M:\n{M_torch.numpy()}")

# 我们的方法
scores32_mlx = attn_scores_mlx
max_scores_mlx = mx.max(scores32_mlx, axis=1, keepdims=True)
exp_scores_mlx = mx.exp(scores32_mlx - max_scores_mlx)

target_mlx = mx.sum(exp_scores_mlx, axis=1)
M_mlx = exp_scores_mlx[:, indices]

print(f"\n我们 (MLX):")
print(f"  target: {np.array(target_mlx)}")
print(f"  M shape: {M_mlx.shape}")
print(f"  M:\n{np.array(M_mlx)}")

# 检查一致性
print(f"\n一致性检查:")
print(f"  target 一致: {np.allclose(target_torch.numpy(), np.array(target_mlx))}")
print(f"  M 一致: {np.allclose(M_torch.numpy(), np.array(M_mlx))}")

# ===================================================================
# Step 2: 调用 NNLS 求解器
# ===================================================================
print(f"\n[Step 2] NNLS 求解")

# 作者的 NNLS (从 base.py)
from compaction.algorithms.base import CompactionAlgorithm

# 调用作者的 NNLS (静态方法)
B_author = CompactionAlgorithm._nnls_pg(M_torch, target_torch, iters=1000, debug=True)

print(f"\n作者的 NNLS 结果:")
print(f"  B: {B_author.numpy()}")
print(f"  beta = log(B): {torch.log(B_author).numpy()}")

# 我们的 NNLS
B_ours = nnls_pg_author(M_mlx, target_mlx, iters=1000, debug=True)

print(f"\n我们的 NNLS 结果:")
print(f"  B: {np.array(B_ours)}")
print(f"  beta = log(B): {np.array(mx.log(B_ours))}")

# ===================================================================
# Step 3: 验证 NNLS 解
# ===================================================================
print(f"\n[Step 3] 验证 NNLS 解")

# 作者
with torch.no_grad():
    residual_author = M_torch @ B_author - target_torch
    loss_author = (residual_author ** 2).sum().item()

print(f"  作者: M @ B = {(M_torch @ B_author).numpy()}")
print(f"  作者: target = {target_torch.numpy()}")
print(f"  作者: loss = {loss_author:.6e}")

# 我们
residual_ours = M_mlx @ B_ours - target_mlx
loss_ours = float(mx.sum(residual_ours ** 2))

print(f"  我们: M @ B = {np.array(M_mlx @ B_ours)}")
print(f"  我们: target = {np.array(target_mlx)}")
print(f"  我们: loss = {loss_ours:.6e}")

# ===================================================================
# 结论
# ===================================================================
print(f"\n" + "=" * 60)
print("结论")
print("=" * 60)

print(f"\nB 值对比:")
print(f"  作者: {B_author.numpy()}")
print(f"  我们: {np.array(B_ours)}")

print(f"\nBeta 值对比:")
beta_author = torch.log(B_author).numpy()
beta_ours = np.array(mx.log(B_ours))
print(f"  作者: {beta_author}")
print(f"  我们: {beta_ours}")

print(f"\n差异分析:")
for i in range(t):
    diff_B = abs(B_author[i].item() - float(B_ours[i]))
    diff_beta = abs(beta_author[i] - beta_ours[i])
    if diff_B > 0.1 or diff_beta > 1.0:
        print(f"  ⚠️ B[{i}]: 作者={B_author[i].item():.6f}, 我们={float(B_ours[i]):.6f}, diff={diff_B:.6f}")
        print(f"     beta[{i}]: 作者={beta_author[i]:.4f}, 我们={beta_ours[i]:.4f}, diff={diff_beta:.4f}")

if np.allclose(B_author.numpy(), np.array(B_ours), rtol=1e-4):
    print("\n✅ NNLS 求解一致")
else:
    print("\n❌ NNLS 求解不一致 - 需要调查求解器差异")
