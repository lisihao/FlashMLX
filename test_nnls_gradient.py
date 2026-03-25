#!/usr/bin/env python3
"""
调试 NNLS 梯度下降过程
"""
import mlx.core as mx

# 创建测试数据（与 test_compression_debug.py 相同）
mx.random.seed(42)
T, d, n, t = 20, 16, 3, 5
K = mx.random.normal((T, d))
queries = mx.random.normal((n, d))

scale = 1.0 / (d ** 0.5)
attn_scores = queries @ K.T * scale
attn_weights = mx.softmax(attn_scores, axis=-1)

# 选择 top-t keys
key_scores = mx.mean(attn_weights, axis=0)
indices_mx = mx.argsort(-key_scores)[:t]
indices = sorted([int(i) for i in indices_mx])

# NNLS 问题
max_scores = mx.max(attn_scores, axis=1, keepdims=True)
exp_scores = mx.exp(attn_scores - max_scores)
target = mx.sum(exp_scores, axis=1)
M = exp_scores[:, indices]

print("=== NNLS 梯度下降调试 ===")
print(f"M shape: {M.shape}")
print(f"target: {target}")
print()

# 初始化
min_val = 1e-12
B_init_val = float(mx.mean(target) / (mx.mean(M) + 1e-12))
B = mx.ones((t,), dtype=M.dtype) * max(B_init_val, min_val)

print(f"B_init: {float(B[0]):.6f}")
print()

# 计算 Lipschitz 常数
u = mx.random.normal((t,), dtype=M.dtype)
u = u / (mx.linalg.norm(u) + 1e-12)

for _ in range(3):
    v = M @ u
    if mx.linalg.norm(v) == 0:
        break
    u_new = M.T @ v
    u_new = u_new / (mx.linalg.norm(u_new) + 1e-12)
    u = u_new

Mu = M @ u
L = float((mx.linalg.norm(Mu)) ** 2)
step_size = 1.0 / L

print(f"Lipschitz constant L: {L:.6f}")
print(f"Step size: {step_size:.6e}")
print()

# 手动梯度下降
print("=== 手动梯度下降 ===")
for it in range(20):
    # 计算梯度
    residual = M @ B - target
    grad = M.T @ residual

    # 梯度下降步
    B_new = B - step_size * grad

    # 投影到约束集
    B_new = mx.maximum(B_new, min_val)

    # 计算 loss
    residual_new = M @ B_new - target
    loss = float(mx.sum(residual_new ** 2))

    if it < 10 or it == 19:
        print(f"Iter {it:2d}: loss={loss:12.6e}, B range=[{float(mx.min(B_new)):.6f}, {float(mx.max(B_new)):.6f}], grad_norm={float(mx.linalg.norm(grad)):.6e}")

    B = B_new

print()
print(f"最终 B: {B}")
print(f"M @ B: {M @ B}")
print(f"target: {target}")
print(f"最终 loss: {float(mx.sum((M @ B - target) ** 2)):.6e}")
print()

# 测试：如果从更好的初始化开始呢？
print("=== 测试：从 B = ones 开始 ===")
B2 = mx.ones((t,), dtype=M.dtype)

for it in range(20):
    residual = M @ B2 - target
    grad = M.T @ residual
    B2_new = B2 - step_size * grad
    B2_new = mx.maximum(B2_new, min_val)

    residual_new = M @ B2_new - target
    loss = float(mx.sum(residual_new ** 2))

    if it < 10 or it == 19:
        print(f"Iter {it:2d}: loss={loss:12.6e}, B range=[{float(mx.min(B2_new)):.6f}, {float(mx.max(B2_new)):.6f}]")

    B2 = B2_new

print()
print(f"最终 B2: {B2}")
print(f"M @ B2: {M @ B2}")
print(f"target: {target}")
print(f"最终 loss: {float(mx.sum((M @ B2 - target) ** 2)):.6e}")
