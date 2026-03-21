# KV Cache Compaction - 核心算法实现细节

**基于**: https://github.com/adamzweiger/compaction (PyTorch参考实现)

---

## 1. NNLS Beta拟合算法

### 1.1 问题定义

```
目标: softmax(Q·C1^T + beta) ≈ softmax(Q·K^T)
转换为: 求解 min ||M·B - target||^2, s.t. lower_bound <= B <= upper_bound

其中:
- exp_scores = exp(Q·K^T / sqrt(d))  # (n_queries, T)
- target = exp_scores.sum(dim=1)  # (n_queries,) - 每个query的总attention mass
- M = exp_scores[:, selected_indices]  # (n_queries, t) - 选中keys的exp scores
- B >= 0 (非负约束)
- beta = log(B)  # 最终的bias
```

### 1.2 求解器实现

#### Option 1: Clamped Least Squares (iters=0, 默认)

```python
def _nnls_pg(M, y, iters=0, lower_bound=1e-12, upper_bound=None):
    """
    快速近似: 使用无约束最小二乘 + clamp

    Time: ~0.5ms per head
    Accuracy: 通常足够好
    """
    # Step 1: 无约束最小二乘
    B = torch.linalg.lstsq(M, y).solution  # (t,)

    # Step 2: 投影到约束集合
    B = B.clamp(min=lower_bound, max=upper_bound)

    # Step 3: 转换为beta
    beta = torch.log(B)  # (t,)

    return beta
```

**MLX移植**:
```python
import mlx.core as mx

def nnls_clamped(M, y, lower_bound=1e-12, upper_bound=None):
    """
    MLX版本的Clamped Least Squares
    """
    # Step 1: lstsq求解
    B = mx.linalg.lstsq(M, y[..., None])[0].squeeze(-1)  # (t,)

    # Step 2: clamp
    B = mx.clip(B, lower_bound, upper_bound if upper_bound else 1e10)

    # Step 3: log
    beta = mx.log(B)

    return beta
```

#### Option 2: Projected Gradient Descent (iters>0)

```python
def _nnls_pg(M, y, iters=100, lower_bound=1e-12, upper_bound=None):
    """
    精确求解: Projected Gradient Descent

    Time: ~2ms per head (100 iters)
    Accuracy: 更精确，但慢
    """
    n, t = M.shape

    # Step 1: 初始解（无约束lstsq）
    B = torch.linalg.lstsq(M, y).solution  # (t,)
    B = B.clamp(min=lower_bound, max=upper_bound)

    # Step 2: 计算步长（基于spectral norm）
    # Power iteration估算 ||M||_2
    u = torch.randn(t, device=M.device)
    u = u / (u.norm() + 1e-12)
    for _ in range(3):  # 快速收敛
        v = M @ u
        v = v / (v.norm() + 1e-12)
        u = M.T @ v
        u = u / (u.norm() + 1e-12)

    L = (u @ (M.T @ (M @ u))).sqrt().clamp(min=1e-6) ** 2  # ||M||_2^2
    eta = 1.0 / L  # 步长

    # Step 3: Projected Gradient Descent
    for _ in range(iters):
        # Gradient: M^T(M·B - y)
        residual = M @ B - y  # (n,)
        grad = M.T @ residual  # (t,)

        # Gradient step
        B = B - eta * grad

        # Projection
        B = B.clamp(min=lower_bound, max=upper_bound)

    # Step 4: 转换为beta
    beta = torch.log(B)

    return beta
```

**MLX移植**:
```python
def nnls_pgd(M, y, iters=100, lower_bound=1e-12, upper_bound=None):
    """
    MLX版本的Projected Gradient Descent
    """
    n, t = M.shape

    # 初始解
    B = mx.linalg.lstsq(M, y[..., None])[0].squeeze(-1)
    B = mx.clip(B, lower_bound, upper_bound if upper_bound else 1e10)

    # Power iteration估算spectral norm
    u = mx.random.normal((t,))
    u = u / (mx.linalg.norm(u) + 1e-12)
    for _ in range(3):
        v = M @ u
        v = v / (mx.linalg.norm(v) + 1e-12)
        u = M.T @ v
        u = u / (mx.linalg.norm(u) + 1e-12)

    L = (u @ (M.T @ (M @ u))) ** 0.5
    L = L ** 2
    L = mx.clip(L, 1e-6, None)
    eta = 1.0 / L

    # PGD iterations
    for _ in range(iters):
        residual = M @ B - y
        grad = M.T @ residual
        B = B - eta * grad
        B = mx.clip(B, lower_bound, upper_bound if upper_bound else 1e10)

    beta = mx.log(B)
    return beta
```

### 1.3 数值稳定性技巧

```python
# 1. 所有计算在fp32
M32 = M.to(torch.float32)
y32 = y.to(torch.float32)
beta32 = nnls_pg(M32, y32)
beta = beta32.to(original_dtype)  # 最后转回bf16/fp16

# 2. Log-sum-exp技巧
scores = Q @ K.T / sqrt(d)
max_scores = scores.max(dim=1, keepdim=True)
exp_scores = torch.exp(scores - max_scores)  # 避免overflow

# 3. Clamp下界
lower_bound = 1e-12  # 避免log(0)
```

---

## 2. LSQ C2拟合算法

### 2.1 问题定义

```
目标: softmax(Q·C1^T + beta)·C2 ≈ softmax(Q·K^T)·V
简化为: X·C2 ≈ Y

其中:
- X = softmax(Q·C1^T / sqrt(d) + beta)  # (n_queries, t) - compacted attention weights
- Y = softmax(Q·K^T / sqrt(d))·V  # (n_queries, head_dim) - original attention outputs
- C2: (t, head_dim) - 待求解的compacted values
```

### 2.2 求解器实现

#### Option 1: Standard Least Squares (默认)

```python
def _compute_C2(C1, beta, K, V, queries, ridge_lambda=0, solver='lstsq'):
    """
    标准最小二乘求解

    Time: ~1ms per head
    """
    # Step 1: 计算X (compacted attention weights)
    scores_C = queries @ C1.T / sqrt(d) + beta  # (n, t)
    X = torch.softmax(scores_C, dim=1)  # (n, t)

    # Step 2: 计算Y (target outputs)
    scores_K = queries @ K.T / sqrt(d)  # (n, T)
    attn_K = torch.softmax(scores_K, dim=1)  # (n, T)
    Y = attn_K @ V  # (n, head_dim)

    # Step 3: 求解 X·C2 = Y
    if ridge_lambda == 0:
        # 无正则化
        C2 = torch.linalg.lstsq(X, Y).solution  # (t, head_dim)
    else:
        # Ridge regression: C2 = (X^T X + λI)^{-1} X^T Y
        XtX = X.T @ X  # (t, t)
        XtX += ridge_lambda * torch.eye(t)  # 正则化
        XtY = X.T @ Y  # (t, head_dim)
        C2 = torch.linalg.solve(XtX, XtY)  # (t, head_dim)

    return C2
```

**MLX移植**:
```python
def compute_C2_lsq(C1, beta, K, V, queries, ridge_lambda=0):
    """
    MLX版本的LSQ求解器
    """
    d = K.shape[-1]
    inv_sqrt_d = (1.0 / d) ** 0.5

    # X: compacted attention weights
    scores_C = queries @ C1.T * inv_sqrt_d + beta[None, :]  # (n, t)
    X = mx.softmax(scores_C, axis=1)  # (n, t)

    # Y: target outputs
    scores_K = queries @ K.T * inv_sqrt_d  # (n, T)
    attn_K = mx.softmax(scores_K, axis=1)  # (n, T)
    Y = attn_K @ V  # (n, head_dim)

    # 求解
    if ridge_lambda == 0:
        C2 = mx.linalg.lstsq(X, Y)[0]  # (t, head_dim)
    else:
        XtX = X.T @ X
        XtX = XtX + ridge_lambda * mx.eye(XtX.shape[0])
        XtY = X.T @ Y
        C2 = mx.linalg.solve(XtX, XtY)

    return C2
```

#### Option 2: Cholesky Decomposition (更快)

```python
def _compute_C2_cholesky(C1, beta, K, V, queries, ridge_lambda=0):
    """
    Cholesky分解求解（更快，更稳定）

    Time: ~0.5ms per head
    """
    # 计算X, Y同上
    # ...

    n, t = X.shape

    if n < t:
        # Underdetermined: C2 = X^T (XX^T + λI)^{-1} Y
        XXt = X @ X.T
        XXt = 0.5 * (XXt + XXt.T)  # 对称化
        XXt.diagonal().add_(ridge_lambda)
        L = torch.linalg.cholesky(XXt)  # Cholesky分解
        Z = torch.cholesky_solve(Y, L)  # 求解 (XX^T + λI)Z = Y
        C2 = X.T @ Z
    else:
        # Overdetermined: C2 = (X^T X + λI)^{-1} X^T Y
        XtX = X.T @ X
        XtX = 0.5 * (XtX + XtX.T)  # 对称化
        XtX.diagonal().add_(ridge_lambda)
        L = torch.linalg.cholesky(XtX)
        XtY = X.T @ Y
        C2 = torch.cholesky_solve(XtY, L)

    return C2
```

**MLX移植**:
```python
def compute_C2_cholesky(C1, beta, K, V, queries, ridge_lambda=0):
    """
    MLX版本的Cholesky求解器
    """
    # 计算X, Y (同上)
    # ...

    n, t = X.shape

    if n < t:
        XXt = X @ X.T
        XXt = 0.5 * (XXt + XXt.T)
        XXt = XXt + ridge_lambda * mx.eye(n)
        L = mx.linalg.cholesky(XXt)

        # Solve L L^T Z = Y
        # Step 1: L y' = Y (forward substitution)
        # Step 2: L^T Z = y' (backward substitution)
        # MLX没有cholesky_solve，需要手动实现
        Z = mx.linalg.solve(L, Y)  # Forward
        Z = mx.linalg.solve(L.T, Z)  # Backward

        C2 = X.T @ Z
    else:
        XtX = X.T @ X
        XtX = 0.5 * (XtX + XtX.T)
        XtX = XtX + ridge_lambda * mx.eye(t)
        L = mx.linalg.cholesky(XtX)
        XtY = X.T @ Y

        C2 = mx.linalg.solve(L, XtY)
        C2 = mx.linalg.solve(L.T, C2)

    return C2
```

### 2.3 Ridge Regression自适应缩放

```python
# Spectral norm scaling (默认)
lambda_scaled = ridge_lambda * (torch.linalg.matrix_norm(X, ord=2) ** 2)

# Frobenius norm scaling
lambda_scaled = ridge_lambda * (torch.linalg.matrix_norm(X, ord='fro') ** 2) / t

# Fixed scaling
lambda_scaled = ridge_lambda
```

**MLX移植**:
```python
if ridge_scale == 'spectral':
    # MLX没有matrix_norm，用SVD估算
    # ||X||_2 = max singular value
    lambda_scaled = ridge_lambda * (mx.linalg.norm(X, ord=2) ** 2)
elif ridge_scale == 'frobenius':
    lambda_scaled = ridge_lambda * (mx.linalg.norm(X, ord='fro') ** 2) / t
else:  # 'fixed'
    lambda_scaled = ridge_lambda
```

---

## 3. 完整压缩流程

### 3.1 单层单头压缩

```python
def compact_single_head(K, V, queries, budget):
    """
    K, V: (seq_len, head_dim)
    queries: (n_queries, head_dim)
    budget: 目标压缩后的key数量

    Returns:
    C1: (budget, head_dim) - compacted keys
    beta: (budget,) - attention bias
    C2: (budget, head_dim) - compacted values
    """
    # Step 1: 计算attention scores
    scores = queries @ K.T / sqrt(head_dim)  # (n_queries, seq_len)
    attn_weights = torch.softmax(scores, dim=1)  # (n_queries, seq_len)

    # Step 2: 选择top-k keys (highest attention method)
    key_scores = attn_weights.max(dim=0)[0]  # (seq_len,) - 每个key的最大attention
    top_indices = torch.topk(key_scores, budget).indices  # (budget,)

    C1 = K[top_indices]  # (budget, head_dim)

    # Step 3: 拟合beta (NNLS)
    exp_scores = torch.exp(scores - scores.max(dim=1, keepdim=True)[0])
    target = exp_scores.sum(dim=1)  # (n_queries,)
    M = exp_scores[:, top_indices]  # (n_queries, budget)

    B = nnls_pg(M, target, iters=0, lower_bound=1e-12)  # (budget,)
    beta = torch.log(B)  # (budget,)

    # Step 4: 拟合C2 (LSQ)
    C2 = compute_C2_lsq(C1, beta, K, V, queries, ridge_lambda=0)  # (budget, head_dim)

    return C1, beta, C2
```

### 3.2 完整模型压缩

```python
def compact_kv_cache(cache, model, target_ratio=0.2):
    """
    cache: mlx_lm KVCache对象
    model: MLX模型
    target_ratio: 压缩比例（0.2 = 压缩到20%）

    Returns:
    CompactedPrefixCache
    """
    num_layers = len(cache)
    num_heads = cache[0].keys.shape[1]
    seq_len = cache[0].keys.shape[2]
    head_dim = cache[0].keys.shape[3]

    budget_per_head = int(seq_len * target_ratio)

    # Generate queries (repeat-prefill method)
    queries = generate_queries_repeat_prefill(model, cache, n_queries=10)

    # Compact each layer and head
    compacted_layers = []
    for layer_idx in range(num_layers):
        C1_heads = []
        beta_heads = []
        C2_heads = []

        for head_idx in range(num_heads):
            K = cache[layer_idx].keys[0, head_idx, :, :]  # (seq_len, head_dim)
            V = cache[layer_idx].values[0, head_idx, :, :]
            Q_ref = queries[layer_idx, head_idx, :, :]  # (n_queries, head_dim)

            C1, beta, C2 = compact_single_head(K, V, Q_ref, budget_per_head)

            C1_heads.append(C1[None, None, :, :])  # (1, 1, budget, head_dim)
            beta_heads.append(beta[None, None, :])  # (1, 1, budget)
            C2_heads.append(C2[None, None, :, :])

        # Concatenate all heads
        C1_layer = mx.concatenate(C1_heads, axis=1)  # (1, num_heads, budget, head_dim)
        beta_layer = mx.concatenate(beta_heads, axis=1)
        C2_layer = mx.concatenate(C2_heads, axis=1)

        compacted_layers.append((C1_layer, beta_layer, C2_layer))

    return CompactedPrefixCache(compacted_layers, original_seq_len=seq_len)
```

---

## 4. MLX移植关键点

### 4.1 API差异

| PyTorch | MLX | 备注 |
|---------|-----|------|
| `torch.linalg.lstsq(A, b).solution` | `mx.linalg.lstsq(A, b)[0]` | 返回值结构不同 |
| `torch.softmax(x, dim=1)` | `mx.softmax(x, axis=1)` | 参数名不同 |
| `torch.clamp(x, min, max)` | `mx.clip(x, min, max)` | 函数名不同 |
| `torch.cholesky_solve(b, L)` | 需手动实现 | MLX缺少 |
| `torch.linalg.matrix_norm(X, ord=2)` | `mx.linalg.norm(X, ord=2)` | 函数名不同 |

### 4.2 必须实现的辅助函数

```python
# 1. Cholesky solve
def cholesky_solve_mlx(L, b):
    """
    Solve L L^T x = b

    L: Lower triangular matrix
    b: Right-hand side
    """
    # Forward substitution: L y = b
    y = mx.linalg.solve(L, b)
    # Backward substitution: L^T x = y
    x = mx.linalg.solve(L.T, y)
    return x

# 2. Power iteration for spectral norm
def spectral_norm_mlx(M, n_iters=3):
    """
    Estimate ||M||_2 via power iteration
    """
    t = M.shape[1]
    u = mx.random.normal((t,))
    u = u / (mx.linalg.norm(u) + 1e-12)

    for _ in range(n_iters):
        v = M @ u
        v = v / (mx.linalg.norm(v) + 1e-12)
        u = M.T @ v
        u = u / (mx.linalg.norm(u) + 1e-12)

    return (u @ (M.T @ (M @ u))) ** 0.5
```

### 4.3 JIT编译优化

```python
@mx.compile
def compact_single_head_compiled(K, V, queries, budget):
    """
    JIT编译加速
    """
    # ... same as above
    return C1, beta, C2

# 使用
C1, beta, C2 = compact_single_head_compiled(K, V, queries, budget)
mx.eval(C1, beta, C2)  # 强制执行
```

---

## 5. 验收标准

### 5.1 Phase A: 单层单头验证

```python
# Synthetic data
K = mx.random.normal((1000, 128))
V = mx.random.normal((1000, 128))
queries = mx.random.normal((10, 128))

C1, beta, C2 = compact_single_head(K, V, queries, budget=100)

# 验收: Attention mass error
scores_original = queries @ K.T / mx.sqrt(128.0)
attn_original = mx.softmax(scores_original, axis=1)

scores_compacted = queries @ C1.T / mx.sqrt(128.0) + beta[None, :]
attn_compacted = mx.softmax(scores_compacted, axis=1)

# 需要填充compacted attention到原始长度才能对比
# 但这里只验证输出
output_original = attn_original @ V
output_compacted = attn_compacted @ C2

mse = mx.mean((output_original - output_compacted) ** 2)
relative_error = mse / mx.mean(output_original ** 2)

print(f"MSE: {mse.item()}")
print(f"Relative error: {relative_error.item()}")

# 成功标准
assert relative_error < 0.05  # <5% error
```

### 5.2 Phase B: 多头验证

```python
# 验证logical length保留
compacted_cache = compact_kv_cache(cache, model, target_ratio=0.2)

assert compacted_cache.get_seq_length() == cache[0].keys.shape[2]  # Logical length不变
assert compacted_cache[0][0].shape[2] == int(cache[0].keys.shape[2] * 0.2)  # Physical length缩小
```

---

*核心算法文档 v1.0*
*Created: 2026-03-21*
*Based on: https://github.com/adamzweiger/compaction*
