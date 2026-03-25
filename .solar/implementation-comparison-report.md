# Attention Matching 实现对比报告

**对比对象**:
- 论文: Fast KV Compaction via Attention Matching (arXiv:2602.16284)
- 作者实现: https://github.com/adamzweiger/compaction
- FlashMLX 实现: `/Users/lisihao/FlashMLX/src/flashmlx/`

**对比时间**: 2026-03-22 21:00

---

## ✅ 一致性检查清单

### 1. 核心算法流程

| 组件 | 论文/作者实现 | FlashMLX 实现 | 状态 | 备注 |
|------|--------------|--------------|------|------|
| **Input** | K (T,d), V (T,d), queries (n,d), budget t | K (T,d), V (T,d), queries (n,d), budget t | ✅ 一致 | 接口完全相同 |
| **Output** | C1 (t,d), beta (t,), C2 (t,d), indices | C1 (t,d), beta (t,), C2 (t,d), indices | ✅ 一致 | 输出格式相同 |
| **Pipeline** | Key Selection → Beta Fitting → Value Fitting | Key Selection → Beta Fitting → Value Fitting | ✅ 一致 | 三步流程相同 |

---

### 2. Key Selection (C1 计算)

#### 2.1 Attention Score 计算

**作者实现** (`highest_attention_keys.py:156-179`):
```python
# Compute attention scores in fp32
inv_sqrt_d = (1.0 / d) ** 0.5
scores_raw = queries @ K.T                                 # (n, T) original dtype
scores32 = scores_raw.to(torch.float32) * inv_sqrt_d       # (n, T) fp32
max_scores = scores32.max(dim=1, keepdim=True)[0]          # (n, 1) fp32
exp_scores = torch.exp(scores32 - max_scores)              # (n, T) fp32
sum_exp = exp_scores.sum(dim=1, keepdim=True)              # (n, 1)
attention_weights = exp_scores / sum_exp                   # (n, T) normalized
```

**FlashMLX 实现** (`compaction_algorithm.py:119-122`):
```python
# Compute attention scores
scale = 1.0 / mx.sqrt(mx.array(d, dtype=K.dtype))
attn_scores = queries @ K.T * scale  # (n, T)
attn_weights = mx.softmax(attn_scores, axis=-1)  # (n, T)
```

| 特性 | 作者 | FlashMLX | 状态 | 差异 |
|------|------|---------|------|------|
| 缩放因子 | ✅ 1/√d | ✅ 1/√d | ✅ 一致 | - |
| 数值稳定性 | ✅ fp32 + max normalization | ⚠️ 使用 mx.softmax (内部稳定) | ⚠️ 实现不同但等价 | mx.softmax 内部使用 log-sum-exp trick |
| 类型处理 | ✅ 显式 fp32 提升 | ⚠️ 隐式处理 | ⚠️ 可能有精度差异 | MLX 可能在 bf16/fp16 下计算 |

**潜在问题**: FlashMLX 没有显式提升到 fp32，可能在低精度模型上有数值误差。

#### 2.2 Score Aggregation

**作者实现** (`highest_attention_keys.py:181-191`):
```python
if self.score_method == 'rms':
    key_scores = torch.sqrt((attention_weights ** 2).mean(dim=0))  # (T,)
elif self.score_method == 'max':
    key_scores = attention_weights.max(dim=0)[0]  # (T,)
else:  # 'mean'
    key_scores = attention_weights.mean(dim=0)  # (T,)
```

**FlashMLX 实现** (`compaction_algorithm.py:125-130`):
```python
if self.score_method == 'mean':
    key_scores = mx.mean(attn_weights, axis=0)  # (T,)
elif self.score_method == 'max':
    key_scores = mx.max(attn_weights, axis=0)  # (T,)
else:  # 'sum'
    key_scores = mx.sum(attn_weights, axis=0)  # (T,)
```

| 方法 | 作者支持 | FlashMLX 支持 | 状态 |
|------|---------|--------------|------|
| `mean` | ✅ | ✅ | ✅ 一致 |
| `max` | ✅ | ✅ | ✅ 一致 |
| `rms` | ✅ | ❌ | ❌ **缺失** |
| `sum` | ❌ | ✅ | ⚠️ 额外 (不影响) |

**问题**: FlashMLX 缺少 `rms` 方法，但这不是论文核心方法（默认用 `mean`）。

#### 2.3 Top-K Selection

**作者实现** (`highest_attention_keys.py:204-206`):
```python
_, top_indices = torch.topk(key_scores, t, largest=True)
```

**FlashMLX 实现** (`compaction_algorithm.py:133,392-393`):
```python
sorted_indices = mx.argsort(-scores)  # (T,)
indices = sorted_indices[:k]  # (k,)
```

| 特性 | 作者 | FlashMLX | 状态 |
|------|------|---------|------|
| 方法 | `torch.topk` | `argsort + slice` | ⚠️ 实现不同但等价 |
| 性能 | O(T log k) | O(T log T) | ⚠️ FlashMLX 慢一些 |

**问题**: FlashMLX 实现效率略低，但对小 T (< 1000) 影响不大。

---

### 3. Beta Computation (Beta 拟合)

#### 3.1 NNLS 求解

**作者实现** (`base.py:472-605`):
```python
def _nnls_pg(M, y, iters=0, lower_bound=1e-12, upper_bound=None):
    # Step 1: Clamped LSQ if iters == 0
    if iters == 0:
        B = torch.linalg.lstsq(M, y.unsqueeze(1)).solution.squeeze(1)
        B = B.clamp_min_(lower_bound)
        if upper_bound is not None:
            B = B.clamp_max_(upper_bound)
        return B

    # Step 2: Power iteration for step size
    sigma = estimate_spectral_norm(M)
    L = sigma ** 2
    eta = 1.0 / L

    # Step 3: Projected gradient descent
    for _ in range(iters):
        grad = M.T @ (M @ B - y)
        B = B - eta * grad
        B = B.clamp_min_(lower_bound)
```

**FlashMLX 实现** (`solvers.py:79-150`):
```python
def nnls_pgd(M, y, lower_bound=0.0, max_iters=100, step_size=None):
    # Step 1: Auto-compute step size using power iteration
    if step_size is None:
        v = mx.random.normal((n,))
        v = v / mx.linalg.norm(v)
        for _ in range(10):
            Mv = M @ v
            MTMv = M.T @ Mv
            v = MTMv / mx.linalg.norm(MTMv)
        spectral_norm_M = mx.linalg.norm(M @ v) / mx.linalg.norm(v)
        L = spectral_norm_M ** 2
        step_size = 1.0 / (L + 1e-8)

    # Step 2: Warm start with clamped solution
    x = nnls_clamped(M, y, lower_bound)

    # Step 3: Projected gradient descent
    for _ in range(max_iters):
        grad = M.T @ (M @ x - y)
        x = x - step_size * grad
        x = mx.maximum(x, lower_bound)
```

| 特性 | 作者 | FlashMLX | 状态 | 差异 |
|------|------|---------|------|------|
| **Clamped LSQ (iters=0)** | ✅ 支持 | ✅ 支持 (`nnls_clamped`) | ✅ 一致 | - |
| **PGD (iters>0)** | ✅ 支持 | ✅ 支持 (`nnls_pgd`) | ✅ 一致 | - |
| **Step size 估计** | ✅ Power iteration | ✅ Power iteration | ✅ 一致 | 实现略有不同但等价 |
| **Warm start** | ❌ 冷启动 (B = lstsq) | ✅ 暖启动 (nnls_clamped) | ⚠️ FlashMLX 更好 | FlashMLX 可能收敛更快 |
| **Upper bound** | ✅ 支持 | ❌ 不支持 | ⚠️ FlashMLX 缺失 | 不影响核心功能 |

**结论**: FlashMLX 的 NNLS 实现**正确且完整**，甚至在 warm start 上优于作者实现。

#### 3.2 Beta 计算流程

**作者实现** (`highest_attention_keys.py:216-224`):
```python
# Target: sum of exp_scores across all queries
target = exp_scores.sum(dim=1)  # (n,)

# Design matrix: selected exp_scores
M = exp_scores[:, selected_indices]  # (n, t)

# Solve NNLS: min ||M B - target||^2, B >= 0
B = self._nnls_pg(M, target, iters, lower_bound, upper_bound)  # (t,)

# Convert to log-space: beta = log(B)
beta = torch.log(B)  # (t,)
```

**FlashMLX 实现** (`compaction_algorithm.py:143-202`):
```python
# Compute base attention (before beta)
base_attn = mx.softmax(attn_scores_C1, axis=-1)  # (n, t)

# Target: attention on selected keys
target_attn = attn_weights[:, indices]  # (n, t)

# Log-ratio approximation
log_ratio = mx.log(target_attn_safe / base_attn_safe)  # (n, t)

# For each key j, solve NNLS to find beta[j]
beta_list = []
for j in range(t):
    y_j = log_ratio[:, j]  # (n,)
    M_j = mx.ones((n, 1))  # (n, 1)
    beta_j = nnls_pgd(M_j, y_j, lower_bound=-10.0, max_iters=50)
    beta_list.append(float(beta_j[0]))

beta = mx.array(beta_list)
```

| 组件 | 作者实现 | FlashMLX 实现 | 状态 |
|------|---------|--------------|------|
| **目标函数** | 匹配 exp(scores) 的和 | 匹配 softmax 分布 | ❌ **完全不同** |
| **设计矩阵** | exp_scores[:, indices] (n,t) | ones(n,1) for each key | ❌ **完全不同** |
| **NNLS 变量** | B (t,) - unnormalized weights | beta[j] (scalar) - log bias | ❌ **完全不同** |
| **最终输出** | beta = log(B) | beta (直接输出) | ❌ **完全不同** |

### 🔴 **CRITICAL FINDING: Beta 计算方法完全不同！**

**作者方法** (正确):
```
目标: 找到 B (t,) 使得
  sum_j(exp(q @ C1[j]^T) * B[j]) ≈ sum_i(exp(q @ K[i]^T))

求解: NNLS on M @ B ≈ target
  where M = exp_scores[:, selected_indices]  # (n, t)
        target = exp_scores.sum(dim=1)        # (n,)

输出: beta = log(B)
```

**FlashMLX 方法** (错误):
```
目标: 找到 beta[j] 使得
  softmax(q @ C1[j]^T + beta[j])[j] ≈ target_attn[:, j]

求解: 对每个 key j 独立求解
  NNLS on ones @ beta[j] ≈ log_ratio[:, j]

输出: beta (直接)
```

### 🚨 **问题分析**:

1. **作者方法的数学基础**:
   - 基于论文公式: `softmax(q @ C1^T + beta) @ C2 ≈ softmax(q @ K^T) @ V`
   - 在 softmax 之前的 **unnormalized** 层面匹配
   - 使用 `beta = log(B)` 将 multiplicative weights 转换为 additive bias
   - **全局优化**: 所有 beta 值联合求解，确保整体分布匹配

2. **FlashMLX 方法的问题**:
   - 基于 log-ratio 的**局部近似**
   - **独立求解**: 每个 beta[j] 独立优化，忽略 softmax 的耦合效应
   - 使用 `ones` 作为设计矩阵，相当于求 `mean(log_ratio)`
   - **近似误差**: log-ratio 只在 base 和 target 接近时有效

3. **为什么测试通过了**:
   - **Self-study queries 的特殊性**:
     - 生成自 K-means 聚类中心
     - 天然匹配 KV cache 的分布
     - base_attn 和 target_attn 本来就很接近
     - log-ratio ≈ 0，beta ≈ 0 是合理的
   - **测试方法正确**:
     - 压缩和评估使用相同 queries (修复后)
     - 完美质量 (1.000) 说明 beta ≈ 0 时算法也能工作
   - **但这不代表方法正确**:
     - 在 distribution shift 大的场景可能失败
     - 没有利用 NNLS 的全局优化能力

---

### 4. Value Fitting (C2 计算)

#### 4.1 Ridge Regression 求解

**作者实现** (`base.py:61-240`):
```python
def _compute_C2(C1, beta, K, V, queries):
    # Y = softmax((QK)/sqrt(d)) @ V
    sK = queries @ K.T * inv_sqrt_d
    attn_K = softmax(sK)
    Y = attn_K @ V  # (n, d) - target output

    # X = softmax((Q C1^T)/sqrt(d) + beta)
    sC = queries @ C1.T * inv_sqrt_d + beta
    X = softmax(sC)  # (n, t) - compressed attention weights

    # Solve: X @ C2 = Y with ridge regularization
    if solver == 'lstsq':
        C2 = torch.linalg.lstsq(X, Y).solution
    elif solver == 'cholesky':
        if n < t:
            XXt = X @ X.T + lam * I
            Z = cholesky_solve(Y, XXt)
            C2 = X.T @ Z
        else:
            XtX = X.T @ X + lam * I
            XtY = X.T @ Y
            C2 = cholesky_solve(XtY, XtX)
```

**FlashMLX 实现** (`compaction_algorithm.py:212-335`):
```python
def _compute_C2(C1, beta, K, V, queries):
    # Y = softmax((QK)/sqrt(d)) @ V
    scores_K = queries @ K.T * scale
    attn_K = mx.softmax(scores_K, axis=-1)
    y = attn_K @ V  # (n, d) - target

    # X = softmax((Q C1^T)/sqrt(d))
    scores_C1 = queries @ C1.T * scale
    attn_C1 = mx.softmax(scores_C1, axis=-1)
    X = attn_C1  # (n, t) - compressed attention (WITHOUT beta!)

    # Solve: X @ C2 = y with ridge
    XTX = X.T @ X
    XTy = X.T @ y
    XTX_reg = XTX + scaled_lambda * I
    C2 = np.linalg.solve(XTX_reg, XTy)  # NumPy fallback
```

| 组件 | 作者 | FlashMLX | 状态 | 问题 |
|------|------|---------|------|------|
| **Target Y** | ✅ softmax(QK) @ V | ✅ softmax(QK) @ V | ✅ 一致 | - |
| **Design matrix X** | ✅ softmax(QC1 + beta) | ❌ softmax(QC1) | ❌ **缺少 beta!** | **CRITICAL** |
| **Ridge scaling** | ✅ Spectral norm | ✅ Spectral norm | ✅ 一致 | - |
| **Solver** | ✅ lstsq/cholesky | ⚠️ NumPy solve | ⚠️ 实现不同 | 性能问题 |

### 🔴 **CRITICAL FINDING: C2 计算缺少 beta 项！**

**问题**: FlashMLX 在计算 C2 时，压缩 attention weights 没有加入 beta:
```python
# 作者 (正确):
sC = queries @ C1.T * inv_sqrt_d + beta  # beta included!
X = softmax(sC)

# FlashMLX (错误):
scores_C1 = queries @ C1.T * scale  # NO beta!
attn_C1 = mx.softmax(scores_C1, axis=-1)
```

**影响**:
- C2 的拟合目标错误 - 应该匹配 `softmax(QC1 + beta)` 而不是 `softmax(QC1)`
- 破坏了 beta 的作用 - beta 只在 forward 时生效，但 C2 训练时看不到
- 可能导致次优的 C2 值

**为什么测试还能通过**:
- beta ≈ 0 时，`softmax(QC1 + 0) ≈ softmax(QC1)`
- Self-study queries 场景下 beta 很小，所以影响不大
- 但这是一个潜在的 bug！

---

### 5. 数值稳定性

| 特性 | 作者实现 | FlashMLX 实现 | 状态 |
|------|---------|--------------|------|
| **fp32 提升** | ✅ 显式 `.to(torch.float32)` | ⚠️ 隐式（mx.softmax 内部） | ⚠️ 可能有差异 |
| **Max normalization** | ✅ `scores - max_scores` | ✅ mx.softmax 内部实现 | ✅ 等价 |
| **Log-sum-exp** | ✅ 手动实现 | ✅ mx.softmax 内部 | ✅ 等价 |
| **Epsilon 保护** | ✅ lower_bound=1e-12 | ✅ eps=1e-10 | ✅ 一致 |
| **Ridge regularization** | ✅ 多种缩放策略 | ✅ Spectral norm 缩放 | ✅ 一致 |

---

### 6. Query Generation (self-study)

**作者实现** (`query_generation/self_study.py`):
```python
def self_study_kmeans(K, num_clusters, n_init=10):
    # K-means on keys
    kmeans = KMeans(n_clusters=num_clusters, n_init=n_init)
    kmeans.fit(K.cpu().numpy())
    centroids = torch.from_numpy(kmeans.cluster_centers_)
    return centroids
```

**FlashMLX 实现** (`compaction/query_generation.py`):
```python
def self_study_kmeans(keys, num_queries, n_init=10):
    # K-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_queries, n_init=n_init)
    keys_np = np.array(keys)
    kmeans.fit(keys_np)
    centroids = mx.array(kmeans.cluster_centers_)
    return centroids
```

| 特性 | 作者 | FlashMLX | 状态 |
|------|------|---------|------|
| **K-means** | ✅ sklearn | ✅ sklearn | ✅ 一致 |
| **n_init** | ✅ 支持 | ✅ 支持 | ✅ 一致 |
| **返回格式** | centroids | centroids | ✅ 一致 |

---

### 7. OMP Refinement

**作者实现**:
- ✅ 提供完整 OMP 实现 (`algorithms/omp.py`, 300+ lines)
- ✅ 支持 greedy OMP 和 batched OMP
- ✅ 论文核心方法之一

**FlashMLX 实现**:
- ✅ 提供 OMP 实现 (`compaction/query_generation.py:omp_refine_queries`)
- ✅ 支持 greedy OMP
- ⚠️ 缺少 batched OMP (性能优化)

| 特性 | 作者 | FlashMLX | 状态 |
|------|------|---------|------|
| **Greedy OMP** | ✅ | ✅ | ✅ 一致 |
| **Batched OMP** | ✅ | ❌ | ⚠️ 缺失 (不影响正确性) |
| **Fast OMP** | ✅ | ❌ | ⚠️ 缺失 (不影响正确性) |

---

## 📊 总结对比表

| 模块 | 作者实现 | FlashMLX 实现 | 一致性 | 问题等级 |
|------|---------|--------------|--------|---------|
| **Key Selection (C1)** | ✅ 完整 | ⚠️ 缺少 fp32 提升 + rms 方法 | 80% | LOW |
| **Beta Computation** | ✅ 全局 NNLS (正确) | ❌ 局部 log-ratio (近似) | 30% | **CRITICAL** |
| **Value Fitting (C2)** | ✅ 带 beta 的 Ridge | ❌ 不带 beta 的 Ridge | 60% | **CRITICAL** |
| **NNLS Solver** | ✅ Clamped + PGD | ✅ Clamped + PGD (更好) | 110% | ✅ BETTER |
| **Query Generation** | ✅ K-means | ✅ K-means | 100% | ✅ GOOD |
| **OMP Refinement** | ✅ Greedy + Batched | ⚠️ 仅 Greedy | 70% | LOW |
| **数值稳定性** | ✅ 显式 fp32 | ⚠️ 隐式稳定性 | 90% | MEDIUM |

---

## 🚨 关键问题清单

### P0 - CRITICAL (必须修复)

1. **Beta 计算方法错误**
   - **现状**: 使用局部 log-ratio 近似 + 独立求解
   - **正确**: 全局 NNLS on unnormalized attention (作者方法)
   - **影响**: 在 distribution shift 大的场景可能质量下降
   - **修复**: 参考 `highest_attention_keys.py:216-224`

2. **C2 计算缺少 beta**
   - **现状**: `attn_C1 = mx.softmax(scores_C1, axis=-1)` (line 260)
   - **正确**: `attn_C1 = mx.softmax(scores_C1 + beta[None, :], axis=-1)`
   - **影响**: C2 的拟合目标错误，beta 不参与训练
   - **修复**: 参考 `base.py:136`

### P1 - HIGH (建议修复)

3. **缺少显式 fp32 提升**
   - **现状**: 依赖 mx.softmax 内部稳定性
   - **正确**: 显式提升到 fp32 再计算 softmax
   - **影响**: 在 bf16/fp16 模型上可能有数值误差
   - **修复**: 参考 `base.py:108-132`

### P2 - MEDIUM (可选修复)

4. **缺少 rms score 方法**
   - **影响**: 论文支持但非默认，影响不大
   - **修复**: 添加 `rms` 选项到 `score_method`

5. **缺少 Batched OMP**
   - **影响**: OMP 性能较低，但不影响正确性
   - **修复**: 参考 `omp_batched.py`

---

## ✅ 测试为什么能通过

1. **Self-study queries 的救赎**:
   - 生成自 K-means，天然匹配 KV cache 分布
   - base_attn ≈ target_attn，beta ≈ 0
   - 即使 beta 计算方法错误，结果也接近 0
   - C2 缺少 beta 影响不大 (因为 beta ≈ 0)

2. **测试方法正确**:
   - 压缩和评估使用相同 queries
   - 避免了 query mismatch 问题

3. **NNLS 实现正确**:
   - `nnls_pgd` 和 `nnls_clamped` 实现都是正确的
   - 只是用在了错误的地方

---

## 🎯 修复优先级

### 立即修复 (今晚)

1. **修复 C2 计算的 beta 缺失** (10 分钟)
   ```python
   # Line 260 修改为:
   scores_C1_with_beta = scores_C1 + beta[None, :]  # 加入 beta!
   attn_C1 = mx.softmax(scores_C1_with_beta, axis=-1)
   ```

2. **修复 Beta 计算方法** (30 分钟)
   ```python
   # 参考作者实现，使用全局 NNLS
   # 1. 计算 exp_scores = exp(queries @ K.T * scale - max)
   # 2. target = exp_scores.sum(dim=1)
   # 3. M = exp_scores[:, indices]
   # 4. B = nnls_pgd(M, target)
   # 5. beta = log(B)
   ```

### 短期修复 (明天)

3. **添加显式 fp32 提升** (20 分钟)
4. **添加 rms score 方法** (10 分钟)

### 长期优化 (下周)

5. **添加 Batched OMP** (2 小时)

---

*报告生成时间: 2026-03-22 21:00*
*状态: 待修复*
