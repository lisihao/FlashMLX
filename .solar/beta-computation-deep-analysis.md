# Beta 计算深度对比分析

## 作者的方法（完整推导）

### 数学目标

原始 attention:
```
output = softmax(q @ K^T / sqrt(d)) @ V
       = [exp(q @ K[i]^T / sqrt(d)) / Z_K] @ V
```

压缩 attention:
```
output ≈ softmax(q @ C1^T / sqrt(d) + beta) @ C2
       = [exp(q @ C1[j]^T / sqrt(d) + beta[j]) / Z_C] @ C2
       = [exp(q @ C1[j]^T / sqrt(d)) * exp(beta[j]) / Z_C] @ C2
       = [exp(q @ C1[j]^T / sqrt(d)) * B[j] / Z_C] @ C2  (定义 B[j] = exp(beta[j]))
```

其中partition functions:
```
Z_K = sum_i exp(q @ K[i]^T / sqrt(d))
Z_C = sum_j exp(q @ C1[j]^T / sqrt(d)) * B[j]
```

### NNLS 公式化

**关键洞察**: 要让压缩 attention 接近原始 attention，需要让 **partition functions 匹配**:

```
Z_K ≈ Z_C
sum_i exp(q @ K[i]^T / sqrt(d)) ≈ sum_j exp(q @ C1[j]^T / sqrt(d)) * B[j]
```

对于每个 query q，这是一个线性方程：
```
target[q] = M[q, :] @ B
```

其中：
- `target[q] = sum_i exp(q @ K[i]^T / sqrt(d))` (n,) - 原始 partition function
- `M[q, j] = exp(q @ C1[j]^T / sqrt(d))` (n, t) - 压缩 keys 的 exp scores
- `B[j]` (t,) - 未归一化权重，`beta[j] = log(B[j])`

求解 NNLS:
```
min_{B >= 0} ||M @ B - target||^2
```

### 作者代码实现

```python
# Step 1: 计算 unnormalized attention scores
scores32 = queries @ K.T * inv_sqrt_d  # (n, T)
max_scores = scores32.max(dim=1, keepdim=True)[0]  # (n, 1)
exp_scores = torch.exp(scores32 - max_scores)  # (n, T) - 数值稳定

# Step 2: NNLS target = partition function
target = exp_scores.sum(dim=1)  # (n,) - Z_K for each query

# Step 3: 设计矩阵 M = selected exp_scores
M = exp_scores[:, selected_indices]  # (n, t) - Z_C 的各项

# Step 4: 求解 NNLS
B = nnls_pg(M, target)  # (t,) - 满足 M @ B ≈ target, B >= 0

# Step 5: 转换到 log-space
beta = torch.log(B)  # (t,) - beta = log(B)
```

---

## FlashMLX 的方法（完整推导）

### 数学目标

FlashMLX 使用的是 **log-ratio 近似**：

```
target_attn[q, j] = softmax(q @ K^T)[selected_indices[j]]
base_attn[q, j] = softmax(q @ C1[j]^T)

目标: softmax(q @ C1^T + beta) ≈ target_attn
```

**一阶近似** (Taylor expansion):
```
softmax(scores + delta)[j] ≈ softmax(scores)[j] * exp(delta[j] - sum_k softmax(scores)[k] * delta[k])
```

**简化假设** (忽略耦合项):
```
softmax(scores + delta)[j] ≈ softmax(scores)[j] * exp(delta[j])
```

因此：
```
target_attn[q, j] ≈ base_attn[q, j] * exp(beta[j])
beta[j] ≈ log(target_attn[q, j] / base_attn[q, j])
```

### NNLS 公式化

对于每个 key j **独立求解**:
```
min_{beta[j] >= -10} || ones @ beta[j] - log_ratio[:, j] ||^2
```

其中：
- `log_ratio[q, j] = log(target_attn[q, j] / base_attn[q, j])`
- `ones` (n, 1) - 所有 queries 共享同一个 beta[j]

求解结果：
```
beta[j] = mean(log_ratio[:, j])  # 因为 M = ones，所以 NNLS 解就是平均值
```

### FlashMLX 代码实现

```python
# Step 1: 计算 normalized attention
attn_scores = queries @ K.T * scale  # (n, T)
attn_weights = mx.softmax(attn_scores, axis=-1)  # (n, T) - 归一化

# Step 2: 提取目标 attention
target_attn = attn_weights[:, indices]  # (n, t)

# Step 3: 计算基础 attention (without beta)
attn_scores_C1 = queries @ C1.T * scale  # (n, t)
base_attn = mx.softmax(attn_scores_C1, axis=-1)  # (n, t) - 归一化

# Step 4: Log-ratio 近似
log_ratio = mx.log(target_attn / base_attn)  # (n, t)

# Step 5: 对每个 key 独立求解
for j in range(t):
    y_j = log_ratio[:, j]  # (n,)
    M_j = mx.ones((n, 1))  # (n, 1) - 共享 beta
    beta[j] = nnls_pgd(M_j, y_j, lower_bound=-10.0)  # (1,) -> scalar
```

---

## 🚨 核心差异总结

| 维度 | 作者方法 | FlashMLX 方法 |
|------|---------|--------------|
| **优化层面** | Unnormalized (partition function) | Normalized (softmax output) |
| **耦合性** | 全局联合优化 (t 个 beta 一起求解) | 局部独立优化 (t 个 beta 分别求解) |
| **设计矩阵 M** | `exp_scores[:, indices]` (n, t) | `ones` (n, 1) for each key |
| **NNLS 变量** | B (t,) - multiplicative weights | beta[j] (scalar) - additive bias |
| **最终输出** | beta = log(B) | beta (直接) |
| **数学正确性** | ✅ 严格匹配 partition function | ⚠️ 一阶近似 + 忽略耦合 |

---

## 🔍 为什么 FlashMLX 方法是近似？

### 问题 1: 忽略了 Softmax 的耦合效应

Softmax 是一个**耦合操作**：
```
softmax(x + delta)[j] = exp(x[j] + delta[j]) / sum_k exp(x[k] + delta[k])
```

修改任何一个 delta[k] 都会影响**所有**的 softmax[j]（因为分母变了）。

FlashMLX 的假设：
```
softmax(x + delta)[j] ≈ softmax(x)[j] * exp(delta[j])  # 错误！忽略了分母变化
```

正确的公式（作者使用）：
```
softmax(x + delta) = exp(x + delta) / sum_k exp(x[k] + delta[k])
                   = exp(x) * exp(delta) / sum_k exp(x[k]) * exp(delta[k])  # 保留分母耦合
```

### 问题 2: 独立优化 vs 全局优化

**FlashMLX**: 每个 beta[j] 独立求解
```
for j in range(t):
    beta[j] = argmin || beta[j] - log_ratio[:, j] ||^2
```

**作者**: 所有 beta 联合求解
```
B = argmin || M @ B - target ||^2  # B 的各个元素互相影响
```

因为 partition function 是一个**全局约束**：
```
sum_j exp(scores[j]) * B[j] = constant
```

修改任何一个 B[j] 都会影响整个和，所以需要联合优化。

### 问题 3: Normalized vs Unnormalized

**FlashMLX**: 在 normalized attention (softmax 输出) 上优化
- 问题：softmax 抹去了 scale 信息
- 例如：`softmax([1, 2, 3])` 和 `softmax([10, 20, 30])` 完全不同，但 FlashMLX 的 log-ratio 看到的是一样的

**作者**: 在 unnormalized scores (partition function) 上优化
- 保留了 scale 信息
- 可以准确匹配原始分布的"质量"(mass)

---

## 📊 近似误差分析

### 何时 FlashMLX 方法有效？

FlashMLX 方法在以下情况下**近似误差小**：

1. **Base 和 Target 接近**: `base_attn ≈ target_attn`
   - log-ratio ≈ 0
   - beta ≈ 0
   - 一阶近似有效

2. **Self-study queries**: K-means 聚类生成的 queries
   - 天然匹配 KV cache 分布
   - 选中的 keys 本来就是高 attention 的
   - base_attn 已经接近 target_attn

3. **低压缩比**: compression_ratio = 2x, 3x
   - 保留了大部分高 attention keys
   - 分布变化小

### 何时 FlashMLX 方法失效？

1. **Distribution shift 大**:
   - Evaluation queries 与 training queries 分布不同
   - Base 和 target 差异大
   - log-ratio 近似误差大

2. **高压缩比**: compression_ratio = 8x, 16x
   - 丢失了很多 keys
   - 需要更强的 beta 补偿
   - 局部优化不够

3. **复杂的 attention patterns**:
   - Multi-modal distributions
   - Long-tail attention
   - Softmax 耦合效应强

---

## ✅ 我们的测试为什么通过了？

完全符合 FlashMLX 方法有效的条件：

1. ✅ **Self-study queries** - K-means 生成
2. ✅ **Base ≈ Target** - beta ≈ 0 (mean=-0.000000)
3. ✅ **中等压缩比** - 4x (92 -> 23 tokens)
4. ✅ **相同 queries** - 压缩和评估用同一批 queries

**结论**: 在这个特定场景下，两种方法都给出 beta ≈ 0，所以质量都是完美的 (1.000)。

但这**不代表方法在所有场景下等价**！

---

## 🎯 推荐修复方案

### Option A: 完全采用作者方法 (推荐)

**优势**:
- 数学严格正确
- 全局优化，质量更好
- 适用于更广泛的场景

**实现** (30 分钟):
```python
# compaction_algorithm.py:143-202 替换为:
# 1. 计算 exp_scores (unnormalized)
scores = queries @ K.T * scale
max_scores = mx.max(scores, axis=1, keepdims=True)
exp_scores = mx.exp(scores - max_scores)  # (n, T)

# 2. NNLS target
target = mx.sum(exp_scores, axis=1)  # (n,)

# 3. 设计矩阵
M = exp_scores[:, indices]  # (n, t)

# 4. 求解
from ..compaction.solvers import nnls_pgd
B = nnls_pgd(M, target, lower_bound=1e-12, max_iters=100)  # (t,)

# 5. 转换
beta = mx.log(B)  # (t,)
```

### Option B: 保持当前方法 + 添加警告

**优势**:
- 无需修改
- 在 self-study 场景下已验证

**劣势**:
- 可能在其他场景下失效
- 不符合论文方法

**实现**:
```python
# 添加 docstring 说明
"""
Note: This implementation uses a log-ratio approximation which is valid
when base and target attention distributions are similar (e.g., with
self-study queries). For more general cases, use the global NNLS method
from the original paper.
"""
```

---

*分析完成时间: 2026-03-22 21:30*
*结论: 方法确实不同，但在特定场景下等价*
