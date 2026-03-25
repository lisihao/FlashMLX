# Attention Matching 完整移植计划

> **原则**: 论文+代码已证明正确，完整移植，只改环境差异（PyTorch→MLX）

---

## 🎯 移植范围

### 源代码（作者实现）
- **仓库**: https://github.com/adamzweiger/compaction
- **核心文件**:
  - `compaction/algorithms/base.py` - 基类 + C2 计算
  - `compaction/algorithms/highest_attention_keys.py` - 主算法
  - `compaction/algorithms/optim.py` - NNLS/OMP solvers

### 目标代码（FlashMLX）
- **路径**: `src/flashmlx/cache/compaction_algorithm.py`
- **Solver**: `src/flashmlx/compaction/solvers.py`

---

## 📋 逐项移植清单

### **P0 CRITICAL - 必须完全照抄**

| 组件 | 作者实现 | FlashMLX 当前 | 移植动作 |
|------|---------|--------------|---------|
| **Beta 计算** | Global NNLS on partition function | Local log-ratio approximation | ✅ **完全替换** |
| **C2 计算** | 包含 beta 的 softmax | 缺少 beta | ✅ **添加 beta** |
| **数值稳定性** | 显式 fp32 提升 | 隐式（MLX 自动） | ⚠️ **验证 MLX 行为** |

---

### **Beta 计算移植细节**

#### 作者实现逻辑（PyTorch）
```python
# Step 1: 计算 unnormalized attention scores (fp32)
scores32 = queries @ K.T * inv_sqrt_d           # (n, T)
max_scores = scores32.max(dim=1, keepdim=True)[0]  # (n, 1)
exp_scores = torch.exp(scores32 - max_scores)   # (n, T) - 数值稳定

# Step 2: NNLS target = partition function
target = exp_scores.sum(dim=1)                  # (n,) - Z_K

# Step 3: 设计矩阵 M
M = exp_scores[:, selected_indices]             # (n, t) - Z_C 各项

# Step 4: 求解 NNLS
B = nnls_pg(M, target, nnls_iters, lower_bound, upper_bound)  # (t,)

# Step 5: 转换到 log-space
beta32 = torch.log(B)                           # (t,) - beta = log(B)
```

#### MLX 移植版本
```python
# Step 1: 计算 unnormalized attention scores
scores = queries @ K.T * inv_sqrt_d             # (n, T)
max_scores = mx.max(scores, axis=1, keepdims=True)  # (n, 1)
exp_scores = mx.exp(scores - max_scores)        # (n, T)

# Step 2: NNLS target
target = mx.sum(exp_scores, axis=1)             # (n,)

# Step 3: 设计矩阵 M
M = exp_scores[:, indices]                      # (n, t)

# Step 4: 求解 NNLS（已有 nnls_pgd）
from ..compaction.solvers import nnls_pgd
B = nnls_pgd(M, target, lower_bound=1e-12, max_iters=100)  # (t,)

# Step 5: 转换
beta = mx.log(B)                                # (t,)
```

**环境差异**:
- `torch.exp` → `mx.exp` ✅
- `dim=` → `axis=` ✅
- `keepdim=` → `keepdims=` ✅
- NNLS solver 已实现 ✅

---

### **C2 计算移植细节**

#### 作者实现逻辑（PyTorch）
```python
# Step 1: 计算原始 attention 输出（包含 attention_bias）
scores_K = queries @ K.T * inv_sqrt_d           # (n, T)
if attention_bias is not None:
    scores_K = scores_K + attention_bias
max_K = scores_K.max(dim=1, keepdim=True)[0]
exp_K = torch.exp(scores_K - max_K)
sum_K = exp_K.sum(dim=1, keepdim=True)
attn_K = exp_K / sum_K                          # (n, T)
Y = attn_K @ V                                  # (n, d) - 原始输出

# Step 2: 计算压缩 attention weights（包含 beta！）
scores_C = queries @ C1.T * inv_sqrt_d + beta   # (n, t) - 加上 beta!
max_C = scores_C.max(dim=1, keepdim=True)[0]
exp_C = torch.exp(scores_C - max_C)
sum_C = exp_C.sum(dim=1, keepdim=True)
X = exp_C / sum_C                               # (n, t)

# Step 3: Ridge Regression
if ridge_lambda > 0:
    # 计算 scale
    if ridge_scale == 'spectral':
        scale = torch.linalg.matrix_norm(X, ord=2) ** 2
    # 求解
    XtX = X.T @ X
    XtY = X.T @ Y
    C2 = torch.linalg.solve(XtX + ridge_lambda * scale * I, XtY)
else:
    C2 = torch.linalg.lstsq(X, Y).solution
```

#### MLX 移植版本
```python
# Step 1: 计算原始 attention 输出
scores_K = queries @ K.T * inv_sqrt_d
if attention_bias is not None:
    scores_K = scores_K + attention_bias
attn_K = mx.softmax(scores_K, axis=-1)          # (n, T)
Y = attn_K @ V                                  # (n, d)

# Step 2: 计算压缩 attention weights（添加 beta！）
scores_C = queries @ C1.T * inv_sqrt_d + beta   # ⚠️ 当前缺少 + beta
attn_C = mx.softmax(scores_C, axis=-1)          # (n, t)
X = attn_C                                      # (n, t)

# Step 3: Ridge Regression
if ridge_lambda > 0:
    if ridge_scale == 'spectral':
        scale = float(mx.max(mx.linalg.svdvals(X)) ** 2)
    XtX = X.T @ X
    XtY = X.T @ Y
    I = mx.eye(X.shape[1], dtype=X.dtype)
    C2 = mx.linalg.solve(XtX + ridge_lambda * scale * I, XtY)
else:
    C2 = mx.linalg.lstsq(X, Y)[0]
```

**环境差异**:
- MLX 的 `mx.softmax` 内部已做 fp32 提升（需验证）
- `torch.linalg.lstsq().solution` → `mx.linalg.lstsq()[0]` ✅
- `torch.linalg.matrix_norm(ord=2)` → `mx.max(mx.linalg.svdvals())` ✅

---

### **P1 HIGH - 强烈建议照抄**

| 组件 | 作者实现 | FlashMLX 当前 | 移植动作 |
|------|---------|--------------|---------|
| **类型提升策略** | 显式标注 fp32 区域 | 依赖 MLX 自动提升 | ⚠️ **添加注释说明** |
| **NNLS 参数** | `lower_bound`, `upper_bound`, `nnls_iters` | 只有 `lower_bound` | ✅ **添加参数** |

---

### **P2 MEDIUM - 可选功能**

| 组件 | 作者实现 | FlashMLX 当前 | 移植动作 |
|------|---------|--------------|---------|
| **Score method** | `mean`, `max`, `rms` | `mean`, `max`, `sum` | ✅ **添加 rms** |
| **Pooling** | `avgpool`, `maxpool` | 无 | ⚠️ **可选添加** |
| **C2 method** | `lsq`, `direct` | `lsq`, `direct` | ✅ 已有 |
| **Beta method** | `nnls`, `zero` | `nnls`, `zeros`, `ones` | ✅ 已有 |

---

## 🔧 环境适配规则

### **PyTorch → MLX 映射表**

| PyTorch | MLX | 说明 |
|---------|-----|------|
| `torch.exp()` | `mx.exp()` | 完全等价 |
| `torch.log()` | `mx.log()` | 完全等价 |
| `torch.sum(x, dim=1)` | `mx.sum(x, axis=1)` | 参数名不同 |
| `torch.max(x, dim=1, keepdim=True)` | `mx.max(x, axis=1, keepdims=True)` | 参数名不同 |
| `torch.softmax(x, dim=-1)` | `mx.softmax(x, axis=-1)` | 完全等价 |
| `x.to(torch.float32)` | `mx.astype(x, mx.float32)` | 方法不同 |
| `torch.linalg.lstsq(A, b).solution` | `mx.linalg.lstsq(A, b)[0]` | 返回格式不同 |
| `torch.linalg.solve(A, b)` | `mx.linalg.solve(A, b)` | 完全等价 |
| `torch.linalg.matrix_norm(X, ord=2)` | `mx.max(mx.linalg.svdvals(X))` | 需计算 SVD |
| `torch.eye(n)` | `mx.eye(n)` | 完全等价 |

### **数值稳定性差异**

| 操作 | PyTorch 策略 | MLX 策略 | 移植建议 |
|------|-------------|---------|---------|
| Softmax | 手动 fp32 提升 | 自动 fp32（内置） | ✅ 依赖 MLX |
| Matmul | Kernel 自动 fp32 累加 | 同 PyTorch | ✅ 无需改动 |
| Exp/Log | 显式 .to(float32) | 自动提升 | ✅ 可省略显式转换 |

---

## 📝 修改文件清单

### 1. `src/flashmlx/cache/compaction_algorithm.py`

**修改点**:
1. ✅ **Beta 计算**（lines 143-202）：
   - 删除 log-ratio 方法
   - 替换为 global NNLS 方法（照抄作者逻辑）

2. ✅ **C2 计算**（line 260）：
   - 添加 `+ beta` 到 scores_C1
   - 验证 ridge regression 实现

3. ✅ **添加 score_method='rms'**（可选）

4. ✅ **添加参数**：
   - `nnls_upper_bound`
   - 文档说明

### 2. `src/flashmlx/compaction/solvers.py`

**验证点**:
1. ✅ `nnls_pgd` 已实现（已验证 quality=1.000）
2. ⚠️ 确认支持 `upper_bound` 参数
3. ⚠️ 确认支持 `max_iters` 参数

### 3. 测试文件

**新增测试**:
1. ✅ 验证 beta 计算与作者一致
2. ✅ 验证 C2 计算包含 beta
3. ✅ 端到端质量测试（cosine similarity ≥ 0.99）

---

## 🚀 实施步骤

### Phase 1: P0 修复（必须）

1. **修改 Beta 计算**:
   - 文件: `compaction_algorithm.py:143-202`
   - 替换为 global NNLS 方法
   - 预计时间: 30 分钟

2. **修改 C2 计算**:
   - 文件: `compaction_algorithm.py:260`
   - 添加 `+ beta` 到 scores
   - 预计时间: 10 分钟

3. **验证测试**:
   - 运行 `test_highest_attention_keys.py`
   - 确认 quality ≥ 0.99
   - 预计时间: 10 分钟

### Phase 2: P1 增强（建议）

1. **添加 NNLS 参数**:
   - `nnls_upper_bound`
   - 预计时间: 15 分钟

2. **添加文档注释**:
   - 说明 fp32 提升策略
   - 预计时间: 10 分钟

### Phase 3: P2 可选（低优先级）

1. **添加 rms score method**
2. **添加 pooling 选项**

---

## ✅ 验收标准

### 必须通过:
- [ ] Beta 计算使用 global NNLS（partition function matching）
- [ ] C2 计算包含 beta 项
- [ ] 端到端测试 cosine similarity ≥ 0.99
- [ ] 与作者实现逻辑一致（环境差异除外）

### 可选通过:
- [ ] 支持 rms score method
- [ ] 支持 pooling
- [ ] 完整文档注释

---

## 📊 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| MLX 数值行为与 PyTorch 不同 | HIGH | 添加数值精度测试 |
| Ridge regression 实现差异 | MEDIUM | 对比作者输出验证 |
| NNLS solver 精度不足 | LOW | 已验证 quality=1.000 ✅ |

---

## 📚 参考资料

1. **论文**: Fast KV Compaction via Attention Matching (2026)
2. **作者代码**: https://github.com/adamzweiger/compaction
3. **FlashMLX 实现**: `src/flashmlx/cache/compaction_algorithm.py`
4. **深度分析**: `.solar/beta-computation-deep-analysis.md`
5. **实现对比**: `.solar/implementation-comparison-report.md`

---

*移植计划版本: v1.0*
*创建时间: 2026-03-22*
*原则: 完整移植，只改环境*
