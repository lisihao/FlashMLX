# Attention Matching 完整移植状态

> **原则铁律**: 论文+代码已证明正确，完整移植，只改环境差异（PyTorch→MLX）

---

## Mission
完整移植 Attention Matching 到 FlashMLX（作者实现 100% 照抄，只改 PyTorch→MLX API）

---

## Constraints
- **禁止改动算法逻辑** - 作者方法已证明正确
- **禁止"优化"** - 不准自作聪明改进
- **只改环境差异** - `torch` → `mx`，其他禁止修改
- **必须通过质量验证** - cosine similarity ≥ 0.99
- **数学逻辑 100% 一致** - Beta 计算、C2 计算、NNLS 求解

---

## Current Plan

### Phase 1: P0 修复（必须，约 50 分钟）

**1. 修改 Beta 计算** (30 min)
- 文件: `src/flashmlx/cache/compaction_algorithm.py:143-202`
- 动作:
  - ❌ 删除 log-ratio 方法（local 优化，错误）
  - ✅ 替换为 global NNLS（partition function matching，正确）
- 参考: `.solar/migration-plan-complete.md` Beta 计算移植细节

**2. 修改 C2 计算** (10 min)
- 文件: `src/flashmlx/cache/compaction_algorithm.py:260`
- 动作:
  - ❌ 当前: `scores_C1 = queries @ C1.T * scale`
  - ✅ 正确: `scores_C1 = queries @ C1.T * scale + beta`  # 添加 + beta
- 原因: C2 regression 必须在包含 beta 的 attention weights 上进行

**3. 验证测试** (10 min)
- 运行: `python -m pytest tests/compaction/test_highest_attention_keys.py -v`
- 预期: quality ≥ 0.99

### Phase 2: P1 增强（建议，约 25 分钟）

**1. 添加 NNLS 参数**
- 添加 `nnls_upper_bound` 参数
- 与作者保持一致

**2. 添加文档注释**
- 说明 fp32 提升策略（MLX 自动 vs PyTorch 显式）
- 说明环境差异

### Phase 3: P2 可选（低优先级）

**1. 添加 rms score method**
- 作者有，FlashMLX 缺少

**2. 添加 pooling 选项**
- 作者有 avgpool/maxpool
- FlashMLX 无

---

## Decisions

- [2026-03-22] **核心决策**: 完整移植作者实现，禁止"优化"或"改进"
- [2026-03-22] **Beta 计算**: 必须用 global NNLS（partition function matching）
- [2026-03-22] **C2 计算**: 必须包含 beta 项
- [2026-03-22] **环境适配**: 只改 PyTorch→MLX API，其他禁止修改
- [2026-03-22] **质量标准**: cosine similarity ≥ 0.99（作者论文标准）

---

## Progress

### Done
- ✅ NNLS solver 实现和验证（quality=1.000）
- ✅ 基础测试框架搭建
- ✅ 深度分析作者实现
  - `.solar/implementation-comparison-report.md` - 完整对比
  - `.solar/beta-computation-deep-analysis.md` - 数学分析
- ✅ 制定完整移植计划
  - `.solar/migration-plan-complete.md` - 逐项移植清单

### In-Progress
- 🔄 准备开始 Phase 1（等待监护人批准）

### Blocked
无

---

## Next Actions（精确到文件和行号）

### 1️⃣ 修改 Beta 计算 (30 min)

**文件**: `src/flashmlx/cache/compaction_algorithm.py`

**删除** (lines 143-202):
```python
# 当前的 log-ratio 方法（错误）
log_ratio = mx.log(target_attn / base_attn)
for j in range(t):
    beta[j] = nnls_pgd(ones, log_ratio[:, j])
```

**替换为**:
```python
# 作者的 global NNLS 方法（正确）
# Step 1: 计算 unnormalized attention scores
scores = queries @ K.T * inv_sqrt_d
max_scores = mx.max(scores, axis=1, keepdims=True)
exp_scores = mx.exp(scores - max_scores)

# Step 2: NNLS target = partition function
target = mx.sum(exp_scores, axis=1)  # (n,)

# Step 3: 设计矩阵 M
M = exp_scores[:, indices]  # (n, t)

# Step 4: 求解 NNLS
from ..compaction.solvers import nnls_pgd
B = nnls_pgd(M, target, lower_bound=1e-12, max_iters=100)

# Step 5: 转换到 log-space
beta = mx.log(B)
```

**参考**: `.solar/migration-plan-complete.md` 第 "Beta 计算移植细节" 部分

---

### 2️⃣ 修改 C2 计算 (10 min)

**文件**: `src/flashmlx/cache/compaction_algorithm.py`

**找到** (约 line 260):
```python
scores_C1 = queries @ C1.T * scale
```

**修改为**:
```python
scores_C1 = queries @ C1.T * scale + beta  # ⚠️ 添加 + beta
```

**原因**: C2 的 ridge regression 必须在"压缩后的 attention weights"上进行，而压缩后的 weights 定义为 `softmax(scores + beta)`

---

### 3️⃣ 运行验证 (10 min)

```bash
cd /Users/lisihao/FlashMLX

# 运行测试
python -m pytest tests/compaction/test_highest_attention_keys.py -v

# 预期输出:
# ✅ test_basic_compression - PASSED
# ✅ test_beta_quality - PASSED
# ✅ Cosine similarity ≥ 0.99
```

---

## 验收标准

### 必须通过 (Phase 1):
- [ ] Beta 计算使用 global NNLS（partition function matching）
- [ ] C2 计算包含 beta 项
- [ ] 端到端测试 cosine similarity ≥ 0.99
- [ ] 与作者实现逻辑一致（环境差异除外）

### 可选通过 (Phase 2/3):
- [ ] 支持 `nnls_upper_bound` 参数
- [ ] 支持 rms score method
- [ ] 支持 pooling
- [ ] 完整文档注释

---

## 参考资料

1. **论文**: Fast KV Compaction via Attention Matching (2026)
   - https://arxiv.org/abs/2602.16284

2. **作者代码**: https://github.com/adamzweiger/compaction
   - `compaction/algorithms/base.py` - 基类 + C2 计算
   - `compaction/algorithms/highest_attention_keys.py` - 主算法
   - `compaction/algorithms/optim.py` - NNLS solver

3. **FlashMLX 实现**:
   - `src/flashmlx/cache/compaction_algorithm.py` - 待修改
   - `src/flashmlx/compaction/solvers.py` - NNLS solver（已验证）

4. **分析报告**:
   - `.solar/beta-computation-deep-analysis.md` - 数学推导
   - `.solar/implementation-comparison-report.md` - 实现对比
   - `.solar/migration-plan-complete.md` - 移植清单

---

*创建时间: 2026-03-22*
*原则: 完整移植，只改环境*
*目标: 与作者实现 100% 一致*
