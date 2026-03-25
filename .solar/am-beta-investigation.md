# Attention Matching Beta 计算调查报告

**日期**: 2026-03-22
**任务**: 集成作者 Attention Matching 代码到 FlashMLX

---

## 调查过程

### 1. 初始问题

NNLS 方法质量低（cos ≈ 0.7-0.8），远低于 log-ratio 方法（cos = 1.000）。

### 2. 发现的问题

#### 问题 A: NNLS 初始化错误（已修复）

**现象**:
- 当 t=T（无压缩）时，NNLS 返回稀疏解（9/20 非零）
- Beta 范围 [-27.63, 2.61] 而非预期的 0

**根因**:
- 初始化公式 `B_init = mean(y) / mean(M)` 假设 B 是标量，但实际是向量
- 当 t=T 时，正确解是 `B = ones(t)`，但从 `B=20*ones` 开始导致收敛到错误解

**修复**:
- 添加 t=T 的特殊处理：直接返回 `beta = zeros(t)`（跳过 NNLS）
- 改进 NNLS 初始化：检测是否为 full-rank 情况

**修复后效果**:
```python
# t=T (无压缩)
beta range: [0.000000e+00, 0.000000e+00]  ✅
Cosine similarity: 0.996  ✅ PASS (≥ 0.99)
```

#### 问题 B: t<T 的质量仍然很差（未解决）

**现象**:
```python
# t=5, T=20, n=3 (测试数据)
Beta values: [-27.631, 0.425, 1.337, 0.628, 1.721]
Cosine similarity: 0.640  ❌ FAIL
```

**NNLS 求解验证**:
- NNLS 迭代正常收敛：
  - 10 iters: loss = 0.899
  - 100 iters: loss = 0.001
  - 500+ iters: loss ≈ 0（完美匹配 M @ B = target）

**但**：尽管 NNLS 数学上正确（M @ B = target），最终输出质量仍然很差。

---

## 根本问题分析

### NNLS 目标 vs. 输出质量

**NNLS 优化目标**:
```
min ||M @ B - target||^2
其中:
  M = exp_scores[:, selected_indices]  # (n, t)
  target = sum(exp_scores, axis=1)     # (n,)
  B >= 0
```

**目标含义**: 匹配 partition function Z

**但**：匹配 partition function ≠ 匹配 attention output

**证据**:
```
M @ B = target  ✅ (loss ≈ 0)
attention_weights_comp ≈ attention_weights_orig  ❌ (L2 error = 0.616)
cosine_similarity(output_orig, output_comp)  ❌ (0.640)
```

### 可能的原因

1. **测试数据太小**:
   - T=20, t=5, n=3
   - 真实场景：T=几千, t=几百, n=几十/几百
   - 小数据可能不具代表性

2. **NNLS 目标不充分**:
   - 作者方法可能需要更大的 query 集合（n）来有效训练
   - 或者需要配合其他技巧（如 C2 的特殊处理）

3. **实现细节差异**:
   - 虽然 NNLS 求解器是直接移植的，但可能有其他细微差异
   - 需要端到端测试作者的原始代码

---

## 当前状态

### ✅ 已完成

1. **t=T 特殊处理**:
   - 直接返回 beta=0
   - 质量 0.996 ✅

2. **NNLS 求解器修复**:
   - 改进初始化（B=ones for full-rank, B=mean(y)/mean(M) for compression）
   - 验证收敛（loss ≈ 0）

3. **代码文件**:
   - `src/flashmlx/cache/compaction_algorithm.py`: 添加 t=T 特殊处理
   - `src/flashmlx/compaction/nnls_author.py`: 改进初始化
   - 测试脚本: `test_nnls_method.py`, `test_nocompression_debug.py`

### ❌ 未解决

1. **t<T 的质量问题**:
   - 当前质量: 0.640
   - 目标质量: ≥ 0.99
   - 根本问题: NNLS 目标（匹配 partition function）不足以保证输出质量

---

## 下一步建议

### 方案 A: 端到端测试作者代码

**目的**: 验证作者的实现在相同测试数据下的表现

**步骤**:
1. 在作者的代码库中运行相同的测试（T=20, t=5, n=3）
2. 对比质量指标
3. 如果作者代码也质量差 → 说明测试数据太小
4. 如果作者代码质量好 → 说明我们实现有差异

### 方案 B: 使用更大的测试数据

**目的**: 验证 NNLS 方法是否需要大规模数据才能有效

**步骤**:
1. 生成更大的测试数据（T=1000, t=100, n=100）
2. 测试质量
3. 如果质量改善 → 说明方法依赖数据规模
4. 如果质量仍差 → 说明实现有问题

### 方案 C: 放弃 NNLS，专注 log-ratio

**目的**: 如果 NNLS 方法确实不可行，使用已验证的 log-ratio 方法

**优点**:
- log-ratio 方法已验证质量 1.000
- 实现简单
- 不依赖 NNLS 求解器（避免 MLX GPU linalg 限制）

**缺点**:
- 与作者实现不一致
- 可能不是最优方法

---

## 技术细节

### 文件修改记录

1. **src/flashmlx/cache/compaction_algorithm.py**:
   ```python
   # Line 178-182: 添加 t=T 特殊处理
   if t == T:
       beta = mx.zeros((t,), dtype=K.dtype)
   else:
       # 原 NNLS 逻辑
   ```

2. **src/flashmlx/compaction/nnls_author.py**:
   ```python
   # Line 56-78: 改进初始化
   test_result = M @ mx.ones((t,), dtype=M.dtype)
   is_full_rank = float(mx.sum((test_result - y) ** 2)) < 1e-6 * float(mx.sum(y ** 2))

   if is_full_rank:
       B = mx.ones((t,), dtype=M.dtype)
   else:
       B_init_val = float(mx.mean(y) / (mx.mean(M) + 1e-12))
       B = mx.ones((t,), dtype=M.dtype) * max(B_init_val, min_val)
   ```

### 测试脚本

- `test_nnls_method.py`: 测试 NNLS beta 方法
- `test_nocompression_debug.py`: 调试 t=T 情况
- `test_nnls_gradient.py`: 验证梯度下降过程
- `test_compression_debug.py`: 调试 t<T 压缩情况

---

## 结论

**已完成**: t=T 的特殊处理，质量达标（0.996）

**未解决**: t<T 的 NNLS 方法质量不达标（0.640）

**建议**:
1. 先进行端到端测试（方案 A），验证作者代码的表现
2. 如果作者代码质量好，深入对比实现差异
3. 如果作者代码质量也差，使用更大测试数据（方案 B）
4. 如果问题无法解决，考虑使用 log-ratio 方法（方案 C）
