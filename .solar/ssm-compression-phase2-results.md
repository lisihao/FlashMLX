# SSM State 压缩 Phase 2 测试结果

**日期**: 2026-03-21 15:00
**模型**: Qwen3.5-35B-A3B (6-bit)
**测试范围**: 全部 30 个 SSM 层

---

## 测试结果总览

| 方法 | 压缩比 | 速度 | 输出质量 | 最终评价 |
|------|--------|------|----------|----------|
| **Baseline** | 1.0x | 25.57 tok/s | ✅ 正常 | 基准 |
| **Quantization (8-bit)** | 2.00x | 46.04 tok/s | ❌ **完全乱码** | **失败** |
| **Low-Rank (rank=32)** | 2.39x | **1.35 tok/s** | ✅ 正常 | ⚠️ **成功但极慢 (19x slower)** |
| **Random Projection (dim=32)** | 6.00x | 39.46 tok/s | ❌ **重复乱码** | **失败** |

---

## 1. Quantization (8-bit) - **完全失败** ❌

### 测试配置
- Bits: 8-bit
- Quantization: Per-head (每个 head 独立 scale/zero_point)
- 压缩比: 2.00x

### 输出样例

**Baseline**:
```
机器学习（Machine Learning, ML）是人工智能（AI）的核心分支之一...
```

**Quantization (8-bit)**:
```
。足球俱乐部除草健康 😀chetto Especi时代.lot Tent revan�ل.和别人ل发展和改革 certif和别人 Verka denun家家家。ai。esz一

猜 ` ` Austral招 escal دي error惊恐只 escal【 ship （
```

### 失败原因分析

1. **SSM State 对量化误差极其敏感**
   - 即使 8-bit per-head quantization 误差 ~0.008（Phase 1 测试）
   - 在真实 SSM 递推中累积导致完全崩溃
   - 输出变为随机 token 组合（多语言混杂、emoji、乱码）

2. **递推放大效应**
   - SSM state 通过递推更新：`state[t+1] = f(state[t], input[t])`
   - 量化误差在每步递推中累积和放大
   - 30 个 SSM 层串联 → 误差指数级增长

3. **Phase 1 vs Phase 2 差异**
   - Phase 1: 随机数据单步测试 → 误差 0.008（看似可行）
   - Phase 2: 真实递推多步生成 → 质量完全崩溃
   - **教训**: 静态误差 ≠ 递推稳定性

### 速度表现

- 46.04 tok/s（比 baseline 快，但没意义）
- 快的原因：模型在胡言乱语，提前终止了某些逻辑

---

## 2. Low-Rank (rank=32) - **质量正常但速度极慢** ⚠️

### 测试配置
- Rank: 32
- SVD: Per-head SVD，保留 top-32 奇异值
- 压缩比: 2.39x

### 输出样例

**Baseline**:
```
机器学习（Machine Learning, ML）是人工智能（AI）的核心分支之一，其核心思想是**让计算机从数据中自动学习规律，而无需进行显式的编程指令**。
```

**Low-Rank (rank=32)**:
```
<think>
Here's a thinking process that leads to the suggested response:

1.  **Understand the Goal:** The user wants an explanation of "what is machine learning"...
```

✅ **输出质量正常**：完整句子，语义连贯，逻辑正确。

### 性能问题 - **关键瓶颈** 🔥

| 指标 | Baseline | Low-Rank | 倍数 |
|------|----------|----------|------|
| 速度 | 25.57 tok/s | **1.35 tok/s** | **慢 19 倍** |
| 100 tokens 耗时 | 3.91s | **73.91s** | **慢 19 倍** |

### 为什么这么慢？

1. **SVD 计算开销巨大**
   - 每次 compress: 需要对 (Dv=128, Dk=192) 矩阵做 SVD
   - 64 个 heads × 30 个 SSM 层 = **1920 次 SVD**
   - SVD 必须在 CPU 上运行（MLX 限制）
   - 每次 SVD 需要 float32 转换（bfloat16 → float32 → bfloat16）

2. **解压缩开销**
   - 每个 token 生成时，需要解压缩所有 30 个 SSM state
   - 每个 state 解压缩：64 次矩阵乘法 `U @ diag(S) @ Vt`
   - 30 层 × 64 heads = 1920 次矩阵乘法/token

3. **CPU-GPU 数据传输**
   - SVD 在 CPU，主计算在 GPU
   - 频繁的 CPU-GPU 传输成为瓶颈

### 压缩比分析

**原始**: 1,572,864 elements/layer
**压缩**:
- U: 64 × 128 × 32 = 262,144
- S: 64 × 32 = 2,048
- Vt: 64 × 32 × 192 = 393,216
- **Total**: 657,408 elements/layer

**压缩比**: 1,572,864 / 657,408 = **2.39x**

---

## 3. Random Projection (dim=32) - **失败（重复乱码）** ❌

### 测试配置
- Target dim: 32
- Projection matrix: Gaussian random (Dk=192 → 32)
- 压缩比: 6.00x (最高!)

### 输出样例

**Baseline**:
```
机器学习（Machine Learning, ML）是人工智能（AI）的核心分支之一...
```

**Random Projection (dim=32)**:
```
。

<think>
用户不不不
从
（（（））
eneg
 defini，从
从


 erville
 </think>
```

### 失败原因

1. **重建不精确**
   - 使用伪逆 `pinv` 重建：`state ≈ compressed @ pinv(proj_matrix)`
   - 伪逆只是最小二乘近似，不保证精确重建
   - SSM 对重建误差极其敏感

2. **维度压缩过激进**
   - Dk=192 → 32 (压缩 6x)
   - 丢失了太多信息，无法保持 SSM 递推的数值稳定性

3. **误差模式**
   - 重复字符（"不不不"、"从从"）
   - 乱码 token（"eneg"、"erville"）
   - 输出提前终止（31 tokens）

### 速度表现

- 39.46 tok/s（看起来快，但输出无效）

---

## 对比分析

### 误差敏感度排序

**SSM State 对压缩误差的敏感度**:

```
Quantization (8-bit) ❌
  ↓ 最敏感（即使 0.008 误差也崩溃）
Random Projection (dim=32) ❌
  ↓ 中等敏感（重建误差 0.728 导致重复乱码）
Low-Rank (rank=32) ✅
  ↓ 相对鲁棒（误差 0.534 但质量正常）
```

### 为什么 Low-Rank 成功而其他失败？

1. **理论保证**
   - Low-Rank SVD: 保留最重要的奇异值，误差有界
   - Quantization: 均匀误差，可能破坏关键数值
   - Random Projection: 伪逆近似，无精确重建保证

2. **误差分布**
   - Low-Rank: 误差集中在不重要的维度
   - Quantization: 误差均匀分布到所有维度
   - Random Projection: 误差不可控

3. **数值稳定性**
   - Low-Rank: SVD 分解本身数值稳定
   - Quantization: 量化可能破坏数值范围
   - Random Projection: 伪逆可能数值不稳定

---

## 速度分析

| 方法 | 速度 | 相对 Baseline | 主要开销 |
|------|------|--------------|----------|
| Baseline | 25.57 tok/s | 1.0x | - |
| Quantization | 46.04 tok/s | 1.8x (快，但无效) | 量化/反量化（极快） |
| **Low-Rank** | **1.35 tok/s** | **0.05x (慢 19 倍)** | **SVD + 矩阵乘法** |
| Random Projection | 39.46 tok/s | 1.5x (快，但无效) | 矩阵乘法 + pinv |

**Low-Rank 慢的本质原因**:
1. SVD 在 CPU 上运行（O(Dv × Dk²)）
2. 频繁的类型转换（bfloat16 ↔ float32）
3. 每个 token 生成时都需要解压缩（1920 次矩阵乘法）

---

## Phase 1 vs Phase 2 对比

### Phase 1: 随机数据单步测试

| 方法 | 压缩比 | 重建误差 | 结论 |
|------|--------|----------|------|
| Quantization (8-bit) | 2.00x | **0.008** ⭐ | "最佳候选" |
| Low-Rank (rank=32) | 2.39x | 0.534 | "备选方案" |
| Random Projection (dim=32) | 6.00x | 0.728 | "高压缩备选" |

### Phase 2: 真实模型递推生成

| 方法 | 压缩比 | 输出质量 | 速度 | 实际结论 |
|------|--------|----------|------|----------|
| Quantization (8-bit) | 2.00x | ❌ **完全乱码** | 46.04 tok/s | **完全失败** |
| **Low-Rank (rank=32)** | 2.39x | ✅ **正常** | **1.35 tok/s** | **唯一可用（但极慢）** |
| Random Projection (dim=32) | 6.00x | ❌ **重复乱码** | 39.46 tok/s | **失败** |

### 教训：静态误差 ≠ 递推稳定性

**Phase 1 的静态误差测试无法预测递推生成的质量**:
- Quantization: 静态误差最低 (0.008) → 递推完全崩溃
- Low-Rank: 静态误差中等 (0.534) → 递推质量正常

**根本原因**:
- SSM state 通过递推更新，误差会累积和放大
- 量化误差虽小但均匀分布，破坏数值稳定性
- Low-Rank 保留主成分，误差不影响关键维度

---

## 下一步行动 🎯

### Option A: 优化 Low-Rank 速度（推荐）

**目标**: 将 Low-Rank 速度从 1.35 tok/s 提升到接近 baseline (25 tok/s)

**优化方向**:

1. **缓存 SVD 结果**
   - 观察：SSM state 结构可能在生成过程中相对稳定
   - 方案：只在 state 变化较大时重新计算 SVD
   - 预期提升：5-10x

2. **增量更新 Low-Rank**
   - 使用增量 SVD 算法（Incremental SVD）
   - 不重新计算完整 SVD，而是更新 U/S/Vt
   - 预期提升：3-5x

3. **混合策略**
   - 只压缩 "冷" SSM 层（不常更新的层）
   - "热" 层保持未压缩
   - 预期提升：根据冷热比例

4. **更高 Rank**
   - 尝试 rank=64 或 rank=96
   - 牺牲压缩比换取更快的重建速度
   - 预期：压缩比 1.5x → 速度提升 2-3x

### Option B: 探索其他压缩方法

1. **4-bit Quantization**
   - 更激进的量化（8-bit → 4-bit）
   - 但 8-bit 已经失败，4-bit 成功概率极低

2. **混合方法**
   - 对不同层使用不同方法
   - 例如：前 10 层 Low-Rank，后 20 层 Quantization（如果能修复）

3. **学习型压缩**
   - 训练一个小型 Autoencoder 压缩 SSM state
   - 但需要训练数据和额外计算

### Option C: 接受现状（不推荐）

- Low-Rank 压缩比只有 2.39x，速度慢 19 倍
- 不如不压缩，直接使用 baseline

---

## 监护人决策点 🤔

**请监护人选择下一步方向**:

- [ ] **Option A**: 优化 Low-Rank 速度（推荐尝试）
- [ ] **Option B**: 探索其他压缩方法（风险较高）
- [ ] **Option C**: 放弃 SSM 压缩，回到 Task #55/56（诊断 Attention 层问题）
- [ ] **Option D**: 其他方向（请说明）

**关键问题**:
1. 2.39x 压缩比是否值得优化？
2. 能接受的最低速度是多少？(目前 1.35 tok/s，需要提升到多少？)
3. 如果 Low-Rank 优化失败，是否继续探索其他方法？

---

*Phase 2 测试报告创建于: 2026-03-21 15:00*
*作者: Solar (Task #53)*
*结论: Low-Rank 可行但极慢，需要优化或探索其他方向*
