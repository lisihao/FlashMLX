# DoubleLayerKVCache 质量下降分析

**测试场景**: 846 tokens prefill + 100 tokens generate

---

## 问题现象

| Cache 策略 | 输出质量 | 输出内容 |
|-----------|---------|----------|
| **Baseline** | ✅ 正确 | "The breakthrough achievement in July 2022 was achieving stable quantum coherence at room temperature..." |
| **RotatingKVCache** | ❌ 乱码 | "[... [RE\n\n**. The lab, 202020..." |
| **DoubleLayerKVCache** | ❌ 错误 | "The\n\nThe story ends with Dr. Chen's lab open-source..." |

**关键差异**:
- Baseline 正确回答了问题（关于 2022 年 7 月的突破）
- DoubleLayerKVCache 跳到了文章末尾（关于开源和全球传播）

---

## 根因分析

### 1. 校准文件不匹配 ⚠️

```
Debug 输出: old_prefix_len=590, selected_calibration=L710, budget=384
```

**问题**:
- L710 校准文件是为 **710 tokens** 设计的
- 实际 old_prefix 只有 **590 tokens**
- 虽然有 dynamic clipping，但 L710 的 selected_indices 分布可能不适合 590 tokens

**为什么不合适**:
```
L710 的 indices 分布（假设）:
[0-100]: 稀疏（文章开头）
[100-500]: 密集（文章中间）
[500-710]: 稀疏（文章末尾）

但 590 tokens 的实际分布可能是:
[0-100]: 应该密集（问题相关）
[100-500]: 中等密集
[500-590]: 无（不存在）
```

L710 的 indices 在 [500-710] 范围内会被 clipped，导致 **末尾区域完全丢失**。

---

### 2. 压缩比过高 ⚠️

```
Compression stats:
- old_prefix: 215 tokens (压缩后)
- compressions: 2
- avg ratio: 1.44x
```

**计算实际压缩比**:
```
第一次压缩: 847 tokens → split at 591 → old_prefix = 591
第二次压缩: 591 tokens (经过 L710 calibration) → 215 tokens

实际压缩比: 591 / 215 = 2.74x (远高于目标 2.0x)
```

**为什么会这样**:
- L710 的 budget = 384 (为 710 tokens 设计)
- 经过 dynamic clipping 后，valid indices < 591
- 最终只保留了 215 个 indices

**问题**: 压缩比 2.74x 太激进，丢失了关键信息。

---

### 3. Recent Window 太小 ⚠️

```
Recent window: 256 tokens (固定)
Total length: 846 tokens
Recent 占比: 256 / 846 = 30.3%
```

**问题**:
- 只保留最近 30% 的上下文
- 问题 "Question: What was the breakthrough achievement in July 2022?" 在文本末尾
- 但答案的关键信息在文本中间（"July 15, 2022, at 3:47 AM"）
- Recent window 可能只覆盖了问题，没有覆盖答案

---

### 4. 多次压缩的累积误差 ⚠️

```
compressions: 2
```

**流程**:
```
Prefill 846 tokens
    ↓
Generate 1st token → 847 tokens → 超过 threshold (300)
    ↓
第一次压缩: 847 → [215 compressed] + [256 recent] = 471
    ↓
Generate more tokens → 471 + N → 超过 threshold?
    ↓
第二次压缩: ... → [215 compressed] + [256 recent] = 471 (稳定)
```

**问题**: 每次压缩都会引入误差，两次压缩累积误差更大。

---

## 关键信息丢失分析

**问题在文本中的位置**:
```
[0-500]: 背景介绍（Dr. Chen, team, early work）
[500-600]: ✅ 核心答案：July 15, 2022 breakthrough
[600-846]: 后续发展（open-source, global spread, question）
```

**DoubleLayerKVCache 的处理**:
```
old_prefix [0-590]: 压缩到 215 tokens
    - 可能丢失了 [500-600] 的关键信息
    - 因为 L710 calibration 的 indices 分布不匹配
recent_window [591-846]: 精确保留 256 tokens
    - 只保留了问题和后续发展
    - 没有保留核心答案
```

**结果**: 模型只看到 "open-source" 和 "global spread"，所以回答了 "The story ends with..."

---

## 解决方案

### 方案 1: 生成更匹配的校准文件 ✅ 推荐

**当前校准**:
```
L249, L466, L710, L944, L1403, L1863
```

**问题**: L466 → L710 跨度太大（244 tokens），590 tokens 距离 L710 太远。

**改进**: 生成更密集的校准文件：
```
L300, L400, L500, L600, L700, L800, L1000, L1200, L1500, L2000
```

**优势**:
- L600 更接近 590 tokens
- indices 分布更适合实际长度

---

### 方案 2: 降低压缩比

**当前**: compression_ratio = 2.0

**改进**: compression_ratio = 1.5

**效果**:
- 590 tokens → 393 tokens (vs 当前 215 tokens)
- 保留更多信息，减少丢失风险

---

### 方案 3: 增加 Recent Window

**当前**: recent_window_size = 256

**改进**: recent_window_size = 512

**效果**:
- 覆盖 [335-846] 范围（vs 当前 [591-846]）
- 更可能包含核心答案区域 [500-600]

---

## 推荐策略

**综合方案** (三管齐下):
1. ✅ 生成密集校准文件 (L300-L2000, 步长 100-200)
2. ✅ 降低压缩比至 1.5x
3. ✅ 增加 recent window 至 512

**预期效果**:
- 更精确的 calibration 选择
- 更保守的压缩（保留更多信息）
- 更大的 recent window（覆盖更多关键区域）

---

## 数学验证

**假设**: 590 tokens old_prefix, compression_ratio=1.5, recent_window=512

```
第一次压缩:
- 847 tokens → split at (847 - 512) = 335
- old_prefix = 335 tokens
- 使用 L400 calibration (最接近)
- budget = 335 / 1.5 = 223 tokens

最终 cache:
- old_prefix: 223 tokens (保留了 [0-335] 的 67%)
- recent_window: 512 tokens (精确保留 [335-846])
- total: 735 tokens (vs 当前 471 tokens)

覆盖核心答案区域 [500-600]:
- [500-600] 在 recent_window [335-846] 中 ✅ 完全覆盖
```

**结论**: 新方案应该能正确保留答案区域。
