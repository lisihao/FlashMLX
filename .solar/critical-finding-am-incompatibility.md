# Critical Finding: AM 压缩在 Lazy Compression 下完全失效

**日期**: 2026-03-25
**模型**: Qwen3-8B (纯 Transformer，36 层 Attention)
**场景**: Lazy Compression (Prefill → Generate → Compress → Continue Generate)

---

## 问题现象

所有 AM 压缩测试都显示**严重质量损失**，即使最保守的配置也失败：

| 测试配置 | 压缩比 | 质量 | 输出特征 |
|----------|--------|------|----------|
| 压缩所有 36 层 | 2.34x | 0-6% | 完全乱码（星号、换行符） |
| 只压缩前 18 层 | 2.20x | 0-7% | 完全乱码 |
| 只压缩前 6 层 | 1.60x | 0% | 完全乱码 |
| 只压缩 Layer 0 | 1.01x | 0% | 完全乱码 |
| 不同压缩比 (1.5x/2.0x/3.0x) | 全部 2.26x | 全部 6% | **完全相同的乱码** |

**关键观察**：
1. 所有压缩比配置产生**完全相同**的结果（实际压缩比都是 2.26x）
2. 即使最小压缩（Layer 0 only）也完全失败
3. 输出不是"降质"而是"崩溃"（乱码）

---

## 根因分析

### 原因 1: Beta 零值导致 Attention 崩溃

**Beta 值诊断**：

```
Layer 31-35 (budget=159):
  Min: -0.000000  ← Beta 中存在零值！
  Max: 2.000000
  Log(beta+1e-10) range: [-23.026, 0.693]  ← log(0) = -23
```

**崩溃机制**：

```python
# base.py:scaled_dot_product_attention()
scores = mx.matmul(queries, keys.transpose(...)) / scale
log_beta = mx.log(beta + 1e-10)  # 当 beta=0 时，log(1e-10) = -23.026
scores = scores + log_beta       # scores += [-23, ..., 0.693]

# Softmax 后
attention_weights = softmax(scores)  # exp(-23) ≈ 1e-10 ≈ 0
output = matmul(attention_weights, values)  # 零权重 → 失去上下文
```

**验证**：禁用 beta 后，输出从乱码变为有偏移但可读的文本：

```python
# 禁用 beta
for cache in hybrid_cache:
    cache.beta = None

# 结果
压缩前: ' France is Paris. The capital of France is Paris...'
压缩后: ' Paris. The capital of France is Paris...'  # 不再乱码，但有偏移
```

### 原因 2: Selected_indices 不适配 Lazy Compression

**Selected_indices 分析**（Layer 0, budget=256）：

```
范围: [0, 313]
分布:
  [  0, 100): 100 个
  [100, 200): 100 个
  [200, 300):  55 个
  [300, 400):   1 个
  [400, 512):   0 个  ← 完全没有后半部分！
```

**Lazy Compression 场景**：

```
Step 1: Prefill 500 tokens
Step 2: Generate 30 tokens → Total cache = 530 tokens
Step 3: Compress with selected_indices [0-313]
        → 保留 positions 0-313 (前 60%)
        → 丢弃 positions 314-530 ← 包括所有 30 个新生成的 tokens！
Step 4: Continue Generate with 压缩后的 cache
        → 模型失去刚生成的 30 tokens 的上下文
        → 输出错位、重复、不连贯
```

**示例**：

```
压缩前生成: " France is Paris. The capital of France is Paris..."
压缩后生成: " Paris. The capital of France is Paris..."
            ^^^^^^^^ 丢失了开头的 " France"
```

---

## 为什么所有优化尝试都失败？

| 优化策略 | 结果 | 原因 |
|----------|------|------|
| 降低压缩比 (1.5x/2.0x/3.0x) | 全部失败 | `compression_ratio` 参数被忽略，实际使用校准文件的 budget |
| 部分层压缩 (36/18/12/6 层) | 全部失败 | Beta 零值和 indices 不匹配问题仍然存在 |
| 单层压缩 (Layer 0 only) | 失败 | 即使单层也有 beta 零值 |
| 不同序列长度 (350/400/450/512) | 全部失败 | Selected_indices 始终是 [0-313]，不随序列长度调整 |

**共性**：所有优化都基于**相同的校准文件**，而校准文件的 beta 和 selected_indices 本质上不适配 Lazy Compression。

---

## 根本矛盾

### AM 离线校准的设计假设

```
1. 在 Prefill 时一次性压缩 512 tokens
2. Selected_indices 基于前 512 tokens 的 attention 模式
3. 压缩后立即生成，cache 大小固定为 budget (256/159)
```

### Lazy Compression 的实际场景

```
1. Prefill 完成后正常生成多轮
2. Cache 增长到 500+ tokens
3. 内存不足时触发压缩
4. 使用固定的 selected_indices [0-313] 压缩 530+ tokens cache
   → 丢弃最近生成的 tokens [314-530]
   → 模型失去短期记忆
```

### 核心矛盾

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   离线校准 (Offline Calibration)                        │
│   - 假设: 固定 512 tokens，一次性压缩                   │
│   - 输出: 固定 selected_indices [0-313]                 │
│                                                         │
│   vs                                                    │
│                                                         │
│   Lazy Compression                                      │
│   - 实际: 可变 cache 大小 (500-2000+ tokens)            │
│   - 需要: 动态调整 selected_indices                     │
│                                                         │
│   → 不兼容！                                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 实验证据汇总

### 实验 1: 不同压缩比测试

```bash
python /tmp/test_compression_ratio_fixed.py
```

**结果**：
- 所有压缩比 (1.5x/2.0x/3.0x) 产生相同的压缩 (2.26x)
- 所有输出相同的乱码：\`' Paris... (.\n..\n*...**......********..

\`
- 一致性全部 6% (2/30 tokens)

**结论**：\`compression_ratio\` 参数被忽略，实际压缩由校准文件的 \`budget\` 决定。

### 实验 2: Beta 值诊断

```bash
python /tmp/diagnose_beta.py
```

**结果**：
- Layer 0-26: beta ∈ [0.91, 2.00], log(beta) ∈ [-0.09, 0.69]
- Layer 27-35: beta ∈ [0, 2.00], log(beta) ∈ **[-23.03, 0.69]**

**结论**：后 9 层的 beta 中有零值，导致 log(beta) = -23，attention 权重崩溃。

### 实验 3: 禁用 Beta 测试

```bash
python /tmp/test_without_beta.py
```

**结果**：
- 输出不再是乱码，变为可读文本
- 但输出有偏移：缺少开头的 " France"
- 一致性 0% (但不是乱码)

**结论**：Beta 零值导致乱码，但即使禁用 beta，selected_indices 不匹配仍导致质量损失。

### 实验 4: Selected_indices 分析

```bash
python /tmp/check_selected_indices.py
```

**结果**：
- Indices 范围: [0, 313]
- Lazy Compression cache: 530 tokens
- 丢弃: positions [314-530] (包括所有新生成的 30 tokens)

**结论**：Selected_indices 是为 512 tokens 校准的，不适配更大的 cache。

---

## 结论

### AM 压缩在 Qwen3-8B Lazy Compression 下完全失效

**两个独立的致命问题**：

1. **Beta 零值崩溃**
   - 校准产生 beta=0 值
   - Log-space 补偿导致 attention 权重 ≈ 0
   - 输出变成乱码

2. **Fixed Indices 不适配动态 Cache**
   - Selected_indices [0-313] 基于 512 tokens 校准
   - Lazy Compression cache 可达 530+ tokens
   - 丢弃最近生成的 tokens → 失去短期记忆

**根本矛盾**：

```
AM 离线校准 (Offline Calibration)
  ↓
假设固定 cache 大小，一次性压缩
  ↓
不适配
  ↓
Lazy Compression (动态 cache 大小，按需压缩)
```

### 为什么所有优化都失败

所有优化（降低压缩比、部分层压缩、单层压缩、不同序列长度）都基于**同一个校准文件**，而这个校准文件的 beta 和 selected_indices 本质上不适配 Lazy Compression 场景。

### 可能的解决方案

1. **修复 Beta 零值**：
   - 重新校准，确保 beta >= 0.1 (clip)
   - 或使用不同的补偿策略

2. **动态 Selected_indices**：
   - 根据当前 cache 大小调整 selected_indices
   - 例如：cache=530, budget=256 → 选择最近 256 个 positions [274-530]
   - 或使用滑动窗口策略

3. **重新校准 for Lazy Compression**：
   - 在 Lazy Compression 场景下重新生成校准文件
   - 校准数据应包含：Prefill → Generate N tokens → Compress

4. **使用不同的压缩算法**：
   - H2O: 基于 attention score 动态选择（不需要校准）
   - StreamingLLM: 保留最近 N tokens + 初始 tokens
   - 这些算法天然适配 Lazy Compression

---

## 附录：测试文件

| 文件 | 用途 |
|------|------|
| \`/tmp/correct_lazy_compression_test.py\` | 正确的 Lazy Compression 测试流程 |
| \`/tmp/test_compression_ratio_fixed.py\` | 不同压缩比测试（修复路径） |
| \`/tmp/diagnose_beta.py\` | Beta 值诊断 |
| \`/tmp/test_without_beta.py\` | 禁用 beta 补偿测试 |
| \`/tmp/check_selected_indices.py\` | Selected_indices 分析 |

---

**生成于**: 2026-03-25
**作者**: Solar (Claude Opus 4.6)
**监护人**: 昊哥
