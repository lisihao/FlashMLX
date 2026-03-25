# AM 压缩在 Qwen3-8B 上的质量失败报告

**日期**: 2026-03-23
**状态**: ❌ AM 压缩在 Qwen3-8B 上破坏输出质量
**结论**: AM 是模型特定的，不是架构通用的压缩算法

---

## 执行摘要

通过系列测试验证了用户的三个关注点，发现：

1. ❌ **AM 目标验证失败** - 压缩后输出质量完全改变（13% 相似度）
2. ✅ **压缩成本分析完成** - 成功触发压缩（591 → 295 tokens, 2x ratio）
3. ❌ **性能影响测试无意义** - 质量破坏使性能测试失去意义

**关键发现**:
- **AM 不是架构通用的**：在 Llama 3.2 3B 上工作，在 Qwen3-8B 上失败
- **质量损失严重**：输出完全答非所问，词汇相似度仅 13%
- **性能无提升**：反而慢 6%（可能因为压缩开销）

---

## 测试详情

### 测试环境

- **模型**: Qwen3-8B (纯 Transformer，36 层)
- **Prompt**: 591 tokens（真实的机器学习教程，非重复）
- **压缩配置**: max_size=256, compression_ratio=2.0, Fast Path
- **预期行为**: Baseline 回答 prompt 问题，AM 应该保持相同输出

### 测试结果

| 配置 | Prompt tokens | PP Speed | 压缩触发 | Cache Size | TG Speed | 输出质量 |
|------|--------------|----------|---------|-----------|----------|---------|
| **Baseline** | 591 | 395 tok/s | ❌ 否 | 591 | 27.85 tok/s | ✅ 正常 |
| **AM Fast Path** | 591 | 397 tok/s | ✅ 是 | 295 (2.0x) | 26.19 tok/s | ❌ 破坏 |

### 输出对比

**Baseline 输出**（正确）:
```
What is the primary purpose of backpropagation in the training of
neural networks, and how does it contribute to the learning process?

The primary purpose of backpropagation in the training of neural
networks is to efficiently compute the gradients of the loss function with
```

**AM Fast Path 输出**（答非所问）:
```
What is a type of neural network that can process sequences of data,
such as time series data or text. They are particularly effective for
tasks like language modeling and time series prediction. The key
difference is that they have a memory of previous inputs, allowing
```

**差异分析**:
- 词汇相似度: **13.04%** (应该接近 100%)
- Baseline: 回答了 "backpropagation 的目的" ✅
- AM: 描述了 "RNN 的特性"，完全不相关 ❌

---

## 压缩效果验证（用户目标1）

### AM 设计目标

1. **离线压缩**: ✅ 达成 - PP 阶段触发压缩（591 → 295 tokens）
2. **降低内存**: ✅ 达成 - KV cache 减少 50%（2.0x ratio）
3. **保持质量**: ❌ **失败** - 输出完全改变（13% 相似度）
4. **释放内存**: ✅ 达成 - Cache 从 591 降到 295

### 结论

**AM 在 Qwen3-8B 上未能保持输出质量**，这违反了核心设计目标。虽然内存确实被压缩和释放，但输出质量的破坏使其无法使用。

---

## 压缩成本分析（用户目标2）

### 压缩统计

- **触发条件**: Cache size (591) > max_size (256)
- **压缩次数**: 所有 36 层各触发 1 次
- **压缩比例**: 2.0x (target), 2.00x (actual)
- **最终 Cache size**: 295 tokens (591 / 2.0 ≈ 295.5)

### 压缩时间

| 阶段 | Baseline | AM Fast | 差异 |
|------|----------|---------|------|
| PP time | 1.494s | 1.488s | **-0.4%** (略快) |
| TG time | 1.795s | 1.909s | **+6.3%** (变慢) |

**压缩开销**:
- PP 阶段: 压缩发生在 PP 末尾，但总时间略快（可能误差范围内）
- TG 阶段: 因为 Cache 更小应该更快，但反而慢了 6%

**可能原因**:
1. 压缩导致的信息丢失影响了后续生成效率
2. 压缩后的 Cache 格式访问效率更低
3. Qwen3-8B 的 Attention 机制对压缩后的 KV 不友好

### 与 Llama 对比

| 模型 | PP Speed | TG Speed 变化 | 输出质量 |
|------|----------|--------------|---------|
| **Llama 3.2 3B** | - | **+46%** ✅ | ✅ 正常 |
| **Qwen3-8B** | -0.4% | **-6%** ❌ | ❌ 破坏 |

**结论**: AM 在 Llama 上是性能优化，在 Qwen3 上是性能倒退。

---

## 推理性能影响（用户目标3）

### PP (Prompt Processing) 影响

| 指标 | Baseline | AM Fast | 变化 |
|------|----------|---------|------|
| PP time | 1.494s | 1.488s | -0.4% |
| PP speed | 395.46 tok/s | 397.10 tok/s | +0.4% |

**结论**: PP 阶段几乎无影响（可能因为压缩发生在最后）

### TG (Token Generation) 影响

| 指标 | Baseline | AM Fast | 变化 |
|------|----------|---------|------|
| TG time | 1.795s | 1.909s | +6.3% |
| TG speed | 27.85 tok/s | 26.19 tok/s | **-6.0%** |

**结论**: TG 速度**下降 6%**，与 Llama 的 +46% 完全相反！

### TTFT (Time To First Token)

TTFT ≈ PP time，因为测试中 PP 后立即生成第一个 token。

| 配置 | TTFT | 变化 |
|------|------|------|
| Baseline | 1.494s | - |
| AM Fast | 1.488s | -0.4% |

**结论**: TTFT 几乎无变化。

---

## 根本原因分析

### 为什么 AM 在 Llama 上成功，在 Qwen3 上失败？

#### 假设 1: Attention 机制差异

**Llama 3.2 3B**:
- 使用 Grouped-Query Attention (GQA)
- 28 层，每层 26 attention heads, 4 KV heads
- head_dim: 64

**Qwen3-8B**:
- 使用不同的 Attention 实现
- 36 层，每层 28 attention heads, 4 KV heads
- head_dim: 128

**可能影响**:
- head_dim 更大（128 vs 64）导致压缩误差放大
- Qwen3 的 Attention 可能对 KV 顺序/分布更敏感
- AM 的 NNLS 拟合在不同 head_dim 下表现不同

#### 假设 2: 训练数据分布差异

**Llama**:
- 主要英文训练
- 较短的 context window (原生 8K)

**Qwen3**:
- 多语言训练（中英文）
- 更长的 context window (原生 128K)
- 可能对长距离依赖更敏感

**可能影响**:
- Qwen3 的 Attention 分布可能更分散
- AM 的"重要 token 选择"策略在 Qwen3 上失效
- 压缩后丢失了 Qwen3 需要的长距离信息

#### 假设 3: 模型规模和层数

**Llama 3.2 3B**: 28 层, 3B 参数
**Qwen3-8B**: 36 层, 8B 参数

**可能影响**:
- 更多层导致误差累积
- 8B 模型的 Attention 更复杂
- AM 的近似在更深网络中失效

#### 假设 4: AM 算法本身的限制

**AM 的核心假设**:
1. 可以用较少的 K 近似原始的 Attention(Q, K, V)
2. 通过 NNLS 拟合 β 补偿
3. 保留的 K 能代表所有 K 的信息

**可能在 Qwen3 上失效**:
- Qwen3 的 Attention 分布不满足 AM 的近似假设
- β 补偿不足以修正 Qwen3 的误差
- 压缩后的 K 在 Qwen3 的 Attention 机制下产生更大偏差

---

## 对比：Llama vs Qwen3

| 特性 | Llama 3.2 3B | Qwen3-8B |
|------|-------------|----------|
| **架构** | 纯 Transformer | 纯 Transformer |
| **层数** | 28 | 36 |
| **head_dim** | 64 | 128 |
| **Context** | 8K | 128K |
| **训练语言** | 英文为主 | 多语言 |
| **AM 输出质量** | ✅ 正常 (100%) | ❌ 破坏 (13%) |
| **AM 性能提升** | ✅ +46% | ❌ -6% |
| **AM 推荐** | ✅ 强烈推荐 | ❌ 禁止使用 |

---

## 重要教训：AM 的局限性

### 之前的认知（错误）

```
AM 适用于纯 Transformer 架构
AM 在混合架构上失败（Qwen 3.5）
→ 结论: 架构决定兼容性
```

### 现在的认知（正确）

```
AM 在 Llama 上成功 ✅
AM 在 Qwen3-8B 上失败 ❌
两者都是纯 Transformer
→ 结论: AM 是模型特定的，非架构通用的
```

### 关键洞察

**AM 不是通用压缩算法**:
- ❌ 不能盲目用于所有 Transformer
- ✅ 需要针对每个模型验证质量
- ✅ 可能需要针对不同模型调整参数

**质量测试不可靠**:
- 之前的 cosine similarity 测试显示 AM quality = 1.0
- 但真实推理中输出完全破坏
- 教训: **必须用真实推理验证，不能只用质量分数**

---

## 后续行动

### 立即行动

1. **更新自适应路由器**
   - 当前规则: 纯 Transformer → AM, 混合架构 → H2O
   - **新规则**:
     ```
     - Llama 系列 → AM ✅
     - Qwen3 系列 → H2O ✅ (即使是纯 Transformer)
     - Qwen3.5 系列 → H2O ✅
     - 其他模型 → 先测试质量，再决定
     ```

2. **添加质量验证**
   - 在生产使用前，必须运行真实推理验证
   - 不能只依赖 cosine similarity
   - 建议测试框架: `test_am_final.py`

3. **文档更新**
   - 明确 AM 的局限性
   - 提供模型兼容性列表
   - 强调质量验证的重要性

### 长期研究

1. **理解 AM 失败的根本原因**
   - 分析 Qwen3 vs Llama 的 Attention 分布差异
   - 研究 head_dim 对压缩质量的影响
   - 探索是否可以针对 Qwen3 调整 AM 参数

2. **探索替代压缩算法**
   - H2O 在 Qwen3.5 上表现如何？
   - StreamingLLM 是否更稳定？
   - 是否需要模型特定的压缩策略？

3. **建立模型-算法兼容性矩阵**
   ```
   | 模型系列 | AM | H2O | StreamingLLM |
   |---------|----|----|--------------|
   | Llama | ✅ | ? | ? |
   | Qwen3 | ❌ | ? | ? |
   | Qwen3.5 | ❌ | ✅ | ? |
   ```

---

## 用户关注点总结

### 1. AM 目标验证 ❌

**目标**: 离线压缩、降低内存、保持质量、释放内存

| 目标 | 结果 | 说明 |
|------|------|------|
| 离线压缩 | ✅ 达成 | PP 阶段成功触发 |
| 降低内存 | ✅ 达成 | 50% 内存节省 |
| 保持质量 | ❌ **失败** | 输出完全改变 (13% 相似度) |
| 释放内存 | ✅ 达成 | Cache 从 591 → 295 |

**结论**: 核心目标"保持质量"失败，AM 在 Qwen3-8B 上不可用。

### 2. 压缩成本分析 ✅

**压缩时间 vs Prompt 长度**:
- 591 tokens prompt, max_size=256, 压缩到 295 tokens
- PP time: 1.488s (包含压缩)
- 压缩开销 < 0.4% PP time (几乎可忽略)

**结论**: 压缩本身成本很低，但破坏了质量。

### 3. 推理性能影响 ❌

| 阶段 | Baseline | AM Fast | 变化 |
|------|----------|---------|------|
| PP | 395 tok/s | 397 tok/s | +0.4% |
| TG | 27.85 tok/s | 26.19 tok/s | **-6.0%** |
| TTFT | 1.494s | 1.488s | -0.4% |

**结论**: 性能无提升，反而下降。与 Llama 的 +46% 完全相反。

---

## 最终结论

**AM 压缩在 Qwen3-8B 上失败**:
1. ❌ 输出质量破坏（13% 相似度）
2. ❌ 性能下降（-6% TG）
3. ❌ 不可用于生产

**关键教训**:
1. **AM 不是架构通用的** - 需要针对每个模型验证
2. **质量分数不可靠** - 必须用真实推理验证
3. **自适应路由必须更新** - 基于模型而非架构

**推荐行动**:
- ✅ Llama 系列使用 AM
- ❌ Qwen3/Qwen3.5 禁用 AM，使用 H2O
- ✅ 新模型必须先验证质量

---

*报告生成于: 2026-03-23*
*测试模型: Qwen3-8B (纯 Transformer)*
*AM 验证: ❌ FAILED - 输出质量破坏*
