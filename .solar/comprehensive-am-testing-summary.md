# AM 压缩性能测试完整总结

**日期**: 2026-03-23
**测试模型**: Qwen3-8B (纯 Transformer, 36 层)
**状态**: ✅ 测试完成，发现关键问题

---

## 执行摘要

用户要求测试 AM 压缩在 Qwen3-8B 上的三个关注点：

1. **AM 目标验证** ❌ - 发现 AM 在 Qwen3-8B 上**破坏输出质量**
2. **压缩成本分析** ✅ - 成功测量压缩时间和计算成本
3. **推理性能影响** ❌ - 因质量破坏，性能测试失去意义

**关键发现**:
- **AM 不是架构通用的**：Llama 3.2 3B 成功，Qwen3-8B 失败
- **质量破坏严重**：输出词汇相似度仅 13%，完全答非所问
- **性能无提升**：TG 速度反降 6%（vs Llama 的 +46%）

**核心教训**:
```
AM 是模型特定的，不是架构通用的压缩算法！
仅基于架构类型（纯 Transformer vs 混合）选择算法是错误的。
必须基于模型系列（Llama vs Qwen3 vs Qwen3.5）选择。
```

---

## 用户关注点详细分析

### 1. AM 目标验证 ❌

**AM 的设计目标**:
1. 离线压缩：在 PP 阶段压缩 KV cache
2. 降低内存：减少 cache 占用
3. 保持质量：输出与 baseline 一致
4. 释放内存：实际减少内存使用

**测试结果** (max_size=256, ratio=2.0, 591 tokens prompt):

| 目标 | 结果 | 证据 |
|------|------|------|
| 离线压缩 | ✅ 达成 | PP 末尾触发，36 层各压缩 1 次 |
| 降低内存 | ✅ 达成 | Cache: 591 → 295 tokens (50% 减少) |
| 保持质量 | ❌ **失败** | **输出词汇相似度仅 13%，完全答非所问** |
| 释放内存 | ✅ 达成 | Cache size 从 591 降到 295 |

**输出对比**:

**Baseline** (正确回答):
```
What is the primary purpose of backpropagation in the training of
neural networks, and how does it contribute to the learning process?

The primary purpose of backpropagation in the training of neural
networks is to efficiently compute the gradients of the loss function with
```

**AM Fast Path** (完全答非所问):
```
What is a type of neural network that can process sequences of data,
such as time series data or text. They are particularly effective for
tasks like language modeling and time series prediction. The key
difference is that they have a memory of previous inputs, allowing
```

**分析**:
- Baseline 正确回答了 prompt 问题（"backpropagation 的目的是..."）
- AM 输出描述 RNN 的特性，与问题无关
- 词汇相似度：13.04%（应该接近 100%）

**结论**: **AM 在 Qwen3-8B 上违反了核心设计目标"保持质量"，无法使用**。

---

### 2. 压缩成本分析 ✅

**压缩时间 vs Prompt 长度**:

| Prompt tokens | max_size | 压缩触发 | 最终 Cache size | 压缩比例 |
|--------------|----------|---------|----------------|---------|
| 375 | 512 | ❌ 否 | 375 | 1.0x (无压缩) |
| 591 | 256 | ✅ 是 | 295 | 2.0x |
| 2374 | 256 | ✅ 是 | 791 | 3.0x |

**压缩开销分析**:

| 阶段 | Baseline | AM Fast | 差异 |
|------|----------|---------|------|
| PP time | 1.494s | 1.488s | **-0.4%** (略快，误差范围) |
| TG time | 1.795s | 1.909s | **+6.3%** (变慢) |

**关键发现**:
1. **PP 阶段几乎无开销**：压缩发生在 PP 末尾，总时间甚至略快（可能误差）
2. **TG 阶段反而变慢**：+6.3%，与预期的"cache 更小应该更快"相反
3. **压缩触发条件**：Cache size > max_size 时触发
4. **压缩比例精确**：实际比例与配置的 compression_ratio 一致

**与 Llama 对比**:

| 模型 | PP 影响 | TG 速度变化 | 结论 |
|------|--------|------------|------|
| **Llama 3.2 3B** | - | **+46%** ✅ | 性能优化 |
| **Qwen3-8B** | -0.4% | **-6%** ❌ | 性能倒退 |

**为什么 Qwen3-8B 上 TG 变慢？**

可能原因：
1. 压缩后的 Cache 访问效率更低（格式问题）
2. 信息丢失导致后续生成效率下降
3. Qwen3-8B 的 Attention 机制对压缩后的 KV 不友好

---

### 3. 推理性能影响 ❌

**PP (Prompt Processing) 影响**:

| 指标 | Baseline | AM Fast | 变化 |
|------|----------|---------|------|
| PP time | 1.494s | 1.488s | -0.4% |
| PP speed | 395.46 tok/s | 397.10 tok/s | +0.4% |

**结论**: PP 阶段几乎无影响（压缩发生在最后，对整体 PP 时间影响极小）。

**TG (Token Generation) 影响**:

| 指标 | Baseline | AM Fast | 变化 |
|------|----------|---------|------|
| TG time | 1.795s | 1.909s | +6.3% |
| TG speed | 27.85 tok/s | 26.19 tok/s | **-6.0%** |

**结论**: TG 速度**下降 6%**，与 Llama 的 +46% 完全相反！

**TTFT (Time To First Token) 影响**:

| 配置 | TTFT | 变化 |
|------|------|------|
| Baseline | 1.494s | - |
| AM Fast | 1.488s | -0.4% |

**结论**: TTFT 几乎无变化。

**性能总结**:
- ❌ AM 在 Qwen3-8B 上无性能提升，反而倒退
- ✅ Llama 上有显著提升（+46%）
- **AM 的性能提升是模型特定的，非架构通用的**

---

## 根本原因分析

### 为什么 AM 在 Llama 上成功，在 Qwen3 上失败？

#### 假设 1: Attention 机制差异

| 特性 | Llama 3.2 3B | Qwen3-8B |
|------|-------------|----------|
| Attention 类型 | Grouped-Query Attention (GQA) | 不同的 Attention 实现 |
| head_dim | 64 | **128** (2x) |
| 层数 | 28 | 36 |

**可能影响**:
- **head_dim 更大**（128 vs 64）→ 压缩误差放大
- Qwen3 的 Attention 对 KV 顺序/分布更敏感
- AM 的 NNLS 拟合在不同 head_dim 下表现不同

#### 假设 2: 训练数据分布差异

| 特性 | Llama | Qwen3 |
|------|-------|-------|
| 训练语言 | 英文为主 | 多语言（中英文） |
| Context window | 8K | **128K** (16x) |

**可能影响**:
- Qwen3 对**长距离依赖更敏感**
- AM 的"重要 token 选择"策略在 Qwen3 上失效
- 压缩后丢失了 Qwen3 需要的长距离信息

#### 假设 3: 模型规模

| 特性 | Llama 3.2 3B | Qwen3-8B |
|------|-------------|----------|
| 参数 | 3B | **8B** (2.7x) |
| 层数 | 28 | 36 |

**可能影响**:
- 更多层 → 误差累积
- 8B 模型的 Attention 更复杂
- AM 的近似在更深网络中失效

#### 假设 4: AM 算法的核心假设不成立

**AM 的核心假设**:
1. 可以用较少的 K 近似原始的 Attention(Q, K, V)
2. 通过 NNLS 拟合 β 补偿
3. 保留的 K 能代表所有 K 的信息

**可能在 Qwen3 上失效**:
- Qwen3 的 Attention 分布不满足 AM 的近似假设
- β 补偿不足以修正 Qwen3 的误差
- 压缩后的 K 在 Qwen3 的 Attention 机制下产生更大偏差

---

## 测试过程回顾

### 测试 1: 简短 Prompt（不触发压缩）

**配置**: 375 tokens prompt, max_size=512

**结果**: 压缩未触发，输出完全相同 ✅

**教训**: 必须用足够长的 prompt 触发压缩才能测试质量

### 测试 2: 重复 Prompt（触发压缩）

**配置**: 2374 tokens (重复 600 次 "Machine learning is a powerful technique"), max_size=256

**结果**: 压缩触发，但输出重复（"the system. the system..."）❌

**教训**: 重复 prompt 导致重复输出，无法准确评估质量

### 测试 3: 真实 Prompt（触发压缩）✅

**配置**: 591 tokens (真实的机器学习教程), max_size=256

**结果**: 压缩触发，输出完全答非所问（13% 相似度）❌

**教训**: 这是真实的质量破坏，不是 prompt 问题

---

## 重要教训

### 之前的认知（错误）

```
✅ 纯 Transformer → AM
❌ 混合架构 → H2O

结论：架构决定兼容性
```

### 现在的认知（正确）

```
✅ Llama 系列 → AM (质量 1.0, 速度 +46%)
❌ Qwen3 系列 → H2O (AM 破坏质量)
❌ Qwen3.5 系列 → H2O (混合架构，AM 崩溃)

结论：模型系列决定兼容性，架构类型不够
```

### 关键洞察

1. **AM 是模型特定的**
   - ❌ 不能盲目用于所有 Transformer
   - ✅ 需要针对每个模型验证质量
   - ✅ 可能需要针对不同模型调整参数

2. **质量测试不可靠**
   - 之前的 cosine similarity 测试显示 AM quality = 1.0
   - 但真实推理中输出完全破坏
   - **教训**: 必须用真实推理验证，不能只用质量分数

3. **性能提升也是模型特定的**
   - Llama: +46% ✅
   - Qwen3: -6% ❌
   - 不能假设 AM 在所有模型上都有性能提升

---

## 解决方案

### 立即行动 ✅

#### 1. 更新自适应路由器（已完成）

**V1 路由器** (错误):
```python
if architecture == 'pure_transformer':
    return 'AM'
elif architecture == 'hybrid':
    return 'H2O'
```

**V2 路由器** (正确):
```python
if model_series == 'llama':
    return 'AM'  # 验证成功
elif model_series == 'qwen3':
    return 'H2O'  # AM 破坏质量
elif model_series == 'qwen3.5':
    return 'H2O'  # 混合架构，AM 崩溃
else:
    return 'H2O'  # 保守选择
```

**测试结果**: 4/4 测试通过 ✅

#### 2. 添加质量验证

**测试脚本**: `benchmarks/test_am_final.py`

**用途**: 在生产使用前，运行真实推理验证

**检查项**:
- [ ] 压缩是否触发
- [ ] 输出是否与 baseline 一致
- [ ] 是否有重复模式
- [ ] 性能是否提升

#### 3. 文档更新（已完成）

- ✅ AM 质量失败报告: `.solar/am-qwen3-quality-failure-report.md`
- ✅ 自适应路由器 V2: `src/flashmlx/cache/adaptive_compressor_v2.py`
- ✅ 路由器测试: `benchmarks/test_adaptive_router_v2.py`

### 长期研究

1. **理解 AM 失败的根本原因**
   - 分析 Qwen3 vs Llama 的 Attention 分布差异
   - 研究 head_dim 对压缩质量的影响
   - 探索是否可以针对 Qwen3 调整 AM 参数

2. **探索替代压缩算法**
   - H2O 在 Qwen3 上的表现如何？
   - StreamingLLM 是否更稳定？
   - 是否需要模型特定的压缩策略？

3. **建立模型-算法兼容性矩阵**

   | 模型系列 | AM | H2O | StreamingLLM | 推荐 |
   |---------|----|----|--------------|------|
   | **Llama** | ✅ 1.0 (+46%) | ❓ 未测 | ❓ 未测 | **AM** |
   | **Qwen3** | ❌ 0.13 (-6%) | ❓ 未测 | ❓ 未测 | **H2O** |
   | **Qwen3.5** | ❌ 崩溃 | ✅ 0.69 | ✅ 0.66 | **H2O** |

---

## 模型-算法兼容性矩阵（当前状态）

| 模型系列 | 架构类型 | AM 质量 | AM 速度 | H2O 质量 | H2O 速度 | 推荐算法 |
|---------|---------|--------|--------|---------|---------|---------|
| **Llama 3.2 3B** | 纯 Transformer | ✅ 1.0 | ✅ +46% | ❓ 未测 | ❓ 未测 | **AM** |
| **Qwen3-8B** | 纯 Transformer | ❌ 0.13 | ❌ -6% | ❓ 未测 | ❓ 未测 | **H2O** |
| **Qwen3.5-0.8B** | 混合架构 | ❌ 崩溃 | ❌ 崩溃 | ✅ 0.69 | ❓ 未测 | **H2O** |
| **Qwen3.5-2B** | 混合架构 | ❌ 崩溃 | ❌ 崩溃 | ✅ 0.69 | ❓ 未测 | **H2O** |
| **Qwen3.5-35B** | 混合架构 | ❌ 崩溃 | ❌ 崩溃 | ✅ 0.69 | ❓ 未测 | **H2O** |

**图例**:
- ✅ 成功/推荐
- ❌ 失败/禁止
- ❓ 未测试

---

## 后续建议

### 必须做（高优先级）

1. **禁止在 Qwen3 系列上使用 AM**
   - 更新代码强制检查
   - 更新文档明确警告
   - 自适应路由器已更新（V2）

2. **测试 H2O 在 Qwen3-8B 上的表现**
   - 验证质量是否正常
   - 测量性能影响
   - 对比 AM 的失败案例

3. **为新模型建立验证流程**
   - 先运行质量测试（真实推理）
   - 再运行性能测试
   - 最后更新兼容性矩阵

### 可选做（中优先级）

1. **深入分析 AM 失败原因**
   - 对比 Llama 和 Qwen3 的 Attention 分布
   - 分析 head_dim 影响
   - 研究是否可以调整 AM 参数适配 Qwen3

2. **扩展兼容性矩阵**
   - 测试 Mistral, GPT, Gemma 等其他模型
   - 建立完整的模型-算法兼容性数据库

3. **优化 H2O 算法**
   - 如果 H2O 是 Qwen3 的最佳选择
   - 可以针对 Qwen3 优化 H2O 参数

---

## 文件清单

### 测试脚本

1. `benchmarks/test_am_aggressive.py` - 强制触发压缩（使用极长 prompt）
2. `benchmarks/test_am_vs_baseline.py` - AM vs Baseline 对比
3. `benchmarks/test_am_compression_quality.py` - 压缩质量验证
4. `benchmarks/test_am_real_prompt.py` - 真实 prompt 测试
5. `benchmarks/test_am_final.py` - 最终验证测试（推荐使用）✅
6. `benchmarks/test_adaptive_router_v2.py` - V2 路由器测试

### 报告文档

1. `.solar/am-qwen3-quality-failure-report.md` - 质量失败详细报告 ✅
2. `.solar/am-aggressive-test-report.md` - 激进压缩测试报告
3. `.solar/am-performance-report.md` - 性能测试报告（旧版）
4. `.solar/adaptive-routing-test-report.md` - V1 路由器测试报告
5. `.solar/adaptive-router-v2-test-report.md` - V2 路由器测试报告 ✅
6. `.solar/comprehensive-am-testing-summary.md` - 本文档 ✅

### 代码实现

1. `src/flashmlx/cache/adaptive_compressor.py` - V1 路由器（已过时）
2. `src/flashmlx/cache/adaptive_compressor_v2.py` - V2 路由器（推荐使用）✅
3. `mlx-lm-source/mlx_lm/models/compacted_cache.py` - AM 实现

### 测试日志

1. `/tmp/am_performance_test.log` - 旧性能测试日志
2. `/tmp/am_aggressive_test.log` - 激进压缩测试日志
3. `/tmp/am_vs_baseline_test.log` - 对比测试日志
4. `/tmp/am_quality_test.log` - 质量测试日志
5. `/tmp/am_real_prompt_test.log` - 真实 prompt 测试日志
6. `/tmp/am_final_test.log` - 最终测试日志 ✅
7. `/tmp/adaptive_router_v2_test.log` - V2 路由器测试日志 ✅

---

## 最终结论

### 用户关注点总结

1. **AM 目标验证** ❌
   - 离线压缩: ✅ 达成
   - 降低内存: ✅ 达成
   - **保持质量: ❌ 失败** (13% 相似度，答非所问)
   - 释放内存: ✅ 达成

2. **压缩成本分析** ✅
   - 压缩开销 < 0.4% PP time
   - 压缩比例精确 (2.0x / 3.0x)
   - 但质量破坏使成本分析失去意义

3. **推理性能影响** ❌
   - PP: -0.4% (几乎无影响)
   - TG: **-6.0%** (性能倒退)
   - TTFT: -0.4% (几乎无影响)
   - 与 Llama 的 +46% 完全相反

### 核心发现

**AM 是模型特定的压缩算法，不是架构通用的**:
- ✅ Llama 系列：质量 1.0, 速度 +46%
- ❌ Qwen3 系列：质量破坏（13%），速度倒退（-6%）
- ❌ Qwen3.5 系列：崩溃（混合架构）

### 推荐行动

1. **立即禁用**：Qwen3/Qwen3.5 系列禁止使用 AM
2. **自适应路由**：使用 V2 路由器（基于模型系列）
3. **质量验证**：新模型必须先用 `test_am_final.py` 验证
4. **H2O 测试**：验证 H2O 在 Qwen3 上的表现

---

*报告生成于: 2026-03-23*
*测试模型: Qwen3-8B (纯 Transformer)*
*AM 验证: ❌ FAILED - 输出质量破坏*
*后续行动: ✅ V2 路由器已实现并测试通过*
