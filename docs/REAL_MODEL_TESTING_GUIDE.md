# Real Model Generation Quality Testing Guide

> **目的**: 测试压缩算法对真实模型生成质量的影响
> **关键**: Token 重叠度、BLEU/ROUGE 分数、人工质量评估

---

## 🎯 测试目标

### 主要指标

| 指标 | 目标 | 说明 |
|------|------|------|
| **Token 重叠度** | ≥70% (4x) | 与 baseline 生成的 token 匹配度 |
| **BLEU Score** | ≥0.6 (4x) | 机器翻译质量评估指标 |
| **连贯性** | 保持 | 生成文本仍然流畅连贯 |
| **关键信息** | 不丢失 | 事实性内容准确 |

### 次要指标

- ROUGE Score (召回率)
- Perplexity (困惑度)
- 人工质量评分 (1-5 分)

---

## 📋 测试场景

### 1. 事实性问答 (Factual QA)

**Prompt**:
```
What is the capital of France?
```

**预期**:
- Baseline: "Paris"
- 压缩后: "Paris" (完全一致)

**关键**: 事实准确性

---

### 2. 摘要生成 (Summarization)

**Prompt**:
```
Summarize: Artificial intelligence (AI) is intelligence demonstrated
by machines, in contrast to the natural intelligence displayed by
humans and animals.
```

**预期**:
- 核心信息保留
- 语言流畅

**关键**: 信息完整性

---

### 3. 创意生成 (Creative)

**Prompt**:
```
Write a short poem about spring.
```

**预期**:
- 风格一致
- 结构相似

**关键**: 创造性保留

---

### 4. 逻辑推理 (Reasoning)

**Prompt**:
```
If all roses are flowers and some flowers fade quickly,
can we conclude that some roses fade quickly?
```

**预期**:
- 逻辑推理正确
- 解释清晰

**关键**: 推理能力

---

### 5. 长文本生成 (Long-form)

**Prompt**:
```
Explain the concept of quantum entanglement in simple terms.
```

**预期**:
- 长篇连贯性
- 概念准确

**关键**: 长上下文质量

---

## 🔧 实现方案

### 方案 A: 离线压缩 + 新生成 (当前可行)

**流程**:
```
1. 准备 prompt prefix (长上下文)
2. 生成 baseline (无压缩):
   - 编码 prefix → 生成 KV cache
   - 继续生成 → 记录 tokens

3. 压缩版本:
   - 编码 prefix → 生成 KV cache
   - 压缩 KV cache (使用我们的算法)
   - 用压缩 cache 继续生成 → 记录 tokens

4. 对比:
   - Token 重叠度
   - BLEU/ROUGE
   - 人工评估
```

**优点**:
- ✅ 使用现有压缩算法
- ✅ 不需要修改 MLX-LM

**缺点**:
- ⚠️ 需要访问 MLX-LM 内部 KV cache
- ⚠️ 需要实现 cache 替换逻辑

---

### 方案 B: Attention Patcher + CompactedKVCache (推荐)

**流程**:
```
1. 加载模型
2. Patch attention (注入 beta 支持)
3. 离线压缩 prefix:
   - 编码 prefix → KV cache
   - 压缩 → CompactedKVCache

4. 生成:
   - Baseline: 用原始 cache 生成
   - Compressed: 用 CompactedKVCache 生成

5. 对比
```

**优点**:
- ✅ 使用现有 CompactedKVCache 和 patcher
- ✅ 更接近实际使用场景

**缺点**:
- ⚠️ 仍需访问 MLX-LM 内部
- ⚠️ 需要实现 cache 替换

---

### 方案 C: 完整集成 (理想但复杂)

**流程**:
```
1. Fork MLX-LM
2. 添加 KV cache compression 支持:
   - 在生成过程中动态压缩
   - 无缝集成到 generate() 函数

3. 对比模式:
   - generate(prompt, cache_compression=None)  # Baseline
   - generate(prompt, cache_compression=4)      # 4x compressed

4. 自动对比和报告
```

**优点**:
- ✅ 最真实的使用场景
- ✅ 易于测试

**缺点**:
- ❌ 需要修改 MLX-LM 源代码
- ❌ 开发时间长

---

## 🚀 推荐实现路径

### 阶段 1: 快速验证 (1-2 天)

使用**方案 B** 进行初步测试：

```python
# 伪代码
from mlx_lm import load
from flashmlx.cache import (
    create_compaction_algorithm,
    create_compacted_cache_list,
    patch_attention_for_compacted_cache,
)

# 1. 加载模型
model, tokenizer = load("mlx-community/Qwen3-8B-Instruct")

# 2. Patch attention
patch_attention_for_compacted_cache(model)

# 3. 准备 prompt
prompt = "Explain quantum computing:"

# 4. 生成 baseline
baseline_output = model.generate(prompt, max_tokens=100)

# 5. 压缩 prefix KV cache
# (需要实现: 获取 model 内部 cache)
prefix_cache = get_model_cache(model, prompt)
compressed_cache = compress_cache(prefix_cache, ratio=4)

# 6. 用压缩 cache 生成
compressed_output = model.generate(
    prompt,
    max_tokens=100,
    cache=compressed_cache  # 替换 cache
)

# 7. 对比
overlap = calculate_token_overlap(baseline_output, compressed_output)
print(f"Token Overlap: {overlap:.1f}%")
```

**关键挑战**: 访问和替换 MLX-LM 的内部 KV cache

---

### 阶段 2: 完整测试 (3-5 天)

1. **实现 MLX-LM 集成**
   - 研究 MLX-LM 代码结构
   - 找到 KV cache 存储位置
   - 实现 cache 提取和替换

2. **运行全面测试**
   - 5 个测试场景
   - 3 个压缩比 (2x, 4x, 8x)
   - 多次运行取平均

3. **生成质量报告**
   - Token 重叠度统计
   - BLEU/ROUGE 分数
   - 人工质量评分
   - 典型案例展示

---

## 🔍 当前进度

### ✅ 已完成

1. **压缩算法实现** - Phase 6
   - HighestAttentionKeysCompaction
   - Beta solving (NNLS)
   - Ridge Regression for C2

2. **数据结构** - Phase 2-3
   - CompactedKVCache
   - Attention patcher (beta injection)

3. **基础测试** - Phase 7
   - 端到端工作流验证
   - 质量指标 (cosine similarity)
   - 多层压缩测试

4. **性能基准** - Phase 8
   - 压缩时间测试
   - 内存节省验证
   - 推理速度测试

5. **测试框架** - 刚刚创建
   - `benchmark_generation_quality.py`
   - 测试场景设计
   - 指标计算框架

### ⏳ 待完成 (当前阶段)

1. **MLX-LM 内部访问**
   - 研究 MLX-LM 源代码
   - 找到 KV cache 存储位置
   - 实现 cache 提取函数

2. **Cache 替换逻辑**
   - 实现 cache 替换函数
   - 确保与 GQA 兼容
   - 验证形状和类型

3. **真实生成测试**
   - 运行 5 个测试场景
   - 计算 token 重叠度
   - 生成对比报告

---

## 📊 预期结果

### 4x 压缩 (推荐配置)

基于我们的数学分析 (Phase 7 cosine similarity = 49.9%)，预期：

| 指标 | 预期值 | 解释 |
|------|--------|------|
| **Token 重叠度** | 70-80% | Cosine similarity 映射到 token 重叠 |
| **BLEU Score** | 0.6-0.7 | 高度相似但有变化 |
| **连贯性** | 保持 | Attention 方向大致相同 |
| **事实准确性** | 保持 | 关键 token 应该一致 |

**不确定性**: Cosine similarity 不能直接预测 token 重叠度，需要实际测试验证。

---

## 🎓 为什么需要真实模型测试？

### 数学指标 vs 实际质量

| 指标类型 | 示例 | 说明 |
|----------|------|------|
| **数学指标** | Cosine similarity: 49.9% | 向量空间中的相似度 |
| **实际质量** | Token overlap: ?% | 真正生成的 token 是否一致 |

**关键差异**:
- Cosine similarity 测量**方向**相似度
- Token generation 是**离散选择**（argmax/sampling）
- 即使向量方向相似，argmax 的结果可能不同

**例子**:
```
Baseline logits:    [2.1, 2.0, 1.9, 1.8, ...]
Compressed logits:  [1.9, 2.1, 1.8, 2.0, ...]

Cosine similarity: 0.95 (很高!)
但 argmax 结果:
  Baseline:   token_0
  Compressed: token_1  (不同!)
```

因此，必须进行真实 token 生成测试。

---

## 🛠️ 实现检查清单

### MLX-LM 集成

- [ ] 研究 MLX-LM 源代码 (github.com/ml-explore/mlx-lm)
- [ ] 定位 KV cache 存储位置
- [ ] 实现 `get_model_cache(model)` 函数
- [ ] 实现 `set_model_cache(model, cache)` 函数
- [ ] 验证 cache 格式兼容性

### 测试实现

- [ ] 完成 `benchmark_generation_quality.py` 的 TODO 部分
- [ ] 实现 `compress_kv_cache()` 真实逻辑
- [ ] 实现 `generate_with_compression()` 真实逻辑
- [ ] 添加更多测试场景 (10+ prompts)
- [ ] 实现 ROUGE 计算 (除了 BLEU)

### 质量验证

- [ ] 运行 baseline vs 2x vs 4x vs 8x
- [ ] 收集 token 重叠度数据
- [ ] 计算 BLEU/ROUGE 分数
- [ ] 人工评估生成质量 (抽样检查)
- [ ] 生成详细质量报告

---

## 📝 质量报告模板

```markdown
# Generation Quality Report

## Test Configuration
- Model: Qwen3-8B-Instruct
- Compression: 4x
- Test Cases: 5 scenarios
- Runs per case: 3

## Token Overlap Results

| Test Case | 2x | 4x | 8x |
|-----------|----|----|-----|
| Factual QA | 95% | 85% | 70% |
| Summarization | 88% | 75% | 60% |
| Creative | 82% | 68% | 50% |
| Reasoning | 90% | 78% | 65% |
| Long-form | 85% | 72% | 58% |
| **Average** | **88%** | **75.6%** | **60.6%** |

## BLEU Scores

| Compression | BLEU-4 | Grade |
|-------------|--------|-------|
| 2x | 0.82 | 🟢 Excellent |
| 4x | 0.71 | 🟡 Good |
| 8x | 0.55 | 🟠 Acceptable |

## Qualitative Analysis

### 4x Compression Examples

**Factual QA**:
- Baseline: "The capital of France is Paris."
- 4x Compressed: "The capital of France is Paris."
- ✅ Identical

**Summarization**:
- Baseline: "AI refers to machine intelligence..."
- 4x Compressed: "AI is machine intelligence..."
- ✅ Same meaning, slight wording difference

**Creative**:
- Baseline: "Spring awakens the sleeping earth..."
- 4x Compressed: "Spring brings life to dormant lands..."
- ⚠️ Different expression, similar theme

## Conclusions

- ✅ 4x compression maintains 75.6% token overlap
- ✅ Factual accuracy preserved
- ✅ Logical reasoning intact
- ⚠️ Creative generation shows more variation
- 🎯 **Recommendation**: 4x compression safe for production
```

---

## 🚧 当前障碍

### 主要挑战

1. **MLX-LM 内部访问**
   - MLX-LM 没有公开的 cache API
   - 需要直接访问模型内部状态
   - 可能需要修改 MLX-LM 源代码

2. **Cache 格式兼容性**
   - MLX-LM 使用的 cache 格式可能与我们的不同
   - 需要确保 CompactedKVCache 兼容

3. **GQA 处理**
   - Qwen3-8B 使用 GQA (8 KV heads)
   - 需要正确处理 head 维度

### 可能的解决方案

1. **Fork MLX-LM** (推荐)
   - Fork 并修改源代码
   - 添加 cache 访问 API
   - 提交 PR 到上游

2. **Monkey Patching**
   - 运行时修改 MLX-LM 类
   - 注入 cache 访问方法
   - 风险较高但不需要 fork

3. **Wrapper Approach**
   - 创建 MLX-LM 的 wrapper
   - 拦截生成过程
   - 在关键点替换 cache

---

## 📞 需要的帮助

如果您有以下经验，可以大大加速测试：

1. **MLX-LM 内部结构**
   - 熟悉 MLX-LM 源代码
   - 知道 KV cache 存储位置
   - 了解生成流程

2. **模型内部访问**
   - 如何获取中间状态
   - 如何替换 cache
   - GQA 的处理

3. **质量评估**
   - BLEU/ROUGE 计算库
   - 人工评估标准
   - 质量报告模板

---

## ✅ 下一步行动

### 立即可做 (不需要模型)

1. ✅ 完善测试框架 - 已完成
2. ✅ 设计测试场景 - 已完成
3. ✅ 实现指标计算 - 已完成

### 需要模型访问

4. ⏳ 研究 MLX-LM 源代码
5. ⏳ 实现 cache 提取/替换
6. ⏳ 运行真实生成测试
7. ⏳ 生成质量报告

### 预计时间

- 研究 MLX-LM: 2-3 小时
- 实现集成: 3-4 小时
- 运行测试: 1-2 小时
- 生成报告: 1 小时

**总计**: 7-10 小时

---

*Real Model Testing Guide*

*Created: 2026-03-22*

*Status: Framework Ready, Integration Pending*

*Next: MLX-LM Internal Access*
