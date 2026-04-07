# Multi-Task Quality Evaluation for FlashMLX

## 概述

多任务质量评估系统，用于检测 KV cache 压缩/量化对模型质量的影响。

**核心目标**: 超越单一 PPL 指标，在实际任务上验证配置质量。

## 评估任务

### 1. Needle-in-Haystack Recall

**目的**: 测试长上下文信息检索能力

**方法**:
- 在 2K/4K/8K token 的"干草堆"中嵌入"针"（独特信息）
- 测试模型能否从长上下文中准确召回
- 针的位置: 20% 和 80% （测试早期和晚期记忆）

**示例**:
```
Haystack: [5000 tokens of filler text]
Needle: "The magic number is 7284 and the secret code is PHOENIX."
Question: "What is the magic number and code mentioned above?"
```

**评分**: Recall = 正确召回数 / 总测试数

### 2. Reasoning Accuracy

**目的**: 测试多选推理能力

**方法**:
- MMLU 风格的多选题（A/B/C/D）
- 涵盖常识、数学、历史、科学等
- 温度=0.0 （确定性生成）

**示例**:
```
Question: What is the capital of France?
A) London
B) Paris
C) Berlin
D) Madrid

Answer: B
```

**评分**: Accuracy = 正确答案数 / 总题数

### 3. Composite Score

**综合评分** = 0.5 × Needle Recall + 0.5 × Reasoning Accuracy

权重设计:
- Needle Recall 占 50%: 长上下文检索是 KV cache 优化的关键场景
- Reasoning Accuracy 占 50%: 推理能力反映整体质量

## 使用方法

### 基础用法

```bash
python3 flashmlx_multi_task_eval.py \
  /path/to/model \
  --output results.json
```

自动测试 3 个配置:
1. Standard (baseline)
2. Triple PQ + PolarQuant 4bit
3. Triple PQ + TurboAngle K128V64

### 自定义配置

```python
from flashmlx_multi_task_eval import MultiTaskEvaluator
from flashmlx_meta_harness import BenchmarkConfig

evaluator = MultiTaskEvaluator('/path/to/model')

# 自定义配置
configs = [
    BenchmarkConfig(kv_cache='standard'),
    BenchmarkConfig(
        kv_cache='scored_pq',
        strategy='scored_pq',
        flat_quant='q8_0',
        density_mode='balanced'
    ),
]

comparison = evaluator.compare_configs(configs)
print(f"Best: {comparison['best_config']}")
```

## 输出格式

### JSON 结构

```json
{
  "model_path": "/path/to/model",
  "results": [
    {
      "task_name": "needle_in_haystack",
      "config": {...},
      "score": 0.83,
      "num_samples": 6,
      "num_failed": 1,
      "failed_samples": [
        {
          "context_length": 8192,
          "needle_position": 0.8,
          "needle": "The magic number is...",
          "response": "...",
          "expected_number": "7284",
          "expected_code": "PHOENIX"
        }
      ]
    },
    {
      "task_name": "reasoning",
      "config": {...},
      "score": 1.0,
      "num_samples": 5,
      "num_failed": 0,
      "failed_samples": []
    }
  ]
}
```

### 对比输出

```
================================================================================
CONFIGURATION COMPARISON
================================================================================
Rank   Config                                   Needle     Reasoning    Composite
--------------------------------------------------------------------------------
1      BenchmarkConfig(kv_cache='standard'...)  100.0%     100.0%       1.000
2      triple_pq polarquant 4bit                83.3%      100.0%       0.917
3      triple_pq turboangle K128V64             83.3%      80.0%        0.817
```

## Edge Case 分析

### 失败样本记录

每个失败的样本会记录:
- **Needle 任务**: 上下文长度、针位置、预期答案、实际响应
- **Reasoning 任务**: 问题、预期答案、预测答案、完整响应

### 边缘案例挖掘

**自动发现导致质量下降的输入模式**:

1. **长上下文失败模式**:
   - 8K 上下文 + 80% 位置 → 高失败率 → KV cache 驱逐问题
   - 特定 filler 模式 → 低 importance scores → 错误驱逐

2. **推理失败模式**:
   - 数学计算题失败 → 量化精度不足
   - 知识问答失败 → 权重退化

## 性能基准

### 预期结果（Llama-3.2-3B）

| 配置 | Needle Recall | Reasoning | Composite | 相对标准 |
|------|--------------|-----------|-----------|----------|
| **Standard** | 100% | 100% | 1.000 | baseline |
| **Triple PQ 4bit** | 83-100% | 100% | 0.92-1.00 | -8% ~ 0% |
| **Scored PQ Q8** | 67-83% | 100% | 0.84-0.92 | -16% ~ -8% |

### 质量阈值

**可接受的质量下降** (相对 standard):
- ✅ Composite Score ≥ 0.95 → 质量损失可忽略
- ⚠️  Composite Score 0.85-0.95 → 需要权衡
- ❌ Composite Score < 0.85 → 质量下降显著

## 与 Meta-Harness 集成

### 联合优化

```python
from flashmlx_meta_harness import FlashMLXMetaHarness
from flashmlx_multi_task_eval import MultiTaskEvaluator

# Step 1: Meta-Harness 找最优配置（速度/内存）
meta_harness = FlashMLXMetaHarness(model_path)
candidates = meta_harness.get_search_space('balanced')
best_config = meta_harness.optimize()

# Step 2: 质量验证（过滤低质量配置）
evaluator = MultiTaskEvaluator(model_path)
quality_scores = evaluator.evaluate_config(best_config)

if quality_scores['composite_score'] < 0.90:
    print("⚠️  Warning: Configuration has quality degradation")
```

### Model Card 更新

质量评估结果应添加到 Model Card:

```json
{
  "model_id": "llama-3.2-3b-mlx",
  "optimal": {...},
  "quality_evaluation": {
    "needle_recall": 0.833,
    "reasoning_acc": 1.0,
    "composite_score": 0.917,
    "relative_to_standard": -0.083
  }
}
```

## 扩展方向

### 未来任务

1. **HellaSwag**: Commonsense reasoning + continuation selection
2. **TruthfulQA**: Truthfulness detection
3. **MMLU (完整)**: 57 个学科的多选题
4. **HumanEval**: Code generation (如果模型支持)

### Adversarial Search

**主动寻找失败案例**:

```python
def find_adversarial_examples(model, config, num_trials=100):
    """
    Generate inputs that maximize quality degradation.

    Strategy:
    1. Generate random long contexts
    2. Measure perplexity/recall drop
    3. Keep worst-performing examples
    4. Mutate to find even worse cases
    """
    worst_cases = []

    for _ in range(num_trials):
        context = generate_random_context()
        quality_drop = measure_quality_drop(model, config, context)

        if quality_drop > threshold:
            worst_cases.append((context, quality_drop))

    return worst_cases
```

## 局限性

1. **小样本**: 当前使用 5-6 个样本（快速验证）
   - 完整评估需要 100+ 样本
   - 可扩展: 加载更大数据集

2. **任务覆盖**: 仅 2 个任务
   - MMLU/HellaSwag 需要完整数据集
   - TruthfulQA 需要专门的评估器

3. **生成质量**: 未测试长文本生成
   - 可添加: ROUGE/BLEU 评分
   - 人工评估: 流畅度、连贯性

## 总结

多任务评估提供了超越 PPL 的质量保证，确保 KV cache 优化不会牺牲实际任务性能。

**核心价值**:
- ✅ 检测长上下文检索退化
- ✅ 发现推理能力下降
- ✅ 边缘案例自动挖掘
- ✅ 配置质量排序

**下一步**: 集成到 Meta-Harness 流程，自动过滤低质量配置。
