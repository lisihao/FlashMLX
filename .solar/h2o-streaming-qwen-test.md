# H2O 和 StreamingLLM 在 Qwen3.5 混合架构上的验证

**日期**: 2026-03-23 10:08:58
**模型**: /Volumes/toshiba/models/qwen3.5-35b-mlx
**目的**: 验证 H2O 和 StreamingLLM 是否能在混合架构上工作

## 对比结果

| 方法 | 成功 | 乱码 | Tokens | 耗时 | 速度 |
|------|------|------|--------|------|------|
| Baseline (无压缩) | ✅ | ✅ 否 | 200 | 2.51s | 79.81 tok/s |
| H2O (compression_ratio=2.0) | ❌ | - | - | - | - |
| StreamingLLM (compression_ratio=2.0) | ❌ | - | - | - | - |

## 生成内容对比

### Baseline (无压缩)

**状态**: ✅ 正常

```
。

<think>
Let me think about how to approach this coding problem.

Problem: 介绍机器学习的基本概念和应用场景。

Key considerations:
- Approach: 机器学习（Machine Learning, ML）是人工智能（AI）的一个子领域，其核心思想是让计算机从数据中学习规律，而不是通过显式编程来执行特定任务。

Let me implement this solution.
</think>

# 机器学习基本概念与应用场景详解

## 一、机器学习的基本概念

### 1. 什么是机器学习？

机器学习（Machine Learning, ML）是人工智能（AI）的一个子领域，其核心思想是让计算机从数据中学习规律，而不是通过显式编程来执行特定任务。

**核心公式**：
$$f(x) = \text{模型从数据中学习到的映射关系}$$

### 2. 机器学习的三大要素

| 要素 | 说明 |
|------|------|
| **数据** | 训练模型的原材料
```

## 结论

### ❌ H2O 在 Qwen3.5 混合架构上失败

- H2O 执行失败: CompactedKVCache.__init__() got an unexpected keyword argument 'compression_method'. Did you mean 'compression_ratio'?
### ❌ StreamingLLM 在 Qwen3.5 混合架构上失败

- StreamingLLM 执行失败: CompactedKVCache.__init__() got an unexpected keyword argument 'compression_method'. Did you mean 'compression_ratio'?
