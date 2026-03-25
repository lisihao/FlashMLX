# Phase 3 诊断 - 单层压缩测试报告

**日期**: 2026-03-23 09:57:46
**模型**: /Volumes/toshiba/models/qwen3.5-35b-mlx
**测试层**: Layer 39 (最后一个 Attention 层)
**Prompt**: '介绍机器学习的基本概念和应用场景'
**Max tokens**: 200

## 测试目标

1. 验证单个 Attention 层压缩是否会产生乱码
2. 隔离问题：是 AM 本身失效，还是层间交互导致

## 对比结果

| 配置 | 成功 | 乱码 | Tokens | 耗时 | 速度 |
|------|------|------|--------|------|------|
| Baseline (no compression) | ✅ | ✅ 否 | 200 | 2.52s | 79.36 tok/s |
| Layer 39 - Conservative (ratio=2.0) | ✅ | ❌ 是 | 200 | 2.41s | 83.05 tok/s |
| Layer 39 - Moderate (ratio=3.0) | ✅ | ❌ 是 | 200 | 2.42s | 82.66 tok/s |
| Layer 39 - Aggressive (ratio=5.0) | ✅ | ❌ 是 | 200 | 2.41s | 82.97 tok/s |

## 生成内容对比

### Baseline (no compression)

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
| **数据** | 训练模型的原材料```

### Layer 39 - Conservative (ratio=2.0)

**状态**: ❌ 乱码检测到

```
的,
#
#
#
#
#
let's the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the t
... (截断, 总共 758 字符)
```

### Layer 39 - Moderate (ratio=3.0)

**状态**: ❌ 乱码检测到

```
的,
#
#
#
#
#
let's the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the t
... (截断, 总共 758 字符)
```

### Layer 39 - Aggressive (ratio=5.0)

**状态**: ❌ 乱码检测到

```
的,
#
#
#
#
#
let's the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the t
... (截断, 总共 758 字符)
```

## 结论

### 结论 1: AM 在单层压缩时就完全失效

即使只压缩一个 Attention 层，所有压缩比都产生乱码。

**推论**:
- AM 算法本身在 Qwen3.5 Attention 层上不适用
- 问题不是层间交互，而是 AM 与这个架构的基本不兼容
- 需要检查：β 值分布、NNLS 求解状态、queries 生成

