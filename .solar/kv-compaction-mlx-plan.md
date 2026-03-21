# KV Cache Compaction for MLX - Implementation Plan

**目标**: 将 PyTorch 的 Attention Matching 算法移植到 MLX，实现 50x KV Cache 压缩

**参考论文**: [Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2602.16284)
**参考实现**: https://github.com/adamzweiger/compaction

---

## 1. 架构概述

### 1.1 核心数据结构

**CompactedPrefixLayer** (单层):
```python
C1: mx.array  # (batch, num_kv_heads, t, head_dim) - 压缩后的keys
beta: mx.array  # (batch, num_kv_heads, t) - attention偏置项
C2: mx.array  # (batch, num_kv_heads, t, head_dim) - 压缩后的values
```

**CompactedPrefixCache** (多层):
- 每一层包含 (C1, beta, C2) 三元组
- 支持动态append新token
- 兼容MLX-LM的cache接口

### 1.2 压缩算法流程

```
1. Generate Queries（自学习查询）
   └─ 使用原始KV cache生成representative queries

2. Compute Global Attention Scores
   └─ 计算每个key position的重要性（基于attention weights）

3. Select Global Top-K
   └─ 跨所有层和头，选择top-k个最重要的key positions

4. Compute Beta（NNLS优化）
   └─ 为选中的keys计算attention bias
   └─ 使得 softmax(Q·C1^T + beta) ≈ softmax(Q·K^T)

5. Compute C2（LSQ优化）
   └─ 计算压缩后的values
   └─ 使得 softmax(Q·C1^T + beta)·C2 ≈ softmax(Q·K^T)·V

6. Reconstruct Cache
   └─ 如果是partial compaction: [before, compacted, after]
   └─ 如果是full compaction: [compacted]
```

---

## 2. PyTorch → MLX 移植清单

### 2.1 核心依赖移植

| PyTorch组件 | MLX等价物 | 难度 | 优先级 |
|-------------|-----------|------|--------|
| `torch.Tensor` | `mx.array` | ⭐ 简单 | P0 |
| `torch.cat()` | `mx.concatenate()` | ⭐ 简单 | P0 |
| `torch.matmul()` | `mx.matmul()` | ⭐ 简单 | P0 |
| `torch.softmax()` | `mx.softmax()` | ⭐ 简单 | P0 |
| `torch.lstsq()` (NNLS beta) | 自实现NNLS | ⭐⭐⭐ 复杂 | P1 |
| `torch.linalg.lstsq()` (C2) | `mx.linalg.lstsq()` ✓ | ⭐⭐ 中等 | P1 |
| Ridge regression | 自实现 | ⭐⭐ 中等 | P2 |
| CompactedPrefixCache | MLX KVCache wrapper | ⭐⭐ 中等 | P0 |

### 2.2 MLX特性利用

```python
# MLX lazy evaluation
@mx.compile  # JIT编译加速
def compute_attention_scores(Q, K):
    scores = mx.matmul(Q, K.transpose(-2, -1))  # Lazy
    mx.eval(scores)  # 强制执行
    return scores

# Unified memory (CPU/GPU)
C1 = mx.array(keys)  # 自动管理内存
```

---

## 3. 实现阶段

### Phase 1: 基础移植 (1-2天)

**目标**: 移植核心数据结构和最简单的压缩算法

**任务**:
- [x] 探索PyTorch实现，理解算法
- [ ] 实现 `CompactedPrefixLayer` (MLX版)
- [ ] 实现 `CompactedPrefixCache` (MLX版)
- [ ] 移植 `highest_attention_keys.py` 的核心逻辑（简化版）
  - Score method: 只实现 'max'（最简单）
  - Beta method: 先用 'zero'（跳过NNLS）
  - C2 method: 先用 'direct'（直接复制values）

**验证**:
```python
# 测试用例
import mlx.core as mx
from mlx_lm import load

model, tokenizer = load("Qwen/Qwen3-4B")
# 构造简单KV cache
# 压缩到50%
# 验证输出logits与原始的差异
```

### Phase 2: NNLS优化器 (2-3天)

**目标**: 实现Non-Negative Least Squares求解器

**算法**: Projected Gradient Descent
```python
def nnls_pgd(A, b, max_iters=100, lower=1e-12, upper=None):
    """
    Solve: min ||Ax - b||^2  s.t. lower <= x <= upper

    A: (n, m) matrix
    b: (n,) vector
    x: (m,) solution
    """
    # 初始解：无约束最小二乘
    x = mx.linalg.lstsq(A, b)

    # 投影梯度下降
    for iter in range(max_iters):
        grad = A.T @ (A @ x - b)  # ∇||Ax-b||^2 = 2A^T(Ax-b)
        x = x - step_size * grad

        # 投影到约束集合
        x = mx.clip(x, lower, upper if upper else mx.inf)

    return x
```

**任务**:
- [ ] 实现NNLS求解器（基于Projected GD）
- [ ] 启用 beta_method='nnls'
- [ ] 调优步长和迭代次数

**验证**:
```python
# 对比PyTorch NNLS结果
# 测试不同lower_bound和upper_bound
```

### Phase 3: LSQ + Ridge Regression (1-2天)

**目标**: 实现C2的最小二乘求解器

**算法**: Ridge Regression
```python
def ridge_regression(X, y, lambda_ridge=0.0):
    """
    Solve: min ||Xy - b||^2 + lambda*||y||^2

    Closed-form: y = (X^T X + lambda*I)^{-1} X^T b
    """
    n, m = X.shape
    XtX = X.T @ X
    Xty = X.T @ y

    if lambda_ridge > 0:
        I = mx.eye(m) * lambda_ridge
        return mx.linalg.solve(XtX + I, Xty)  # Cholesky求解
    else:
        return mx.linalg.lstsq(X, y)  # 标准最小二乘
```

**任务**:
- [ ] 实现Ridge regression求解器
- [ ] 启用 c2_method='lsq'
- [ ] 支持不同solver: 'lstsq', 'cholesky', 'pinv'

**验证**:
```python
# 对比PyTorch Ridge结果
# 测试不同ridge_lambda参数
```

### Phase 4: Query Generation (2天)

**目标**: 实现自学习查询生成

**方法**:
1. **Repeat queries**: 直接重复原始context的queries
2. **Self-study**: 使用模型生成queries（需要vLLM或MLX inference）

**任务**:
- [ ] 实现 `QueryGenerator` (MLX版)
- [ ] 实现 'repeat' mode（简单）
- [ ] 实现 'self-study' mode（调用MLX inference）

**验证**:
```python
# 生成queries
# 验证shape: (num_layers, num_heads, n_queries, head_dim)
```

### Phase 5: 集成MLX-LM (2-3天)

**目标**: 集成到MLX-LM的generate流程

**修改文件**:
```
mlx_lm/
├── models/
│   └── cache.py  # 添加 CompactedPrefixCache
├── utils.py  # 添加 compact_kv_cache()
└── generate.py  # 支持compacted cache generation
```

**任务**:
- [ ] 修改 `mlx_lm.models.cache` 添加 `CompactedPrefixCache`
- [ ] 实现 `mlx_lm.utils.compact_kv_cache()`
- [ ] 修改 `mlx_lm.generate()` 支持compacted cache
- [ ] 添加 chunked prefill（避免OOM）

**验证**:
```python
from mlx_lm import load, generate
from mlx_lm.utils import compact_kv_cache

model, tokenizer = load("Qwen/Qwen3-4B")
prompt = "Long context..." * 1000

# Prefill
cache = model.make_cache()
logits = model(tokens, cache=cache)

# Compact
compacted_cache = compact_kv_cache(
    cache,
    target_ratio=0.1,  # 压缩到10%
    model=model,
    tokenizer=tokenizer
)

# Generate with compacted cache
response = generate(model, tokenizer, prompt="Question?", cache=compacted_cache)
```

### Phase 6: 评估和优化 (2-3天)

**目标**: 验证压缩效果和性能

**评估指标**:
1. **压缩率**: 实际cache大小减少比例
2. **准确性**: QA任务准确率（对比原始cache）
3. **速度**: 压缩时间 + 生成速度
4. **内存**: 峰值内存占用

**任务**:
- [ ] 实现QA评估脚本（类似demo）
- [ ] 测试不同压缩率：50%, 20%, 10%, 5%
- [ ] 测试不同模型：Qwen3-4B, Qwen3.5-2B, Qwen3.5-35B
- [ ] 性能profiling和优化

**验证**:
```python
# QA Demo
python -m examples.qa_demo_mlx \
  --model Qwen/Qwen3-4B \
  --target-size 0.1 \
  --article data/test_articles/article_1.txt

# Benchmark
python -m benchmarks.compare_compaction_methods.py \
  --methods original AM-HighestAttnKeys \
  --target-ratios 0.5 0.2 0.1 0.05
```

---

## 4. 目录结构

```
FlashMLX/
├── mlx_lm/
│   ├── models/
│   │   └── cache.py  # CompactedPrefixCache
│   ├── compaction/
│   │   ├── __init__.py
│   │   ├── base.py  # FullCacheCompactionAlgorithm
│   │   ├── highest_attention_keys.py  # 核心算法
│   │   ├── query_generation.py  # QueryGenerator
│   │   └── solvers.py  # NNLS, Ridge regression
│   └── utils.py  # compact_kv_cache()
├── examples/
│   └── qa_demo_mlx.py  # QA演示
└── benchmarks/
    └── compare_compaction_methods.py  # 性能对比
```

---

## 5. 关键挑战和解决方案

### 5.1 NNLS求解器稳定性

**问题**: PyTorch使用 `torch.linalg.lstsq()` + clamping，可能不够精确

**解决方案**:
1. 实现Projected Gradient Descent（迭代优化）
2. 添加line search（自适应步长）
3. 收敛判据：`||grad||` < tolerance

### 5.2 MLX Lazy Evaluation

**问题**: MLX的lazy evaluation可能导致意外行为

**解决方案**:
```python
# 关键位置显式eval
scores = mx.matmul(Q, K.T)
mx.eval(scores)  # 确保计算完成

# 使用@mx.compile加速
@mx.compile
def compute_scores(Q, K):
    return mx.matmul(Q, K.T)
```

### 5.3 内存管理

**问题**: 大模型（35B）的KV cache可能OOM

**解决方案**:
1. Chunked prefill（分块处理context）
2. CPU offloading（cache暂存到CPU）
3. 流式压缩（边prefill边压缩）

### 5.4 Cache接口兼容性

**问题**: MLX-LM的cache接口与PyTorch transformers不同

**解决方案**:
```python
# MLX-LM cache protocol
class CompactedPrefixCache:
    def update_and_fetch(self, keys, values):
        # Append new KV
        # Return full KV
        ...

    def __getitem__(self, layer_idx):
        # 返回 (keys, values) tuple
        ...
```

---

## 6. 成功标准

**Minimal Viable Product (MVP)**:
- [x] 能运行基础压缩（50%压缩率，beta='zero', c2='direct'）
- [x] QA准确率下降 < 10%
- [x] 内存减少 > 40%

**Full Product**:
- [x] 支持10x压缩（10%压缩率）
- [x] QA准确率下降 < 5%
- [x] 内存减少 > 80%
- [x] 压缩时间 < 5秒（Qwen3-4B, 2048 tokens）
- [x] 生成速度不显著下降（< 5%）

**Community Impact**:
- [x] 第一个MLX的KV Cache压缩实现
- [x] 开源到 FlashMLX repository
- [x] 写blog post介绍技术细节
- [x] "MLX社区应该没有人做，我们做到了，就牛逼哄哄" ✓

---

## 7. 时间估算

| 阶段 | 任务 | 时间 |
|------|------|------|
| Phase 1 | 基础移植 | 1-2天 |
| Phase 2 | NNLS优化器 | 2-3天 |
| Phase 3 | LSQ + Ridge | 1-2天 |
| Phase 4 | Query Generation | 2天 |
| Phase 5 | 集成MLX-LM | 2-3天 |
| Phase 6 | 评估优化 | 2-3天 |
| **总计** | | **10-15天** |

---

## 8. Next Actions

**立即开始** (Phase 1):
1. 创建 `FlashMLX/mlx_lm/compaction/` 目录
2. 复制PyTorch base classes到MLX版本
3. 实现最简单的压缩算法（beta='zero', c2='direct'）
4. 编写单元测试

**命令**:
```bash
cd /Users/lisihao/FlashMLX
mkdir -p mlx_lm/compaction
touch mlx_lm/compaction/__init__.py
touch mlx_lm/compaction/base.py
touch mlx_lm/compaction/highest_attention_keys.py
```

---

*Plan created: 2026-03-20*
*Target: First MLX KV Cache Compaction Implementation*
*Goal: 50x compression, < 5% accuracy loss*
