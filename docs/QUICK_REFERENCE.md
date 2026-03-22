# FlashMLX Attention Matching v2: Quick Reference

> **快速查询卡** - 压缩算法使用和配置参考

---

## 📦 安装

```bash
pip install flashmlx
```

---

## 🚀 快速开始 (离线压缩)

```python
from flashmlx.cache import create_compaction_algorithm

# 1. 创建压缩算法
algo = create_compaction_algorithm(
    score_method='mean',      # 注意力分数聚合: 'mean', 'max', 'sum'
    beta_method='nnls',       # Beta 求解方法: 'nnls', 'ones'
    c2_method='lsq',          # C2 计算方法: 'lsq', 'direct'
    c2_ridge_lambda=0.01      # Ridge 正则化参数
)

# 2. 压缩 KV cache (per head)
# K, V: (T, head_dim) - 原始 keys 和 values
# queries: (n, head_dim) - 查询样本
# t: 目标压缩长度 (e.g., T/4 for 4x compression)
C1, beta, C2, indices = algo.compute_compacted_cache(K, V, queries, t)

# 3. 形状验证
assert C1.shape == (t, head_dim)    # 压缩后的 keys
assert beta.shape == (t,)           # Bias terms
assert C2.shape == (t, head_dim)    # 压缩后的 values
assert len(indices) == t            # 选中的 key indices
```

---

## 🎯 推荐配置

### 配置速查表

| 场景 | 压缩比 | 内存节省 | 质量 | 推理速度 | 适用 |
|------|--------|----------|------|----------|------|
| **长上下文对话** | 4x | 75% | 50% | +11% | 客服、RAG、文档问答 |
| **质量敏感** | 2x | 50% | 75% | +8% | 创作、翻译、代码生成 |
| **极限内存** | 8x | 87% | 31% | +20% | 移动设备、嵌入式 |

### 通用配置 (推荐)

```python
# 4x 压缩 - 最佳平衡点
algo = create_compaction_algorithm(
    score_method='mean',
    beta_method='nnls',
    c2_method='lsq',
    c2_ridge_lambda=0.01
)

t = T // 4  # 4x compression
C1, beta, C2, indices = algo.compute_compacted_cache(K, V, queries, t)
```

### 质量优先配置

```python
# 2x 压缩 - 高质量输出
algo = create_compaction_algorithm(
    score_method='mean',
    beta_method='nnls',
    c2_method='lsq',
    c2_ridge_lambda=0.005  # 降低正则化
)

t = T // 2  # 2x compression
```

### 内存优先配置

```python
# 8x 压缩 - 极限内存节省
algo = create_compaction_algorithm(
    score_method='max',        # 使用 max 更激进
    beta_method='ones',        # 简化 beta 求解
    c2_method='direct',        # 直接复制
    c2_ridge_lambda=0.02       # 增加正则化
)

t = T // 8  # 8x compression
```

---

## 🔧 参数详解

### score_method (注意力分数聚合)

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `'mean'` | 平均聚合 (推荐) | 通用场景 |
| `'max'` | 最大值聚合 | 更激进的压缩 |
| `'sum'` | 求和聚合 | 长查询序列 |

### beta_method (Beta 求解方法)

| 方法 | 说明 | 速度 | 质量 |
|------|------|------|------|
| `'nnls'` | Log-ratio approximation (推荐) | 快 | 好 |
| `'ones'` | 全为 1 (无 bias) | 最快 | 中 |

### c2_method (C2 计算方法)

| 方法 | 说明 | 速度 | 质量 |
|------|------|------|------|
| `'lsq'` | Ridge Regression (推荐) | 中 | 好 |
| `'direct'` | 直接复制选中的 V | 快 | 中 |

### c2_ridge_lambda (Ridge 正则化参数)

| 值 | 说明 | 适用场景 |
|----|------|----------|
| `0.005` | 弱正则化 | 质量优先 |
| `0.01` | 中等正则化 (推荐) | 平衡场景 |
| `0.02` | 强正则化 | 内存优先 |

---

## 📊 性能指标参考

### 压缩时间 (Qwen3-8B 维度)

| 原始长度 | 2x | 4x | 8x |
|----------|----|----|-----|
| 256 | 8ms | 6ms | 6ms |
| 512 | 11ms | 8ms | 6ms |
| 1024 | 16ms | **11ms** | 7ms |
| 2048 | 29ms | 17ms | 10ms |

### 内存节省 (1024 tokens)

| 压缩比 | 原始 | 压缩后 | 节省 |
|--------|------|--------|------|
| 2x | 8192 KB | 4112 KB | 49.8% |
| 4x | 8192 KB | **2056 KB** | **74.9%** |
| 8x | 8192 KB | 1028 KB | 87.5% |

### 质量保留 (Cosine Similarity)

| 压缩比 | 相似度 | 等级 |
|--------|--------|------|
| 2x | 75.5% | 🟢 优秀 |
| 4x | **49.9%** | 🟡 **良好** |
| 8x | 30.8% | 🟠 可用 |

### 推理速度变化

| 压缩比 | TTFT | 速度变化 |
|--------|------|----------|
| 无压缩 | 0.126ms | 1.00x |
| 4x | 0.114ms | **1.11x (加速!)** |

---

## 🔄 完整工作流

### Step 1: 离线压缩 (一次性)

```python
from flashmlx.cache import create_compaction_algorithm

# 为每一层、每个 head 压缩
algo = create_compaction_algorithm(
    score_method='mean',
    beta_method='nnls',
    c2_method='lsq',
    c2_ridge_lambda=0.01
)

compacted_cache_per_layer = []

for layer_idx in range(num_layers):
    layer_cache = []

    for head_idx in range(n_kv_heads):
        K_head = K[layer_idx, head_idx]  # (T, head_dim)
        V_head = V[layer_idx, head_idx]  # (T, head_dim)

        C1, beta, C2, indices = algo.compute_compacted_cache(
            K_head, V_head, queries, t
        )

        layer_cache.append((C1, beta, C2))

    compacted_cache_per_layer.append(layer_cache)
```

### Step 2: 创建 CompactedKVCache

```python
from flashmlx.cache import create_compacted_cache_list

# 包装为 (B, n_kv_heads, t, head_dim)
wrapped_cache = []

for layer_cache in compacted_cache_per_layer:
    C1_layer = mx.stack([c1 for c1, _, _ in layer_cache])
    beta_layer = mx.stack([beta for _, beta, _ in layer_cache])
    C2_layer = mx.stack([c2 for _, _, c2 in layer_cache])

    # Expand batch dimension
    C1_layer = mx.expand_dims(C1_layer, axis=0)    # (1, n_kv_heads, t, head_dim)
    beta_layer = mx.expand_dims(beta_layer, axis=0)  # (1, n_kv_heads, t)
    C2_layer = mx.expand_dims(C2_layer, axis=0)    # (1, n_kv_heads, t, head_dim)

    wrapped_cache.append((C1_layer, beta_layer, C2_layer))

# 创建 CompactedKVCache list
cache = create_compacted_cache_list(
    compacted_cache=wrapped_cache,
    original_seq_len=T
)
```

### Step 3: Patch Attention

```python
from flashmlx.cache import patch_attention_for_compacted_cache
from mlx_lm import load

# 加载模型
model, tokenizer = load("mlx-community/Qwen3-8B-Instruct")

# Patch attention (一次性)
patch_attention_for_compacted_cache(model, verbose=True)
```

### Step 4: 推理

```python
# 正常使用模型，cache 会被自动处理
input_ids = tokenizer.encode("Your prompt here")
output = model(input_ids, cache=cache)
```

---

## 🐛 常见问题

### Q1: 压缩时间太长？

**A1**: 降低压缩比或减少 query samples:
```python
# 减少 queries 数量
queries = queries[:20]  # 从 50 降到 20

# 或降低压缩比
t = T // 2  # 从 4x 降到 2x
```

### Q2: 质量不够好？

**A2**: 降低压缩比或调整参数:
```python
# 降低压缩比
t = T // 2  # 从 4x 降到 2x

# 或降低正则化
algo = create_compaction_algorithm(
    score_method='mean',
    beta_method='nnls',
    c2_method='lsq',
    c2_ridge_lambda=0.005  # 从 0.01 降到 0.005
)
```

### Q3: 内存节省不够？

**A3**: 提高压缩比:
```python
# 提高压缩比
t = T // 8  # 从 4x 提高到 8x

# 使用更激进的配置
algo = create_compaction_algorithm(
    score_method='max',      # 从 mean 改为 max
    beta_method='ones',      # 简化 beta
    c2_method='direct',      # 直接复制
)
```

### Q4: 如何选择 query samples？

**A4**: Query samples 应该代表实际推理中的查询分布:
```python
# 方案 1: 使用最近的 queries (推荐)
recent_tokens = K[-50:]  # 最后 50 个 tokens
queries = recent_tokens

# 方案 2: 随机采样
indices = mx.random.permutation(T)[:50]
queries = K[indices]

# 方案 3: 均匀采样
indices = mx.arange(0, T, T // 50)
queries = K[indices]
```

### Q5: NumPy fallback 太慢？

**A5**: 当前版本使用 NumPy fallback，未来 MLX 添加 GPU 支持后会自动替换。临时解决方案:
```python
# 使用 'direct' 方法避免 Ridge Regression
algo = create_compaction_algorithm(
    score_method='mean',
    beta_method='nnls',
    c2_method='direct',  # 避免 NumPy fallback
)
```

---

## 📚 API 速查

### 主要类

```python
# 压缩算法
from flashmlx.cache import HighestAttentionKeysCompaction

# KV Cache
from flashmlx.cache import (
    CompactedKVCache,
    CompactedKVCacheLayer,
    create_compacted_cache_list,
)

# Attention Patcher
from flashmlx.cache import (
    patch_attention_for_compacted_cache,
    repeat_kv,
)
```

### 工厂函数

```python
# 创建压缩算法
from flashmlx.cache import create_compaction_algorithm

algo = create_compaction_algorithm(
    score_method='mean',
    beta_method='nnls',
    c2_method='lsq',
    c2_ridge_lambda=0.01
)
```

### 核心方法

```python
# 压缩
C1, beta, C2, indices = algo.compute_compacted_cache(K, V, queries, t)

# 创建 cache
cache = create_compacted_cache_list(compacted_cache, original_seq_len=T)

# Patch attention
patch_attention_for_compacted_cache(model, verbose=True)
```

---

## 🔗 相关文档

- **完整报告**: `docs/FINAL_SUMMARY.md`
- **性能报告**: `benchmarks/PERFORMANCE_REPORT.md`
- **实现细节**: `docs/COMPRESSION_ALGORITHM_COMPLETE.md`
- **迁移指南**: `docs/MIGRATION_COMPLETE.md`

---

## 💡 最佳实践

1. **默认使用 4x 压缩** - 最佳平衡点
2. **Query samples 使用最近的 50 个 tokens** - 代表实际分布
3. **离线压缩一次，推理多次使用** - 摊销压缩开销
4. **质量敏感场景降到 2x** - 保持高质量输出
5. **内存受限场景提高到 8x** - 极限内存节省
6. **定期验证质量** - 使用 cosine similarity 监控

---

*Quick Reference v1.0*
*Last Updated: 2026-03-22*
*FlashMLX Attention Matching v2*
