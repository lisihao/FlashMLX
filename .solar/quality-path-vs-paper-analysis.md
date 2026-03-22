# Quality Path 实现 vs 论文对比分析

**日期**: 2026-03-21
**发现**: 我们的实现**缺少论文中最耗时的 Query Generation 部分**

---

## 论文的完整流程（60k tokens on H200）

| 步骤 | 时间 | 占比 | 说明 |
|------|------|------|------|
| 1. repeat-prefill | 8s | 1.1% | 生成查询向量 |
| **2. self-study** | **139s** | **19.3%** | **Query generation (subset selection)** |
| 3. Highest attention key selection | 3s | 0.4% | 选择 top-k keys |
| **4. OMP** | **565s** | **78.6%** | **Orthogonal Matching Pursuit** |
| 5. NNLS | 2.2s | 0.3% | 拟合 beta (attention bias) |
| 6. value fitting (LSQ) | 1.8s | 0.2% | 拟合压缩后的 values |
| **总计** | **719s** | **100%** | |

**关键发现**：
- **Query generation (self-study + OMP) 占 98% 的时间（704s / 719s）**
- 论文明确说："query generation dominates runtime"
- 其他步骤（key selection, NNLS, LSQ）只占 2% 的时间

---

## 我们的实现（FlashMLX Quality Path）

| 步骤 | 实现状态 | 文件 | 说明 |
|------|----------|------|------|
| 1. Query generation | ❌ **缺失** | - | 我们用 `queries = keys`（全量） |
| 2. Self-study | ❌ **缺失** | - | 跳过了 query subset selection |
| 3. OMP | ❌ **缺失** | - | 没有实现 Orthogonal Matching Pursuit |
| 4. Attention-aware key selection | ✅ 实现 | `quality.py:15-76` | `select_keys_attention_aware` |
| 5. NNLS (beta fitting) | ✅ 实现 | `quality.py:152-194` | `nnls_clamped`, `nnls_pgd` |
| 6. LSQ (value fitting) | ✅ 实现 | `quality.py:196-223` | `compute_C2_lstsq/cholesky/pinv` |

---

## 核心差异

### 论文的方法

```python
# 论文的完整流程
def paper_quality_path(K, V, budget):
    # Step 1: Query generation (self-study + OMP) - 704s (98%)
    queries = self_study(K)  # 选择代表性的 query subset
    queries = OMP(queries, K, V, budget)  # Orthogonal Matching Pursuit

    # Step 2: Key selection - 3s
    indices = select_top_k_keys(queries, K, budget)
    C1 = K[indices]

    # Step 3: Beta fitting (NNLS) - 2.2s
    beta = fit_beta_nnls(queries, C1)

    # Step 4: Value fitting (LSQ) - 1.8s
    C2 = fit_values_lsq(queries, C1, beta, V)

    return C1, beta, C2
```

**关键特征**：
- `queries` 是通过 self-study + OMP 生成的**少量**代表性查询（可能几百个）
- 时间复杂度：O(query_subset × seq_len) ≈ O(100 × 60k) = O(6M)
- 98% 时间花在 query generation 上

### 我们的实现

```python
# 我们的简化实现
def our_quality_path(K, V, budget):
    # Step 1: 跳过 query generation，直接用全量 keys
    queries = K  # ❌ 简化：用全量 keys 作为 queries

    # Step 2: Key selection
    indices = select_keys_attention_aware(queries, K, budget)
    C1 = K[indices]

    # Step 3: Beta fitting (NNLS)
    beta = fit_beta_nnls(queries, C1)

    # Step 4: Value fitting (LSQ)
    C2 = fit_values_lsq(queries, C1, beta, V)

    return C1, beta, C2
```

**关键特征**：
- `queries = keys`（全量 60k tokens）
- 时间复杂度：O(seq_len²) ≈ O(60k × 60k) = O(3.6B)
- **比论文慢 600 倍**（3.6B / 6M）

---

## 问题分析

### 1. 时间复杂度爆炸

| 项目 | 论文 | 我们的实现 | 差距 |
|------|------|-----------|------|
| Query 数量 | ~100（代表性子集） | 60,000（全量） | **600x** |
| Key selection 复杂度 | O(100 × 60k) | O(60k × 60k) | **600x** |
| NNLS 矩阵大小 | (100, budget) | (60k, budget) | **600x** |
| LSQ 矩阵大小 | (100, budget) | (60k, budget) | **600x** |

### 2. 内存占用

**论文**：
- Attention scores: (100 × 60k) = 6M floats = 24MB
- NNLS matrix: (100 × budget) ≈ 100 × 12k = 1.2M floats = 4.8MB

**我们**：
- Attention scores: (60k × 60k) = 3.6B floats = **14.4GB** ❌
- NNLS matrix: (60k × budget) = 60k × 12k = 720M floats = **2.88GB** ❌

### 3. 为什么我们的实现仍然工作？

虽然我们跳过了 query generation，但我们的实现仍然有效，因为：

1. **测试规模小**：
   - 我们测试的上下文：~500 tokens（不是 60k）
   - 时间复杂度：O(500²) = 250k（可接受）
   - 内存：(500 × 500) = 250k floats = 1MB（可接受）

2. **质量仍然好**：
   - 用全量 keys 作为 queries 是"过拟合"（使用更多信息）
   - 在小规模数据上，更多信息 = 更好的质量
   - 但在大规模数据上，会爆内存

3. **Fast Path vs Quality Path 对比有效**：
   - 我们证明了 Quality Path > Fast Path（在质量上）
   - 但没有证明在**大规模场景**（60k tokens）的可行性

---

## 论文的 Query Generation 算法

### Self-Study

**目的**：从全量 keys 中选择代表性的 query subset

**方法**（论文 Algorithm 1）：
```python
def self_study(K, num_queries):
    """
    Select representative queries from keys.

    Uses clustering or importance sampling to select
    a small subset of keys that best represent the
    attention patterns.
    """
    # 方法1: K-means clustering
    centroids = kmeans(K, num_queries)
    return centroids

    # 方法2: Importance sampling
    # (based on key norms or attention patterns)
    importance = compute_importance(K)
    indices = sample_top_k(importance, num_queries)
    return K[indices]
```

### Orthogonal Matching Pursuit (OMP)

**目的**：进一步优化 query subset，使其能最好地重建 attention output

**算法**（贪心迭代）：
```python
def OMP(queries, K, V, budget, max_iters=100):
    """
    Orthogonal Matching Pursuit for query refinement.

    Iteratively selects queries that maximally reduce
    reconstruction error.
    """
    selected = []
    residual = compute_original_output(K, V)

    for i in range(max_iters):
        # Find query that best reduces residual
        best_query = None
        best_error = float('inf')

        for q in queries:
            if q in selected:
                continue

            # Try adding this query
            test_selected = selected + [q]
            reconstruction = compute_compressed_output(test_selected, K, V, budget)
            error = norm(residual - reconstruction)

            if error < best_error:
                best_error = error
                best_query = q

        # Add best query
        selected.append(best_query)
        residual = residual - compute_output(best_query, K, V)

        # Stop if error small enough
        if best_error < threshold:
            break

    return selected
```

**时间复杂度**：O(max_iters × query_subset × seq_len) ≈ O(100 × 1000 × 60k) = O(6B)

---

## 修复建议

### Option 1: 实现完整的 Query Generation（推荐）

**优点**：
- ✅ 符合论文算法
- ✅ 支持大规模场景（60k+ tokens）
- ✅ 时间复杂度合理

**缺点**：
- ❌ 实现复杂（需要 self-study + OMP）
- ❌ 增加约 700s 的开销（在 60k 规模）

**实现**：
```python
# 添加到 quality.py
def self_study_kmeans(keys, num_queries):
    """Use K-means to select representative queries."""
    # TODO: 实现 K-means 聚类
    pass

def OMP(initial_queries, keys, values, budget, max_iters=100):
    """Orthogonal Matching Pursuit for query refinement."""
    # TODO: 实现 OMP 算法
    pass

def compact_multi_head_quality_with_query_gen(
    keys, values, budget,
    num_queries=100,  # 新参数
    use_omp=True      # 新参数
):
    # Step 1: Query generation
    queries = self_study_kmeans(keys, num_queries)
    if use_omp:
        queries = OMP(queries, keys, values, budget)

    # Step 2-4: 现有实现
    return compact_multi_head_quality(keys, values, budget, queries=queries)
```

### Option 2: 保持简化实现（当前）

**优点**：
- ✅ 实现简单
- ✅ 在小规模场景（< 2K tokens）工作良好
- ✅ 已验证有效

**缺点**：
- ❌ 不支持大规模场景（> 10K tokens）
- ❌ 时间复杂度 O(n²) 不可扩展
- ❌ 与论文不完全一致

**文档更新**：
```markdown
## Quality Path 限制

**重要**: 当前实现使用简化的 query generation：
- 使用全量 keys 作为 queries（`queries = keys`）
- 适用于中小规模场景（< 2K tokens）
- 大规模场景（> 10K tokens）可能耗时或 OOM

**与论文的差异**:
- 论文使用 self-study + OMP 生成少量代表性 queries
- 我们跳过 query generation，直接用全量 keys
- 在小规模数据上质量相同或更好
- 在大规模数据上不可扩展
```

### Option 3: Hybrid Approach（折中）

**实现自适应策略**：
```python
def compact_multi_head_quality_adaptive(
    keys, values, budget, queries=None
):
    seq_len = keys.shape[1]

    # 小规模：用全量 keys（简单快速）
    if seq_len < 2000:
        if queries is None:
            queries = keys
        return compact_multi_head_quality(keys, values, budget, queries)

    # 大规模：用 query generation（节省时间）
    else:
        if queries is None:
            num_queries = min(100, seq_len // 10)
            queries = self_study_kmeans(keys, num_queries)
        return compact_multi_head_quality(keys, values, budget, queries)
```

---

## 结论

### 当前实现状态

| 方面 | 状态 | 说明 |
|------|------|------|
| Key selection | ✅ 正确 | Attention-aware selection |
| Beta fitting (NNLS) | ✅ 正确 | 多种 NNLS solver |
| Value fitting (LSQ) | ✅ 正确 | 多种 LSQ solver |
| Query generation | ❌ **缺失** | **跳过了论文 98% 的计算** |
| 小规模场景 (< 2K) | ✅ 工作良好 | 已验证有效 |
| 大规模场景 (> 10K) | ❌ **不可用** | 时间/内存爆炸 |

### 性能对比

**小规模（500 tokens）**：
- 我们的实现：< 1s（可接受）
- 与论文相近

**大规模（60k tokens）**：
- 论文：~719s
- 我们的实现：**估计 > 10 小时**（不可用）

### 建议

1. **短期**：
   - ✅ 保持当前实现
   - ✅ 在文档中明确标注限制（< 2K tokens）
   - ✅ 添加序列长度检查，超过阈值时警告

2. **中期**（如需支持长上下文）：
   - 实现 self-study (K-means)
   - 添加可选的 OMP
   - 使用自适应策略（小规模用简化版，大规模用完整版）

3. **长期**（如需完全符合论文）：
   - 完整实现 Algorithm 1 (self-study)
   - 完整实现 Algorithm 2 (OMP)
   - 优化大规模场景性能

---

*分析完成于: 2026-03-21*
*关键发现: 我们跳过了论文 98% 的计算（query generation）*
*影响: 小规模场景工作良好，大规模场景不可用*
