# 完整实现论文算法 - 行动计划

**日期**: 2026-03-22
**目标**: 正确实现离线 KV Cache 压缩（GC 式内存管理）
**原则**: 质量第一，速度第二

**状态**: ✅ **Phase 1 完成** | Phase 2 待开始 | Phase 3 待定

---

## ✅ Phase 1 完成摘要

**完成时间**: 2026-03-22
**耗时**: ~2 小时

### 实现内容
- ✅ Self-Study: K-means + 重要性采样（自动回退）
- ✅ OMP: 简化版（实用主义）
- ✅ 离线压缩主流程：单头 + 多头 + MLX-LM 集成
- ✅ 单元测试：6/6 通过
- ✅ E2E 测试：3/3 通过

### 关键文件
- `src/flashmlx/compaction/query_generation/self_study.py`
- `src/flashmlx/compaction/query_generation/omp.py`
- `src/flashmlx/compaction/offline_compressor.py`
- `tests/compaction/test_query_generation.py`
- `tests/compaction/test_offline_compression_e2e.py`

详见 **PHASE_1_COMPLETE.md**

---

## 问题根源

### ❌ 错误理解
- 以为是实时加速生成
- 跳过了 Query Generation（为了快）
- 用简单的 "最后 N tokens" 代替

### ✅ 正确理解
- **离线 GC 式压缩**（类似 JVM GC）
- 目的：清理 KV cache，释放内存
- 可以慢慢做（10 分钟、1 小时都可以）
- **不追求速度，追求质量和压缩比**

---

## Phase 1: 完整实现（1-2 天）

### 1.1 Self-Study: Query Selection

**文件**: `src/flashmlx/compaction/query_generation/self_study.py`

```python
"""
Self-Study: Select representative queries from full key sequence

Methods:
1. K-means clustering (preferred)
2. Importance sampling (alternative)
"""

def self_study_kmeans(
    keys: mx.array,  # (seq_len, head_dim)
    num_queries: int = 100
) -> mx.array:
    """
    K-means clustering to select representative queries

    Args:
        keys: Full key sequence (60k tokens)
        num_queries: Number of representative queries to select

    Returns:
        Representative queries (num_queries, head_dim)
    """
    from sklearn.cluster import KMeans

    # Convert to numpy for sklearn
    keys_np = np.array(keys)

    # K-means clustering
    kmeans = KMeans(n_clusters=num_queries, random_state=42)
    kmeans.fit(keys_np)

    # Cluster centers are representative queries
    centroids = mx.array(kmeans.cluster_centers_)

    return centroids


def self_study_importance_sampling(
    keys: mx.array,
    num_queries: int = 100
) -> mx.array:
    """
    Importance sampling based on key norms

    Alternative to K-means, faster but potentially lower quality
    """
    # Compute importance scores (L2 norms)
    importance = mx.sum(keys ** 2, axis=-1)  # (seq_len,)

    # Sample top-k by importance
    top_indices = mx.argsort(importance)[-num_queries:]

    return keys[top_indices]
```

### 1.2 OMP: Orthogonal Matching Pursuit

**文件**: `src/flashmlx/compaction/query_generation/omp.py`

```python
"""
Orthogonal Matching Pursuit for query refinement

Iteratively selects queries that maximally reduce reconstruction error
"""

def omp_refine_queries(
    initial_queries: mx.array,  # (num_queries, head_dim)
    keys: mx.array,             # (seq_len, head_dim)
    values: mx.array,           # (seq_len, head_dim)
    budget: int,                # Target compressed size
    max_iters: int = 100
) -> mx.array:
    """
    Refine query subset using Orthogonal Matching Pursuit

    Args:
        initial_queries: Initial query set from self-study
        keys: Full key sequence
        values: Full value sequence
        budget: Target compression budget
        max_iters: Maximum OMP iterations

    Returns:
        Refined query subset
    """
    selected_queries = []
    candidate_pool = list(range(len(initial_queries)))

    # Compute original attention output (target)
    target_output = compute_attention_output(keys, keys, values)
    residual = target_output

    for iteration in range(max_iters):
        if len(selected_queries) >= budget:
            break

        best_query_idx = None
        best_error = float('inf')

        # Try each candidate query
        for idx in candidate_pool:
            if idx in selected_queries:
                continue

            # Test adding this query
            test_queries = [initial_queries[i] for i in selected_queries + [idx]]
            test_queries = mx.stack(test_queries)

            # Compress with these queries
            compressed_output = compute_compressed_output(
                test_queries, keys, values, budget
            )

            # Measure reconstruction error
            error = mx.sum((residual - compressed_output) ** 2)

            if error < best_error:
                best_error = error
                best_query_idx = idx

        # Add best query
        selected_queries.append(best_query_idx)

        # Update residual
        refined_queries = mx.stack([initial_queries[i] for i in selected_queries])
        compressed_output = compute_compressed_output(
            refined_queries, keys, values, budget
        )
        residual = target_output - compressed_output

        # Early stopping if error is small enough
        if mx.sum(residual ** 2) < 1e-6:
            break

    return mx.stack([initial_queries[i] for i in selected_queries])


def compute_attention_output(queries, keys, values):
    """Compute standard attention output"""
    # Q @ K^T
    scores = queries @ keys.T  # (num_queries, seq_len)

    # Softmax
    weights = mx.softmax(scores, axis=-1)

    # Weighted sum of values
    output = weights @ values  # (num_queries, head_dim)

    return output


def compute_compressed_output(queries, keys, values, budget):
    """Compute compressed attention output"""
    # This is a placeholder - needs full compression pipeline
    # Including key selection, beta fitting, value fitting
    from flashmlx.cache import create_compaction_algorithm

    algo = create_compaction_algorithm()
    C1, beta, C2, _ = algo.compute_compacted_cache(
        keys, values, queries, budget
    )

    # Compute output with compressed cache
    scores = queries @ C1.T
    scores = scores + beta[None, :]  # Add bias
    weights = mx.softmax(scores, axis=-1)
    output = weights @ C2

    return output
```

### 1.3 集成到主流程

**文件**: `src/flashmlx/compaction/offline_compressor.py`

```python
"""
Offline KV Cache Compressor (完整论文实现)

用于离线 GC 式内存压缩，不追求速度，追求质量
"""

def offline_compress_kv_cache(
    keys: mx.array,      # (B, n_heads, seq_len, head_dim)
    values: mx.array,    # (B, n_heads, seq_len, head_dim)
    compression_ratio: int = 4,
    num_queries: int = 100,
    use_omp: bool = True,
    max_omp_iters: int = 100,
    verbose: bool = True
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    完整的离线压缩流程（论文算法）

    可以慢慢做，不限制时间

    Args:
        keys: Full key cache
        values: Full value cache
        compression_ratio: Target compression ratio (e.g., 4x)
        num_queries: Number of representative queries to select
        use_omp: Whether to use OMP refinement (slower but better quality)
        max_omp_iters: Maximum OMP iterations
        verbose: Print progress

    Returns:
        (C1, beta, C2): Compressed cache
    """
    B, n_heads, seq_len, head_dim = keys.shape
    budget = seq_len // compression_ratio

    if verbose:
        print(f"Offline KV Cache Compression")
        print(f"  Sequence length: {seq_len} tokens")
        print(f"  Target budget: {budget} tokens ({compression_ratio}x)")
        print(f"  Query generation: {num_queries} queries")
        print(f"  Use OMP: {use_omp}")

    compacted_data = []

    for head_idx in range(n_heads):
        if verbose:
            print(f"\n[Head {head_idx + 1}/{n_heads}]")

        K_head = keys[0, head_idx]  # (seq_len, head_dim)
        V_head = values[0, head_idx]

        # Step 1: Self-Study (Query Generation)
        if verbose:
            print("  Step 1: Self-Study (K-means clustering)...")

        import time
        t0 = time.time()
        queries = self_study_kmeans(K_head, num_queries)
        t1 = time.time()

        if verbose:
            print(f"    ✓ Selected {num_queries} representative queries ({t1-t0:.2f}s)")

        # Step 2: OMP Refinement (optional)
        if use_omp:
            if verbose:
                print("  Step 2: OMP Refinement...")

            t0 = time.time()
            queries = omp_refine_queries(
                queries, K_head, V_head, budget, max_omp_iters
            )
            t1 = time.time()

            if verbose:
                print(f"    ✓ Refined queries ({t1-t0:.2f}s)")

        # Step 3-6: Existing pipeline
        if verbose:
            print("  Step 3: Key selection + Beta + Value fitting...")

        from flashmlx.cache import create_compaction_algorithm
        algo = create_compaction_algorithm(
            score_method='mean',
            beta_method='nnls',
            c2_method='lsq',
            c2_ridge_lambda=0.01
        )

        t0 = time.time()
        C1, beta, C2, _ = algo.compute_compacted_cache(
            K_head, V_head, queries, budget
        )
        t1 = time.time()

        if verbose:
            print(f"    ✓ Compression complete ({t1-t0:.2f}s)")

        compacted_data.append((C1, beta, C2))

    # Stack results
    C1_all = mx.stack([c1 for c1, _, _ in compacted_data], axis=1)[None, ...]  # (1, n_heads, budget, head_dim)
    beta_all = mx.stack([b for _, b, _ in compacted_data], axis=1)[None, ...]  # (1, n_heads, budget)
    C2_all = mx.stack([c2 for _, _, c2 in compacted_data], axis=1)[None, ...]

    if verbose:
        print(f"\n✅ Offline compression complete")
        print(f"   Original: {seq_len} tokens")
        print(f"   Compressed: {budget} tokens")
        print(f"   Ratio: {compression_ratio}x")
        print(f"   Memory saved: {(1 - budget/seq_len) * 100:.1f}%")

    return C1_all, beta_all, C2_all
```

---

## Phase 2: 质量验证（0.5 天）

### 评估脚本

**文件**: `benchmarks/evaluate_offline_compression.py`

```python
"""
评估离线压缩质量（论文标准）
"""

def evaluate_compression_quality(
    model,
    tokenizer,
    compressed_cache,
    original_cache,
    test_prompts: List[str]
):
    """
    使用论文标准评估压缩质量

    Metrics:
    1. Perplexity (主要指标)
    2. Reconstruction error
    3. Memory saving
    """
    results = {
        'perplexity_original': [],
        'perplexity_compressed': [],
        'reconstruction_error': [],
        'memory_saving': 0
    }

    for prompt in test_prompts:
        # Original
        ppl_orig = compute_perplexity(model, tokenizer, prompt, original_cache)
        results['perplexity_original'].append(ppl_orig)

        # Compressed
        ppl_comp = compute_perplexity(model, tokenizer, prompt, compressed_cache)
        results['perplexity_compressed'].append(ppl_comp)

    # Summary
    avg_ppl_orig = np.mean(results['perplexity_original'])
    avg_ppl_comp = np.mean(results['perplexity_compressed'])
    ppl_increase = (avg_ppl_comp - avg_ppl_orig) / avg_ppl_orig * 100

    print(f"Quality Evaluation Results:")
    print(f"  Original Perplexity: {avg_ppl_orig:.2f}")
    print(f"  Compressed Perplexity: {avg_ppl_comp:.2f}")
    print(f"  Increase: {ppl_increase:.2f}%")
    print(f"  ✅ PASS" if ppl_increase < 5 else "❌ FAIL")

    return results
```

---

## Phase 3: 优化耗时（1-2 天，可选）

**只有在 Phase 2 验证合格后才进行**

### 优化方向

1. **并行化**
   - 多头并行压缩
   - 多核 K-means

2. **近似算法**
   - Mini-batch K-means
   - Randomized OMP
   - 验证质量下降 < 2%

3. **Early stopping**
   - 收敛检测
   - 残差阈值

---

## 时间估计

| Phase | 时间 | 重要性 |
|-------|------|--------|
| Phase 1: 完整实现 | 1-2 天 | **CRITICAL** |
| Phase 2: 质量验证 | 0.5 天 | **CRITICAL** |
| Phase 3: 优化耗时 | 1-2 天 | Optional |

**总计**: 2-5 天

---

## 成功标准

### Phase 1
- [ ] Self-Study 实现并测试
- [ ] OMP 实现并测试
- [ ] 集成到主流程
- [ ] 单元测试通过

### Phase 2
- [ ] Perplexity 增加 < 5%
- [ ] 压缩比 ≥ 4x
- [ ] 内存节省 ≥ 75%

### Phase 3（可选）
- [ ] Query Generation 耗时 < 原始的 50%
- [ ] 质量下降 < 2%

---

## 下一步行动

1. **立即开始 Phase 1.1**: 实现 Self-Study
2. 不追求速度，追求正确性
3. 每个步骤都要单元测试
4. Phase 1 完成后再考虑 Phase 2

---

*Updated: 2026-03-22*
*Status: Ready to implement*
