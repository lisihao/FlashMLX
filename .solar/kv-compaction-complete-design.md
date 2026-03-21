# KV Cache Compaction for MLX - 完整项目设计

**版本**: v2.0 Final
**日期**: 2026-03-21
**状态**: 设计审批中

---

## 目录

1. [项目背景](#1-项目背景)
2. [技术原理](#2-技术原理)
3. [核心价值定位](#3-核心价值定位)
4. [设计目标和约束](#4-设计目标和约束)
5. [技术方案](#5-技术方案)
6. [实施路线图](#6-实施路线图)
7. [风险和权衡](#7-风险和权衡)
8. [成功标准](#8-成功标准)
9. [资源和时间估算](#9-资源和时间估算)

---

## 1. 项目背景

### 1.1 问题描述

**LLM长文本处理的内存瓶颈**：

```
Qwen3-4B, 60K context:
- 模型权重: ~8GB
- KV Cache: 40 layers × 32 heads × 60K tokens × 128 dim × 2 (K+V) × 2 bytes
           = ~15GB
- Activations: ~2GB
Total: ~25GB

Apple Silicon (M4 Pro 48GB):
- 可用内存: 48GB
- 可处理最长context: ~80K tokens
- 并发用户数: 1-2个
```

**核心矛盾**：
- 用户需求：处理超长文档（100K+ tokens）
- 硬件限制：Apple Silicon统一内存有限
- 现有方案：Sliding Window（丢失长距离依赖）、Quantization（压缩比有限）

### 1.2 方案来源

**论文**: [Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2602.16284)
**作者**: Adam Zweiger, Xinghong Fu, Han Guo, Yoon Kim
**发表**: 2026年2月

**核心思想**：
- 不在token空间压缩（删除、摘要）
- 在latent KV space压缩（重建attention行为）
- 通过数学优化保持attention输出接近

**论文声称**：
- 压缩比：10-50x（取决于数据集和任务）
- 质量损失：部分任务 < 5%，部分任务更大
- 时间成本：8-139s（取决于query generation方法）

**媒体报道**（VentureBeat）：
- 标题："50x without accuracy loss"
- **注意**：这是媒体口径，不是论文原话

### 1.3 项目动机

**为什么在MLX上做这个**：

1. **社区空白**：MLX没有KV cache压缩方案
2. **硬件契合**：Apple Silicon统一内存架构，内存优化价值大
3. **技术挑战**：PyTorch → MLX移植，涉及数学优化算法
4. **研究价值**：验证Attention Matching在不同框架的可行性

**不做的理由**：
- 如果只追求速度，FlashAttention更成熟
- 如果只追求内存，Quantization更简单
- 如果论文效果不可复现，价值有限

---

## 2. 技术原理

### 2.1 Attention Matching核心思想

**原始Attention**：
```
output = softmax(Q·K^T / sqrt(d)) · V

其中:
K: (seq_len, head_dim) - 原始keys
V: (seq_len, head_dim) - 原始values
Q: (1, head_dim) - query
```

**压缩后Attention**：
```
output ≈ softmax(Q·C1^T / sqrt(d) + beta) · C2

其中:
C1: (budget, head_dim) - 压缩后的keys (budget << seq_len)
beta: (budget,) - attention bias (补偿信息损失)
C2: (budget, head_dim) - 压缩后的values
```

**目标**：通过优化C1, beta, C2，使压缩后的attention输出尽量接近原始输出。

### 2.2 算法流程（论文方法）

#### Step 1: Generate Queries

**目的**：生成代表性queries，用于指导压缩

**方法**：
- **Repeat-prefill**（8s）：重用prefill时的queries
- **Self-study**（139s）：模型自己生成queries
- **On-policy**（更慢）：根据生成任务定制queries

**选择**：Repeat-prefill（最快且有效）

#### Step 2: Select Keys (C1)

**目的**：从T个keys中选出budget个最重要的

**方法**：
- **Highest Attention Keys**（3s）：计算attention scores，选top-k
- **OMP Keys**（565s）：Orthogonal Matching Pursuit，更精确但极慢

**选择**：Highest Attention Keys

#### Step 3: Fit Beta (NNLS)

**目的**：拟合attention mass，使 softmax(Q·C1^T + beta) ≈ softmax(Q·K^T)

**数学**：
```
exp_scores = exp(Q·K^T / sqrt(d))
target = exp_scores.sum(dim=1)  # 每个query的总mass

M = exp_scores[:, selected_indices]  # 选中keys的exp scores
Solve: min ||M·B - target||^2, s.t. B >= 0  (NNLS)

beta = log(B)
```

**求解器**：
- **Clamped Least Squares**（快，iters=0）：lstsq + clamp
- **Projected Gradient Descent**（慢，iters=100）：迭代优化

**时间**：2.2s

#### Step 4: Fit C2 (LSQ)

**目的**：拟合attention outputs，使 softmax(Q·C1^T + beta)·C2 ≈ softmax(Q·K^T)·V

**数学**：
```
X = softmax(Q·C1^T / sqrt(d) + beta)  # (n_queries, budget)
Y = softmax(Q·K^T / sqrt(d)) · V      # (n_queries, head_dim)

Solve: X·C2 = Y  (Least Squares)

可选: Ridge regression (C2 = (X^T X + λI)^{-1} X^T Y)
```

**求解器**：
- **Lstsq**（默认）：torch.linalg.lstsq
- **Cholesky**（推荐）：Cholesky分解，更快更稳定
- **Pinv**（慢）：伪逆

**时间**：1.8s

### 2.3 PyTorch实现要点

**文件结构**（参考实现）：
```
compaction/
├── compaction_methods/
│   ├── base.py - FullCacheCompactionAlgorithm基类
│   └── global_highest_attention_keys.py - 全局压缩算法
├── algorithms/
│   ├── base.py - CompactionAlgorithm基类，NNLS/LSQ实现
│   └── highest_attention_keys.py - Per-head压缩算法
models/
├── cache.py - CompactedPrefixCache实现
├── generate.py - 生成工具，chunked_prefill
└── qwen3/modeling_qwen3.py - Qwen3集成
```

**关键类**：
```python
class CompactedPrefixLayer:
    """单层cache"""
    keys: Tensor    # (batch, num_heads, budget, head_dim) - C1
    beta: Tensor    # (batch, num_heads, budget) - beta
    values: Tensor  # (batch, num_heads, budget, head_dim) - C2

class CompactedPrefixCache:
    """多层cache容器"""
    layers: List[CompactedPrefixLayer]
    original_seq_len: int  # 保留logical length（重要！）
```

**Logical length保留**：
- 压缩后physical length变小（budget << seq_len）
- 但logical length不变（position IDs不变）
- 新token的position = original_seq_len + offset
- 保证RoPE等位置编码正确

---

## 3. 核心价值定位

### 3.1 主要价值：内存 → 容量提升

**量化收益**：
```
压缩比 5x:
- 原始: 60K tokens → 15GB cache
- 压缩: 60K tokens → 3GB cache
- 收益: 可处理 5x longer context (300K tokens)
        或 5x more concurrent users
```

**适用场景**：
1. **超长文档处理**：法律合同、学术论文、代码仓库
2. **多用户并发**：边缘设备serving多个用户
3. **内存受限设备**：Apple Silicon, Jetson等

### 3.2 次要价值：TG吞吐量提升

**理论分析**：
```
Token Generation每步:
1. Q @ K^T  ← K从(seq_len, d)变成(0.2*seq_len, d)  ⬇️ 5x操作量
2. softmax  ← 计算量减少5x
3. attn @ V ← V从(seq_len, d)变成(0.2*seq_len, d)  ⬇️ 5x操作量

理论加速: ~5x (如果attention是瓶颈)
```

**实际情况**（需验证）：
- 如果attention占TG 50%时间 → 实际加速 2.5x
- 如果attention占TG 20%时间 → 实际加速 1.2x
- 需要profiling确定

**ROI分析**：
```
Scenario 1: 单次生成100 tokens
- 压缩成本: 4s
- TG加速: 100 tokens × 1ms saved = 0.1s
- Net: 亏3.9s ❌

Scenario 2: Cache复用，10用户各生成100 tokens
- 压缩成本: 4s (一次性)
- TG加速: 10 × 100 × 1ms = 1s
- Net: 亏3s ❌

Scenario 3: Cache serving，100用户各生成1000 tokens
- 压缩成本: 4s
- TG加速: 100 × 1000 × 1ms = 100s
- Net: 赚96s ✅
```

**结论**：TG吞吐量提升**仅在cache复用场景有价值**

### 3.3 成本：压缩时间Overhead

**论文数据**（60K tokens, H200）：
- Repeat-prefill: 8s
- Self-study: 139s
- OMP: 565s

**可接受范围**（用户体验）：
- < 1s: 无感知 ✅
- 1-3s: 可接受 ✅
- 3-10s: 需要progress bar ⚠️
- > 10s: 体验差 ❌

**优化目标**：压缩时间降到 < 2s

### 3.4 与其他方案对比

| 方案 | 内存减少 | 质量损失 | 适用场景 | MLX实现 |
|------|----------|----------|----------|---------|
| **Sliding Window** | 50% | 显著 | 短context | ✅ 有 |
| **Token Dropping** | 70% | 中等 | 摘要任务 | ❌ 无 |
| **Quantization (KV)** | 75% (fp16→int4) | < 1% | 所有场景 | ✅ 有 |
| **Attention Matching** | 80% (5x) | 需验证 | 长context QA | ❌ 无 |

**组合方案**（最大化收益）：
```
Quantization (4x) + Attention Matching (5x) = 20x总压缩

示例:
- 原始: fp16, 60K tokens → 15GB
- Quant: int4, 60K tokens → 3.75GB
- Quant+Compact: int4, 12K tokens → 0.75GB

收益: 可处理 20x longer context 或 20x batch size
```

---

## 4. 设计目标和约束

### 4.1 核心设计原则（来自用户反馈）

**Priority 1: 输出质量**
- 数学正确性 > 速度
- 与PyTorch参考实现对齐
- 质量可量化、可解释

**Priority 2: PP & TG吞吐量**
- 压缩时间优化（降低overhead）
- TG性能提升（cache变小的收益）

**Priority 3: 工程可用性**
- 集成mlx-lm（显式API，不自动触发）
- 用户可控（ratio, method参数）

### 4.2 设计目标（务实版）

**不追求**（过于乐观）：
- ❌ 10x压缩（时间成本太高）
- ❌ 50x压缩（媒体夸张，论文未普遍达到）
- ❌ < 5%质量损失（论文数据，不保证在所有任务）
- ❌ 10-15天完成（低估难度）

**追求**（可实现）：
- ✅ 5x压缩（务实，时间可接受）
- ✅ < 10%质量损失（单层单头数学验证）
- ✅ < 2s压缩时间（用户可接受）
- ✅ 3周研究原型（含质量验证）

### 4.3 约束条件

**技术约束**：
- MLX API限制（缺少cholesky_solve等）
- 需要手动实现部分数学算法
- Lazy evaluation需要显式eval

**质量约束**：
- 与PyTorch对齐（误差 < 1e-5）
- 通过多种压缩比测试（5x, 7x, 10x）
- 可视化验证合理性

**性能约束**：
- M4 Pro硬件（Metal GPU）
- 不能牺牲prefill速度太多（< 20% overhead）
- TG吞吐量不能下降

**工程约束**：
- 不破坏mlx-lm现有API
- 用户显式调用（不埋入generate内部）
- 支持Qwen3系列模型（优先）

---

## 5. 技术方案

### 5.1 方案架构

**三层架构**：

```
┌─────────────────────────────────────────┐
│  User API Layer (mlx_lm.compaction)     │
│  - compact_kv_cache()                   │
│  - CompactedPrefixCache                 │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  Compaction Algorithm Layer             │
│  - highest_attention_keys()             │
│  - query_generation                     │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  Math Solver Layer                      │
│  - nnls_solver()                        │
│  - lsq_solver()                         │
│  - cholesky_solve_mlx()                 │
└─────────────────────────────────────────┘
```

### 5.2 核心算法：三种档位

#### **Fast Path**（首选，实用）

**目标**：
- 时间: < 2s (60K tokens, M4 Pro)
- 压缩比: 5x
- 质量: 需验证（< 10%损失）

**算法**：
```python
def compact_fast(K, V, budget):
    """
    Time: ~1-2s for 60K tokens
    Compression: 5x (20% retention)
    """
    # 1. No query generation (0ms)
    #    假设: Keys本身包含足够语义信息

    # 2. Recent + Stride key selection (10ms)
    #    保留最近25% + 远处均匀采样75%
    n_recent = int(budget * 0.25)
    recent_indices = mx.arange(seq_len - n_recent, seq_len)

    stride = (seq_len - n_recent) // (budget - n_recent)
    stride_indices = mx.arange(0, seq_len - n_recent, stride)

    indices = mx.concatenate([stride_indices, recent_indices])
    C1 = K[indices]

    # 3. Beta = 0 (0ms)
    #    跳过NNLS，假设直接attention足够
    beta = mx.zeros(budget)

    # 4. C2 = direct (1ms)
    #    直接复制values
    C2 = V[indices]

    return C1, beta, C2
```

**优势**：
- 极快（~2s）
- 实现简单
- 无query generation依赖

**风险**：
- 质量未知（需Phase A验证）
- Beta=0可能损失较大
- Recent+Stride假设可能不适用所有场景

#### **Medium Path**（平衡）

**目标**：
- 时间: 5-10s
- 压缩比: 7x
- 质量: Good（< 7%损失）

**算法**：
```python
def compact_medium(K, V, budget):
    """
    Time: ~5-10s for 60K tokens
    Compression: 7x
    """
    # 1. PCA queries (per-layer, 50ms × 40 = 2s)
    K_layer_mean = K.mean(axis=0)  # 平均所有heads
    U, S, Vt = mx.linalg.svd(K_layer_mean)
    queries = Vt[:5, :]  # 取top-5主成分

    # 2. L2 norm pre-filtering (100ms)
    #    先用norm筛选候选keys（2x budget）
    key_norms = mx.linalg.norm(K, axis=1)
    candidate_indices = mx.argsort(key_norms)[-(budget * 2):]
    K_candidates = K[candidate_indices]

    # 3. Highest attention on candidates (500ms)
    scores = queries @ K_candidates.T
    attn = mx.softmax(scores, axis=1)
    key_scores = attn.max(axis=0)
    top_in_candidates = mx.argsort(key_scores)[-budget:]
    indices = candidate_indices[top_in_candidates]

    C1 = K[indices]

    # 4. Simplified NNLS (500ms)
    #    Clamped LS (不用PGD)
    beta = nnls_clamped(M, y, lower_bound=1e-12)

    # 5. Simplified LSQ (2s)
    #    只用2个queries，Cholesky求解
    C2 = compute_C2_cholesky(C1, beta, K, V, queries[:2])

    return C1, beta, C2
```

**优势**：
- 有query generation（质量更好）
- 有beta拟合（保持attention mass）
- 时间可接受（5-10s）

**风险**：
- SVD可能慢（需profiling）
- 仍需Phase A验证质量

#### **Slow Path**（论文方法，参考）

**目标**：
- 时间: 8-12s
- 压缩比: 10x
- 质量: Best（论文标准）

**算法**：完全按论文实现（repeat-prefill + highest attention + NNLS + LSQ）

**用途**：
- 质量基准（对比Fast/Medium）
- 离线压缩（不追求实时）

### 5.3 Incremental Compaction（创新方案）

**核心思想**：边prefill边压缩，overhead分摊

```python
def prefill_with_incremental_compaction(
    model, tokens,
    chunk_size=4096,
    compact_interval=4  # 每4个chunks压缩一次
):
    """
    渐进式压缩：用户无感知

    Time: 原始prefill + 20% overhead
    Compression: 5x
    """
    cache = model.make_cache()
    num_chunks = (len(tokens) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(tokens))

        # Prefill this chunk
        logits = model(tokens[start:end], cache=cache)

        # Compact periodically
        if (i + 1) % compact_interval == 0:
            # Fast compact recent chunks only
            compact_start = max(0, start - chunk_size * compact_interval)
            cache = compact_cache_incremental(
                cache,
                start_idx=compact_start,
                end_idx=end,
                target_ratio=0.2
            )

    return cache, logits
```

**优势**：
- 用户体验最佳（无明显等待）
- 可以更频繁压缩（内存峰值更低）
- 适合streaming场景

**挑战**：
- 实现复杂度高（需要partial compaction）
- 需要careful cache管理（logical length）

### 5.4 MLX移植关键点

#### API差异wrapper

```python
# PyTorch → MLX API映射
def lstsq_mlx(A, b):
    """torch.linalg.lstsq兼容wrapper"""
    solution = mx.linalg.lstsq(A, b)[0]  # MLX返回tuple
    return solution

def softmax_mlx(x, axis):
    """torch.softmax兼容wrapper"""
    return mx.softmax(x, axis=axis)  # axis不是dim

def clip_mlx(x, min_val, max_val):
    """torch.clamp兼容wrapper"""
    return mx.clip(x, min_val, max_val)
```

#### 缺失函数实现

```python
def cholesky_solve_mlx(L, b):
    """
    Solve L L^T x = b

    PyTorch有torch.cholesky_solve，MLX需手动实现
    """
    # Forward: L y = b
    y = mx.linalg.solve(L, b)
    # Backward: L^T x = y
    x = mx.linalg.solve(L.T, y)
    return x

def spectral_norm_mlx(M, n_iters=3):
    """
    Estimate ||M||_2 via power iteration

    PyTorch有torch.linalg.matrix_norm(ord=2)，MLX需手动实现
    """
    _, t = M.shape
    u = mx.random.normal((t,))
    u = u / (mx.linalg.norm(u) + 1e-12)

    for _ in range(n_iters):
        v = M @ u
        v = v / (mx.linalg.norm(v) + 1e-12)
        u = M.T @ v
        u = u / (mx.linalg.norm(u) + 1e-12)

    sigma = (u @ (M.T @ (M @ u))) ** 0.5
    return sigma
```

#### 数值稳定性

```python
# 所有数学计算在fp32
def compute_in_fp32(func):
    """Decorator: fp32计算 + 转回原始dtype"""
    def wrapper(*args, **kwargs):
        # Upcast to fp32
        args_fp32 = [arg.astype(mx.float32) if isinstance(arg, mx.array) else arg
                     for arg in args]

        # Compute
        result = func(*args_fp32, **kwargs)

        # Downcast to original dtype
        if isinstance(result, mx.array) and args[0].dtype != mx.float32:
            result = result.astype(args[0].dtype)

        return result
    return wrapper

# Log-sum-exp trick
def safe_softmax(scores):
    """数值稳定的softmax"""
    max_scores = scores.max(axis=-1, keepdims=True)
    exp_scores = mx.exp(scores - max_scores)  # 避免overflow
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)
```

#### JIT编译

```python
@mx.compile
def compact_single_head_jit(K, V, queries, budget):
    """
    JIT编译加速

    注意: 只在Phase B性能优化时使用
           Phase A质量验证时不用（方便调试）
    """
    # ... algorithm
    return C1, beta, C2

# 使用
C1, beta, C2 = compact_single_head_jit(K, V, queries, budget)
mx.eval(C1, beta, C2)  # 显式执行
```

---

## 6. 实施路线图

### 6.1 Phase A: 质量验证（Week 1-2）

**目标**: 验证Fast Path数学正确性和质量

#### A.1 基础数学实现（3-4天）

**任务**:
1. 实现MLX缺失函数
   - `cholesky_solve_mlx()`
   - `spectral_norm_mlx()`
   - API wrapper（lstsq, softmax, clip）

2. 实现NNLS求解器
   - `nnls_clamped()` - Clamped LS (快速版)
   - `nnls_pgd()` - Projected GD (精确版，用于对照)

3. 实现C2求解器
   - `compute_C2_lstsq()` - Lstsq版本
   - `compute_C2_cholesky()` - Cholesky版本（推荐）

**验收**:
```python
# 对比PyTorch结果
def test_nnls_against_pytorch():
    M_torch = torch.rand(50, 20)
    y_torch = torch.rand(50)
    B_torch = pytorch_nnls(M_torch, y_torch)

    M_mlx = mx.array(M_torch.numpy())
    y_mlx = mx.array(y_torch.numpy())
    B_mlx = nnls_clamped(M_mlx, y_mlx)

    error = np.linalg.norm(B_torch.numpy() - B_mlx) / np.linalg.norm(B_torch.numpy())
    assert error < 1e-5  # 相对误差 < 0.001%
```

#### A.2 Fast Path实现（2天）

**任务**:
```python
def compact_single_head_fast(K, V, budget):
    """
    Fast Path完整实现
    """
    # Recent + Stride selection
    indices = select_keys_recent_stride(len(K), budget, recent_ratio=0.25)
    C1 = K[indices]

    # Beta = 0
    beta = mx.zeros(budget)

    # C2 = direct
    C2 = V[indices]

    return C1, beta, C2, indices
```

**验收**:
```python
# Synthetic data测试
K = mx.random.normal((1000, 128))
V = mx.random.normal((1000, 128))
queries = mx.random.normal((10, 128))

C1, beta, C2, indices = compact_single_head_fast(K, V, budget=200)

# 计算误差
output_original = compute_attention_output(queries, K, V)
output_compacted = compute_attention_output(queries, C1, C2, beta)

relative_error = mx.mean((output_original - output_compacted) ** 2) / mx.mean(output_original ** 2)

print(f"Fast Path: {relative_error.item():.4f}")
# 目标: < 0.10 (10%)
```

#### A.3 多场景质量测试（2-3天）

**测试矩阵**:
```python
test_cases = [
    # (seq_len, budget, compression_ratio, expected_error)
    (1000, 500, 2.0, 0.03),   # Easy: 50% compression
    (2000, 400, 5.0, 0.08),   # Medium: 5x compression
    (4000, 400, 10.0, 0.15),  # Hard: 10x compression
    (8000, 400, 20.0, 0.25),  # Extreme: 20x compression
]

for seq_len, budget, ratio, expected in test_cases:
    K, V, queries = generate_synthetic_data(seq_len)
    C1, beta, C2, _ = compact_single_head_fast(K, V, budget)
    error = evaluate_quality(K, V, queries, C1, beta, C2)

    print(f"Ratio {ratio}x: error={error:.4f} (expected <{expected:.4f})")
    assert error < expected
```

**分析维度**:
1. Error vs compression ratio（绘图）
2. Error vs sequence length（绘图）
3. Key selection quality（是否选中高attention keys）
4. Attention heatmap对比（可视化）

**产出**: `Phase-A-Quality-Report.md`

**Go/No-Go判断**:
- ✅ Go: Fast Path在5x压缩下，error < 10%
- ❌ No-Go: Fast Path error > 15%，需要回退到Medium Path

### 6.2 Phase B: 性能优化（Week 2-3）

**前提**: Phase A通过（Fast Path质量可接受）

#### B.1 Profiling（1天）

**任务**:
```python
import time

def profile_compaction(K, V, budget):
    """
    详细profiling
    """
    times = {}

    # 1. Key selection
    start = time.time()
    indices = select_keys_recent_stride(len(K), budget)
    times['key_selection'] = time.time() - start

    # 2. C1提取
    start = time.time()
    C1 = K[indices]
    mx.eval(C1)
    times['c1_extract'] = time.time() - start

    # 3. Beta
    start = time.time()
    beta = mx.zeros(budget)
    mx.eval(beta)
    times['beta'] = time.time() - start

    # 4. C2
    start = time.time()
    C2 = V[indices]
    mx.eval(C2)
    times['c2_extract'] = time.time() - start

    return times

# Run profiling
K = mx.random.normal((60000, 128))
V = mx.random.normal((60000, 128))
times = profile_compaction(K, V, budget=12000)

print("Profiling results:")
for component, t in times.items():
    print(f"  {component}: {t*1000:.2f}ms")
```

**产出**: 瓶颈分析报告

#### B.2 JIT编译优化（2天）

**任务**:
```python
@mx.compile
def compact_single_head_fast_jit(K, V, budget, recent_ratio):
    """JIT编译版本"""
    # ... same as before
    return C1, beta, C2

# Benchmark
import time

# Without JIT
start = time.time()
for _ in range(10):
    C1, beta, C2 = compact_single_head_fast(K, V, 12000)
    mx.eval(C1, beta, C2)
time_no_jit = (time.time() - start) / 10

# With JIT
start = time.time()
for _ in range(10):
    C1, beta, C2 = compact_single_head_fast_jit(K, V, 12000, 0.25)
    mx.eval(C1, beta, C2)
time_jit = (time.time() - start) / 10

print(f"Speedup: {time_no_jit / time_jit:.2f}x")
```

#### B.3 全模型集成（2-3天）

**任务**:
```python
# mlx_lm/compaction/__init__.py
def compact_kv_cache(
    cache,
    model,
    target_ratio=0.2,
    method='fast',  # 'fast' | 'medium' | 'slow'
):
    """
    Compact entire KV cache

    Parameters
    ----------
    cache : mlx_lm KVCache
        Original cache from prefill
    model : mlx_lm Model
        Model instance
    target_ratio : float
        Target compression ratio (0.2 = 5x compression)
    method : str
        Compaction method

    Returns
    -------
    compacted_cache : CompactedPrefixCache
        Compacted cache, can be used for generation
    """
    num_layers = len(cache)
    budget_per_layer = int(cache[0].keys.shape[2] * target_ratio)

    compacted_layers = []
    for layer_idx in range(num_layers):
        K = cache[layer_idx].keys  # (batch, num_heads, seq_len, head_dim)
        V = cache[layer_idx].values

        # Per-head compaction
        C1_heads = []
        beta_heads = []
        C2_heads = []

        for head_idx in range(K.shape[1]):
            K_head = K[0, head_idx, :, :]
            V_head = V[0, head_idx, :, :]

            if method == 'fast':
                C1, beta, C2, _ = compact_single_head_fast(
                    K_head, V_head, budget_per_layer
                )
            # ... other methods

            C1_heads.append(C1[None, None, :, :])
            beta_heads.append(beta[None, None, :])
            C2_heads.append(C2[None, None, :, :])

        # Concatenate
        C1_layer = mx.concatenate(C1_heads, axis=1)
        beta_layer = mx.concatenate(beta_heads, axis=1)
        C2_layer = mx.concatenate(C2_heads, axis=1)

        compacted_layers.append((C1_layer, beta_layer, C2_layer))

    return CompactedPrefixCache(
        compacted_layers,
        original_seq_len=cache[0].keys.shape[2]
    )
```

#### B.4 TG吞吐量验证（1-2天）

**任务**:
```python
def benchmark_tg_throughput(model, tokenizer, context):
    """
    验证TG吞吐量提升
    """
    tokens = tokenizer.encode(context)

    # Original cache
    cache_original = model.make_cache()
    logits = model(mx.array(tokens)[None, :], cache=cache_original)

    # Measure TG speed
    start = time.time()
    for _ in range(100):
        logits = model(mx.array([0])[None, :], cache=cache_original)
        mx.eval(logits)
    time_original = (time.time() - start) / 100

    # Compacted cache
    cache_compacted = compact_kv_cache(cache_original, model, ratio=0.2)

    # Measure TG speed
    start = time.time()
    for _ in range(100):
        logits = model(mx.array([0])[None, :], cache=cache_compacted)
        mx.eval(logits)
    time_compacted = (time.time() - start) / 100

    speedup = time_original / time_compacted
    print(f"TG Speedup: {speedup:.2f}x")
    print(f"  Original: {1000/time_original:.1f} tok/s")
    print(f"  Compacted: {1000/time_compacted:.1f} tok/s")

    return speedup
```

**期望**:
- 理论: 5x speedup (5x compression)
- 实际: 1.5-3x speedup（取决于attention占比）

### 6.3 Phase C: 端到端验证（Week 3）

#### C.1 QA Demo（2天）

**任务**:
```python
# examples/qa_demo_mlx.py
def main():
    # Load model
    model, tokenizer = load("Qwen/Qwen3-4B")

    # Article
    article = """Long article..."""

    # Prefill
    tokens = tokenizer.encode(article)
    cache = model.make_cache()
    logits = model(mx.array(tokens)[None, :], cache=cache)

    # Compact
    print("Compacting cache...")
    start = time.time()
    cache_compacted = compact_kv_cache(cache, model, ratio=0.2)
    compact_time = time.time() - start
    print(f"  Compression: {len(tokens)} -> {int(len(tokens) * 0.2)} tokens")
    print(f"  Time: {compact_time:.2f}s")

    # QA with original cache
    question = "What is the main topic?"
    answer_original = generate_answer(model, tokenizer, question, cache)

    # QA with compacted cache
    answer_compacted = generate_answer(model, tokenizer, question, cache_compacted)

    # Compare
    print("\nOriginal answer:", answer_original)
    print("Compacted answer:", answer_compacted)
```

#### C.2 Memory Benchmark（1天）

**任务**:
```python
import psutil
import os

def measure_memory_usage():
    """
    测量实际内存占用
    """
    process = psutil.Process(os.getpid())

    # Baseline
    mem_before = process.memory_info().rss / 1024**3

    # Load model
    model, _ = load("Qwen/Qwen3-4B")
    mem_model = process.memory_info().rss / 1024**3

    # Prefill
    cache = model.make_cache()
    tokens = mx.random.randint(0, 1000, (60000,))
    logits = model(tokens[None, :], cache=cache)
    mx.eval(cache)
    mem_cache = process.memory_info().rss / 1024**3

    # Compact
    cache_compacted = compact_kv_cache(cache, model, ratio=0.2)
    mx.eval(cache_compacted)
    mem_compacted = process.memory_info().rss / 1024**3

    print(f"Memory usage:")
    print(f"  Model: {mem_model - mem_before:.2f} GB")
    print(f"  Cache (60K tokens): {mem_cache - mem_model:.2f} GB")
    print(f"  Compacted cache: {mem_compacted - mem_cache:.2f} GB")
    print(f"  Reduction: {(mem_cache - mem_compacted) / (mem_cache - mem_model) * 100:.1f}%")
```

#### C.3 文档和示例（1天）

**产出**:
- `README_compaction.md` - 使用指南
- `examples/compaction_demo.py` - 基础示例
- `examples/long_context_demo.py` - 超长文档处理
- API文档

---

## 7. 风险和权衡

### 7.1 技术风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| **Fast Path质量不够** | 中 | 高 | Phase A验证，不通过则回退Medium |
| **MLX API限制** | 低 | 中 | 手动实现缺失函数，已验证可行 |
| **TG加速不明显** | 中 | 中 | Profiling验证，调整预期 |
| **内存减少不足80%** | 低 | 高 | 数学上确定，风险低 |

### 7.2 质量 vs 性能权衡

**Trade-off矩阵**:

| 方案 | 压缩时间 | 压缩比 | 质量损失 | 适用场景 |
|------|----------|--------|----------|----------|
| **Fast** | < 2s ✅ | 5x | ~10% | 实时场景 |
| **Medium** | 5-10s | 7x | ~7% | 可接受等待 |
| **Slow** | 8-12s | 10x | ~5% | 离线压缩 |
| **Incremental** | 分摊 ✅ | 5x | ~10% | Streaming |

**推荐策略**:
- 默认: Fast Path（实用）
- 高质量需求: Medium Path
- 离线场景: Slow Path（论文方法）

### 7.3 工程权衡

**显式API vs 自动触发**:

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **显式调用** | 用户可控，易调试 | 需要手动调用 | ✅ 采用 |
| **自动触发** | 无感知，简单 | 难调试，magic | ❌ 不采用 |

**Per-head vs Per-layer**:

| 方案 | 时间 | 质量 | 选择 |
|------|------|------|------|
| **Per-head** | 慢（1280 heads） | 最优 | ❌ 太慢 |
| **Per-layer** | 快（40 layers） | 次优 | ✅ 采用 |

### 7.4 场景适用性

**适合**:
- ✅ 超长文档（100K+ tokens）
- ✅ 多用户cache serving
- ✅ 内存受限设备（Apple Silicon）
- ✅ QA任务（保留关键信息）

**不适合**:
- ❌ 短文本（< 10K tokens）- 压缩overhead不值得
- ❌ 生成任务（摘要、创作）- 质量损失明显
- ❌ 单次生成100 tokens - ROI负

---

## 8. 成功标准

### 8.1 Phase A成功标准（质量）

**数学正确性**:
```python
# 1. 与PyTorch对齐
assert ||B_mlx - B_torch|| / ||B_torch|| < 1e-5

# 2. Synthetic data测试
for ratio in [2x, 5x, 10x]:
    error = test_compaction(ratio)
    assert error < threshold[ratio]
    # 2x: <3%, 5x: <10%, 10x: <20%

# 3. 多样化测试
# 100组随机数据，全部通过
```

**可解释性**:
```python
# 1. Attention heatmap对比（可视化）
plot_attention_comparison(original, compacted)

# 2. Key selection分析
# - 选中keys的attention分布
# - 是否选中了high-attention positions

# 3. Error breakdown
# - Per-query error
# - 哪些query压缩效果好，哪些不好
```

**产出**: `Phase-A-Quality-Report.md`

### 8.2 Phase B成功标准（性能）

**压缩时间**:
```
60K tokens, M4 Pro:
- Fast Path: < 2s ✅
- Medium Path: < 10s ✅
```

**TG吞吐量**:
```
压缩5x后:
- TG speedup: > 1.5x ✅ (至少)
- 或 TG time不增加（压缩overhead在生成中摊销）
```

**内存减少**:
```
压缩5x后:
- 内存减少: > 70% ✅
```

**产出**: `Phase-B-Performance-Report.md`

### 8.3 Phase C成功标准（可用性）

**端到端Demo**:
```python
# 可以运行的Demo
python examples/qa_demo_mlx.py --model Qwen/Qwen3-4B --ratio 0.2

# 输出:
# - Compression time: X.Xs
# - Memory reduction: XX%
# - Original answer: ...
# - Compacted answer: ...
# - Quality: acceptable/not acceptable
```

**文档完整**:
- [ ] README with usage examples
- [ ] API documentation
- [ ] Limitations and known issues

**产出**: 可运行的研究原型

### 8.4 项目整体成功标准

**Minimum Viable Product (MVP)**:
- ✅ Fast Path质量可接受（< 10% error at 5x compression）
- ✅ 压缩时间 < 2s
- ✅ 内存减少 > 70%
- ✅ 与PyTorch数学对齐
- ✅ 可运行Demo

**Good to Have**:
- ✅ TG吞吐量提升 > 1.5x
- ✅ Medium Path实现
- ✅ Incremental compaction
- ✅ 多模型支持（Qwen3, Llama3）

**Not Required (Phase 1)**:
- ❌ 10x压缩（Fast Path只做5x）
- ❌ Self-study query generation（太慢）
- ❌ On-policy queries（复杂）
- ❌ 自动触发（Phase 1只做显式API）

---

## 9. 资源和时间估算

### 9.1 时间估算（保守）

| Phase | 任务 | 时间 | 依赖 |
|-------|------|------|------|
| **Phase A** | 质量验证 | | |
| A.1 | 基础数学实现 | 3-4天 | - |
| A.2 | Fast Path实现 | 2天 | A.1 |
| A.3 | 多场景测试 | 2-3天 | A.2 |
| A.小计 | | **7-9天** | |
| **Phase B** | 性能优化 | | |
| B.1 | Profiling | 1天 | A完成 |
| B.2 | JIT编译 | 2天 | B.1 |
| B.3 | 全模型集成 | 2-3天 | B.2 |
| B.4 | TG吞吐量验证 | 1-2天 | B.3 |
| B.小计 | | **6-8天** | |
| **Phase C** | 端到端验证 | | |
| C.1 | QA Demo | 2天 | B完成 |
| C.2 | Memory Benchmark | 1天 | C.1 |
| C.3 | 文档和示例 | 1天 | C.2 |
| C.小计 | | **4天** | |
| **总计** | | **17-21天** | |
| **Buffer** | Debug + 迭代 | **3-4天** | |
| **最终** | | **3-4周** | |

### 9.2 关键里程碑

**Week 1 结束**:
- ✅ Phase A.1-A.2完成
- ✅ Fast Path可以运行
- ✅ 初步质量数据

**Week 2 结束**:
- ✅ Phase A.3完成
- ✅ 质量报告
- ✅ Go/No-Go决策

**Week 3 结束**:
- ✅ Phase B完成
- ✅ 性能优化
- ✅ 全模型集成

**Week 4 结束**:
- ✅ Phase C完成
- ✅ Demo可运行
- ✅ 文档完整

### 9.3 资源需求

**硬件**:
- MacBook Pro M4 Pro (48GB)
- 用于开发和测试

**软件**:
- MLX (最新版)
- mlx-lm (最新版)
- PyTorch (参考实现对照)

**数据**:
- 合成数据（自己生成）
- 测试文档（公开数据集）

### 9.4 风险buffer

**预留时间**（3-4天）用于：
- Phase A质量不达标 → 回退Medium Path
- MLX API问题 → 寻找workaround
- TG加速不明显 → 调整策略
- Bug修复和调试

---

## 10. 决策点和审批

### 10.1 Phase A结束决策点

**Go条件**:
- Fast Path在5x压缩下，error < 10%
- 与PyTorch数学对齐（误差 < 1e-5）
- 通过100组随机测试

**No-Go条件**:
- Fast Path error > 15%
- 数学实现bug无法修复

**No-Go应对**:
- 回退到Medium Path
- 重新评估时间（+1周）

### 10.2 Phase B结束决策点

**Go条件**:
- 压缩时间 < 3s（稍放宽）
- TG吞吐量不下降
- 内存减少 > 70%

**No-Go条件**:
- 压缩时间 > 10s（不可用）
- TG吞吐量下降 > 20%

**No-Go应对**:
- 定位为"离线压缩"方案
- 或者focus Incremental compaction

### 10.3 最终审批标准

**研究原型交付标准**:
- [ ] Fast Path质量报告
- [ ] 性能benchmark报告
- [ ] 可运行Demo
- [ ] API文档
- [ ] 已知限制说明

**不要求**:
- [ ] 生产就绪
- [ ] 多模型支持
- [ ] 完整测试覆盖

---

## 11. 下一步行动

### 11.1 立即行动（批准后）

**Day 1**:
```bash
cd /Users/lisihao/FlashMLX
mkdir -p mlx_lm/compaction
mkdir -p mlx_lm/compaction/solvers
mkdir -p tests/compaction
mkdir -p examples/compaction

# 创建基础文件
touch mlx_lm/compaction/__init__.py
touch mlx_lm/compaction/base.py
touch mlx_lm/compaction/fast.py
touch mlx_lm/compaction/solvers/nnls.py
touch mlx_lm/compaction/solvers/lsq.py
touch mlx_lm/compaction/solvers/utils.py
```

**Day 1-2**:
- 实现`cholesky_solve_mlx()`
- 实现`spectral_norm_mlx()`
- 单元测试（对比numpy）

**Day 3-4**:
- 实现`nnls_clamped()`
- 单元测试（对比PyTorch）

### 11.2 每周Review

**Week 1 Friday**:
- Review Phase A.1-A.2进度
- 评估质量初步数据
- 调整计划（如需要）

**Week 2 Friday**:
- Review Phase A.3完整质量报告
- **Go/No-Go决策**
- 规划Phase B（如Go）

**Week 3 Friday**:
- Review Phase B性能数据
- 评估是否满足成功标准
- 规划Phase C

**Week 4 Friday**:
- 项目总结
- Demo演示
- 下一步roadmap

---

## 12. 附录

### 12.1 参考文献

1. **论文**: Fast KV Compaction via Attention Matching
   - https://arxiv.org/abs/2602.16284

2. **参考实现**: https://github.com/adamzweiger/compaction
   - PyTorch实现
   - 包含Qwen3, Llama, Gemma3支持

3. **相关技术**:
   - FlashAttention: https://arxiv.org/abs/2205.14135
   - Quantization: https://arxiv.org/abs/2306.00978

### 12.2 术语表

| 术语 | 定义 |
|------|------|
| **KV Cache** | Key-Value cache，存储attention的keys和values |
| **Compaction** | 压缩，减少cache大小但保持功能 |
| **Attention Matching** | 通过优化使压缩后attention行为接近原始 |
| **NNLS** | Non-Negative Least Squares，非负最小二乘 |
| **LSQ** | Least Squares，最小二乘 |
| **Logical Length** | 逻辑长度，压缩后保留的原始序列长度信息 |
| **Physical Length** | 物理长度，压缩后实际存储的token数量 |
| **PP** | Prefill，预填充阶段 |
| **TG** | Token Generation，生成阶段 |

### 12.3 文档组织

**已创建文档**:
- `kv-compaction-mlx-plan.md` - 初始计划（已过时）
- `kv-compaction-core-algorithms.md` - 核心算法实现细节
- `kv-compaction-complete-design.md` - 本文档（完整设计）

**待创建文档**（Phase A-C产出）:
- `Phase-A-Quality-Report.md`
- `Phase-B-Performance-Report.md`
- `README_compaction.md`
- `API_documentation.md`

---

## 13. 总结

### 13.1 核心要点

1. **价值定位**：内存→容量提升（主要），TG吞吐量提升（次要）
2. **设计原则**：质量优先（NO1），性能优化（NO2）
3. **技术方案**：Fast/Medium/Slow三档，推荐Fast Path（5x压缩，<2s）
4. **实施路径**：Phase A质量验证 → Phase B性能优化 → Phase C端到端
5. **成功标准**：Fast Path error < 10%，压缩时间 < 2s，内存减少 > 70%
6. **时间估算**：3-4周研究原型
7. **关键风险**：Fast Path质量不够（有Medium Path fallback）

### 13.2 与初始方案的变化

| 维度 | 初始方案（v1.0） | 最终方案（v2.0） | 原因 |
|------|------------------|------------------|------|
| **压缩比** | 10x（50x媒体说法） | 5x（务实） | 时间成本考虑 |
| **时间目标** | 10-15天 | 3-4周 | 低估了复杂度 |
| **质量标准** | <5%损失 | <10%损失 | 更现实 |
| **压缩时间** | 8-12s（论文） | <2s（优化） | 用户体验需求 |
| **优先级** | 未明确 | 质量NO1，性能NO2 | 用户反馈 |
| **方案** | 单一方案 | Fast/Medium/Slow | 灵活性 |

### 13.3 关键学习

从讨论中学到的：

1. **不要过度承诺**：50x无损是媒体口径，不是论文标准
2. **query generation是大头**：8s占66%，不是NNLS/LSQ
3. **质量优先于速度**：Phase A完全focus质量，Phase B才优化性能
4. **务实的trade-off**：压缩比10x→5x，时间8s→2s，可接受
5. **用户体验重要**：Incremental compaction（分摊overhead）价值大
6. **明确价值主张**：内存→容量（确定），TG加速（需验证）

---

**设计完成日期**: 2026-03-21
**下一步**: 等待审批，批准后开始Phase A.1实施

**审批人签字**: _____________

**批准日期**: _____________
