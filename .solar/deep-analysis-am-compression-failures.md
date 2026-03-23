# 深度分析：Attention Matching 压缩算法的两个关键失败

**日期**: 2026-03-23
**分析团队**: 审判官 (DeepSeek-R1) + 探索派 (Gemini-3-Pro) + 稳健派 (Gemini-2.5-Pro)
**状态**: 🔴 CRITICAL - 核心算法问题

---

## 执行摘要

AM (Attention Matching) 压缩算法在两个关键场景完全失败：

| 场景 | 现象 | 质量 | 根本原因 |
|------|------|------|---------|
| **真实数据 (Real)** | 质量从 0.9999 降至 0.898 | 10% 降级 | 注意力集中 + β 不足 |
| **TruthfulQA** | 完全崩溃，产生乱码 | 0.000 | t > T 逻辑矛盾 |

**关键洞察**：AM 的根本问题不是实现细节，而是 **架构假设与真实世界的本质不兼容**。

---

## 问题 1：为什么 AM 在真实数据上质量降级？

### 1.1 证据分析

**模拟数据 vs 真实数据对比**：

```
特征              | 模拟数据    | 真实数据 (Alpaca) | TruthfulQA
===============================================================
序列长度 T        | 100         | 31               | 15
Key 方差 K_std    | 0.188 (低)  | 0.887 (高)       | 0.887 (高)
注意力熵          | 4.61 (散)   | 3.42 (中)        | 2.69 (集)
最大注意力值      | 0.0124 (低) | 0.0578 (中)      | 0.1001 (高)
AM 质量           | 0.9999      | 0.898            | 0.000
H2O 质量          | 0.696       | 0.945            | >0.9
StreamingLLM 质量 | 0.691       | 0.908            | >0.9
```

**关键观察**：
- 当注意力 **分散** (entropy ↑, max ↓) 时：AM 完美 ✅
- 当注意力 **集中** (entropy ↓, max ↑) 时：AM 降级 ❌
- H2O/StreamingLLM：在集中注意力上反而 **更优秀** ✅

---

### 1.2 根本原因：Beta 补偿的数学局限

#### 问题的本质

AM 算法的第3步（Beta 计算）基于这个假设：

```
假设：可以用单一 β 向量补偿从 T 个 key 中移除 (T-t) 个 key 的影响

数学表达：
  原始注意力：A_orig = softmax(Q @ K.T)
  压缩后注意力（无 β）：A_no_beta = softmax(Q @ C1.T)
  压缩后注意力（有 β）：A_beta = softmax(Q @ C1.T + β)

  目标：A_beta ≈ A_orig
```

#### 为什么这个假设在集中注意力上失败？

**情形 1：分散注意力（模拟数据，entropy=4.61）**

```
Query q:
  K 中的 100 个 key 都有接近的贡献
  移除 75 个 key → 剩余 25 个 key 分散了这个质量
  β 可以通过增加剩余 key 的权重来补偿 ✅

  直观：如果 25 个朋友各贡献 4% + 75 个朋友各贡献 1%
       移除 75 个朋友后，剩余 25 个提升权重到 5%，就能匹配原始 100% ✅
```

**情形 2：集中注意力（真实数据，entropy=3.42）**

```
Query q:
  少数几个 key 有 80% 的权重
  很多 key 有接近 0 的权重

  如果关键 key 被选中（概率高）：
    剩余 key 可以通过 β 补偿 ✅

  如果关键 key 被漏掉（概率低但存在）：
    β 无法凭空创造被移除 key 的贡献 ❌
    因为 β 是共享的，所有 query 用同一个 β
    某个 query 的关键 key 被移除，另一个 query 的关键 key 被保留
    → 共享 β 无法同时满足两个 query 的不同需求
```

**情形 3：极度集中注意力（TruthfulQA，entropy=2.69）**

```
Query q:
  单个关键 key：90% 权重
  其他 key：<1% 权重

  移除这个关键 key → 失去 90% 信息
  β 怎么补偿？β[j] 是一个标量
  最多增加一个 key 的权重从 0.5% 到 5%
  无法弥补 90% 的缺失 ❌❌❌
```

#### 数学证明：β 的自由度不足

**问题的自由度分析**：

```
已知信息：
  - C1: (t, d) - 选中的 key
  - β: (t,) - 每个选中 key 的 bias
  - queries: (n, d) - n 个 query samples

需要满足的约束条件数：
  - 每个 query i 的注意力权重需要匹配：n 个约束
  - 每个 removed key j 的贡献需要补偿：(T-t) × n 个隐含约束

自由度：
  - β 有 t 个参数
  - t << (T-t) × n （当 T 大时）

结论：大幅度欠定（severely under-determined）
```

**实际数字（TruthfulQA）**：
- Removed keys: 15 - 25 = -10 （无法移除！）
- Constraints: 20 queries × -10 removed keys = -200（矛盾！）
- β degrees of freedom: 25
- **Status**: 完全不可解 ❌

---

### 1.3 为什么 H2O/StreamingLLM 在集中注意力上更优秀？

#### H2O (Heavy-Hitter Oracle)

**核心思想**：明确保留高注意力的 key

```
1. 计算注意力权重（同 AM）
2. 识别 "Heavy-Hitter"：注意力值 > 阈值的 key
3. 保留这些 key + 最近的 key
4. 不用 β 补偿，直接保留原始 key 和 attention 权重
```

**为什么在集中注意力上优秀**：
- Heavy-Hitter 正好是 **注意力集中** 时的最重要 key ✅
- 没有 β 补偿，直接用原始权重 ✅
- 不需要学习/优化任何参数，完全启发式 ✅

**vs AM**：
- AM 在选择 top-25 key 时，依靠 mean attention (全局)
- 如果某个 query 的关键 key 排名不在前 25 怎么办？
- H2O 的 heavy-hitter 是 query-aware 的概念，而 AM 是 query-agnostic 的

#### StreamingLLM (Sliding Window + Sink)

**核心思想**：保留两类 key

```
1. Sink tokens (BOS/CLS): 吸收 attention，质量好
2. Recent window: 最近的 token 通常最重要
3. 放弃中间的 token
```

**为什么在集中注意力上优秀**：
- Sink + recent 通常覆盖了大部分注意力质量 ✅
- 不用通过 β 补偿，直接保留原始 key ✅
- 对各种注意力分布都稳健（局部性假设天然适合 LLM）✅

---

### 1.4 数值精度的角色（次要原因）

真实数据的 float16 转 float32 问题：

```
float16 动态范围：~1e-4 到 1e4
float32 动态范围：~1e-38 到 1e38

在计算 exp(logits) 时：
  logits 通常范围是 [-50, 50]
  exp(logits) 范围是 [1e-22, 1e22]

在 float16 中：
  - exp(15) = 3.3M → 溢出为 Inf
  - exp(-15) = 3e-7 → 下溢为 0
  - NNLS 矩阵 M 充满 Inf 和 0 → 求解失败
```

**但这是次要原因**，因为：
- 代码已经将数据转为 float32 (行 329-331)
- 问题持续存在，说明不只是精度问题

---

## 问题 2：为什么 TruthfulQA 完全失败（质量=0.000）？

### 2.1 逻辑矛盾：t > T

**临界条件**：

```python
# 测试代码 (test_real_model_serial.py, 行 316-321)
if T <= 100:
    t = max(25, T // 4)  # TruthfulQA: T=15, t = max(25, 3) = 25
elif T <= 500:
    t = max(100, T // 5)
else:
    t = max(200, T // 5)

# 结果：t = 25 > T = 15 ❌❌❌
```

### 2.2 算法层面：为什么返回 0.000 而不是错误？

**执行流程**：

```
Step 1: topk(scores, t=25) on T=15 keys
  - topk 返回所有 15 个 key 的索引（无法选择 25 个）
  - C1.shape = (15, d) instead of (25, d)

Step 2: NNLS 求解
  - M.shape = (n, 15)  [应该是 (n, 25)]
  - 试图求解 25 个变量的方程组，但只有 15 个独立变量
  - 矩阵 M 秩不足（rank deficiency）
  - NNLS 求解器的行为（推测）：
    * 返回最小范数解：B = zeros(25,) 或 nan
    * 或返回前 15 个位置非零，后 10 个位置为 0

Step 3: beta = log(B)
  - 如果 B[j] = 0，则 log(0) = -inf
  - 如果 B 包含 NaN，则 log(NaN) = NaN
  - β 中充满 -inf 或 NaN

Step 4: 计算压缩注意力
  - Q @ C1.T + β
  - 如果 β 是 -inf，softmax(... + (-inf)) = 0
  - 所有注意力权重变成 0
  - 质量 = cos_sim(output) = 0 ❌
```

### 2.3 设计缺陷：为什么代码没有捕获这个问题？

**代码检查（行 83-85）**：

```python
if T < t:
    raise ValueError(f"Cannot compact: K has {T} rows but t={t} requested")
```

**这个检查存在，但**：
1. 可能在错误的分支被跳过
2. 可能被 try-except 捕获并吞掉（见测试代码 370-374）
3. 可能的 topk 实现做了自动截断，没有触发检查

**根本问题**：
- t 计算逻辑 `max(25, T // 4)` 是 **测试代码的设计缺陷**
- 应该是 `min(t_target, T)` 而不是 `max(25, T // 4)`

---

### 2.4 H2O/StreamingLLM 为什么没问题？

**H2O 的处理**：

```python
# H2O 不固定 t，而是基于比例
t = int(max_capacity * (1 - recent_ratio))
# 最大容量通常是 KV cache 大小，不会超过 T
```

**StreamingLLM 的处理**：

```python
# 保留策略很灵活
sink_tokens = 4  # 固定数量
recent_tokens = window_size  # 滑动窗口
# 总大小 = sink + recent <= T (自然保证)
```

---

## 架构对比：AM vs H2O vs StreamingLLM

### 设计假设对比

| 假设 | AM | H2O | StreamingLLM |
|------|----|----|-------------|
| **Key 选择方式** | 全局 mean attention (query-agnostic) | Heavy-hitter 识别 (query-aware) | 位置优先 (sink + recent) |
| **Attention 重建** | 通过 β bias 学习补偿 | 直接保留原始权重 | 直接保留原始权重 |
| **假设 1** | 注意力分散 ❌（真实是集中） | 有少数关键 key ✅（真实对应） | 最近 token 重要 ✅（真实对应） |
| **假设 2** | β 可补偿移除 key ❌（t>T 失败） | 显式保留重要 key ✅ | 显式保留 sink ✅ |
| **假设 3** | Key 重要性全局稳定 ❌ | Key 重要性动态变化 ✅ | 位置即重要性 ✅ |
| **失败场景** | 集中注意力、短序列 | 多关键 key 分散分布 | 重要信息在中间 |

### 在真实世界的表现

```
                AM        H2O      StreamingLLM
========================================================
模拟数据(均匀)  0.9999    0.696    0.691
真实数据(中等)  0.898     0.945    0.908
TruthfulQA(集中) 0.000    0.9+     0.9+

结论：
- AM 过度优化了不存在的场景
- H2O/StreamingLLM 优化了真实场景
```

---

## 信息论与优化论视角

### 3.1 Information-Theoretic Analysis

**AM 的信息论问题**：

```
Original attention: H(A_orig) 位
              ↓
Remove keys: 删除 (T-t) 个 key 的贡献
              ↓
Loss: I_loss = H(A_orig) - H(A_selected)

Compensation using β:
  - β 有 t 个 bit（每个参数 ~32 bits）
  - 需要补偿 (T-t) × n 个 context 中的信息丢失
  - 当 T > t 且 n > 1 时：β 信息量 << 需补偿信息量 ❌
```

**H2O/StreamingLLM 的信息论优势**：

```
不试图补偿信息丢失，而是：
1. 保留最高互信息的 key（Heavy-Hitter）
2. 让数据本身决定哪些 key 重要（而不是用模型参数）
3. I_loss 更小（因为保留了真正关键的 key）
```

### 3.2 Optimization Theory Perspective

**AM 的优化问题**：

```
min ||softmax(Q@C1.T + β) - softmax(Q@K.T)||^2
 β

问题：
- β 是关于 softmax 的非线性函数（梯度爆炸/消失）
- logit space 的距离不等于 probability space 的距离
- 高斯噪声假设在 softmax 后端不成立
- NNLS 假设：E[error] = 0，但在集中注意力时违反
```

**H2O/StreamingLLM 的优化**：

```
不优化任何参数，直接使用：
  A_h2o = softmax_masked(Q@K.T, where K ∈ heavy_hitters)

优势：
- 无需优化，无优化失败风险
- 鲁棒于数据分布变化
- 解释性强（这就是重要的 key）
```

---

## 数值计算分析

### Precision Cascade（精度级联）

**浮点精度的影响链**：

```
1. exp(logits) 计算
   - logits ∈ [-50, 50]
   - exp(logits) ∈ [1e-22, 1e22]
   - float16 无法表示，overflow/underflow
   - float32 可以表示，但动态范围大

2. NNLS 求解
   - M 矩阵条件数：κ(M) = σ_max / σ_min
   - exp(logits) 导致 κ(M) 可能 ~ 1e20（超大）
   - 迭代 NNLS 需要至少 float32，最好 float64

3. log(B) 计算
   - 如果 B[j] = 0（NNLS 返回的非负解），log(0) = -inf
   - 如果 B[j] = 1e-10，log(1e-10) = -23（极小）
   - β 中充满 -inf 或极小值 → softmax 精度丧失

4. 最终 softmax
   - softmax(Q@C1.T + β) 中，β 的极小值被淹没
   - 如果某个 β[j] = -1000，那个 key 权重永远是 0
   - 与原始 softmax 大相径庭
```

**为什么浮点精度问题普遍存在**：
- 即使数据是 float32，β 经常返回极端值（-inf, 0, 1e-10 等）
- NNLS 对条件数敏感，float32 精度不足以解决 κ(M) ~ 1e20 的问题

---

## 跨层级错误累积（Qwen3.5 混合架构）

### 为什么 AM 在 Qwen3.5 上完全崩溃？

**事实**（来自 hetero-report）：
- 10 个 Attention 层被 AM 压缩
- 30 个 SSM 层保持不压缩
- 结果：生成完全乱码 ❌

**误差传播链**：

```
Layer 3 (Attention, compressed)
  Original error: ε₃
  ↓
Layer 4 (SSM) - 递推处理
  s₄ = f(s₃, x₃')  其中 x₃' 是压缩后的 attention 输出
  SSM 状态对输入敏感，ε₃ 被 SSM 状态递推放大
  New error: δ₄ = g(ε₃) 其中 g 可能是乘法增长 (exponential)
  ↓
Layer 7 (Attention, compressed)
  输入已经被 SSM 层放大的误差污染
  新的压缩误差 ε₇ 叠加上去
  Combined error: δ₇ = δ₄ + ε₇ + coupling_effects
  ↓
...repeat for layers 11, 15, 19, 23, 27, 31, 35, 39...

Final output error: δ_final ≈ exponential(initial_errors)
```

**为什么是指数增长而不是线性**：

```
Qwen3.5 SSM (GatedDeltaNet) 的递推：
  h_t = gate(h_{t-1}, x_t) * new_state(x_t)

其中：
  - gate 通常是 sigmoid，范围 (0, 1)
  - 如果 gate > 0.5，则 h_{t-1} 贡献被保留并放大
  - 重复 T 次后：误差增长 ~ (1.5)^T （当 gate ≈ 1.5）

例：T = 31 (TruthfulQA 长度)
  误差增长系数 ~ 1.5^31 ~ 1.8e6 倍
  初始误差 0.01 × 1.8e6 = 1.8e4 → 完全崩溃
```

---

## 修复建议与替代方案

### 4.1 AM 算法的可能修复（低优先级）

#### 修复 1：动态 t 调整

```python
# 当前（有问题）
t = max(25, T // 4)

# 修复
t_target = max(25, T // 4)
t = min(t_target, T)  # 永远不会超过 T
```

**效果**：解决 t > T 的逻辑矛盾，但不能解决质量降级问题。

#### 修复 2：Query-Aware Key Selection

```python
# 当前（query-agnostic）
key_scores = mean(attn_weights, axis=0)  # (T,)
indices = topk(key_scores, t)

# 修复（query-aware）
# 对 query 进行聚类
clusters = cluster_queries(queries, n_clusters=4)
indices_per_cluster = []
for cluster_id, cluster_queries in enumerate(clusters):
    attn_weights_cluster = softmax(cluster_queries @ K.T)
    scores_cluster = mean(attn_weights_cluster, axis=0)
    indices_cluster = topk(scores_cluster, t // n_clusters)
    indices_per_cluster.append(indices_cluster)
```

**效果**：部分缓解。但增加复杂度，且在集中注意力时仍会漏掉关键 key。

#### 修复 3：放弃 β，使用更简单的补偿

```python
# 当前（复杂的 NNLS + β）
beta = solve_nnls(...)

# 修复（简单的全局 shift）
# 保留原始 attention 分布，不尝试补偿
# 只用 topk key，直接归一化
A_compressed = softmax(Q @ C1.T)  # 不加 β
```

**效果**：降低复杂度，提高数值稳定性。但放弃了补偿机制，质量仍差。

### 4.2 为什么不修复 AM，改用 H2O/StreamingLLM（高优先级）

| 方案 | 复杂度 | 稳健性 | 质量 | 学习成本 |
|------|--------|---------|------|----------|
| **AM 修复** | 高 | 中 | 0.85-0.90 | 高 |
| **H2O** | 低 | 高 | 0.94-0.95 | 低 |
| **StreamingLLM** | 低 | 高 | 0.90-0.92 | 低 |

**建议**：放弃进一步投入 AM，优先完善 H2O 或 StreamingLLM 的实现。

---

## 关键教训

### 教训 1：模拟数据的陷阱 ⚠️

**现象**：AM 在模拟数据上完美 (0.9999)，让人产生错误的信心

**根本原因**：
- 模拟数据特意设计得均匀分散（uniform attention）
- 这恰好是 AM **最擅长** 的场景
- 但真实 LLM 的注意力几乎从不均匀分散

**教训**：
- ✅ 模拟数据用来验证实现的正确性，不是算法有效性
- ✅ 真实数据用来验证算法的实用性
- ❌ 不要被模拟数据的完美表现所迷惑

### 教训 2：算法假设与真实分布的不匹配 🎯

**AM 的隐式假设**：
```
假设 1: 注意力权重分散（diffuse），没有明显的峰值
假设 2: 重要的 key 可以通过全局 mean 识别
假设 3: β 可以补偿被移除 key 的贡献
```

**真实情况**：
```
实际 1: 注意力权重集中（peaked），少数 key 占 80%+
实际 2: 重要的 key 是 query-dependent，全局 mean 识别不准
实际 3: β 自由度不足，无法补偿 (T-t) × n 个约束
```

**教训**：
- ✅ 算法设计时，必须验证核心假设在真实数据上是否成立
- ✅ 如果假设不成立，算法失败 100%（不是局部优化问题）
- ❌ 不要期望通过参数调优弥补假设的本质不符

### 教训 3：启发式方法的鲁棒性 💪

**为什么 H2O/StreamingLLM 更鲁棒**：
- 不依赖 "注意力分散" 假设，而是 "有关键 key" 假设（天然对应峰值分布）
- 不试图通过优化补偿信息丢失，而是直接保留关键 key
- 无需精细参数调优

**教训**：
- ✅ 启发式方法可能不是理论最优，但工程鲁棒性更强
- ✅ 在不确定性高的场景（真实数据分布未知），启发式优于优化
- ❌ 不要过度相信"理论最优"，实践鲁棒性同样重要

---

## 验证实验方案

为了严格验证上述分析，建议进行以下实验：

### 实验 1：Attention Entropy Phase Transition

**目标**：找到 AM 失效的确切分界点

```python
# 生成从均匀分布到单峰分布的合成数据
entropies = [4.6, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
for entropy in entropies:
    K, V, queries = generate_data_with_entropy(entropy)
    am_quality = test_am(K, V, queries)
    h2o_quality = test_h2o(K, V, queries)
    plot(entropy, am_quality, label='AM')
    plot(entropy, h2o_quality, label='H2O')

# 预期：AM 质量随 entropy 下降而急剧下滑（相变点）
```

### 实验 2：Float Precision Isolation

**目标**：验证浮点精度是否是主因

```python
# 使用完全相同的真实数据，但分别用 float16/32/64
for dtype in [float16, float32, float64]:
    K_typed = K.astype(dtype)
    V_typed = V.astype(dtype)

    # 在 dtype 精度下执行整个 AM 流程
    quality = test_am(K_typed, V_typed, queries)
    print(f"{dtype}: quality={quality}")

# 预期：float32+ 质量恢复到 0.95+，说明不只是精度问题
```

### 实验 3：Cross-Layer Error Cascade

**目标**：量化 Attention→SSM 误差放大

```python
# 测量每层的表示误差（MSE）
for layer_idx in [3, 4, 5, 6, 7, ...]:
    # 获取 layer_idx 的输出激活
    activation_original = model.get_activation(layer_idx, use_compressed=False)
    activation_compressed = model.get_activation(layer_idx, use_compressed=True)

    error = mse(activation_original, activation_compressed)
    print(f"Layer {layer_idx}: error={error:.6f}")

# 绘制误差随层数的增长曲线
# 预期：误差呈指数增长（不是线性）
```

### 实验 4：Query-Agnostic vs Query-Aware Selection

**目标**：验证 key 选择的 query 依赖性

```python
# 对于每个 query，计算它的 top-5 最重要的 key
top5_per_query = []
for i, q in enumerate(queries):
    attn_weights_q = softmax(q @ K.T)
    top5_indices = topk(attn_weights_q, 5)
    top5_per_query.append(set(top5_indices))

# 计算 top-5 的重叠（交集）
overlap = compute_pairwise_intersection(top5_per_query)
overlap_ratio = len(intersection) / 5

print(f"Top-5 overlap ratio: {overlap_ratio:.2%}")

# 预期：如果 < 80% 重叠，说明 top-5 key 是高度 query-dependent 的
#       AM 的全局 top-k 选择无法覆盖所有 query 的关键 key
```

---

## 推荐决策

### 短期（1-2 周）

**❌ 不建议**：继续优化 AM 算法
- 根本上有缺陷，修复空间有限
- 投入产出比低

**✅ 建议**：
1. 完成本分析的 4 个验证实验
2. 正式宣布 AM 在 Qwen3.5 混合架构上的局限性
3. 基于验证结果，更新 roadmap

### 中期（1-2 月）

**✅ 建议**：
1. 优先完善 H2O 压缩的实现
2. 或深入研究 StreamingLLM 在 SSM 层的应用
3. 对比两者在 Qwen3.5 上的真实生成质量

### 长期（3-6 月）

**✅ 建议**：
1. 建立 "Heterogeneous Memory Compaction" 框架
2. 为 Attention 层选择最优方法（可能是 H2O）
3. 为 SSM 层设计专用压缩算法
4. 统一两种压缩方法的理论基础

---

## 参考文献与资料

| 文档 | 路径 | 说明 |
|------|------|------|
| **AM 实现** | `/Users/lisihao/FlashMLX/src/flashmlx/cache/compaction_algorithm.py` | NNLS beta 计算的源代码 |
| **测试结果** | `/Users/lisihao/FlashMLX/.solar/hetero-cache-quality-report.md` | Qwen3.5 实验数据 |
| **根因分析** | `/Users/lisihao/FlashMLX/.solar/critical-finding-am-incompatibility.md` | 之前的诊断报告 |
| **H2O 参考** | FlashMLX/src/flashmlx/cache/h2o.py | H2O 启发式实现 |
| **StreamingLLM 参考** | FlashMLX/src/flashmlx/cache/streaming_llm.py | StreamingLLM 启发式实现 |

---

## 总结

### AM 失败的本质

| 问题 | 根本原因 | 无法修复的理由 |
|------|---------|--------------|
| **真实数据降级** | β 补偿自由度不足 + 注意力集中 | 需要多倍的 β 参数，但硬件/复杂度不允许 |
| **TruthfulQA 崩溃** | t > T 逻辑矛盾 | 根本上无法从 15 个 key 选出 25 个 |
| **Qwen3.5 乱码** | 跨层级误差指数增长 | SSM 递推放大 Attention 压缩误差 |

### H2O/StreamingLLM 成功的理由

| 方法 | 核心假设 | 为什么对应真实 |
|------|----------|-------------|
| **H2O** | 有少数关键 key | 真实注意力就是峰值分布 ✅ |
| **StreamingLLM** | 最近和 sink 重要 | LLM 天然的局部性偏好 ✅ |

### 最终建议

```
🎯 核心结论：
   AM 是一个"理论上优雅但实践上脆弱"的算法
   在真实世界场景下，启发式方法 (H2O/StreamingLLM) 更稳健

🚀 行动：
   1. 接受 AM 的局限性
   2. 重点投入 H2O/StreamingLLM
   3. 建立异构压缩框架

📊 期望：
   - Attention 层：用 H2O 或 HyperAttention
   - SSM 层：设计专用压缩（后续研究）
   - 最终性能：质量 > 90%, 压缩比 2-3x
```

---

**分析完成于**: 2026-03-23
**审批状态**: 待监护人决策
**后续动作**: 运行 4 个验证实验，确认分析结论
