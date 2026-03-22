# 混合架构 KV Cache 压缩失败 - 根本原因分析

**日期**: 2026-03-21
**分析来源**: 监护人深度分析
**核心发现**: AM 不是"通用记忆压缩器"，它是"softmax-attention KV 压缩器"

---

## 🔥 核心问题：我们误解了问题的本质

### 我们的错误假设

**假设 1**：KV cache = 模型的全部记忆
- ❌ **错误**：混合架构的记忆不只在 KV cache

**假设 2**：所有层都可以用相同的压缩方法
- ❌ **错误**：不同类型的层需要不同的压缩策略

**假设 3**：Attention Matching 是通用方法
- ❌ **错误**：AM 只适用于标准 softmax attention

**假设 4**：论文方法可以直接应用到所有混合架构
- ❌ **错误**：论文在 Gemma-3 上也只压缩了 global-attention 层

---

## 📊 监护人分析的四个核心观点

### 观点 1: 混合架构的记忆不只在 KV cache ⭐⭐⭐⭐⭐

**问题描述**：

论文方法默认"记忆 = KV cache"，但混合架构里，记忆可能存在于：

| 层类型 | 记忆存储位置 | 是否在 KV cache | 能否用 AM 压缩 |
|--------|-------------|----------------|---------------|
| Standard Attention | Keys, Values | ✅ 是 | ✅ 可以 |
| Linear Attention | 累积状态 (accumulated state) | ❌ 否 | ❌ 不能 |
| Sliding Window | 局部窗口 (local window) | 部分 | ⚠️ 部分可以 |
| SSM/Mamba | conv_state, ssm_state | ❌ 否 | ❌ 不能 |
| Recurrent | 隐状态 (hidden state) | ❌ 否 | ❌ 不能 |

**实际例子（Qwen3.5）**：

```python
# Qwen3.5 的混合架构（40 层）
layers = [
    # 30 个 SSM 层（GatedDeltaNet）
    GatedDeltaNetBlock(
        # 记忆在：
        conv_state,  # (B, kernel_size-1, dim) ← 不在 KV cache
        ssm_state,   # (B, state_dim, dim) ← 不在 KV cache
    ),

    # 10 个 Attention 层
    AttentionBlock(
        # 记忆在：
        keys,    # (B, seq_len, head_dim) ← 在 KV cache ✅
        values,  # (B, seq_len, head_dim) ← 在 KV cache ✅
    )
]
```

**为什么压缩 KV cache 不够？**

```
假设 Qwen3.5 的信息流：

Input → SSM Layer 1 (conv_state₁, ssm_state₁)
      → SSM Layer 2 (conv_state₂, ssm_state₂)
      → ...
      → SSM Layer 30 (conv_state₃₀, ssm_state₃₀)
      → Attention Layer 31 (keys₃₁, values₃₁) ← 我们只压缩这里！
      → Attention Layer 32 (keys₃₂, values₃₂)
      → ...
      → Attention Layer 40 (keys₄₀, values₄₀)
      → Output

问题：
- 我们只压缩了 Attention 层的 KV (layers 31-40)
- 但 SSM 层的 conv_state 和 ssm_state (layers 1-30) 没有压缩
- SSM 层的状态仍然占用大量内存
- 更重要的是：压缩 Attention KV 后，residual stream 开始漂移
- SSM 层收到漂移的输入 → 状态更新错误 → 生成歪掉
```

**具体崩溃路径**：

```
Step 1: 前 30 层（SSM）正常运行
  conv_state 累积历史信息
  residual stream 正常

Step 2: Layer 31-40（Attention）KV cache 被压缩
  4096 tokens → 820 tokens (5x 压缩)
  residual stream 开始漂移（信息丢失）

Step 3: 后续生成时
  新 token 输入 → Layer 1-30 (SSM)
  SSM 收到漂移的 residual stream
  conv_state 更新错误
  累积误差

Step 4: 输出崩溃
  "the the the the..." 重复
  token 数量下降 (487 → 172)
```

**结论**：
- ✅ **正确理解**：混合架构的记忆 = KV cache + conv_state + ssm_state + ...
- ❌ **错误理解**：混合架构的记忆 = KV cache
- 🎯 **解决方向**：需要异构记忆压缩（Heterogeneous Memory Compaction）

---

### 观点 2: Attention mass 在混合层未必存在 ⭐⭐⭐⭐⭐

**论文方法的核心假设**：

AM (Attention Matching) 的 β 偏置依赖于 **attention mass**：

```
Mass(q; K) = Σ exp(q·K^T / √d)

目标：让压缩后的 mass 匹配原始 mass
方法：引入 β 偏置

Compressed Mass: Σ exp(q·C₁^T / √d + β)
Original Mass:   Σ exp(q·K^T / √d)

求解：β = argmin ||Compressed Mass - Original Mass||²
```

**问题：非 softmax attention 没有这个量**

| 层类型 | Attention 形式 | 是否有 Mass | β 是否有效 |
|--------|---------------|-----------|-----------|
| Standard Attention | softmax(Q·K^T)·V | ✅ 有 | ✅ 有效 |
| Linear Attention | φ(Q)·φ(K)^T·V | ❌ 无 | ❌ 无效 |
| SSM/Mamba | State-Space Update | ❌ 无 | ❌ 无效 |
| Kernelized | kernel(Q, K)·V | ⚠️ 不同形式 | ⚠️ 需重新推导 |

**实际例子（Linear Attention）**：

```python
# Standard Attention (有 mass)
def standard_attention(Q, K, V):
    scores = Q @ K.T / sqrt(d)  # (query_len, seq_len)
    mass = exp(scores).sum(axis=1)  # (query_len,) ← 存在！
    attn_weights = softmax(scores, axis=1)  # 归一化
    output = attn_weights @ V
    return output

# Linear Attention (无 mass)
def linear_attention(Q, K, V):
    # 使用 feature map: φ(x) = elu(x) + 1
    Q_feat = elu(Q) + 1  # (query_len, d)
    K_feat = elu(K) + 1  # (seq_len, d)

    # 不经过 softmax，直接计算
    kv = K_feat.T @ V  # (d, d) ← 累积的键值对
    output = Q_feat @ kv  # (query_len, d)

    # 没有 exp(scores)，没有 mass 的概念！
    return output
```

**为什么 β 在混合层失效？**

```
假设 Layer 15 是 Linear Attention：

Original:
  output = φ(Q) @ (φ(K)^T @ V)

Compressed (尝试用 β):
  output = φ(Q) @ (φ(C₁)^T @ C₂) + β ← β 该加在哪里？

问题：
1. 没有 softmax，β 无法通过 exp 映射到权重空间
2. φ(Q) @ (φ(C₁)^T @ C₂) 的结构与标准 attention 不同
3. β 的理论基础（mass matching）完全不适用
```

**SSM/Mamba 更复杂**：

```python
# Mamba/GatedDeltaNet 的状态更新
def mamba_layer(x, conv_state, ssm_state):
    # 卷积部分
    x_conv = conv1d(x, conv_state)  # 使用历史 conv_state

    # SSM 状态更新（递推）
    ssm_state_new = A @ ssm_state + B @ x_conv
    output = C @ ssm_state_new

    # 没有 attention weights！
    # 没有 Q·K^T！
    # 没有 mass！
    return output, conv_state_new, ssm_state_new
```

**结论**：
- ✅ **论文的 β 只对标准 softmax attention 有效**
- ❌ **Linear attention / SSM / Recurrent 不适用**
- 🎯 **解决方向**：每种层类型需要独立的压缩理论

---

### 观点 3: Future concatenation invariance 在混合层不成立 ⭐⭐⭐⭐

**论文的重要卖点**：

AM 保证了 **future concatenation invariance**（附录 A.2）：

```
压缩后的 cache 可以与未来 tokens 拼接，行为保持一致

Original:
  KV = [k₁, k₂, ..., k_n]
  output = Attention(q_future, KV)

Compressed:
  C₁ = [c₁, c₂, ..., c_m]  (m << n)
  output ≈ Attention(q_future, C₁)

Concatenation:
  KV_extended = [KV, k_new]
  C₁_extended = [C₁, k_new]

  Attention(q, KV_extended) ≈ Attention(q, C₁_extended)
  ↑ 这个性质很关键！
```

**为什么这个性质重要？**

在自回归生成时，每生成一个新 token，都需要：
1. 拼接新的 key/value 到 cache
2. 用扩展后的 cache 继续生成

如果拼接后行为不一致，生成就会歪掉。

**这个性质依赖的假设**：

论文附录 A.2 证明，这个不变性依赖于：

1. **Block mixture identity**：
   ```
   softmax([s₁, s₂]) @ [v₁, v₂]
   = α·softmax(s₁)@v₁ + (1-α)·softmax(s₂)@v₂
   ```

2. **Attention 是全局的**：
   - 每个 query 可以看到所有 keys
   - Keys 之间没有依赖关系

**混合架构为什么不满足？**

| 层类型 | 是否全局 | 是否满足 block mixture | Invariance 成立 |
|--------|---------|---------------------|----------------|
| Full Attention | ✅ 全局 | ✅ 满足 | ✅ 成立 |
| Sliding Window | ❌ 局部 | ❌ 不满足 | ❌ 不成立 |
| Linear Attention | ✅ 全局 | ⚠️ 不同形式 | ⚠️ 需重新证明 |
| SSM/Recurrent | ❌ 递推 | ❌ 不满足 | ❌ 不成立 |

**实际例子（Sliding Window）**：

```python
# Sliding Window Attention (窗口大小 = 4)
def sliding_window_attention(q, K, V, window=4):
    output = []
    for i, query in enumerate(q):
        # 只看最近 4 个 keys
        start = max(0, i - window + 1)
        end = i + 1

        local_K = K[start:end]
        local_V = V[start:end]

        scores = query @ local_K.T
        attn = softmax(scores)
        output.append(attn @ local_V)

    return output

# 问题：压缩后 concatenation 不一致
Original KV:  [k₁, k₂, k₃, k₄, k₅, k₆, k₇, k₈]
Compressed C: [c₁, c₃, c₆, c₈]  # 压缩到 4 个

Query at position 7 (看窗口 [4, 5, 6, 7]):
  Original:   看到 [k₄, k₅, k₆, k₇]
  Compressed: 看到 [c₁, c₃, c₆] ← 不一致！

→ Invariance 破坏
```

**SSM/Recurrent 更严重**：

```python
# Recurrent Layer (状态递推)
def recurrent_layer(x, hidden_state):
    # 状态更新依赖历史
    new_state = f(x, hidden_state)
    output = g(new_state)
    return output, new_state

# 问题：状态是递推的，不能简单拼接
Original:
  h₁ = f(x₁, h₀)
  h₂ = f(x₂, h₁)
  h₃ = f(x₃, h₂)
  ...
  h_n = f(x_n, h_{n-1})

Compressed:
  如何压缩 h₁, h₂, ..., h_n？
  不能简单选择一部分，因为有依赖关系！

  如果压缩成 [h₁, h₅, h_n]，那么：
  - h₅ 依赖 h₄, h₃, h₂
  - 你丢掉了中间状态，递推链断了

→ 完全不支持 concatenation
```

**结论**：
- ✅ **Future concatenation invariance 只对全局 attention 成立**
- ❌ **局部窗口 / 递推状态不支持**
- 🎯 **混合架构需要重新设计压缩协议**

---

### 观点 4: 论文自己已经给了信号 ⭐⭐⭐⭐⭐

**关键证据：Gemma-3-12B 的处理方式**

论文第 4.2 节（Experiments）明确说：

> For Gemma-3-12B, we only compact **global-attention layers**,
> leaving **sliding-window layers** unchanged.
> The reported compaction ratio is calculated based on
> **global-attention KV cache only**.

**这说明什么？**

| 模型 | 架构 | 论文的做法 | 为什么 |
|------|------|-----------|--------|
| Llama 3 | 纯 Transformer | 全层压缩 | 所有层都是标准 attention ✅ |
| Mistral | 纯 Transformer | 全层压缩 | 所有层都是标准 attention ✅ |
| Gemma-3-12B | Hybrid (global + sliding) | **只压缩 global 层** | 滑动窗口层不适合压缩 ⚠️ |
| Qwen3.5 | Hybrid (SSM + attention) | ？？？ | **我们尝试全层压缩 → 失败** ❌ |

**论文作者的隐含态度**：

```
作者自己都没敢把 AM 直接套到整个混合架构上！

他们的策略：
- ✅ 识别哪些层是 global attention
- ✅ 只压缩这些层
- ✅ 其他层（sliding window）完全不动

这说明：
- 作者知道 AM 不是通用方法
- 作者知道混合架构有风险
- 作者选择了保守策略
```

**我们的错误**：

```
我们的做法：
- ❌ 没有识别层类型
- ❌ 尝试全层压缩（包括 SSM 层）
- ❌ 假设 AM 是通用方法

结果：
- Qwen3.5 生成崩溃
- "the the the" 重复
- Token 数量下降 65%
```

**结论**：
- ✅ **论文作者自己也只压缩标准 attention 层**
- ❌ **我们不应该比论文作者更激进**
- 🎯 **应该学习论文在 Gemma-3 上的策略**

---

## 🎯 根因排序（按概率）

### A. 把非全注意力层也压了 ⭐⭐⭐⭐⭐（第一嫌疑人）

**证据**：
1. 我们对 Qwen3.5 的所有 40 层都应用了 CompactedKVCache
2. 其中 30 层是 SSM（GatedDeltaNet），10 层是 Attention
3. SSM 层根本不应该用 AM 压缩

**验证方法**：
```python
# 检查：是否只压缩了 Attention 层？
for i, layer in enumerate(model.layers):
    if hasattr(layer, 'self_attn'):
        print(f"Layer {i}: Attention ← 应该压缩")
    elif hasattr(layer, 'linear_attn'):
        print(f"Layer {i}: SSM ← 不应该压缩")
```

**后果**：
- SSM 层的 conv_state 和 KV cache 概念不匹配
- 压缩破坏了 SSM 的状态更新机制
- Residual stream 漂移

---

### B. 沿用了 softmax-attention 的 β / mass matching ⭐⭐⭐⭐（理论不适用）

**证据**：
1. 我们的 Quality Path 实现了 β fitting（NNLS）
2. 但 SSM 层没有 attention mass 的概念
3. β 补偿失去意义

**验证方法**：
```python
# 检查：β 在非 softmax 层是否有意义？
if layer_type == "SSM":
    # SSM 没有 exp(scores)，β 该加在哪里？
    print("Warning: β fitting not applicable to SSM")
```

---

### C. Query distribution drift 更严重 ⭐⭐⭐（累积误差）

**论文已提到**（第 3.1 节）：

> Compacting layer by layer causes query distribution shift.
> We use on-policy queries to mitigate this.

**混合架构更严重**：
- SSM 层对 residual stream 更敏感
- 早期压缩改变后续层的状态演化
- 不只是 residual stream 漂，memory state 也在漂

---

### D. Chunked compaction 对 hybrid 更脆 ⭐⭐（实现细节）

**论文强调**（第 3.5 节）：

> KV-based chunking is more stable than text-based chunking,
> because text-based chunking ignores cross-chunk interactions.

**如果我们用了近似 chunk 方案**：
- 混合架构的 cross-chunk interaction 更复杂
- SSM 的状态跨 chunk 传播
- 更容易出错

---

## 🔧 工程改进路线（基于监护人建议）

### 路线 1: 只对标准全注意力层做 compaction ⭐⭐⭐⭐⭐（最务实）

**这是论文在 Gemma-3 上的做法，也是最稳妥的方案**

#### 实现步骤

**Step 1: 层分类器**

```python
def classify_layer(layer):
    """
    分类 Qwen3.5 的层类型

    Returns:
        "full_attention": 标准全注意力层 → 应该压缩
        "ssm": SSM/Mamba 层 → 不压缩
        "sliding": 滑动窗口层 → 不压缩或轻量压缩
        "linear": Linear attention → 需要专门方法
    """
    if hasattr(layer, 'self_attn'):
        # 进一步检查是否是滑动窗口
        if hasattr(layer.self_attn, 'sliding_window'):
            return "sliding"
        else:
            return "full_attention"

    elif hasattr(layer, 'linear_attn'):
        # Qwen3.5 的 SSM 层（GatedDeltaNet）
        return "ssm"

    elif hasattr(layer, 'conv1d'):
        return "ssm"

    else:
        return "unknown"
```

**Step 2: 选择性压缩**

```python
def create_selective_cache(model, max_size, compression_ratio):
    """
    为每层创建合适的 cache：
    - Full attention: CompactedKVCache
    - SSM: 标准 ArraysCache（不压缩）
    - Sliding: 保留窗口，不压缩
    """
    caches = []

    for i, layer in enumerate(model.layers):
        layer_type = classify_layer(layer)

        if layer_type == "full_attention":
            # 只有这里用 CompactedKVCache
            cache = CompactedKVCache(
                max_size=max_size,
                compression_ratio=compression_ratio
            )
            print(f"Layer {i}: Full Attention → CompactedKVCache")

        elif layer_type == "ssm":
            # SSM 层：标准 cache，不压缩
            cache = ArraysCache()
            print(f"Layer {i}: SSM → ArraysCache (no compression)")

        elif layer_type == "sliding":
            # 滑动窗口：保留窗口，不压缩
            cache = ArraysCache()  # 或 SlidingWindowCache
            print(f"Layer {i}: Sliding Window → ArraysCache (no compression)")

        else:
            # 未知类型：保守策略，不压缩
            cache = ArraysCache()
            print(f"Layer {i}: Unknown → ArraysCache (no compression)")

        caches.append(cache)

    return caches
```

**Step 3: 使用**

```python
# 加载模型
model, tokenizer = load("qwen3.5-35b-mlx")

# 创建选择性 cache
cache = create_selective_cache(
    model,
    max_size=4096,
    compression_ratio=5.0
)

# 生成
response = generate(
    model,
    tokenizer,
    prompt="介绍机器学习",
    cache=cache  # 混合 cache
)
```

**预期效果**：
- ✅ Attention 层（10 层）：压缩，节省内存
- ✅ SSM 层（30 层）：不压缩，保持正确性
- ✅ 输出质量：不崩溃
- ⚠️ 内存节省：只节省 Attention 部分（约 25% 的总 cache）

---

### 路线 2: 异构记忆压缩 (Heterogeneous Memory Compaction) ⭐⭐⭐（创新方向）

**核心思想**：不再把所有"上下文记忆"都叫 KV cache

#### 抽象设计

```python
class HeterogeneousCache:
    """
    混合架构的异构记忆系统

    管理两类记忆：
    - Attention Memory: 适用 AM compaction
    - State Memory: 适用 state summarization
    """

    def __init__(self):
        self.attention_memory = {}  # {layer_id: CompactedKVCache}
        self.state_memory = {}      # {layer_id: StateCache}

    def get_cache_for_layer(self, layer_id, layer_type):
        if layer_type == "attention":
            return self.attention_memory[layer_id]
        elif layer_type == "ssm":
            return self.state_memory[layer_id]

    def compress_attention_memory(self, budget):
        """压缩 Attention memory（用 AM）"""
        for layer_id, cache in self.attention_memory.items():
            cache.compress(budget)

    def compress_state_memory(self, budget):
        """压缩 State memory（用专门方法）"""
        for layer_id, cache in self.state_memory.items():
            cache.compress_state(budget)
```

#### State Memory 的压缩方法

```python
class StateCache:
    """
    SSM/Recurrent 层的状态缓存

    压缩方法（不同于 AM）：
    - State projection（低秩投影）
    - Learned state merge（学习的状态合并）
    - Recurrent state summary（递推状态摘要）
    """

    def compress_state(self, budget):
        """
        压缩 SSM 状态

        方法 1: 低秩投影
          conv_state: (B, kernel_size, dim) → (B, kernel_size, dim//4)

        方法 2: 状态池化
          conv_state: 每 4 个时间步合并为 1 个

        方法 3: 截断
          conv_state: 只保留最近 N 个时间步
        """
        # TODO: 实现
        pass
```

**优势**：
- ✅ 针对性设计
- ✅ 理论清晰
- ✅ 可扩展

**挑战**：
- 需要深入理解 SSM 的状态语义
- 需要设计新的压缩算法
- 需要大量实验验证

---

### 路线 3: 改目标函数（新的研究工作）⭐⭐⭐⭐（学术价值高）

**核心思想**：不再强行做 mass matching，改成更泛化的目标

#### 新的目标函数

**原始目标（AM）**：
```
min ||Compressed Attention Mass - Original Attention Mass||²
```

**新目标（泛化）**：

**选项 1: Match Layer Output**
```
min ||Compressed Layer Output - Original Layer Output||²

优势：
- 适用于所有层类型
- 直接优化最终目标

挑战：
- 计算开销大
- 需要前向传播
```

**选项 2: Match Residual Contribution**
```
min ||Compressed Residual - Original Residual||²

优势：
- 关注对 residual stream 的影响
- 防止 drift

挑战：
- 需要理解每层的贡献
```

**选项 3: Match Next-Token Logit Impact**
```
min ||Compressed Logits - Original Logits||²

优势：
- 直接优化生成质量
- 端到端

挑战：
- 需要解码器
- 计算开销更大
```

**这已经是新的研究方向了**，不是简单实现能解决的。

---

## 🎯 最实用的建议（Layerwise Ablation）

### 监护人的关键建议

> 你下一轮别再"整模型全压"了，直接这么干：
>
> 1. 把 Qwen3.5 每层分类
> 2. 只对 full attention 层启用 AM
> 3. local/sliding 层完全跳过
> 4. 非 softmax memory 层单独建压缩器
> 5. **先做 layerwise ablation**
>    - 单独压某几层
>    - **看哪类层一压就坏**

### Layerwise Ablation 实验设计

**目标**：找出哪些层可以压缩，哪些层不能压缩

#### 实验矩阵

| 实验 | 压缩策略 | 预期结果 | 验证假设 |
|------|---------|---------|---------|
| Baseline | 不压缩任何层 | 输出正常 | 基准 |
| Exp 1 | 只压缩 Layer 31-40 (Attention) | ✅ 应该正常 | Attention 层可压缩 |
| Exp 2 | 只压缩 Layer 1-30 (SSM) | ❌ 应该崩溃 | SSM 层不可压缩 |
| Exp 3 | 压缩所有层 | ❌ 崩溃（已验证） | 混合压缩失败 |
| Exp 4 | 只压缩 Layer 31 (单个 Attention) | ✅ 应该正常 | 单层压缩可行 |
| Exp 5 | 只压缩 Layer 1 (单个 SSM) | ❌ 应该崩溃 | 单个 SSM 也不行 |

#### 实验代码

```python
def layerwise_ablation_test(model, tokenizer, compress_layers):
    """
    Layerwise ablation 测试

    Args:
        compress_layers: List[int] - 要压缩的层索引

    Returns:
        输出质量指标
    """
    # 创建 cache
    cache = []
    for i, layer in enumerate(model.layers):
        if i in compress_layers:
            # 压缩这一层
            cache.append(CompactedKVCache(max_size=4096, compression_ratio=5.0))
            print(f"Layer {i}: Compressed")
        else:
            # 不压缩
            cache.append(ArraysCache())
            print(f"Layer {i}: Not compressed")

    # 生成
    prompt = "介绍机器学习"
    response = generate(model, tokenizer, prompt, cache=cache)

    # 评估
    return {
        'num_tokens': len(response),
        'has_repetition': check_repetition(response),
        'quality_score': evaluate_quality(response)
    }

# 运行实验
experiments = {
    "Baseline": [],  # 不压缩
    "Attention Only": list(range(31, 40)),  # 只压缩 Attention
    "SSM Only": list(range(0, 30)),  # 只压缩 SSM
    "All Layers": list(range(0, 40)),  # 压缩所有
    "Single Attention": [31],  # 单个 Attention 层
    "Single SSM": [0],  # 单个 SSM 层
}

results = {}
for exp_name, layers in experiments.items():
    print(f"\n{'='*50}")
    print(f"Experiment: {exp_name}")
    print(f"Compress layers: {layers}")
    print(f"{'='*50}")

    result = layerwise_ablation_test(model, tokenizer, layers)
    results[exp_name] = result

    print(f"Result: {result}")
```

**预期输出**：

```
==================================================
Experiment: Baseline
Compress layers: []
==================================================
Layer 0-39: Not compressed
Result: {'num_tokens': 497, 'has_repetition': False, 'quality_score': 9.5}

==================================================
Experiment: Attention Only
Compress layers: [31, 32, 33, 34, 35, 36, 37, 38, 39]
==================================================
Layer 0-30: Not compressed
Layer 31-39: Compressed
Result: {'num_tokens': 495, 'has_repetition': False, 'quality_score': 9.2}
  ✅ 成功！Attention 层可以压缩

==================================================
Experiment: SSM Only
Compress layers: [0, 1, ..., 29]
==================================================
Layer 0-29: Compressed
Layer 30-39: Not compressed
Result: {'num_tokens': 172, 'has_repetition': True, 'quality_score': 2.1}
  ❌ 失败！SSM 层不能用 AM 压缩

==================================================
Experiment: Single Attention
Compress layers: [31]
==================================================
Layer 31: Compressed
Result: {'num_tokens': 496, 'has_repetition': False, 'quality_score': 9.4}
  ✅ 单个 Attention 层也可以

==================================================
Experiment: Single SSM
Compress layers: [0]
==================================================
Layer 0: Compressed
Result: {'num_tokens': 180, 'has_repetition': True, 'quality_score': 2.5}
  ❌ 单个 SSM 层也会崩溃
```

**这会快速告诉我们**：
- 哪些层可以压缩
- 哪些层不能碰
- 问题的边界在哪里

---

## 🎯 核心判断（监护人观点）

### AM 的本质边界

**AM 不是"通用记忆压缩器"，它是"softmax-attention KV 压缩器"**

| 特性 | AM 是否支持 | 原因 |
|------|-----------|------|
| Softmax Attention | ✅ 支持 | 设计初衷 |
| Linear Attention | ❌ 不支持 | 无 mass 概念 |
| SSM/Mamba | ❌ 不支持 | 非 KV 结构 |
| Sliding Window | ⚠️ 部分支持 | 需要特殊处理 |
| Recurrent | ❌ 不支持 | 递推状态 |

**这个边界必须划清！**

---

## 📊 刷新后的研究优先级

### 🔥 立即执行（本周）：Layerwise Ablation ⭐⭐⭐⭐⭐

**目标**：验证假设，找到边界

**任务**：
1. ✅ 实现层分类器（classify_layer）
2. ✅ 实现选择性 cache（create_selective_cache）
3. ✅ 运行 layerwise ablation 实验
4. ✅ 分析结果，确认哪些层可以压缩

**预期时间**：2-3 天

**产出**：
- 实验报告
- 明确的层类型边界
- 下一步方向

---

### 📋 短期（1-2 周）：只压缩 Attention 层 ⭐⭐⭐⭐

**目标**：在 Qwen3.5 上实现稳定的部分压缩

**任务**：
1. 基于 ablation 结果，实现选择性压缩
2. 在 Qwen3.5 上验证
3. 性能和质量评估
4. 撰写技术报告

**预期时间**：1-2 周

**产出**：
- 工作原型
- 实验报告
- 可能贡献给 MLX-LM

---

### 🔬 中期（1-2 月）：异构记忆压缩 ⭐⭐⭐⭐

**目标**：为 SSM 层设计专门的压缩方法

**任务**：
1. 文献调研（SSM state compression）
2. 设计 State Memory 压缩算法
3. 实现并验证
4. 与 Attention Memory 压缩联合优化

**预期时间**：1-2 月

**产出**：
- 新的压缩算法
- 实验结果
- 可能的论文

---

### 📝 长期（3-6 月）：新的理论框架 ⭐⭐⭐⭐⭐

**目标**：建立混合架构记忆压缩的统一理论

**任务**：
1. 泛化目标函数（不限于 mass matching）
2. 为不同层类型设计专门方法
3. 系统性实验和分析
4. 撰写论文

**预期时间**：3-6 月

**产出**：
- 理论贡献
- 学术论文
- 开源实现

---

## 📝 总结

### 核心发现

1. **混合架构的记忆 ≠ KV cache**
   - 包括 conv_state, ssm_state 等多种状态
   - 不能只压缩 KV，要考虑异构记忆

2. **Attention mass 不适用于所有层**
   - β fitting 只对 softmax attention 有效
   - Linear attention / SSM 需要重新设计

3. **Future concatenation invariance 有边界**
   - 只对全局 attention 成立
   - 局部窗口 / 递推状态不支持

4. **论文作者自己也只压缩标准 attention 层**
   - Gemma-3: 只压缩 global attention
   - 我们不应该更激进

5. **AM 的本质：softmax-attention KV 压缩器**
   - 不是通用记忆压缩器
   - 必须划清边界

### 下一步行动

**本周（立即开始）**：
1. 实现层分类器
2. 运行 layerwise ablation
3. 确认边界

**1-2 周**：
1. 只压缩 Attention 层
2. 验证稳定性
3. 撰写报告

**1-2 月（如果有时间）**：
1. 设计 SSM state 压缩
2. 异构记忆系统
3. 可能的论文

---

*分析完成于: 2026-03-21*
*基于: 监护人深度分析*
*核心发现: AM 是 softmax-attention KV 压缩器，不是通用方法*
