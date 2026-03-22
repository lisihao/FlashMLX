# Query Generation 执行时机详解

**关键问题**: Query Generation 是对每个用户 prompt 都执行吗？

**答案**: ❌ **不是**。只在 KV cache **需要压缩时**才执行，通常是累积到一定规模后。

---

## 🔄 完整的对话流程

### 场景：用户与 AI 的长对话

```
用户: "介绍一下机器学习" (Prompt 1, 10 tokens)
  ↓ 模型处理
AI: "机器学习是..." (Response 1, 200 tokens)
  ↓ KV cache 累积
KV cache: 210 tokens (10 + 200)

用户: "深度学习和机器学习的区别？" (Prompt 2, 15 tokens)
  ↓ 模型处理（使用之前的 KV cache）
AI: "深度学习是机器学习的子集..." (Response 2, 300 tokens)
  ↓ KV cache 累积
KV cache: 525 tokens (210 + 15 + 300)

用户: "给我举个例子" (Prompt 3, 8 tokens)
  ↓ 模型处理
AI: "比如图像识别..." (Response 3, 400 tokens)
  ↓ KV cache 累积
KV cache: 933 tokens (525 + 8 + 400)

... 对话继续 ...

KV cache: 4200 tokens ← 超过阈值（max_size=4096）
  ↓ 🔥 触发压缩！
  ↓ 执行 Query Generation + Compression
KV cache: 840 tokens (4200 / 5 = 压缩到 1/5)
```

---

## 🎯 Query Generation 的执行时机

### 时机 1：KV Cache 达到压缩阈值

**触发条件**：
```python
if cache.offset > cache.max_size:  # 例如 > 4096 tokens
    # 🔥 执行压缩！
    cache._compress()
```

**执行内容**：
```python
def _compress(self):
    """压缩累积的 KV cache"""

    # 当前 cache: 60k tokens 的历史对话
    current_keys = self.keys[..., :self.offset, :]    # (n_heads, 60k, head_dim)
    current_values = self.values[..., :self.offset, :]

    # Step 1: Query Generation（只在这里执行一次）
    # 从这 60k tokens 中生成 100 个代表性 queries
    queries = self_study(current_keys)  # 60k → 100
    queries = OMP(queries, current_keys, current_values)  # 优化

    # Step 2: 用这 100 个 queries 压缩 KV cache
    compressed_keys, beta, compressed_values = compress(
        queries, current_keys, current_values, budget=12000
    )

    # Step 3: 替换原来的 cache
    self.keys = compressed_keys    # 60k → 12k
    self.values = compressed_values
    self.offset = 12000
```

**频率**：
- 不是每个 prompt 都执行
- 只有当 cache 累积到 max_size（如 4096）时才执行
- 可能几十轮对话才触发一次

---

## 📊 具体示例：长对话场景

### 配置

```python
cache = CompactedKVCache(
    max_size=4096,           # 累积到 4096 tokens 时压缩
    compression_ratio=5.0    # 压缩到 1/5 (4096 → ~820)
)
```

### 对话过程

| 轮次 | 用户输入 | AI 输出 | KV Cache 累积 | 是否压缩 | Query Generation |
|------|---------|---------|--------------|---------|-----------------|
| 1 | 10 tokens | 200 tokens | 210 tokens | ❌ | ❌ 不执行 |
| 2 | 15 tokens | 300 tokens | 525 tokens | ❌ | ❌ 不执行 |
| 3 | 8 tokens | 400 tokens | 933 tokens | ❌ | ❌ 不执行 |
| 4 | 12 tokens | 500 tokens | 1445 tokens | ❌ | ❌ 不执行 |
| 5 | 20 tokens | 600 tokens | 2065 tokens | ❌ | ❌ 不执行 |
| 6 | 18 tokens | 700 tokens | 2783 tokens | ❌ | ❌ 不执行 |
| 7 | 25 tokens | 800 tokens | 3608 tokens | ❌ | ❌ 不执行 |
| 8 | 30 tokens | 650 tokens | **4288 tokens** | ✅ 触发！ | ✅ **执行一次** |
| → | - | - | 858 tokens (压缩后) | - | - |
| 9 | 15 tokens | 400 tokens | 1273 tokens | ❌ | ❌ 不执行 |
| 10 | 20 tokens | 500 tokens | 1793 tokens | ❌ | ❌ 不执行 |
| ... | ... | ... | ... | ... | ... |
| 20 | 22 tokens | 600 tokens | **4500 tokens** | ✅ 再次触发！ | ✅ **再次执行** |

**关键点**：
- Query Generation 只在第 8 轮和第 20 轮执行（KV cache 超过 4096 时）
- 其他轮次（1-7, 9-19）都不执行 Query Generation
- 每次压缩是针对**累积的所有历史 KV cache**

---

## 🔍 Query Generation 的输入

### 问题：Queries 从哪里来？

**答案：从累积的历史 KV cache（keys）中生成**

```python
# 假设累积了 60k tokens 的对话历史
# keys 的内容包括：
# - 所有历史用户输入（prompt 1, 2, 3, ...）
# - 所有历史 AI 输出（response 1, 2, 3, ...）

keys: [
    "介绍",      # prompt 1 token 1
    "一下",      # prompt 1 token 2
    "机器",      # prompt 1 token 3
    "学习",      # prompt 1 token 4
    "机器",      # response 1 token 1
    "学习",      # response 1 token 2
    "是",        # response 1 token 3
    ...
    (60k tokens total)
]

# Query Generation 的目标：
# 从这 60k 个 keys 中找到 100 个代表性的
# 这 100 个能反映整个对话的 attention 模式

representative_queries: [
    q1,  # 代表 "关于定义的查询"
    q2,  # 代表 "关于区别的查询"
    q3,  # 代表 "关于举例的查询"
    ...
    q100
]
```

---

## 🎯 为什么不是每个 Prompt 都执行？

### 原因 1：没必要

**小规模 cache 不需要压缩**：
- 用户输入：10 tokens
- AI 输出：200 tokens
- KV cache: 210 tokens（很小）
- 内存占用：< 1MB
- **不需要压缩，直接存储即可** ✅

### 原因 2：太频繁

**每个 prompt 都压缩会很慢**：
```
用户: "你好"
  ↓ 执行 Query Generation (700s) ❌
  ↓ 压缩 10 tokens
AI: (等了 12 分钟才回复)

→ 体验极差！
```

### 原因 3：累积效应

**压缩大 cache 效率更高**：
- 压缩 100 tokens → 20 tokens：省 80 tokens（不划算）
- 压缩 60k tokens → 12k tokens：省 48k tokens（很划算）

---

## 📊 成本收益分析

### 场景 1：每个 Prompt 都压缩（❌ 不好）

```
对话 10 轮：
- 每轮压缩一次
- 每次 Query Generation: 假设 10s（小规模）
- 总开销: 10 × 10s = 100s
- 节省内存: ~1K tokens（微不足道）

→ 性价比极低
```

### 场景 2：累积到阈值再压缩（✅ 好）

```
对话 10 轮：
- 累积到 4096 tokens 时压缩一次
- Query Generation: 1 次 × 20s = 20s
- 总开销: 20s
- 节省内存: ~3200 tokens（显著）

→ 性价比高
```

---

## 🔧 实际执行流程

### 用户视角

```
用户: "介绍一下机器学习"
  ↓ [200ms] 模型推理
AI: "机器学习是..."
  ↓ KV cache: 210 tokens

用户: "深度学习和机器学习的区别？"
  ↓ [220ms] 模型推理（使用 cache，更快）
AI: "深度学习是..."
  ↓ KV cache: 525 tokens

... (多轮对话) ...

用户: "给我总结一下"
  ↓ [250ms] 模型推理
  ↓ KV cache: 4200 tokens → 触发压缩
  ↓ [2s] 压缩（包括 Query Generation）
  ↓ KV cache: 840 tokens
AI: "综上所述..."

→ 用户感受：稍微卡了一下（2s），但总体流畅
```

### 开发者视角

```python
# 用户发送 prompt
prompt = "介绍一下机器学习"

# 模型处理
for token in generate(model, tokenizer, prompt, cache=cache):
    print(token)

    # 内部逻辑（自动）：
    # if cache.offset > cache.max_size:
    #     cache._compress()  ← 在这里执行 Query Generation
```

---

## 🎯 总结

### Query Generation 不是每个 Prompt 都执行

**执行条件**：
- ✅ KV cache 累积到阈值（如 4096 tokens）
- ✅ 对整个累积的历史 cache 执行一次
- ❌ 不是每个 prompt 都执行

**执行频率**：
- 小规模对话（< 10 轮）：可能不触发
- 中等对话（10-50 轮）：触发 1-2 次
- 长对话（> 100 轮）：触发多次

**执行对象**：
- 输入：累积的所有历史 KV cache（60k tokens）
- 输出：100 个代表性 queries
- 用途：用这 100 个 queries 压缩 60k cache → 12k

### 用户体验

**无 Query Generation（我们当前的实现）**：
- 小规模：流畅 ✅
- 大规模：卡死 ❌

**有 Query Generation（论文实现）**：
- 小规模：流畅 ✅
- 大规模：偶尔卡 2-3 秒（压缩时），但不会卡死 ✅

### 类比

就像**垃圾回收**：
- 不是每次分配内存都回收垃圾
- 累积到一定程度，触发垃圾回收
- 回收时会"停顿"一下，但整体更流畅

**KV cache 压缩 = 内存垃圾回收**
**Query Generation = 选择哪些"垃圾"可以回收**

---

*文档创建于: 2026-03-21*
*关键发现: Query Generation 只在压缩时执行，不是每个 prompt 都执行*
