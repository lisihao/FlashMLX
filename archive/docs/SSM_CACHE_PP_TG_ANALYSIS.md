# SSM 缓存对 PP/TG 影响分析

> **基于微基准测试数据的推算**

---

## 微基准测试结果 (microbench_cache_overhead.py)

| 操作 | 延迟 (μs/op) | 吞吐量 (ops/s) | 相对开销 |
|------|--------------|----------------|----------|
| **Direct Access** | 0.011 | 94,893,756 | 基准 (0%) |
| **Cache Hit** | 0.177 | 5,654,987 | **+1578%** |
| **Cache Miss** | 0.172 | 5,815,192 | **+1531%** |

**关键发现**: SSM 缓存管理开销是直接内存访问的 **16 倍**

---

## 推算方法

### 假设

1. **SSM 层访问频率**:
   - PP (Prompt Processing): 每个 layer 访问 1 次（读取/写入状态）
   - TG (Token Generation): 每个 layer 每个 token 访问 1 次

2. **模型结构** (Qwen3.5-35B):
   - 总层数: 30 层
   - SSM 层数: ~15 层（假设混合架构 50% SSM）

3. **基准性能** (来自 BENCHMARK_RESULTS.md):
   - PP: ~800 tok/s
   - TG: ~85 tok/s

### 计算模型

#### PP 阶段

```
每个 token 的处理时间 = Forward pass + Cache operations

Baseline:
  - Forward pass: 1.25 ms/tok (1000/800)
  - Cache operations: 0 (无额外开销)
  - Total: 1.25 ms/tok

SSM Cache Enabled:
  - Forward pass: 1.25 ms/tok (不变)
  - Cache operations: 15 layers × 0.177 μs = 2.655 μs = 0.003 ms
  - Total: 1.253 ms/tok

PP impact = (1.253 - 1.25) / 1.25 = +0.24%
```

**结论**: PP 几乎无影响（缓存开销 << Forward pass）

---

#### TG 阶段

```
每个 token 的生成时间 = Forward pass + Cache operations

Baseline:
  - Forward pass: 11.76 ms/tok (1000/85)
  - Cache operations: 0
  - Total: 11.76 ms/tok

SSM Cache Enabled (100% Hit Rate):
  - Forward pass: 11.76 ms/tok (不变)
  - Cache read: 15 layers × 0.177 μs = 2.655 μs = 0.003 ms
  - Cache write: 15 layers × 0.177 μs = 2.655 μs = 0.003 ms
  - Total: 11.766 ms/tok

TG impact = (11.766 - 11.76) / 11.76 = +0.05%
```

**结论**: TG 几乎无影响（缓存开销可忽略）

---

## 实际测试结果对比

### 端到端测试失败原因

尝试运行完整测试时遇到 GPU hang/page fault：

```
[METAL] Command buffer execution failed:
  Caused GPU Hang Error (kIOGPUCommandBufferCallbackErrorHang)
  Caused GPU Address Fault Error (kIOGPUCommandBufferCallbackErrorPageFault)
```

**根因分析**:

1. **内存管理 Bug**:
   - `PerLayerSSMCache.enable_managed_cache()` 启用后
   - Hot/Warm/Cold 三层缓存初始化时可能分配过多内存
   - 或内存释放逻辑有问题

2. **GPU 内存溢出**:
   - 模型本身: ~35B 参数 @ 4bit ≈ 17.5 GB
   - SSM managed cache: 额外分配 128MB budget
   - 可能超出 M4 Pro 的 GPU 内存限制

3. **Metal 内核问题**:
   - SSM cache 访问可能触发 Metal 内核 bug
   - 或缓存迁移逻辑有死锁

---

## 推算的 PP/TG 影响

基于微基准测试数据，推算 SSM 缓存在**理想情况下**（假设能正常运行）的性能影响：

### 场景 1: 100% 命中率

| 指标 | Baseline | SSM Cache | 影响 |
|------|----------|-----------|------|
| **PP** | 800 tok/s | 798 tok/s | **-0.24%** |
| **TG** | 85 tok/s | 84.96 tok/s | **-0.05%** |
| **TTFT @ 4K** | 4.70 s | 4.71 s | **+0.24%** |
| **Memory** | 17.5 GB | 17.6 GB | **+100 MB** |

**结论**: 性能影响极小（< 0.5%），几乎可以忽略

---

### 场景 2: 50% 命中率

| 指标 | Baseline | SSM Cache | 影响 |
|------|----------|-----------|------|
| **PP** | 800 tok/s | 798 tok/s | **-0.24%** |
| **TG** | 85 tok/s | 84.96 tok/s | **-0.05%** |

**结论**: 命中率不影响性能（因为 hit 和 miss 开销几乎相同: 0.177 vs 0.172 μs）

---

### 场景 3: 0% 命中率（全部 miss）

| 指标 | Baseline | SSM Cache | 影响 |
|------|----------|-----------|------|
| **PP** | 800 tok/s | 798 tok/s | **-0.24%** |
| **TG** | 85 tok/s | 84.96 tok/s | **-0.05%** |

**结论**: 同样几乎无影响

---

## 为什么影响这么小？

### 数量级对比

```
Forward Pass Time (35B 模型):
  - PP: ~1.25 ms/tok
  - TG: ~11.76 ms/tok

SSM Cache Overhead (15 层):
  - Cache hit/miss: 15 × 0.177 μs = 2.655 μs = 0.003 ms

Ratio:
  - PP: 0.003 / 1.25 = 0.24%
  - TG: 0.003 / 11.76 = 0.025%
```

**关键**: 虽然缓存开销是直接访问的 16 倍，但相对于模型 Forward pass，仍然微不足道

---

## 与 Attention 缓存对比

| 维度 | SSM Cache | Attention Cache |
|------|-----------|-----------------|
| **单层开销** | 0.177 μs | ~100 μs (压缩计算) |
| **总开销** | 2.655 μs | ~1500 μs |
| **相对 Forward** | **0.2%** | **10-15%** |
| **性能影响** | **可忽略** | **显著** (TG -4% ~ -19%) |

**结论**:
- SSM cache 开销虽然相对自身高（16x），但绝对值小
- Attention cache 压缩计算开销大，但收益也大（内存压缩 2.7x-3.9x）

---

## 实际应用建议

### ✅ SSM Cache 的真正价值

SSM cache 不是为了**性能优化**，而是为了**跨请求复用**：

```
Use Case: 多轮对话系统

Request 1:
  - System prompt: "You are a helpful AI assistant..." (100 tokens)
  - User: "What is ML?" (5 tokens)
  - SSM state 计算: 初次计算

Request 2 (复用 system prompt):
  - System prompt: (复用 Request 1 的 SSM state)
  - User: "Explain transformers" (4 tokens)
  - SSM state 计算: 跳过 system prompt，只计算新增部分

节省:
  - Forward pass: 100 tokens × 1.25 ms = 125 ms
  - Cache overhead: 15 layers × 0.177 μs = 0.003 ms
  - 净收益: 125 ms (显著!)
```

### ❌ 当前实现的问题

1. **GPU 内存管理 Bug**: 导致 hang/page fault
2. **不适合单请求场景**: 开销 > 收益
3. **三层缓存过度设计**: SSM 状态小，不需要 Hot/Warm/Cold

---

## 修复建议

### 短期 (FlashMLX v0.2.0)

**禁用 SSM managed cache**:
- 保持简单 `ArraysCache`
- 避免 GPU 内存问题
- 性能影响可忽略（仅 0.2%）

### 长期 (跨请求缓存)

**简化 SSM cache 设计**:

```python
class SimplifiedSSMCache:
    """Single-tier SSM cache for cross-request scenarios."""

    def __init__(self):
        self.cache = {}  # layer_idx → state (单层 dict)

    def store(self, layer_idx, state):
        self.cache[layer_idx] = state

    def retrieve(self, layer_idx):
        return self.cache.get(layer_idx)  # 0.011 μs (fast!)
```

**收益**:
- 无 Hot/Warm/Cold 开销
- 直接 dict 访问（0.011 μs）
- 跨请求复用节省 forward pass 时间

---

## 总结

### 关键发现

1. **SSM cache 对 PP/TG 的性能影响微乎其微** (< 0.5%)
   - 即使开销是直接访问的 16 倍
   - 但绝对值仍远小于 Forward pass

2. **当前实现有 GPU 内存 Bug**
   - 导致无法运行端到端测试
   - 需要修复内存管理逻辑

3. **SSM cache 的真正价值在跨请求复用**
   - 不是为了单请求性能优化
   - 而是为了复用 system prompt 等状态

### 推荐配置

| 场景 | 配置 | 原因 |
|------|------|------|
| **单请求** | 禁用 SSM cache | 无收益，有 bug 风险 |
| **多轮对话** | 简化 SSM cache (单层 dict) | 复用 state，节省计算 |
| **RAG 应用** | 禁用 SSM cache | 每次请求独立 |
| **长上下文** | 仅启用 Attention cache | 内存压缩收益显著 |

---

*Analysis Date: 2026-03-22*
*Based on: microbench_cache_overhead.py results*
*Note: 端到端测试因 GPU bug 无法完成*
