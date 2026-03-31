# SSM Cache 设计说明

**日期**: 2026-03-21
**状态**: 重新设计 - 简化为单请求缓存

---

## 问题背景

**原设计**：
```
PerLayerSSMCache
  → 本地缓存 (ArraysCache)
  → Managed Cache (Hot/Warm/Cold)
```

**问题**：
1. 本地缓存永远有值 → Managed Cache 永远不会被查询
2. Hot/Warm/Cold 命中率 0%（机制完全失效）
3. 设计复杂，但没有实际效果

---

## 根本原因：缓存需求不同

| 维度 | Attention 层 | SSM 层 |
|------|-------------|--------|
| **缓存对象** | KV cache (seq_len × hidden) | SSM state (fixed size) |
| **内存压力** | 极大（随 seq_len 线性增长） | 较小（固定大小） |
| **单请求内** | 需要压缩（节省内存） | 不需要压缩（内存充足） |
| **跨请求** | 不需要（每次都不同） | 可能需要（复用 prefix state） |
| **优化目标** | Attention Matching 压缩 | 跨请求状态共享 |

**关键洞察**：
- **Attention 层**：单请求内就有巨大内存压力 → 需要压缩
- **SSM 层**：单请求内内存充足 → **不需要 Hot/Warm/Cold**

---

## 新设计

### PerLayerSSMCache (简化)

```python
class PerLayerSSMCache(ArraysCache):
    """
    Per-layer cache for SSM layers.

    Design:
    - Single-request: Use simple ArraysCache (no compression needed)
    - Cross-request: Future work for request-level caching
    """

    def __init__(self, manager, layer_idx, size=2):
        super().__init__(size)
        self.manager = manager
        self.layer_idx = layer_idx

    # Use default ArraysCache behavior (no __setitem__/__getitem__ override)
```

**优点**：
- ✅ 简单，符合 MLX-LM 规范
- ✅ 单请求内性能最优（直接内存访问）
- ✅ 没有复杂的 Hot/Warm/Cold 逻辑

**缺点**：
- ❌ 不支持跨请求状态复用

---

## Hot/Warm/Cold 何时使用？

### 场景 1: 跨请求状态复用（未来）

**示例**：多用户共享相同 system prompt
```
Request 1: system_prompt + user_query_1
Request 2: system_prompt + user_query_2  # 复用 system_prompt 的 SSM state
Request 3: system_prompt + user_query_3
```

**实现**：
- Request-level cache manager
- 基于 prompt prefix 的状态匹配
- Hot/Warm/Cold 基于访问频率

**当前状态**: 未实现（benchmark 是单请求测试，命中率 0% 是正常的）

### 场景 2: 长上下文内状态压缩（不需要）

**原因**：
- SSM state 是固定大小（不随 seq_len 增长）
- 内存压力远小于 Attention KV cache
- 不需要压缩

---

## Benchmark 结果解读

### 为什么 SSM hit rate = 0%？

**答案**: 正常！因为：
1. 每次都是新的 prompt（没有跨请求复用）
2. SSM state 是 per-request 的
3. 当前设计不支持跨请求缓存

### 为什么 Attention compression > 0？

**答案**: 单请求内压缩！因为：
1. KV cache 随 seq_len 增长
2. Attention Matching 在单请求内就压缩
3. 节省内存，允许更长上下文

---

## 对比总结

| 维度 | SSM Cache | Attention Cache |
|------|-----------|----------------|
| **当前实现** | 简单 ArraysCache | Attention Matching 压缩 |
| **单请求优化** | 无需优化（内存充足） | ✅ 压缩 KV cache |
| **跨请求优化** | ❌ 未实现 | 无需优化 |
| **Hit rate** | 0%（正常） | N/A（压缩率 > 0） |
| **内存节省** | 无 | ✅ 2.7x-3.9x |

---

## 未来工作

### Phase 1: Request-level SSM Cache (跨请求)

**目标**: 复用相同 prefix 的 SSM state

**设计**：
```python
class RequestLevelSSMCache:
    """
    Cross-request SSM state cache.

    - Hash prompt prefix → SSM state
    - Hot/Warm/Cold based on access frequency
    - LRU eviction when budget exceeded
    """

    def get_prefix_state(self, prompt_prefix: str, layer_idx: int):
        # Try Hot → Warm → Cold
        # Update access count
        # Promote if needed
```

**Benchmark**: 多请求测试
```python
# Test 1: Same system prompt
for i in range(100):
    generate(f"{system_prompt}\nUser query {i}")

# Expected: SSM hit rate > 80% (reuse system_prompt state)
```

### Phase 2: Adaptive Compression (可选)

**触发条件**: 极长上下文 + 内存不足

**方法**: SSM state 量化 (FP16 → INT8)

---

## 总结

**当前设计（方案 C）**：
- ✅ SSM Cache: 简化为单请求 ArraysCache
- ✅ Attention Cache: 单请求内 Attention Matching 压缩
- ✅ 清晰的职责划分

**Benchmark 结果**：
- SSM hit rate 0%：**正常**（单请求测试，没有跨请求复用）
- Attention compression 2.7x-3.9x：**有效**（单请求内压缩）

**下一步**：
- [ ] 实现 Request-level SSM Cache（跨请求状态复用）
- [ ] 多请求 benchmark 测试 SSM hit rate
