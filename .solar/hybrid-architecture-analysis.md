# 混合架构 Cache 接口分析

**日期**: 2026-03-21
**任务**: Task #51 - 设计混合架构适配方案
**模型**: Qwen3.5 35B (混合架构: SSM + Attention)

---

## 执行摘要

CompactedKVCache 与 Qwen3.5 混合架构不兼容的**根本原因**：

- **SSM 层**期望 **subscriptable cache** (`cache[0]`, `cache[1]`)
- **Attention 层**期望 **对象方法** (`cache.update_and_fetch()`, `cache.offset`)
- CompactedKVCache 只实现了 Attention 接口，不支持 subscript 访问

**解决方案**：设计兼容层，让两种 cache 接口共存。

---

## Cache 接口对比

### SSM 层 (GatedDeltaNet) 需求

**文件**: `mlx-lm-source/mlx_lm/models/qwen3_next.py` (Lines 236-305)

| 操作 | 代码位置 | 接口需求 |
|------|----------|----------|
| 读取 conv_state | Line 247 | `cache[0]` (subscript) |
| 更新 conv_state | Line 267-269 | `cache[0] = ...` (subscript assignment) |
| 读取 ssm_state | Line 282 | `cache[1]` (subscript) |
| 更新 ssm_state | Line 301 | `cache[1] = ...` (subscript assignment) |
| 推进序列位置 | Line 302 | `cache.advance(S)` (方法) |
| 检查长度 | Line 264 | `cache.lengths` (可选属性) |

**代码示例**：
```python
# Line 247-253: 读取 conv_state
if cache is not None and cache[0] is not None:
    conv_state = cache[0]
else:
    conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim))

# Line 262-269: 更新 conv_state
if cache is not None:
    if cache.lengths is not None:
        ends = mx.clip(cache.lengths, 0, S)
        positions = (ends[:, None] + mx.arange(n_keep))[..., None]
        cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
    else:
        cache[0] = conv_input[:, -n_keep:, :]

# Line 282: 读取 ssm_state
state = cache[1] if cache else None

# Line 300-302: 更新 ssm_state + advance
if cache is not None:
    cache[1] = state
    cache.advance(S)
```

### Attention 层 需求

**文件**: `mlx-lm-source/mlx_lm/models/qwen3_next.py` (Lines 121-158)

| 操作 | 代码位置 | 接口需求 |
|------|----------|----------|
| 获取序列偏移 | Line 146-147 | `cache.offset` (属性) |
| 更新并获取 KV | Line 148 | `cache.update_and_fetch(keys, values)` (方法) |

**代码示例**：
```python
# Line 145-151: Attention 层使用 cache
if cache is not None:
    queries = self.rope(queries, offset=cache.offset)
    keys = self.rope(keys, offset=cache.offset)
    keys, values = cache.update_and_fetch(keys, values)
else:
    queries = self.rope(queries)
    keys = self.rope(keys)
```

---

## CompactedKVCache 当前接口

**文件**: `mlx-lm-source/mlx_lm/models/compacted_cache.py`

| 接口 | 实现状态 | 说明 |
|------|----------|------|
| `cache.offset` | ✅ 已实现 | 属性 |
| `cache.update_and_fetch()` | ✅ 已实现 | 方法 |
| `cache[0]` | ❌ **未实现** | 没有 `__getitem__` |
| `cache[1]` | ❌ **未实现** | 没有 `__getitem__` |
| `cache.advance()` | ❌ **未实现** | 没有此方法 |
| `cache.lengths` | ❌ **未实现** | 没有此属性 |

**错误信息**：
```
TypeError: 'CompactedKVCache' object is not subscriptable
```

---

## 架构差异图

### Qwen3.5 混合架构

```
40 层混合:
┌─────────────────────────────┐
│ Layer 0: Linear Attention   │ ← SSM, 需要 cache[0], cache[1]
├─────────────────────────────┤
│ Layer 1: Linear Attention   │ ← SSM, 需要 cache[0], cache[1]
├─────────────────────────────┤
│ Layer 2: Linear Attention   │ ← SSM, 需要 cache[0], cache[1]
├─────────────────────────────┤
│ Layer 3: Full Attention     │ ← CompactedKVCache ✅
├─────────────────────────────┤
│ Layer 4: Linear Attention   │ ← SSM, 需要 cache[0], cache[1]
├─────────────────────────────┤
│         ...                 │
├─────────────────────────────┤
│ Layer 39: Full Attention    │ ← CompactedKVCache ✅
└─────────────────────────────┘

问题：SSM 层期望 cache[0], cache[1]，但 CompactedKVCache 不支持
```

### Qwen3 纯 Transformer（对照组）

```
36 层全部为 Full Attention:
┌─────────────────────────────┐
│ Layer 0: Full Attention     │ ← CompactedKVCache ✅
├─────────────────────────────┤
│ Layer 1: Full Attention     │ ← CompactedKVCache ✅
├─────────────────────────────┤
│ Layer 2: Full Attention     │ ← CompactedKVCache ✅
│            ...              │
├─────────────────────────────┤
│ Layer 35: Full Attention    │ ← CompactedKVCache ✅
└─────────────────────────────┘

✅ 所有层使用相同 cache 接口，完美兼容
```

---

## 三种适配方案

### 方案 A: 扩展 CompactedKVCache 支持 Subscript

**核心思想**：让 CompactedKVCache 同时支持两种接口

**实现**：
```python
class CompactedKVCache:
    def __init__(self, ...):
        # 现有属性
        self.keys = None
        self.values = None
        self.offset = 0

        # 新增：SSM 状态存储
        self._ssm_states = [None, None]  # [conv_state, ssm_state]
        self.lengths = None  # 可选：用于 SSM 的长度跟踪

    # ✅ 现有方法（Attention 层用）
    def update_and_fetch(self, keys, values):
        ...

    # ✅ 新增：subscript 支持（SSM 层用）
    def __getitem__(self, index):
        if index not in [0, 1]:
            raise IndexError(f"Cache only supports indices 0 and 1, got {index}")
        return self._ssm_states[index]

    def __setitem__(self, index, value):
        if index not in [0, 1]:
            raise IndexError(f"Cache only supports indices 0 and 1, got {index}")
        self._ssm_states[index] = value

    # ✅ 新增：advance 方法（SSM 层用）
    def advance(self, n):
        self.offset += n
```

**优点**：
- ✅ 最小化代码修改（只改 CompactedKVCache 一个类）
- ✅ 保持 Attention 层的压缩优化
- ✅ SSM 层使用标准状态存储（不压缩）
- ✅ 对外接口兼容性好

**缺点**：
- ⚠️ SSM 状态不压缩（但 SSM 状态本身很小）
- ⚠️ CompactedKVCache 变得更复杂

**适用性**：
- ✅ 混合架构（Qwen3.5）
- ✅ 纯 Transformer（Llama, Qwen3）

---

### 方案 B: 混合 Cache 包装器

**核心思想**：根据层类型路由到不同 cache 实现

**实现**：
```python
class HybridCache:
    """适配混合架构的 cache 包装器"""

    def __init__(self, layer_type, **kwargs):
        """
        Args:
            layer_type: "ssm" 或 "attention"
        """
        self.layer_type = layer_type

        if layer_type == "ssm":
            # SSM 层使用简单的 tuple cache
            self._cache = [None, None]  # [conv_state, ssm_state]
            self.lengths = None
            self.offset = 0

        elif layer_type == "attention":
            # Attention 层使用 CompactedKVCache
            self._cache = CompactedKVCache(**kwargs)

        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    # SSM 接口
    def __getitem__(self, index):
        if self.layer_type == "ssm":
            return self._cache[index]
        else:
            raise TypeError(f"CompactedKVCache does not support subscript")

    def __setitem__(self, index, value):
        if self.layer_type == "ssm":
            self._cache[index] = value
        else:
            raise TypeError(f"CompactedKVCache does not support subscript")

    # Attention 接口
    def update_and_fetch(self, keys, values):
        if self.layer_type == "attention":
            return self._cache.update_and_fetch(keys, values)
        else:
            raise AttributeError(f"SSM cache does not have update_and_fetch")

    @property
    def offset(self):
        if self.layer_type == "attention":
            return self._cache.offset
        else:
            return self._offset

    def advance(self, n):
        if self.layer_type == "ssm":
            self.offset += n
        elif self.layer_type == "attention":
            self._cache.offset += n

# 初始化时根据层类型创建
cache = [
    HybridCache("ssm") if model.layers[i].is_linear
    else HybridCache("attention", max_size=4096, compression_ratio=5.0)
    for i in range(len(model.layers))
]
```

**优点**：
- ✅ 清晰的职责分离（SSM 和 Attention 完全独立）
- ✅ CompactedKVCache 不需要修改
- ✅ 可以为 SSM 层单独优化（如果需要）

**缺点**：
- ❌ 需要在初始化时知道每层类型（需要检查 `layer.is_linear`）
- ❌ 新增一个包装类（增加代码复杂度）
- ⚠️ 用户需要手动构建 cache 列表

**适用性**：
- ✅ 混合架构（Qwen3.5）
- ⚠️ 纯 Transformer 不需要（多余）

---

### 方案 C: 修改 SSM 层代码

**核心思想**：改 SSM 层使用对象属性而非 subscript

**实现**：
```python
# 修改 qwen3_next.py 的 GatedDeltaNet 类

# 原代码 (Line 247-253)
if cache is not None and cache[0] is not None:
    conv_state = cache[0]
else:
    conv_state = mx.zeros(...)

# 改为
if cache is not None and hasattr(cache, 'conv_state') and cache.conv_state is not None:
    conv_state = cache.conv_state
else:
    conv_state = mx.zeros(...)

# 原代码 (Line 267-269)
if cache is not None:
    cache[0] = conv_input[:, -n_keep:, :]

# 改为
if cache is not None:
    cache.conv_state = conv_input[:, -n_keep:, :]

# 原代码 (Line 282)
state = cache[1] if cache else None

# 改为
state = cache.ssm_state if (cache and hasattr(cache, 'ssm_state')) else None

# 原代码 (Line 301)
if cache is not None:
    cache[1] = state

# 改为
if cache is not None:
    cache.ssm_state = state
```

**然后在 CompactedKVCache 中添加**：
```python
class CompactedKVCache:
    def __init__(self, ...):
        # 现有属性
        self.keys = None
        self.values = None
        self.offset = 0

        # 新增：SSM 状态
        self.conv_state = None
        self.ssm_state = None
        self.lengths = None
```

**优点**：
- ✅ CompactedKVCache 修改最小（只加属性）
- ✅ 接口统一（都用属性，不用 subscript）

**缺点**：
- ❌ **需要修改 MLX-LM 的模型代码**（qwen3_next.py, qwen3_5.py）
- ❌ 破坏了与原始 MLX-LM 的兼容性
- ❌ 每次 MLX-LM 更新都需要重新 patch
- ❌ 其他混合架构模型可能也需要修改

**适用性**：
- ⚠️ 不推荐（维护成本高）

---

## 推荐方案

### ✅ 推荐：方案 A（扩展 CompactedKVCache）

**原因**：

1. **最小侵入性**
   - 只修改 CompactedKVCache 一个文件
   - 不修改 MLX-LM 的模型代码
   - 保持与上游兼容

2. **功能完整性**
   - Attention 层继续使用压缩（性能优化）
   - SSM 层使用简单状态存储（状态很小，无需压缩）
   - 两种接口共存

3. **易用性**
   - 用户无需知道层类型
   - 初始化方式与纯 Transformer 一致：
     ```python
     cache = [
         CompactedKVCache(max_size=4096, compression_ratio=5.0)
         for _ in range(len(model.layers))
     ]
     ```
   - CompactedKVCache 自动处理两种接口

4. **性能**
   - Attention 层的 KV cache 被压缩（80% 内存节省）
   - SSM 层的状态不压缩（但状态很小，影响可忽略）
   - 整体性能依然优于 baseline

5. **可维护性**
   - 单一职责：CompactedKVCache 负责所有 cache 管理
   - 易于测试
   - 易于文档化

---

## 实现计划

### Phase 1: 扩展 CompactedKVCache

1. 添加 SSM 状态存储：
   ```python
   self._ssm_states = [None, None]
   self.lengths = None
   ```

2. 实现 subscript 支持：
   ```python
   def __getitem__(self, index): ...
   def __setitem__(self, index, value): ...
   ```

3. 实现 advance 方法：
   ```python
   def advance(self, n):
       self.offset += n
   ```

4. 更新文档：
   - 在 `docs/COMPACTED_CACHE_USAGE.md` 中说明混合架构支持
   - 添加 Qwen3.5 的示例

### Phase 2: 测试验证

1. 单元测试：
   - 测试 `__getitem__` 和 `__setitem__`
   - 测试 `advance()` 方法
   - 测试 Attention 和 SSM 接口共存

2. 集成测试：
   - 在 Qwen3.5 35B 上运行
   - 对比输出质量（与 baseline）
   - 测试性能（期望 > baseline）

3. 回归测试：
   - 在 Llama 3.2 3B 上验证（纯 Transformer）
   - 在 Qwen3-8B 上验证（纯 Transformer）
   - 确保没有破坏现有功能

### Phase 3: 性能优化（可选）

1. 如果 SSM 状态太大，考虑压缩
2. 如果 advance() 调用频繁，考虑批处理

---

## 预期结果

| 指标 | 目标 | 说明 |
|------|------|------|
| **输出质量** | ✅ 正常 | 无 "the the the" 重复 |
| **Token 数量** | ✅ 与 baseline 一致 | 无减少 |
| **性能** | ✅ > baseline | 期望 +10% 以上 |
| **内存** | ✅ < baseline | KV cache 压缩 80% |
| **兼容性** | ✅ 纯 Transformer 无影响 | Llama, Qwen3 继续工作 |

---

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| SSM 状态存储逻辑错误 | 中 | 高 | 单元测试 + 输出质量对比 |
| 性能回退 | 低 | 中 | Benchmark 对比 |
| 破坏纯 Transformer 兼容性 | 低 | 高 | 回归测试（Llama, Qwen3） |

---

## 参考

- **Qwen3.5 模型代码**: `mlx-lm-source/mlx_lm/models/qwen3_next.py`
- **CompactedKVCache 实现**: `mlx-lm-source/mlx_lm/models/compacted_cache.py`
- **Llama 成功报告**: `.solar/llama-test-success-report.md`
- **Qwen3.5 问题分析**: `.solar/output-quality-critical-issue.md`

---

*分析完成于: 2026-03-21*
*下一步: 实现方案 A*
