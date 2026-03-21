# 混合架构适配修复记录

**日期**: 2026-03-21
**任务**: Task #51 - 修复混合架构兼容性问题
**状态**: ✅ 修复完成，测试中

---

## 问题发现

### 问题 1: CompactedKVCache 不支持 SSM 接口

**错误**：
```
TypeError: 'CompactedKVCache' object is not subscriptable
```

**原因**：
- SSM 层期望 `cache[0]`, `cache[1]` (subscript 访问)
- CompactedKVCache 没有实现 `__getitem__` 和 `__setitem__`

**解决**：✅ 已修复
- 添加 `_ssm_states = [None, None]` 存储 SSM 状态
- 实现 `__getitem__` 和 `__setitem__` 方法
- 实现 `advance(n)` 方法

---

### 问题 2: SSM Mask 类型错误

**错误**：
```python
File "/Users/lisihao/FlashMLX/mlx-lm-source/mlx_lm/models/qwen3_5.py", line 157, in __call__
    qkv = mx.where(mask[..., None], qkv, 0)
                   ~~~~^^^^^^^^^^^
TypeError: string indices must be integers, not 'tuple'
```

**原因**：
- `create_ssm_mask` 调用 `cache.make_mask(N)` 返回字符串 "causal"
- SSM 层期望数组，可以用 `mask[..., None]` 索引
- `CompactedKVCache.make_mask` 默认返回 "causal"（Line 345）

**解决**：✅ 已修复

**文件**: `mlx-lm-source/mlx_lm/models/base.py` (Line 58-61)

**修改前**：
```python
def create_ssm_mask(h, cache=None):
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(h.shape[1])  # 返回 "causal" 字符串
    return None
```

**修改后**：
```python
def create_ssm_mask(h, cache=None):
    if cache and hasattr(cache, "make_mask"):
        # SSM layers need actual mask array, not "causal" string
        return cache.make_mask(h.shape[1], return_array=True)  # 返回实际数组
    return None
```

---

### 问题 3: ArraysCache 不支持 return_array 参数

**错误**：
```
TypeError: ArraysCache.make_mask() got an unexpected keyword argument 'return_array'
```

**原因**：
- `ArraysCache.make_mask` 没有 `return_array` 参数
- Baseline 测试（无压缩）使用 `ArraysCache`，不是 `CompactedKVCache`
- 修复问题 2 后，`create_ssm_mask` 传入 `return_array=True` 导致报错

**解决**：✅ 已修复

**文件**: `mlx-lm-source/mlx_lm/models/cache.py` (Line 767)

**修改前**：
```python
def make_mask(self, N: int):
    if self.left_padding is not None:
        pos = mx.arange(N)
        return pos >= self.left_padding[:, None]
    elif self.lengths is not None:
        pos = mx.arange(N)
        return pos < self.lengths[:, None]
    else:
        return None
```

**修改后**：
```python
def make_mask(self, N: int, return_array: bool = False, **kwargs):
    if self.left_padding is not None:
        pos = mx.arange(N)
        return pos >= self.left_padding[:, None]
    elif self.lengths is not None:
        pos = mx.arange(N)
        return pos < self.lengths[:, None]
    else:
        return None
```

**说明**：
- 添加 `return_array` 参数（默认 False）保持向后兼容
- 添加 `**kwargs` 接受其他参数
- ArraysCache 总是返回实际数组或 None，不返回 "causal" 字符串
- 所以 `return_array` 参数对 ArraysCache 无影响

---

## 修复总结

### 修改文件

| 文件 | 修改内容 | 行数 |
|------|----------|------|
| `mlx-lm-source/mlx_lm/models/compacted_cache.py` | 添加 SSM 支持 | +98 lines |
| `mlx-lm-source/mlx_lm/models/base.py` | 修复 SSM mask 创建 | 1 line |
| `mlx-lm-source/mlx_lm/models/cache.py` | 兼容 return_array 参数 | 1 line |

### 修改详情

**1. compacted_cache.py**

```python
# 添加 SSM 状态存储
self._ssm_states = [None, None]  # [conv_state, ssm_state]
self.lengths = None

# 添加 subscript 支持
def __getitem__(self, index): ...
def __setitem__(self, index, value): ...

# 添加 advance 方法
def advance(self, n: int): ...
```

**2. base.py**

```python
# Line 61: 传入 return_array=True
return cache.make_mask(h.shape[1], return_array=True)
```

**3. cache.py**

```python
# Line 767: 添加 return_array 参数
def make_mask(self, N: int, return_array: bool = False, **kwargs):
```

---

## 根本原因分析

### 为什么 SSM 需要实际 mask 数组？

**SSM 层代码** (qwen3_5.py Line 156-157):
```python
if mask is not None:
    qkv = mx.where(mask[..., None], qkv, 0)  # 需要索引 mask
```

**解释**：
- `mx.where` 需要布尔数组作为条件
- `mask[..., None]` 将 mask 从 (B, S) 扩展到 (B, S, 1)
- 字符串 "causal" 无法扩展维度

### 为什么 Attention 层可以接受 "causal" 字符串？

**Attention 层** 使用 `scaled_dot_product_attention`：
```python
output = scaled_dot_product_attention(
    queries, keys, values, cache=cache, scale=self.scale, mask=mask
)
```

**`scaled_dot_product_attention` 内部**：
```python
if mask == "causal":
    mask = create_causal_mask(...)  # 内部创建实际数组
```

**解释**：
- Attention 函数内部检查 `mask == "causal"`
- 如果是字符串，内部创建实际数组
- SSM 层没有这个逻辑，直接使用 mask

---

## 设计反思

### 为什么最初没有考虑 SSM？

1. **CompactedKVCache 设计初衷**：
   - 针对标准 Transformer（纯 Attention）
   - 压缩 KV cache（keys + values）
   - 论文验证模型：Qwen3-4B（纯 Transformer）

2. **混合架构是新趋势**：
   - Qwen3.5（2025）首次引入 SSM + Attention 混合
   - CompactedKVCache 设计时（2024）混合架构尚未流行

3. **SSM 状态与 KV cache 本质不同**：
   - KV cache：tokens × heads × dim（大，需压缩）
   - SSM 状态：conv_state + ssm_state（小，无需压缩）

### 设计教训

**教训 1**：Cache 接口应该更通用
- ✅ 支持多种访问模式（属性 + subscript）
- ✅ 支持多种状态类型（KV + SSM）

**教训 2**：Mask 类型应该一致
- ❌ 有的函数接受 "causal" 字符串
- ❌ 有的函数需要实际数组
- ✅ 应该统一为实际数组（性能开销可忽略）

**教训 3**：向后兼容很重要
- ✅ 添加 `return_array` 参数（默认 False）
- ✅ 添加 `**kwargs` 接受未知参数
- ✅ 保持现有 API 不变

---

## 测试验证

### 测试脚本

**文件**: `benchmarks/qwen3_5_hybrid_test.py`

**测试配置**：
1. Baseline（无压缩，ArraysCache）
2. CompactedKVCache Fast Path（5x 压缩）
3. CompactedKVCache Quality Path（5x 压缩）

**预期结果**：
- ✅ Baseline 正常运行（验证 ArraysCache 兼容性）
- ✅ CompactedKVCache 正常运行（验证 SSM 支持）
- ✅ 输出质量正常（无 "the the the" 重复）
- ✅ Token 数量一致（±5%）
- ✅ 性能 ≥ baseline

**当前状态**: 🔄 测试运行中

---

## 风险评估

### 已缓解风险

| 风险 | 状态 | 缓解措施 |
|------|------|----------|
| CompactedKVCache 不支持 subscript | ✅ 已修复 | 实现 `__getitem__`, `__setitem__` |
| SSM mask 类型错误 | ✅ 已修复 | `create_ssm_mask` 传入 `return_array=True` |
| ArraysCache 不兼容 | ✅ 已修复 | 添加 `return_array` 参数 |

### 剩余风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| SSM 状态逻辑错误 | 低 | 高 | 测试验证 |
| offset 更新冲突 | 低 | 中 | 观察测试结果 |
| 性能回退 | 低 | 中 | Benchmark 对比 |

---

## 下一步

### Phase 1: 等待测试结果（进行中）

- 🔄 Qwen3.5 35B 测试运行中
- 预计时间：~5-10 分钟

### Phase 2: 回归测试（如果 Phase 1 通过）

1. Llama 3.2 3B（纯 Transformer）
2. Qwen3-8B（纯 Transformer）
3. 确认无破坏性修改

### Phase 3: 文档更新（如果测试通过）

1. 更新 `docs/COMPACTED_CACHE_USAGE.md`
2. 添加混合架构说明
3. 添加 Qwen3.5 示例

---

## 参考

- **实现报告**: `.solar/hybrid-architecture-implementation.md`
- **分析报告**: `.solar/hybrid-architecture-analysis.md`
- **修改文件**:
  - `mlx-lm-source/mlx_lm/models/compacted_cache.py`
  - `mlx-lm-source/mlx_lm/models/base.py`
  - `mlx-lm-source/mlx_lm/models/cache.py`
- **测试脚本**: `benchmarks/qwen3_5_hybrid_test.py`

---

*修复完成于: 2026-03-21*
*测试状态: 🔄 运行中*
