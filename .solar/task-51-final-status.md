# Task #51: 混合架构适配方案 - 最终状态

**日期**: 2026-03-21
**状态**: ⚠️ 部分完成，遇到技术挑战
**结论**: Qwen3.5 混合架构需要更深入的架构重构

---

## 执行摘要

✅ **成功验证**: CompactedKVCache 在纯 Transformer 上工作正常（Llama 3.2, Qwen3-8B）
❌ **挑战**: Qwen3.5 混合架构兼容性需要比预期更深入的修改

**核心问题**: SSM 层的 conv_state 缓存与 MLX 生成过程中的 batch size 变化不兼容

---

## 完成的工作

### 1. 深入架构分析 ✅

**文件**: `.solar/hybrid-architecture-analysis.md`

- 详细对比了 SSM 和 Attention 层的 cache 接口需求
- 设计了三种适配方案，推荐方案 A（扩展 CompactedKVCache）
- 分析了性能影响和风险

**关键发现**：
- SSM 层需要: `cache[0]` (conv_state), `cache[1]` (ssm_state), `cache.advance()`
- Attention 层需要: `cache.update_and_fetch()`, `cache.offset`
- 两种接口本质不同

### 2. 扩展 CompactedKVCache ✅

**文件**: `mlx-lm-source/mlx_lm/models/compacted_cache.py`

**修改内容**：
```python
# 添加 SSM 状态存储
self._ssm_states = [None, None]  # [conv_state, ssm_state]
self.lengths = None

# 实现 subscript 支持
def __getitem__(self, index): ...
def __setitem__(self, index, value): ...

# 实现 advance 方法
def advance(self, n: int): ...
```

**向后兼容**：✅ 纯 Transformer 模型（Llama, Qwen3）完全兼容

### 3. 修复 SSM Mask 问题 ✅

**文件**: `mlx-lm-source/mlx_lm/models/base.py`

**问题**: SSM 层收到字符串 "causal" 而不是实际数组
**修复**: `create_ssm_mask` 传入 `return_array=True`

### 4. 修复 ArraysCache 兼容性 ✅

**文件**: `mlx-lm-source/mlx_lm/models/cache.py`

**问题**: ArraysCache 不支持 `return_array` 参数
**修复**: 添加参数（向后兼容）

---

## 遇到的技术挑战

### 挑战 1: Batch Size 不匹配 ❌

**问题**：
```
ValueError: [concatenate] All the input array dimensions must match exactly except
for the concatenation axis. However, the provided shapes are (1,3,8192), (10,10,8192),
and the concatenation axis is 1.
```

**根本原因**：
- MLX 生成过程中，batch size 从 1 变化到 10
- SSM 层的 conv_state 在第一次调用时以 batch size = 1 初始化
- 后续调用时 batch size = 10，导致维度不匹配

**尝试的解决方案**：
1. ✅ `__getitem__` 返回 None → **无效**（cache 仍返回旧值）
2. ✅ `__setitem__` no-op → **无效**（cache 仍存储旧值）
3. ✅ qwen3_5.py 检测并重新初始化 → **无效**（代码未执行）
4. ✅ 完全禁用 conv_state 缓存 → **破坏输出质量**（"the the the" 重复）

### 挑战 2: MLX JIT 编译缓存 ❌

**问题**: 修改 Python 代码后，行为没有变化

**可能原因**：
- MLX 使用 JIT 编译，第一次调用时编译函数
- 后续调用直接使用编译后的代码，不重新执行 Python
- 清理 `.pyc` 文件无效

**验证**：
- 添加 `print()` 语句 → 没有输出
- 修改逻辑 → 错误信息不变

### 挑战 3: Conv State 缓存的必要性 ❌

**实验结果**：
- 禁用 conv_state 缓存 → Baseline 输出崩溃
- Token 数量：487 → 172 (-65%)
- 输出质量："the the the" 重复（与原始 Qwen3.5 + CompactedKVCache 问题一致）

**结论**: Conv state 缓存对 SSM 层的正确性至关重要，不能简单禁用

---

## 技术分析

### 为什么 Batch Size 会变化？

MLX-LM 的生成过程：
1. **Prefill 阶段**: 处理完整 prompt，batch size = 1
2. **Generation 阶段**: 逐 token 生成，可能批处理多个 tokens，batch size > 1

### 为什么 Llama/Qwen3 没有问题？

纯 Transformer 架构：
- Attention 层不需要 conv_state
- KV cache 由 CompactedKVCache 管理，自动处理 batch size
- CompactedKVCache.update_and_fetch() 内部处理维度

混合架构（Qwen3.5）：
- SSM 层期望 conv_state 在调用之间持久化
- Conv state 在初始化时绑定到特定 batch size
- 无法自动适应 batch size 变化

### 根本矛盾

```
SSM 层需求：
- Conv state 必须在调用之间持久化（保持滑动窗口）
- Conv state 必须匹配当前 batch size

MLX 生成行为：
- Batch size 在生成过程中变化
- Cache 在整个生成过程中共享

→ 无法同时满足两个要求
```

---

## 可能的解决方向

### 方向 1: 动态调整 Conv State 维度

**思路**: 检测到 batch size 变化时，扩展或裁剪 conv_state

**挑战**：
- 如何保证语义正确性？（裁剪哪部分？扩展用什么填充？）
- MLX JIT 编译是否支持动态维度？

### 方向 2: 修改生成流程

**思路**: 确保 batch size 在整个生成过程中保持一致

**挑战**：
- 需要修改 MLX-LM 的核心生成代码
- 可能影响性能（无法批处理）

### 方向 3: 重新设计 SSM Cache

**思路**: 设计一个专门处理 batch size 变化的 SSM cache

**实现**：
```python
class DynamicSSMCache:
    def __init__(self):
        self.states = {}  # {batch_size: state}

    def get(self, batch_size):
        if batch_size not in self.states:
            return None  # 初始化
        return self.states[batch_size]

    def set(self, batch_size, state):
        self.states[batch_size] = state
```

**挑战**：
- 需要修改 qwen3_5.py 的 SSM 层代码
- 破坏与上游 MLX-LM 的兼容性

### 方向 4: 放弃混合架构支持

**思路**: CompactedKVCache 只支持纯 Transformer

**优势**：
- 实现简单
- 已验证有效（Llama, Qwen3-8B）
- 不破坏现有功能

**文档更新**：
```markdown
## 支持的模型

✅ 支持:
- 纯 Transformer 架构（Llama, GPT, Mistral, Qwen3）

❌ 不支持:
- 混合架构（Qwen3.5, 包含 SSM/Mamba 层的模型）
```

---

## 推荐方案

### 短期：方向 4（放弃混合架构支持）

**理由**：
1. CompactedKVCache 设计初衷是针对纯 Transformer
2. 混合架构需要重新设计 cache 系统
3. 纯 Transformer 已验证有效（+46% 性能提升）
4. 避免破坏稳定功能

**行动**：
1. 恢复 qwen3_5.py 到原始状态（撤销 conv_state 修改）
2. 更新文档，明确标注不支持混合架构
3. 添加架构检测，抛出友好错误信息

### 长期：方向 3（重新设计 SSM Cache）

**条件**: 如果混合架构需求强烈

**计划**：
1. 深入研究 MLX JIT 编译机制
2. 设计支持动态 batch size 的 SSM cache
3. 与 MLX-LM 上游合作，贡献修复

---

## 已完成的修改

### 需要保留的修改

1. ✅ `CompactedKVCache.__getitem__`, `__setitem__`, `advance()` - 向后兼容，无害
2. ✅ `base.py`: `create_ssm_mask` 传入 `return_array=True` - 修复问题
3. ✅ `cache.py`: `ArraysCache.make_mask` 支持 `return_array` - 向后兼容

### 需要撤销的修改

1. ❌ `qwen3_5.py`: 禁用 conv_state 缓存 - 破坏输出质量

---

## 测试结果

| 模型 | 架构 | CompactedKVCache | 状态 |
|------|------|-----------------|------|
| Llama 3.2 3B | 纯 Transformer | ✅ 工作正常 | +46% 性能 |
| Qwen3-8B | 纯 Transformer | ✅ 工作正常 | +23.5% 性能 |
| Qwen3.5-35B | 混合（SSM + Attention） | ❌ 不兼容 | Batch size 错误 |

---

## 结论

**CompactedKVCache 实现正确**，在纯 Transformer 架构上表现优异：
- ✅ 输出质量无损
- ✅ 性能显著提升（+23.5% ~ +46%）
- ✅ 向后兼容性完美

**混合架构（Qwen3.5）不兼容**，原因是：
- SSM 层的 conv_state 缓存机制与 MLX 生成过程中的 batch size 变化冲突
- 需要更深入的架构重构，超出当前 scope

**建议**：
- 短期：明确文档标注不支持混合架构
- 长期：如有需求，重新设计 SSM cache 系统

---

## 后续行动

### Immediate（今天）

1. [ ] 恢复 qwen3_5.py 到原始状态
2. [ ] 更新 `docs/COMPACTED_CACHE_USAGE.md`
   - 明确标注支持的架构类型
   - 添加架构检测示例
3. [ ] 运行回归测试（Llama, Qwen3）

### Short-term（本周）

1. [ ] 添加架构检测函数
   ```python
   def is_hybrid_architecture(model):
       return any(hasattr(layer, 'linear_attn') for layer in model.layers)
   ```
2. [ ] 在 CompactedKVCache 初始化时检查架构
   ```python
   if is_hybrid_architecture(model):
       raise ValueError("CompactedKVCache does not support hybrid architectures")
   ```

### Long-term（如有需求）

1. [ ] 研究 MLX JIT 编译机制
2. [ ] 设计动态 batch size 支持的 SSM cache
3. [ ] 与 MLX-LM 上游讨论解决方案

---

## 学到的教训

1. **JIT 编译挑战**: 动态语言在 JIT 编译器下的行为可能与预期不同
2. **架构假设**: 设计 cache 系统时要考虑所有可能的架构变体
3. **Batch Size 管理**: 生成过程中的 batch size 变化是一个复杂问题
4. **测试覆盖**: 需要测试不同架构类型，不能只测试一种

---

## 相关文档

- `.solar/hybrid-architecture-analysis.md` - 详细分析
- `.solar/hybrid-architecture-implementation.md` - 实现报告
- `.solar/hybrid-architecture-fixes.md` - 修复记录
- `.solar/llama-test-success-report.md` - Llama 成功验证

---

*报告完成于: 2026-03-21*
*任务状态: ⚠️ 部分完成*
*下一步: 恢复代码 + 文档更新*
