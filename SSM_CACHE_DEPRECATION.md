# SSM 缓存废弃决策记录

> **日期**: 2026-03-22
> **决策**: 废弃 SSM 缓存功能，保留代码但封闭入口
> **决策人**: 昊哥

---

## 📋 决策摘要

**废弃组件**：
- Hot/Warm/Cold 三层缓存
- SimplifiedSSMCacheManager
- PerLayerSSMCache
- ManagedArraysCache
- HybridCacheManager（SSM 部分）

**保留组件**：
- AttentionMatchingCompressor
- CompressedKVCache
- BudgetManager
- PinnedControlState

---

## 🎯 废弃原因

### 1. 场景完全重叠

**ThunderLLAMA 已有功能**：
```cpp
✅ Prefix Caching (THUNDER_LMCACHE)
✅ Prefix Matching (THUNDER_PREFIX_MATCHING)
✅ KV Cache Reuse (--cache-reuse 256)
```

**SSM Cache 目标场景**：
- 多轮对话 - 复用 system prompt
- RAG - 复用知识库 context
- Few-shot - 复用 examples

**结论**：100% 场景重叠，ThunderLLAMA 已覆盖所有需求

### 2. GPU 稳定性问题

**问题**：
```
任何 SSM 状态管理（无论架构复杂度）都会触发：
❌ GPU page fault (kIOGPUCommandBufferCallbackErrorPageFault)
❌ 无法在实际推理中使用
```

**已尝试方案**：
- Hot/Warm/Cold 三层 (16x overhead) → GPU fault
- Simplified 单层 (11x overhead) → GPU fault

**根因**：SSM 状态缓存与 MLX Metal 内存管理的根本性冲突

### 3. 架构选择矛盾

**如果用 ThunderLLAMA**：
```
模型: Qwen2.5 (纯 Transformer)
缓存: Prefix Caching 100% 覆盖
结论: SSM Cache 无用
```

**如果用 Qwen3.5-MoE**：
```
Attention 层 (10/40): 可用 Prefix Cache
SSM 层 (30/40):      GPU bug 无法缓存
结论: 得不偿失
```

### 4. 开发成本 vs 收益

| 维度 | ThunderLLAMA | SSM Cache |
|------|--------------|-----------|
| 稳定性 | ✅ 生产级 | ❌ GPU bug |
| 适用模型 | ✅ 所有 Transformer | ⚠️ 仅 Qwen3.5-MoE |
| 场景覆盖 | ✅ 100% | ✅ 100% (理论) |
| 实际可用 | ✅ 是 | ❌ 否 |

---

## 📦 保留代码的原因

虽然废弃，但保留代码而不是删除：

### 1. 技术资产保存

**已完成的工作**：
- 架构简化：Hot/Warm/Cold → Simplified (开销降低 1.5x)
- 跨请求复用机制（单元测试验证通过）
- 完整的缓存管理抽象

**价值**：
- 如果未来 MLX Metal 内存管理改进
- 如果出现新的混合架构模型
- 可以快速恢复和调整

### 2. 学习参考

**设计模式价值**：
- 三层缓存架构设计
- LRU 策略实现
- 跨请求状态管理
- 内存预算管理

### 3. 避免重复造轮子

如果未来有类似需求：
- 不需要从零开始
- 已有完整的实现和测试
- 已知的坑和解决方案

---

## 🔒 封闭措施

### 1. 移除公共接口

**`__init__.py` 变更**：
```python
# ❌ 移除 SSM 相关导出
# from .simplified_ssm_cache import SimplifiedSSMCacheManager
# from .hot_tier_manager import HotTierManager
# from .hybrid_cache_manager import HybridCacheManager

# ✅ 保留 Attention 相关
from .attention_matching_compressor import AttentionMatchingCompressor
from .compressed_kv_cache import CompressedKVCache
```

### 2. 添加 DEPRECATED 标注

所有 SSM 缓存文件头部添加：
```python
"""
DEPRECATED: 2026-03-22

This module is deprecated and should not be used in production.

Reason:
- Functionality overlaps with ThunderLLAMA prefix caching
- GPU stability issues (page fault) in actual inference
- No real-world use case

Preserved for potential future use if:
- MLX Metal memory management improves
- New hybrid architecture models emerge
- Different caching strategies needed

See: SSM_CACHE_DEPRECATION.md
"""
```

### 3. 禁用示例和测试

```python
# examples/cross_request_ssm_reuse.py
# benchmark_simplified_ssm_impact.py
# test_simplified_ssm_cache.py

# 添加跳过标记
@pytest.mark.skip(reason="SSM cache deprecated, see SSM_CACHE_DEPRECATION.md")
```

---

## ✅ 保留的组件（继续优化）

### Attention Matching 相关

**保留并继续优化**：
```python
✅ AttentionMatchingCompressor
✅ CompressedKVCache
✅ BudgetManager
✅ PinnedControlState
✅ β 校准机制
```

**原因**：
- 针对 Attention 层的 KV cache 压缩
- 不依赖 SSM 状态管理
- 没有 GPU 稳定性问题
- 有实际优化空间

**优化方向**（后续）：
1. Quality Path 优化 (#48)
2. 选择性压缩 (#52)
3. 保守的压缩方法 (#56)

---

## 📊 性能数据（存档）

### 简化架构改进

| 指标 | Hot/Warm/Cold | Simplified | 改进 |
|------|--------------|------------|------|
| 开销 | 16x (0.177 μs) | 11x (0.131 μs) | 1.5x ↓ |
| 代码量 | ~500 行 | ~100 行 | 5x ↓ |
| 层数 | 3 层 | 1 层 | 简化 |

### 跨请求复用（理论）

| 场景 | System Prompt | PP 加速 | TTFT 降低 |
|------|--------------|---------|----------|
| 多轮对话 | 100 tokens | 6x | ~125 ms |
| RAG | 1000 tokens | 50x | ~1250 ms |
| Few-shot | 200 tokens | 10x | ~250 ms |

### GPU 问题

```
测试结果：
✅ 单元测试: 30 层 × 1000 次无 hang
❌ 实际推理: GPU page fault

问题类型：
kIOGPUCommandBufferCallbackErrorPageFault

触发条件：
enable_managed_cache() + 实际 model.generate()
```

---

## 🔮 未来可能的场景

**如果以下情况出现，可以考虑恢复 SSM Cache**：

1. **MLX Metal 改进**
   - MLX 修复了 SSM 状态的内存管理问题
   - 提供更稳定的 GPU 内存分配机制

2. **新的混合架构**
   - 出现 SSM 占比更高的模型（如 80% SSM + 20% Attention）
   - ThunderLLAMA prefix cache 无法覆盖

3. **不同的缓存策略**
   - 需要 SSM 特定的压缩算法
   - 需要 SSM 状态的跨请求迁移

4. **性能关键场景**
   - 超长 context (> 100K tokens)
   - 极低延迟要求 (< 10ms TTFT)

---

## 📝 相关文档

- `SSM_CACHE_IMPROVEMENT_SUMMARY.md` - 改进工作总结
- `benchmark_simplified_results_final.log` - 最终测试结果
- `.solar/performance-regression-analysis-2026-03-15.md` - 性能分析

---

## ✍️ 签名

**决策人**: 昊哥
**执行人**: Solar (Claude Sonnet 4.5)
**日期**: 2026-03-22
**版本**: v1.0

---

*"好的代码不应该被删除，而应该被封存。"*
