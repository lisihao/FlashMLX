# FlashMLX 活跃功能清单

> **更新日期**: 2026-03-22

---

## ✅ 活跃功能（生产可用）

### Attention Matching Compressor

**位置**: `src/flashmlx/cache/attention_matching_compressor.py`

**功能**: KV Cache 压缩（针对 Attention 层）

**特性**:
- Attention Matching 算法
- β 校准机制
- Quality Path 保护
- 可配置压缩率

**状态**: ✅ 生产可用

**文档**:
- 核心实现：`src/flashmlx/cache/attention_matching_compressor.py`
- 测试：单元测试通过
- 示例：待补充

**后续优化方向**:
1. Quality Path 优化 (Task #48)
2. 选择性压缩 (Task #52)
3. 保守的压缩方法 (Task #56)

---

### Compressed KV Cache

**位置**: `src/flashmlx/cache/compressed_kv_cache.py`

**功能**: 压缩 KV Cache 包装器

**特性**:
- 与 MLX-LM 兼容
- 自动压缩触发
- 统计信息收集

**状态**: ✅ 生产可用

---

### Budget Manager

**位置**: `src/flashmlx/cache/budget_manager.py`

**功能**: 内存预算管理

**特性**:
- 三层预算分配（Hot/Warm/Cold）
- 动态内存分配
- 预算监控

**状态**: ✅ 可用（用于 Attention 层）

---

### Pinned Control State

**位置**: `src/flashmlx/cache/pinned_control_state.py`

**功能**: 关键通道保护

**特性**:
- Control channel 检测
- 关键通道 pinning
- 质量保护机制

**状态**: ✅ 可用

---

## ⚠️ 废弃功能（已封存）

### SSM Cache (所有组件)

**废弃日期**: 2026-03-22

**组件列表**:
- `HybridCacheManager`
- `SimplifiedSSMCacheManager`
- `HotTierManager`
- `WarmTierManager`
- `ColdArchive`
- `MigrationTrigger`
- `LayerScheduler`
- `ManagedArraysCache`
- `PerLayerSSMCache`

**废弃原因**:
1. 场景与 ThunderLLAMA prefix caching 重叠
2. GPU 稳定性问题（page fault）
3. 无实际生产使用价值

**代码状态**:
- 代码保留但封存
- 入口已关闭（从 `__init__.py` 移除）
- 测试/示例已禁用

**详细说明**: `SSM_CACHE_DEPRECATION.md`

**替代方案**: 使用 ThunderLLAMA prefix caching

---

## 🎯 当前开发重点

### 1. Attention Matching 优化

**目标**: 提升 KV Cache 压缩质量和效率

**任务**:
- [ ] Quality Path 优化 (#48)
- [ ] 选择性压缩（只压缩非关键层）(#52)
- [ ] 保守的压缩方法（避免质量损失）(#56)

### 2. 性能测试

**目标**: 量化压缩收益

**任务**:
- [ ] 内存节省测试 (#79)
- [ ] 性能开销测试 (#80)
- [ ] 参数调优 (#81)

### 3. 文档完善

**目标**: 提供清晰的使用指南

**任务**:
- [ ] Attention Matching 使用示例
- [ ] 压缩参数调优指南
- [ ] 最佳实践文档

---

## 📦 项目结构

```
FlashMLX/
├── src/flashmlx/cache/
│   ├── attention_matching_compressor.py  ✅ ACTIVE
│   ├── compressed_kv_cache.py            ✅ ACTIVE
│   ├── budget_manager.py                 ✅ ACTIVE
│   ├── pinned_control_state.py           ✅ ACTIVE
│   ├── per_layer_attention_cache.py      ✅ ACTIVE
│   │
│   ├── hybrid_cache_manager.py           ⚠️ DEPRECATED
│   ├── simplified_ssm_cache.py           ⚠️ DEPRECATED
│   ├── hot_tier_manager.py               ⚠️ DEPRECATED
│   ├── warm_tier_manager.py              ⚠️ DEPRECATED
│   ├── cold_archive.py                   ⚠️ DEPRECATED
│   ├── migration_trigger.py              ⚠️ DEPRECATED
│   ├── layer_scheduler.py                ⚠️ DEPRECATED
│   ├── managed_arrays_cache.py           ⚠️ DEPRECATED
│   └── per_layer_ssm_cache.py            ⚠️ DEPRECATED
│
├── examples/
│   ├── attention_matching_demo.py        🔜 TODO
│   └── cross_request_ssm_reuse.py        ⚠️ DEPRECATED
│
├── tests/
│   └── test_simplified_ssm_cache.py      ⚠️ DEPRECATED
│
└── docs/
    ├── SSM_CACHE_DEPRECATION.md          📋 决策记录
    ├── SSM_CACHE_IMPROVEMENT_SUMMARY.md  📋 存档
    └── ACTIVE_FEATURES.md                📋 本文件
```

---

## 🚀 快速开始

### 使用 Attention Matching Compressor

```python
from flashmlx.cache import (
    AttentionMatchingCompressor,
    CompressedKVCache,
    BudgetManager,
    BudgetConfig
)

# 创建 budget manager
budget_config = BudgetConfig(total_budget_bytes=128 * 1024 * 1024)
budget_manager = BudgetManager(budget_config)

# 创建 compressor
compressor = AttentionMatchingCompressor(
    compression_ratio=3.0,
    budget_manager=budget_manager
)

# 创建压缩 KV cache
kv_cache = CompressedKVCache(
    compressor=compressor,
    layer_idx=0
)

# 使用...
```

### 禁用 SSM Cache（确认已封闭）

```python
# ❌ 以下导入会失败（已从 __init__.py 移除）
from flashmlx.cache import SimplifiedSSMCacheManager  # ImportError!
from flashmlx.cache import HybridCacheManager         # ImportError!

# ✅ 只能导入 Attention 相关组件
from flashmlx.cache import AttentionMatchingCompressor  # OK
from flashmlx.cache import CompressedKVCache            # OK
```

---

## 📞 联系与反馈

如有问题或建议：
1. 查看文档：`SSM_CACHE_DEPRECATION.md`
2. 查看任务列表：`.solar/tasks/`
3. 联系维护者

---

*Last updated: 2026-03-22*
*Status: Attention Matching only, SSM cache deprecated*
