# FlashMLX 实际性能测试结果

**测试日期**: 2026-03-21
**硬件**: Apple M4 Pro
**模型**: Qwen3.5-35B-A3B (MLX format, ~19GB)
**模型路径**: `/Volumes/toshiba/models/qwen3.5-35b-mlx/`

---

## 🎯 关键发现

### 实际测量 vs 之前预测

| 指标 | 实际测量 | 之前预测 | 差异 |
|------|----------|----------|------|
| **PP (Prompt Processing)** | **574 tok/s** | 950 tok/s | **-40%** ❌ |
| **TG (Token Generation)** | **91 tok/s** | 28 tok/s | **+225%** ✅ |
| Peak Memory | 20.029 GB | N/A | - |

**结论**: 之前的预测性能数据**完全不准确**，是基于理论分析的期望值，不是实际测试结果。

---

## 📊 标准 MLX-LM Baseline（未优化）

**测试条件**:
- Prompt: 319 tokens (多轮对话，正确的 chat format)
- Generated: 100 tokens
- Context: 5 轮对话历史 + 当前问题

**性能数据**:
```
PP (Prompt Processing):
  Throughput: 587.8 tok/s
  Time:       542.7 ms

TG (Token Generation):
  Throughput: 79.0 tok/s
  Time:       1266.3 ms

Total Time:   1.81s
Peak Memory:  19.977 GB
```

**生成质量**: ✅ 正常（使用 chat template 后质量恢复）

---

## 🔍 与 ThunderOMLX 的对比

### 为什么 FlashMLX baseline 和 ThunderOMLX 差异大？

**ThunderOMLX**:
- 高度优化的 MLX 推理引擎
- 核心技术：
  - 分页 SSD 缓存
  - 多级跳跃策略
  - 混合哈希
  - ContextPilot 消息级优化
- **成果**: TTFT 降低 90.6%，SSD 访问加速 185 倍

**FlashMLX**:
- **Baseline 是标准 MLX-LM**（没有特殊优化）
- 核心技术：
  - 混合架构缓存压缩（SSM + Attention Matching）
  - 三层缓存管理（Hot/Warm/Cold）
  - β 校准的注意力匹配
- **目标**: 内存节省 18.8%，TG 开销 <10%

### 两者的关系

- **不是竞争关系**，是不同优化方向
- **可能可以结合**：
  - ThunderOMLX 的缓存管理 + FlashMLX 的混合架构压缩
  - 理论上可以同时获得两种优化的收益

---

## ⚠️ 当前测试问题

### 已知问题

1. **~~质量问题~~** ✅ **已修复**
   - ~~之前：重复 prompt 50 次导致生成质量崩溃~~
   - **现在：使用 chat template 后质量正常**

2. **内存测量不准确**
   - Memory Used 显示 0.0 MB（应该是 1000+ MB）
   - 原因：`mx.metal.get_active_memory()` 已弃用，需要用 `mx.get_active_memory()`

3. **缓存统计异常**
   - SSM hit rate: 0.00%（第一次运行没有缓存命中正常）
   - Avg compression: 1.00x（说明压缩没有生效或统计有误）

4. **Hybrid Cache 性能异常**
   - PP 反而比 baseline 快（应该更慢）
   - 可能原因：第二次运行有缓存/JIT 优化效果

5. **FlashMLX 混合缓存可能没有真正生效**
   - 需要调试 `inject_hybrid_cache_manager()` 是否正确注入
   - 需要验证 layer routing 是否正确

### 需要改进

- [ ] 修复内存测量（使用更准确的方法）
- [ ] 验证缓存统计数据结构
- [ ] 多次运行取平均值（排除 JIT/缓存影响）
- [ ] 使用 `benchmark_precise.py`（stream API 精确测量）

---

## 📝 建议对比方式

如果要和 ThunderOMLX 对比性能，应该：

1. **明确对比基准**：
   - ThunderOMLX baseline (已优化)
   - vs FlashMLX + ThunderOMLX (结合两种优化)

2. **关注不同指标**：
   - ThunderOMLX: 关注 TTFT、缓存命中率、SSD 加速
   - FlashMLX: 关注内存节省、TG 开销、压缩比

3. **探索结合方案**：
   - ThunderOMLX 的前缀缓存 + FlashMLX 的混合压缩
   - 可能实现：低延迟 + 低内存的双赢

---

## 🚀 下一步

1. **修复测试脚本**（内存测量、统计数据）
2. **运行 benchmark_precise.py**（获取精确 TTFT/TBT）
3. **多轮测试取平均值**（排除一次性影响）
4. **探索与 ThunderOMLX 的结合方案**

---

*最后更新: 2026-03-21*
