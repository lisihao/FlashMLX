# FlashMLX 上下文长度性能测试报告

**测试日期**: 2026-03-21
**硬件**: Apple M4 Pro
**模型**: Qwen3.5-35B-A3B (MLX format, ~19GB)
**模型路径**: `/Volumes/toshiba/models/qwen3.5-35b-mlx/`

---

## 📊 完整测试结果

### Baseline（标准 MLX-LM，未优化）

| Context Length | TTFT (ms) | PP (tok/s) | TG (tok/s) | Memory (MB) |
|----------------|-----------|------------|------------|-------------|
| **512 tokens** | 1,022.5 | 549.6 | 88.1 | 0.5 |
| **1K tokens** | 1,273.8 | 848.6 | 88.6 | 0.5 |
| **2K tokens** | 2,406.0 | 884.0 | 87.4 | 0.5 |
| **4K tokens** | 4,576.3 | 906.4 | 84.6 | 0.5 |
| **8K tokens** | 9,694.1 | 851.3 | 80.3 | 0.5 |
| **16K tokens** | 20,827.0 | 790.8 | 73.7 | 0.5 |

### Hybrid Cache（混合缓存 + Attention Matching）

| Context Length | TTFT (ms) | PP (tok/s) | TG (tok/s) | Memory (MB) | Overhead |
|----------------|-----------|------------|------------|-------------|----------|
| **512 tokens** | 783.5 | 717.3 | 88.9 | 0.5 | **-23.4%** ⚠️ |
| **1K tokens** | 1,264.1 | 855.1 | 88.8 | 0.5 | **-0.8%** |
| **2K tokens** | 2,403.8 | 884.8 | 87.5 | 0.5 | **-0.1%** |
| **4K tokens** | 4,652.9 | 891.5 | 85.6 | 0.5 | **+1.7%** |
| **8K tokens** | 9,761.1 | 845.5 | 80.1 | 0.5 | **+0.7%** |
| **16K tokens** | 21,409.4 | 769.3 | 73.7 | 0.5 | **+2.8%** |

---

## 🔍 性能分析

### 1. PP (Prompt Processing) 趋势

```
PP Throughput vs Context Length (Baseline)

1000 ┤
 900 ┤         ┌─────┐
 800 ┤    ┌────┤     │
 700 ┤    │    │     │                      ┌───
 600 ┤────┤    │     │              ┌───────┤
 500 ┼────┘    │     └──────────────┤       └───
     └────┴────┴────┴────┴────┴────┴────┴────┴───
     512  1K   2K   4K   8K   16K  tokens
```

**观察**：
- ✅ 549 tok/s (512) → 906 tok/s (4K)：提升 65%
- ⚠️ 906 tok/s (4K) → 790 tok/s (16K)：下降 13%
- **原因**: 4K 以下受益于批处理优化，16K 时内存压力增大

### 2. TG (Token Generation) 趋势

```
TG Throughput vs Context Length (Baseline)

90 ┤─────┐
85 ┤     └──┐
80 ┤        └───┐
75 ┤            └───────┐
70 ┤                    └────
   └────┴────┴────┴────┴────┴────
   512  1K   2K   4K   8K   16K  tokens
```

**观察**：
- ⚠️ 88 tok/s (512) → 73 tok/s (16K)：下降 17%
- **原因**: 更大的 KV cache 访问开销
- **影响**: 对用户体验的影响有限（仍然很快）

### 3. TTFT (首 token 延迟) 趋势

```
TTFT vs Context Length (线性增长)

21s ┤                              ┌──
18s ┤                          ┌───┤
15s ┤                      ┌───┤   │
12s ┤                  ┌───┤   │   │
 9s ┤              ┌───┤   │   │   │
 6s ┤          ┌───┤   │   │   │   │
 3s ┤      ┌───┤   │   │   │   │   │
 0s ┼──────┤   │   │   │   │   │   │
    └──────┴───┴───┴───┴───┴───┴───
    512   1K  2K  4K  8K  16K tokens
```

**观察**：
- ✅ 近乎完美的线性增长
- 比例：~1.3 ms/token (TTFT / context_length)

---

## 🚨 关键问题：混合缓存未生效

### 证据

**所有测试中**：
- ❌ SSM hit rate: **0.0%** (预期: 70-90%)
- ❌ Attention compression: **1.00x** (预期: 3.5-4.5x)
- ❌ Memory saved: **0%** (预期: 18-20%)

### 预期行为 vs 实际行为

| 指标 | 预期（如果生效） | 实际测量 | 状态 |
|------|-----------------|---------|------|
| **SSM hit rate** | 70-90% | 0.0% | ❌ 未生效 |
| **Attention compression** | 3.5-4.5x | 1.00x | ❌ 未生效 |
| **Memory saved** | 18-20% | 0% | ❌ 未生效 |
| **PP overhead** | +15-20% | -0.8% to +2.8% | ⚠️ 异常 |
| **TG overhead** | <10% | <1% | ✅ 符合（但因为未生效） |

### 可能原因

1. **注入机制失败**
   - `inject_hybrid_cache_manager()` 可能没有正确替换原生 cache
   - Layer routing 没有生效

2. **缓存对象未被使用**
   - MLX-LM 可能有内部缓存机制绕过了我们的包装
   - 需要检查 monkey patch 是否正确

3. **统计数据收集失败**
   - `get_statistics()` 可能返回空对象
   - 需要添加调试日志

### 诊断建议

```python
# 添加调试日志
print(f"Original cache type: {type(model.layers[0].self_attn.cache)}")
# 注入后
print(f"After injection: {type(model.layers[0].self_attn.cache)}")
# 生成后
print(f"Cache statistics: {cache_wrapper.get_statistics()}")
```

---

## 📈 与 ThunderOMLX 的对比

### 标准 MLX-LM (本次测试)

| Metric | Value |
|--------|-------|
| **PP @ 4K** | 906 tok/s |
| **TG @ 4K** | 84.6 tok/s |
| **TTFT @ 4K** | 4.58s |

### ThunderOMLX（昊哥的优化引擎）

| Optimization | Improvement |
|--------------|-------------|
| **TTFT 降低** | 90.6% |
| **SSD 加速** | 185x |
| **技术** | 分页缓存、多级跳跃、混合哈希、ContextPilot |

### 结合方案（理论）

如果 FlashMLX 混合缓存生效并与 ThunderOMLX 结合：

| Metric | Baseline | + ThunderOMLX | + FlashMLX | Combined (理论) |
|--------|----------|---------------|------------|-----------------|
| **TTFT @ 4K** | 4.58s | 0.43s (-90.6%) | 5.36s (+17%) | 0.50s (-89%) |
| **Memory** | 100% | 100% | 82% (-18%) | 82% (-18%) |
| **技术栈** | MLX-LM | KV Cache 管理 | 混合压缩 | 缓存管理 + 压缩 |

**注**: FlashMLX 目前未生效，需要修复后才能验证结合效果。

---

## 🎯 总结

### ✅ 成功获得

1. **完整的 Baseline 性能数据**（512/1K/2K/4K/8K/16K）
2. **性能趋势分析**（PP/TG/TTFT 随上下文变化）
3. **标准 MLX-LM 在 M4 Pro 上的性能基准**

### ❌ 需要修复

1. **混合缓存注入机制**（完全未生效）
2. **内存测量**（显示 0.5 MB，明显不准确）
3. **统计数据收集**（SSM hit rate 和 compression 都是 0）

### 📋 下一步

1. **调试混合缓存注入**
   - 添加详细日志
   - 验证 monkey patch 是否正确
   - 检查 Layer routing

2. **修复内存测量**
   - 使用 `mx.get_active_memory()` 替代已弃用的 API
   - 在生成前后正确测量

3. **验证压缩机制**
   - 单独测试 AttentionMatchingCompressor
   - 验证 β 校准是否工作

4. **与 ThunderOMLX 结合测试**
   - 在 ThunderOMLX 的基础上测试 FlashMLX
   - 验证两种优化能否叠加

---

**测试数据文件**:
- `benchmark_context_length_results.json` - 完整 JSON 数据
- `benchmark_context_length.log` - 完整测试日志
- 本报告 - 分析和总结

---

*最后更新: 2026-03-21*
*FlashMLX 版本: 1.0 (混合缓存未生效)*
