# FlashMLX Baseline Performance Report

> **日期**: 2026-03-19
> **模型**: Qwen3.5-2B-Opus-Distilled (MLX format)
> **测试环境**: M4 Pro, MLX 0.31.1

---

## 测试配置

| 参数 | 值 |
|------|-----|
| 模型 | qwen3.5-2b-opus-distilled |
| 模型大小 | 2B parameters |
| 测试序列长度 | 512, 2048, 8192 tokens |
| 生成长度 | 128 tokens (目标) |
| 量化 | FP16 (MLX default) |

---

## 性能基准数据

### TTFT (Time To First Token) - Prefill 性能

| Prompt Length | TTFT (ms) | Tokens/ms | Relative |
|---------------|-----------|-----------|----------|
| **745 tokens** | 3137.7 | 0.237 | 1.0x (baseline) |
| **2981 tokens** | 3949.2 | 0.755 | 1.26x |
| **11926 tokens** | 8570.9 | 1.391 | 2.73x |

### 缩放分析

```
序列长度增加: 16.0x  (745 → 11926 tokens)
TTFT 增加:     2.7x  (3137.7 → 8570.9 ms)

结论: ✅ 接近线性缩放 (比例: 2.7/16.0 = 0.17)
```

这表明 **Prefill 阶段的性能瓶颈不严重**，缩放基本符合预期。

---

## 内存使用

| Prompt Length | Peak Memory (GB) | Increase |
|---------------|------------------|----------|
| 745 tokens    | 4.14 GB         | baseline |
| 2981 tokens   | 4.60 GB         | +0.46 GB |
| 11926 tokens  | 4.97 GB         | +0.37 GB |

**分析**:
- 内存增长平缓，主要是 KV cache
- 11926 tokens 的 KV cache 增加了 0.83 GB (相比 baseline)
- 内存管理良好，无明显泄漏

---

## 瓶颈验证

### 与 Kernel 分析报告对比

**Kernel 分析报告预测** (`MLX_KERNEL_ANALYSIS.md`):

1. **Flash Attention 瓶颈**:
   - 指数运算 (fast::exp × 2): 40-50%
   - SIMD Reduction (simd_sum): 20-30%
   - 内存加载: 15-20%

2. **GEMV 瓶颈**:
   - 内存加载: 60-70%
   - SIMD Shuffle: 10-15%
   - MAC 操作: 10-15%

**Baseline 测试观察**:

1. **TTFT 缩放近乎线性** (2.7x vs 16.0x)
   - 说明 Prefill 阶段 (Flash Attention) **不是主要瓶颈**
   - 可能原因：2B 模型较小，内存带宽充足

2. **TG 数据异常** (需要修复测试)
   - 初步数据显示 TG 不稳定
   - 需要正确的流式生成测试

---

## 性能目标

基于 baseline 数据，设定优化目标：

### Phase 3 优化目标

| 优化项 | 当前性能 | 目标性能 | 预期提升 |
|--------|----------|----------|----------|
| **TTFT (11K tokens)** | 8570.9 ms | 7713.8 ms | **-10%** |
| **TG** | TBD | TBD | **+15%** |
| **内存峰值** | 4.97 GB | < 5.0 GB | 保持 |

### 优化优先级

基于 baseline 数据和 kernel 分析，调整优化优先级：

1. **优先级 🟡 中**: Flash Attention 指数优化
   - 原因：TTFT 缩放接近线性，瓶颈不明显
   - 但仍可优化 10-15% (有价值)

2. **优先级 🔴 高**: GEMV 内存访问优化
   - 原因：TG 阶段是 GEMV 密集型
   - TG 不稳定可能与内存访问有关
   - 预期收益 15-20%

3. **优先级 🟢 低**: 其他优化
   - Kernel Fusion
   - 量化优化

---

## 下一步行动

### 1. 修复 TG 测试 ✅ (Next)

需要正确测量 Token Generation 性能：
- 使用 `generate_step()` 获取流式输出
- 准确计时每个 token
- 验证 TG 稳定性

### 2. 详细 Profiling

使用 FlashMLX Profiler 深度分析：
```python
from flashmlx.profiler import Profiler, ProfilerConfig, InstrumentationLevel

config = ProfilerConfig(
    level=InstrumentationLevel.FULL,
    capture_memory=True
)

with Profiler("baseline_detailed", config=config):
    model.generate(prompt, max_tokens=128)
```

分析：
- 哪些函数最耗时？
- Flash Attention vs GEMV 时间占比
- 内存访问模式

### 3. 确定第一个优化目标

基于详细 profiling 数据，在以下两者中选择：
- **Option A**: Flash Attention 指数优化 (+10-15%)
- **Option B**: GEMV 内存访问优化 (+15-20%)

选择标准：
- 实际测量的瓶颈占比
- 实现难度 vs 预期收益
- 风险评估

---

## 关键发现

### ✅ 积极发现

1. **TTFT 缩放良好**: 接近线性，说明 Prefill 优化良好
2. **内存管理稳定**: 无明显泄漏，增长符合预期
3. **基础性能合理**: 2B 模型在 M4 Pro 上表现正常

### ⚠️  需要关注

1. **TG 测试不准确**: 需要修复流式生成测试
2. **TG 可能不稳定**: 初步数据显示变化较大
3. **缺少详细 profiling**: 需要更细粒度的分析

### 🎯 优化方向

**当前结论**: 优先优化 **GEMV 内存访问**，因为：
1. TG 阶段是 GEMV 密集型
2. 预期收益更高 (15-20% vs 10-15%)
3. 影响实际使用体验（生成速度）

**但需要先完成**:
1. 修复 TG 测试
2. 详细 profiling
3. 基于数据确认

---

*Baseline Performance Report v1.0*
*完成于: 2026-03-19*
*测试工具: MLX-LM 0.31.1*
*下一步: 修复 TG 测试 + 详细 Profiling*
