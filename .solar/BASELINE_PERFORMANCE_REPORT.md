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

| Prompt Tokens | TTFT (ms) | Prompt TPS | Relative |
|---------------|-----------|-----------|----------|
| **1084 tokens** | 622.5 | 1741.5 tok/s | 1.0x (baseline) |
| **4340 tokens** | 2011.6 | 2157.5 tok/s | 3.2x |
| **17363 tokens** | 8770.7 | 1979.7 tok/s | 14.1x |

### TG (Token Generation) - Decode 性能

| Prompt Tokens | Decode TPS | Generated Tokens |
|---------------|-----------|------------------|
| **1084 tokens** | 60.3 tok/s | 128 |
| **4340 tokens** | 59.3 tok/s | 128 |
| **17363 tokens** | 55.5 tok/s | 128 |
| **Average** | **58.4 tok/s** | 128 |

### 缩放分析

```
序列长度增加: 16.0x  (1084 → 17363 tokens)
TTFT 增加:     14.1x (622.5 → 8770.7 ms)

结论: ✅ 优秀的线性缩放 (比例: 14.1/16.0 = 0.88)
```

这表明 **Prefill 阶段的性能缩放优秀**，接近理想的线性缩放。

---

## 内存使用

| Prompt Tokens | Peak Memory (GB) | Increase |
|---------------|------------------|----------|
| 1084 tokens   | 4.30 GB         | baseline |
| 4340 tokens   | 4.74 GB         | +0.44 GB |
| 17363 tokens  | 5.23 GB         | +0.49 GB |

**分析**:
- 内存增长平缓，主要是 KV cache
- 17363 tokens 的 KV cache 增加了 0.93 GB (相比 baseline)
- 内存管理良好，无明显泄漏
- 平均每 1000 tokens 增加约 0.057 GB

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

1. **TTFT 缩放优秀** (14.1x vs 16.0x = 88% 效率)
   - 说明 Prefill 阶段 (Flash Attention) **性能优秀**
   - 接近理想的线性缩放
   - 2B 模型的内存带宽利用良好

2. **TG 性能稳定** ✅
   - 平均 decode TPS: 58.4 tok/s
   - 稳定范围: 55.5 - 60.3 tok/s
   - 变化率: < 10%，符合预期

---

## 性能目标

基于 baseline 数据，设定优化目标：

### Phase 3 优化目标

| 优化项 | 当前性能 | 目标性能 | 预期提升 |
|--------|----------|----------|----------|
| **TTFT (17K tokens)** | 8770.7 ms | 7893.6 ms | **-10%** |
| **TG (Decode)** | 58.4 tok/s | 67.2 tok/s | **+15%** |
| **内存峰值** | 5.23 GB | < 5.5 GB | 保持 |

### 优化优先级

基于 baseline 数据和 kernel 分析，调整优化优先级：

1. **优先级 🔴 高**: GEMV 内存访问优化
   - 原因：TG 阶段是 GEMV 密集型
   - 当前 58.4 tok/s 有提升空间
   - 预期收益 15-20%
   - **这是 Phase 3 的主线任务**

2. **优先级 🟡 中**: Flash Attention 指数优化
   - 原因：TTFT 缩放已经很好（88% 效率）
   - 但仍可优化 10-15% (有价值)

3. **优先级 🟢 低**: 其他优化
   - Kernel Fusion
   - 量化优化

---

## 下一步行动

### 1. ✅ 修复 TG 测试 (已完成)

使用新的 `baseline_benchmark_simple.py`：
- ✅ 使用 `stream_generate()` 获取流式输出
- ✅ 准确计时每个 token
- ✅ 验证 TG 稳定性 (58.4 tok/s, 变化率 < 10%)

### 2. 详细 Profiling (Next)

使用 FlashMLX Profiler 深度分析：
```python
from flashmlx.profiler import Profiler, ProfilerConfig, InstrumentationLevel

config = ProfilerConfig(
    level=InstrumentationLevel.FULL,
    capture_memory=True
)

with Profiler("baseline_detailed", config=config):
    # 使用 stream_generate 进行真实场景测试
    for _ in stream_generate(model, tokenizer, prompt, max_tokens=128):
        pass
```

分析目标：
- 哪些函数最耗时？
- Flash Attention vs GEMV 时间占比
- 内存访问模式
- GatedDeltaNet cache/concat 开销

### 3. 开始 GEMV 优化 (Phase 3 主线)

基于 baseline 数据，确定优化方向：
- **首选**: GEMV 内存访问优化 (+15-20%)
  - 使用 `simdgroup_matrix` 优化加载
  - 增加分块大小 (TM=8, TN=8)
  - 目标：TG 从 58.4 tok/s → 67.2 tok/s

---

## 关键发现

### ✅ 积极发现

1. **TTFT 缩放优秀**: 14.1x vs 16.0x = 88% 效率，接近理想线性缩放
2. **TG 性能稳定**: 58.4 tok/s，变化率 < 10%，测试准确可靠
3. **内存管理良好**: 5.23 GB 峰值，无明显泄漏
4. **基础性能扎实**: 2B 模型在 M4 Pro 上表现正常

### 📊 性能特征

1. **Prefill (TTFT)**:
   - Prompt TPS: 1741.5 - 2157.5 tok/s
   - 缩放效率: 88% (优秀)
   - 瓶颈不明显，可以暂缓优化

2. **Decode (TG)**:
   - Decode TPS: 58.4 tok/s (平均)
   - **有提升空间**: 目标 67.2 tok/s (+15%)
   - 这是 Phase 3 的主攻方向

### 🎯 优化方向

**已确认**: 优先优化 **GEMV 内存访问**，因为：
1. ✅ TG 数据准确可靠（不是之前的不稳定）
2. ✅ TG 阶段是 GEMV 密集型
3. ✅ 预期收益更高 (15-20% vs 10-15%)
4. ✅ 影响实际使用体验（生成速度）

**下一步**:
1. ✅ TG 测试已修复
2. ⏳ 详细 profiling (识别 GEMV 热点)
3. ⏳ 开始 GEMV 优化实现

---

*Baseline Performance Report v1.0*
*完成于: 2026-03-19*
*测试工具: MLX-LM 0.31.1*
*下一步: 修复 TG 测试 + 详细 Profiling*
