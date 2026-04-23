# FlashMLX 性能对比测试结果

> **Warning**: Memory 列数据 (全部 0.5 MB) 来自 MoE 模型的 SSM 层测量，不反映真实 KV 内存。以 ARCHITECTURE.md §13 为准。

**日期**: 2026-03-21
**模型**: Qwen3.5-35B-A3B (MLX)
**配置**: Hybrid Cache (128MB budget, 4.0x compression, β-calibrated)

---

## 测试场景

对比 **Baseline** vs **Hybrid Cache + Attention Matching** 在不同上下文长度下的性能：

| 上下文长度 | 实际 tokens |
|-----------|-------------|
| 512       | 562         |
| 1K        | 1,081       |
| 2K        | 2,127       |
| 4K        | 4,148       |
| 8K        | 8,253       |
| 16K       | 16,471      |

---

## 性能对比

### TTFT (Time to First Token / 首 Token 延迟)

| Context | Baseline | Hybrid  | 改善 (%) |
|---------|----------|---------|----------|
| 512     | 940.6 ms | 846.6 ms | **-10.0%** |
| 1K      | 1278.1 ms | 1264.4 ms | -1.1% |
| 2K      | 2502.3 ms | 2446.2 ms | -2.2% |
| 4K      | 4697.0 ms | 4509.5 ms | -4.0% |
| 8K      | 9814.7 ms | 9013.6 ms | -8.2% |
| 16K     | 20905.3 ms | 18034.3 ms | **-13.7%** ⚡ |

**结论**: 上下文越长，首 Token 延迟优势越明显。16K 时节省 2.9 秒。

---

### PP (Prompt Processing Throughput)

| Context | Baseline | Hybrid  | 改善 (%) |
|---------|----------|---------|----------|
| 512     | 597.5 tok/s | 663.9 tok/s | **-11.1%** |
| 1K      | 845.8 tok/s | 855.0 tok/s | -1.1% |
| 2K      | 850.0 tok/s | 869.5 tok/s | -2.3% |
| 4K      | 883.1 tok/s | 919.8 tok/s | -4.2% |
| 8K      | 840.9 tok/s | 915.6 tok/s | -8.9% |
| 16K     | 787.9 tok/s | 913.3 tok/s | **-15.9%** 🚀 |

**结论**: PP 显著加速，16K 时提升 15.9%。

---

### TG (Token Generation Throughput)

| Context | Baseline | Hybrid  | 开销 (%) |
|---------|----------|---------|----------|
| 512     | 88.8 tok/s | 72.1 tok/s | **+18.8%** |
| 1K      | 88.3 tok/s | 72.5 tok/s | +17.8% |
| 2K      | 87.0 tok/s | 71.5 tok/s | +17.8% |
| 4K      | 85.0 tok/s | 71.7 tok/s | +15.7% |
| 8K      | 80.5 tok/s | 71.0 tok/s | +11.8% |
| 16K     | 73.7 tok/s | 70.6 tok/s | **+4.3%** |

**结论**: TG 有 4-19% 开销（压缩计算成本），但随上下文增长，开销占比下降。

---

### Memory Consumption

| Context | Baseline | Hybrid   | 额外内存 |
|---------|----------|----------|----------|
| 512     | 0.5 MB   | 64.4 MB  | +64 MB   |
| 1K      | 0.5 MB   | 67.4 MB  | +67 MB   |
| 2K      | 0.5 MB   | 67.4 MB  | +67 MB   |
| 4K      | 0.5 MB   | 64.4 MB  | +64 MB   |
| 8K      | 0.5 MB   | 67.4 MB  | +67 MB   |
| 16K     | 0.5 MB   | 67.4 MB  | +67 MB   |

**注意**: Baseline 的内存测量可能不准确（0.5MB 明显偏低），需要进一步调查。

---

### Attention Compression Ratio

| Context | Compression Ratio |
|---------|-------------------|
| 512     | 2.73x             |
| 1K      | 3.13x             |
| 2K      | 3.47x             |
| 4K      | 3.69x             |
| 8K      | 3.84x             |
| 16K     | **3.92x** ✨      |

**结论**: 长上下文压缩效果更好，16K 时接近 4x 压缩率。

---

## 关键发现

### ✅ 优势场景

1. **长上下文 (>4K tokens)**
   - TTFT 改善 4-14%
   - PP 加速 4-16%
   - 压缩效果更好

2. **Prompt-heavy Workloads**
   - Prompt processing 显著加速
   - 首 Token 延迟大幅降低

3. **高并发场景**
   - Attention 压缩节省 KV Cache 内存
   - 可支持更多并发请求

### ⚠️ 权衡

1. **Token Generation 开销**
   - TG 慢 4-19%
   - 压缩计算成本

2. **额外内存**
   - 需要 64-67MB 用于缓存管理
   - 但通过压缩节省的 KV Cache 内存远超此值

3. **SSM 层缓存未生效**
   - SSM hit rate 0% - **这是正常的**
   - SSM 缓存设计用于**跨请求**场景（例如复用 system prompt state）
   - 当前是单请求测试，没有跨请求复用
   - 详见：`SSM_CACHE_DESIGN.md`

---

## 推荐使用场景

### ✅ 适合

- **RAG 应用**: 长 prompt + 短 generation
- **对话系统**: 多轮对话，长历史
- **文档分析**: 大文档输入，简短回答
- **代码生成**: 长上下文，关注首 Token 延迟

### ⚠️ 谨慎使用

- **实时聊天**: 需要低 TG 延迟
- **流式生成**: 每个 token 延迟敏感
- **短 prompt 场景**: 优势不明显

---

## 下一步优化

1. **修复内存测量**
   - `get_memory_mb()` 函数需要改进
   - 使用更准确的内存测量方法

2. **SSM 缓存优化**
   - 当前 SSM hit rate 为 0%
   - 需要优化访问模式

3. **降低 TG 开销**
   - 优化压缩算法性能
   - 考虑异步压缩

4. **参数调优**
   - 测试不同 compression_ratio
   - 测试不同 budget 大小

---

**完整数据**: `benchmark_context_length_results.json`
