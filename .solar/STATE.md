# FlashMLX State

## Mission

构建基于 MLX 的高性能推理引擎，专注于 Flash Attention 优化和 Metal GPU 加速。

## Constraints

- 基于 MLX 0.31.2 核心库
- 兼容 MLX-LM 模型接口
- 保持 Apple Silicon 优化
- 不破坏 MLX 原有功能
- **铁律：任何优化前必须先用 Profiler 分析，找到瓶颈**
- **铁律：Profiler 分析不到位时，优先优化 Profiler 而非系统**

## Current Plan

### Phase 1: 项目初始化 + Profiler ✓
- [x] 创建项目结构
- [x] 导入 MLX 和 MLX-LM 源码
- [x] 建立文档体系
- [x] 实现 Profiler 工具 (性能 + 内存 + 延迟 + 锁 + IO + 并发)

### Phase 2: 分析 MLX 现状 ✓
1. ✅ 分析 MLX Flash Attention 实现
2. ✅ 分析 GEMV kernel 实现
3. ✅ 创建性能 baseline

### Phase 3: 核心优化
1. Flash Attention 优化
2. GEMV 优化
3. 量化优化

### Phase 4: 集成测试
1. 性能 benchmark
2. 正确性验证
3. 稳定性测试

## Decisions

- [2026-03-18] 项目定位：基于 MLX 的性能优化版本，不是 fork，而是增强层
- [2026-03-18] 技术栈：MLX 0.31.2 + MLX-LM (commit 4a21ffd) + 自定义优化
- [2026-03-18] Profiler 设计：六维度 (性能+内存+延迟+锁+IO+并发)，TrackedLock 手动插桩，JSON 日志
- [2026-03-19] Kernel 分析结论：
  - Flash Attention 主要瓶颈：指数运算 (40-50%) + SIMD reduction (20-30%)
  - GEMV 主要瓶颈：内存加载 (60-70%)
  - 优化优先级：Flash Attention exp 优化 (预期 +10-15%) 或 GEMV 内存优化 (预期 +15-20%)
  - 决策方法：先创建 baseline，基于实际瓶颈数据确定优化顺序
- [2026-03-19] Baseline 测试结论：
  - TTFT 缩放接近线性 (2.7x / 16.0x)，Prefill 瓶颈不明显
  - 内存管理良好 (4.14 → 4.97 GB)
  - TG 测试数据异常，需要修复
  - **最终决策**：优先优化 GEMV 内存访问 (影响 TG，预期 +15-20%)
  - 理由：(1) TG 是实际使用的关键指标，(2) GEMV 是 TG 阶段瓶颈，(3) 预期收益更高
- [2026-03-20] **KVTC 集成决策**：
  - 基于论文 https://arxiv.org/pdf/2511.01815 实现 KV Cache Transform Coding
  - 现状：已有基础实现（PCA/quantization/DP/DEFLATE），但 CPU-bound（NumPy）
  - 主要瓶颈：encode 500ms+，decode 300ms+，严重影响实际可用性
  - **优化策略**：Metal GPU 加速 (P0) → 预期 10-20× 加速
  - **次要优化**：Incremental compression (P1)、DCT transform (P2)、Per-head calibration (P3)、Magnitude pruning (P4)
  - **决策理由**：KV cache 压缩对长上下文场景至关重要，但当前性能瓶颈限制了实用价值

## Progress

### Done
- ✅ 创建项目目录结构
- ✅ 导入 MLX 源码 (26 Metal kernels)
- ✅ 导入 MLX-LM 源码
- ✅ 创建项目文档
- ✅ 实现基础 Python 包结构
- ✅ 实现 Flash Attention 包装
- ✅ 单元测试通过 (3/3)
- ✅ 性能 benchmark 正常运行
- ✅ 修复 scale 参数和延迟执行问题
- ✅ 构建测试报告
- ✅ **实现全面 Profiler 工具 (6维度)**
  - 性能分析：函数级插桩、三级粒度 (BASIC/DETAILED/FULL)
  - 内存分析：Python heap + Metal GPU 内存跟踪
  - 延迟分析：min/max/mean/P95/P99 + TTFT + inter-token latency
  - 锁分析：TrackedLock、争抢检测、死锁预警
  - IO 分析：吞吐量、最慢操作、文件读写统计
  - 并发分析：线程生命周期、GIL 争抢、并发问题检测
  - 测试覆盖：21 tests 全通过 (3 core + 6 profiler + 5 memory/latency + 7 lock/IO/concurrency)
  - 文档：完整设计文档 + 3个示例
- ✅ **完成 MLX Kernel 深度分析**
  - Flash Attention: 3 种变体 (single-pass, 2-pass)，找到瓶颈 (exp×2 + simd_sum)
  - GEMV: 2 种变体 (standard, transposed)，找到瓶颈 (内存加载)
  - 创建详细分析报告 (MLX_KERNEL_ANALYSIS.md)
  - 确定优化方向：指数运算优化 (+10-15%) 和内存访问优化 (+15-20%)
- ✅ **完成性能 Baseline 测试 (更新)**
  - 测试模型：Qwen3.5-2B (1084, 4340, 17363 tokens)
  - TTFT 缩放：14.1x / 16.0x = 88% 效率 ✅ (优秀)
  - TG 性能：58.4 tok/s (稳定，变化率 < 10%) ✅
  - 内存使用：4.30 → 5.23 GB (稳定)
  - 使用新的 `baseline_benchmark_simple.py` 进行准确测量
  - 创建详细 baseline 报告 (BASELINE_PERFORMANCE_REPORT.md)
  - **关键发现**: TTFT 性能优秀，优先优化 GEMV 内存访问 (目标 TG +15%)
- ✅ **完成 KVTC 模块分析与任务规划**
  - 全面扫描现有 KVTC 实现（kvtc_codec.py, cache.py, tests, benchmarks）
  - 对比论文识别实现差异：已有（PCA/量化/DP/DEFLATE），缺失（Metal GPU加速/增量压缩/DCT/Per-head校准/真正pruning）
  - 创建完整优化任务清单（Task #13-#18）：6个优化阶段，优先级 P0-P4
  - **核心瓶颈确认**：CPU-bound NumPy 实现导致 encode 500ms+，decode 300ms+
  - **最高优先级**：Metal GPU 加速（P0），预期 10-20× 性能提升

### In-Progress
- 🔄 **KVTC 优化** (当前优先级)
  - ✅ 分析现有实现，识别性能瓶颈
  - ✅ Metal GPU 加速 Phase 1 (Task #13 完成)
    - ✅ Metal kernels 实现 (kvtc.metal)
    - ✅ Python wrapper (metal_kvtc_codec.py)
    - ✅ 单元测试通过 (9 tests)
    - ❌ 性能问题：小矩阵慢 9 倍（MLX matmul 瓶颈）
    - ✅ 缓解方案：自动回退 NumPy
  - ✅ Phase 1.5: Optimized Metal 实现 (完成)
    - ✅ optimized_metal_kvtc.py 实现完成
    - ✅ 持久化 GPU 缓冲区（避免重复传输）
    - ✅ 向量化量化（处理所有分组）
    - ✅ 智能回退机制（batch < 300 用 NumPy）
    - ✅ 快速正确性测试通过
    - ✅ 性能测试完成（2026-03-20）
    - ✅ C1 优化完成：threshold 100 → 300（避免性能退化）
  - ✅ 性能测试结果分析
    - Metal vs NumPy: batch 100 慢 17x, batch 500 慢 5x
    - 根因：CPU-GPU 传输开销 + MLX matmul 小矩阵性能差
    - 决策：调整 threshold 到 300，只在超大 batch 使用 Metal
  - ⏳ 增量压缩 (Task #14, P1)
  - ⏳ DCT 变换 (Task #15, P2)
  - ⏳ Per-head 校准 (Task #16, P3)
  - ⏳ 幅度剪枝 (Task #17, P4)

### Blocked
- (无)

## Next Actions

### Phase 3 优化任务清单

1. **详细 Profiling 分析** ⏳ (Next)
   - 目标: 识别 GEMV 热点和内存访问瓶颈
   - 动作:
     - 使用 FlashMLX Profiler (FULL level)
     - 测试真实 stream_generate 场景
     - 分析函数级耗时分布
     - 识别 GatedDeltaNet cache/concat 开销
   - 产出:
     - 函数级性能热点报告
     - Flash Attention vs GEMV 时间占比
     - 内存访问模式分析
     - GEMV 优化的具体切入点

2. **优先做 `GatedDeltaNet` cache / concat**
   - 目标: 减少小而密的缓存拼接和同步点
   - 动作:
     - 验证 `conv_state=3` 是否适合 ring buffer / 预分配滚动窗口
     - 评估是否能减少 `mx.concatenate` 次数或把它挪到更粗粒度
   - 预期:
     - 降低 GPU launch / flush 空洞
     - 降低 decode 阶段的碎片化开销

3. **压 `hidden-size RMSNorm`**
   - 目标: 先动总量最大的 norm 路径
   - 动作:
     - 评估层前 / 层后 / final norm 的融合空间
     - 评估 `linear-attn q/k norm` 与输出 norm 的融合空间
   - 预期:
     - 减少 kernel 数量
     - 减少 Python 侧同步与 GIL 串行放大

4. **回头验证 GEMV baseline**
   - 目标: 保持 Phase 3 主线不丢
   - 动作:
     - 在 Qwen3.5 热区确认后，再回到 GEMV 内存优化的 baseline 计划
   - 说明:
     - GEMV 仍然是候选高 ROI 优化
     - 但当前 Qwen3.5 证据链更明确地指向 `concat` / `rms_norm`

5. **复测 35B / 2B 对照**
   - 目标: 验证优化是否真的降低了 stall 和总时长
   - 动作:
     - 同一 prompt、同一 token 预算、同一 profiler 配置复跑
     - 对比 `function_call sum`、`region time`、`gil_contention_estimate`
   - 产出:
     - 优化前后差异
     - 是否继续深化 ring buffer / kernel fusion

### Phase 3 候选优化

**Option A: GEMV 内存访问优化** (推荐)
- 使用 `simdgroup_matrix` 优化加载
- 增加分块大小 (TM=8, TN=8)
- 预期：+15-20% TG

**Option B: Flash Attention 指数优化**
- 多项式近似代替 exp
- 预期：+10-15% TTFT

---

*最后更新: 2026-03-19*
