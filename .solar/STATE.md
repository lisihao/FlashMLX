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
- ✅ **完成性能 Baseline 测试**
  - 测试模型：Qwen3.5-2B (745, 2981, 11926 tokens)
  - TTFT 缩放：2.7x / 16.0x = 接近线性 ✅
  - 内存使用：4.14 → 4.97 GB (稳定)
  - 创建 baseline 报告 (BASELINE_PERFORMANCE_REPORT.md)
  - **关键发现**: TTFT 瓶颈不明显，优先优化 GEMV (TG 阶段)

### In-Progress
- 🔄 **Phase 2 完成，准备进入 Phase 3**
  - Phase 2 已完成：Kernel 分析 + Baseline 测试
  - Phase 3 目标：核心优化 (GEMV 优先)

### Blocked
- (无)

## Next Actions

### Phase 3 优化任务清单

1. **先把测量边界拆开**
   - 目标: 区分“真实产品开销”与“profiler 同步开销”
   - 动作:
     - 把 `mx.eval` 从“每个函数”尽量收敛到“region 级”
     - 对比有 / 无同步的 wall-clock 差值
   - 产出:
     - 哪些 stall 是观测放大的
     - 哪些 stall 是模型本身的真实热点

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
