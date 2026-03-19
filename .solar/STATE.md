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

### Phase 2: 分析 MLX 现状 (Next)
1. 分析 MLX Flash Attention 实现
2. 评估 Metal kernels 性能瓶颈
3. 确定优化方向（GEMV, MatMul, Attention）

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

### In-Progress
- 🔄 Phase 1 完成，准备进入 Phase 2：分析 MLX 现状

### Blocked
- (无)

## Next Actions

1. **分析 Flash Attention 实现**
   ```bash
   cat ~/FlashMLX/mlx-source/mlx/backend/metal/kernels/scaled_dot_product_attention.metal
   ```

2. **分析 GEMV kernel**
   ```bash
   cat ~/FlashMLX/mlx-source/mlx/backend/metal/kernels/gemv.metal
   ```

3. **创建性能 baseline**
   - 使用 MLX-LM 跑 Qwen 模型
   - 记录 PP/TG 性能
   - 作为优化基准

---

*最后更新: 2026-03-18*
