# FlashMLX State

## Mission

构建基于 MLX 的高性能推理引擎，专注于 Flash Attention 优化和 Metal GPU 加速。

## Constraints

- 基于 MLX 0.31.2 核心库
- 兼容 MLX-LM 模型接口
- 保持 Apple Silicon 优化
- 不破坏 MLX 原有功能

## Current Plan

### Phase 1: 项目初始化 ✓
- [x] 创建项目结构
- [x] 导入 MLX 和 MLX-LM 源码
- [x] 建立文档体系

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

## Progress

### Done
- ✅ 创建项目目录结构
- ✅ 导入 MLX 源码 (26 Metal kernels)
- ✅ 导入 MLX-LM 源码
- ✅ 创建项目文档

### In-Progress
- 🔄 分析 MLX 架构

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
