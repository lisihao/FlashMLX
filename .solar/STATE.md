# FlashMLX State

## Mission
优化 MLX-LM 推理性能，专注运行时加速和内存优化。

## Current Status
✅ **KVTC 调查完成** - 发现 KVTC 设计用于 Prompt-Cache 持久化，不适合 FlashMLX 的运行时优化目标。

## Constraints
- 不破坏现有 MLX-LM API 兼容性
- 性能优化必须有可测量的改进
- 优先运行时优化，而非离线/存储优化

## Decisions

### [2026-03-20] KVTC 不集成到 FlashMLX，迁移到 ThunderOMLX
- **发现**: KVTC 设计用于 Prompt-Cache 持久化（磁盘压缩），不是运行时内存压缩
- **证据**:
  1. 官方 Benchmark 测试的是 "serialization" 而非 "generation"
  2. 必须解压后使用，运行时内存占用无改善
  3. E2E 测试显示直接用压缩 Cache 质量完全崩溃
- **结论**:
  - ❌ 不符合 FlashMLX 运行时优化目标
  - ✅ 适合 ThunderOMLX 长上下文持久化场景
- **详细分析**: `.solar/kvtc-design-intent-analysis.md`

## Progress

### Done
- ✅ KVTC P0: Metal GPU 加速 - 编解码性能优化
- ✅ KVTC P1: 增量压缩 - 动态 Cache 增长优化
- ✅ KVTC P2: DCT Transform - 无校准快速压缩
- ✅ KVTC P3: Per-Head 校准 - 精度提升优化
- ✅ KVTC P4: Magnitude + 分级量化
- ✅ KVTC PCA: 数据驱动维度压缩
- ✅ KVTC E2E 测试: 0.5B, 3B 模型质量验证
- ✅ KVTC 设计意图调查: 源码分析确认真实用途

### In-Progress
- ⏸️ KVTC 集成工作暂停（待迁移到 ThunderOMLX）

### Blocked
- 无

## Next Actions

### FlashMLX 方向
1. 探索真正的运行时压缩方案
   - 研究 MQA/GQA (减少 KV heads)
   - 研究 StreamingLLM (保留关键 tokens)
   - 研究 Paged Attention (内存分页)

2. 专注 FlashAttention 优化
   - Metal GPU Kernel 优化
   - Memory layout 优化
   - Kernel fusion

### ThunderOMLX 方向（建议）
1. 集成 KVTC 持久化能力
   - 实现 CacheManager API
   - 支持长提示缓存保存/加载
   - 支持 RAG 文档缓存

2. 优化存储格式
   - 研究更好的压缩算法
   - 支持分块加载

## Files Modified
- `mlx_lm/models/kvtc_pca_codec.py` - PCA 压缩实现
- `tests/test_kvtc_pca.py` - PCA 单元测试
- `tests/test_kvtc_pca_e2e.py` - PCA E2E 测试
- `tests/test_kvtc_e2e_serial.py` - 集成测试（错误用法）
- `.solar/kvtc-design-intent-analysis.md` - 调查报告

## Metrics
- KVTC 压缩率: 40-50x（磁盘）
- KVTC 运行时内存节省: 0x（不适用）
- E2E 测试模型: 0.5B, 3B
- 测试配置: DCT-Fixed, Magnitude, PCA-8, PCA-16
