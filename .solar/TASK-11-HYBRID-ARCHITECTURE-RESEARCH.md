# Task #11: 混合架构压缩算法研究

**创建日期**: 2026-03-23
**状态**: 🔄 In Progress
**优先级**: P0 (CRITICAL)

---

## 执行摘要

基于 AM 算法修复的成功 (quality 0.898 → 0.999)，重新验证 Qwen3.5 混合架构的 KV cache 压缩方案。

**核心假设**: 之前的 Heterogeneous Cache 质量崩溃可能是由 AM 的两个 bug 引起：
1. 短序列 t > T 问题
2. Beta 补偿自由度不足 (约束比 20:1)

**验证目标**: 使用修复后的 AM 重新测试，预期质量从"完全乱码"恢复至 ≥0.90

---

## 背景回顾

### Qwen3.5-35B 架构

**混合架构**:
- 30 层 SSM (GatedDeltaNet) - State-Memory
- 10 层 Attention (Softmax Attention) - Attention-Memory
- 总计 40 层

**内存特性**:
- SSM 层: 小状态向量 (conv_state + ssm_state)
- Attention 层: 大 KV cache (grows with sequence length)

### 之前的失败 (2026-03-21)

**实验配置**:
| 配置 | compression_ratio | 生成质量 |
|------|-------------------|----------|
| Baseline | - | ✅ 正常 |
| Conservative | 2.0 | ❌ 乱码 |
| Moderate | 3.0 | ❌ 乱码 (相同) |
| Aggressive | 5.0 | ❌ 乱码 (相同) |

**关键发现**:
1. 所有压缩比产生**完全相同**的乱码
2. 只压缩 10/40 层就完全破坏质量
3. 无 shape mismatch，但输出完全崩溃

**当时的假设**:
- Hypothesis 1: AM 假设在混合架构中被打破 (Attention → SSM 误差累积)
- Hypothesis 2: Qwen3.5 Attention 层特殊实现，β 补偿失效
- Hypothesis 3: 累积误差在 SSM 层中放大

### AM 修复成功 (2026-03-23)

**修复内容**:
1. **Fix #1**: 短序列 t > T 问题
   - 问题: TruthfulQA T=15, t=25 导致质量=0.000
   - 修复: `if t >= T: t = max(T // 2, T - 1)`
   - 效果: 0.000 → 1.000

2. **Fix #2**: Beta DOF 优化
   - 问题: 约束比 20:1 导致 NNLS 不稳定
   - 修复: `n_effective = min(n_original, max(t // 2, 5))`
   - 效果: 约束比 20:1 → 7-12:1，质量 0.898 → 0.999

**关键洞察**:
- AM 的问题不是算法本身，而是实现细节
- 边界条件检查和约束比优化是关键

---

## 新假设

### Hypothesis: 之前的失败是 AM bugs 引起的

**推理链**:

1. **所有压缩比产生相同乱码** → 说明不是压缩太激进，而是 AM 本身失败
   - 可能原因: t > T bug 或 Beta DOF bug 导致 AM 完全失效
   - AM 失效 → β = -∞ 或 NaN → attention = 0 → 乱码

2. **压缩 10 层就崩溃** → 说明错误被放大
   - Attention 层输出异常 → SSM 层接收污染输入 → 递推放大误差
   - 即使只有 25% 的层有问题，误差也会累积到整个模型

3. **Qwen3.5 Attention 层可能有短序列** → 触发 t > T bug
   - 如果某些 Attention 层的 KV cache 较短 (T < 25)
   - 原 AM 会计算 t = max(25, T // 4) = 25 > T
   - 导致 NNLS rank deficient → β = -∞ → 质量崩溃

**验证方法**:
- 使用修复后的 AM 重新运行 hetero_cache_quality_test.py
- 如果质量恢复至 ≥0.90，说明假设正确
- 如果仍然乱码，说明有更深层次的不兼容性

---

## 验证计划

### Phase 1: 快速验证 (30 分钟)

**目标**: 确认修复后的 AM 是否解决问题

**步骤**:
1. 确认 `hetero_cache_quality_test.py` 使用的是修复后的 CompactedKVCache
2. 运行 Conservative 配置 (compression_ratio=2.0)
3. 检查生成质量

**判断标准**:
- ✅ 质量正常 (无乱码) → Hypothesis 正确，继续 Phase 2
- ❌ 仍然乱码 → Hypothesis 错误，进入 Phase 3 (深度诊断)

### Phase 2: 全面测试 (1-2 小时)

**目标**: 完整验证所有压缩配置

**测试矩阵**:
| 配置 | compression_ratio | max_size | 预期质量 |
|------|-------------------|----------|----------|
| Conservative | 2.0 | 8192 | ≥0.95 |
| Moderate | 3.0 | 8192 | ≥0.92 |
| Aggressive | 5.0 | 4096 | ≥0.90 |

**测试场景**:
1. 中文生成 (触发中文 token)
2. 长上下文 (触发序列长度变化)
3. 混合语言 (触发多种 token 分布)

**数据收集**:
- 生成质量 (人工检查 + gibberish detection)
- 内存节省 (压缩前后对比)
- 性能开销 (TG 变化)
- 压缩统计 (每层压缩比、β 分布)

### Phase 3: 深度诊断 (如果 Phase 1 失败)

**目标**: 识别混合架构的深层不兼容性

**诊断步骤**:
1. **单层测试**: 只压缩 layer 39 (最后一个 Attention 层)，验证是否仍乱码
2. **前向分析**: 对比压缩前后每层的激活值分布
3. **源码审查**: 检查 Qwen3.5 Attention 实现的特殊性
4. **β 分析**: 检查 β 值分布，确认是否异常 (NaN/-∞)
5. **Query 分析**: 检查查询生成的 queries 是否合理

**产出**:
- 根因分析报告
- 不兼容性的数学证明
- 替代方案建议

---

## 实施细节

### 修改 hetero_cache_quality_test.py

**检查点**:
1. 确认使用的 CompactedKVCache 包含 AM fixes
2. 添加详细日志 (每层压缩统计、β 分布、质量指标)
3. 添加 gibberish detection 自动化检测

**关键代码检查**:
```python
# 确认 CompactedKVCache 使用修复后的算法
cache = CompactedKVCache(
    max_size=self.max_size,
    compression_ratio=self.compression_ratio
)
# 应该内部调用修复后的 HighestAttentionKeysCompaction
```

### 测试脚本更新

**新增功能**:
1. 自动 gibberish detection
2. 逐层压缩统计记录
3. β 值分布可视化
4. 生成质量自动评分

**输出格式**:
```
Configuration: Conservative (ratio=2.0)
=====================================
Layer Statistics:
  Layer 9 (Attention): compressed 512→256, β_mean=0.005, β_std=0.012
  Layer 19 (Attention): compressed 512→256, β_mean=0.003, β_std=0.010
  ...

Generation Quality:
  Gibberish: NO ✅
  Repetition: None
  Coherence: HIGH
  Overall: PASS ✅

Performance:
  Baseline: 36.52 tok/s
  Compressed: 64.68 tok/s (+77%)
  Memory: 8.5 GB → 6.2 GB (-27%)
```

---

## 成功标准

### Must Have (必须满足)

1. **质量**: 所有配置生成质量正常 (无乱码，coherent)
2. **内存**: 内存节省 ≥ 20%
3. **性能**: 性能提升 ≥ 40% (由于压缩减少计算)
4. **稳定性**: 100 次生成无崩溃

### Should Have (应该满足)

1. **质量指标**: 平均质量 ≥ 0.90 (vs baseline)
2. **β 分布**: β 值无 NaN/Inf，分布合理 (均值接近 0，方差 < 0.1)
3. **压缩均匀**: 各层压缩比接近目标值 (±10%)
4. **长上下文**: 支持 8K+ tokens 无质量下降

### Nice to Have (锦上添花)

1. **自适应压缩**: 根据序列长度动态调整压缩比
2. **层级压缩**: 不同层使用不同压缩比
3. **在线监控**: 实时监控压缩质量和性能
4. **自动回退**: 检测到质量下降时自动禁用压缩

---

## 风险与缓解

### Risk #1: 假设错误，问题仍然存在

**概率**: 30%
**影响**: HIGH (阻塞整个 Task #11)

**缓解措施**:
1. 准备 Plan B: H2O 压缩替代 AM
2. 准备 Plan C: 只压缩 SSM 层，Attention 层保持原样
3. 准备 Plan D: 使用量化而非压缩

### Risk #2: 部分场景成功，部分场景失败

**概率**: 40%
**影响**: MEDIUM (需要额外调优)

**缓解措施**:
1. 识别失败场景的共同特征 (序列长度？token 类型？)
2. 针对失败场景调整压缩参数
3. 实现自适应策略 (好场景压缩，坏场景不压缩)

### Risk #3: 质量可接受但性能退化

**概率**: 20%
**影响**: MEDIUM (需要性能优化)

**缓解措施**:
1. Profiling 识别瓶颈
2. 优化 NNLS 求解器
3. 使用更快的 beta 计算方法 (log-ratio 而非 NNLS)

---

## 时间估算

| 阶段 | 工作量 | 预计时间 |
|------|--------|---------|
| Phase 1: 快速验证 | 代码检查 + 单次测试 | 30 分钟 |
| Phase 2: 全面测试 | 3 配置 × 3 场景 | 1-2 小时 |
| Phase 3: 深度诊断 (如果需要) | 5 步诊断 + 报告 | 4-6 小时 |
| 报告撰写 | 结果分析 + 文档 | 1 小时 |
| **总计** | - | **2-10 小时** |

**最佳情况** (Phase 1 成功): 2-3 小时
**最坏情况** (Phase 3 深度诊断): 10 小时

---

## 交付物

### 必须交付

1. ✅ 测试结果报告 (TASK-11-RESULTS.md)
2. ✅ 质量对比分析 (vs 原始失败结果)
3. ✅ 压缩统计数据 (每层、每配置)
4. ✅ 生成样本 (3 配置 × 3 场景)

### 可选交付

1. 📊 β 分布可视化图表
2. 📊 质量 vs 压缩比曲线
3. 📊 内存节省 vs 性能开销权衡图
4. 📄 深度诊断报告 (如果 Phase 1 失败)

---

## 下一步行动

### 立即执行 (Next 30 min)

1. ✅ 创建本文档
2. ⏭️ 检查 CompactedKVCache 是否包含 AM fixes
3. ⏭️ 更新 hetero_cache_quality_test.py (添加日志)
4. ⏭️ 运行 Phase 1 快速验证

### 根据 Phase 1 结果

**如果成功** (质量正常):
1. 运行 Phase 2 全面测试
2. 生成测试报告
3. 更新 STATE.md
4. 继续优化集成

**如果失败** (仍然乱码):
1. 启动 Phase 3 深度诊断
2. 生成诊断报告
3. 评估替代方案
4. 请求监护人决策

---

## 参考文档

1. `.solar/AM-FIX-RESULTS.md` - AM 修复成功报告
2. `.solar/critical-finding-am-incompatibility.md` - 原失败分析
3. `.solar/hetero-cache-quality-report.md` - 质量对比实验
4. `.solar/deep-analysis-am-compression-failures.md` - AM 失败根因
5. `benchmarks/hetero_cache_quality_test.py` - 测试脚本

---

**创建时间**: 2026-03-23
**预计完成**: 2026-03-23 (同日完成)
**负责人**: Solar AI
**审批人**: 监护人昊哥
