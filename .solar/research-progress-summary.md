# FlashMLX Heterogeneous Memory Compaction - 研究进展总结

**研究周期**: 2026-03-21 ~ 2026-03-25
**目标**: 为 Qwen3-8B 实现 Heterogeneous KV cache 压缩 (AM 算法)
**状态**: ❌ 失败 (概念验证成功，但质量完全不可用)

---

## 研究动机

Qwen3.5-32B 混合架构（27 Attention + 5 SSM 层）需要针对不同层的压缩策略。AM (Attention Matching) 算法在论文中声称可以高质量压缩 Attention KV cache。

**研究问题**：AM 算法能否在 Qwen3-8B 纯 Transformer 架构上，以 Lazy Compression 方式工作？

---

## 实现进度

### ✅ 已完成

1. **Heterogeneous KVCache 架构**
   - 支持不同层使用不同 budget (256 vs 159)
   - 支持 Lazy Compression (Prefill → Generate → Compress → Continue)
   - 架构稳定，无崩溃，无 NaN

2. **AM 离线校准系统**
   - On-policy 增量式校准 (36 层，512 tokens)
   - 生成校准文件：\`am_calibration_qwen3-8b_2.0x_onpolicy.pkl\`
   - 包含 selected_indices, beta, Ck, budget

3. **Beta 补偿机制**
   - Log-space beta 补偿集成到 \`base.py\`
   - 支持 GQA (Grouped Query Attention)
   - 自动从 HybridKVCache 读取 beta

4. **完整测试套件**
   - 正确的 Lazy Compression 测试流程
   - 不同压缩比、部分层、序列长度测试
   - Beta 诊断、indices 分析工具

### ❌ 失败

1. **质量验证**
   - 所有压缩配置质量损失 100% (0-6% 一致性)
   - 输出完全乱码或严重错位
   - 即使最保守配置（Layer 0 单层压缩）也失败

2. **问题诊断**
   - Beta 零值导致 attention 崩溃
   - Selected_indices 不适配 Lazy Compression
   - 离线校准与 Lazy Compression 本质不兼容

---

## 关键发现

详见：
- `critical-finding-am-incompatibility.md`
- `hetero-cache-quality-report.md`

### 发现 1: Beta 零值崩溃

```
Layer 27-35 的 beta 中存在 0 值
→ log(0 + 1e-10) = -23.026
→ Attention scores += -23
→ Attention weights = exp(-23) ≈ 1e-10 ≈ 0
→ 模型失去上下文 → 输出乱码
```

### 发现 2: Fixed Indices 不适配动态 Cache

```
校准: 512 tokens → selected_indices [0-313]
Lazy Compression: 530 tokens (500 prefill + 30 generated)
→ Indices 只选择 [0-313]
→ 完全丢弃 [314-530]（包括所有新生成的 30 tokens）
→ 模型失去短期记忆 → 输出错位
```

### 发现 3: 根本矛盾

```
AM 离线校准:
  - 假设: 固定 512 tokens，一次性压缩
  - 输出: 固定 selected_indices

Lazy Compression:
  - 实际: 可变 cache (500-2000+ tokens)，按需压缩
  - 需要: 动态 selected_indices

→ 不兼容！所有优化尝试都基于同一个不适配的校准文件
```

---

## 测试结果汇总

| 测试场景 | 配置 | 压缩比 | 质量 | 输出特征 |
|----------|------|--------|------|----------|
| 完整 36 层压缩 | 500 tokens, 所有层 | 2.34x | 0% | 完全乱码 |
| 部分层压缩 | 18/12/6 层 | 2.20x-1.60x | 0-7% | 乱码 |
| 不同序列长度 | 350/400/450/512 | 2.20x | 0% | 乱码 |
| Layer 0 单层 | 350 tokens, 1 层 | 2.15x | 0% | 乱码 |
| 不同压缩比 | 1.5x/2.0x/3.0x | 全部 2.26x | 全部 6% | 相同乱码 |
| 禁用 beta | 2.0x, 无 beta | 2.26x | 0% | 错位但可读 |

**共性**: 所有配置都失败，且所有压缩比产生相同结果（校准文件决定实际压缩比）。

---

## 错误总结

### 我犯的错误

1. **反复声称 "混合架构"**
   - 用户多次纠正：Qwen3-8B 全是 Attention 层
   - 我仍然错误地说 "Layer 27-35 是 SSM 层"
   - **实际**: Layer 27-35 的 budget=159 是因为校准预算设置，不是因为层类型不同

2. **错误的测试流程**
   - 最初测试了两个独立 baseline，而不是连续的 Lazy Compression
   - 用户明确纠正后才修复

3. **过度优化不适配的方案**
   - 尝试了 7+ 种优化（部分层、不同比例、不同长度等）
   - 所有尝试都基于同一个不适配的校准文件
   - 应该更早发现根本矛盾

### 正确的做法（未来参考）

1. **先验证概念兼容性**
   - 在大量实现前，先验证 AM 离线校准是否适配 Lazy Compression
   - 分析 selected_indices 是否适应动态 cache 大小

2. **诊断优先于优化**
   - 第一次失败时就应该诊断 beta 和 indices
   - 而不是盲目尝试不同参数组合

3. **尊重用户纠正**
   - 用户说"不是混合架构"就是不是
   - 不要反复犯同样的错误

---

## 经验教训

### 对 AM 算法的理解

1. **AM 不是 Attention-Memory 的通用压缩器**
   - 即使是 softmax attention，也可能因为架构交互而失效
   - 混合架构的层间交互比单层特性更重要

2. **离线校准有严格假设**
   - 假设固定 cache 大小
   - 假设一次性压缩
   - 不适配动态、按需的 Lazy Compression

3. **Beta 补偿非常脆弱**
   - Beta=0 会导致 attention 完全崩溃
   - 需要在校准时严格 clip beta >= 0.1

### 对混合架构的理解

（注：Qwen3-8B 不是混合架构，但未来 Qwen3.5-32B 是）

1. **不同层可能需要不同压缩策略**
   - Attention 层: AM / H2O / StreamingLLM
   - SSM 层: 可能不需要压缩（state size 已经固定）

2. **误差会跨层传播**
   - Attention 层的压缩误差 → SSM 层放大
   - 需要整体质量评估，不能只看单层

---

## 下一步方向

### 方向 1: 放弃 AM，使用 H2O / StreamingLLM

**理由**：
- H2O 和 StreamingLLM 天然适配 Lazy Compression
- 不需要离线校准
- 已有成熟实现和验证

**工作量**：中等（1-2 周）

### 方向 2: 修复 AM 以适配 Lazy Compression

**需要**：
1. 重新校准，确保 beta >= 0.1
2. 实现动态 selected_indices（根据 cache 大小调整）
3. 在 Lazy Compression 场景下重新验证

**工作量**：大（3-4 周）

### 方向 3: 设计新的混合压缩算法

**思路**：
- 结合 AM 的精确性（校准）和 H2O 的动态性（实时 attention score）
- On-policy 校准 + 动态 indices
- 专为 Lazy Compression 优化

**工作量**：非常大（6-8 周）

### 推荐

**立即**: 使用 H2O 或 StreamingLLM（方向 1）

**长期**: 如果 H2O 质量不足，考虑方向 3（新算法）

---

## 成果清单

### 文档

1. `critical-finding-am-incompatibility.md` - 完整根因分析
2. `hetero-cache-quality-report.md` - 质量评测报告
3. `research-progress-summary.md` - 本文件（研究总结）

### 代码

1. `hybrid_cache.py` - Heterogeneous KVCache 实现
2. `base.py` - Beta 补偿集成
3. 校准系统：
   - `calibrate_am_offline.py`
   - `am_calibration_qwen3-8b_2.0x_onpolicy.pkl`

### 测试脚本

1. `/tmp/correct_lazy_compression_test.py` - 正确 Lazy Compression 测试
2. `/tmp/test_compression_ratio_fixed.py` - 压缩比测试
3. `/tmp/diagnose_beta.py` - Beta 诊断
4. `/tmp/test_without_beta.py` - 禁用 beta 测试
5. `/tmp/check_selected_indices.py` - Indices 分析

---

## 结论

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   研究结论: AM 压缩不适用于 Qwen3-8B Lazy Compression        │
│                                                             │
│   ✅ 成功: Heterogeneous cache 架构实现正确                 │
│   ✅ 成功: 离线校准系统工作正常                             │
│   ✅ 成功: 概念验证（架构无崩溃）                           │
│                                                             │
│   ❌ 失败: 质量完全不可用（0-6% 一致性）                    │
│   ❌ 失败: 两个根本问题无法通过参数调整修复                 │
│                                                             │
│   根因: 离线校准 ≠ Lazy Compression                         │
│                                                             │
│   建议: 使用 H2O 或 StreamingLLM 替代 AM                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**概念验证的价值**：
- 快速发现 AM 算法的根本性问题
- 避免在不可行方案上浪费更多时间
- 为未来选择合适算法提供清晰方向

---

**生成于**: 2026-03-25
**作者**: Solar (Claude Opus 4.6)
**监护人**: 昊哥
