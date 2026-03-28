# Heterogeneous Memory Compaction - 质量报告

**日期**: 2026-03-25
**模型**: Qwen3-8B
**测试场景**: Lazy Compression (500 tokens Prefill + 30 tokens Generate + Compress + 30 tokens Continue)

---

## 执行摘要

❌ **AM 压缩在 Lazy Compression 下完全失败**

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| 质量一致性 | ≥ 90% | **0-6%** | ❌ 完全失败 |
| 输出可读性 | 正常文本 | **乱码** (星号、换行符) | ❌ 崩溃 |
| 压缩比控制 | 可配置 (1.5x-3.0x) | **固定 2.26x** | ❌ 不可控 |
| 架构稳定性 | 无崩溃 | ✅ 无崩溃 | ✅ 成功 |

**结论**: Heterogeneous cache 架构实现成功（无崩溃，无 NaN），但 AM 压缩算法不适用于 Qwen3-8B 的 Lazy Compression 场景。

---

## 测试结果详情

### 测试 1: 完整 36 层压缩

```bash
python /tmp/correct_lazy_compression_test.py
```

**结果**：
- 压缩比: 2.34x (18000 → 7692 tokens)
- 压缩时长: 315 ms
- 内存节省: 124 MB
- **质量一致性: 0%**
- 输出: 完全乱码（\`'.\n. . . . . . . . . . . . ... . . . . . . . . . . . . . . ...'\`）

### 测试 2: 部分层压缩

```bash
python /tmp/test_partial_layer_compression.py
```

| 策略 | 压缩比 | 质量 |
|------|--------|------|
| 压缩所有 36 层 | 2.20x | 0-3% |
| 只压缩前 18 层 | 2.20x | 0% |
| 只压缩前 12 层 | 1.98x | 3% |
| 只压缩前 6 层 | 1.60x | 7% |

**结论**: 减少压缩层数无法改善质量。

### 测试 3: 不同序列长度

```bash
python /tmp/test_calibration_length.py
```

| 序列长度 | 压缩层数 | 质量 |
|----------|----------|------|
| 350 tokens | 36/36 | 0% |
| 400 tokens | 36/36 | 0% |
| 450 tokens | 36/36 | 0% |
| 512 tokens | 36/36 | 0% |

**结论**: 序列长度不影响质量损失。

### 测试 4: Layer 0 单层压缩

```bash
python /tmp/test_layer0_only_lazy.py
```

**结果**：
- 压缩: Layer 0 only (550 → 256 tokens, 2.15x)
- **质量: 0%**
- 输出仍然是乱码

**结论**: 即使最小压缩也失败。

### 测试 5: 不同压缩比

```bash
python /tmp/test_compression_ratio_fixed.py
```

| 配置压缩比 | 实际压缩比 | 质量 | 输出 |
|-----------|-----------|------|------|
| 1.5x | 2.26x | 6% | 乱码 A |
| 2.0x | 2.26x | 6% | 乱码 A (相同) |
| 3.0x | 2.26x | 6% | 乱码 A (相同) |

**结论**: `compression_ratio` 参数被忽略，所有配置产生相同结果。

---

## 根因分析

详见 `critical-finding-am-incompatibility.md`

**简要总结**：

1. **Beta 零值崩溃**
   - Layer 27-35 的 beta 中有 0 值
   - log(0 + 1e-10) = -23.026
   - Attention 权重崩溃 → 输出乱码

2. **Selected_indices 不适配**
   - 校准基于前 512 tokens，indices 范围 [0-313]
   - Lazy Compression cache 有 530+ tokens
   - 丢弃最近生成的 tokens [314-530]
   - 模型失去短期记忆 → 输出错位

3. **离线校准 vs Lazy Compression 不兼容**
   - AM 假设固定 cache 大小，一次性压缩
   - Lazy Compression 需要动态 cache 大小，按需压缩
   - 本质矛盾，无法通过参数调整修复

---

## 验证实验

### 实验 A: 禁用 Beta 补偿

```bash
python /tmp/test_without_beta.py
```

**结果**：
- 输出不再是乱码
- 但输出有偏移：\`' Paris. The capital of...\` (缺少 \`' France\`)
- 质量: 0% (但可读)

**结论**: Beta 零值导致乱码，但 selected_indices 不匹配仍导致质量损失。

### 实验 B: Selected_indices 分析

```bash
python /tmp/check_selected_indices.py
```

**发现**：
- Indices 范围: [0, 313]
- 分布: 100% 集中在前 60% positions
- Lazy Compression cache: 530 tokens
- **完全丢弃 positions [314-530]**，包括所有新生成的 30 tokens

**结论**: Fixed indices 不适配动态 cache 大小。

---

## 架构验证

✅ **Heterogeneous cache 架构成功**：

1. **无崩溃**: 所有测试稳定运行，无段错误
2. **无 NaN**: Compressed K/V 无 NaN 值
3. **正确形状**: Layer 0-26 budget=256, Layer 27-35 budget=159
4. **压缩执行**: 成功压缩 18828 → 8343 tokens (2.26x)

**但压缩质量完全失败**，说明问题在 AM 算法本身，不在架构实现。

---

## 对比：期望 vs 实际

| 阶段 | 期望 | 实际 |
|------|------|------|
| **Prefill** | 正常推理 | ✅ 正常 |
| **Generate (压缩前)** | 正常生成文本 | ✅ 正常 |
| **Compress** | 无崩溃，质量损失 < 10% | ✅ 无崩溃 / ❌ 质量损失 100% |
| **Continue Generate** | 输出稍有降质但可用 | ❌ 完全乱码/严重错位 |

---

## 结论与建议

### 核心发现

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   AM 压缩算法在 Qwen3-8B Lazy Compression 下完全失效        │
│                                                             │
│   问题 1: Beta 零值 → Attention 崩溃 → 乱码                 │
│   问题 2: Fixed indices → 丢弃新生成 tokens → 错位          │
│   根本矛盾: 离线校准 ≠ Lazy Compression                     │
│                                                             │
│   ✅ Heterogeneous cache 架构正确                           │
│   ❌ AM 算法不适用此场景                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 建议

1. **短期**: 放弃 AM，使用 H2O 或 StreamingLLM
   - H2O: 基于 attention score 动态选择 KV pairs
   - StreamingLLM: 保留最近 N + 初始 M tokens
   - 两者都天然适配 Lazy Compression

2. **中期**: 修复 AM 以适配 Lazy Compression
   - 重新校准，确保 beta >= 0.1
   - 实现动态 selected_indices (根据 cache 大小调整)
   - 在 Lazy Compression 场景下重新评估

3. **长期**: 设计新的压缩算法
   - 结合 AM 的精确性和 H2O 的动态性
   - On-policy 校准 + 动态 indices
   - 专为 Lazy Compression 优化

---

## 附录：测试环境

- **模型**: Qwen3-8B (36 层 Attention)
- **设备**: M4 Pro / Metal
- **Framework**: MLX 0.x
- **校准文件**: \`am_calibration_qwen3-8b_2.0x_onpolicy.pkl\` (512 tokens)
- **测试 Prompt**: "The capital of France is Paris. " × 70 (500 tokens)

---

**生成于**: 2026-03-25
**作者**: Solar (Claude Opus 4.6)
**监护人**: 昊哥
