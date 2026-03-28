# Dynamic Recent Window 质量测试 - 失败报告

**日期**: 2026-03-26 15:33
**测试人员**: Solar (Claude Opus 4.6)
**监护人**: 昊哥

---

## 执行摘要

❌ **测试失败：减小 Recent Window 导致严重质量降低**

| Window Size | 内存节省 | 质量判定 | 关键问题 |
|------------|---------|---------|---------|
| 512 (baseline) | 0% | ✅ 正常 | - |
| 256 (optimized) | +12.9% | ⚠️  降质 | 重复、不连贯 |
| 128 (aggressive) | +19.4% | ❌ 崩溃 | 乱码 (只输出数字) |

---

## 测试配置

**模型**: Qwen3-8B (36 layers)
**Workload**: 3384 tokens agent debugging session
**Compression**: Uniform R1.5 (all layers)
**Generate**: 100 tokens
**Window sizes**: 512, 256, 128

---

## 详细结果

### Window=512 (Baseline)

**内存使用**: 278.0 MB
**质量指标**:
- Repetition: 0.00% ✅
- Coherence: 62.50% ✅
- Relevance: 42.86% ✅

**输出** (First 200 chars):
```
The root cause of the 500 errors was the application's direct database connection leaks from the background worker, scheduler, and batch request handler, and the database connection pool size was too
```

**质量判定**: ✅ **正常** - 连贯、相关、准确回答问题

---

### Window=256 (Optimized)

**内存使用**: 242.0 MB
**内存节省**: +12.9% (vs baseline)
**质量指标**:
- Repetition: 33.33% ⚠️  (+33.3 百分点)
- Coherence: 37.50% ⚠️  (-25 百分点)
- Relevance: 28.57% ⚠️  (-14.3 百分点)

**输出** (First 200 chars):
```
The root cause was direct database connections in the API handlers and background processes were creating connections outside the connection pool
The root cause was direct database connections in the
```

**质量判定**: ⚠️  **明显降质** - 出现重复，开始不连贯

**问题**:
- 同一句子重复出现
- 内容开始循环
- 虽然有节省内存，但质量下降不可接受

---

### Window=128 (Aggressive)

**内存使用**: 224.0 MB
**内存节省**: +19.4% (vs baseline)
**质量指标**:
- Repetition: 0.00% (无意义，因为全是数字)
- Coherence: 25.00% ❌ (-37.5 百分点)
- Relevance: 0.00% ❌ (-42.9 百分点)

**输出** (First 200 chars):
```
The root cause of the 5000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

**质量判定**: ❌ **完全崩溃** - 只输出数字，完全乱码

**问题**:
- 输出完全无意义
- 只生成重复的数字 "0"
- 质量完全崩溃

---

## 根因分析

### 为什么减小 Recent Window 导致质量降低？

**架构回顾**:
```
[Old Prefix (AM compressed)] + [Recent Window (exact)]
```

- **Old Prefix**: 3384 - window_size 个 tokens，被 AM 压缩
- **Recent Window**: window_size 个 tokens，保持精确

**问题机制**:

#### 1. Window=512 → 256: 重复问题

```
Before:
  Old Prefix: 2872 tokens (compressed)
  Recent Window: 512 tokens (exact)

After:
  Old Prefix: 3128 tokens (compressed) ← +256 tokens 被压缩
  Recent Window: 256 tokens (exact) ← 精确上下文减少

问题:
- 更多 tokens 进入 Old Prefix
- AM 压缩误差累积
- 生成时只有 256 tokens 精确上下文
- 缺少足够的最近上下文导致重复
```

#### 2. Window=128: 完全崩溃

```
Old Prefix: 3256 tokens (compressed) ← 大量 tokens 被压缩
Recent Window: 128 tokens (exact) ← 精确上下文严重不足

问题:
- 压缩误差 + 精确上下文不足
- 模型失去最近生成的语义
- 输出完全崩溃（乱码）
```

### 核心矛盾

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Recent Window 的两面性：                               │
│                                                         │
│  从内存角度: 是"边角料"（只占 15%）                      │
│  从质量角度: 是"保护线"（保证生成质量）                  │
│                                                         │
│  减小 Recent Window:                                    │
│    ✅ 确实节省内存 (+12-19%)                            │
│    ❌ 但破坏质量 (重复/乱码)                            │
│                                                         │
│  → Recent Window 不能砍！                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 对比之前的假设

### 原假设 (用户建议)

```
✅ Recent Window 占 15% 内存，是低悬的果实
✅ 减半可以节省 ~7-15%
✅ 动态调整 (QA:128, Chat:256, Code:384)
```

### 实际结果

```
❌ Recent Window 虽然占比小，但对质量至关重要
❌ 减半确实节省内存，但质量下降
❌ 即使 256 也不行，512 是最低安全线
```

---

## 关键教训

### 教训 1: 内存占比 ≠ 重要性

```
Recent Window:
  内存占比: 15% (看似边角料)
  质量影响: 100% (质量保证线)

结论:
  占比小的部分可能是系统的关键约束
  不能只看数字占比，要看功能作用
```

### 教训 2: 工程级优化也需要验证

```
方案看起来合理:
  ✅ 逻辑清晰 (减小不压缩部分)
  ✅ 收益明确 (+12-19%)
  ✅ 实现简单 (改配置)

但实际:
  ❌ 质量崩溃 (重复/乱码)
  ❌ 不可接受

结论:
  即使是"显而易见"的优化也必须实测
  质量 > 内存节省
```

### 教训 3: DoubleLayerKVCache 的设计是有原因的

```
Recent Window = 512 不是随便设的

这个值是平衡的结果:
  ✓ 足够大：保证生成质量
  ✓ 足够小：不浪费内存

减小它会破坏这个平衡
```

---

## 对 Layerwise 压缩的启示

### 新的担忧

如果减小 Recent Window (15% 的内存) 都导致质量崩溃，那么:

```
Layerwise 压缩深层 (可能 50% 的 Old Prefix 不能压):
  风险可能更大！

原因:
  - Recent Window 是"精确保留"，只是范围变小
  - Layerwise 是"压缩"，引入误差
  - 压缩误差 + 范围减小 = 双重打击？
```

### 重新评估期望

```
之前预期:
  Layerwise 可能节省 40-60%

现在担心:
  如果后层不能压（类似 Recent Window 不能减）
  实际节省可能 < 30%
  甚至可能质量崩溃
```

---

## 下一步建议

### 方案 A: 等待 Layerwise 验证 (Task #24-27)

```
优先级: 高
时间: 70 分钟后开始

原因:
- Recent Window 方案失败
- Layerwise 是剩下的主要优化方向
- 必须验证是否也有质量问题
```

### 方案 B: 如果 Layerwise 也失败

```
切换到 Direction 4 (并发吞吐量):
- 不追求单请求内存节省
- 转而优化并发处理能力
- 可能更实际
```

### 方案 C: 研究其他压缩算法

```
H2O:
  - 动态压缩，基于 attention score
  - 可能更稳定

StreamingLLM:
  - 保留最近 + 初始 tokens
  - 质量更可控
```

---

## 结论

### 核心发现

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Recent Window = 512 是质量保证线                       │
│  不能减小，否则质量崩溃                                  │
│                                                         │
│  内存占比小 ≠ 可以牺牲                                   │
│  关键约束往往不是最大的部分                              │
│                                                         │
│  → 优化必须实测，不能靠推理                              │
│  → 质量 > 内存节省                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 对用户方案的回应

```
您的方案逻辑完全正确:
  ✅ Recent Window 确实占 15%
  ✅ 减半确实节省 12.9%
  ✅ 实现确实简单

但实测结果:
  ❌ 质量下降不可接受
  ❌ 即使 256 也有重复问题
  ❌ 128 完全崩溃

结论:
  工程级优化也需要实证验证
  这次实验很有价值 - 排除了一个方向
```

---

**生成于**: 2026-03-26 15:40
**作者**: Solar (Claude Opus 4.6)
**监护人**: 昊哥

**测试文件**:
- Log: `/tmp/window_sizes_quality_test.log`
- Outputs: `/tmp/window_{512,256,128}_output.txt`
