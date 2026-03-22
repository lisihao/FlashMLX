# SSM State 压缩研究：教训总结

**日期**: 2026-03-21
**投入**: ~5 天研究时间
**结果**: No-Go - 放弃 SSM live-state 全局 low-rank 压缩方向

---

## 核心教训

### ❌ 错误 1: 优化了错误的轴

**问题**:
- 我优化的是：state reconstruction（状态重建精度）
- 真正重要的是：behavior preservation（行为保持）

**教训**:
> 压缩算法的目标不是"最小化重建误差"，而是"保持模型行为"

**证据**:
- Option C+ (rank=96) 比 Option C (rank=48) 重建误差更小
- 但 Chinese 场景行为从 10% 差异率跌到 94%（灾难性退化）

---

### ❌ 错误 2: 把结构性冲突当作调参问题

**问题**:
- 不同场景对 rank 需求冲突：Chinese (rank~48) vs Think/Format/Mixed (rank~96)
- 我以为"找一个折中的 rank"就能解决

**真相**:
> 这不是连续优化问题（可以找最优点），而是结构性冲突（无解）

**教训**:
- 当看到"质量跷跷板"现象时，应该立即停止调参
- 改变问题框架，而不是继续暴力搜索

---

### ❌ 错误 3: 过度依赖单一技术（SVD low-rank）

**问题**:
- 花了 5 天时间全在调 SVD low-rank 的 rank 参数
- Option A → Option C → Option C+ → Option C++ 都是同一个框架

**教训**:
> 当一个方法尝试 2-3 次都失败时，应该换方法，而不是继续微调

**专家建议**:
- 审判官：缺乏实测数据支撑，全是推测
- 稳健派：所有方案都达不到生产标准，应立即 No-Go
- 探索派：提出 5 个替代方案（Delta-State, Token Router, Top-K, DCT, 混合量化）

---

### ✅ 正确的做法：监护人的战略转向

**建议**:
```
别再问"哪个 rank 最好"
改问"哪些记忆该压、什么时候压、压成什么层级、
     哪些状态绝不能进热路径压缩"
```

**新主线**:
1. 把 SSM live-state 压缩从主战场降级
2. 主战场切到：Transformer KV Cache Compaction
3. SSM 部分只做保守的分层内存管理
4. 做成 Runtime Memory Manager，不是单一算法

---

## 研究成果（虽然失败，但有价值）

### 发现 1: 质量"跷跷板"现象

**现象**: 增加 rank 改善控制场景，但破坏 Chinese 场景

| 场景 | Option C (rank 48) | Option C+ (rank 96) |
|------|-------------------|---------------------|
| Chinese | 10% ✅ | 94% ❌ |
| Think Tag | 94% ❌ | 54% ✅ |
| Mixed | 84% ❌ | 0% ✅ |

**解释**:
- Chinese 依赖低频语义（rank~48 足够）
- Think/Format 依赖高频控制信号（需要 rank~96）
- 高 rank 引入的高频分量对 Chinese 是噪声，对控制场景是信号

---

### 发现 2: Critical Channels 分离失败的根因

**方法**: 保留 6 个"关键通道"全精度，其他 122 个通道低秩压缩

**结果**: 89% 平均差异率（惨败）

**根因**:
- Critical channels profiling 基于单通道扰动
- 无法捕捉多通道协同（SSM state 是整体，不可分割）
- 破坏了状态的整体结构

**教训**:
> SSM state 是高度耦合的系统，不能简单地分为"关键"和"非关键"

---

### 发现 3: 低秩近似可以工作（在特定场景）

**证据**: Option C 在 Chinese 场景达到 10% 差异率（优秀）

**说明**:
- Low-rank approximation 本身**不是错的**
- 问题是"不同场景需要不同 rank"（无法统一）

**启示**:
- 可以在**非热路径**（Warm tier）使用 low-rank 存储
- 但不应该在**热路径**（Hot tier）全局应用

---

## 转向新主线的理由

### 理由 1: Transformer 更适合压缩

**为什么**:
- KV Cache 是独立的 key-value pairs
- 可以基于重要性选择性驱逐（eviction）
- 有成熟的方法：H2O, StreamingLLM, SnapKV

**对比**:
- SSM state 是高度耦合的递归状态
- 压缩任何部分都会影响整体
- 没有成熟的压缩方法

---

### 理由 2: 分层管理比全局压缩更稳健

**全局压缩问题**:
- 必须在热路径实时压缩/解压
- 任何质量损失都直接影响生成
- 难以预测不同场景的影响

**分层管理优势**:
- Hot tier 完全不压缩（质量无损）
- Warm/Cold tier 压缩不在热路径（容忍更大误差）
- 更符合"记忆分级"的认知模型

---

### 理由 3: Runtime Manager 是推理引擎的正确抽象

**单一算法的问题**:
- 只解决一个子问题（SSM state 压缩）
- 无法全局优化内存使用
- 难以适应不同场景和内存压力

**Runtime Manager 的优势**:
- 统一管理 Attention + SSM 的内存
- 动态分配资源（基于内存预算和重要性）
- 可扩展到其他优化（如 quantization）

---

## 数据留存

### 实验数据
- `.solar/compression-comparison-results.json` - Option A vs C 对比
- `.solar/option-c-plus-results.json` - Option C+ 测试结果

### 分析报告
- `.solar/compression-strategies-comparison-report.md` - 详细分析
- `.solar/option-c-plus-analysis.md` - 跷跷板现象分析
- `.solar/compression-research-summary-for-experts.md` - 专家会审材料

### 专家意见
- 审判官 (deepseek-r1): 根因分析，缺乏验证数据
- 稳健派 (gemini-2.5-pro): 工程评估，建议 No-Go
- 探索派 (gemini-3-pro): 5 个创新方案

---

## 未来可能的方向（如果必须压缩 SSM state）

基于探索派的建议，如果未来必须解决 SSM state 压缩问题，可以尝试：

1. **Delta-State 压缩** (置信度 0.8)
   - 只压缩增量 $\Delta h_t$，不压缩绝对状态
   - 避免在历史语义中注入噪声

2. **Token-Driven Router** (置信度 0.8)
   - 基于当前 token embedding 动态选择 rank
   - 无需显式场景分类

3. **Top-K 稀疏性** (置信度 0.7)
   - 保留最大值，零化其他（替代 SVD）
   - 保持锐利度，适合控制场景

4. **混合 Rank-量化** (置信度 0.6)
   - Top 48 奇异值用 FP16，Bottom 48 用 INT4
   - 既有容量又减少噪声

---

## 时间线

- 2026-03-15: 开始 Phase 1 (Critical Channels Profiling)
- 2026-03-17: 完成 30 层 profiling (3,840 次测试)
- 2026-03-18: Option A 测试失败 (89% 差异率)
- 2026-03-19: Option C 测试部分成功 (Chinese 10%, 其他失败)
- 2026-03-21: Option C+ 测试，发现跷跷板现象
- 2026-03-21: 三位专家会审，建议 No-Go
- 2026-03-21: 监护人战略决策，转向新主线

**总投入**: ~5 个工作日
**结果**: 明确了不可行的方向，避免了更大的浪费

---

## 引用监护人的原话

> "你现在这条路——Option C / C+ / C++ 这种全局 low-rank rank 调参——已经暴露出一个根本问题：你优化的是 state reconstruction 轴，但真正决定可用性的，是行为保持轴。"

> "比'继续调 SSM 全局 low-rank rank'更好的路，是换问题表述。别再问'哪个 rank 最好'，改问'哪些记忆该压、什么时候压、压成什么层级、哪些状态绝不能进热路径压缩'。这才是推理引擎路线。"

---

**总结**:
- ✅ 及时止损（5 天 vs 可能的 2-3 周）
- ✅ 明确了正确的主线（推理引擎路线）
- ✅ 保留了有价值的发现（跷跷板现象、分层管理思路）
- ✅ 专家会审验证了决策的正确性

**下一步**:
- Task #61: 实现 Transformer KV Cache Compaction
- Task #62: 实现 SSM 保守分层内存管理
- Task #63: 构建 Runtime Memory Manager
