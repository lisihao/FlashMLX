# KV Cache Compaction 专家会审报告

**日期**: 2026-03-21
**任务**: Task #61 - 实现 Transformer KV Cache Compaction
**专家组**: 审判官 (deepseek-r1) + 探索派 (gemini-3-pro) + 稳健派 (gemini-2.5-pro)

---

## 执行总结

**会审结论**: 采用**两步走战略**
- ✅ **Phase 1 (2 周)**: StreamingLLM 快速交付（稳健派推荐）
- 🚀 **Phase 2 (探索)**: UMA-Dynamic Paging 研究（探索派创新，符合监护人战略）

**关键决策依据**:
1. StreamingLLM 是唯一能在 2-3 周内完成的生产级方案
2. UMA-Dynamic Paging 与已批准的 Hybrid Memory Manager v3 (Task #63) 完全一致
3. 需要微基准测试验证 UMA 特性后再决定 Phase 2

---

## 专家 1: 审判官 (deepseek-r1) - 深度推理

### 核心观点

**最可疑方法**: H2O (Heavy-Hitter Oracle)
- **质疑**: Attention Score ≠ 重要性（概念混淆）
- **风险**: 可能错误驱逐关键信息（如否定词、条件状语）
- **失败模式**: 破坏长程逻辑依赖，导致灾难性生成错误

**最可能成功**: SnapKV
- **理由**: 建模注意力行为本身，理论最严谨
- **前提**: 注意力头满足"局部平稳性"假设

### 关键未知问题

> "在 Attention+SSM 混合架构中，SSM 状态与 Attention KV Cache 的信息冗余度和互补性如何？这是所有三种方法都面临的共同但未被探索的核心问题。"

### 三种方法分析

#### H2O
- **核心假设**: Attention score 能准确反映 KV pair 的重要性
- **质疑**: 高 attention score 可能只是当前关注度高，而非信息重要
- **隐藏成本**:
  - 计算：实时追踪 attention scores
  - 实现：侵入式修改 Attention 核心
  - 调参：阈值高度依赖任务，泛化能力存疑

#### StreamingLLM
- **核心假设**: 初始 tokens 充当 "Attention Sink"，窗口外信息可丢弃
- **质疑**:
  - 依赖模型架构和训练分布，泛化风险高
  - 硬性边界假设，超窗口信息一律丢弃
- **隐藏成本**: 极低（仅维护滑动窗口指针）

#### SnapKV
- **核心假设**: 注意力模式平稳可预测，可通过线性聚合压缩
- **质疑**:
  - 平稳性假设在场景切换频繁时可能被打破
  - 线性聚合能否捕获非线性的注意力交互？
- **隐藏成本**: 极高（周期性前向传播生成快照）

### 综合判断

**需要验证的关键假设**:
1. Qwen3.5-35B 的注意力头是否满足"局部平稳性"？
2. SSM 状态与 Attention KV Cache 的信息冗余度和互补性？

---

## 专家 2: 探索派 (gemini-3-pro) - 创新突破

### 颠覆性洞察

> **"Eviction 可能是伪命题"** - 在 Mac UMA 极高带宽下，传统"驱逐"策略是对硬件优势的浪费。我们不需要减少 Token 数量，只需改变内存驻留层级。

### 基准方法的共同局限

1. **永久性信息丢失 (Hard Eviction)**: H2O 和 SnapKV 直接丢弃"不重要" token，无法恢复
2. **硬件不可知 (Hardware-Agnostic)**: 假设 PCIe 带宽瓶颈，未利用 Mac UMA 零拷贝优势
3. **架构孤立 (Architecture Isolation)**: 仅优化 Attention，忽略 SSM 已压缩历史信息

### 创新方案

#### 方案 1: UMA-Native 动态预取缓存 (UMA-Dynamic Paging)

**核心思路**:
- GPU 显存 (Hot Tier): Sink Tokens + 最近 Window
- CPU 内存 (Warm Tier): 被挤出的 KV Cache（不丢弃）
- 利用当前 Query 异步预取 CPU 中最相关的 KV 块回 GPU

**与基准方法的区别**: 零信息丢失，基于 UMA 的内存调度

**潜在优势**:
- 近乎无限 Context Length，100% 理论精度
- 512KB 传输仅需 2.5 μs，可被 10-30ms 推理计算掩盖

**实现难点**: 需要轻量级路由网络预测"哪些 CPU 块会被 Query 命中"

**置信度**: 0.9 (High)

#### 方案 2: SSM-Guided 交叉层语义融合 (SSM-Merge)

**核心思路**:
- 提取 SSM 状态转移矩阵，识别已被遗忘的 token
- 在 Attention 层对这些 token 的 KV 向量执行**融合**（而非丢弃）

**与基准方法的区别**:
- 无需 H2O 的昂贵 Attention Score 计算
- 用"融合"代替"丢弃"，保留背景语义

**置信度**: 0.6 (Med)

#### 方案 3: Semantic KV Merging

**核心思路**: 结合 SnapKV + Token Merging，对非关键 Token 加权平均融合

**潜在风险**: 可能破坏 RoPE 相位连续性

**置信度**: 0.7 (Med)

### 混合策略

**三段式缓存**:
- 头部：4 个 Sink Tokens (StreamingLLM)
- 尾部：局部滑动窗口 (StreamingLLM)
- 中间：H2O 保留 Top-10% Heavy-Hitters，其余降级到 UMA Warm Tier

### 惊人发现

> **混合架构的天然互补性**：SSM 本质上是无限窗口的、有损的 KV Compactor。Attention 层根本不需要记忆长程背景，只需作为高频信号的"放大器"。Attention KV Cache 可以被激进压缩（仅保留 5% Heavy-Hitters）。

### 快速验证方法

1. **UMA 性能极限测试**: MLX 微基准，注入 512KB-2MB CPU 拉取，测量对 TBT 影响
2. **SSM-Guided 相关性验证**: 记录 SSM 隐状态范数变化率 vs Attention H2O 分数，计算相关性

---

## 专家 3: 稳健派 (gemini-2.5-pro) - 工程评估

### 实现复杂度对比

| 方法 | 复杂度 (1-10) | 代码行数 | 改动模块 | 测试难度 |
|------|---------------|----------|----------|----------|
| **StreamingLLM** | 3 | 50-100 | KV Cache 管理 | 中 |
| **H2O** | 8 | 200-400 | Attention 计算 + KV Cache | 高 |
| **SnapKV** | 9 | 300-500+ | KV Cache + 压缩模块 | 极高 |

**评估依据**:
- StreamingLLM: 本质是索引和切片，不触及核心 Attention，风险可控
- H2O: 需要提取 attention scores，可能无法使用 MLX fast 内核，性能大幅下降
- SnapKV: 引入全新压缩子系统，数值稳定性和性能都需验证

### 质量风险分析

#### SSM 压缩失败教训的适用性

> "SSM 压缩失败的教训（优化 reconstruction loss ≠ 保持 behavior）**完全适用于**此任务。任何压缩方案都必须以端到端的行为作为最终衡量标准。"

#### KV Cache 压缩的特殊风险

1. **错误累积 (Cascading Errors)**: 自回归模型中，t 时刻的错误会传播到 t+1
2. **关键信息丢失 (Key Information Loss)**:
   - StreamingLLM: 风险明确（窗口外信息必然丢失）
   - H2O: 风险隐蔽（可能错误驱逐重要 token）
3. **性能退化 (Latency)**:
   - H2O: 计算 importance score 增加开销
   - SnapKV: 压缩/解压直接增加延迟

### 测试场景设计

**场景 1: 长上下文问答 (Needle in a Haystack)**
- 在 32K tokens 文档中隐藏独特事实，末尾提问
- 验收标准: 100% 准确回答
- **StreamingLLM 预期直接失败**

**场景 2: 长篇指令遵循 (Long Instruction Following)**
- 多步骤、有依赖、跨度长的复杂指令
- 验收标准: 完整正确执行所有步骤

**场景 3: 代码生成与补全**
- 依赖上下文多处定义的变量/函数
- 验收标准: 生成代码语法和逻辑正确，通过单元测试

**场景 4: 基准困惑度 (Perplexity)**
- 标准数据集 (PG-19)
- 验收标准: PPL 上升 < 5%

**总体验收标准**: PPL < 5% **且** 特定任务指标下降 < 5%

### 时间成本估计

**假设**: 1 名熟悉 MLX 的资深工程师

| 阶段 | StreamingLLM | H2O | SnapKV |
|------|--------------|-----|--------|
| 研究 | 1 天 | 3 天 | 4 天 |
| 实现 | 2 天 | 7+ 天 | 10+ 天 |
| 测试 | 5 天 | 5 天 | 5 天 |
| **总计** | **8 天** | **15+ 天** | **19+ 天** |

**结论**:
- ✅ StreamingLLM 是唯一能在 2-3 周完成的方案
- ⚠️ H2O 实现阶段高度不确定
- ❌ SnapKV 几乎不可能在规定时间内完成

### 推荐方案

**第一选择: StreamingLLM**

**理由**:
1. 简单可控：实现复杂度最低，不触及核心计算
2. 行为可预测：缺陷明确（窗口外信息丢失），可预测适用场景
3. 快速交付：唯一能在 2-3 周完成并充分测试的方案

**风险**: 对长距离依赖任务完全无效

**缓解措施**:
- 明确标注局限性，仅在适合场景开启
- Attention sink 大小可配置，允许内存-性能权衡

**备选方案**: 无（当前阶段不推荐其他方案）

### Go/No-Go 决策

**Go 标准**:
1. 前置验证通过（见下方清单）
2. PoC 中额外延迟 < 5%
3. 有效上下文长度提升 ≥ 50%（如 8K → 12K+）

**No-Go 标准**:
1. 前置验证失败（MLX KV Cache 结构需破坏性大改）
2. 收益不明显（内存节省无法满足目标硬件需求）

**前置验证清单 (必须在启动前完成)**:
1. **[1天] 基线性能复现**: 无压缩的可重复性能/质量基线
2. **[1天] MLX KV Cache API 审查**: 确认支持高效张量切片和拼接
3. **[2天] 最小可行原型 (MVP)**: 验证"大海捞针"失败模式 + 短上下文一致性

---

## Solar 战略家综合判断

### 专家意见审计（基于全程参与）

#### ✅ 稳健派 - **最靠谱**
- 工程评估**完全可信**（时间、复杂度、风险）
- 测试场景设计**严谨**
- Go/No-Go 清单**防止重蹈覆辙**
- **过于保守**: 低估了 UMA-Dynamic Paging 战略价值（没看到 v3 架构）

#### 🚀 探索派 - **战略洞察最强**
- **天才级洞察**: "Eviction 是伪命题" 完全符合监护人战略
- UMA-Dynamic Paging **与 Task #63 (v3 架构) 完全一致**
- 这不是"创新探索"，而是**已批准的战略主线**
- SSM-Guided Eviction 需要验证（置信度 0.6）

#### 🔍 审判官 - **质疑能力强，但方案偏理论**
- H2O 质疑**完全正确**（假设脆弱）
- 混合架构未知问题**是关键盲区**
- **不同意 SnapKV 推荐**:
  - 忽略了 SSM 压缩教训（reconstruction ≠ behavior）
  - SnapKV 假设"平稳性"，但 SSM 压缩暴露了**非平稳性**（场景跷跷板）
  - 时间不允许（19+ 天）

### 关键分歧

**专家们不知道的信息**:
1. ✅ 监护人已批准 Hybrid Memory Manager v3（Task #63）
2. ✅ v3 = Hot/Warm/Cold 三层 = UMA-Dynamic Paging
3. ✅ SSM 压缩失败教训：结构性冲突（Chinese rank~48 vs Think rank~96）
4. ✅ Mac UMA 18GB CPU 闲置（硬件事实）

**因此**:
- 探索派的 UMA-Dynamic Paging **不是探索，是战略主线**
- 稳健派低估了它（没看到 v3 架构）
- 审判官高估了 SnapKV（没经历 SSM 压缩失败）

### 最终执行计划

#### Phase 1 (2 周): StreamingLLM 实现
```
Day 1-2: 前置验证（基线性能 + MLX API 审查 + MVP）
Day 3-5: 实现 StreamingLLM（KV Cache 窗口滑动 + Sink）
Day 6-8: 测试四场景 + 质量验收
```

**关键约束**（专家们未提到）:
- **必须测试四场景**: Chinese / Think Tag / Format / Mixed
- 不能只测 PPL，要防止"平均提升，特定场景崩溃"（SSM 教训）

#### Phase 1 并行 (2 天): UMA 微基准测试
```
分配: 小快手 (glm-4-flash)
任务: 验证 CPU/GPU 512KB 传输对 TBT 影响 < 5%
```

**为什么必须做**:
- UMA-Dynamic Paging = Task #63 核心技术
- 这不是"备选"，是**战略主线的可行性验证**

#### Phase 2 决策点 (Day 10):
- ✅ **微基准通过** → 立即启动 UMA-Dynamic Paging（Task #63）
- ❌ **微基准失败** → StreamingLLM 生产化 + 重新评估 v3

### 不推荐方案

**H2O** - 三位专家 + 战略家一致反对
- ❌ 假设脆弱、实现复杂、可能重蹈 SSM 覆辙

**SnapKV** - 不同意审判官推荐
- ❌ 19+ 天（时间不允许）
- ❌ 假设"平稳性"，但 SSM 压缩暴露了场景跷跷板
- ❌ 线性聚合能否处理高频/低频冲突？未知

---

## 关键洞察（写入 Cortex）

### 洞察 1: 混合架构的天然互补性

```
SSM = 无限窗口的有损 KV Compactor（负责长程背景）
Attention = 高频信号放大器（负责局部精确控制）

→ Attention KV Cache 可以被激进压缩（仅保留 5% Heavy-Hitters）
→ SSM 状态已承担长程记忆职责
```

**证据**: SSM 压缩研究发现 Chinese 场景依赖低频语义（rank~48），Think/Format 依赖高频控制（rank~96）

### 洞察 2: Mac UMA 改变压缩范式

```
传统思路: 减少 Token 数量（Eviction）
UMA 新思路: 改变内存驻留层级（Paging）

GPU Hot (10GB) → CPU Warm (15GB) → SSD Cold (100GB)
传输成本: 2.5 μs/512KB（可被 10-30ms 推理计算掩盖）
```

**硬件事实**: Mac M4 Pro CPU 闲置 18GB，CPU/GPU 共享物理内存

### 洞察 3: SSM 压缩失败的通用教训

```
优化 Reconstruction Loss ≠ 保持 Behavior Preservation

→ 任何压缩方案必须以端到端行为作为最终衡量标准
→ 不能只看重建误差或 PPL
→ 必须测试多场景（防止平均提升、特定场景崩溃）
```

**教训来源**: Option C (rank 48) Chinese 10% 优秀，但 Option C+ (rank 96) Chinese 94% 灾难

---

## 附录：专家原始输出

### 审判官完整分析
[见 AUDIT FLAGS 部分的原始输出]

### 探索派完整分析
[见 HYPOTHESES 和 EXPLORATION RESULTS 部分]

### 稳健派完整分析
[见工程评估各部分]

---

**报告生成时间**: 2026-03-21
**下一步**: 执行 Phase 1 (StreamingLLM) + 并行 UMA 微基准测试
**决策点**: Day 10 根据微基准结果决定 Phase 2
