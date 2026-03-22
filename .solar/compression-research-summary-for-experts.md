# SSM State 压缩研究总结（专家会审版）

## 背景

**目标**: 为 FlashMLX 的 State-Memory 机制设计压缩算法，降低内存占用同时保持生成质量

**SSM State 结构**:
- Shape: (B, Hv, Dv, Dk) 其中 Dv=128 个通道
- 每层 SSM state 大小: ~512KB (bfloat16)
- 39 层总计: ~20MB

**质量阈值**: Token 差异率 < 15% 视为可接受

---

## 研究历程

### Phase 1: Critical Channels Profiling (已完成)

**方法**: 单通道扰动测试，识别"关键通道"

**结果**:
- 完成了 30 层的 profiling（3,840 次测试）
- 每层识别出 6 个"关键通道"（5% ratio）
- 发现：后期层 (26-38) 重要性比早期层高 12.6%

---

## 三种压缩方案对比

### Option A: 混合压缩（Critical + Bulk）

**思路**:
```
Critical channels (6/128): 全精度保留
Bulk channels (122/128): 低秩近似 (rank=32)
```

**压缩比**: 28.52% 保留

**结果**: **失败** ❌
- 平均 Token 差异率: **89.00%**
- 所有场景都失败

**根因推测**:
- Critical channels profiling 基于单通道扰动，无法捕捉多通道协同
- Critical 和 Bulk 的二分法过于简化
- 重组后的 state 失去了原始的整体结构

---

### Option C: 分层压缩（全低秩）

**思路**:
```
早期层 (0-12):  rank=16
中期层 (13-25): rank=32
后期层 (26-38): rank=48
```

**压缩比**: 25.00% 保留

**结果**: **部分成功** ⚠️
- 平均 Token 差异率: **70.50%**
- Chinese 场景: **10.0%** ✅ (优秀！)
- Think Tag: 94.0% ❌
- Format: 94.0% ❌
- Mixed Language: 84.0% ❌

**关键发现**: Chinese 场景成功证明了低秩近似**可以工作**，但其他场景失败说明 rank 不够

---

### Option C+: 增加 Rank

**思路**: 将 Option C 的 rank 翻倍
```
早期层 (0-12):  rank=32  (vs C: 16)
中期层 (13-25): rank=64  (vs C: 32)
后期层 (26-38): rank=96  (vs C: 48)
```

**压缩比**: 50.00% 保留

**结果**: **质量"跷跷板"** 🎢
- 平均 Token 差异率: **56.50%** (比 Option C 好 14%)
- Chinese 场景: **94.0%** ❌ (从 10% 大幅退化！)
- Think Tag: **54.0%** ✅ (改善 40%)
- Format: **78.0%** ✅ (改善 16%)
- Mixed Language: **0.0%** ✅ (完美！)

**困惑点**: 为什么增加 rank 反而破坏了 Chinese 场景？

---

## 详细数据

### Chinese 场景输出对比

**Baseline** (无压缩):
```
My reasoning: 人工智能（Artificial Intelligence，简称 AI）的定义与内涵

1
```

**Option C (rank=48)** - 10% 差异率:
```
My reasoning: 人工智能（Artificial Intelligence，简称 AI）是一个涵盖范围很广
```
→ ✅ 中文生成流畅，语义连贯

**Option C+ (rank=96)** - 94% 差异率:
```
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Answer the question "What is Artificial Intelligence?" (什么是人工智能？).
    *   Language: Chinese (中文).
```
→ ❌ 切换到英文，语义完全偏离

---

### Mixed Language 场景输出对比

**Baseline**:
```
Answer (Chinese): 深度学习是
```

**Option C (rank=48)** - 84% 差异率:
```
Answer (Chinese): 深度学习是机器学习的一个子集，
```
→ ❌ 语言切换不完整

**Option C+ (rank=96)** - 0% 差异率:
```
Answer (Chinese): 深度学习是
```
→ ✅ 完全一致，语言切换完美

---

## 需要专家解答的问题

1. **为什么 Option C 的 rank=48 对 Chinese 场景效果很好，但 rank=96 反而破坏了？**
   - 是否因为高 rank 引入了噪声？
   - 还是因为过拟合了某些不重要的信息？

2. **为什么 Option C+ 对 Mixed Language 场景完美（0% 差异率）？**
   - 是否因为跨语言切换需要更高维度的状态表示？

3. **为什么控制场景（Think Tag, Format）需要更高的 rank？**
   - 是否因为控制信号编码在高频分量？

4. **是否存在"场景特定的最优 rank"？**
   - Chinese: rank~48
   - Mixed Language: rank~96
   - Think/Format: rank~96
   - 如果是，如何统一？

5. **下一步应该怎么做？**
   - Option C++ (rank: 24/48/72) - 折中方案？
   - 混合策略 (早期层低 rank，后期层高 rank)？
   - 场景自适应 rank（运行时检测场景并动态调整）？
   - 还是重新评估方向？

---

## SVD 低秩近似原理 (供参考)

```python
# SVD 分解
U, S, Vt = np.linalg.svd(state_matrix)

# 低秩近似
state_compressed = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]

# 奇异值分布假设 (未验证)
# 前 16 个: 低频语义（占 80% 能量）
# 17-48 个: 中频结构（占 15% 能量）
# 49-96 个: 高频细节 + 噪声（占 5% 能量）
```

**疑问**: 上述奇异值分布假设是否成立？如何验证？

---

## 测试设置

**模型**: Qwen3.5-35B-MLX (39 层混合架构)

**测试场景**:
1. Chinese: 纯中文生成
2. Think Tag: `<think>` 标签控制
3. Format: 格式化输出（列表）
4. Mixed Language: 英文+中文混合

**生成长度**: 50 tokens

**重复次数**: 每个配置测试 1 次

---

## 请专家分析

1. 解释"跷跷板"现象的根因
2. 评估三种方案的优劣
3. 提出下一步优化方向
4. 判断是否应该继续 SSM 压缩方向，还是转向其他方法

感谢！
