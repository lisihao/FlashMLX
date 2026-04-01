# Route 0: Density Router — 设计文档

> 基于论文 [2603.25926](https://arxiv.org/abs/2603.25926) 的核心洞察，结合 FlashMLX Route 5 (Context Recall) 构建的统一压缩决策框架。

**状态**: Phase 1-4 已实现 (基础设施 + 参数穿透 + 三模式 + Budget + ThunderOMLX)
**日期**: 2026-04-01
**评审**: 老专家 9/10 分

---

## 一、论文核心洞察 (2603.25926)

### 论文做了什么

论文标题: "Density-aware Soft Context Compression with Semi-Dynamic Compression Ratio"

这篇论文不是又一个 KV cache 压缩方法。它是 **embedding 级上下文压缩**：用一个 0.6B encoder (bidirectional attention) 处理全文，mean-pooling 压缩后送 decoder。

### 我们从中提取的三个可移植认知

| 论文洞察 | 原始实现 | 我们的移植方式 |
|---------|---------|-------------|
| **连续动态比率会崩，离散化才稳** | 5 个离散比率 {2x,4x,8x,16x,32x} | 把 AM adaptive compression 形式化为 5 个离散级别 |
| **信息密度差异极大，不能一刀切** | 训练 DRS (regression head) 预测密度 | 用 AM score 分布作为密度信号 (零成本) |
| **scale 旋钮统一调节激进度** | log2 空间加偏移 | 同样的 log2 空间偏移，一个参数控制 Pareto 曲线 |

### 关键决策：我们不需要论文的 0.6B encoder

论文需要一个单独的 encoder 做 bidirectional attention + regression head 来预测密度。

**我们已经有 AM scoring。** AM 打分本身就是密度信号：
- 高密度文本 → AM score 方差大、"必须保留" token 占比高
- 低密度文本 → AM score 普遍低，大部分 token 可牺牲

这把"前端语义压缩模型"的成本，降成了"已有 runtime 信号上的决策层"。

### 理论风险 (专家泼的冷水)

**AM score ≠ 语义密度。** 这两个高度相关，但不是同一个量：
- AM score 是 attention matching / runtime importance 信号
- 语义密度是面向压缩的信息量度量

可能失败的场景：重复性强但 attention 模式固定的文本 → AM 分数高但语义密度低。

**这个风险只能用实验自证，不能靠推理代替。** 见 Benchmark A 组。

---

## 二、架构设计

### Route 0 的定位

> **Route 0 不是新模型，是新决策层。**

```
                  ┌─────────────────────────────────┐
                  │  Route 0: Density Router (新)     │
                  │                                   │
                  │  输入: AM score 分布 (已有计算)      │
                  │  输出: 离散压缩级别 + scale 偏移     │
                  │  成本: 零 (复用 AM scoring)         │
                  └─────────────┬─────────────────────┘
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Route 2/3    │  │ Route 5      │  │ SSD Layer    │
    │ 运行时治理    │  │ Context      │  │ 密度感知      │
    │              │  │ Recall       │  │ 持久化        │
    │ chunked PP   │  │              │  │              │
    │ scored evict │  │ h^(0) 存档    │  │ KVTC/lz4     │
    │ flat Q8      │  │ 按需重建      │  │ 自适应选择    │
    │              │  │              │  │              │
    │ budget ← R0  │  │ 后盾 ← R0    │  │ 阈值 ← R0   │
    └──────────────┘  └──────────────┘  └──────────────┘
```

### Route 0 工作原理

**Step 1: 密度信号提取 (零成本)**

在 `_scored_compress_prefix` 已有的 AM scoring 之后，提取密度指标：

```python
# AM scores 已经算好了 (shape: [n_tokens])
am_scores = self._compute_am_scores(keys, values)

# 密度信号 = 高分 token 占比 + 分布形状
density_metrics = {
    "high_score_ratio": (am_scores > threshold).float().mean(),  # 0.0~1.0
    "score_variance": am_scores.var(),
    "tail_mass": (am_scores > p90).float().mean(),  # 尾部质量
}
```

**Step 2: 离散化 (论文核心洞察)**

```python
# 离散压缩级别 (论文证明: 5个级别足够)
COMPRESSION_LEVELS = {
    "keep_80": 1.25,  # 保留 80%，轻微压缩
    "keep_50": 2.0,   # 保留 50%，当前默认
    "keep_33": 3.0,   # 保留 33%，中等压缩
    "keep_20": 5.0,   # 保留 20%，激进压缩
    "keep_10": 10.0,  # 保留 10%，极端压缩 (Route 5 兜底)
}

# 密度 → log2 空间 → 加 scale → 量化到最近级别
log_ratio = density_to_log_ratio(density)  # 连续值
log_ratio_scaled = log_ratio + scale        # 用户旋钮
level = snap_to_nearest(log_ratio_scaled, COMPRESSION_LEVELS)
```

**Step 3: Scale 旋钮 (产品级控制)**

```python
# scale 在 log2 空间，每 +1 = 压缩翻倍
# scale = 0.0  → 密度自适应决定
# scale = +1.0 → 整体翻倍压缩（更激进）
# scale = -1.0 → 整体减半压缩（更保守）
```

---

## 三、Route 0 ↔ Route 5 耦合

### 核心关系

| 没有 Route 5 | 有 Route 5 |
|:---:|:---:|
| Route 0 只能保守压 | Route 0 可以大胆压 |
| 极端级别 (keep_10) 不敢用 | keep_10 变得可用 |
| 压错了无法补救 | 压错了还能回头看 |

**这不是两个压缩算法堆一起，而是：**
- 前端做密度感知压缩决策
- 后端做按需上下文召回

### Reconstruction Budget (专家关键建议)

**不是"有 Route 5 就可以随便压"，而是"Route 5 的摊销重建成本可接受时，才允许更激进压缩"。**

```python
class ReconstructionBudget:
    """控制 Route 0 可以多激进地压缩。

    防止系统过度依赖重建路径，导致：
    - tail latency 飙升
    - 用户感知"平时挺快，一问细节突然顿一下"
    """
    max_recall_per_turn: int = 1        # 每轮最多触发 1 次重建
    max_recon_tokens: int = 2048        # 单次重建上限 token 数
    max_recon_latency_ms: float = 500   # 重建延迟红线 (ms)
    cooldown_turns: int = 2             # 触发后冷却轮次

    def can_afford(self, estimated_recon_tokens: int) -> bool:
        """Route 0 决策时查询：是否还有预算允许激进压缩。"""
        if self.recalls_this_turn >= self.max_recall_per_turn:
            return False
        if estimated_recon_tokens > self.max_recon_tokens:
            return False
        if self.turns_since_last_recall < self.cooldown_turns:
            return False
        return True
```

Route 0 在决策时：

```python
if h0_store is not None and recon_budget.can_afford(estimated_cost):
    max_scale = 3.0   # 允许激进压缩
else:
    max_scale = 1.0   # 限制在保守范围
```

---

## 四、产品模式

### 三个模式

| 模式 | Route 0 scale | Route 5 触发策略 | 适用场景 |
|------|:---:|:---:|------|
| **Balanced** | 0.0 | 仅显式请求 | 日常对话、写代码、短文本 |
| **Ultra Long** | +1.5 | 回忆窗口自动 | 长文档、会议记录、多轮文档 QA |
| **Recall-first** | +2.5 | 频繁主动触发 | 先粗读后追问、RAG + 长前缀 |

### 模式定义的要求

**每个模式必须在自己的场景上有客观优势，不能只是三组拍脑袋参数。**

- Balanced: TG 最快，延迟最低
- Ultra Long: 内存最低，PP 最快
- Recall-first: 回答质量最高 (尤其是追问早期细节时)

这需要 Benchmark C 组验证。

---

## 五、ThunderOMLX 集成设计

### 1. 配置层 (settings_v2.py)

```python
class FlashMLXSettingsV2(BaseModel):
    # 现有字段...

    # Route 0: Density Router (新增)
    density_mode: str = "balanced"        # balanced | ultra_long | recall_first | manual
    density_scale: float = 0.0            # manual 模式下用户直接设 scale
    density_levels: int = 5               # 离散级别数
```

### 2. CacheConfig (config.py)

```python
class CacheConfig(BaseModel):
    # 现有字段...

    # Route 0
    density_mode: str = "off"             # off | balanced | ultra_long | recall_first
    density_scale: float = 0.0
```

### 3. Scheduler (scheduler.py)

```python
# 请求初始化时，根据 density_mode 设置 Route 0 + Route 5 联动
if flashmlx_settings.density_mode == "recall_first":
    cache_kwargs["density_scale"] = 2.5
    cache_kwargs["kv_cache"] = "scored_kv_direct"  # 自动启用 Route 5
elif flashmlx_settings.density_mode == "ultra_long":
    cache_kwargs["density_scale"] = 1.5
    cache_kwargs["kv_cache"] = "scored_kv_direct"
# balanced 模式不启用 Route 5，只用离散密度路由
```

### 4. Model Card 扩展

```json
{
  "optimal": {
    "strategy": "scored_pq",
    "density_mode": "balanced"
  },
  "modes": {
    "balanced": { "density_scale": 0.0, "strategy": "scored_pq" },
    "ultra_long": { "density_scale": 1.5, "strategy": "scored_kv_direct" },
    "recall_first": { "density_scale": 2.5, "strategy": "scored_kv_direct" }
  }
}
```

### 5. SSD 层密度感知持久化

Route 0 的密度信号可以回灌到 ThunderOMLX 的 SSD cache 层：

```python
# 当前: kvtc_threshold = 2048 (固定阈值)
# Route 0 增强: 用密度信号动态调整阈值
if density_metrics["high_score_ratio"] > 0.6:
    # 高密度内容，用 KVTC 精细压缩
    use_kvtc = True
else:
    # 低密度内容，lz4 快速压缩即可
    use_kvtc = False
```

---

## 六、实现路线图

### Phase 1: 离散化 AM 压缩比 (纯 FlashMLX)

**目标**: 把 `compression_ratio=0.0` (adaptive) 替换为离散级别系统

| 改动 | 文件 | 行数 |
|------|------|:---:|
| `DensityLevel` enum + snap 逻辑 | `config.py` | ~30 |
| `_scored_compress_prefix` 用离散级别 | `triple_layer_cache.py` | ~20 |
| 密度信号提取 (AM score 分布统计) | `triple_layer_cache.py` | ~15 |
| `density_scale` 参数穿透 | `config.py → cache.py → cache_factory.py` | ~10 |
| 单元测试 | `tests/` | ~40 |

**前置**: 无
**验证**: Benchmark A 组

### Phase 2: Scale 旋钮 + 产品模式

**目标**: 三个模式按钮，一键切换压缩激进度

| 改动 | 文件 | 行数 |
|------|------|:---:|
| `density_mode` 字段 + mode→scale 映射 | `config.py` | ~20 |
| Model Card `modes` 字段 | `model_cards.py` + JSON | ~30 |
| bench 脚本跑三模式对比 | `benchmarks/bench_density_modes.py` | ~120 |

**前置**: Phase 1
**验证**: Benchmark B + C 组

### Phase 3: Reconstruction Budget + Route 0↔5 联动

**目标**: 模式自动联动 Route 5，带预算控制

| 改动 | 文件 | 行数 |
|------|------|:---:|
| `ReconstructionBudget` 类 | `kv_direct_cache.py` | ~40 |
| 模式自动选择 scored_kv_direct | `cache_factory.py` | ~15 |
| 压缩日志 (哪个 chunk 压了多少) | `triple_layer_cache.py` | ~20 |
| Route 5 触发策略 (基于压缩日志 + budget) | `kv_direct_cache.py` | ~30 |
| Route 0 scale 上限联动 budget | `cache_factory.py` | ~10 |

**前置**: Phase 2 + Route 5 (已完成)
**验证**: Benchmark D 组

### Phase 4: ThunderOMLX 集成

**目标**: 通过 ThunderOMLX settings 一键使用三模式

| 改动 | 文件 | 行数 |
|------|------|:---:|
| `FlashMLXSettingsV2` 加 density_mode | `settings_v2.py` | ~10 |
| Scheduler 联动逻辑 | `scheduler.py` | ~15 |
| SSD 层密度感知 (Route 0 信号→KVTC 阈值) | `paged_ssd_cache.py` | ~20 |

**前置**: Phase 3
**验证**: 端到端集成测试

### Phase 5: 密度信号增强 (可选研究方向)

当 AM 分数不够准时，可以尝试：
- 用 ThunderOMLX 本地的 Qwen3-1.7B 计算 perplexity 作为辅助密度信号
- 这和论文的 encoder 思路一致，但不需要 bidirectional attention
- 只在 Phase 1 的 Benchmark A 组证明 AM 不够用时才启动

---

## 七、Benchmark 计划

### A 组：打论文主张 — 连续 vs 离散

**目的**: 证明离散化比连续 adaptive 更好或至少不弱

| 对比项 | 方法 | 测什么 |
|--------|------|--------|
| 当前 adaptive (continuous) | `compression_ratio=0.0` | PP/TG/Mem/Quality |
| 离散 5 档 + AM density | Route 0 实现 | 同上 |
| 离散 5 档 + scale sweep | scale={-1, 0, +1, +2} | Pareto 曲线是否平滑 |

**上下文**: 4K / 16K / 32K
**模型**: Qwen3-8B
**成功标准**:
- 离散 5 档在所有上下文长度上质量不退化
- scale 移动产生平滑的质量-压缩 tradeoff 曲线
- 不出现"scale +1 突然质量崩塌"的断崖

### B 组：打 FlashMLX 主线 — 逐级叠加

**目的**: 证明 Route 0 是在现有系统上真的抬高了

| 配置 | 策略 |
|------|------|
| Baseline | scored_pq Q8 (v2.0 官方基线) |
| +Route 0 | scored_pq Q8 + 离散密度路由 |
| +Route 0+5 | scored_kv_direct + 离散密度路由 + h^(0) |

**上下文**: 4K / 16K / 32K
**模型**: Qwen3-8B
**指标**: PP tok/s, TG tok/s, KV Mem, TTFT, Quality
**成功标准**:
- +Route 0 在内存上优于 Baseline (密度低的部分压更狠)
- +Route 0+5 在质量上不弱于 Baseline (即使压更狠)
- TTFT 不退化

### C 组：打产品模式 — 三模式各有冠军区间

**目的**: 证明三个模式不是拍脑袋，各自在对应场景有优势

| 模式 | 测试场景 | 预期冠军指标 |
|------|---------|------------|
| Balanced | 日常对话 (4K)、代码补全 | TG 最快，延迟最低 |
| Ultra Long | 32K+ 长文档 QA | 内存最低，PP 最快 |
| Recall-first | 多轮追问早期细节 | 回答质量最高 |

**成功标准**: 每个模式在至少一个场景上是三者中最优

### D 组：打回忆能力 — 用户价值

**目的**: 证明 Route 0 压得更狠 + Route 5 回忆 = 质量不掉

| 场景 | 测试方法 |
|------|---------|
| 长文档问前文细节 | 32K 文档，问第 2K 处的具体数字/名字 |
| 多轮跨轮回指 | 5 轮对话，第 5 轮问第 1 轮定义的概念 |
| RAG 先粗读后细问 | 8K 检索文档，先概括再追问具体段落 |

**对比矩阵**:
- standard (无压缩)
- scored_pq (当前基线)
- scored_pq + Route 0 aggressive (scale=+2)
- scored_pq + Route 0 aggressive + Route 5

**成功标准**:
- Route 0 aggressive 单独跑，质量会掉
- Route 0 aggressive + Route 5，质量恢复到 standard 水平
- 这就证明了"前端敢压，后端能救"的闭环

---

## 八、系统叙事

### 当前叙事 (v2.0)

> FlashMLX: KV cache 压缩 + 专家卸载

### 目标叙事 (v3.0)

> **FlashMLX / ThunderOMLX 正在从"压缩系统"升级成"上下文操作系统"。**
>
> - Route 0 负责**理解**上下文值得保留多少
> - Route 2/3 负责在运行时高效地**保留和忘记**
> - Route 5 负责在必要时把过去重新**想起来**
>
> 这不是"上下文太长怎么办"。
> 这是"上下文应该怎么被理解、怎么被保留、怎么被忘、怎么被想起"。

### 对外一句话

> **语义密度感知 + 运行时记忆治理 + 上下文回忆 = 压缩—遗忘—召回的闭环系统**

---

## 九、风险清单

| 风险 | 严重度 | 缓解措施 |
|------|:------:|---------|
| AM score 不等于语义密度 | 高 | Benchmark A 组实验验证；Phase 5 perplexity 备选 |
| 5 档不是最优离散集合 | 中 | Benchmark C 组在 4 种场景验证；可调整为 4 档或 6 档 |
| Route 0→5 过度依赖重建 | 高 | ReconstructionBudget 限制；cooldown 机制 |
| Scale 极端值质量崩塌 | 中 | Benchmark A 组 scale sweep；设 max_scale 硬限 |
| ThunderOMLX 集成冲突 | 低 | Phase 4 单独做，不影响 Phase 1-3 |

---

## 十、参考文献

1. [2603.25926] "Density-aware Soft Context Compression with Semi-Dynamic Compression Ratio" — DRS 离散化洞察、scale 旋钮、mean-pooling 骨干
2. [2603.19664] "KV-Direct" — h^(0) 存档 + 前缀重建，Route 5 基础
3. FlashMLX v2.0 — scored_pq + AM scoring + flat buffer Q8，Route 2/3 基础
4. ThunderOMLX — 调度器 + 缓存层 + SSD 持久化 + KVTC codec，集成目标

---

## 十一、文件索引

| 文件 | 角色 |
|------|------|
| `docs/route0-density-router-design.md` | 本文档 (设计 + 规划) |
| `docs/kv-direct-v2-paper-comparison.md` | Route 5 论文对比 (已有) |
| `model_cards/qwen3-8b-mlx-4bit.json` | v2.0 官方基线数据 |
| `src/flashmlx/config.py` | CacheConfig (待加 density_mode) |
| `mlx-lm-source/.../triple_layer_cache.py` | AM scoring + 压缩逻辑 (待改离散化) |
| `mlx-lm-source/.../kv_direct_cache.py` | Route 5 实现 (待加 ReconstructionBudget) |
| `mlx-lm-source/.../cache_factory.py` | 工厂 (待加 Route 0↔5 联动) |
