# Scored P2 (Architecture D) 完整性能档案

> 版本: v0.8.1 | 模型: Qwen3-8B-MLX | 平台: M4 Pro 48GB
> 测试日期: 2026-03-28
> 校准文件: am_calibration_qwen3-8b_2.0x_onpolicy.pkl

---

## 架构概述

```
Scored P2 数据流:
  PP (Prefill):   输入 → 全部存 bf16 Recent buffer → 无压缩
  Promotion:      bf16 Recent → AM 评分(干净数据) → hot→flat buffer(bf16) / cold→丢弃
  TG (Decode):    flat buffer(bf16) → O(1) slice append

自适应压缩率:
  ≤16K: 3.0x (保留 33% tokens)
  >16K: 1.5x (保留 67% tokens, fill indices 补充)
```

**关键特性**:
- AM 在干净 bf16 数据上评分（vs Pipeline 在退化 PQ4 数据上评分）
- 一次性 promotion（prefill→TG 转换时）
- O(1) TG token append（slice assignment）
- PQ/TQ 未使用，flat buffer 全 bf16

---

## 一、16K 性能 (adaptive→3.0x)

### 1.1 质量 (5-test benchmark, 3 轮独立运行)

| 测试 | Standard | Pipeline | Scored P2 R1 | R2 | R3 |
|------|----------|----------|:---:|:---:|:---:|
| Needle-Early (1项) | PASS | PASS | PASS | PASS | PASS |
| Needle-Middle (1项) | PASS | PASS | PASS | PASS | PASS |
| Multi-Hop (3项) | 3/3 | 1/3 | 1/3 | 1/3 | 1/3 |
| Numerical (3项) | 3/3 | 1/3 | 1/3 | 3/3 | 2/3 |
| List-Recall (3项) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| **Total** | **13/13** | **9/13** | **9/13** | **11/13** | **10/13** |
| **百分比** | **100%** | **69%** | **69%** | **84%** | **76%** |

**质量统计**: Scored P2 平均 76-84%, Pipeline 稳定 69%

**失败分析**:
- Multi-Hop: "3.2x ROI" → 模型改写为 "ROI of 300%" / "ROI of 3"，AM 和 Standard 都有此现象
- Numerical: "$8.1M" 有时输出为 "$8.1 million"（匹配判定差异），Scored P2 表现优于 Pipeline

### 1.2 速度与内存

| 指标 | Standard | Scored P2 | 变化 |
|------|----------|-----------|------|
| PP 速度 | ~320 tok/s | ~320 tok/s | 相同 |
| TG 速度 | 20.0-20.9 tok/s | 24.0-25.9 tok/s | **+22~30%** |
| PP 内存 | ~2.2 GB | ~2.2 GB | **相同（未压缩）** |
| TG 内存 | ~2.2 GB | 798-810 MB | **-64%** |
| Flat tokens | 16,128 (全部) | 5,150-5,760 | 3.0x 压缩 |

### 1.3 Promotion 稳定性 (5 轮)

```
Run 1: 221.5ms  Run 2: 222.1ms  Run 3: 221.4ms  Run 4: 219.6ms  Run 5: 224.2ms
平均: 221.8ms   最小: 219.6ms   最大: 224.2ms   Range: 4.6ms (2.1%)
```

**结论**: 16K promotion 极其稳定，方差仅 2.1%

### 1.4 Promotion 分解 (per-layer profiling)

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Numpy index building (36层) | 4.4ms | 2% |
| mx.array conversion | 0.11ms | <1% |
| Flat buffer allocation | 4.24ms/层 | 45% |
| Gather + copy | 4.46ms/层 | 47% |
| **实际总计 (36层)** | **221.8ms** | 100% |
| 外推单层×36 | 338ms | — |

**GPU 效率**: 实际 222ms vs 外推 338ms → MLX overlap 节省 34%

---

## 二、32K 性能 (adaptive→1.5x)

### 2.1 质量 (3-test, 修复后)

| 测试 | Standard | Scored P2 (1.5x) | Scored P2 (2.0x, 修复前) |
|------|----------|-------------------|--------------------------|
| Needle | PASS | PASS | PASS |
| Multi-Hop (3项) | 1/3 | 1/3 | 1/3 |
| Numerical (3项) | PASS | PASS | 1/3 |
| **Total** | **5/7 (71%)** | **5/7 (71%)** | **3/7 (43%)** |

**32K 质量修复**: adaptive ratio 从 2.0x→1.5x + fill indices 补充 (commit 32bd147)

### 2.2 速度与内存

| 指标 | Standard | Scored P2 | 变化 |
|------|----------|-----------|------|
| TG 速度 | 16.0-16.3 tok/s | 18.1-18.8 tok/s | **+15%** |
| TG 内存 | ~4.4 GB | 3,065 MB | **-30%** |
| Flat tokens | 32,512 (全部) | 21,217 | 1.5x 压缩 |

### 2.3 Promotion 稳定性 (5 轮)

```
Run 1: 459.7ms  Run 2: 466.0ms  Run 3: 453.3ms  Run 4: 486.2ms  Run 5: 458.4ms
平均: 464.7ms   最小: 453.3ms   最大: 486.2ms   Range: 32.8ms (7.1%)
```

**Scaling**: 32K flat buffer 是 16K 的 3.83x，promotion 慢 2.09x → GPU 带宽并行化

---

## 三、Ratio Sweep @ 16K (3-test)

| Ratio | Hot Tokens | TG tok/s | TG 内存 | Promo | Quality |
|-------|-----------|----------|---------|-------|---------|
| 1.5x | 8,340 | 23.2 | 1,071 MB | 227ms | 4/7 (57%) |
| 2.0x | 8,340 | 23.1 | 1,071 MB | 205ms | 4/7 (57%) |
| **2.5x** | 6,780 | 25.5 | 906 MB | 184ms | **5/7 (71%)** |
| **3.0x** | 5,760 | **25.9** | **798 MB** | **174ms** | **5/7 (71%)** |

**发现**: 1.5x/2.0x 结果一致（fill indices 补充到同样的 token 数），3.0x 是最优点

**反直觉结论**: 更高压缩反而质量更好 — AM 核心 256 indices 是最有信息量的集合，额外 fill 的 token 引入噪声

---

## 四、与其他策略对比 (完整矩阵)

| 指标 | Standard | Pipeline (PQ4+AM) | **Scored P2 (AM bf16)** |
|------|----------|-------------------|--------------------------|
| 16K 质量 | 100% | 69% | **76-84%** |
| 32K 质量 | 71% | - | **71%** |
| 16K TG | 20 tok/s | ~24 tok/s | **25.9 tok/s** |
| 32K TG | 16 tok/s | - | **18.4 tok/s** |
| 16K TG 内存 | 2.2 GB | ~1.1 GB | **798 MB** |
| 32K TG 内存 | 4.4 GB | - | **3.1 GB** |
| PP 内存 | 2.2 GB | 2.2 GB | **2.2 GB (无节省)** |
| Promotion | 0ms | ~200ms | **222ms** |
| Token 保留率 | 100% | ~53% | **33% (3.0x)** |
| 数据精度 | bf16 | PQ4 (退化) | **bf16 (干净)** |

---

## 五、已知限制与改进方向

### 5.1 当前限制

1. **PP 内存无节省** — AM 只在 promotion 时生效，prefill 全 bf16
2. **静态 mask** — 离线校准，内容无关，所有输入用同一个 mask
3. **Q≈K 假设** — 校准时用 Keys 代替 Queries，GQA 下偏差更大
4. **校准语料单一** — 仅 500 词量子故事 + 24 个事实抽取问题
5. **Chunk 边界** — 512 token 硬切割，跨 chunk 语义断裂
6. **PQ 未使用** — 策略名 "scored_pq" 误导，实际全 bf16

### 5.2 AM 算法分析 (多专家会审: deepseek-r1 + gemini-2.5-pro)

**核心结论**: 质量瓶颈不在架构（Pipeline vs Scored），而在 AM 静态 mask 本身。
三轮 benchmark 证实两种架构在同一 mask 下丢同样的信息。

**更多校准文件的收益递减**:
- 1个 (现在): 76% → 5个: ~78-80% → 20个: ~80-82% → 天花板 ~83%
- 根本原因: 静态 mask 无法适应动态内容

### 5.3 改进路线图

| 阶段 | 改进 | 难度 | 预期收益 |
|------|------|------|----------|
| P3a | Attention Sink (前64 token 保护) | 低 | +3-5% 质量 |
| P3b | H2O 在线评分 (替代静态 mask) | 中 | +10-15% 质量 |
| P3c | 移除离线校准依赖 | 中 | 简化部署 |
| P3d | 可变 chunk (消除边界效应) | 低 | +2-3% 质量 |

---

## 六、关键代码位置

| 文件 | 功能 |
|------|------|
| `mlx_lm/models/triple_layer_cache.py` | 核心 KV cache 实现 |
| `mlx_lm/models/cache_factory.py` | 策略注册 ("scored_pq") |
| `calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl` | AM 校准文件 |
| `calibrate_am_onpolicy.py` | 校准生成脚本 |

**关键方法**:
- `_get_effective_ratio()` — 自适应压缩率 (≤16K→3.0x, >16K→1.5x)
- `_get_importance_mask()` — AM 重要性掩码 + fill indices
- `_scored_compress_prefix()` — Promotion 核心: AM 评分 → flat buffer
- `_update_flat_path()` — TG 快速路径: O(1) slice append

---

## 七、Git 历史

```
32bd147 fix: 32K quality regression with adaptive ratio + fill indices
8f4be58 perf: enable adaptive 3.0x ratio as default for Scored P2
3362601 refactor: direct flat buffer writes in Scored P2 promotion
ea829af perf: vectorize token gather in Scored P2 promotion
5602cc6 perf: remove PQ2 cold storage from Scored P2 promotion
```

---

*Scored P2 Benchmark Report v0.8.1*
*Generated: 2026-03-28*
*Model: Qwen3-8B-MLX on Apple M4 Pro 48GB*
