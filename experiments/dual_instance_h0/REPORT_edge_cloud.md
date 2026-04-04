# 端云协同 Pipeline Split 技术报告

## 实验模型
- **Qwen3-8B-MLX-4bit**: 36 layers, d=4096, 8 KV heads, head_dim=128
- 模型权重: 4,394 MB (Q4 量化)
- KV 每 token 每层: 4,096 B (bf16)

---

## 1. 核心架构

```
Cloud (GPU):  layers 0..cut-1 → 计算 h^(cut) + KV[0:cut-1]
              ↓ 传输: h^(cut) + KV[0:cut-1] (int8 压缩)
Edge (设备):  注入云端 KV → 从 h^(cut) 重建 KV[cut:L-1] → decode
```

Pipeline split 在任意层切分均保持 **EXACT** 质量 (零 token 偏差)。切分层选择影响的是计算/传输/内存的 trade-off，不影响质量。

---

## 2. 实验矩阵与结果

### 2.1 Context = 2K tokens (scored_pq 甜点)

| 配置 | KV Memory | 传输量 | 计算 | 质量 |
|------|-----------|--------|------|------|
| Baseline 全量 | 288 MB | - | 36L | baseline |
| Pipeline@18 bf16 | 288 MB | 97 MB (int8) | 18L, -50% | EXACT |
| scored_pq 全量 | ~100 MB | - | 36L | EXACT |

> 2K ≤ scored_max_cache=2048，scored_pq 无需 eviction，质量完美。

### 2.2 Context = 4K tokens (端侧甜点)

| 配置 | KV Memory | 节省 | 传输 (int8) | 质量 (重复) | 质量 (多样) | TTFT | TG |
|------|-----------|------|-------------|------------|------------|------|-----|
| A) Standard 全量 | 576 MB | - | - | baseline | baseline | - | - |
| B) Pipeline@18 bf16 | 576 MB | 0% | 194 MB | **EXACT** | **EXACT** | 150 ms | 22 ms/tok |
| C) scored_pq 全量 | 239 MB | 58% | - | 5/50 | 2/50 | 159 ms | 20 ms/tok |
| D) Pipe@18 + scored_pq | 408 MB | 29% | - | 23/50 | 2/50 | - | - |

> Pipeline@18 bf16 质量完美，传输 3× 压缩，50% 计算节省。
> scored_pq 在 4K 触发 eviction (4096→1702, ratio=3.0×)，质量崩溃。

### 2.3 Context = 16K tokens (上限测试)

| 配置 | KV Memory | 节省 | 传输 (int8) | 质量 (重复) | 质量 (多样) | TTFT | TG |
|------|-----------|------|-------------|------------|------------|------|-----|
| A) Standard 全量 | 2,304 MB | - | - | baseline | baseline | - | - |
| B) Pipeline@18 bf16 | 2,304 MB | 0% | 776 MB | **EXACT** | **EXACT** | 267 ms | 32 ms/tok |
| C) scored_pq 全量 | 813 MB | 65% | - | 5/50 | 6/50 | 586 ms | 22 ms/tok |
| D) Pipe@18 + scored_pq | 1,559 MB | 32% | - | **EXACT** | 2/50 | - | - |

> 16K 下 scored_pq eviction 更激进 (16384→5782→4085)，全量质量极差。
> D) 重复文本 EXACT 说明质量损失主要来自前 18 层 eviction，后 18 层 scored_pq 在重复文本上尚可。

---

## 3. 端侧内存画像

### Pipeline@18 (50-50 切分) 端侧内存

| 组件 | 4K | 16K |
|------|-----|------|
| 模型权重 (layers 18-35 + embed + lm_head) | 2,531 MB | 2,531 MB |
| KV Cache (36 layers × bf16) | 576 MB | 2,304 MB |
| **Total** | **3,107 MB** | **4,835 MB** |
| vs 全量本地 | 4,970 MB | 6,698 MB |
| **内存节省** | **37%** | **28%** |

> 内存节省来自**模型权重减半** (4,394→2,531 MB, 节省 1,863 MB)。
> KV 内存无节省 — 所有 36 层 KV 仍需在端侧存在。

### Pipeline@1 (极端切分) 端侧内存

| 组件 | 4K | 16K |
|------|-----|------|
| 模型权重 (layers 1-35 + embed + lm_head) | 4,291 MB | 4,291 MB |
| KV Cache | 576 MB | 2,304 MB |
| **Total** | **4,867 MB** | **6,595 MB** |
| vs 全量本地 | 4,970 MB | 6,698 MB |
| **内存节省** | **2%** | **2%** |

> Cut@1 几乎没有意义: 只省 1 层权重 (103 MB)，传输也最大化。

---

## 4. 传输分析

### 传输组成

```
传输 = h^(cut) + KV[0:cut-1]

h^(cut) = batch × N × d_model × 2B (bf16)
        = 1 × N × 4096 × 2B
        = N × 8 KB

KV[0:cut-1] = cut × N × 2 × n_kv_heads × head_dim × 2B (bf16)
            = cut × N × 4 KB
```

### 实测传输量

| 切分 | 4K bf16 | 4K int8 | 16K bf16 | 16K int8 | 压缩比 (vs 全量 KV) |
|------|---------|---------|----------|----------|---------------------|
| @1 | 48 MB | 41 MB | 192 MB | 164 MB | **14×** |
| @18 | 320 MB | 194 MB | 1,280 MB | 776 MB | **3×** |
| 全量 KV | 576 MB | - | 2,304 MB | - | 1× |

> int8 gs=32 对 KV 量化: ~1.78× 压缩 (含 scales/biases 开销)

---

## 5. 计算性能

### Prefill / 重建

| | 4K | 16K |
|---|-----|------|
| 全量 prefill (36L) | 10.3 s | 61.1 s |
| Pipeline@18 重建 (18L) | 5.2 s | 30.5 s |
| **计算节省** | **50%** | **50%** |
| Pipeline@1 重建 (35L) | 10.0 s | 61.9 s |
| 计算节省 | 3% | -1% |

### Decode 速度

| | 4K | 16K |
|---|-----|------|
| TTFT (Pipeline@18) | 150 ms | 267 ms |
| TG (Pipeline@18) | 22 ms/tok | 32 ms/tok |
| TG (scored_pq) | 20 ms/tok | 22 ms/tok |

> scored_pq 的 TG 更快 (KV 少 → attention 计算量少)，但质量不可接受。

---

## 6. 网络延迟模型

### TTFT = 传输时间 + RTT + 端侧重建时间

以 Pipeline@18 int8 为例:

| 网络 | 传输时间 (4K) | TTFT (4K) | 传输时间 (16K) | TTFT (16K) |
|------|-------------|-----------|--------------|------------|
| LAN (1Gbps, 0.5ms) | 2 ms | 5.2 s | 7 ms | 30.5 s |
| WiFi (50Mbps, 5ms) | 31 ms | 5.2 s | 124 ms | 30.6 s |
| 5G (100Mbps, 20ms) | 16 ms | 5.2 s | 62 ms | 30.6 s |
| 4G (20Mbps, 50ms) | 78 ms | 5.3 s | 311 ms | 30.8 s |

> TTFT 被端侧重建时间主导 (5-30s)，网络传输时间相对可忽略。
> 这意味着: 传输压缩 (int8 vs bf16) 价值不大，瓶颈在端侧计算。

---

## 7. Edge-side AM 压缩探索

### 7.1 朴素 token eviction — 全部失败

直接从所有层删除 token (K-norm / Sinks+Recent 策略):

| 策略 | 保留率 | 质量 |
|------|--------|------|
| K-norm top 75% | 75% | 0/50 |
| K-norm top 50% | 50% | 0/50 |
| Sinks+Recent 50% | 50% | 13/50 |
| Sinks+K-norm 50% | 50% | 9/50 |

> **根因**: Token eviction 直接移除 softmax 分母项，36 层累积误差导致灾难性崩溃。

### 7.2 FlashMLX scored_pq — 质量受限

scored_pq 使用 AM 评分 + PQ4/PQ2 差分量化 (非 token 删除):

| Context | KV 节省 | 质量 |
|---------|---------|------|
| 2K (≤ max_cache) | ~60% | EXACT (无 eviction) |
| 4K (2× max_cache) | 58% | 2-5/50 |
| 16K (8× max_cache) | 65% | 5-6/50 |

> scored_pq 的 `scored_prefill_chunk_evict` 在 context > scored_max_cache 时触发 3.0× ratio eviction，质量不可接受。
> 仅在 context ≤ scored_max_cache 时质量完美。

### 7.3 关键洞察

- **Beta 补偿**: FlashMLX 的 AM 通过 `log(beta)` 补偿 attention scores (NNLS 拟合)，理论上可保持 softmax 分布。但 prefill chunk eviction 的大比例删除超出了 beta 补偿能力。
- **前层比后层更敏感**: Pipeline@18 + scored_pq 仅压缩后 18 层时，重复文本 EXACT，说明后层对 eviction 容忍度更高。
- **PQ 量化 vs token 删除**: scored_pq 的 PQ 压缩 (不删除 token) 效果好于纯 eviction，但 prefill eviction 阶段仍是质量瓶颈。

---

## 8. 结论

### Pipeline Split 的价值定位

| 维度 | 效果 | 评估 |
|------|------|------|
| **质量** | EXACT (零损失) | 核心优势 |
| **计算** | -50% prefill (cut@18) | 显著 |
| **模型内存** | -43% 权重 (4.4→2.5 GB) | 显著 |
| **KV 内存** | 0% 节省 | **无改善** |
| **传输** | 3× 压缩 (int8 cut@18) | 有用但非瓶颈 |

### Pipeline Split 不解决的问题

1. **KV 内存不减少**: 所有 36 层 KV 在 decode 时必须全部驻留端侧
2. **长 context 内存仍是瓶颈**: 16K 时 KV = 2.3 GB，远超模型权重
3. **端侧 AM 压缩质量不可接受**: scored_pq eviction 在 context > 2K 时崩溃

### 适用场景

**Pipeline@18 bf16 (推荐配置)**:
- 4K context: 端侧 3.1 GB (iPhone 15 Pro 8GB 可跑)
- 质量: EXACT
- 限制: 端侧仍需全量 KV，更长 context 内存紧张

**真正需要 Pipeline Split 的场景**:
- 模型太大放不进端侧 (14B/32B)
- 需要 cloud prefill 加速 (利用 GPU 集群)
- 对 TTFT 敏感且网络延迟 < 端侧计算延迟

### 4K 甜点参数

```
模型:     Qwen3-8B-MLX-4bit
切分:     @18 (50-50)
传输:     h^(18) + KV[0:17] int8 = 194 MB
端侧:
  权重    2,531 MB
  KV      576 MB
  Total   3,107 MB
性能:
  重建    5.2 s
  TTFT    150 ms
  TG      22 ms/tok
质量:     EXACT
```

---

*实验环境: Apple Silicon, MLX, bf16 KV, greedy decode*
*日期: 2026-04-03*
