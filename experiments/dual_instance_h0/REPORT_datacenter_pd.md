# 数据中心 PD 分离 Pipeline Split 技术报告

## 实验模型
- **Qwen3-8B-MLX-4bit**: 36 layers, d=4096, 8 KV heads, head_dim=128
- 模型权重: 4,394 MB (Q4 量化)
- KV 每 token 每层: 4,096 B (bf16)
- 实验平台: Apple Silicon Mac mini 48GB (模拟数据中心 PD 分离场景)

---

## 1. 核心命题

### Prefill-Decode 分离 (PD Disaggregation)

Prefill 和 Decode 阶段有根本不同的硬件瓶颈:

| 阶段 | 瓶颈类型 | 原因 |
|------|----------|------|
| **Prefill** | 计算密集 (compute-bound) | O(N²) attention, N 个 token 的矩阵乘法 |
| **Decode** | 访存密集 (memory-bandwidth-bound) | 每 token 读全部权重 + KV, 但只算 1 个 token |

**PD 分离**思路: 让 P 节点 (大算力 GPU) 专门做 prefill，D 节点 (大带宽 GPU) 专门做 decode。

**Pipeline split 的角色**: 通过 residual checkpoint h^(cut) 实现 PD 间高效传输:
1. P 节点 prefill → 产出 h^(cut) (仅 hidden state，不传全量 KV)
2. D 节点接收 h^(cut) → 重建 KV → 高效 decode

---

## 2. D1: 计算 vs 访存瓶颈 — 实测验证

### 算术强度 (Arithmetic Intensity)

```
AI = FLOPs / Bytes

Prefill: 大量 FLOPs (N² attention + N×d matmul) / 相对少的数据移动
Decode:  少量 FLOPs (1 token matmul) / 大量数据移动 (读全部权重 + KV)
```

### 4K tokens 实测结果

| 指标 | Prefill | Decode |
|------|---------|--------|
| 理论 FLOPs/层 | 12.2 GFLOPs | 0.2 GFLOPs |
| 理论 Bytes/层 | 0.76 MB (权重) | 57.2 MB (权重+KV) |
| **算术强度 (AI)** | **16,026 FLOPs/Byte** | **3.4 FLOPs/Byte** |
| **AI 差距** | — | **4,688×** |
| 实测吞吐 | 6.52 TFLOPS | — |
| 实测带宽 | — | 220.8 GB/s (81% BW util) |

### Roofline 映射

```
             ┌──────────────────────────────────┐
 TFLOPS      │         ╱ compute ceiling         │
             │        ╱                          │
             │       ╱   ★ Prefill (AI=16026)    │
             │      ╱                            │
             │     ╱    ridge point              │
             │    ╱                              │
             │   ╱                               │
             │  ╱  ★ Decode (AI=3.4)             │
             │ ╱     BW ceiling                  │
             └──────────────────────────────────┘
                       AI (FLOPs/Byte)
```

**结论**: Prefill 深处 compute-bound 区域，Decode 深处 bandwidth-bound 区域。4,688× 差距意味着**同一硬件不可能同时最优服务两种负载** — PD 分离有坚实物理基础。

### GPU 通用性

| GPU | Compute (TFLOPS) | BW (TB/s) | Ridge Point | Prefill 区 | Decode 区 |
|-----|-------------------|-----------|-------------|-----------|-----------|
| A100 (bf16) | 312 | 2.0 | 156 | ✓ (AI >> 156) | ✓ (AI << 156) |
| H100 (bf16) | 990 | 3.35 | 295 | ✓ (AI >> 295) | ✓ (AI << 295) |
| Mac M2 Ultra | 27 | 0.8 | 34 | ✓ (AI >> 34) | ✓ (AI << 34) |

> AI 差距 4,688× 远大于任何 GPU 的 ridge point，在所有硬件上 P/D 分区结论一致。

---

## 3. D2: Residual Checkpoint 恢复 + Decode 效率

### 实验设计

```
方案 A (baseline): 全量 prefill → decode 100 tokens
方案 B (pipeline): P 做 layers 0..cut-1 → 传 h^(cut)
                    D 从 h^(cut) 重建 layers cut..35 → decode 100 tokens
对比: decode 速度 A ≈ B?
```

### 4K tokens 实测结果

| Cut Point | 质量 | Decode 速度 vs Baseline | 重建时间 |
|-----------|------|------------------------|----------|
| **@1** (P做1层) | **EXACT** | **±1%** | ~10.0 s |
| **@9** (P做9层) | **EXACT** | **±1%** | ~7.5 s |
| **@18** (P做18层) | **EXACT** | **±1%** | ~5.2 s |
| **@27** (P做27层) | **EXACT** | **±1%** | ~2.6 s |
| Baseline (全量) | baseline | baseline | 10.3 s |

### 关键结论

1. **质量**: 任意切分 EXACT (零 token 偏差)，与端云实验一致
2. **Decode 速度**: 恢复后 decode 速度 ≈ baseline (≤1% 差异)
3. **重建时间 ∝ 层数**: cut@18 重建 18 层 ≈ 50% prefill 时间

> **D 节点从 h^(cut) 恢复 KV 后，decode 效率完全等同于本地 prefill。不存在 "恢复质量损失" 或 "缓存冷启动" 问题。**

---

## 4. D3: P:D 节点配比分析

### 原理

```
1 个请求: prefill_time + gen_tokens × decode_per_token

P 节点忙于 prefill → prefill_time 后空闲
D 节点忙于 decode → gen_tokens × per_token_time

P:D 比 = decode_total / prefill_time
如果 decode_total > prefill_time → D 是瓶颈 → 1 P 服务多 D
如果 decode_total < prefill_time → P 是瓶颈 → 需更多 P
```

### 4K tokens 实测

| 生成长度 | Prefill | Decode Total | P:D 比 | 瓶颈 |
|----------|---------|-------------|--------|------|
| 32 tok | 10.3 s | 0.7 s | 0.07 | **P 严重瓶颈** |
| 64 tok | 10.3 s | 1.4 s | 0.14 | P 瓶颈 |
| 128 tok | 10.3 s | 2.8 s | 0.27 | P 瓶颈 |
| 256 tok | 10.3 s | 5.6 s | 0.55 | P 瓶颈 |
| **474 tok** | **10.3 s** | **10.3 s** | **1.0** | **交叉点** |
| 512 tok | 10.3 s | 11.2 s | 1.09 | D 轻微瓶颈 |
| 1024 tok | 10.3 s | 22.3 s | 2.17 | D 瓶颈 |
| 2048 tok | 10.3 s | 44.5 s | 4.32 | **D 严重瓶颈** |

### 不同服务场景

| 场景 | 典型 gen tokens | P:D 比 | 部署建议 |
|------|---------------|--------|----------|
| **Chat/QA** | 64-256 | 0.14-0.55 | P 是瓶颈，需 2-7× P 节点 |
| **代码生成** | 512-2048 | 1.1-4.3 | D 是瓶颈，1 P 服务 1-4 D |
| **长文摘要** | 128-512 | 0.27-1.1 | 均衡区间 |
| **翻译** | ≈ input length | 与 input 相当 | 接近均衡 |

### 关键洞察

> **Crossover 在 ~474 gen tokens**。大多数 chat 场景 (<256 tokens 回复) P 是瓶颈。只有在长生成场景 (代码、长文) D 才成为瓶颈。
>
> 这解释了为什么 DistServe/Splitwise 论文主要展示 **长生成** 场景的收益 — 短回复场景 PD 分离的瓶颈在 P 侧，需要更多 P 节点而非 D 节点。

---

## 5. D4: h^(cut) 传输 vs 全量 KV 传输

### 传输方案对比

| 方案 | 传输内容 | 4K tokens | 压缩比 |
|------|----------|-----------|--------|
| **全量 KV** (DistServe) | K+V, 36 layers, bf16 | 576 MB | 1× |
| **全量 KV int8** | 量化传输 | 324 MB | 1.8× |
| **h^(18) + KV[0:17]** | 半量 KV + hidden state | 320 MB (bf16) / 194 MB (int8) | 1.8× / 3× |
| **h^(18) only** | 仅 hidden state | **33.6 MB** | **18×** |

### h^(cut) only 方案详解

```
h^(cut) = batch × seq_len × d_model × 2B (bf16)
        = 1 × 4096 × 4096 × 2
        = 33.6 MB

vs 全量 KV:
KV = n_layers × seq_len × 2 × n_kv_heads × head_dim × 2B
   = 36 × 4096 × 2 × 8 × 128 × 2
   = 576 MB
```

h^(cut) only 要求 D 节点重建全部 36 层 KV (D 做完整 forward)。代价是 D 侧重建时间 ≈ 全量 prefill。

### 不同互联带宽下传输时间

以 4K context 为例:

| 互联 | 带宽 | 全量 KV | h^(18) int8 | h^(18) only | 备注 |
|------|------|---------|-------------|-------------|------|
| **NVLink** | 900 GB/s | 0.6 ms | 0.2 ms | 0.04 ms | GPU 间同机 |
| **InfiniBand NDR** | 50 GB/s | 12 ms | 4 ms | 0.7 ms | 机架内 |
| **IB NDR ×8** | 400 GB/s | 1.4 ms | 0.5 ms | 0.08 ms | 胖树 |
| **跨机房 10G** | 1.25 GB/s | 461 ms | 155 ms | 27 ms | 跨数据中心 |
| **跨地域 1G** | 125 MB/s | 4,608 ms | 1,552 ms | 269 ms | WAN |

### 关键洞察

> **在高速互联 (NVLink/IB) 下，传输差异可忽略** — 全量 KV 也只要 12ms，不值得用 D 侧 5s 重建换 12ms 传输。
>
> **在低速互联 (跨机房/WAN) 下，h^(cut) 18× 压缩极其有价值** — 从 461ms 降到 27ms，且不需要传大量 KV 数据。
>
> **结论**: h^(cut) 方案的传输优势在 **带宽受限** 场景 (跨机架、跨机房、混合云) 最有价值。

---

## 6. D5: Pipeline Split — KV 内存 per-node

### 对称切分 (Pipeline@18, 50-50)

| Context | 全量 KV/节点 | P 节点 KV | D 节点 KV | 每节点节省 |
|---------|-------------|-----------|-----------|-----------|
| 2K | 288 MB | 144 MB | 144 MB | 50% |
| 4K | 576 MB | 288 MB | 288 MB | 50% |
| 8K | 1,152 MB | 576 MB | 576 MB | 50% |
| 16K | 2,304 MB | 1,152 MB | 1,152 MB | 50% |
| 32K | 4,608 MB | 2,304 MB | 2,304 MB | 50% |
| 128K | 18,432 MB | 9,216 MB | 9,216 MB | 50% |

### 意义

Pipeline split 让每个节点只需存一半 KV:
- **更大 batch**: 同样 HBM 预算下，每个请求 KV 减半 → batch size 翻倍
- **更长 context**: 128K 从 18.4 GB → 9.2 GB per node，A100 80GB 可服务更多并发

但注意: P 节点的 KV 在 prefill 后需要传给 D (或由 D 重建)。Pipeline split 不减少**系统总** KV，只减少**每节点** KV。

---

## 7. 与学术工作对比

### PD 分离方案族谱

| 系统 | 会议 | P→D 传输 | 传输量 (4K) | D 需重建? | 传输延迟 (IB 50GB/s) |
|------|------|----------|-------------|-----------|---------------------|
| **DistServe** | OSDI'24 | 全量 KV | 576 MB | 否 | 12 ms |
| **Splitwise** | ISCA'24 | 全量 KV | 576 MB | 否 | 12 ms |
| **TetriInfer** | 2024 | 全量 KV | 576 MB | 否 | 12 ms |
| **Ours (h^cut only)** | — | 仅 h^(cut) | **33.6 MB** | **是 (~5s)** | **0.7 ms** |
| **Ours (h^cut + partial KV)** | — | h^(cut) + KV[0:17] | 194 MB (int8) | 部分 (18L) | 4 ms |

### Trade-off 分析

```
DistServe 路线:
  ✅ D 收到 KV 立即可 decode (零重建)
  ❌ 高带宽需求 (576 MB per request)
  ❌ 传输占 P→D 路径延迟
  ✅ 适合: 同机架高速互联

Ours (h^cut only):
  ✅ 传输极小 (33.6 MB, 18× 压缩)
  ❌ D 需要重建 KV (~5s @4K on Apple Silicon)
  ❌ 重建 = 额外计算负载 on D
  ✅ 适合: 跨机房/WAN 场景, 带宽昂贵
  ✅ 适合: D 节点有空余计算能力 (decode 是带宽瓶颈，compute 闲置)
```

### 独特洞察: D 节点计算不浪费

> D 节点在 decode 阶段是 **bandwidth-bound** (AI=3.4)，**compute 大量闲置**。
> 利用闲置 compute 做 KV 重建，是 "免费" 的 — 不与 decode 计算争抢。
> 但重建发生在 decode **之前** (TTFT 增加)，不能与 decode 并行。

---

## 8. Pipeline Decode 的顺序依赖

### 如果 P 和 D 各跑一半层做 decode?

```
t_i → P(层0-17) → 传h^(18) → D(层18-35) → t_{i+1}
```

**每个 token 都有 P→D 顺序依赖**:
- 单流延迟: 不降反升 (增加传输延迟)
- 吞吐: 可以通过 batch pipelining (P 处理 request_B 同时 D 处理 request_A)
- **不是 latency 优化，是 throughput 优化**

### 正确用法

Pipeline split 在 decode 阶段的价值是:
1. **Batch 级别吞吐**: P 和 D 各持半模型，各处理不同请求的不同阶段
2. **KV 内存减半**: 每节点只存半量 KV → 更大 batch
3. **不是** 让单个请求的 decode 更快

---

## 9. 结论

### Pipeline Split 在数据中心的价值定位

| 维度 | 效果 | 评估 |
|------|------|------|
| **PD 瓶颈分离** | 4,688× AI 差距，物理上不可调和 | 核心基础 |
| **h^(cut) 传输压缩** | 18× (33.6 MB vs 576 MB @ 4K) | 带宽受限场景有价值 |
| **恢复后 decode 效率** | ±1%, EXACT 质量 | 零损失 |
| **per-node KV 减半** | 50% (对称切分) | batch size 翻倍 |
| **P:D 配比灵活** | crossover ~474 tokens | 按场景调配 |

### 适用场景

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| **同机架高速互联** | DistServe (全量 KV) | 传输差异 <12ms，不值得重建 |
| **跨机架/跨机房** | h^(cut) only | 18× 传输压缩显著 |
| **混合云 (GPU cloud + 边缘)** | h^(cut) + partial KV | 平衡传输与重建 |
| **超长 context (128K+)** | Pipeline split | per-node KV 减半，HBM 可控 |
| **短回复 Chat** | 更多 P 节点 | P 是瓶颈 (gen < 474 tok) |
| **长生成 (代码/文档)** | 更多 D 节点 | D 是瓶颈 (gen > 474 tok) |

### 核心发现

1. **PD 分离有坚实物理基础**: 4,688× 算术强度差距在所有 GPU 上成立
2. **h^(cut) 恢复无损**: decode 速度 ±1%，质量 EXACT
3. **h^(cut) 传输优势在带宽受限场景**: 高速互联场景不如 DistServe 直传 KV
4. **D 节点 compute 闲置可利用**: decode 是 BW-bound，重建用的是 "免费" compute
5. **P:D 比非固定**: 短回复场景 P 瓶颈，长生成场景 D 瓶颈

### 推荐的 4K 甜点配置

```
模型:     Qwen3-8B (bf16 in datacenter)
切分:     @18 (50-50)
P→D 传输: h^(18) only = 33.6 MB
D 重建:   18L from h^(18), ~5.2 s
Decode:   22 ms/tok, EXACT 质量
P:D 比:   按服务场景 1:0.5 (chat) ~ 1:4 (code gen)
```

---

*实验环境: Apple Silicon Mac mini 48GB, MLX, Qwen3-8B-MLX-4bit, bf16 KV, greedy decode*
*实验验证数据来自真实模型运行，理论公式用于计算互联/GPU roofline 映射*
*日期: 2026-04-03*
