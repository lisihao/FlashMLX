# h^(0) Residual Checkpoint on NVIDIA Rubin R100 — 投影分析

## Executive Summary

基于 Qwen3-8B 在 M4 Pro 上的完整实测数据 (D6-D10)，投影到 NVIDIA Rubin R100 GPU 的 PD 分离部署。h^(0) 方案在三个维度带来量化优势：

| 维度 | DistServe (全量 KV) | h^(0) 方案 | 优势 |
|------|---------------------|-----------|------|
| P→D 传输 @128K | 368 ms / 18.4 GB | 20.5 ms / 1.0 GB | **18× 带宽节省** |
| Session 容量 @128K | 14 sessions | 60 sessions | **4.3× 密度** |
| 跨机架部署 | 受限于 IB 带宽 | IB 永远不饱和 | **解除拓扑约束** |
| KV 重建延迟 @4K | 0 (直接 decode) | 13 ms (单节点) / 3.4 ms (4D 并行) | **可忽略** |

**核心洞察**: 在 Rubin R100 上, h^(0) 的重建时间从 M4 Pro 的 11 秒降到 13 毫秒 — 从 "不可接受" 变成 "用户无感"。这使得 h^(0) 方案在数据中心场景中实际可行, 同时保留其 18× 传输压缩和 4.3× session 密度优势。

---

## 1. NVIDIA Rubin R100 硬件画像

### 1.1 核心规格

| 参数 | R100 | 对比 H100 | 对比 M4 Pro |
|------|------|----------|-------------|
| bf16 TFLOPS | ~8,000 | 990 | 27 |
| FP8 TFLOPS | ~16,000 | 1,979 | — |
| HBM 容量 | 288 GB (HBM4) | 80 GB (HBM3e) | 48 GB (unified) |
| HBM 带宽 | 22 TB/s | 3.35 TB/s | 273 GB/s |
| NVLink 6 | 3.6 TB/s | 900 GB/s (NVL4) | — |
| IB NDR | 50 GB/s | 50 GB/s | — |
| 架构 | Vera Rubin (2025) | Hopper (2022) | Apple M4 (2024) |

### 1.2 Ridge Point 分析

```
Ridge Point = Peak TFLOPS / Peak BW = Compute-bound 与 BW-bound 的分界线

R100 bf16:  8,000 / 22,000 = 363.6 FLOPs/Byte
H100 bf16:    990 /  3,350 = 295.5 FLOPs/Byte
M4 Pro bf16:   27 /    273 =  98.9 FLOPs/Byte
```

Qwen3-8B 的 Arithmetic Intensity (实测):
- **Prefill AI = 16,026 FLOPs/Byte** → 远超所有 GPU 的 ridge point → **恒定 compute-bound**
- **Decode AI = 3.4 FLOPs/Byte** → 远低于所有 GPU 的 ridge point → **恒定 BW-bound**

**结论**: R100 的更高 ridge point (363.6) 不改变 P/D 分离的物理基础。
Prefill 仍然 compute-bound, Decode 仍然 BW-bound。PD 分离在 Rubin 上依然有效。

---

## 2. Qwen3-8B bf16 在 R100 上的投影

### 2.1 模型参数

| 参数 | 值 |
|------|-----|
| 架构 | 36L, d=4096, GQA 4:1, head_dim=128 |
| bf16 权重 | 16 GB |
| KV/tok/layer | 4,096 B |
| Full KV/tok (36L) | 147,456 B (0.14 MB) |
| h^(0)/tok | 8,192 B (0.008 MB) |
| h^(0) 压缩比 | **18×** |

### 2.2 Prefill 性能

```
实测 FLOPs (M4 Pro @4K): 5.83 TFLOPS × 10.815s = 63.05 TFLOP
独立验证 (架构计算):
  QKV: 2 × 4096 × 4096 × 6144 = 206 GFLOP/layer
  Attn: 2 × 32 × 4096² × 128 = 137 GFLOP/layer
  Out:  2 × 4096 × 4096²     = 137 GFLOP/layer
  FFN:  6 × 4096 × 4096 × 12288 = 1,237 GFLOP/layer
  ─────────────────────────────────────────
  Total: 1,717 GFLOP/layer × 36 = 61.8 TFLOP  ✓ (vs 实测 63)
```

| GPU | MFU | Effective TFLOPS | Prefill @4K | Throughput |
|-----|-----|-----------------|-------------|------------|
| M4 Pro | 21.6% | 5.83 | 10,815 ms | 0.09 req/s |
| H100 | 60% | 594 | 106 ms | 9.4 req/s |
| **R100** | **60%** | **4,800** | **13.1 ms** | **76 req/s** |
| R100 | 70% | 5,600 | 11.3 ms | 88 req/s |

> MFU 60% 假设: H100 实测 prefill MFU 通常 50-70% (vLLM/TensorRT-LLM), Rubin 架构更优化, 60% 为保守估计。

### 2.3 Decode 吞吐

```
Decode 是 BW-bound → 性能由 HBM 带宽决定

Available HBM = 288 - 16 (weights) = 272 GB
Full KV @4K/user = 147,456 × 4,096 = 576 MB

Max active batch @4K = 272 GB / 576 MB = 472 users
```

| Batch | Read/Step | Step Time | tok/s/user | Total tok/s |
|-------|-----------|-----------|------------|-------------|
| 1 | 16.6 GB | 0.75 ms | 1,333 | 1,333 |
| 100 | 73.6 GB | 3.35 ms | 299 | 29,900 |
| 472 | 288 GB | 13.1 ms | 76.3 | 36,036 |

> 关键: Active decode 时, h^(0) 方案与 DistServe 完全相同。
> 两方案的 active 用户都需要完整 KV cache 在 HBM 中。
> h^(0) 的优势不在 active decode, 而在传输、存储和重建。

### 2.4 KV 重建时间 (h^(0) → Full KV)

```
重建 = 完整 forward pass (与 prefill 计算量相同)
63 TFLOP on R100 @60% MFU = 13.1 ms

Sparse Checkpoint 并行重建:
  [0]           → 1 D-node:  13.1 ms
  [0,18]        → 2 D-nodes: 6.6 ms   (2.0× 加速)
  [0,9,18,27]   → 4 D-nodes: 3.4 ms   (3.9× 加速)
```

| GPU | 单节点 Recon @4K | 4D 并行 | 对比 |
|-----|-----------------|---------|------|
| M4 Pro | 11,049 ms | 2,791 ms | 基准 |
| H100 | 106 ms | 27 ms | 104× faster |
| **R100** | **13.1 ms** | **3.4 ms** | **843× faster** |

**结论**: R100 上重建从 "秒级" 变成 "毫秒级"。
13ms 的重建延迟 vs 13ms 的 decode step → 仅增加 1 个 decode step 的 TTFT。
用户几乎无感知。

---

## 3. P→D 传输: h^(0) 的杀手级优势

### 3.1 传输量对比

| Context | DistServe (Full KV) | h^(0) | 压缩 |
|---------|-------------------|-------|------|
| 4K | 576 MB | 32 MB | 18× |
| 8K | 1,152 MB | 64 MB | 18× |
| 32K | 4,608 MB | 128 MB | 18× |
| 128K | 18,432 MB | 1,024 MB | 18× |

### 3.2 传输时间 (IB NDR 50 GB/s)

| Context | DistServe | h^(0) | 加速 |
|---------|----------|-------|------|
| 4K | 11.5 ms | 0.64 ms | 18× |
| 32K | 92.2 ms | 5.1 ms | 18× |
| 128K | **368 ms** | **20.5 ms** | **18×** |

### 3.3 IB 链路饱和分析 (关键发现)

```
P prefill 速度: 76 req/s @4K
P 产生的数据流:

DistServe @4K:  76 × 576 MB = 43.8 GB/s  ← 接近 IB NDR 50GB/s 极限!
h^(0) @4K:      76 × 32 MB  =  2.4 GB/s  ← IB NDR 利用率仅 4.8%

DistServe @128K: 传输时间 368ms > 预期 prefill 时间 → IB 成为瓶颈!
h^(0) @128K:     传输时间 20.5ms << prefill 时间 → IB 永远不饱和
```

### 3.4 网络拓扑解放

```
                    DistServe                         h^(0)
                    ─────────                         ──────

  ┌─── Rack 0 ───┐                    ┌─── Rack 0 ───┐
  │ P pool        │    43.8 GB/s       │ P pool        │   2.4 GB/s
  │ R100 × N      │════════════════    │ R100 × N      │───────────── ···
  └───────────────┘    ↓ (需要          └───────────────┘   ↓ (IB NDR
                       NVLink 或                             足够, 可
  ┌─── Rack 0 ───┐    同机架)          ┌─── Rack 1 ───┐    跨机架!)
  │ D pool        │                    │ D pool        │
  │ R100 × M      │                    │ R100 × M      │
  └───────────────┘                    └───────────────┘

  DistServe:                           h^(0):
  P/D 必须在同机架                      P/D 可以跨机架
  (NVLink 连接)                        (IB NDR 足够)
  拓扑约束 → 资源利用率低               拓扑自由 → 灵活调度
```

**这是 h^(0) 在数据中心的最大价值**: 将 P/D 从 NVLink 拓扑约束中解放出来,
使得 P 和 D 可以部署在不同机架, 按需独立扩缩容。

---

## 4. Session 密度: 长 Context 下的 HBM 效率

### 4.1 内存模型

```
DistServe: 所有 session 存 Full KV (不管是否正在 decode)
  Memory/session @ctx = 147,456 × ctx bytes

h^(0): Active sessions → Full KV, Idle sessions → h^(0) archive
  Active: Full KV (已重建)
  Idle:   h^(0) = 8,192 × ctx bytes (18× 更小)
```

### 4.2 Session 容量对比 (R100 272 GB 可用)

假设 20% active rate (典型 chatbot: 用户看回复时 session idle):

| Context | DistServe Sessions | h^(0) Sessions | 放大 |
|---------|-------------------|----------------|------|
| 4K | 472 | 1,814 | **3.8×** |
| 8K | 236 | 786 | **3.3×** |
| 32K | 59 | 217 | **3.7×** |
| **128K** | **14** | **60** | **4.3×** |

计算公式 (h^(0)):
```
T × (active_rate × full_kv + (1-active_rate) × h0_size) ≤ available_hbm
T × (0.2 × 18.432 + 0.8 × 1.024) ≤ 272     (@128K)
T × 4.499 ≤ 272
T = 60
```

### 4.3 不同 Active Rate 下的 Session 密度 (@128K)

| Active Rate | DistServe | h^(0) | 放大 |
|------------|----------|-------|------|
| 100% (全活跃) | 14 | 14 | 1.0× (无优势) |
| 50% | 14 | 27 | 1.9× |
| 20% (典型) | 14 | **60** | **4.3×** |
| 10% (长文档) | 14 | **104** | **7.4×** |
| 5% (超长等待) | 14 | **177** | **12.6×** |

> DistServe 不受 active rate 影响 (所有 session 都存 full KV)。
> h^(0) 在低 active rate 下优势急剧放大。

---

## 5. P:D 最优扇出比

### 5.1 @4K Context

```
P 节点 (compute-bound):
  Prefill: 13.1 ms/req → 76 req/s per R100

D 节点 (BW-bound):
  Active batch: 472 users
  Decode: 76.3 tok/s per user
  Average response: 512 tokens → 6.7s per session
  Turnover: 472 / 6.7 = 70.4 new req/s per R100

  DistServe 最优 P:D = 76 / 70.4 ≈ 1:1
  h^(0) P:D = 相同 (active decode 性能一致)
```

### 5.2 @128K Context

```
P 节点:
  Prefill @128K FLOPs ≈ 63 × (128/4)^1.3 ≈ 5,200 TFLOP (attention quadratic)
  Time @60% MFU: 5,200 / 4,800 = 1.08s → 0.92 req/s

D 节点:
  Active batch: 14 (DistServe) / 14 (h^(0) active, 相同)
  Decode: ~80 tok/s/user
  Response 512 tok → 6.4s
  Turnover: 14 / 6.4 = 2.2 req/s

  但 h^(0) D 节点可同时维持 60 个 session (含 idle)!
  → 需要更少的 D 节点来服务相同的用户群

DistServe: P:D = 0.92 / 2.2 ≈ 1:2
  16 个 R100: 8P + 8D = 7.4 req/s, 可服务 8 × 14 = 112 active sessions

h^(0): P:D = 0.92 / 2.2 ≈ 1:2 (active turnover 相同)
  16 个 R100: 8P + 8D = 7.4 req/s, 可服务 8 × 60 = 480 total sessions
                                                        ↑ 4.3× more!
```

### 5.3 h^(0) 方案的 P:D 扇出优势

```
┌─────────────────────────────────────────────────────────┐
│                    @128K, 16 R100 集群                    │
├────────────────────────┬────────────────────────────────┤
│     DistServe          │        h^(0)                   │
│                        │                                │
│  8P + 8D               │  8P + 8D                      │
│  112 active sessions   │  480 total sessions           │
│  112 total sessions    │  (96 active + 384 idle)       │
│                        │                                │
│  P→D: 43.8 GB/s       │  P→D: 2.4 GB/s               │
│  需要 NVLink           │  IB NDR 足够                   │
│  同机架约束            │  跨机架自由                     │
│                        │                                │
│  每用户成本: 100%      │  每用户成本: 23.3%             │
└────────────────────────┴────────────────────────────────┘
```

---

## 6. 大模型投影: Llama-3.1-70B

### 6.1 模型参数

| 参数 | Llama-3.1-70B | Qwen3-8B | 对比 |
|------|---------------|----------|------|
| Layers | 80 | 36 | 2.2× |
| d_model | 8,192 | 4,096 | 2× |
| GQA | 8:1 | 4:1 | — |
| n_kv | 8 | 8 | 相同! |
| KV/tok/layer | 4,096 B | 4,096 B | 相同! |
| bf16 weight | 140 GB | 16 GB | 8.75× |
| h^(0) compression | **20×** | **18×** | 更优 |

### 6.2 R100 投影 (2× TP, 576 GB total)

| 指标 | DistServe | h^(0) | 优势 |
|------|----------|-------|------|
| Available HBM | 296 GB | 296 GB | — |
| Full KV @128K/user | 40 GB | 40 GB (active) | — |
| h^(0) @128K/user | — | 2.0 GB (idle) | 20× |
| Active batch @128K | 7 | 7 | — |
| Sessions @128K (20%) | **7** | **30** | **4.3×** |
| P→D @128K (IB) | 800 ms | 40 ms | **20×** |

### 6.3 NVL72 机架级分析 (70B)

```
NVL72: 72 × R100, 20,736 GB HBM, NVLink 全互联

70B with 8-way TP → 9 个 model 副本
每副本: 8 GPU, 2,304 GB HBM, ~1,164 GB for KV

@128K Context:
  DistServe: 1,164 / 40 = 29 sessions × 9 replicas = 261 total
  h^(0) (20%): 1,164 / 9.6 = 121 sessions × 9 replicas = 1,089 total
  放大: 4.2×

P→D 带宽 (NVL72 内, NVLink 3.6 TB/s per GPU):
  DistServe: 40 GB → 11.1 ms (可接受, 但占 NVLink 份额)
  h^(0): 2.0 GB → 0.56 ms (可忽略)
```

---

## 7. h^(0) 缓存: System Prompt 的零成本复用

### 7.1 场景

大量用户共享相同 system prompt (如 ChatGPT 的系统指令, 企业知识库注入)。

```
System Prompt: 4K tokens (典型)
  Full KV:   576 MB (每用户一份, 不可共享)
  h^(0):     32 MB (可缓存, 所有用户共享!)

h^(0) 缓存流程:
  1. 首次: P prefill → 生成 h^(0) → 缓存到 KV store
  2. 后续: D 直接从缓存取 h^(0) → 重建 KV → 跳过 P 节点!
     TTFT = cache_lookup + recon = ~0 + 13ms = 13ms
     vs DistServe TTFT = prefill = 13ms (相同!)
     但 P 节点完全不参与 → P pool 释放给非缓存请求
```

### 7.2 缓存经济性

| System Prompt | Full KV 缓存 | h^(0) 缓存 | 压缩 |
|---------------|-------------|-----------|------|
| 4K tokens | 576 MB | 32 MB | 18× |
| 32K tokens | 4,608 MB | 256 MB | 18× |
| 128K tokens (RAG) | 18,432 MB | 1,024 MB | 18× |

1000 个热门 system prompt @4K:
- Full KV: 576 GB (超过 2 个 R100)
- h^(0): 32 GB (1 个 R100 的 11%)

---

## 8. 综合对比: DistServe vs h^(0) on Rubin R100

### 8.1 Qwen3-8B @4K (单 R100)

| 指标 | DistServe | h^(0) | Δ |
|------|----------|-------|---|
| P→D 传输 | 576 MB, 11.5ms (IB) | 32 MB, 0.64ms | **18× 小** |
| IB 链路利用 | 87.6% (接近饱和) | 4.8% | **不饱和** |
| Active batch | 472 | 472 | = |
| Total sessions (20%) | 472 | 1,814 | **3.8×** |
| Decode 速度 | 76.3 tok/s/user | 76.3 tok/s/user | = |
| TTFT (cold) | 13.1 ms (prefill) | 26.2 ms (prefill + recon) | +13 ms |
| TTFT (cached prompt) | 13.1 ms | **13.1 ms** (recon only) | = |
| P/D 拓扑 | 需 NVLink | IB 足够 | **跨机架** |

### 8.2 Qwen3-8B @128K (单 R100)

| 指标 | DistServe | h^(0) | Δ |
|------|----------|-------|---|
| P→D 传输 | 18.4 GB, 368ms (IB) | 1.0 GB, 20.5ms | **18× 小** |
| IB 饱和? | **是** (P→D > prefill) | 否 | **解除瓶颈** |
| Active batch | 14 | 14 | = |
| Total sessions (20%) | **14** | **60** | **4.3×** |
| 每 session 成本 (HBM) | 18.4 GB | 4.5 GB (平均) | **4.1× 便宜** |

### 8.3 Llama-70B @128K (2× R100 TP)

| 指标 | DistServe | h^(0) | Δ |
|------|----------|-------|---|
| P→D 传输 | 40 GB, 800ms (IB) | 2 GB, 40ms | **20× 小** |
| Total sessions (20%) | **7** | **30** | **4.3×** |

---

## 9. 成本效益分析

### 9.1 每 Session 成本 (以 R100 HBM 为基准)

| 场景 | DistServe HBM/session | h^(0) HBM/session | 节省 |
|------|----------------------|-------------------|------|
| 8B @4K (20% active) | 576 MB | 150 MB | 74% |
| 8B @128K (20% active) | 18.4 GB | 4.5 GB | 76% |
| 70B @128K (20% active) | 40 GB | 9.6 GB | 76% |

### 9.2 等效 GPU 节省 (@128K, 1000 sessions)

```
Qwen3-8B, 1000 sessions, 20% active:
  DistServe: 1000 × 18.4 GB = 18,400 GB → 68 R100 (D pool)
  h^(0):     1000 × 4.5 GB  =  4,500 GB → 17 R100 (D pool)
  节省: 51 R100!

Llama-70B, 1000 sessions, 20% active (2× TP):
  DistServe: 1000 × 40 GB = 40,000 GB → 139 R100
  h^(0):     1000 × 9.6 GB = 9,600 GB → 34 R100
  节省: 105 R100!
```

### 9.3 IB 链路节省

```
@128K, 不用 h^(0):
  每 P 节点产生 18.4 GB/req → 需要多条 IB 链路或 NVLink 直连
  P/D 必须同机架 → 资源碎片化

用 h^(0):
  每 P 节点产生 1.0 GB/req → 单条 IB NDR (50 GB/s) 可支撑 ~48 req/s
  P/D 可跨机架 → 全局统一调度
```

---

## 10. 跨模型 × 跨 Checkpoint 策略分析

### 10.1 模型参数总览

| 模型 | L | d_model | GQA | n_kv | head_dim | bf16 | h^(0)/tok | KV/tok (all L) | 压缩 | Tier |
|------|---|---------|-----|------|----------|------|-----------|---------------|------|------|
| Qwen3-0.6B | 28 | 1,024 | 2:1 | 8 | 64 | 1.2 GB | 2,048 B | 57,344 B | **28×** | A |
| Qwen3-1.7B | 28 | 2,048 | 2:1 | 8 | 128 | 3.4 GB | 4,096 B | 114,688 B | **28×** | A |
| Qwen3-8B | 36 | 4,096 | 4:1 | 8 | 128 | 16 GB | 8,192 B | 147,456 B | **18×** | B |
| Llama-3.1-8B | 32 | 4,096 | 4:1 | 8 | 128 | 16 GB | 8,192 B | 131,072 B | **16×** | B |
| Mistral-7B | 32 | 4,096 | 4:1 | 8 | 128 | 15 GB | 8,192 B | 131,072 B | **16×** | B |
| Gemma-4-E4B | 42 | 2,560 | 4:1 | 2 | 256 | 8 GB | 5,120 B | 86,016 B | **17×** | B |
| Llama-3.1-70B | 80 | 8,192 | 8:1 | 8 | 128 | 140 GB | 16,384 B | 327,680 B | **20×** | B |
| Gemma-4-31B | 60 | 5,376 | 2:1 | 16 | 256 | 62 GB | 10,752 B | 983,040 B* | **91×*** | S |
| DeepSeek-V3 | 61 | 7,168 | MLA | — | 128 | 1.4 TB | 14,336 B | 62,464 B | **4.4×** | C |

> *Gemma-4-31B: 理论 91×。实际因 hybrid sliding window attention, @128K 有效 ≈ 50×。
> DeepSeek-V3: MLA 已将 KV 压缩到 latent space (1024 B/layer)，h^(0) 价值有限。

### 10.2 Checkpoint 数量 vs 压缩比 (通用公式)

```
压缩比(k个checkpoint) = Full_KV_per_tok / (k × h^(0)_per_tok)
                        = (L × kv_ptpl) / (k × d_model × 2)
                        = 2L × (n_kv / n_q) / k

其中 k = checkpoint 数量 (等间距放置)
```

| 模型 | 1-ckpt h^(0) | 2-ckpt | 4-ckpt | 8-ckpt | 最小 k_min |
|------|-------------|--------|--------|--------|-----------|
| Qwen3-0.6B | 28× | 14× | 7× | 3.5× | 2 |
| Qwen3-1.7B | 28× | 14× | 7× | 3.5× | 2 |
| **Qwen3-8B** | **18×** | **9×** | **4.5×** | **2.25×** | 3 |
| Llama-3.1-8B | 16× | 8× | 4× | 2× | 3 |
| Mistral-7B | 16× | 8× | 4× | 2× | 3 |
| Gemma-4-E4B | 17× | 8.4× | 4.2× | 2.1× | 3 |
| **Llama-70B** | **20×** | **10×** | **5×** | **2.5×** | 5 |
| **Gemma-4-31B** | **91×** | **46×** | **23×** | **11×** | 2 |
| DeepSeek-V3 | 4.4× | 2.2× | 1.1× | ❌ | — |

> k_min = n_q/(2×n_kv) 的上取整 + 1。checkpoint 间隔 < k_min 时, h^(0) 比该段 KV 还大。
> DeepSeek-V3 8-ckpt = 1.1× (无意义)，2-ckpt 已 < 3× (不推荐)。

### 10.3 P→D 传输对比 (@128K, IB NDR 50 GB/s)

#### 传输量 (GB)

| 模型 | DistServe (Full KV) | 1-ckpt | 2-ckpt | 4-ckpt | 8-ckpt |
|------|-------------------|--------|--------|--------|--------|
| Qwen3-0.6B | 7.0 | 0.25 | 0.50 | 1.0 | 2.0 |
| Qwen3-1.7B | 14.0 | 0.50 | 1.0 | 2.0 | 4.0 |
| **Qwen3-8B** | **18.0** | **1.0** | **2.0** | **4.0** | **8.0** |
| Llama-3.1-8B | 16.0 | 1.0 | 2.0 | 4.0 | 8.0 |
| **Llama-70B** | **40.0** | **2.0** | **4.0** | **8.0** | **16.0** |
| **Gemma-4-31B** | **62*** | **1.3** | **2.6** | **5.3** | **10.5** |

> *Gemma-4-31B: 修正为 hybrid attention 下 ~62 GB (非全量 120 GB)

#### IB NDR 传输时间 (ms)

| 模型 | DistServe | 1-ckpt | 2-ckpt | 4-ckpt | 8-ckpt |
|------|----------|--------|--------|--------|--------|
| Qwen3-0.6B | 140 ms | 5 | 10 | 20 | 40 |
| Qwen3-1.7B | 280 | 10 | 20 | 40 | 80 |
| **Qwen3-8B** | **360** | **20** | **40** | **80** | **160** |
| Llama-3.1-8B | 320 | 20 | 40 | 80 | 160 |
| **Llama-70B** | **800** | **40** | **80** | **160** | **320** |
| **Gemma-4-31B** | **1,240** | **26** | **53** | **105** | **210** |

**关键洞察**: 即使使用 8-ckpt (最多 checkpoint), 传输时间仍远小于 DistServe:
- Qwen3-8B 8-ckpt: 160ms vs DistServe 360ms → 仍有 2.25× 优势
- 但压缩比降到 2.25×, session 密度优势大幅缩水

### 10.4 Session 密度 (@128K, 20% active, R100 288 GB)

**公式**: `Sessions = Available_HBM / (0.2 × Full_KV + 0.8 × Ckpt_Size)`

| 模型 | Available | DistServe | 1-ckpt | 2-ckpt | 4-ckpt | 8-ckpt |
|------|-----------|----------|--------|--------|--------|--------|
| Qwen3-0.6B | 287 GB | 41 | **179** (4.4×) | 159 (3.9×) | 130 (3.2×) | 95 (2.3×) |
| Qwen3-1.7B | 285 GB | 20 | **89** (4.4×) | 79 (3.9×) | 65 (3.2×) | 47 (2.4×) |
| **Qwen3-8B** | **272 GB** | **15** | **62** (4.1×) | 52 (3.5×) | 40 (2.7×) | 27 (1.8×) |
| Llama-3.1-8B | 272 GB | 17 | **68** (4.0×) | 57 (3.3×) | 42 (2.5×) | 29 (1.7×) |
| Mistral-7B | 273 GB | 17 | **68** (4.0×) | 57 (3.3×) | 43 (2.5×) | 29 (1.7×) |
| **Llama-70B†** | **436 GB** | **11** | **45** (4.1×) | 39 (3.5×) | 30 (2.7×) | 21 (1.9×) |
| **Gemma-31B** | **226 GB** | **4*** | **17** (4.3×) | 15 (3.8×) | 14 (3.5×) | 11 (2.8×) |

> † Llama-70B: 2× R100 TP, 576 GB - 140 GB = 436 GB
> * Gemma-31B: hybrid attention @128K, effective full KV ≈ 62 GB

**规律**:
1. GQA 2:1 模型 (Qwen 0.6B/1.7B) 的 1-ckpt 放大最高 (4.4×)
2. GQA 4:1 模型 (8B class) 约 4.0-4.1×
3. GQA 8:1 模型 (70B) 同样 4.1× — **压缩比受 GQA 影响, 但 session 放大率趋同**
4. 每增加 1 倍 checkpoint 数量, session 放大衰减 ~15-25%
5. 8-ckpt 时所有模型仍有 1.7-2.4× 优势 (不会负收益)

### 10.5 KV 重建时间 on R100 @60% MFU

```
估算公式: FLOPs ≈ 2 × model_params × seq_len
R100 effective: 4,800 TFLOPS
Recon_time = FLOPs / 4,800 TFLOPS
k-ckpt parallel: Recon_time / k
```

#### @4K Context (单 D 节点)

| 模型 | FLOPs @4K | 1-ckpt | 2-ckpt parallel | 4-ckpt parallel | 8-ckpt parallel |
|------|----------|--------|-----------------|-----------------|-----------------|
| Qwen3-0.6B | 4.9 T | **1.0 ms** | 0.5 ms | 0.25 ms | 0.13 ms |
| Qwen3-1.7B | 13.9 T | **2.9 ms** | 1.5 ms | 0.7 ms | 0.4 ms |
| **Qwen3-8B** | **63 T** | **13.1 ms** | 6.6 ms | 3.3 ms | 1.6 ms |
| Llama-3.1-8B | 65.5 T | **13.6 ms** | 6.8 ms | 3.4 ms | 1.7 ms |
| Mistral-7B | 59.9 T | **12.5 ms** | 6.2 ms | 3.1 ms | 1.6 ms |
| Gemma-4-E4B | 32.8 T | **6.8 ms** | 3.4 ms | 1.7 ms | 0.9 ms |
| **Llama-70B†** | **574 T** | **59.8 ms** | 29.9 ms | 15.0 ms | 7.5 ms |
| **Gemma-4-31B** | **254 T** | **52.9 ms** | 26.5 ms | 13.2 ms | 6.6 ms |

> † Llama-70B: 2× R100 TP, 单 D 节点 = 2 GPU 并行 forward → 59.8ms

#### @128K Context

长 context 下 prefill/recon FLOPs 超线性增长 (attention O(N²)):

| 模型 | 1-ckpt @128K | 4-ckpt @128K | 8-ckpt @128K |
|------|-------------|-------------|-------------|
| Qwen3-0.6B | ~50 ms | 12 ms | 6 ms |
| Qwen3-1.7B | ~150 ms | 38 ms | 19 ms |
| **Qwen3-8B** | **~700 ms** | 175 ms | 88 ms |
| Llama-3.1-8B | ~600 ms | 150 ms | 75 ms |
| **Llama-70B†** | **~3.2 s** | 800 ms | 400 ms |
| **Gemma-4-31B** | **~2.8 s** | 700 ms | 350 ms |

> @128K 估算: @4K_time × (128/4)^1.3 (attention quadratic + linear component)
> 这些是 R100 @60% MFU 的投影, 实际可能 ±30%

**关键观察**:
1. **小模型 (0.6B-1.7B)**: 重建代价在 R100 上接近 0 → 1-ckpt 即可, 无需 sparse
2. **中模型 (7-8B)**: 1-ckpt ≈ 13ms @4K (可接受), @128K ~700ms → 建议 2-4 ckpt
3. **大模型 (31-70B)**: 1-ckpt @4K = 53-60ms → 建议 4-8 ckpt; @128K 超过 1s → 必须 sparse

### 10.6 Checkpoint 策略决策矩阵

```
                     压缩比高 ←─────────────→ 重建快
                     Session多                 TTFT低

  ┌─────────────────────────────────────────────────────────┐
  │         1-ckpt      2-ckpt      4-ckpt      8-ckpt     │
  │         h^(0)      +h^(mid)   +h^(q1,q3)   +more      │
  ├─────────────────────────────────────────────────────────┤
  │ 0.6B    ★最优      过度                                 │
  │ 1.7B    ★最优      可选                                 │
  │ 8B      ★推荐      平衡★      可选                      │
  │ L-8B    ★推荐      平衡★                                │
  │ 7B      ★推荐      平衡★                                │
  │ E4B     ★推荐      ★最优                                │
  │ 70B     保守       ★推荐      ★最优       可选          │
  │ 31B     保守       可选       ★推荐       ★最优        │
  │ DS-V3   勉强       ❌         ❌          ❌            │
  └─────────────────────────────────────────────────────────┘

  ★最优: 该模型的最佳性价比点
  ★推荐: 合理选择
  保守: 可用但 recon 偏慢
  可选: 牺牲压缩换速度
  ❌: 压缩比 < 3×, 不值得
```

### 10.7 三类模型的最优策略 (R100)

#### 类型 A: 小模型 (< 3B) — 1-ckpt, 不需要 sparse

```
代表: Qwen3-0.6B, Qwen3-1.7B

• Recon @4K = 1-3 ms → 几乎免费
• Recon @128K = 50-150 ms → 仍可接受
• 压缩比 28× → session 密度 4.4×
• 结论: h^(0) only 即可, sparse 是过度工程
```

#### 类型 B: 中模型 (7-8B) — 1-2 ckpt, 按 SLA 选

```
代表: Qwen3-8B, Llama-3.1-8B, Mistral-7B

• 1-ckpt: recon @4K = 13ms (可接受), @128K = 700ms (SLA > 1s 可接受)
• 2-ckpt: recon @4K = 6.6ms, @128K = 350ms, 压缩 9×
• 结论:
  - SLA > 500ms (chatbot): 1-ckpt 即可
  - SLA < 200ms (实时): 2-ckpt
  - @128K 场景: 2-4 ckpt 推荐
```

#### 类型 C: 大模型 (30-70B) — 4-8 ckpt, 必须 sparse

```
代表: Llama-3.1-70B, Gemma-4-31B

• 1-ckpt: recon @4K = 53-60ms (可接受), @128K > 2.5s (太慢!)
• 4-ckpt: recon @4K = 13-15ms, @128K = 700-800ms, 压缩仍 5-23×
• 8-ckpt: recon @128K = 350-400ms, 压缩仍 2.5-11×
• 结论:
  - 4K context: 4-ckpt 推荐 (recon < 15ms, 压缩 5-23×)
  - 128K context: 8-ckpt (recon < 400ms, 但 70B 压缩仅 2.5×)
  - Gemma-4-31B 即使 8-ckpt 仍有 11× 压缩 → S-tier 优势!
```

### 10.8 Gemma-4-31B: h^(0) 的理想目标

```
为什么 Gemma-4-31B 是 h^(0) 的 "杀手级应用":

1. 压缩比 91× (S-tier) — 即使 8-ckpt 仍有 11×
2. KV 爆炸最严重 — @128K Full KV = 62 GB (hybrid), 几乎吃掉整张 R100
3. DistServe @128K 只能放 4 个 session → h^(0) 放 17 个 (4.3×)
4. P→D DistServe @128K = 1.24s → h^(0) 4-ckpt = 105ms (12×!)

原因: GQA 2:1 + 60 layers + head_dim=256 → 每层 KV 极大 (16,384 B)
      而 d_model=5,376 相对不大 → h^(0) 极小 (10,752 B)
      比率 = 60 × 16,384 / 10,752 = 91.4×

对比 DeepSeek-V3 (h^(0) 的 "天敌"):
  MLA 已将 KV 压到 1,024 B/layer → 61L × 1,024 = 62 KB/tok
  h^(0) = 14,336 B/tok → 压缩比仅 4.4× → 不值得
```

### 10.9 跨模型 Rubin R100 完整对比 (@128K, 20% active, 最优策略)

| 模型 | 最优策略 | 压缩 | DistServe Sessions | h^(0) Sessions | 放大 | Recon @4K | P→D @128K IB |
|------|---------|------|-------------------|---------------|------|----------|-------------|
| Qwen3-0.6B | 1-ckpt | 28× | 41 | **179** | **4.4×** | 1.0 ms | 5 ms |
| Qwen3-1.7B | 1-ckpt | 28× | 20 | **89** | **4.4×** | 2.9 ms | 10 ms |
| Qwen3-8B | 1-ckpt | 18× | 15 | **62** | **4.1×** | 13.1 ms | 20 ms |
| Llama-3.1-8B | 2-ckpt | 8× | 17 | **57** | **3.3×** | 6.8 ms | 40 ms |
| Mistral-7B | 1-ckpt | 16× | 17 | **68** | **4.0×** | 12.5 ms | 20 ms |
| Gemma-4-E4B | 2-ckpt | 8.4× | 27 | **84** | **3.1×** | 3.4 ms | 25 ms |
| Llama-70B | 4-ckpt | 5× | 11 | **30** | **2.7×** | 15.0 ms | 160 ms |
| **Gemma-4-31B** | **4-ckpt** | **23×** | **4** | **14** | **3.5×** | 13.2 ms | 105 ms |
| DeepSeek-V3 | 1-ckpt | 4.4× | — | — | ~1.5× | — | — |

### 10.10 GQA Ratio 是决定性因素

```
h^(0) 压缩比 = 2L × (n_kv / n_q)

GQA ratio (n_kv/n_q):
  MHA 1:1   → 压缩 = 2L          (Gemma-4-31B: 2×60=120... 但 n_kv≠n_q)
  GQA 2:1   → 压缩 = L           (Qwen 0.6B/1.7B: 28×)
  GQA 4:1   → 压缩 = L/2         (Qwen-8B: 18×, Llama-8B: 16×)
  GQA 8:1   → 压缩 = L/4         (Llama-70B: 20×, 靠 80 层补偿)
  MLA       → 压缩 ≈ 小           (DeepSeek-V3: 4.4×, KV 已被 MLA 压过)

规律:
  • GQA 比例越接近 1:1 → h^(0) 价值越大 (KV 每层大, h^(0) 不变)
  • 层数越多 → h^(0) 价值越大 (跨更多层分摊)
  • MLA 是 h^(0) 的天敌 (KV 已被压缩, h^(0) 无法进一步缩减)

最佳目标: GQA ≤ 2:1, L > 40 → Gemma-4-31B (91×), Qwen3 系列 (28×)
最差目标: MLA → DeepSeek 系列 (< 5×)
```

---

## 11. 跨模型关键结论

1. **所有 GQA 模型都受益于 h^(0)** — session 密度放大 1.7-4.4× (视 checkpoint 数量)
2. **GQA 2:1 是甜蜜点** — Qwen3 系列 28×, Gemma-4-31B 91× (S-tier)
3. **MLA 模型不适合** — DeepSeek-V3 仅 4.4×, KV 已被 MLA 压过
4. **小模型不需要 sparse** — 1-ckpt 重建 < 3ms on R100
5. **大模型必须 sparse** — 70B 1-ckpt @128K > 3s, 4-ckpt 降到 800ms
6. **Gemma-4-31B 是杀手级目标** — 即使 8-ckpt 仍有 11× 压缩
7. **传输优势随 checkpoint 数量线性衰减** — 但到 8-ckpt 仍比 DistServe 快 2-12×

---

## 12. 关键数字速查卡

```
┌──────────────────────────────────────────────────────────────────┐
│  h^(0) on NVIDIA Rubin R100 — Qwen3-8B bf16                     │
│                                                                  │
│  硬件                                                            │
│  ├── R100: 8000 TFLOPS bf16, 288GB HBM4, 22TB/s                │
│  ├── Ridge Point: 363.6 FLOPs/Byte                              │
│  └── Prefill: compute-bound, Decode: BW-bound → PD 分离有效     │
│                                                                  │
│  Prefill                                                         │
│  ├── @4K: 13.1ms (76 req/s) on R100 @60% MFU                   │
│  ├── FLOPs: 63 TFLOP (实测 + 架构验证)                          │
│  └── vs M4 Pro: 823× faster                                     │
│                                                                  │
│  Decode                                                          │
│  ├── @4K batch=472: 76.3 tok/s/user, 36K tok/s total            │
│  ├── Active decode 速度 = DistServe (无退化)                     │
│  └── AI gap: 4,689× (compute 98.4% 闲置)                        │
│                                                                  │
│  h^(0) 重建                                                      │
│  ├── @4K 单节点: 13.1ms (vs M4 Pro 11s → 843× 加速)            │
│  ├── @4K 4D 并行: 3.4ms                                         │
│  └── TTFT 增量: +13ms (1 个 decode step, 用户无感)              │
│                                                                  │
│  传输 (IB NDR 50GB/s)                                            │
│  ├── @4K:   DistServe 11.5ms → h^(0) 0.64ms (18×)              │
│  ├── @128K: DistServe 368ms  → h^(0) 20.5ms  (18×)             │
│  └── IB 饱和: DistServe @128K → 饱和! h^(0) → 永远不饱和        │
│                                                                  │
│  Session 密度 (@128K, 20% active)                                │
│  ├── DistServe: 14 sessions per R100                             │
│  ├── h^(0): 60 sessions per R100 (4.3×)                         │
│  └── 1000 sessions: 68 R100 → 17 R100 (节省 51 张!)             │
│                                                                  │
│  跨机架部署                                                       │
│  ├── DistServe: P/D 必须 NVLink 直连 (同机架)                   │
│  └── h^(0): IB NDR 足够, P/D 可跨机架自由调度                   │
│                                                                  │
│  质量                                                            │
│  └── 所有场景: EXACT (bit-identical output)                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 13. 投影假设与风险

| 假设 | 依据 | 风险 |
|------|------|------|
| R100 bf16 ~8000 TFLOPS | Blackwell 2250 × ~3.5× | ±20%, NVIDIA 未公布确切值 |
| R100 HBM4 22 TB/s | 公开 spec | 较确定 |
| R100 288 GB | 公开 spec | 确定 |
| MFU 60% prefill | H100 实测 50-70% | 保守端, 实际可能更高 |
| Decode BW utilization ~70% | M4 Pro 实测 46%, GPU 通常更高 | 可能影响 batch throughput |
| 20% active rate | 典型 chatbot 使用模式 | 因应用而异 |
| h^(0) 重建可利用 idle compute | 理论可行, 需 CUDA stream 并行 | 实现复杂度待验证 |
| 63 TFLOP @4K | M4 Pro 实测 + 架构计算双重验证 | 较确定 |

### 最大不确定性

1. **R100 实际 TFLOPS**: NVIDIA 尚未公布 bf16 确切值, 8000 为合理估计
2. **MFU**: 实际 prefill MFU 可能 50-75%, 报告用 60%
3. **Decode + Recon overlap**: GPU 能否真正在 decode 同时重建? 需要 CUDA stream 级别验证

---

## 14. 结论

h^(0) Residual Checkpoint 在 Rubin R100 上的三大支柱:

**支柱 1: 传输压缩 18×** — 将 P→D 通信从 "IB 瓶颈" 变成 "IB 闲置"。
使得 PD 分离可以跨机架部署, 彻底解除拓扑约束。

**支柱 2: Session 密度 4.3×** — 在 128K 长 context 下,
同一张 R100 可服务 60 个 session (vs DistServe 的 14 个), 节省 76% HBM 成本。

**支柱 3: 重建代价趋零** — R100 的 8000 TFLOPS 使得
4K context 的完整 KV 重建仅需 13ms (vs M4 Pro 的 11 秒)。
用户 TTFT 增加不到 1 个 decode step, 几乎无感知。

**代价**: Cold start TTFT 增加 ~13ms (@4K)。对于 SLA > 100ms 的场景完全可接受。

**底线**: 相同的 16 R100 集群, 服务 1000 个 128K session,
DistServe 需要 68 张 D 卡, h^(0) 只需 17 张。**节省 51 张 R100 (75%)**。

---

*基于 FlashMLX 实测数据 (M4 Pro) + NVIDIA Rubin R100 公开/估计规格*
*模型: Qwen3-8B bf16 (36L, d=4096, GQA 4:1, h^(0) compression 18×)*
*投影模型: Llama-3.1-70B bf16 (80L, d=8192, GQA 8:1, h^(0) compression 20×)*
*日期: 2026-04-03*
