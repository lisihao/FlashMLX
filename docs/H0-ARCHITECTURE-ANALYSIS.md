# h^(0) 残差流架构分析 — 从实验到产业级推理

> 基于 FlashMLX dual-instance h^(0) 实验数据 + NVIDIA Groq 3 LPU/Vera Rubin AFD 架构分析
> 日期: 2026-04-03

## 1. 实验背景

FlashMLX 实现了 dual-instance h^(0) disaggregated inference:
- **Instance A**: 生产 h^(0) (embed_tokens 输出)
- **Instance B**: 接收 h^(0)，重建 KV cache，解码生成

5 项优化已实现并验证 (commits `254d9ce`, `e186eaa`, `9dea43b`):
- OPT-0: Embed-Only A (686× 加速)
- OPT-1: h^(0) 持久化缓存
- OPT-2: 增量追加 (多轮对话)
- OPT-3: 一对多扇出
- OPT-4: 流水线化

---

## 2. Benchmark 数据

### 2.1 Qwen3-1.7B-MLX-4bit (28L, 16Q/8KV, d=2048, M4 Max)

| N tokens | Trad Prefill | B Recon | Recon/PP | ΔTTFT | Trad TG | B TG | ΔTG | Trad Peak | B Peak | ΔMem |
|----------|-------------|---------|----------|-------|---------|------|-----|-----------|--------|------|
| 1,024 | 602ms | 492ms | 82% | +17% | 157 t/s | 156 t/s | 0% | 1973MB | 1668MB | -15% |
| 2,048 | 1185ms | 958ms | 81% | +18% | 144 t/s | 147 t/s | 0% | 2293MB | 1847MB | -19% |
| 2,751 | 2093ms | 1348ms | 84% | +15% | 140 t/s | 139 t/s | 0% | 2767MB | 1822MB | -34% |

- h^(0) 压缩比: **28×**
- embed_tokens 权重: 158 MB (18.1%)
- 有效吞吐: 传统 5.0-5.4 TFLOP/s, B 6.1-6.5 TFLOP/s

### 2.2 Qwen3-8B-MLX-4bit (36L, 32Q/8KV, d=4096, M4 Max)

| N tokens | Trad Prefill | B Recon | Recon/PP | ΔTTFT | Trad TG | B TG | ΔTG | Trad Peak | B Peak | ΔMem |
|----------|-------------|---------|----------|-------|---------|------|-----|-----------|--------|------|
| 1,024 | 2468ms | 2227ms | 90% | +5% | 45 t/s | 46 t/s | 0% | 5373MB | 5122MB | -5% |
| 2,048 | 4953ms | 4503ms | 91% | +7% | 43 t/s | 42 t/s | 0% | 5860MB | 5186MB | -11% |
| 2,751 | 6724ms | 6310ms | 94% | +5% | 40 t/s | 40 t/s | 0% | 6200MB | 5235MB | -16% |

- h^(0) 压缩比: **18×**
- embed_tokens 权重: 334 MB (7.6%)
- 有效吞吐: 传统 5.9-6.0 TFLOP/s, B 6.4-6.6 TFLOP/s

### 2.3 1.7B vs 8B 对比规律

| 指标 | 1.7B (28L, d=2048) | 8B (36L, d=4096) | 趋势 |
|------|-------------------|-----------------|------|
| h^(0) 压缩比 | 28× | 18× | 层数越多，KV 越大，压缩比下降 |
| Recon vs Prefill | 81-84% | 90-94% | 模型越大，FFN 占比越高，chunk attention 优势缩小 |
| TTFT 提升 | +15-18% | +5-7% | 同上 |
| 峰值内存节省 | -15% ~ -34% | -5% ~ -16% | 模型权重占主导后，activation 节省占比缩小 |
| TG 速度回归 | 0% | 0% | 零回归，两者 KV cache 结构相同 |

---

## 3. 关键发现: B 的 Recon 比传统 Prefill 更快

### 3.1 原因分析

理论 FLOPs 相同，但 B 的 chunked reconstruction 比传统 unchunked prefill 快 6-19%。

**根因: Attention Score 矩阵的内存压力**

传统 prefill (unchunked):
```
Attention scores = Q @ K^T → shape: (n_q_heads, N, N)
1.7B, 2.7K tokens: 16 × 2751 × 2751 × 4B = 460 MB 峰值/层
```

B reconstruction (chunked, chunk_size=512):
```
Attention scores = Q_chunk @ K^T → shape: (n_q_heads, 512, N)
1.7B, 2.7K tokens: 16 × 512 × 2751 × 4B = 86 MB 峰值/层
```

- 5.4× 更小的 attention 矩阵 → Metal GPU cache 命中率大幅提升
- 1.7B: attention 占总 FLOPs 比例较高 → chunk 效果显著 (18% 加速)
- 8B: FFN 占总 FLOPs 更大比例 → chunk 效果被稀释 (6-10% 加速)

### 3.2 峰值内存节省

| N tokens | 传统峰值 | B 峰值 | 差值 | 主因 |
|----------|---------|--------|------|------|
| 1K (1.7B) | 1973MB | 1668MB | -305MB | unchunked attn ~120MB + embed 158MB |
| 2.7K (1.7B) | 2767MB | 1822MB | -945MB | unchunked attn ~820MB 被 chunk 消解 |
| 2.7K (8B) | 6200MB | 5235MB | -965MB | 模型权重 4.4GB 占主导，attention 节省占比小 |

传统 prefill 峰值含 **O(N²) attention 激活**，chunked recon 控制在 O(chunk × N)。

---

## 4. h^(0) 的本质局限: 计算负担在错误的一端

### 4.1 embed_tokens 的真实开销

| 模型 | embed 权重 | 占全模型 | embed 耗时 | 占 prefill 比例 |
|------|-----------|---------|-----------|---------------|
| 1.7B | 158 MB | 18.1% | 0.3ms | 0.05% |
| 8B | 334 MB | 7.6% | ~0.5ms | 0.008% |

**embed_tokens 是查表操作，接近零 FLOPs。** h^(0) 架构本质上做的是:

```
Server (A):  0.008% 的活 (embed 查表)
Client (B):  99.992% 的活 (36 层 forward + decode)
```

### 4.2 为什么不适合 Client-Server 分离

目标: Server 做重活 (prefill)，Client 做轻活 (decode only)。

```
h^(0) 架构:
  Server → h^(0) (32MB, 小) → Client
  Client: 收到 h^(0) → 重建 KV (= 跑完整 36 层 forward) → 解码
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        这就是 prefill! 客户端根本没减负
```

Client 仍需:
- 完整模型权重 (4.4 GB for 8B)
- 完整 prefill 计算量 (14.5 TFLOP for 8B @ 4K)
- O(N²) attention 峰值内存

### 4.3 正确的 Client-Server 分离: 传 KV Cache

```
正确架构:
  Server: 完整 prefill → KV cache → 压缩 → 传给 Client
  Client: 收到 KV → 解压 → 注入 cache → 只跑 TG (零 prefill!)
```

Client TTFT = KV 传输时间 + 1 次 forward (~25ms)。

---

## 5. KV 传输 vs 残差流传输 — 数学对比

### 5.1 per-token 数据量 (Qwen3-8B)

```
残差流 h^(l):    d_model × 2 bytes        = 4096 × 2 = 8,192 bytes/token/layer
KV cache:       2 × kv_heads × head_dim × 2 = 2 × 8 × 128 × 2 = 4,096 bytes/token/layer
```

**KV 是残差流经 W_K/W_V 投影后的低维表示 (4096→1024)。KV 每层比残差流小 50%。**

### 5.2 五种传输方案对比

| 方案 | 传什么 | bytes/token | 4K 总量 | 客户端计算 | 客户端权重 |
|------|--------|------------|---------|-----------|-----------|
| A: 传 KV | 所有层 K+V | 147,456 | **576 MB** | 零 | 4.4 GB (TG) |
| B: 传 h^(0) | 初始残差 | 8,192 | **32 MB** | 全量重建 4.5s | 4.4 GB (全模型) |
| C: 传每层残差 | h^(0)..h^(35) | 294,912 | **1,152 MB** | 72 matmuls ~120ms | 600 MB (W_K,W_V) + 4.4 GB (TG) |
| D: 低秩残差 | PCA(h^(l), r=256) | 18,432 | **72 MB** | 解压+投影 ~200ms | 72 MB 基矩阵 + 4.4 GB (TG) |
| A': 传 int4 KV | 量化 KV | ~36,864 | **144 MB** | ~10ms 解压 | 4.4 GB (TG) |

### 5.3 方案 E 不可行的数学原因

"只用 h^(0) + W_K/W_V 得到所有层 KV" — 数学上不可能:

```python
K_1 = norm(h^(0)) @ W_K_1    # ✅ 第 1 层可以
K_2 = norm(h^(1)) @ W_K_2    # ❌ 需要 h^(1)
# h^(1) = h^(0) + Attn_1(h^(0)) + FFN_1(h^(0))
#                 需要 W_Q, W_K, W_V, W_O + softmax + 完整 FFN 权重
```

每层残差依赖上一层的完整计算结果。无法跳过中间层。

### 5.4 传输带宽需求 (4K context)

| 网络 | Raw KV (576MB) | int4 KV (144MB) | h^(0) (32MB) |
|------|---------------|-----------------|--------------|
| WiFi 100Mbps | 47s | 12s | 2.6s |
| 5G 1Gbps | 4.7s | **1.2s** | 0.3s |
| 局域网 10Gbps | 0.5s | 0.1s | 0.03s |
| Thunderbolt 40Gbps | 0.12s | 0.03s | 0.007s |

**5G + int4 KV: 4K context 1.2 秒传完，Client 零 prefill。**

---

## 6. NVIDIA AFD 架构: 正确的 GPU+LPU 切分

### 6.1 不是按 Phase 切，是按 Layer Component 切

NVIDIA Groq 3 LPU + Vera Rubin GPU 采用 **AFD (Attention-Feed Forward Disaggregation)**:

```
Phase Split (h^(0) approach — 不 work):
  GPU: Prefill ──h^(0)──▶ LPU: Decode (需全模型+KV, SRAM装不下)

AFD (NVIDIA approach — 正确):
  GPU: Attention (有 HBM 存 KV)  ←──residual──▶  LPU: FFN (SRAM 存权重, 无 KV)
```

### 6.2 AFD 数据流

```
每层每 token:
  h^(l-1) ──▶ [GPU: Attention + KV cache] ──▶ h' ──▶ [LPU: FFN] ──▶ h^(l)
                KV stays in HBM                      FFN weights in SRAM
```

| | GPU (Attention) | LPU (FFN) |
|---|---|---|
| **存什么** | Attn 权重 + KV cache | FFN/MoE 权重 |
| **随 context 增长** | KV 线性增长 (HBM 扩容) | **不变** |
| **SRAM 需求** | N/A (用 HBM) | 模型相关, 上下文无关 |

### 6.3 硬件规格 (Groq 3 LPU)

| 指标 | Groq 3 LPU (per chip) | Vera Rubin GPU |
|------|----------------------|----------------|
| 内存类型 | SRAM 500MB | HBM |
| 带宽 | 150 TB/s | 22 TB/s |
| 机架级 | 128 GB SRAM, 40 PB/s | 大容量 HBM |
| 芯片间互联 | 640 TB/s (机架级) | NVLink |

### 6.4 为什么 LPU 永远不需要 KV

AFD 的核心设计:
- **KV cache 是 context-dependent** → 留在 GPU (HBM 可扩展)
- **FFN 权重是 context-independent** → 放在 LPU (SRAM 固定大小)
- **Context 增长只影响 GPU 侧，LPU 完全不变**

```
8B 模型:
  LPU 需要: FFN 权重 ~3.6 GB (固定, 用 8 chips × 500MB SRAM)
  GPU 需要: Attn 权重 ~300 MB + KV cache (随 context 增长)

4K context:   GPU KV = 576 MB    | LPU: 不变
128K context: GPU KV = 18.4 GB   | LPU: 不变
1M context:   GPU KV = 144 GB    | LPU: 还是不变
```

### 6.5 GPU↔LPU 间传输的就是残差流

每层每 token:
- 传输量: d_model × 2 bytes = 8,192 bytes (8B 模型)
- Decode 总量: 8KB × 2 方向 × 36 层 = 576 KB/token
- 640 TB/s 互联: 576 KB → ~0.001 μs

带宽完全不是瓶颈，延迟是考量因素 (36 次 round-trip/token)。

---

## 7. h^(0) 在 AFD 架构中的真正价值

h^(0) 不是 GPU→LPU 的桥梁，而是 **GPU 侧的 KV 内存优化**:

```
AFD 架构 + h^(0) 增强:

GPU 侧 (HBM):
  ┌──────────────────────────────────────────┐
  │ Attention 引擎                             │
  │                                          │
  │ 热 KV: 最近 W tokens (窗口, 576 MB)       │  ← HBM, 直接用
  │ 温 h^(0): 历史 tokens (32 MB/4K)          │  ← 按需重建 KV
  │ 冷 h^(0): 磁盘缓存 (.npz)                 │  ← 重复 prompt 秒加载
  │                                          │
  │ 128K context 内存对比:                     │
  │   传统 KV:     18.4 GB  (可能 OOM)         │
  │   h^(0)+窗口:  1 GB + 576 MB = 1.6 GB     │  ← 11.5× 省内存
  └──────────┬──────────────────┬─────────────┘
             │ residual (8KB)   │
             ▼                  ▲
  ┌──────────────────────────────────────────┐
  │ LPU: FFN 引擎 (SRAM, 500MB)               │
  │ 只存当前层 FFN 权重，逐层流式               │
  │ 完全不关心 context 长度                     │
  └──────────────────────────────────────────┘
```

### 7.1 GPU 侧 h^(0) 三大用途

| 用途 | 机制 | 收益 |
|------|------|------|
| **KV 内存压缩** | 存 h^(0) (8KB/tok) 替代 KV (144KB/tok) | 128K ctx: 18.4 GB → 1.6 GB |
| **Prompt 缓存** | 相同 prompt → 相同 h^(0), 磁盘持久化 | 重复请求跳过全部 prefill |
| **Fan-out** | 1 个 h^(0) → N 个 decode session | Prefill 成本 N× 摊薄 |

### 7.2 长上下文 KV 内存对比

| Context | 传统 KV (8B) | h^(0) + 4K 窗口 KV | 节省 |
|---------|-------------|-------------------|------|
| 4K | 576 MB | 608 MB | 0 (窗口覆盖全部) |
| 32K | 4.5 GB | 832 MB | **5.4×** |
| 128K | 18.4 GB | 1.6 GB | **11.5×** |
| 1M | 144 GB | 8.6 GB | **16.8×** |

---

## 8. 结论

### 8.1 h^(0) 的正确定位

| 定位 | 可行? | 说明 |
|------|-------|------|
| Client-Server prefill/decode 分离 | ❌ | Client 仍需全模型 + 全量重建 = 没减负 |
| GPU→LPU 桥梁 (Phase Split) | ❌ | LPU SRAM 装不下模型权重 + KV |
| **Server/GPU 端 KV 内存管理** | ✅ | 18× 压缩, 按需重建, 缓存, fan-out |
| **AFD 架构的 GPU 侧增强** | ✅ | 长上下文 HBM 节省, prompt 缓存 |

### 8.2 产业级推理架构启示

1. **NVIDIA 的答案**: 不是"怎么把 KV 塞进 SRAM"，而是"别把 KV 给 SRAM 设备"
2. **AFD > Phase Split**: 按 Attention/FFN 切比按 Prefill/Decode 切更优
3. **h^(0) 的价值在 server 端**: KV 内存压缩 (11-17×) + prompt 缓存 + fan-out
4. **Client 要轻量**: 传 int4 量化 KV (5G 下 4K ctx 1.2s)，不传 h^(0)

### 8.3 FlashMLX h^(0) 实验的贡献

| 发现 | 意义 |
|------|------|
| embed_tokens ≡ h^(0), bit-exact | 证明 h^(0) 作为 KV 压缩格式的可行性 |
| h^(0) 18-28× 压缩比 | Server 侧 KV 内存管理的理论基础 |
| Chunked recon 比 unchunked prefill 快 | 非直觉优化: O(chunk×N) vs O(N²) attention peak |
| TG 零回归 | h^(0) 重建的 KV 功能完全等价 |
| POSIX SHM 传输 < 6ms | 进程间 h^(0) 传输可忽略 |

---

## 参考资料

- [Inside NVIDIA Groq 3 LPX (NVIDIA Blog)](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/)
- [GTC 2026: SRAM-Decode Implications](https://www.viksnewsletter.com/p/gtc-2026-preview-implications-of-sram-decode)
- [Beyond GTC: Compute, LPX, and SpecDec](https://www.viksnewsletter.com/p/beyond-gtc-a-deep-dive-into-compute-lpx)
- [NVIDIA Groq 3 LPU — IEEE Spectrum](https://spectrum.ieee.org/nvidia-groq-3)
- [NVIDIA Groq KV Cache Context Extension](https://blocksandfiles.com/2026/01/07/nvidia-groq-kv-cache-context-memory-extension-and-phisons-edge-inferencing-sw/)
- FlashMLX experiments: `experiments/dual_instance_h0/`
- Benchmark script: `experiments/dual_instance_h0/bench_b_vs_traditional.py`
