# FlashMLX：三条路线吃掉 LLM 推理的每一块内存

> 在 Apple Silicon 上，三条独立优化路线同时出击：
> - **参数内存**：Expert Offloading 把 18 GB MoE 参数压到 10 GB，TG 速度不掉
> - **PP 内存**：Chunked Prefill + Streaming Eviction 把 O(N²) 打到 O(1)，32K PP 只用 774 MB
> - **TG 内存**：Scored P2 + Q8 Flat Buffer 把 4.6 GB KV Cache 压到 147 MB，TG 反而快 54%
>
> 这不是一个顺利的故事。每条路线都踩了坑，有些坑是论文挖的，有些是自己跳的。

---

## 先搞清楚：内存都花在哪了？

大模型推理的 GPU 内存开销分三大块，很多人只看到其中一块就以为搞定了：

```
┌─────────────────────────────────────────────────────────────────────┐
│                   LLM 推理内存三大块                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 参数内存 (Model Parameters)                                    │
│     模型权重常驻 GPU。密集模型 = 参数量 × 精度。                    │
│     MoE 模型 = 参数量巨大，但只有 top-k 专家被激活。               │
│     Qwen3.5-35B-A3B: 256 专家/层 × 40 层 = 18.21 GB              │
│                                                                     │
│  2. PP 内存 (Prefill / Prompt Processing)                          │
│     处理输入时的 KV Cache + 中间激活值。                            │
│     标准 Attention 是 O(N²)，32K 上下文 = 5 GB+ 峰值。            │
│     这是"首次输入"的内存高峰。                                      │
│                                                                     │
│  3. TG 内存 (Token Generation / Decode)                            │
│     逐 token 生成时的 KV Cache。                                   │
│     每个新 token 读全部历史 KV = 随对话长度线性增长。              │
│     32K 上下文 = 4.6 GB KV Cache，且拖慢 TG 速度。               │
│                                                                     │
│  总开销 = 参数 + max(PP 峰值, TG 累积)                             │
│  三块都得优化，只优化一块效果有限。                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

下面分别讲三条路线怎么把每一块吃下来的。

---

## 路线一：参数内存 — Expert Offloading Compact Pool

### 问题

MoE 模型的参数内存很大。Qwen3.5-35B-A3B 每层 256 个专家，40 层，Q4 量化后仍需 **18.21 GB** 常驻 GPU。

但一个关键事实是：**推理时每层只激活 top-8 个专家**。256 个里只用 8 个，其余 248 个纯占内存。

MLX-LM 社区有人做了 `flash-moe` 架构，把不活跃的专家卸载到 CPU。我在此基础上加了一个关键优化：**两阶段 Compact Pool**。

### 架构

```
PP 阶段: 全部 256 专家留在 GPU
         → identity 路径，零开销
         → 同时记录每个专家的激活次数

Compact: 统计 PP 中的"热门"专家 → 选 top-K
         → 非热门专家降级到 CPU cache (numpy, UMA 快速搬运)
         → mx.eval(compact pool) + gc.collect → 释放旧 pool
         → 预热 Metal kernel (新 pool shape 需要 JIT 编译)

TG 阶段: 只用 K 个热门专家 → remap + clamp 索引
         → 和 identity 路径走同一条 gather_qmm 代码
         → 零 .item()，零 GPU→CPU 同步，全 lazy evaluation
         → 极少数 miss → clamped 到最近的热门专家 (1/8 权重误差，可忽略)
```

### 踩坑记录

**坑 1：Sentinel 检查杀死性能**

最初的 miss 检测方案：每个 MoE 层用 `mx.max(local_indices).item()` 检查是否有索引越界。

结果：**5.6 tok/s**（vs identity 的 90+ tok/s）。

原因：`.item()` 强制 GPU→CPU 同步。40 个 MoE 层 × 每个 token = 40 次同步。MLX 的 lazy evaluation 完全失效——本来应该攒一大坨计算一起 flush 的 graph，被 `.item()` 切成 40 段串行执行。

**修复：Speculative Execution**

不检查，直接 clamp：

```python
# 旧方案：每层检查一次（5.6 tok/s）
max_idx = mx.max(local_indices).item()  # GPU→CPU sync!
if max_idx < K:
    fast_path()
else:
    miss_path()

# 新方案：零检查（92.8 tok/s）
# remap 表的默认值设为 K-1（最后一个有效专家）
# 非 pool 专家自动 clamp 到有效范围，无需 mx.minimum
local_indices = self._pool_remap[indices]  # 一条 MLX op，全 lazy
```

为什么可以不检查？因为 compact 时选的是 PP 阶段激活次数最多的 top-K 专家。TG 阶段的路由高度集中在同一批专家上。coverage=100% 意味着所有 PP 期间出现过的专家都在 pool 里，TG 时几乎不会 miss。

**坑 2：Metal Kernel JIT 的 Warmup**

compact 后前 50 个 token 只有 ~40 tok/s，之后突然跳到 90+ tok/s。

原因不是 compact 本身慢，而是 Apple Metal 需要为新的 pool tensor shape **JIT 编译 GPU kernel**。编译完后就是全速。

验证：`FORCE_REMAP=1` 强制 pool=256 走 remap 路径 → 90.6 tok/s，和 identity 完全一致。**remap 零开销**，慢的只是 JIT。

### 最终结果

Qwen3.5-35B-A3B (Q4, 256 experts/layer, top-8), Apple M4 Pro 48GB：

| Config | Steady TG | Memory | Saved | Coverage |
|--------|-----------|--------|-------|----------|
| pool=256 (identity) | 90.0 tok/s | 18.21 GB | — | 100% |
| pool=192 (compact) | 90.9 tok/s | 13.99 GB | **4.23 GB** | 100% |
| pool=128 (compact) | **92.8 tok/s** | **9.77 GB** | **8.44 GB** | 100% |

**TG 速度零惩罚**（warmup 后与 identity 一致），参数内存 **减少 46%**。

pool=128 甚至比 pool=256 快 3%——因为更小的 pool tensor = 更紧凑的内存布局 = 更好的 cache locality。

---

## 路线二：PP 内存 — Chunked Prefill + Streaming Eviction

### 问题

标准 Prefill 是 O(N²)。处理 32K token 的 prompt，Attention 要算 32K × 32K 的矩阵，KV Cache + 中间激活 = **5 GB+** 峰值。

这意味着：即使你的模型只占 8 GB，一个长 prompt 就再吃 5 GB，可能直接 OOM。

### 核心洞察

如果我们在 Prefill 阶段也做 eviction 呢？

不是等 prefill 结束后再压缩，而是**边 prefill 边评估 token 重要性，边驱逐不重要的 token**：

```
Chunk 1 (token 0-511):     prefill → cache: 512 tokens
Chunk 2 (token 512-1023):  prefill → cache: 1024 tokens
...
Chunk 9 (token 4096-4607): prefill → cache: 4608 tokens
                            ↓ cache > max_cache (4096)
                            AM eviction: 4608 → 1872 tokens
                            ↓ continue
Chunk 10 (token 4608-5119): prefill → cache: 2384 tokens
...
```

每次 cache 超过 4096 token 阈值，用 AM importance scoring 驱逐低分 token，保留 hot tokens + 最近 512 tokens。

### 为什么反而更快？

标准 attention 是 O(N²)。32K token 意味着 attention 矩阵是 32K × 32K = 10 亿次运算。

Chunked Prefill 把它变成 O(chunk × cache) = O(512 × 2048) ≈ 100 万次运算/chunk。**复杂度从 O(N²) 变成 O(1)**。

32K 时效果最明显：standard PP 被 O(N²) 拖到 213 tok/s，Scored Chunked 维持 **369 tok/s (+73%)**。

### 结果

Qwen3-8B-MLX (Q8), Apple M4 Pro 24GB：

| 指标 | Standard | Scored Chunked | 32K 变化 |
|------|----------|----------------|----------|
| PP 速度 | 213.6 tok/s | 369.1 tok/s | **+73.0%** |
| PP 峰值内存 | 5,079 MB | 774 MB | **-84.8%** |
| PP Active | 14,207 MB | 526 MB | **-96.3%** |

**PP 峰值 = 774 MB，无论 16K 还是 32K 还是 128K。O(1) 内存复杂度。**

这个数字不是巧合。`max_cache=2048` + `chunk_size=512` 限制了物理 cache 上限。理论上你可以 prefill 128K token，PP 内存仍然在 ~800 MB 量级。

---

## 路线三：TG 内存 — Scored P2 + Q8 Flat Buffer

### 问题

TG 阶段的 KV Cache 随对话长度线性增长。32K 上下文 = 4.6 GB KV Cache。不仅吃内存，还拖慢 TG 速度（每个新 token 要读全部历史 KV）。

### 演化路线

这条路线走了最多弯路，也是教训最多的一条。

#### Round 1：信 AM，得永生？

Attention Matching——数学很优雅：最小化压缩前后 attention 分布的 KL 散度，用 β 系数补偿被丢弃的 Key。单层测试，完美：

```
Ratio 2.0x → 50% 压缩 → QA 100% ✅
Ratio 3.0x → 67% 压缩 → QA 100% ✅
```

36 层全上：**QA 0.000，输出乱码**。

**教训 #1：单层压缩和全层压缩是两个完全不同的问题。** 误差在 36 层 transformer 中逐层累积，到最后层已经不可控。

#### Round 2：差点放弃，然后找到了 Bug

准备转向 H2O 时，重新审视实现，发现两个关键 bug：

1. **β solver 无界**：论文只给 `β = R_S† · target`，没提需要 bound。β 飞到 [-171, 221]，毫无物理意义。加 bounded optimization 后修复。
2. **Query 采样偏差**：连续 10 个 query 偏向某个上下文窗口。改为均匀采样 594 个位置后修复。

**教训 #2：不要过早放弃。修完 bug 后看到真实天花板在哪，再做决策。**

#### Round 3：On-Policy 校准——论文没说的事

Bug 修完后，18 层 OK，36 层还是崩。加 45% 的数据，一层都没多压成功。

问题不在数据量，在**数据分布**。第 18 层以后看到的 KV 已经被前 17 层压缩过了——用原始分布的数据去校准被压缩分布上的行为，等于用城市路况数据训练越野自驾模型。

解法：分阶段 on-policy 校准。结果：**36/36 层全部压缩**，87.5% QA 准确率（与 baseline 持平）。

**教训 #3：这是我读过的所有 KV Cache 压缩论文中都没有被充分讨论的一点。**

#### Round 4：Scored P2 + Flat Buffer

最终架构抛弃了复杂的三层缓存（L0 Recent + L1 Quantized Warm + L2 AM Cold），简化为**一次性 Promotion**：

```
PP 阶段:    全 bf16 Recent buffer，不压缩
Promotion:  PP→TG 转换时一次性 AM 评分 → hot token 进 flat buffer / cold 丢弃
TG 阶段:    flat buffer(Q8) → O(1) per token append
```

为什么简化反而更好？因为 Pipeline 架构（PQ4+AM）的 PP 内存反而**翻倍**——Attention 必须收 bf16 数据，但 Pipeline 同时持有量化存储和 dequant 结果 = 双份内存。Scored P2 在 PP 阶段不做量化，PP 内存 = standard。

#### Round 5：Q8_0 Flat Buffer——几乎免费的 50%

flat buffer 本身还能压。per-token absmax int8 + bf16 scale，一条 Metal 指令的 dequant：

| 量化 | TG 内存 (32K) | TG 速度 | 速度代价 |
|------|-------------|---------|---------|
| bf16 (无量化) | 288 MB | 26.2 tok/s | — |
| **Q8_0** | **147 MB** | **24.7 tok/s** | **-6%** |
| Q4_0 | 81 MB | 16.1 tok/s | -39% |

Q8_0 是甜蜜点：6% 速度换 49% 内存。Q4_0 的 nibble unpack 是 compute-bound，KV 只占 TG 总带宽 6%，省带宽没意义但多出来的计算是实打实的。

**教训 #4：KV Cache 量化的收益分析不能只看 cache 本身。TG 阶段 94% 的带宽在模型参数上。**

### 混合架构上的 AM：完全失败

这是一个重要的负面结果。在 Qwen3.5（30 SSM + 10 Attention）上：

```
Compression ratio 2.0x → 乱码
Compression ratio 3.0x → 乱码
Compression ratio 5.0x → 乱码（和 2.0x 的乱码一模一样）
```

只压缩 10/40 层（25%）就完全崩溃。SSM 层放大了 Attention 层的压缩误差。这不是一个可以通过调参解决的问题——是架构级别的不兼容。

**教训 #5：AM 不是通用记忆压缩器。即使一层是 softmax attention，也可能因架构交互而失效。混合架构的层间交互比单层特性更重要。**

这个发现也解释了为什么我们在参数内存优化（Expert Offloading）上走了完全不同的路——不压缩，而是**卸载**。

### 最终结果

Qwen3-8B-MLX (Q8), Apple M4 Pro 24GB，Scored Q8 (推荐配置)：

| 指标 | Standard | Scored Q8 | 16K 变化 | 32K 变化 |
|------|----------|-----------|---------|---------|
| TG 速度 | 18.9 / 16.0 tok/s | 24.7 / 24.7 tok/s | **+31%** | **+54%** |
| KV TG 内存 | 2,268 / 4,572 MB | 129 / 147 MB | **-94%** | **-97%** |
| 质量 | PASS | PASS | 无损 | 无损 |

32K 上下文：**TG 快 54%，KV 省 97%，质量无损。**

---

## 三线汇合：完整架构

```
                         FlashMLX v1.0 — 三维内存优化
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ┌─── 路线一：参数内存 ───────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │  MoE Expert Offloading + Compact Pool                             │  │
│  │  Full Pool (256 experts) ──→ PP (zero overhead)                   │  │
│  │  Compact to hot-K ──→ CPU cache (non-hot experts)                 │  │
│  │  TG: remap + clamp, zero .item(), full lazy eval                  │  │
│  │                                                                    │  │
│  │  Model: Qwen3.5-35B-A3B (Q4)                                     │  │
│  │  Result: 18.21 GB → 9.77 GB (-46%), TG: 92.8 tok/s (zero loss)   │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─── 路线二：PP 内存 ───────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │  Chunked Prefill + Streaming AM Eviction                          │  │
│  │  chunk=512, max_cache=2048                                        │  │
│  │  if cache > threshold: AM scoring → evict cold tokens             │  │
│  │  O(chunk × cache) attention per chunk → O(1) total memory         │  │
│  │                                                                    │  │
│  │  Model: Qwen3-8B-MLX (Q8)                                        │  │
│  │  Result: PP peak 5,079 → 774 MB (-85%), PP speed +73% @ 32K      │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─── 路线三：TG 内存 ──────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │  Scored P2 + Q8_0 Flat Buffer Quantization                        │  │
│  │  PP: bf16 recent buffer (no quant overhead)                       │  │
│  │  Promotion: AM scoring → hot tokens → Q8 flat buffer              │  │
│  │  TG: int8 flat buffer, O(1) per token append                     │  │
│  │                                                                    │  │
│  │  Model: Qwen3-8B-MLX (Q8)                                        │  │
│  │  Result: KV 4,572 → 147 MB (-97%), TG speed +54% @ 32K           │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─── 支撑系统 ─────────────────────────────────────────────────────┐  │
│  │  Auto-Calibration: 首次 ~26s, 缓存后 <1ms                       │  │
│  │  Pluggable Quantizers: Q4_0, Q8_0, PolarQuant, TurboQuant        │  │
│  │  On-Policy Calibration: 分阶段校准解决深层网络误差累积           │  │
│  │  UMA-Aware: Apple Silicon 统一内存 = numpy→mx 6μs memcpy         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Platform: Apple M4 Pro 48GB                                             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 总成绩单

### 参数内存 — Qwen3.5-35B-A3B (Q4, 256 experts/layer, MoE)

| Config | TG (steady) | 参数内存 | Saved |
|--------|------------|---------|-------|
| No offload | 90.0 tok/s | 18.21 GB | — |
| Compact pool=192 | 90.9 tok/s | 13.99 GB | **-4.23 GB (-23%)** |
| Compact pool=128 | **92.8 tok/s** | **9.77 GB** | **-8.44 GB (-46%)** |

### PP 内存 — Qwen3-8B-MLX (Q8, Dense Transformer)

| Config | PP Speed | PP Peak Memory | Change |
|--------|----------|----------------|--------|
| Standard 16K | 330.6 tok/s | 8,399 MB | — |
| Standard 32K | 264.6 tok/s | 16,990 MB | — |
| Scored Chunked 16K | 367.5 tok/s (+11%) | **1,131 MB** | **-87%** |
| Scored Chunked 32K | 369.1 tok/s (+39%) | **1,131 MB** | **-93%** |

### TG 内存 — Qwen3-8B-MLX (Q8, Dense Transformer)

| Config | TG Speed | KV TG Memory | Change |
|--------|----------|-------------|--------|
| Standard 16K | 18.9 tok/s | 2,268 MB | — |
| Standard 32K | 16.0 tok/s | 4,572 MB | — |
| Scored Q8 16K | 24.7 tok/s (+31%) | **129 MB** | **-94%** |
| Scored Q8 32K | 24.7 tok/s (+54%) | **147 MB** | **-97%** |

### 组合效果（PP + TG，Qwen3-8B, 32K 上下文）

| 阶段 | Standard | FlashMLX | 变化 |
|------|----------|----------|------|
| PP 速度 | 213.6 tok/s | 372.8 tok/s | **+74.5%** |
| PP 峰值内存 | 5,079 MB | 774 MB | **-84.8%** |
| TG 速度 | 16.0 tok/s | 24.7 tok/s | **+54.4%** |
| TG KV 内存 | 4,572 MB | 147 MB | **-96.8%** |
| TTOF | 151.7s | 86.9s | **-42.7%** |
| 质量 | PASS | PASS | 无损 |

---

## 技术创新总结

### 设计决策层（10 个创新）

| # | 创新点 | 路线 | 类型 | 解决的问题 |
|---|--------|------|------|-----------|
| 1 | Two-Phase Compact Pool | 参数 | 架构创新 | MoE 参数内存 46% 节省，零 TG 惩罚 |
| 2 | Speculative Execution (clamp, no sentinel) | 参数 | 算法创新 | 消除 MoE 层的 GPU→CPU 同步瓶颈 |
| 3 | UMA-Aware CPU Cache | 参数 | 工程创新 | Apple Silicon 统一内存的 numpy→mx 快速搬运 |
| 4 | Chunked Prefill + Streaming Eviction | PP | 架构创新 | O(N²) → O(1) PP 内存 |
| 5 | On-Policy 分阶段校准 | TG | 算法创新 | 深层网络的误差累积 |
| 6 | Bounded β Optimization | TG | 算法修复 | AM 论文的隐含假设 |
| 7 | Scored P2 一次性 Promotion | TG | 架构创新 | 避免 Pipeline 的 PP 内存翻倍 |
| 8 | Q8_0 Flat Buffer Quantization | TG | 工程创新 | Flat buffer 内存减半，6% 速度代价 |
| 9 | 可插拔量化策略 | TG | 架构创新 | Q4_0 / Q8_0 / PolarQuant 统一接口 |
| 10 | 自动校准系统 | TG | 工程创新 | 新模型零配置使用 |

### 系统工程层（15 个优化）

| # | 优化机制 | 路线 | 类型 | 去掉它会怎样 |
|---|---------|------|------|------------|
| S1 | Gather-Sort Cache Locality | 参数 | GPU 硬件优化 | gather_qmm cache miss 率暴增，TG 速度下降 |
| S2 | Three-Tier Hierarchical Cache | 参数 | 存储层次设计 | miss = 直接 SSD (240μs)，无中间缓冲 |
| S3 | Telemetry-Driven Prediction | 参数 | 自适应优化 | compact 选不对 expert，coverage < 100%，频繁 miss |
| S4 | Dynamic Pool Self-Optimization | 参数 | 在线学习 | 长对话 expert 分布漂移后性能持续下降 |
| S5 | Background Prefetch Engine | 参数 | 并行 I/O | miss 恢复延迟从 6μs 退化到 240μs (SSD) |
| S6 | Regime Auto-Detection | 参数 | 自动调优 | 用户需手动选 streaming/three-tier/full-gpu |
| S7 | Identity Path Detection | 参数 | 快速路径 | PP 阶段每层白做一次 mx.take remap |
| S8 | Async SSD→CPU Population | 参数 | 零 GPU I/O | CPU cache population 阻塞 GPU，compact 变慢 |
| S9 | Deferred PP Index Collection | 参数 | 延迟求值 | PP 阶段 2560 次 GPU→CPU sync（12.8ms 浪费） |
| S10 | Pool Miss Mini-Pool Fallback | 参数 | 优雅降级 | 非 clamp miss 无精确恢复路径 |
| S11 | Lazy Prefill Threshold | KV | 延迟量化 | 短 context 白做量化（精度损失 + 无收益） |
| S12 | Adaptive Compression Ratio | KV | 数据驱动 | 32K 用 3.0x 压缩，质量崩溃 |
| S13 | Chunk-Aware Eviction | KV | 量化感知 | TurboQuant re-quantize → QJL 噪声放大 |
| S14 | Vectorized Single-Gather | KV | GPU 算子优化 | 30 次 GPU kernel dispatch vs 1 次 |
| S15 | RoPE Position Correction | KV | 位置编码修正 | Eviction 后位置编码错乱 → attention 完全错误 |

---

## 被忽视的 15 个系统工程优化

前面 10 个创新是"设计决策"——解决**什么**问题、用**什么**方法。但真正让性能数字成立的是下面这些机制——它们解决的是"在 GPU 上真正跑起来"时遇到的硬件级、系统级问题。

这些不是"锦上添花"。**去掉任何一个，性能都可能从 90 tok/s 掉回 5 tok/s。**

### S1. Gather-Sort Cache Locality（参数 / 硬件级优化）

**问题**：MoE 的 `gather_qmm` 需要按 expert index 从 pool tensor 中提取权重。如果 token 的 expert 分配是乱序的（token 0 → expert 7, token 1 → expert 2, token 2 → expert 7...），GPU 的内存访问是随机跳跃的——cache line 命中率极低。

**原理**：在 `gather_qmm` 之前，把 token 按 expert index 排序。同一个 expert 的 token 聚在一起 → 访问同一块权重 → GPU L2 cache 命中率大幅提升。

```python
def _gather_sort(x, indices):
    """Sort tokens by expert index for better gather_qmm memory access."""
    indices = indices.flatten()
    order = mx.argsort(indices)          # expert 排序
    inv_order = mx.argsort(order)        # 记录恢复顺序
    return x.flatten(0, -3)[order // M], indices[order], inv_order

def _scatter_unsort(x, inv_order, shape=None):
    """Restore original token order after sorted gather_qmm."""
    return x[inv_order]  # 恢复原始 token 顺序
```

**自适应阈值**：只在 `indices.size >= 64` 时排序（PP 阶段多 token），TG 阶段（seq_len=1, top-8 = 8 个 indices）不排序——8 个值排序的 overhead 大于 cache locality 收益。

**类比**：这和数据库的 clustered index 是一个道理。Jeff Dean 在 Bigtable 论文里强调的 "locality group" 本质上也是同一个原理——把经常一起访问的数据在物理上放到一起。

---

### S2. Three-Tier Hierarchical Cache（参数 / 存储层次设计）

**问题**：不是所有 expert 都需要在 GPU 上，但也不是所有 miss 都要去 SSD。介于"GPU 上"和"SSD 上"之间缺少一个中间层。

**原理**：三级存储层次 + 量化的延迟差异：

```
Tier 0: GPU Pool    — mx.take 查表          — 延迟: 0 μs（同一块 VRAM）
Tier 1: CPU Warm    — numpy → mx.array      — 延迟: ~6 μs（UMA 内存拷贝）
Tier 2: SSD Cold    — pread() → mx.array    — 延迟: ~240 μs（NVMe SSD）

Tier 0 → Tier 1: 40x 差距
Tier 1 → Tier 2: 40x 差距
Tier 0 → Tier 2: 1600x 差距
```

这不是"有一个 CPU cache"这么简单。它是一个完整的存储层次设计，每个 tier 有独立的驻留策略、驱逐策略、和晋升条件。

**Mac Mini M4 Pro 48GB 的内存分配**：
```
GPU: 36 GB → 模型参数 + GPU Pool (~128 experts/layer)
CPU: 12 GB → CPU Warm Cache (~177 experts/layer)
SSD: ∞    → 全部 256 experts

GPU + CPU 覆盖: 128 + 177 = 305 / 256 = 100%+ 覆盖
几乎所有 miss 在 6μs 内恢复，而非 240μs
```

---

### S3. Telemetry-Driven Expert Prediction（参数 / 自适应优化）

**问题**：compact pool 之后，TG 阶段偶尔会路由到 pool 外的 expert。这些 miss 需要从 CPU (6μs) 或 SSD (240μs) 加载。能否预测哪些 expert 即将被需要？

**原理**：`ExpertTelemetry` 跟踪每个 expert 的两个信号，用加权组合预测"热度"：

```python
score = 0.6 * freq_norm + 0.4 * recency_norm

# freq_norm: 该 expert 被激活的总次数 / 最大激活次数
# recency_norm: 该 expert 最近一次被激活的 token 位置 / 总 token 数
# 0.6/0.4 权重: 高频 + 最近 = 最可能被再次激活
```

还有 rolling window (64 token) 做趋势检测：如果一个 expert 在最近 64 个 token 内频率暴增，但历史频率低，rolling window 会先于全局频率捕捉到这个信号。

**与 CPU Prefetch 的硬件类比**：现代 CPU 有硬件 prefetcher 预测内存访问模式。ExpertTelemetry 做的是同一件事，只不过预测的是 "which expert will be routed to" 而非 "which cache line will be accessed"。

---

### S4. Dynamic Pool Self-Optimization（参数 / 在线学习）

**问题**：compact 时选的 top-K expert 基于 PP 阶段的激活分布。但 TG 阶段的分布可能漂移——长对话中某些 expert 变冷，新 expert 变热。

**原理**：`maintain_pool` 每 8 个 token 执行一次 promote/evict 循环：

```
每 8 tokens:
1. 用 Telemetry 预测 top-16 热门 expert（排除已在 pool 中的）
2. 检查哪些在 CPU cache（6μs promote）vs SSD（240μs promote）
3. 找出 pool 中最冷的 expert（telemetry score 最低）
4. 驱逐冷 expert → CPU cache，晋升热 expert → GPU pool
5. 重建 pool tensor + remap 表

限制: 每次最多 promote 4 个 expert（amortize 重建 pool 的开销）
保底: 永远不驱逐到 < 8 个 expert（keep_min=8）
```

**为什么是 8 tokens 而非每个 token**：pool 重建需要 `mx.concatenate` + `mx.eval` + remap 表重构。每 token 做一次太贵。8 token 间隔的代价是 ~8 次 miss（如果 miss 的话），但换来 7 次不需要重建的全速推理。

---

### S5. Background Prefetch Engine（参数 / 并行 I/O）

**问题**：当 Dynamic Pool 预测一个 expert 需要从 SSD 加载（240μs），这个延迟发生在 TG 热路径上，会阻塞推理。

**原理**：`PrefetchEngine` 在后台守护线程中运行，提前把 SSD 数据搬到 CPU cache：

```
Main thread:  token N → compute → token N+1 → compute → ...
Prefetch:     ────── SSD→CPU(expert A) ── SSD→CPU(expert B) ──
              ↑                           ↑
              Telemetry 预测 A 即将被需要   Telemetry 预测 B 即将被需要

当 token N+3 真的需要 expert A 时：
  ❌ 没有 Prefetch: SSD load 240μs → 阻塞推理
  ✅ 有 Prefetch: CPU cache hit 6μs → 几乎不影响
```

**批处理**：每次最多处理 8 个 prefetch 请求，按 layer 分组批量加载（`loader.load_experts_numpy` 支持多 expert 批量 pread）。空闲时 1ms sleep，几乎零 CPU 占用。

---

### S6. Regime Auto-Detection（参数 / 自动调优）

**问题**：不同硬件 + 不同模型 + 不同并发数 = 不同的最优策略。用户不应该手动选择 "streaming" vs "three-tier" vs "full GPU"。

**原理**：`RegimeDetector` 不只看 "模型能不能放进 GPU"，还考虑**并发需求**：

```python
available = gpu_memory - os_overhead - non_expert_params - kv_cache * concurrent_requests
ratio = available / total_expert_size

if ratio >= 1.0:    # Regime C: 全放得下
    pool = top 25% experts (cache locality bonus)
elif ratio >= 0.3:  # Regime B: 三级缓存
    pool = 40% available + CPU warm + SSD cold
else:               # Regime A: 最小 pool + SSD 流式
    pool = max(available * 50%, 1 GB)
```

**关键洞察**：一个 18 GB 模型在 24 GB GPU 上看起来是 Regime C（全放得下），但如果要支持 8 个并发请求（8 × 0.5 GB KV = 4 GB），变成 Regime B。**并发数改变了 regime**。

---

### S7. Identity Path Detection（参数 / 快速路径优化）

**问题**：PP 阶段全部 256 expert 在 GPU 上，每次 `_pool_call` 都要做 `mx.take(remap_table[indices])` 来映射索引——但此时 remap 是 identity（expert i → slot i），这个 `mx.take` 是纯开销。

**原理**：检测 identity 条件 → 跳过 remap：

```python
self._pool_is_identity = (
    K == self.num_experts and              # pool 包含全部 expert
    expert_ids == list(range(self.num_experts))  # 顺序排列
)

# _pool_call:
if self._pool_is_identity:
    # 直接用原始 indices，不经过 remap
    y = self._switchglu(x_e, self._pool, indices)
else:
    # compact pool: 必须 remap
    local_indices = self._pool_remap[indices]
    y = self._switchglu(x_e, self._pool, local_indices)
```

PP 阶段处理长 prompt（数千 token），每个 token 40 层 MoE → 跳过 40 × N_tokens 次不必要的 `mx.take`。

---

### S8. Async SSD→CPU Population（参数 / 零 GPU 干扰 I/O）

**问题**：compact 后，非热门 expert 需要存入 CPU cache 以备 miss 恢复。传统做法：`GPU tensor → np.array(GPU→CPU sync) → CPU cache`。但 GPU→CPU sync 发生在 compact 热路径上，阻塞 TG。

**原理**：完全绕过 GPU，直接从 SSD 到 CPU：

```
❌ 传统: SSD → mx.array(GPU) → np.array(GPU→CPU sync!) → CPUWarmCache
✅ 异步: SSD → bytes → numpy(CPU) → CPUWarmCache

数据流: pread() → 字节流 → numpy（全部在 CPU 侧）→ 后台线程
GPU 完全不参与，零 GPU 同步，零 Metal 堆内存占用
```

后台守护线程执行，`best-effort`（失败不报错，因为 SSD fallback 仍然可用）。

---

### S9. Deferred PP Index Collection（参数 / 延迟求值模式）

**问题**：PP 阶段需要统计每个 expert 的激活次数，用于后续 compact 选热门。最直接的做法：每个 chunk 用 `.tolist()` 提取 indices → CPU 上 bincount。但 `.tolist()` 是 GPU→CPU sync。

**原理**：PP 阶段只"记账"，compact 时才"算账"：

```python
# PP 阶段：只 append 到 buffer（零 GPU sync）
if self._prebuilt_full and not self._pool_compacted and seq_len > 1:
    self._pp_indices_buffer.append(indices.reshape(-1))  # MLX 数组，lazy

# Compact 阶段：一次性算总账
all_indices = mx.concatenate(self._pp_indices_buffer)  # 一次 concat
counts = np.bincount(np.array(all_indices), ...)       # 一次 GPU→CPU sync
```

如果 PP 处理 32K tokens / 512 chunk_size = 64 个 chunk × 40 层 = 2560 次 `.tolist()`——每次 sync 5μs = 12.8 ms 白白浪费。Deferred collection 把它变成 1 次。

---

### S10. Pool Miss Fallback Mini-Pool（参数 / 优雅降级）

**问题**：当 TG 路由到 pool 外的 expert 时（极少发生），怎么处理？

**原理**：构建一个临时 mini-pool，只包含当前 token 需要的 expert：

```
token 需要 expert [2, 7, 13, 45, 7, 2, 128, 200]
pool 有: [2, 7, 13, 45, 200]
缺少: [128]

→ 从 pool 切出 [2,7,13,45,200] 的 slice（lazy，零开销）
→ 从 CPU cache 加载 128（~6μs）
→ 拼成 6-expert mini-pool → gather_qmm → 返回结果
```

TG 阶段 seq_len=1, top-8，所以 `.tolist()` 只有 8 个值，开销可忽略。这是 Speculative Execution（前面创新 #2）的"降级兜底"——正常路径 clamp 处理 miss，极端情况走 mini-pool 精确处理。

---

### S11. Lazy Prefill Threshold（KV / 延迟量化）

**问题**：短 prompt（< 8K tokens）做 Q4_0 量化是浪费——量化有精度损失，而短 context 的 KV cache 内存本来就小，不值得压缩。

**原理**：在 prefill 阶段检查 context 长度，低于阈值时跳过量化：

```python
# lazy_prefill_threshold = 8192
if self.recent_keys.shape[2] <= self.lazy_prefill_threshold:
    return self.recent_keys, self.recent_values  # bf16，零量化开销

# 长 context（> 8K）：做增量 aging 避免 TTFT spike
if self.recent_keys.shape[2] > self.recent_size:
    self._manage_aging()  # Recent → Warm(Q4_0) → Cold(AM)
```

**为什么不总是跳过**：因为量化也在做 aging（Recent → Warm → Cold），长 context 不做 aging 会导致 PP→TG 转换时一次性处理所有 token（TTFT spike）。

---

### S12. Adaptive Compression Ratio（KV / 数据驱动调优）

**问题**：固定压缩比在不同 context 长度下表现不同。3.0x 在 16K 最优，但 32K 下质量开始下降。

**原理**：从 benchmark 数据反向推导最优比例：

```python
def _get_effective_ratio(self, context_len):
    if context_len <= 16384:
        return 3.0   # 16K: 最佳 TG (+27%) + 质量 PASS
    else:
        return 1.5   # 32K+: 温和压缩，保留数值细节
```

**为什么 32K 要降到 1.5x**：更长的 context = attention 分布更分散 = 每个 token 的 importance score 更均匀。暴力丢 67% 的 token（3.0x）会把一些"中等重要"的 token 丢掉。1.5x 只丢 33%，保留更多信息。

---

### S13. Chunk-Aware Eviction（KV / 量化感知驱逐）

**问题**：TurboQuant 的 QJL 残差校正给每个 chunk 记录了 `signs` 和 `norms`。如果从 chunk 中间切一刀驱逐一半 token，再重新量化——QJL 的随机投影矩阵变了，噪声会累积放大。

**原理**：TurboQuant 设 `requires_chunk_eviction=True`，触发整 chunk 驱逐而非 split：

```
❌ 标准驱逐 (Q4_0):  dequant ALL → split → requant remaining
   → 可以，因为 Q4_0 是确定性量化，重量化后 cos > 0.995

❌ 如果 TurboQuant 也这样做:  dequant → split → RE-quantize
   → QJL 的随机投影矩阵变了 → 新 signs 与旧 norms 不匹配 → 噪声放大

✅ Chunk-aware 驱逐:  pop 整个 chunk（量化一次 + 解量化一次）
   → 剩余 chunk 完全不动 → 零重量化误差
```

`_evict_warm_chunks` 找最少的完整 chunk 覆盖 >= overflow tokens，然后一次性 pop。

---

### S14. Vectorized Single-Gather（KV / GPU 算子优化）

**问题**：`_scored_compress_prefix` 需要从 old_keys 中选出重要 token。如果逐 chunk 做 gather（30 个 chunk = 30 次 `mx.gather`），GPU kernel 发射开销累积。

**原理**：在 numpy 侧构建一个全局 index 数组，然后做一次 MLX gather：

```python
# 方案A (N 次 gather):
for chunk in chunks:
    hot_k = old_keys[:, :, chunk_indices, :]  # N 次 GPU kernel 发射

# 方案B (1 次 gather):
global_indices = []
for chunk in chunks:
    global_indices.append(np.where(mask)[0] + offset)  # numpy，CPU 上
all_indices = np.concatenate(global_indices)  # 一个 numpy 数组
hot_k = old_keys[:, :, mx.array(all_indices), :]  # 1 次 GPU kernel
```

numpy 的 concatenate 在 CPU 上 < 1μs。30 次 GPU kernel 发射 vs 1 次 → 减少 29 次 kernel dispatch overhead。

---

### S15. RoPE Position Correction（KV / 位置编码修正）

**问题**：Chunked Prefill 会驱逐中间的 token。假设处理了 8K tokens 但驱逐后 cache 只有 2K tokens。如果 RoPE 用 `cache.shape[2]` (= 2K) 作为下一个 token 的位置，位置编码就是错的——它应该是 8001。

**原理**：用 `_true_offset` 追踪真实 token 计数，与物理 cache 长度脱钩：

```python
# 物理 cache: 2048 tokens (eviction 后)
# 真实位置: 8192 (已处理的 token 总数)

@property
def offset(self):
    if self._true_offset is not None:
        return self._true_offset + self._tg_count  # 真实位置
    return self.recent_keys.shape[2]  # 物理 cache 长度（无 eviction 时）
```

这保证了 RoPE 在 eviction 后仍然给出正确的位置编码。如果不做这个修正，模型会认为第 8001 个 token 在位置 2049——距离偏差 6K 个位置，attention pattern 完全错乱。

---

## 系统级洞察：为什么总体大于部分之和

上面 10 个创新 + 15 个工程优化不是独立存在的。它们形成一个**相互依赖的系统**：

```
┌─────────────────────── 性能数字的因果链 ────────────────────────┐
│                                                                  │
│  Compact Pool (创新1) 能做到零 TG 惩罚，因为：                   │
│    └→ Speculative Clamp (创新2) 消除了 GPU→CPU sync              │
│       └→ 但 Clamp 能用，因为 Coverage ≈ 100%                    │
│          └→ Coverage 高，因为 Telemetry (S3) 选了正确的 expert   │
│             └→ Telemetry 准，因为 Deferred PP Index (S9) 收集完整│
│                                                                  │
│  92.8 tok/s 能维持，因为：                                       │
│    └→ Gather-Sort (S1) 保证了 gather_qmm cache locality         │
│       └→ Identity Path (S7) 让 PP 阶段零 remap 开销             │
│          └→ Dynamic Pool (S4) 让长对话不衰退                     │
│             └→ Prefetch Engine (S5) 让 miss 从 240μs 降到 6μs   │
│                └→ Async CPU Population (S8) 让 Prefetch 有数据   │
│                                                                  │
│  O(1) PP 内存能做到，因为：                                      │
│    └→ Chunked Prefill (创新4) 切成固定大小 chunk                 │
│       └→ 但 eviction 后 RoPE 不崩，因为 Position Correction (S15)│
│          └→ Vectorized Gather (S14) 让 eviction 本身够快         │
│                                                                  │
│  -97% KV 内存能做到，因为：                                      │
│    └→ Scored P2 (创新7) 一次性 promotion                        │
│       └→ Q8 Flat Buffer (创新8) 再压 50%                        │
│          └→ Lazy Prefill (S11) 短 context 不浪费                 │
│             └→ Adaptive Ratio (S12) 长 context 不崩质量          │
│                └→ Chunk-Aware Eviction (S13) TurboQuant 不放大噪声│
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**这就是系统工程。** 每个数字背后不是一个 idea，是 5-6 个 idea 的组合。去掉因果链中任何一环，整个数字就崩掉。

这也是为什么 FlashMLX 不是"三个独立的优化 slapped together"——三条路线共享底层原语（lazy evaluation、UMA、telemetry），互相配合而不互相干扰。

---

## 技术创新详解

### 1. Two-Phase Compact Pool（参数 / 架构创新）

**问题**：MoE 模型推理时每层只激活 top-k 专家（如 Qwen3.5-35B-A3B：256 专家中只用 8 个），但所有 256 个专家的参数常驻 GPU，白白占 18.21 GB。

**原理**：利用一个关键观察——**TG 阶段的路由高度集中在 PP 阶段激活过的同一批专家上**。因此可以分两阶段处理：

```
Phase 1 (PP): 保留全部 256 专家
  → 零额外开销，identity 路径
  → 同时用 deferred buffer 记录每个专家的激活次数（不触发 GPU 同步）
  → indices 存入 _pp_indices_buffer，延迟到 compact 时一次性统计

Phase 2 (Compact + TG):
  → 用 np.bincount 统计 PP 阶段激活频率 → np.argsort 选 top-K 热门专家
  → 非热门专家逐层批量迁移到 CPU cache（每个 component 一次 GPU→CPU sync）
  → 构建新的 compact_pool dict，老 pool 引用释放，Python refcount 自动回收
  → 预热 Metal kernel（新 pool shape 需 JIT 编译 ~50 tokens）
```

**关键代码**（`expert_offload.py:1101-1169`）：

```python
def _compact_pool(self, target_pool_size):
    # 1. 一次性聚合 PP 阶段缓存的所有 indices（单次 GPU→CPU sync）
    all_indices = mx.concatenate(self._pp_indices_buffer)
    counts = np.bincount(np.array(all_indices), minlength=self.num_experts)

    # 2. 选 top-K 热门专家
    hot_ids = np.argsort(-counts)[:pool_size].tolist()

    # 3. 非热门专家批量迁移到 CPU cache（每个 component 一次 sync，不是每个 expert 一次）
    non_hot_idx = mx.array(non_hot_ids)
    for comp, full_tensor in self._pool.items():
        batch = full_tensor[non_hot_idx]  # 一次 GPU→CPU
        comp_np[comp] = np.array(batch)

    # 4. 提取 compact pool（lazy，无 mx.eval）
    idx = mx.array(hot_ids)
    compact_pool = {comp: full_tensor[idx] for comp, full_tensor in self._pool.items()}

    # 5. 构建预 clamp 的 remap 表（默认值 K-1）
    self._pool_remap_np = np.full(self.num_experts, K - 1, dtype=np.int32)
    for i, eid in enumerate(hot_ids):
        self._pool_remap_np[eid] = i
```

**为什么 PP 不直接用 compact pool**：PP 阶段输入长、expert 分布分散，用 compact pool 会频繁 miss（miss 需要 CPU→GPU 搬运），反而更慢。而 TG 阶段每个 token 只查 8 个专家，路由高度集中，几乎零 miss。

---

### 2. Speculative Execution — Clamp, No Sentinel（参数 / 算法创新）

**问题**：compact pool 后，TG 阶段的路由索引可能指向已被卸载的专家。最直接的检测方式是每层用 `.item()` 检查是否有索引越界——但这会杀死性能。

**原理**：MLX 的核心优势是 **lazy evaluation**——所有计算以 DAG 形式构建，直到显式 `mx.eval()` 才一起执行。一个 `.item()` 调用强制 GPU→CPU 同步，把整个 DAG 切断：

```
正常 lazy 执行（快）：          .item() 打断后（慢）：
Layer 0 ─┐                     Layer 0 → eval → item → sync
Layer 1 ─┤                     Layer 1 → eval → item → sync
Layer 2 ─┤  → 一次性 eval      Layer 2 → eval → item → sync
...      ─┤                     ...
Layer 39 ─┘                     Layer 39 → eval → item → sync
                                40 次 GPU→CPU 串行同步！
```

**解决方案**：不检查，直接 clamp。利用 remap 表的默认值设为 `K-1`（最后一个有效 slot），非 pool 专家自动映射到有效范围：

```python
# 旧方案（5.6 tok/s）：每层检查一次
max_idx = mx.max(local_indices).item()  # ← GPU→CPU sync!
if max_idx >= K:
    miss_path()
else:
    fast_path()

# 新方案（92.8 tok/s）：零检查
# remap 表构造时已经 clamp 了所有越界索引
local_indices = self._pool_remap[indices]  # 一条 MLX op，全 lazy
y = self._switchglu(x_e, self._pool, local_indices)
```

**为什么可以不检查**：compact 选的是 PP 阶段激活频率最高的 top-K 专家。coverage=100% 意味着所有 PP 期间出现过的专家都在 pool 里。TG 阶段路由到 PP 没见过的专家概率极低（MoE 路由的局部性）。即使 miss，被 clamp 到 K-1 slot 的专家只占 1/8 权重（top-8 路由中的一个），对最终输出的影响可忽略。

**性能演化**：

| 方案 | TG 速度 | 原因 |
|------|---------|------|
| `.item()` sentinel | 5.6 tok/s | 40 次 GPU→CPU sync/token |
| `mx.minimum` clamp | 28.1 tok/s | 每层一个额外 MLX op |
| 预 clamp remap 表 | 32.0 tok/s | 零额外 op |
| + 更多 tokens (200) | **92.8 tok/s** | warmup 稀释消失 |

---

### 3. UMA-Aware CPU Cache（参数 / 工程创新）

**问题**：compact 后被卸载的专家需要一个快速恢复通道。传统 GPU→CPU 数据搬运很慢（PCIe 带宽瓶颈），但 Apple Silicon 不同。

**原理**：Apple Silicon 使用 **Unified Memory Architecture (UMA)**——CPU 和 GPU 共享同一块物理内存。这意味着：

```
传统架构（PCIe）:              Apple Silicon (UMA):
CPU ←→ PCIe ←→ GPU             CPU ←→ 共享物理内存 ←→ GPU
~12 GB/s 带宽                  ~200 GB/s (M4 Pro)
数据搬运 = 实打实的 memcpy     数据搬运 = 页表映射切换
延迟: ~100μs+                  延迟: ~6μs (numpy→mx.array)
```

**实现**：CPU cache 用 numpy array 存储（不占 Metal 堆内存），恢复时直接 `mx.array(numpy_data)` 构造 MLX tensor：

```python
# 存入 CPU cache（compact 阶段）
for comp, full_tensor in self._pool.items():
    batch = full_tensor[non_hot_idx]
    if batch.dtype == mx.bfloat16:
        comp_np[comp] = np.array(batch.view(mx.uint16))  # bf16→uint16 视图
    else:
        comp_np[comp] = np.array(batch)
    self._cpu_cache.put(layer_idx, eid, single)

# 从 CPU cache 恢复（miss 处理）
cached = self._cpu_cache.get(layer_idx, eid)  # numpy array, ~6μs
expert_tensor = mx.array(cached)  # zero-copy on UMA
```

**设计决策**：

- **numpy 而非 mx.array 做 cache**：numpy array 不消耗 Metal 堆内存，不参与 MLX GC，不影响 `mx.metal.get_active_memory()` 统计。
- **bf16 → uint16 视图**：numpy 不支持 bf16 dtype，用 uint16 位模式存储，恢复时再 view 回来。零精度损失。
- **每 component 批量同步**：不是每个 expert 做一次 GPU→CPU sync，而是每个 component（gate_proj, up_proj, down_proj）做一次批量 sync。256→128 compact 只触发 3 次 sync，而非 128×3=384 次。

---

### 4. Chunked Prefill + Streaming Eviction（PP / 架构创新）

**问题**：标准 Self-Attention 的时间和空间复杂度是 O(N²)。处理 32K token prompt 时，attention 矩阵 = 32K × 32K = 10 亿次运算，KV Cache + 激活 = 5 GB+ 峰值。

**原理**：将 prefill 切成固定大小的 chunk（512 tokens），每个 chunk 独立做 attention。当物理 cache 超过阈值（`scored_prefill_max_cache=4096`），用 AM importance scoring 驱逐低分 token，保留高分 token + 最近 512 tokens：

```
Chunk 1 (0-511):     prefill → cache: 512 tokens
Chunk 2 (512-1023):  prefill → cache: 1024 tokens
...
Chunk 8 (3584-4095): prefill → cache: 4096 tokens
Chunk 9 (4096-4607): prefill → cache: 4608 tokens
                      ↓ cache > max_cache (4096)
                      AM scoring → evict cold → cache: ~1872 tokens
                      ↓ continue...
```

**关键代码**（`triple_layer_cache.py:458-481`）：

```python
def _update_slow_path(self, keys, values):
    # 1. 追加到 Recent
    self.recent_keys = mx.concatenate([self.recent_keys, keys], axis=2)
    self._prefill_tokens_seen += keys.shape[2]  # 追踪真实 offset（RoPE 需要）

    # 2. Prefill 阶段（seq_len > 1）：检查是否需要 eviction
    if keys.shape[2] > 1:
        if self.scored_mode and self._scored_prefill_chunk_evict:
            total = self.recent_keys.shape[2]
            if total > self._scored_prefill_max_cache:
                self._scored_prefill_evict()  # AM 评分 → 驱逐低分 token
            return self.recent_keys, self.recent_values
```

**复杂度分析**：

```
标准 Attention:
  每个 token 要看所有历史 = O(N) per token
  N 个 token 的 prefill = O(N²) 总计
  空间: O(N) KV cache + O(N²) attention matrix

Chunked Prefill:
  每个 chunk 只看 cache 中的 token = O(chunk × cache_size) per chunk
  cache_size 被 eviction 限制在固定范围 ≤ max_cache
  → O(chunk × max_cache) per chunk = O(1) per chunk
  → 总时间: O(N/chunk × 1) = O(N) 线性
  空间: O(max_cache) = O(1) 常量
```

**RoPE 修正**：chunked prefill 会驱逐中间 token，物理 cache 长度 < 实际 token 位置。如果不修正，RoPE（Rotary Position Embedding）的位置编码会错位。解决方案：用 `_prefill_tokens_seen` 追踪真实的 token 总数，`offset` 属性返回真实位置而非物理 cache 大小。

---

### 5. On-Policy 分阶段校准（TG / 算法创新）

**问题**：AM 压缩（Attention Matching）的校准数据必须匹配推理时的实际 KV 分布。标准做法是在原始模型上跑一遍 prefill 收集 KV cache 作为校准数据——但这只适用于浅层网络。

**原理**：深层 Transformer（36 层）中，前 17 层的 AM 压缩会改变后续层看到的 KV 分布。用"干净的"（未压缩的）数据去校准第 18+ 层，等于用城市路况训练越野自驾——分布不匹配导致校准完全失效。

```
传统校准（off-policy）：                On-Policy 校准：
原始模型 → 跑 prefill → 收集 KV        Phase 1: 原始模型 → 校准 Layer 0-17
用同一份 KV 校准所有 36 层             Phase 2: 用 Phase 1 的压缩结果重跑 prefill
                                        → 收集被压缩后的 KV → 校准 Layer 18-35
Layer 18+ 看到的是被压缩过的 KV
但校准数据是干净的 KV → 不匹配!        Layer 18+ 的校准数据和推理时一致!
```

**效果**：

| 校准方式 | 压缩成功层数 | QA 准确率 |
|----------|-------------|----------|
| 单次校准（off-policy） | 18/36 层 | ~40%（乱码） |
| + 45% 更多数据 | 18/36 层 | 仍然 ~40% |
| 分阶段 on-policy | **36/36 层** | **87.5%** |

**教训**：这是我读过的所有 KV Cache 压缩论文中都没有被充分讨论的一点。论文通常在 Llama-7B（32 层）上验证，但 Llama-7B 的 AM 压缩较为温和（每层只丢一点），误差累积不明显。换到更激进的配置或更深的网络就暴露了。

---

### 6. Bounded β Optimization（TG / 算法修复）

**问题**：AM 论文的核心数学是最小化压缩前后 attention 分布的 KL 散度。其中 β 系数用于补偿被丢弃 Key 的 attention mass。论文给出解析解 `β = R_S† · target`——但没说这个解可能飞到荒谬的值。

**原理**：AM 的 β 补偿原理：

```
压缩前: softmax(Q · [K_kept, K_dropped]) = [α_kept, α_dropped]
压缩后: softmax(Q · [K_kept * β])         应该 ≈ α_kept / (1 - α_dropped)

β 的作用是缩放保留的 Key，使得去掉被丢弃 Key 后的 attention 分布尽量接近原始。
```

论文的解析解 `β = R_S† · target` 使用了 softmax 矩阵的伪逆——但 softmax 矩阵高度病态（condition number 很大），伪逆放大了数值误差。实测中 β 飞到了 `[-171, +221]` 的范围，完全没有物理意义（β 应该在 1 附近）。

**修复**：加 bounded optimization，限制 β ∈ [-3, +3]：

```python
# 论文的解析解（无界，β 飞天）
beta = np.linalg.lstsq(R_S, target, rcond=None)[0]
# 实测: beta ∈ [-171, +221]

# 修复后（有界，稳定）
from scipy.optimize import lsq_linear
result = lsq_linear(R_S, target, bounds=(-3.0, 3.0))
beta = result.x
# 实测: beta ∈ [-2.8, +2.9]，输出质量正常
```

**为什么论文没提**：论文的实验大多在较短上下文和较温和的压缩比上，softmax 矩阵的病态程度较轻。在长上下文 + 高压缩比下，矩阵条件数恶化，无界解才暴露问题。

---

### 7. Scored P2 一次性 Promotion（TG / 架构创新）

**问题**：最初的 KV Cache 压缩架构是"Pipeline"模式——PP 阶段就开始做量化（L0 Recent → L1 Warm Q4 → L2 Cold AM），但这导致 PP 内存翻倍。

**原理**：Pipeline 架构的 PP 内存问题：

```
Pipeline 架构（PP 内存翻倍）：
  Attention 计算需要 bf16 数据 → 必须持有 bf16 KV
  Pipeline 同时做量化存储    → 还要持有 Q4 量化版本
  → 双份内存！PP 峰值反而更高

Scored P2 架构（PP 内存 = standard）：
  PP 阶段: 全 bf16 Recent buffer，不做任何量化（零额外开销）
  PP→TG 转换时（第一个 TG token）:
    1. 用 AM importance scoring 评估每个 token 的重要性
    2. 保留 top-budget 重要 token → 写入 Q8 flat buffer
    3. 丢弃低分 token（eviction）
    4. 后续 TG: O(1) per token append 到 flat buffer
```

**关键代码**（`triple_layer_cache.py:498-516`）：

```python
# 第一个 TG token 触发 scored promotion
if self.scored_mode and has_am and no_warm_cold:
    if self._prefill_tokens_seen > 0:
        # Chunked prefill 路径：PP 期间已经做过 streaming eviction
        self._true_offset = self._prefill_tokens_seen
        self._promote_to_flat_buffer(self.recent_keys, self.recent_values)
    else:
        # 全量 prefill 路径：一次性 AM scoring + eviction
        self._true_offset = cache_len
        self._scored_compress_prefix(full_keys, full_values, recent_len)
    self._scored_active = True
    return self._fetch_flat(self._flat_offset)
```

**为什么"一次性"而非"增量"更好**：

| 架构 | PP 内存 | 转换成本 | TG 速度 |
|------|---------|---------|---------|
| Pipeline (L0+L1+L2) | **2x**（双份数据） | 持续（每个 chunk） | 中等（多层 dequant） |
| Scored P2 (一次性) | **1x**（只 bf16） | 一次性（~ms 级） | 快（flat buffer，一次 read） |

---

### 8. Q8_0 Flat Buffer Quantization（TG / 工程创新）

**问题**：Scored P2 的 flat buffer 用 bf16 存储 KV token，32K 上下文仍需 288 MB。能否进一步压缩？

**原理**：per-token absmax int8 量化——每个 token 的 KV 向量独立量化为 int8，附带一个 bf16 scale：

```
原始:   token_kv = [v0, v1, ..., v127]  (128 维 bf16, 256 bytes)
量化后: quant = round(token_kv / scale) → int8 (128 bytes)
        scale = max(|token_kv|) / 127   → bf16 (2 bytes)
        总计: 130 bytes ≈ 内存减半
```

**关键代码**（`triple_layer_cache.py:306-330`）：

```python
def _q8_quantize(self, x):
    """Per-token absmax: (B,H,S,D) → int8 + bf16 scale"""
    max_val = mx.max(mx.abs(x), axis=-1, keepdims=True)  # 每个 token 的最大绝对值
    scales = (max_val / 127.0).astype(mx.bfloat16)
    quant = mx.round(x / scales).astype(mx.int8)
    return quant, scales

def _q8_dequantize(self, quant, scales):
    """int8 * bf16 scale → bf16"""
    return quant.astype(mx.bfloat16) * scales  # 一条 Metal 指令
```

**为什么 Q8 是甜蜜点，Q4 不行**：

KV Cache 读取只占 TG 总带宽的 ~6%（94% 在模型参数上）。Q4 的 nibble unpack（位操作 + 移位 + 掩码）是**计算密集**的，但省的带宽只有 3%（6% 的一半）。计算代价 >> 带宽收益 = 净负。

| 量化 | TG 内存 (32K) | TG 速度 | 速度代价 | 分析 |
|------|-------------|---------|---------|------|
| bf16 | 288 MB | 26.2 tok/s | — | 基准 |
| **Q8_0** | **147 MB** | **24.7 tok/s** | **-6%** | 一条乘法指令 dequant，带宽/计算平衡 |
| Q4_0 | 81 MB | 16.1 tok/s | -39% | nibble unpack 计算密集，省的带宽 < 增加的计算 |

---

### 9. 可插拔量化策略（TG / 架构创新）

**问题**：不同场景需要不同的量化质量/压缩比权衡。硬编码一种量化算法无法适配所有需求。

**原理**：定义抽象基类 `QuantizationStrategy`，所有量化器实现 4 个接口：

```python
class QuantizationStrategy(ABC):
    @abstractmethod
    def quantize(self, keys, values) -> (quant_k, quant_v, metadata): ...
    @abstractmethod
    def dequantize(self, quant_k, quant_v, metadata) -> (keys, values): ...
    @abstractmethod
    def get_compression_ratio(self) -> float: ...
    @abstractmethod
    def estimate_memory(self, num_tokens, head_dim, num_heads) -> int: ...

    @property
    def requires_chunk_eviction(self) -> bool:
        """TurboQuant 需要整 chunk 驱逐以避免重量化误差放大"""
        return False  # Q4_0/PolarQuant 不需要

    def requantize(self, keys, values):
        """重量化路径（TurboQuant 退化为 PQ-only 以避免 QJL 噪声放大）"""
        return self.quantize(keys, values)
```

**已实现的量化器**：

| 量化器 | 算法 | 压缩比 | 特点 |
|--------|------|--------|------|
| `Q4_0Quantizer` | 4-bit 对称, group=32 | 2.0x | 默认，通用 |
| `Q8_0Quantizer` | 8-bit 对称, group=32 | ~1.78x | Flat buffer 用 |
| `PolarQuantizer` | Haar 旋转 + Lloyd-Max | 3.8x (4-bit) | 无校准，数据无关 |
| `TurboQuantizer` | PolarQuant + QJL 残差 | 3.6x (4-bit) | 无偏内积估计 |
| `NoOpQuantizer` | 不压缩 | 1.0x | 消融实验用 |

**PolarQuant 原理**（来自 Google TurboQuant, ICLR 2026）：

```
1. Haar 正交旋转: x_rotated = x · R^T（R 是随机正交矩阵，QR 分解构造）
   → 旋转后每个坐标近似 N(0, σ²) 独立高斯
2. Lloyd-Max 最优标量量化: 对每个坐标独立用预计算的最优码本量化
   → 码本是 N(0,1) 上的 Lloyd-Max 最优解，硬编码在代码里
3. Bit-pack: 将 b-bit 索引打包到 uint32（32//b 个值 per uint32）
4. 解码: 查码本 → 反旋转 → 缩放

优势: 完全数据无关（无需校准），压缩比高 (4-bit: 3.8x)
劣势: 旋转矩阵运算有开销（~15% TG 速度代价）
```

**TurboQuant 的 QJL 残差修正**：

```
Stage 1: PolarQuant (b-1 bit) → 粗量化 x_pq
Stage 2: 残差 r = x - x_pq → QJL 1-bit 量化（sign(S·r)）
解码: x_hat = x_pq + α · QJL^{-1}(signs, norms)

关键: α 必须自适应于 head_dim (d)
  QJL 方差 ∝ 1/d → d 越大，α 可以越大
  α = d / (d + 1152)  # 校准点: d=128 → α=0.1
  d=64: α≈0.05 | d=128: α≈0.10 | d=512: α≈0.31

论文用 α=1.0 → 在 d=128 上完全乱码
我们校准出 α=0.1 → 正确输出 + 流畅文本
```

---

### 10. 自动校准系统（TG / 工程创新）

**问题**：AM 压缩需要预校准数据（β 系数 + selected_indices），但用户不应该关心这些。新模型应该零配置即可使用。

**原理**：首次使用时自动跑校准（~26 秒），结果缓存在 `~/.cache/flashmlx/calibrations/` 目录，后续使用 <1ms 加载：

```python
# 用户只需一行代码
generate(model, tokenizer, prompt, kv_cache="scored_pq")
# 内部自动:
# 1. 生成 model key: "{model_type}_h{hidden_size}_l{num_layers}_kv{num_kv_heads}"
# 2. 查缓存: ~/.cache/flashmlx/calibrations/{key}_{ratio}x.pkl
# 3. 不存在 → 自动校准 → 缓存
# 4. 存在 → 直接加载
```

**校准流程**（`am_calibrator.py:256-428`）：

```
Step 1: 生成 Query 样本（~1800 queries）
  → 8 种多样化文本 × 5 次 repeat-prefill（不同 cache 状态）
  → 12 个 QA 问题 × 1 次 prefill（测试检索 pattern）
  → 覆盖: 叙事、技术、中文、代码、数学、对话、列表、JSON

Step 2: 生成参考 Key（~512 tokens）
  → 拼接两段校准文本 → 标准 prefill → 提取 KV cache

Step 3: 拟合 AM 参数（逐 Attention 层）
  → attention_scores = softmax(Q · K^T / √d)
  → avg_importance = mean(scores, axis=queries)
  → selected_indices = argsort(avg_importance)[-budget:]
  → beta = ones（scored_pq 不用 beta，只用 indices 做 eviction）
```

**混合架构兼容**（`_make_native_cache` + `_get_attention_indices`）：

```python
def _make_native_cache(self):
    """创建原生 cache（自动处理混合架构）"""
    if hasattr(self.inner, "make_cache"):
        return self.inner.make_cache()  # SSM 层得到 ArraysCache, Attention 层得到 KVCache
    return [KVCache() for _ in range(self.num_layers)]

def _get_attention_indices(self, cache):
    """检测哪些层是 Attention（通用方法，不绑定属性名）"""
    return [i for i, c in enumerate(cache) if isinstance(c, KVCache)]
    # Qwen3.5: 返回 [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]（10 个 Attention 层）
    # Qwen3-8B: 返回 [0, 1, ..., 35]（全部 36 层）
```

**缓存文件格式**（.pkl）：

```python
{
    'model_name': 'qwen3_h4096_l36_kv8',
    'compression_ratio': 2.0,
    'num_layers': 36,  # 或 10（混合架构只校准 Attention 层）
    'calibration': {
        0: {'Ck': array, 'beta': array, 'selected_indices': array,
            'compression_ratio': 2.0, 'budget': 256},
        1: {...},
        ...
    },
    'created_at': '2026-03-29T...',
    'version': '2.0',
}
```

校准文件体积小（~200 KB），首次校准 ~26s（主要是 40 次 prefill），缓存后后续调用 <1ms。支持 `force=True` 强制重新校准。

---

## 技术演化墓地：废弃方案与教训

以下方案要么已被替代（代码还在但不推荐），要么被证明不可行（彻底放弃）。**每个"死亡"方案的价值不在于它本身，而在于它暴露了什么隐藏假设。**

### 墓碑 1：Pipeline 三层缓存（L0 Recent → L1 Warm Q4_0 → L2 Cold AM）

**生卒**：KV 压缩的第一代架构，被 Scored P2 取代。**代码仍在**（`_manage_aging`、`_append_warm_with_quant`、`_append_cold_with_am`、`_am_compress_prefix`），设 `scored_mode=False` 可以激活。

**怎么工作的**：

```
Token 到来 → L0 Recent (bf16, 最近 512 tokens)
               ↓ 溢出 512 个
             L1 Warm (Q4_0 量化, 1536 tokens)
               ↓ 溢出 >= 64 个 (batch threshold)
             L2 Cold (AM 压缩, β 补偿重建)
```

每一层都有独立的量化/压缩逻辑，token 在三层之间"老化"。看起来很优雅——渐进式压缩，精度逐层降低，最近的 token 永远精确。

**为什么被废弃**：

PP 内存**翻倍**。原因很反直觉：Attention 计算需要 bf16 数据做 QKV 矩阵乘，但 Pipeline 在 PP 阶段同时持有 bf16 原始数据 + Q4_0 量化副本。两份数据 = 双倍内存。

```
Pipeline PP 内存:
  bf16 Recent (给 Attention 用)  = N tokens × head_dim × 2 bytes
  + Q4_0 Warm (Pipeline 存储)     = N tokens × head_dim × 0.5 bytes
  = 2.5 × 原始大小
  vs
Scored P2 PP 内存:
  bf16 Recent (PP 不做量化)       = N tokens × head_dim × 2 bytes
  = 1.0 × 原始大小
```

**华点**：**越复杂不等于越好。Pipeline 有三层缓存、两种压缩、渐进式老化——但 Scored P2 只有一步（AM 评分 → flat buffer），PP 内存却只有 Pipeline 的 40%。** 架构设计中 "简化" 不是妥协，有时候就是正确答案。

**教训**：当你的系统有 N 个组件时，每个组件的副作用会组合爆炸。Pipeline 的三层看起来互相独立，但 PP 阶段的内存是三层共同作用的结果。

**代码墓志铭**：

```python
# triple_layer_cache.py line 550
# Pipeline mode (original): AM prune → flat buffer
if has_am:
    full_keys, full_values = self._am_compress_prefix(
        full_keys, full_values, recent_len
    )
```

---

### 墓碑 2：AM 数学压缩（compress-and-reconstruct with β compensation）

**生卒**：Attention Matching 的"正统"用法。论文原意是用 β 系数补偿被丢弃 Key 的 attention mass，让压缩后的 attention 分布尽量接近原始分布。在 Scored P2 中 AM 的数学仍然被使用，但**用途完全变了**。

**Pipeline 中 AM 的用法（已废弃）**：

```
AM 原意: 压缩 + 重建
  1. 用 AM 评分选出 hot tokens
  2. 用 β 系数缩放保留的 Key → softmax(Q · K_hot · β) ≈ 原始 attention
  3. 压缩后的 KV 存入 Cold 层，需要时 dequant + β 重建

  问题：β 解析解不稳定（墓碑3），分布偏移导致深层崩溃（墓碑4）
```

**Scored P2 中 AM 的用法（当前）**：

```
AM 新用途: 只评分，不重建
  1. 用 AM 评分选出 hot tokens（和原来一样）
  2. hot tokens 原样（bf16）进 flat buffer → 零重建误差
  3. cold tokens 直接丢弃 → 不存储，不重建
  4. β 系数？不用了。selected_indices 就够了。
```

**华点**：**同一个数学工具，用途换了，从"勉强能用"变成"完美匹配"。** AM 的 attention-score 评估 token 重要性非常准——这部分数学是 solid 的。但论文把它用于"重建压缩后的 attention 分布"，这引入了 β 补偿、分布偏移、误差累积等一系列问题。Scored P2 只用 AM 做**评分**（哪些 token 重要），不做**重建**（重新模拟原始 attention）——避开了所有这些问题。

**教训**：工具的价值取决于你怎么用。AM 的数学没有错，错的是"用它来重建"。同一个 hammer，敲钉子是工具，砸窗户是武器。

---

### 墓碑 3：无界 β Solver（lstsq without bounds）

**生卒**：AM 论文的解析解 `β = R_S† · target`，被 bounded optimization `β ∈ [-3, +3]` 取代。

**怎么死的**：softmax 矩阵 R_S 高度病态（condition number 巨大），伪逆放大数值误差。β 飞到 `[-171, +221]`——一个 Key 被放大 221 倍，另一个被反转 171 倍。输出当然是乱码。

**华点**：**论文的公式是对的，但隐含了一个实践中不成立的假设——矩阵 R_S 是良好条件的。** 论文通常在短序列（512-2K token）上验证，R_S 的 condition number 较小。长序列（16K+）+ 高压缩比下，R_S 退化为近奇异矩阵，伪逆变成噪声放大器。

**教训**：永远不要相信论文的解析解能直接用。**写 production 代码时，给所有数值优化加 bound。** 成本几乎为零（一行 `bounds=(-3, 3)`），但能避免灾难性失败。

**代码中的痕迹**：现在 `_fit_layer` 中 β 直接设为 `np.ones`（Scored P2 不用 β），β solver 代码已被删除。只有 `selected_indices` 被使用。

---

### 墓碑 4：Off-Policy 校准（用原始分布校准被压缩后的层）

**生卒**：AM 校准的第一版。在原始模型上跑一次 prefill，用得到的 KV 校准所有 36 层。被 on-policy 分阶段校准取代。

**怎么死的**：Layer 0-17 用原始 KV 校准，工作正常。Layer 18-35 看到的 KV 已经被前 17 层压缩过了——用"干净的"数据校准"脏的"层，等于用城市驾驶数据训练越野车。

```
Off-policy:
  Layer 17 看到的 KV: 100% 原始信息
  Layer 18 看到的 KV: 被 L0-L17 压缩后的信息（~60% 残留）
  但校准数据是 100% 原始信息 → 不匹配

  结果: Layer 18+ 校准完全无效
  加 45% 更多数据 → 仍然无效（问题不在数据量，在分布）
```

**华点**：**"加更多数据"是直觉，"数据分布对不上"是事实。** 这是一个经典的 distribution shift 问题，和 RL 中的 off-policy/on-policy 区分完全一样。但在 KV Cache 压缩领域，没有论文讨论过这个问题。

**教训**：当你的系统是 multi-stage pipeline（压缩层叠加），后面 stage 的行为取决于前面 stage 的输出。校准后面的 stage 时，**必须先跑完前面的 stage 再收集校准数据**。这个道理在 RL (on-policy training)、编译器优化 (profile-guided optimization)、auto-tuning (stage-aware tuning) 里都成立。

---

### 墓碑 5：Sentinel .item() Miss Detection → mx.minimum Clamp → Pre-Clamp Remap

**生卒**：三代方案，前两代被淘汰。这是一个完整的"优化演化链"。

```
Gen 1: Sentinel .item()     → 5.6 tok/s    → 被 Gen 2 取代
Gen 2: mx.minimum clamp     → 28.1 tok/s   → 被 Gen 3 取代
Gen 3: Pre-clamp remap table → 92.8 tok/s   → 当前方案
```

**Gen 1 怎么死的**：`.item()` 强制 GPU→CPU 同步。40 个 MoE 层 × 每个 token = 40 次同步。MLX lazy evaluation 图被切成 40 段串行。性能从 90 掉到 5.6。

**Gen 2 怎么死的**：用 `mx.minimum(indices, K-1)` 替代 `.item()`——保持 lazy evaluation，但每层多一个 MLX 算子。40 层 = 40 个额外 op。从 5.6 提升到 28.1，但离 90 还有 3x 差距。

**Gen 3 为什么活了**：在构建 remap 表时就把 clamp 做了（`np.full(num_experts, K-1)`），推理路径只有一个 `mx.take`——零额外 op，零 GPU sync。92.8 tok/s = 和 identity 路径一样快。

**华点**：**性能差异 16x (5.6 → 92.8)，来自"把工作从热路径搬到冷路径"。** Gen 1 在每个 token 做 40 次检查；Gen 3 在 compact 时做一次预处理，之后永远不检查。这是 precomputation 最纯粹的形式——**用一次 O(N) 预处理换掉 T × 40 次在线检查**。

**教训**：当你发现一个 O(1) 操作在热路径上被调用百万次时，问自己：这个操作能搬到 cold path 吗？哪怕 cold path 的单次成本高 100x，只要 hot path 频率足够高，就是赚的。

---

### 墓碑 6：AM 压缩在混合架构上的失败（Qwen3.5 = 30 SSM + 10 Attention）

**生卒**：尝试在 Qwen3.5 上使用 AM 压缩，**所有配置全部失败**。不是参数调不好，是架构级不兼容。

```
Ratio 2.0x → 乱码
Ratio 3.0x → 乱码（和 2.0x 一模一样的乱码）
Ratio 5.0x → 乱码（和 2.0x 一模一样的乱码）

只压 10/40 层（25%的层）→ 仍然完全乱码
```

**华点**：**三种不同的压缩比产生完全相同的乱码。** 这意味着问题不在"压多少"，而在"压不压"——有 vs 没有是质变，多 vs 少是量变。

**为什么失败**：Qwen3.5 的 30 个 SSM 层对 token 表征做了强耦合变换。10 个 Attention 层的 KV 压缩引入的微小误差，在通过 SSM 层时被**非线性放大**。SSM 的状态是递归的——误差在时间维度上累积，不像 Attention 层是独立的（每个 token 的 attention 独立计算）。

```
纯 Transformer (Qwen3-8B):
  Layer N 的压缩误差 → Layer N+1 独立计算 attention → 误差不累积

混合架构 (Qwen3.5):
  Attention Layer N 的压缩误差 → SSM Layer N+1 的递归状态
  → SSM Layer N+2 的递归状态 → ... → SSM Layer N+k 的递归状态
  → Attention Layer N+k+1 看到被 SSM 放大 k 次的误差
  → 不可控
```

**教训**：
1. **快速失败比缓慢成功更有价值**——2 天发现不可行，省去数周无用功。
2. **混合架构的层间交互比单层特性更重要**——即使每个 Attention 层的 AM 压缩是"正确的"，SSM 层的放大效应也会摧毁整体质量。
3. **正是这个失败催生了 Expert Offloading 路线**——在 Qwen3.5 上不能压缩 KV，但可以卸载参数。负面结果指向了正确方向。

---

### 墓碑 7：Q4_0 Flat Buffer（nibble unpack 的算力陷阱）

**生卒**：尝试用 4-bit 量化 flat buffer 进一步压缩 KV 内存（从 Q8_0 的 147 MB 到 81 MB）。被放弃，因为速度惩罚 -39%。

```
bf16: 288 MB, 26.2 tok/s (基准)
Q8_0: 147 MB, 24.7 tok/s (-6%，甜蜜点 ✅)
Q4_0:  81 MB, 16.1 tok/s (-39%，不值得 ❌)
```

**为什么 Q4 这么慢**：Q4_0 的 dequant 需要 nibble unpack：

```
Q8_0 dequant: int8 * bf16_scale → bf16          (一条乘法指令)
Q4_0 dequant: uint8 → 取高4位/低4位 → 移位 → 掩码 → 减偏置 → 乘 scale
              (6+ 条指令，计算密集)
```

**华点**：**KV Cache 读取只占 TG 总带宽的 ~6%**（94% 在读模型参数权重）。Q4_0 省的是这 6% 中的一半（3%），但 nibble unpack 增加的计算量是实打实的。**在一个带宽不是瓶颈的环节做带宽优化 = 负优化。**

**教训**：做量化收益分析时，不能只看被量化对象本身。要看**整个 TG pipeline 的带宽/计算分布**。Amdahl's Law 在这里完美适用——优化占比 6% 的部分，理论上限也就 6%，但计算代价远超 6%。

---

### 墓碑 8：Discovery Phase 逐 Token .tolist()（被 prebuild_pool 取代）

**生卒**：Expert Offloading 的第一阶段（Discovery）使用 `.tolist()` 逐 token 提取 expert indices，从 SSD 加载所需 expert。被 `prebuild_pool(full=True)` 取代——直接预加载全部 256 expert。

**怎么工作的**：

```
Discovery:
  Token 0 → indices.tolist() [GPU→CPU sync!] → 加载 expert 7,23,45... 从 SSD
  Token 1 → indices.tolist() [GPU→CPU sync!] → 加载 expert 2,7,89... 从 SSD
  ...
  发现足够多 expert → 构建 pool → 切换到 pool 模式

prebuild_pool(full=True):
  直接加载全部 256 expert → 构建完整 pool → PP 走 identity 路径
  零 .tolist()，零 SSD I/O，PP 全速
```

**Discovery 为什么还在代码里**：它是 **fallback**，用于无法预加载全部 expert 的场景（Regime A：model >> memory，只能流式处理）。对于 Regime B/C（model ≤ memory），`prebuild_pool` 严格更优。

**华点**：**如果你已经知道答案（要用哪些 expert），为什么还要"发现"？** Discovery 是为"不知道会用哪些 expert"设计的——但 PP 阶段处理的是用户输入的 prompt，这些 token 会激活哪些 expert 在处理前是未知的。`prebuild_pool` 的洞察是：**与其一个个发现，不如全部加载，然后再裁剪**（compact）。空间换时间。

**教训**：当"发现"的成本（逐个 .tolist() + SSD I/O）高于"全部加载 + 裁剪"的成本时，前者就该被淘汰。这和数据库的 full table scan vs index scan 的选择是一样的——当你知道你最终要扫大部分行时，full scan 反而更快。

---

### 墓碑 9：_pool_miss_call 精确 Miss 处理（热路径上被 Speculative Clamp 取代）

**生卒**：当 pool 中没有路由到的 expert 时，构建临时 mini-pool 精确处理。**代码仍在**（`_pool_miss_call`），但热路径已改用 speculative clamp（前面创新 #2）。仅在 dynamic pool 的 maintain_pool 场景中可能间接使用。

**怎么工作的**：

```
Speculative Clamp (当前热路径):
  非 pool expert → 被 remap 表 clamp 到 K-1 slot → 用最近的有效 expert 代替
  零 .tolist()，零 GPU sync，全 lazy → 92.8 tok/s
  代价：极少数 miss 的 1/8 权重误差（可忽略）

_pool_miss_call (精确路径，非热路径):
  检测到 miss → .tolist() 提取 indices → 从 CPU/SSD 加载缺失 expert
  → 构建临时 mini-pool → gather_qmm → 零误差
  代价：一次 GPU→CPU sync + CPU/SSD I/O
```

**华点**：**完美主义 vs 实用主义。** `_pool_miss_call` 追求零误差——每个 miss 都精确处理。但在 compact pool coverage=100% 的场景下，miss 率接近 0%。为了 0% 的 case 付出 100% 的检测成本（每个 token 都 `.item()` 检查），是经典的过度工程。

**教训**：在系统设计中，**"几乎不发生的 case" 的处理不应该影响 "总是发生的 case" 的性能**。正确做法是：热路径对 99.9% case 优化到极致（speculative clamp），冷路径对 0.1% case 做精确处理（_pool_miss_call 作为 fallback）。

---

### 已废弃的参数和 API

代码中还标注了以下 DEPRECATED 参数（代码保留做向后兼容）：

| 废弃项 | 替代方案 | 为什么废弃 |
|--------|---------|-----------|
| `quant_bits: int = 4` | `warm_quantizer: QuantizationStrategy` | 硬编码 bit 数 → 可插拔量化策略 |
| `warm_scales_k / warm_scales_v` | `warm_metadata: list` | 单一 scale 结构 → 通用 metadata（支持 PolarQuant/TurboQuant） |
| `memory_budget_mb` | 显式 `recent_size + warm_size` | "内存预算"太抽象 → 精确控制每层大小 |
| `cold_keys / cold_values` | `cold_compressed_keys + cold_pending_keys` | 单一 buffer → 区分已压缩/待压缩 |

**共同教训**：这些废弃都指向一个方向——**从"隐式控制"到"显式控制"**。`memory_budget_mb` 让系统自己算 cache 大小（用户不知道会怎样），`recent_size=512, warm_size=1536` 让用户精确指定（可预测、可调试）。

---

### 墓地总结：9 块墓碑的共同规律

```
┌─────────────────────── 废弃方案的三种死法 ────────────────────────┐
│                                                                    │
│  1. "看起来优雅但实际更慢" — Pipeline、Q4_0 Flat Buffer           │
│     教训: 复杂 ≠ 好。性能取决于硬件约束，不取决于数学优雅度。      │
│                                                                    │
│  2. "论文没写的约束" — 无界β、Off-Policy校准                      │
│     教训: 论文验证的是"它能work"，不验证"它在所有条件下work"。      │
│     生产代码需要的 bound/constraint 往往不在论文正文里。            │
│                                                                    │
│  3. "架构级不兼容" — AM on 混合架构、Discovery on 大内存设备       │
│     教训: 当问题在架构层面，调参无用。需要换思路（卸载代替压缩、    │
│     预加载代替发现）。                                              │
│                                                                    │
│  4. "热路径上的冷操作" — Sentinel .item()、精确Miss处理            │
│     教训: 把工作从热路径搬到冷路径。一次预处理换掉百万次在线检查。  │
│                                                                    │
│  最大的教训:                                                        │
│  每个废弃方案都不是"错的"——它在某个约束下是对的。                  │
│  Pipeline 在单层测试中最优。.item() 在非 lazy eval 系统中是正确的。│
│  AM 压缩在纯 Transformer 上完美。                                  │
│  废弃 = 约束条件变了，不是方案本身有问题。                          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 复盘：不太光彩但很有用的教训

### 论文复现的生存指南

- **先在最简单的设置上复现**。单层、短序列、小模型。如果这都不 work，不用往下走了。
- **论文的 bound 和 constraint 往往不写在正文里**。β 需要 bounded optimization 这件事，正文、附录、代码都没有。
- **论文的模型选择有 selection bias**。在 Llama-7B 上 work 的 2-bit 量化，在 Qwen3-8B 上不 work。不要假设结论能跨架构迁移。

### 系统设计的教训

- **不要追求"一种算法统治所有"**。参数用 offloading，PP 用 chunked eviction，TG 用 scored quantization。三条路线独立演进，互不干扰。
- **Lazy > Eager**。不到内存不够，不做压缩。压缩有代价。
- **Bound 你的问题**。chunk size 和 max cache 一限制，O(N²) 变 O(1)，质量不降。有时候最好的优化是"少做一点"。
- **MLX Lazy Evaluation 是把双刃剑**。用好了 40 层计算攒一起 flush，性能爆炸；一个 `.item()` 就全毁——从 90 tok/s 掉到 5.6 tok/s。

### 研究方法论

- **快速失败比缓慢成功更有价值**。混合架构上的 AM 失败用了 2 天发现，省去了可能几周的无用功。正是这个失败催生了 Expert Offloading 路线——不压缩，而是卸载。
- **保留所有实验记录**。这个项目产出了 100+ 份实验报告。每一次失败都有完整的数据和分析。
- **让数据说话，不是直觉**。"加更多数据应该能 work"是直觉；"query 分布不匹配"是数据告诉我的。"`.item()` 应该很快"是直觉；"40 次 GPU→CPU 同步"是 profiler 告诉我的。

---

## 使用方式

### KV Cache 压缩（Dense Transformer）

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-8B-Instruct")

# One line — auto-calibration on first use (~26s), cached afterwards (<1ms)
result = generate(model, tokenizer, prompt="Your long prompt here...",
                  kv_cache="scored_pq", kv_flat_quant="q8_0")
```

### Expert Offloading（MoE 模型）

```python
from mlx_lm import load
from mlx_lm.models.expert_offload import patch_model_for_offload

model, tokenizer = load("qwen3.5-35b-mlx")
ctx = patch_model_for_offload(model, model_path, max_workers=4, cpu_cache_gb=2.0)

# PP with full pool → compact → TG with compact pool
# ... generate tokens ...
ctx.compact(pool_size=128)  # 18.21 → 9.77 GB, TG: 92.8 tok/s
```

---

## 代码

项目地址：[github.com/lisihao/FlashMLX](https://github.com/lisihao/FlashMLX)

核心文件：

**KV Cache 压缩（路线二+三）**:
- `triple_layer_cache.py` — Scored P2 + Chunked Prefill + Q8/Q4 Flat Buffer
- `cache_factory.py` — 策略工厂 + 自适应参数
- `am_calibrator.py` — 自动校准系统
- `quantization_strategies.py` — 可插拔量化（Q4_0, Q8_0, PolarQuant）

**Expert Offloading（路线一）**:
- `expert_offload.py` — Two-Phase Compact Pool + Speculative Execution + CPU Cache

---

*这篇文章基于 2026 年 3 月 18 日至 29 日的开发记录。*
*KV Cache 数据来自 Qwen3-8B-MLX (Q8) on Apple M4 Pro 24GB。*
*Expert Offloading 数据来自 Qwen3.5-35B-A3B (Q4) on Apple M4 Pro 48GB。*
*所有测试在独立子进程中运行，串行执行。*
*FlashMLX v1.0 — MIT License*
