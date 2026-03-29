# 从论文到生产：一周做出 O(1) 内存的 KV Cache 压缩系统

> 在 Apple Silicon 上实现 97% KV 内存节省、54% TG 加速、O(1) PP 内存、0% 质量损失。
> 这不是一个顺利的故事。

---

## 背景

大语言模型推理有一个众所周知的内存瓶颈：KV Cache。

以 Qwen3-8B 为例，32K 上下文的 KV Cache 占用 **4.6 GB**——几乎是模型参数本身的一半。在 Apple Silicon 上，统一内存就这么大，KV Cache 吃掉的每一个 MB 都直接影响你能跑多长的上下文、能不能并发多个请求。

学术界有大量论文在做 KV Cache 压缩：Attention Matching (AM)、H2O、StreamingLLM、PolarQuant、TurboQuant……每篇都声称"无损"或"近乎无损"。

我决定在 MLX 上把这些方案真正落地。

一周后，我得到了一个在所有论文之上的结果。但过程远没有这么优雅。

---

## Day 1-2：信 AM，得永生？

### 起点：Attention Matching

起点是一篇看起来非常漂亮的论文——"Fast KV Compaction via Attention Matching"（arXiv 2602.16284）。

核心思想很直觉：用一组参考 Query 对 Key 做 attention scoring，保留得分最高的 Key 位置，用 β 系数补偿被丢弃的 Key 对 attention 分布的贡献。数学上很优雅：最小化压缩前后 attention 分布的 KL 散度。

我实现了完整管线：OMP key selection、NNLS beta fitting、LSQ C2 fitting。单层测试，完美：

```
Ratio 2.0x → 50% 压缩 → QA 100% ✅
Ratio 3.0x → 67% 压缩 → QA 100% ✅
Ratio 5.0x → 80% 压缩 → QA 100% ✅
```

信心爆棚。把 36 层全上。

### 现实的第一记耳光

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Layers Compressed    QA Accuracy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1 layer              100%     ✅
  18 layers (50%)      100%     ✅
  36 layers (100%)       0%     ❌ 输出乱码
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

TruthfulQA：0.000。输出全是 gibberish。

论文没告诉你的第一件事：**单层压缩和全层压缩是两个完全不同的问题。** 误差在 36 层 transformer 中逐层累积，到最后层已经完全不可控。

### 根因分析：论文的隐含假设不成立

深挖 NNLS solver 的行为，发现 `exp_scores` 矩阵在所有 21 个 (layer, head) 组合上都是 **rank-deficient**（秩 = 1，条件数 = ∞）：

```
Singular values: [5.06e+01, 4.71e-14, ..., 5.04e-132]
β values: 全部打到下界 -13.8
```

AM 论文有三个隐含假设：
1. Attention 分布是"弥散"的——实际上 Qwen3 的 attention 高度集中
2. β 补偿有足够的自由度——实际上 (T-t)×n 的约束空间不够
3. 参考序列足够长——实际上短序列 t > T 会导致逻辑矛盾

**教训 #1：论文的 ablation study 往往只在论文作者选定的模型和设置上做。换一个模型、换一个上下文长度，假设可能完全不成立。**

---

## Day 3：差点放弃，然后找到了 Bug

这是整个项目最关键的转折点。

当时的判断是 AM 算法本身有根本性缺陷，准备转向 H2O。但在写"放弃 AM"的决策文档时，我重新审视了实现细节，发现了两个关键 bug：

### Bug 1：β solver 无界

```python
# 论文暗示的做法
beta = np.linalg.pinv(R_S) @ target
# β ∈ [-171, 221]  ← 没有物理意义

# 修复
res = scipy.optimize.lsq_linear(R_S, target, bounds=(-3, 3))
# β ∈ [-3, 3]  ← 有意义的补偿系数
```

论文只给了公式 `β = R_S† · target`，没提需要 bound。这不是"实现细节"，这是算法能否工作的关键。

### Bug 2：Query 采样方式

```python
# 错误：连续的 10 个 query（偏向某个上下文窗口）
queries = keys[:, :, offset-10:offset, :]

# 正确：均匀采样 594 个 key 位置
indices = np.linspace(0, offset-1, 594, dtype=int)
queries = mx.take(keys, indices, axis=2)
```

这两个修复后，单层压缩的质量指标完美。但 36 层全压缩仍然崩溃——这次不是 bug，是**误差累积**的物理极限。

**教训 #2：不要过早放弃。但也不要把"算法 bug"和"算法局限性"混为一谈。修完 bug，看到真实的天花板在哪里，再做决策。**

---

## Day 3-4：On-Policy 校准——论文没说的事

18 层可以，36 层不行。直觉反应是"加更多训练数据"：

```
15.8K queries → 18 层 ✅
23.0K queries (+45%) → 仍然 18 层 ❌
```

加 45% 的数据，一层都没多压成功。

### 诊断

问题不在数据量，在**数据分布**。离线校准生成的 query 代表的是"原始 KV"上的 attention 分布，但第 18 层以后看到的 KV 已经被前 17 层压缩过了——这是一个完全不同的分布。

这就好比你用城市路况的数据训练了一个自动驾驶模型，然后丢到越野路上——数据再多也没用，因为分布不对。

### 解法：分阶段 On-Policy 校准

```
Phase 1 (离线):    Layers 0-17   (原始 KV 分布)
Phase 2 (on-policy): Layers 18-26  (0-17 已压缩的 KV 分布)
Phase 3 (on-policy): Layers 27-35  (0-26 已压缩的 KV 分布)
```

结果：36/36 层全部压缩，87.5% QA 准确率（与 baseline 持平）。

**教训 #3：深度网络的压缩不是一个静态优化问题，是一个动态系统。后面的层看到的输入取决于前面的层怎么处理——你不能用原始分布去校准被压缩分布上的行为。这一点在我读过的所有 KV Cache 压缩论文中都没有被充分讨论。**

---

## Day 4-5：论文矩阵——哪些能用，哪些是坑

在做 AM 的同时，我并行评估了多个学术方案。

### PolarQuant（Google, AISTATS 2026）

核心思想：4-bit quantization with learned polar coordinates。

**实际表现**：
- 4-bit：质量完美，72% 内存节省
- 3-bit：质量开始退化
- 2-bit：不可用

**问题**：论文声称 2-bit 可用，但实测在 Qwen3-8B 上不行。**论文的 2-bit 实验是在 Llama-7B 上做的，不同架构差异巨大。**

### TurboQuant（PolarQuant + QJL, ICLR 2026）

结合了 PolarQuant 和 Johnson-Lindenstrauss 随机投影。

**实际表现**：QJL 的随机投影在 MLX 的 Metal backend 上没有高效实现，性能反而比纯 PolarQuant 差 15%。最终做了"damped QJL"（α=0.1）来稳定质量。

**教训 #4：算法的理论复杂度和在特定硬件上的实际性能是两回事。QJL 在 CUDA 上有 cuBLAS 加速，在 Metal 上是手写 kernel——同一个算法，两种命运。**

### H2O（Heavy-Hitter Oracle）和 StreamingLLM

两个 token-level eviction 方案。简单、鲁棒，但只能做粗粒度的 cache 管理——丢整个 token，不能做 sub-token 压缩。

在最终架构中被 Scored P2 完全包含（Scored P2 的 eviction 本质上就是 AM-scored 版本的 H2O）。

### Qwen3.5 混合架构上的 AM：完全失败

这是一个并行的研究方向——在 Attention + SSM 混合架构上做 KV Cache 压缩。

```
Compression ratio 2.0x → 乱码
Compression ratio 3.0x → 乱码
Compression ratio 5.0x → 乱码（和 2.0x 的乱码一模一样）
```

只压缩 10/40 层（25%）就完全崩溃。

**根因**：SSM 层放大了 Attention 层的压缩误差。这不是一个可以通过调参解决的问题——是架构级别的不兼容。

**教训 #5：混合架构的层间交互比单层特性更重要。你不能只看"这一层是 softmax attention，所以 AM 应该能用"。你需要看这一层的输出会怎样影响下一层（可能是 SSM）的行为。**

---

## Day 5-6：架构创新——Triple-Layer Cache

到这一步，我已经积累了足够的失败经验来做正确的架构设计。

不再追求"一种算法压缩所有"，而是设计一个**分层内存管理系统**：

```
                ┌─────────────────────────────────────┐
                │          Token Lifecycle             │
                │                                     │
                │  New Token ──→ L0 (Recent, 0-512)   │
                │       exact bf16, 零延迟             │
                │                   │                  │
                │                   ▼ age out          │
                │            L1 (Warm, 512-2048)       │
                │       Q4_0 quantized, ~2x 压缩       │
                │                   │                  │
                │                   ▼ age out          │
                │            L2 (Cold, 2048+)          │
                │       AM compressed, ~3x 压缩        │
                └─────────────────────────────────────┘
```

核心设计原则：

1. **最近的 token 不压缩**——质量保证的最后防线
2. **中间层用量化**——低开销、可逆、不丢信息
3. **老 token 用 AM**——已验证的高压缩比方案
4. **所有策略可插拔**——PolarQuant、Q4_0、TurboQuant 都是可替换的 quantizer

这个架构不是某篇论文告诉我的。是在试过 AM 的 36 层崩溃、PolarQuant 的 2-bit 退化、QJL 的性能问题、混合架构的不兼容之后，从失败中提炼出来的。

---

## Day 6-7：最终形态——Scored P2 Chunked Prefill

Triple-Layer Cache 解决了 TG（decode）阶段的内存问题。但 PP（prefill）阶段仍然是个大问题：处理 32K token 的 prompt 需要一次性分配 17 GB 的 KV Cache。

### 关键洞察

如果我们在 prefill 阶段也做 eviction 呢？

不是等 prefill 结束后再压缩，而是**边 prefill 边评估 token 重要性，边驱逐不重要的 token**。

### 实现

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

### 结果

这是 v0.9.0 的输出。与标准推理相比（Chunked Prefill + Streaming Eviction，bf16 flat buffer）：

**16K 上下文**：

| 指标 | Standard | Scored Chunked | 变化 |
|------|----------|----------------|------|
| PP 速度 | 275.2 tok/s | 362.4 tok/s | **+31.7%** |
| TG 速度 | 18.9 tok/s | 26.4 tok/s | **+39.7%** |
| PP 峰值内存 | 2,785 MB | 773 MB | **-72.2%** |
| TG 内存 | 2,268 MB | 252 MB | **-88.9%** |
| TTOF | 58.4s | 44.2s | **-24.3%** |
| 质量 | PASS | PASS | 无损 |

**32K 上下文**：

| 指标 | Standard | Scored Chunked | 变化 |
|------|----------|----------------|------|
| PP 速度 | 213.6 tok/s | 369.5 tok/s | **+73.0%** |
| TG 速度 | 16.0 tok/s | 26.2 tok/s | **+63.7%** |
| PP 峰值内存 | 5,079 MB | 774 MB | **-84.8%** |
| TG 内存 | 4,572 MB | 288 MB | **-93.7%** |
| TTOF | 151.7s | 87.7s | **-42.2%** |
| 质量 | PASS | PASS | 无损 |

**PP 峰值内存 = 773 MB，无论 16K 还是 32K。O(1) 内存复杂度。**

这个数字不是巧合。`max_cache=2048` + `chunk_size=512` 限制了物理 cache 上限。理论上，你可以 prefill 128K token，PP 内存仍然在 ~800 MB 量级。

更反直觉的是：**压缩后反而更快**。

为什么？因为标准 prefill 的 attention 复杂度是 O(N²)，而 Scored Chunked 的 attention 是 O(chunk × cache)——cache 被 bound 在 2048，所以是 O(512 × 2048) = O(1) per chunk。32K 时，标准 PP 被 O(N²) 严重拖慢（213 tok/s），而 Scored Chunked 维持 370 tok/s。

### 收尾：自动校准

最后一步，我把手动离线校准流程自动化了。新模型第一次用 `scored_pq` 时自动校准（~26 秒），之后缓存在 `~/.cache/flashmlx/` 下，后续调用 <1ms。

```python
from mlx_lm import load, generate

model, tokenizer = load("any-new-model")
# 无需手动校准，无需指定 calibration 文件
result = generate(model, tokenizer, prompt, kv_cache="scored_pq")
```

对比测试显示自动校准与离线校准在质量和性能上没有统计显著差异。

---

## Day 7+：最后一公里——Flat Buffer Quantization (v0.9.2)

自动校准解决了易用性问题，但还有一个没碰的角落：**flat buffer 本身的内存**。

Scored P2 在 eviction 后会把"幸存"的 token 平铺到一个 bf16 flat buffer 里。32K 上下文下，即使只保留 max_cache=4096 个 token，这个 buffer 仍然有 288 MB（36 层 × 8 头 × 4096 token × 128 维 × 2 字节 × 2(K+V)）。

能不能把 flat buffer 本身也量化？

### Q8_0：几乎免费的 50% 内存

第一版实现很直接：per-token absmax int8 + bf16 scale。

```python
# 写入时量化
max_val = mx.max(mx.abs(x), axis=-1, keepdims=True)
scale = (max_val / 127.0).astype(mx.bfloat16)
quantized = mx.round(x / scale).astype(mx.int8)

# 读取时还原
dequantized = quantized.astype(mx.bfloat16) * scale
```

结果出乎意料的好：

| 指标 | bf16 flat | Q8_0 flat | 变化 |
|------|-----------|-----------|------|
| TG 内存 | 252 MB | 129 MB | **-49%** |
| TG 速度 | 26.4 tok/s | 24.7 tok/s | **-6.4%** |
| 质量 | PASS | PASS | 无损 |

6% 的速度代价换 49% 的内存——而且这个速度损失**不是来自 dequant 开销**，而是因为 Q8 的 cache 更小，Metal GPU 的内存带宽利用率更高时反而命中了一些调度瓶颈。

### Q4_0：极限压缩，但不推荐

然后我试了 4-bit：nibble-packed（2 个 4-bit 值塞进 1 个 uint8），per-group bf16 scale（group_size=32）。

```python
# 每组 32 个值共享一个 scale
quant = clip(round(x / scale), -7, 7)
quant_u = (quant + 8).astype(uint8)  # [1, 15]
packed = high_nibble * 16 + low_nibble  # 2 values → 1 byte
```

内存继续下降，但速度不行了：

| 指标 | bf16 flat | Q4_0 flat | 变化 |
|------|-----------|-----------|------|
| TG 内存 | 252 MB | 72 MB | **-71%** |
| TG 速度 | 26.4 tok/s | 19.4 tok/s | **-26.5%** |
| 质量 | PASS | PASS | 无损 |

问题在于 dequant 是 **compute-bound**（nibble unpack + group scale），不是 bandwidth-bound。KV cache 只占 TG 总带宽的 ~6%，所以省带宽没意义，但多出来的计算是实打实的。

**教训 #6：KV Cache 量化的收益分析不能只看 cache 本身。TG 阶段 94% 的带宽在模型参数上，KV 只占 6%。Q8_0 的 dequant 几乎免费（int8 乘 bf16，一条指令），Q4_0 的 nibble unpack 则是真正的额外开销。**

### 最终调参：max_cache 4096 → 2048

另一个发现更反直觉：把 max_cache 从 4096 砍到 2048，TG 速度反而**更快**。

原因：更小的 cache = 更短的 attention 序列 = 更快的 decode。而 AM scoring 足够好，2048 个精选 token 的质量和 4096 个没有明显差异。

### v0.9.2 最终成绩单

所有测试在独立子进程中运行，串行执行，Qwen3-8B-MLX (Q8)，M4 Pro 24GB。

**16K 上下文**：

| 指标 | Standard | Scored Q8 | 变化 |
|------|----------|-----------|------|
| PP 速度 | 275.2 tok/s | 361.8 tok/s | **+31.5%** |
| TG 速度 | 18.9 tok/s | 24.7 tok/s | **+30.7%** |
| TTOF | 58.4s | 44.3s | **-24.1%** |
| KV TG 内存 | 2,268 MB | 129 MB | **-94.3%** |
| 质量 | PASS | PASS | 无损 |

**32K 上下文**：

| 指标 | Standard | Scored Q8 | 变化 |
|------|----------|-----------|------|
| PP 速度 | 213.6 tok/s | 372.8 tok/s | **+74.5%** |
| TG 速度 | 16.0 tok/s | 24.7 tok/s | **+54.4%** |
| TTOF | 151.7s | 86.9s | **-42.7%** |
| KV TG 内存 | 4,572 MB | 147 MB | **-96.8%** |
| 质量 | PASS | PASS | 无损 |

32K 上下文：**TG 快 54%，KV 省 97%，质量无损。**

---

## 复盘：一些不太光彩但很有用的教训

### 1. 论文复现的生存指南

- **先在最简单的设置上复现**。单层、短序列、小模型。如果这都不 work，不用往下走了。
- **论文的 bound 和 constraint 往往不写在正文里**。β 需要 bounded optimization 这件事，我在正文、附录、代码都没找到。
- **论文的模型选择有 selection bias**。在 Llama-7B 上 work 的 2-bit 量化，在 Qwen3-8B 上不 work。不要假设结论能跨架构迁移。

### 2. 系统设计的教训

- **不要追求"一种算法统治所有"**。不同 token 年龄、不同上下文长度、不同架构需要不同策略。分层是对的。
- **Lazy > Eager**。不到内存不够，不做压缩。压缩是有代价的。
- **Bound 你的问题**。chunk size 和 max cache 一限制，O(N²) 变 O(1)，而且质量不降。有时候最好的优化是"少做一点"。

### 3. 研究方法论

- **快速失败比缓慢成功更有价值**。混合架构上的 AM 失败用了 2 天发现，省去了可能几周的无用功。
- **保留所有实验记录**。这一周产出了 100+ 份实验报告。每一次失败都有完整的数据和分析。没有这些记录，很多 insight 会丢失。
- **让数据说话，不是直觉**。"加更多数据应该能 work"是直觉；"query 分布不匹配"是数据告诉我的。

---

## 技术创新总结

| 创新点 | 类型 | 解决的问题 |
|--------|------|-----------|
| On-Policy 分阶段校准 | 算法创新 | 深层网络的误差累积 |
| Bounded β Optimization | 算法修复 | AM 论文的隐含假设 |
| Triple-Layer 分层缓存 | 架构创新 | 不同 token 年龄的差异化处理 |
| Chunked Prefill + Streaming Eviction | 架构创新 | Prefill 阶段的 O(N²) 内存 |
| Q8_0 Flat Buffer Quantization | 工程创新 | Flat buffer 内存减半，几乎零速度代价 |
| 自适应压缩比 | 工程创新 | 短上下文 vs 长上下文的质量/性能平衡 |
| 自动校准系统 | 工程创新 | 新模型的零配置使用 |
| 可插拔量化策略 | 架构创新 | 不同量化方案的统一接口 |

---

## 最终架构

```
                    FlashMLX KV Cache Architecture (v0.9.2)
    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │   Prefill (Chunked)                     Decode (Streaming)         │
    │   ┌──────────────┐                      ┌──────────────────┐      │
    │   │ chunk=512     │                      │ Flat Buffer (Q8) │      │
    │   │ ──→ model()   │                      │ max_cache=2048   │      │
    │   │ ──→ eval()    │                      │ int8 + bf16 scale│      │
    │   │ ──→ if >2048: │                      │ O(1) per step    │      │
    │   │    AM evict   │─── promote ─────────→│                  │      │
    │   └──────────────┘                      └──────────────────┘      │
    │        │                                        │                  │
    │        │ PP Peak: ~773 MB (O(1))                │ TG: 24.7 tok/s  │
    │        │ PP Speed: 373 tok/s                    │ TG KV: 147 MB   │
    │                                                                    │
    │   ┌────────────────────────────────────────────────────────────┐   │
    │   │  Auto-Calibration                                          │   │
    │   │  First use: 26s (8 corpus × 5 repeat + 12 QA → scoring)   │   │
    │   │  Cached: <1ms (~/.cache/flashmlx/calibrations/)            │   │
    │   └────────────────────────────────────────────────────────────┘   │
    │                                                                    │
    │   Model: Qwen3-8B-MLX (Q8) | Platform: Apple M4 Pro 24GB          │
    └────────────────────────────────────────────────────────────────────┘
```

---

## 代码

项目地址：[github.com/lisihao/FlashMLX](https://github.com/lisihao/FlashMLX)

核心文件：
- `triple_layer_cache.py` — 三层缓存 + Scored P2 + Chunked Prefill
- `cache_factory.py` — 策略工厂 + 自适应参数
- `am_calibrator.py` — 自动校准系统
- `quantization_strategies.py` — 可插拔量化（Q4_0, PolarQuant, TurboQuant）

---

*这篇文章基于 2026 年 3 月 22 日至 28 日的开发记录，所有数据来自真实测试（Qwen3-8B-MLX Q8, Apple M4 Pro 24GB）。*
*FlashMLX v0.9.2 — MIT License*
