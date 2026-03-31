# FlashMLX

> **Apple Silicon 上的本地 LLM 推理，不该只是“能跑”，而应该是：更快出首 Token、更高吞吐、更低峰值内存、对 35B MoE 真正可用。**

**FlashMLX** 不是又一个 KV cache 小优化，也不是把 CUDA 世界的 serving 方案硬翻成 Metal。  
它是一套面向 **Apple Silicon** 的 **memory-policy runtime**：把 **参数内存、Prefill 峰值、Decode 常驻、批量调度、质量保护** 当成一个整体来重写。

---

## 为什么是 FlashMLX

在本地长上下文推理里，真正把你拖死的从来不只是一件事：

- 35B MoE 模型参数太大，常驻就是浪费
- Prefill 峰值内存按 batch 叠加，先把你顶爆
- Decode 又会被长上下文 KV cache 拖慢
- 批量请求下，用户为了等完整 prefill，首 Token 体验灾难
- Apple Silicon 不是“低配 CUDA”，UMA / Metal / cache 行为都不一样

**FlashMLX 的答案不是单点 trick，而是一整套系统打法：**

1. **Parameter Memory** — MoE Expert Offloading + Compact Pool  
2. **PP Memory** — Chunked Prefill + Streaming Eviction  
3. **TG Memory** — Scored Promotion + Flat Buffer Quantization  
4. **v2 Scheduler** — Chunked Interleaved Scheduling（交错式 Prefill / Decode）

换句话说：

> **FlashMLX 不是在做一个“压缩算法”。它在做 Apple Silicon 上的本地推理操作系统。**

---

## 一句话定位

### 如果 vLLM 是为 CUDA 数据中心 serving 写的，
### 那 FlashMLX 就是为 Apple Silicon 本地长上下文与 35B MoE 推理写的 runtime。

---

## 核心结果

### FlashMLX v2.0：16K / Batch=4 / Qwen3.5-35B-A3B / Apple M4 Pro 48GB

| 指标 | 社区版 mlx-lm | FlashMLX v2.0 | 提升 |
|---|---:|---:|---:|
| Decode 吞吐 (TG) | 115.8 tok/s | **196.9 tok/s** | **+70.0%** |
| 首 Token 延迟 (TTFT) | 82.0s | **21.1s** | **-74.3%** |
| GPU 内存峰值 | 28.01 GB | **13.78 GB** | **-50.8%** |
| 模型显存占用 | 18.21 GB | **11.42 GB** | **-37.3%** |
| 生成质量 | 4/4 PASS | **4/4 PASS** | 无损 |

> 测试环境：Qwen3.5-35B-A3B (6-bit, 256 experts/layer × 40 MoE layers) / Apple M4 Pro 48GB

---

## FlashMLX 到底牛在哪

### 1) 它不是单点优化，而是“三维内存系统”

大多数方案只打一件事：
- 只管 KV 分页
- 只管 KV 量化
- 只管 eviction
- 只管 serving 调度

**FlashMLX 同时打三块内存：**

| 维度 | 问题 | FlashMLX 做法 | 结果 |
|---|---|---|---:|
| Parameter Memory | 35B MoE 专家权重常驻太浪费 | Expert Offloading + Compact Pool | 18.21 → **9.77 GB** |
| PP Memory | Prefill 峰值按 batch 叠加 | Chunked Prefill + Streaming Eviction | 5,079 → **774 MB** |
| TG Memory | 长上下文 KV cache 拖慢 decode | Scored Promotion + Flat Buffer | 2,454 → **80 MB** |

**这不是“某个 cache trick”能解释的结果。**

---

### 2) 它真正解决的是本地 batch 推理的死穴：TTFT

传统本地批量推理是这么跑的：

```text
Prefill ALL requests completely -> then Decode
```

问题很简单，也很致命：

- batch 越大，prefill activation 叠得越高
- 第一个用户必须等最后一个请求 prefill 完
- decode 单元在长时间 prefill 期间基本空转

**FlashMLX v2.0 干的事，是直接改执行秩序：**

```text
Decode ALL active requests -> Prefill one chunk (512 tokens) -> repeat
```

这不是参数调优，这是**系统调度重构**。

---

### 3) `chunk=512` 不是小参数，而是关键系统发现

很多人会凭直觉认为：prefill chunk 越大越高效。  
FlashMLX 把这件事做实测扫全了，结论正相反。

| chunk_size | TG 吞吐 | 解码步延迟 | 峰值内存 | 质量 |
|---:|---:|---:|---:|---:|
| 4096 | 131.4 tok/s | 30.4 ms | 17.01 GB | 4/4 |
| 2048 | 136.5 tok/s | 29.3 ms | 15.15 GB | 4/4 |
| 1024 | 155.6 tok/s | 25.7 ms | 14.36 GB | 4/4 |
| **512** | **198.3 tok/s** | **20.2 ms** | **13.99 GB** | **4/4** |

**为什么？**
因为大 chunk 会把 activation 塞满 GPU cache，接下来 decode 读取 KV cache + 参数时 cache miss 暴增；小 chunk 反而让 decode 热数据留在 cache 里。  
这不是纸面理论，这是**Apple Silicon 上真实跑出来的系统行为**。

---

## 与竞争路线的区别

> 下面不是“别人不行”，而是 **FlashMLX 解决的问题边界更完整**。

### 与社区版 `mlx-lm`

| 维度 | 社区版 mlx-lm | FlashMLX |
|---|---|---|
| 目标 | 通用 Apple Silicon LLM 推理与微调 | **Apple Silicon 本地长上下文 / 35B MoE / batch 推理 runtime** |
| 调度 | 传统 prefill-then-decode | **Chunked Interleaved Scheduling** |
| MoE 内存 | 通用路径 | **Expert Offloading + Compact Pool** |
| 长上下文 KV | 基础能力 | **Scored eviction + flat buffer + 可插拔量化** |
| 优先级 | 通用性 / 生态 | **端到端 TTFT / 峰值内存 / TG balance** |

**一句话：** 社区版 mlx-lm 是底座，FlashMLX 是在 Apple Silicon 上把“本地推理体验”狠狠干到位的系统层。

---

### 与 `vLLM` / `vllm-metal`

| 维度 | vLLM | FlashMLX |
|---|---|---|
| 主战场 | **CUDA / HIP 数据中心 serving** | **Apple Silicon 本地推理** |
| 核心武器 | PagedAttention、continuous batching、serving 吞吐 | **三维内存系统 + 交错调度 + MoE 专家池** |
| KV 管理 | 强，尤其在高并发 serving | 强，但更关注**本地长上下文 + TTFT + UMA 行为** |
| 平台假设 | 高并发服务端 | **单机本地 / M4 Pro 48GB / Apple UMA** |
| 风格 | Datacenter serving engine | **Apple-native inference runtime** |

**一句话：**  
如果你要的是机房里的吞吐机器，去看 vLLM。  
如果你要的是 **Apple Silicon 上把 35B MoE 真正跑成产品体验**，FlashMLX 才是对的问题。

---

### 与 `TurboQuant+`

| 维度 | TurboQuant+ | FlashMLX |
|---|---|---|
| 核心定位 | **KV codec / quantization backend** | **端到端 memory-policy runtime** |
| 关注点 | PolarQuant / QJL / 低 bit KV 压缩 | **调度 + eviction + quantization + expert residency** |
| 强项 | 压缩率高、作为 backend 很强 | **整体 balance：TTFT / TG / 峰值内存 / 质量** |
| 在 FlashMLX 中的位置 | 可插拔后端 | **系统的一部分，而不是系统本身** |

FlashMLX 自己的移植与 benchmark 已经说明：
- `scored+q8_0` 是默认最优平衡点
- `scored+tq` 可以把 KV 从 153 MB 压到 80 MB
- 但 TG 会从 25.9 tok/s 掉到 16.5 tok/s

**一句话：** TurboQuant+ 是好武器，但 FlashMLX 是整套军队。

---

### 与 H2O / StreamingLLM / SnapKV 这类“只打 eviction”的路线

| 维度 | 纯 eviction / streaming 路线 | FlashMLX |
|---|---|---|
| 主要解决 | token retention / heavy hitters / streaming | **PP 峰值 + TG 常驻 + 批量调度 + 可插拔量化** |
| 能力边界 | 主要是长上下文缓存策略 | **直接把调度与内存形态一起重写** |
| 缺点 | 往往只省一块，不改 TTFT 结构性问题 | **正面改批量推理执行秩序** |

**一句话：** 这类方案是在修 cache；FlashMLX 在重写本地推理的执行模型。

---

### 与 LMCache 这类“KV 存储层”方案

| 维度 | LMCache | FlashMLX |
|---|---|---|
| 主问题 | 跨请求 / 跨实例 / 跨引擎 KV 复用 | **单机 Apple Silicon 上的端到端推理体验** |
| 场景 | 数据中心、长前缀复用、分离式 prefill | **本地 batch 推理、长上下文、35B MoE** |
| 本质 | KV cache storage layer | **runtime scheduler + memory policy stack** |

**一句话：** LMCache 是“存储基础设施”，FlashMLX 是“本地推理执行系统”。

---

## 为什么 FlashMLX 在 Apple Silicon 上更有意思

因为它不是把 CUDA 思路翻译一遍，而是按 Apple Silicon 的物理现实设计：

- **UMA**：CPU cache / GPU hot path 可以重新分工
- **Metal cache 行为**：prefill chunk 大小会直接影响 decode hit rate
- **MoE 本地运行**：不是简单地“都塞进显存”，而是动态热池 / 冷池
- **本地交互体验**：TTFT 比“实验室里单项吞吐”更重要

很多项目默认平台是 NVIDIA。  
**FlashMLX 默认平台是 M4 Pro 48GB 这种真实机器。**

这就是它不一样的地方。

---

## 设计哲学

### 1. 热路径零废话
- 不把复杂控制逻辑塞进 decode 热路径
- 能搬到 cold path 的，都搬走
- 先杀同步，再谈算子

### 2. 不迷信单一论文
- 有用就移植
- 不好用就降级
- 不兼容就废弃
- benchmark 比故事更重要

### 3. 默认追求端到端 balance
- 不是只看压缩率
- 不是只看单点 tok/s
- 不是只看某个漂亮图
- 是看 **TTFT / TG / Peak / Quality / Complexity** 的整体最优

### 4. Apple Silicon 不是“平替 CUDA”
它有自己的规律，就应该有自己的 runtime。

---

## 当前系统组成

### Active Stack
- **Chunked Interleaved Scheduling**
- **Expert Offloading + Compact Pool**
- **Attention Matching / Scored Eviction**
- **Flat Buffer Quantization (bf16 / q8_0 / q4_0 / TurboQuant)**
- **Budget Gating / Runtime Monitoring / Layer Scheduling**

### Engineering Attitude
- 保留 benchmark
- 保留墓碑（dead approaches）
- 明确降级策略
- 不把实验结果包装成神话

---

## 什么时候该用 FlashMLX

### 你应该用它，如果你：
- 在 Apple Silicon 上跑本地 LLM
- 关心 TTFT，而不只关心单项吞吐
- 要跑长上下文
- 要把 35B MoE 跑到真正可用
- 想看一个“runtime / memory policy”级别的开源项目

### 你可能不需要它，如果你：
- 只做 NVIDIA 数据中心 serving
- 主要关心多节点、多租户、高并发 API 服务
- 只想要一个简单、通用、默认稳定的基础推理包

---

## 项目状态

FlashMLX 的目标不是“做一个看起来厉害的 benchmark artifact”。  
它的目标是：

> **把 Apple Silicon 上的本地推理，从“能跑”推进到“像真正的系统软件一样可优化、可组合、可扩展”。**

当前阶段，FlashMLX 已经验证了：
- 35B MoE 在 M4 Pro 48GB 上不仅能跑，而且可以显著缩短 TTFT、降低峰值、提升 TG
- 纯量化不是答案，调度 + eviction + residency 才是关键
- Apple Silicon 值得拥有一条自己的 inference runtime 路线

---

## 快速开始

```python
from mlx_lm import load
from mlx_lm.models.expert_offload import patch_model_for_offload, FlashBatchGenerator

model, tokenizer = load("your-moe-model-path")
ctx = patch_model_for_offload(model, "your-moe-model-path")

gen = FlashBatchGenerator(
    model,
    ctx,
    max_tokens=512,
    completion_batch_size=4,
)
```

---

## 路线图

### Next
- 更强的 fused compute path
- 更成熟的 TurboQuant / backend strategy ladder
- 更系统化的 batch policy tuning
- 更明确的 API / packaging / docs 收口
- 更进一步的 Apple Silicon 特化优化

### Long-term
**FlashMLX 的终局，不是一个优化集合。**  
而是：

# **Apple Silicon 的本地推理基础设施。**

---

## 最后一句话

**不是所有推理系统都该长得像 vLLM。**  
**也不是所有本地优化都只配当“小技巧”。**

FlashMLX 代表的是另一条路：

# **从 Apple Silicon 出发，重写本地 LLM 推理的执行秩序。**

如果你也相信这件事值得做，欢迎 star、benchmark、提 issue，或者直接一起把它打磨成真正的下一代本地推理 runtime。
