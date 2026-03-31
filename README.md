# FlashMLX

> **Apple Silicon 上的本地 LLM 推理，不该只是“能跑”。它应该更快出首 Token、更高 decode 吞吐、更低峰值内存、对 35B MoE 真正可用。**

**FlashMLX** 不是又一个 KV cache 小技巧，也不是把 CUDA 世界的 serving 方案硬翻成 Metal。  
它是一套面向 **Apple Silicon** 的 **memory-policy runtime**：把 **参数内存、Prefill 峰值、Decode 常驻、批量调度、质量保护、平台特性** 当成一个整体来重写。

---

## 一句话定位

### 如果 vLLM 是为 CUDA 数据中心 serving 写的，
### 那 FlashMLX 就是为 Apple Silicon 本地长上下文、35B MoE 与真实交互体验写的 runtime。

---

## 为什么是 FlashMLX

本地长上下文推理的真正瓶颈，从来不只是一件事：

- **参数内存**：35B MoE 专家权重常驻，浪费巨大
- **PP 峰值**：prefill activation 和 KV cache 一起把显存顶爆
- **TG 常驻**：长上下文 decode 需要读取全历史，越长越慢
- **TTFT 体验**：批量请求下，第一个用户要等所有请求 prefill 完才能看到首 Token
- **Apple Silicon 物理现实**：UMA / Metal / cache 行为和 CUDA 不是一回事

**FlashMLX 的答案不是单点 trick，而是完整系统打法：**

1. **Parameter Memory** — Expert Offloading + Compact Pool  
2. **PP Memory** — Chunked Prefill + Streaming Eviction  
3. **TG Memory** — Scored Promotion + Flat Buffer Quantization  
4. **Batch Scheduling** — Chunked Interleaved Scheduling  
5. **Support Systems** — Calibration / Quantizer Ladder / Runtime Policy / Fallback  

换句话说：

> **FlashMLX 不是在做一个“压缩算法”。它在做 Apple Silicon 上的本地推理执行系统。**

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

> 测试环境：Qwen3.5-35B-A3B / Apple M4 Pro 48GB

### FlashMLX v1.x：32K / Qwen3-8B / Apple M4 Pro 48GB

| 指标 | 标准路径 | FlashMLX | 变化 |
|---|---:|---:|---:|
| PP Speed | 213.6 tok/s | **372.8 tok/s** | **+74.5%** |
| PP Peak Memory | 5,079 MB | **774 MB** | **-84.8%** |
| TG Speed | 16.0 tok/s | **24.7 tok/s** | **+54.4%** |
| TG KV Memory | 4,572 MB | **147 MB** | **-96.8%** |
| TTOF | 151.7s | **86.9s** | **-42.7%** |
| Quality | PASS | **PASS** | 无损 |

### Parameter Memory：Qwen3.5-35B-A3B / Q4 / 256 experts per layer

| 配置 | TG 吞吐 | 内存 | 节省 |
|---|---:|---:|---:|
| No offload | 90.0 tok/s | 18.21 GB | — |
| Compact pool=192 | 90.9 tok/s | 13.99 GB | -23% |
| **Compact pool=128** | **92.8 tok/s** | **9.77 GB** | **-46%** |

> 这不是“省内存但变慢”，而是 **更省内存，还更快**。根因不是魔法，而是更好的 cache locality 和对 GPU↔CPU sync 的消灭。

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
| TG Memory | 长上下文 KV cache 拖慢 decode | Scored Promotion + Flat Buffer | 4,572 → **147 MB** |

**这不是“某个 cache trick”能解释的结果。**  
这是把推理内存当成一个系统问题来拆解。

---

### 2) 它真正解决的是本地 batch 推理的死穴：TTFT

传统本地批量推理是这样跑的：

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
FlashMLX 把这件事做成了真实系统实验，结论相反：**太大的 chunk 会污染 cache，拖累 decode 热路径**。

| chunk_size | TG 吞吐 | 解码步延迟 | 峰值内存 | 质量 |
|---:|---:|---:|---:|---:|
| 4096 | 131.4 tok/s | 30.4 ms | 17.01 GB | 4/4 |
| 2048 | 136.5 tok/s | 29.3 ms | 15.15 GB | 4/4 |
| 1024 | 155.6 tok/s | 25.7 ms | 14.36 GB | 4/4 |
| **512** | **198.3 tok/s** | **20.2 ms** | **13.99 GB** | **4/4** |

**为什么？**  
因为大 chunk 会把 activation 塞满 GPU cache，接下来 decode 读取 KV cache + 参数时 cache miss 暴增；小 chunk 反而让 decode 热数据留在 cache 里。  
这不是纸面理论，这是 **Apple Silicon 上真实跑出来的系统行为**。

---

## 与竞争路线的区别

> 下面不是“别人不行”，而是 **FlashMLX 解决的问题边界更完整、耦合关系更深**。

### 与社区版 `mlx-lm`

| 维度 | 社区版 mlx-lm | FlashMLX |
|---|---|---|
| 目标 | 通用 Apple Silicon LLM 推理与微调 | **Apple Silicon 本地长上下文 / 35B MoE / batch 推理 runtime** |
| 调度 | 传统 prefill-then-decode | **Chunked Interleaved Scheduling** |
| MoE 内存 | 通用路径 | **Expert Offloading + Compact Pool** |
| 长上下文 KV | 基础能力 | **Scored eviction + flat buffer + 可插拔量化** |
| 优先级 | 通用性 / 生态 | **端到端 TTFT / 峰值内存 / TG balance** |

**一句话：** 社区版 mlx-lm 是底座，FlashMLX 是在 Apple Silicon 上把“本地推理体验”狠狠干到位的系统层。

### 与 `vLLM` / `vllm-metal`

| 维度 | vLLM | FlashMLX |
|---|---|---|
| 主战场 | CUDA / HIP 数据中心 serving | **Apple Silicon 本地推理** |
| 核心武器 | PagedAttention、continuous batching、serving 吞吐 | **三维内存系统 + 交错调度 + MoE 专家池** |
| KV 管理 | 强，尤其在高并发 serving | 强，但更关注**本地长上下文 + TTFT + UMA 行为** |
| 平台假设 | 高并发服务端 | **单机本地 / M4 Pro 48GB / Apple UMA** |
| 风格 | Datacenter serving engine | **Apple-native inference runtime** |

**一句话：** 如果你要的是机房里的吞吐机器，去看 vLLM。  
如果你要的是 **Apple Silicon 上把 35B MoE 真正跑成产品体验**，FlashMLX 才是对的问题。

### 与 `TurboQuant+`

| 维度 | TurboQuant+ | FlashMLX |
|---|---|---|
| 核心定位 | KV codec / quantization backend | **端到端 memory-policy runtime** |
| 关注点 | PolarQuant / QJL / 低 bit KV 压缩 | **调度 + eviction + quantization + expert residency** |
| 强项 | 压缩率高、作为 backend 很强 | **整体 balance：TTFT / TG / 峰值内存 / 质量** |
| 在 FlashMLX 中的位置 | 可插拔后端 | **系统的一部分，而不是系统本身** |

**一句话：** TurboQuant+ 是一把好刀，但 FlashMLX 是整套作战体系。

### 与 H2O / StreamingLLM / SnapKV 一类“只打 eviction”的路线

| 维度 | 纯 eviction / streaming 路线 | FlashMLX |
|---|---|---|
| 主要解决 | token retention / heavy hitters / streaming | **PP 峰值 + TG 常驻 + 批量调度 + 可插拔量化** |
| 能力边界 | 主要是长上下文缓存策略 | **直接把调度与内存形态一起重写** |
| 缺点 | 往往只省一块，不改 TTFT 结构性问题 | **正面改批量推理执行秩序** |

**一句话：** 这类方案是在修 cache；FlashMLX 在重写本地推理的执行模型。

### 与 LMCache 这类“KV 存储层”方案

| 维度 | LMCache | FlashMLX |
|---|---|---|
| 主问题 | 跨请求 / 跨实例 / 跨引擎 KV 复用 | **单机 Apple Silicon 上的端到端推理体验** |
| 场景 | 数据中心、长前缀复用、分离式 prefill | **本地 batch 推理、长上下文、35B MoE** |
| 本质 | KV cache storage layer | **runtime scheduler + memory policy stack** |

**一句话：** LMCache 是“存储基础设施”，FlashMLX 是“本地推理执行系统”。

---

## 10 个大颗粒创新：原理、机制、效果

> 这 10 个不是“小调参”，而是决定 FlashMLX 系统边界的设计层创新。

### 1. Two-Phase Compact Pool
**解决什么问题：** MoE 模型在 PP 和 TG 阶段的专家活跃度完全不同，把 256 个 experts 全常驻 GPU 是纯浪费。  
**核心原理：** PP 阶段保持 full pool，零额外开销地收集专家活跃统计；进入 TG 前再按热度收缩成 hot-K pool。  
**机制：** `full pool -> collect activation counts -> compact hot experts -> remap table -> TG use compact pool`。  
**为什么成立：** PP 更看吞吐和身份路径，TG 更看常驻内存和随机访问局部性。两阶段目标不同，不该强行同构。  
**效果：** 在 Qwen3.5-35B-A3B 上，`18.21 GB -> 9.77 GB (-46%)`，TG 还从 `90.0 -> 92.8 tok/s`。  
**真正牛的点：** 不是“省了内存”，而是把 **参数常驻策略** 做成了 runtime policy。

### 2. Speculative Execution (clamp, no sentinel)
**解决什么问题：** MoE expert miss 检测如果放在热路径上，会引入 GPU→CPU 同步，把 lazy eval 打爆。  
**核心原理：** 把 miss 检查从每 token 的热路径挪到 compact 时的冷路径，运行时只做预先约束过的 remap。  
**机制：** `item() sentinel -> mx.minimum -> pre-clamp remap` 三代演化，最终零 `.item()`、零热路径同步。  
**效果：** TG 从 `5.6 -> 28.1 -> 92.8 tok/s`，16x 提升。  
**真正牛的点：** 这是典型的系统高手操作：**move work from hot path to cold path**。

### 3. UMA-Aware CPU Cache
**解决什么问题：** Apple Silicon 上 CPU/GPU 共享内存，不应该照搬“显存/内存二元对立”的 CUDA 思维。  
**核心原理：** 把冷专家放 CPU 侧缓存，但利用 UMA 让 `numpy -> mx.array` 成本足够低。  
**机制：** 热池在 GPU，冷池在 CPU，必要时快速提升，不把每次 miss 都打到 SSD。  
**效果：** 冷专家不再占满 GPU，同时保留可接受的恢复时延。  
**真正牛的点：** 这是 **按 Apple Silicon 的物理现实设计系统**，不是翻译 CUDA 教条。

### 4. Chunked Prefill + Streaming Eviction
**解决什么问题：** 标准 attention 在长上下文 PP 中是 O(N²) 内存，32K prompt 很容易到 5GB+ 峰值。  
**核心原理：** 把 prefill 切成小 chunk，并在 cache 超阈值时动态淘汰冷 token，把 PP 峰值从“跟长度走”改成“被预算约束”。  
**机制：** `chunk=512` + `max_cache=2048` + AM scoring eviction。  
**效果：** `5079 MB -> 774 MB (-85%)`，PP 还从 `213.6 -> 369.1 tok/s`。  
**真正牛的点：** 不是“省一点 PP 内存”，而是把 **PP 内存曲线改写成近似 O(1)**。

### 5. On-Policy Stage-wise Calibration
**解决什么问题：** 多层深网络的 KV 压缩误差会逐层放大，单层有效不等于 36 层有效。  
**核心原理：** 校准必须用“压缩后的真实分布”继续往后推，而不是用原始分布离线拟合。  
**机制：** 分阶段校准，每一阶段都以前一阶段输出为输入；不是 one-shot 离线拟合。  
**效果：** 从只能稳定覆盖 `18/36 layers` 提升到 `36/36 layers`。  
**真正牛的点：** 你不是在复现 paper，而是在**补论文没讲明白的生产约束**。

### 6. Bounded β Optimization
**解决什么问题：** 论文里的 β 求解默认数值稳定，但真实长序列上会发散。  
**核心原理：** 所有在线优化参数都必须有生产可接受的数值边界。  
**机制：** 对 β 显式设边界，防止漂到 `[-171, +221]` 这种灾难区。  
**效果：** 长序列下校准稳定，避免 reconstruction 直接崩坏。  
**真正牛的点：** 这是典型“论文隐含假设 -> 工业约束显性化”。

### 7. Scored P2 One-Shot Promotion
**解决什么问题：** 多级 aging/pipeline cache 看起来优雅，但会带来 PP 双缓冲和复杂管理成本。  
**核心原理：** 在 PP→TG 交界一次性打分、一次性晋升，比全程维护多层 cache 更直接。  
**机制：** `bf16 during PP -> score once at transition -> hot tokens to quantized flat buffer`。  
**效果：** 避免了 pipeline 方案带来的 PP 内存翻倍。  
**真正牛的点：** **简单模型赢复杂模型**，而且是经过墓碑验证后的结论。

### 8. Q8_0 Flat Buffer Quantization
**解决什么问题：** KV 量化不能只追求更低 bit，必须看 bandwidth、dequant 成本和真实 TG 热路径。  
**核心原理：** 当 KV 读只占 TG 总带宽的一小部分时，Q4 的 nibble unpack 成本可能大于节省下来的带宽。  
**机制：** 保留连续 flat buffer + Q8 dequant，只引入极低额外算术成本。  
**效果：** 在 32K 上把 TG KV 压到 `147 MB`，同时保留 `24.7 tok/s`。  
**真正牛的点：** 不是“追低 bit”，而是**选真实系统 sweet spot**。

### 9. Pluggable Quantization Strategies
**解决什么问题：** 不同模型、不同上下文长度、不同平台约束下，单一量化后端不可能永远最优。  
**核心原理：** 把 quantizer 抽象成可切换 backend，由 runtime 选择平衡点，而不是把某个 codec 神化。  
**机制：** 统一接口下支持 `Q4_0 / Q8_0 / PolarQuant / TurboQuant`。  
**效果：** 允许 FlashMLX 把 TurboQuant 当武器而不是宗教。  
**真正牛的点：** 你做的是系统，不是“某个论文的忠实信徒”。

### 10. Auto-Calibration System
**解决什么问题：** 新模型、新层分布、新量化策略如果都要手工调，根本不具备产品化可能。  
**核心原理：** 首次使用自动校准，之后缓存结果，把工程复杂度从“手工调参”降成“首次预热”。  
**机制：** first use calibration，后续直接命中缓存。  
**效果：** 首次约 `~26s`，之后 `<1ms` 级加载。  
**真正牛的点：** 这一步把研究 prototype 推向了**真实产品接口**。

---

## 15 个工程级技术优化：它们为什么重要

> 下面这 15 个点，决定了 FlashMLX 不是“理论上成立”，而是“在真 GPU 上真的快”。

### S1. Gather-Sort Cache Locality
**问题：** gather_qmm 访问分散，cache miss 高，TG 会被随机访存拖死。  
**做法：** 调整 gather / sort 顺序，让访问更局部。  
**价值：** 提升热池访问局部性，TG 吞吐更稳。

### S2. Three-Tier Hierarchical Cache (GPU / CPU / SSD)
**问题：** 如果冷专家一 miss 就直接打 SSD，恢复时延不可接受。  
**做法：** GPU 热池、CPU 温池、SSD 冷池三级分层。  
**价值：** miss 恢复从“巨慢”变成“可控”，把极端情况也纳入系统边界。

### S3. Telemetry-Driven Expert Prediction
**问题：** compact pool 如果选错专家，后续 miss 会反复发生。  
**做法：** 用历史活跃 telemetry 预测下一阶段热专家。  
**价值：** 让 compact pool 更接近真实访问分布。

### S4. Dynamic Pool Self-Optimization
**问题：** 长对话下 expert 分布会漂移，静态 pool 会越来越失配。  
**做法：** 让热池大小/组成动态自优化。  
**价值：** 长会话性能不衰减。

### S5. Background Prefetch Engine
**问题：** miss 恢复如果是同步的，会直接阻塞 decode。  
**做法：** 后台预取未来可能需要的专家。  
**价值：** 把 miss cost 从用户可感知路径移开。

### S6. Regime Auto-Detection
**问题：** 用户不该手工判断自己现在适合 full-GPU、three-tier 还是 streaming。  
**做法：** 根据上下文、内存、模型形态自动判断 regime。  
**价值：** 系统可用性大幅提高。

### S7. Identity Path Detection
**问题：** PP 阶段很多层其实仍走 identity path，不该在每层都白做 remap。  
**做法：** 自动识别 identity path，跳过无意义 `mx.take`。  
**价值：** PP 少掉一堆无意义操作。

### S8. Async SSD→CPU Population
**问题：** compact 期间同步填充 CPU cache，会阻塞 GPU。  
**做法：** SSD→CPU 异步填充。  
**价值：** 减少 compact 期间的 stall。

### S9. Deferred PP Index Collection
**问题：** PP 阶段每步发现一个索引就回传 CPU，会造成海量同步。  
**做法：** 先在 GPU/批量路径里收齐，再统一裁剪。  
**价值：** 避免 `2560` 次级别的 GPU→CPU sync。

### S10. Pool Miss Mini-Pool Fallback
**问题：** 极少数 miss 不值得用复杂热路径处理，但又不能完全无恢复。  
**做法：** 为边缘 miss 准备轻量 fallback mini-pool。  
**价值：** 既不拖累 99.9% 热路径，又不放弃极端正确性。

### S11. Lazy Prefill Threshold
**问题：** 短上下文一上来就量化/压缩，纯属白折腾。  
**做法：** 只有上下文长到值得压缩时才启用。  
**价值：** 短 prompt 不被“高级优化”反向伤害。

### S12. Adaptive Compression Ratio
**问题：** 32K 长序列如果压得太狠，质量会塌。  
**做法：** 根据上下文长度和误差预算调整压缩比。  
**价值：** 不是死命压，而是按质量预算做控制。

### S13. Chunk-Aware Eviction
**问题：** TurboQuant / QJL 这类残差校正方法，在 chunk 边界反复 re-quant 可能放大噪声。  
**做法：** eviction 和 chunk 结构对齐，避免不必要的重复量化。  
**价值：** 把算法级误差和系统级切块逻辑统一起来。

### S14. Vectorized Single-Gather
**问题：** 30 次 GPU dispatch 干 1 件事，纯属调度自杀。  
**做法：** 向量化 gather，把多次小 dispatch 合成一次大操作。  
**价值：** dispatch overhead 大降，热路径更干净。

### S15. RoPE Position Correction
**问题：** token eviction/压缩后，如果位置处理不对，RoPE 会直接失真。  
**做法：** 在 cache 变形之后做位置修正。  
**价值：** 确保“省了内存”不是以位置语义崩坏为代价。

---

## 墓碑系统：FlashMLX 不是靠运气做出来的

真正成熟的系统，不是只有成功案例，还要有**明确杀掉的失败路线**。FlashMLX 的墓碑很值钱，因为它说明这套系统是被现实打磨出来的。

### 9 条被判死刑的路线
1. **Pipeline L0→L1→L2 cache**：结构优雅，但 PP 内存翻倍，直接判死刑。  
2. **AM compress-and-reconstruct**：AM 适合 scoring，不适合 reconstruction。  
3. **Unbounded β solver**：长序列直接发散。  
4. **Off-policy calibration**：后半网络误差不可控。  
5. **`.item()` / precise miss handling on hot path**：同步毁掉吞吐。  
6. **AM on hybrid architecture (Qwen3.5)**：SSM 层会放大 attention 压缩误差。  
7. **Q4_0 flat buffer as default**：压得更狠，但 TG 速度不值。  
8. **Discovery phase `.tolist()`**：每步回传 CPU，系统性慢。  
9. **Precise miss handling on hot path**：0.1% 边缘 case 不该惩罚 99.9% 热路径。  

### FlashMLX 学到的四条铁律
- **看起来优雅，不代表真的更快**
- **论文隐含假设，必须被工业化约束显性化**
- **结构不兼容，不是调参能救的**
- **冷操作绝不能塞进热路径**

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
- 不只看压缩率
- 不只看单点 tok/s
- 不只看某张漂亮图
- 看的是 **TTFT / TG / Peak / Quality / Complexity** 的整体最优

### 4. Apple Silicon 不是“平替 CUDA”
它有自己的规律，就该有自己的 runtime。

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
