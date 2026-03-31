# FlashMLX v2.0

**在 Apple Silicon 上把 35B MoE 推理内存砍半，吞吐拉满 70%，首 Token 快 4 倍**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-0.31+-green.svg)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> v1.0 用三条路线把 LLM 推理的每一块内存都吃掉了。
> v2.0 又问了一个问题：**如果 prefill 和 decode 不再是先后关系，而是交替进行呢？**
>
> 答案是：吞吐 +70%，首 Token 延迟 -74%，内存峰值 -51%，质量零损失。

## 16K 上下文，Batch=4，35B MoE 模型——一张表说完

| 指标 | 社区版 mlx-lm | FlashMLX v2.0 | 提升 |
|:---:|:---:|:---:|:---:|
| **解码吞吐 (TG)** | 115.8 tok/s | **196.9 tok/s** | **+70.0%** |
| **首 Token 延迟 (TTFT)** | 82.0s | **21.1s** | **-74.3%** |
| **GPU 内存峰值** | 28.01 GB | **13.78 GB** | **-50.8%** |
| **模型显存占用** | 18.21 GB | **11.42 GB** | **-37.3%** |
| **生成质量** | 4/4 PASS | 4/4 PASS | **无损** |

> 测试环境：Qwen3.5-35B-A3B (6-bit, 256 experts/layer x 40 MoE layers) | Apple M4 Pro 48GB

---

## 从 v1.0 到 v2.0：撞墙与顿悟

### v1.0 做了什么

三条路线，25 个创新，把推理内存的每一块都压了下来：

| 路线 | 目标 | 方法 | 成果 |
|------|------|------|------|
| 参数内存 | MoE 专家权重 | Expert Offloading + Compact Pool | 18.21 → 9.77 GB (**-46%**) |
| PP 内存 | Prefill 激活峰值 | Chunked Prefill + Streaming Eviction | 5,079 → 774 MB (**-85%**) |
| TG 内存 | KV Cache 累积 | Scored P2 + Pluggable Flat Buffer | 2,454 → 80 MB (**-97%**) |

但 v1.0 的所有测试都是**单请求**。当我们把 batch 开到 4，问题来了。

### v1.0 撞上的墙

传统批量推理的流程：

```
Step 1: Prefill 所有 4 个请求的完整 prompt（4 x 16K = 64K tokens）
        └── 内存峰值 = 4 个请求的 activation 叠加 = 28 GB
        └── 时间 = 82 秒（用户干等）
Step 2: 全部 prefill 完成后，才开始 decode
        └── 用户看到第一个字：82 秒后
```

三个致命问题：
1. **内存叠加**：4 个请求的 prefill activation 同时存在 = 峰值爆炸
2. **首 Token 灾难**：第一个用户必须等最后一个请求 prefill 完
3. **GPU 饥饿**：82 秒的 prefill 期间，decode 单元完全闲置

我们试过更激进的卸载、更极致的压缩。直到我们问自己：**这堵墙，为什么一定要撞上去？**

---

## v2.0 核心创新：分块交错调度

### 一个简单的想法

如果 prefill 和 decode 不是"先后关系"，而是像操作系统时间片一样**交替执行**呢？

```
传统调度：
  [====== Prefill ALL 82s ======][Decode...Decode...Decode...]
  用户等 82 秒才看到第一个字

FlashMLX v2.0 调度：
  [Decode ALL][Prefill chunk 0-511][Decode ALL][Prefill 512-1023][Decode ALL]...
  用户 5 秒就看到第一个字，后续 chunk 在背后悄悄进行
```

每个调度 tick 的动作：

```python
while not all_done:
    # Phase 1: 所有活跃请求先走一步 decode（出 1 token）
    decode_all_active_requests()

    # Phase 2: 挑一个待处理的 prompt，只 prefill 512 个 token
    if has_pending_and_memory_ok():
        prefill_one_chunk(size=512)

    # Phase 3: 维护专家池热度
    maintain_expert_pool()
```

### 三重颠覆效果

| 传统方案 | FlashMLX v2.0 | 为什么 |
|----------|--------------|--------|
| TTFT = 全 batch prefill 时间 | TTFT = 1 个 chunk 的时间 | 第一个请求只需处理 512 tokens 就开始出字 |
| 内存峰值 = N 个请求叠加 | 内存峰值 = 1 个 chunk | 同一时刻只有一个 chunk 的 activation |
| Decode 等 prefill 全部完成 | Decode 和 prefill 交替 | GPU 利用率更高 |

---

## 为什么 chunk=512 是 Killer 参数

这是整个 v2.0 里最反直觉的发现。

我们做了完整的参数扫描（Turbo Sweep），测试了 chunk_size 从 512 到 4096：

| chunk_size | TG 吞吐 (tok/s) | 解码步延迟 | 内存峰值 | 质量 |
|:---:|:---:|:---:|:---:|:---:|
| 4096 | 131.4 | 30.4ms | 17.01G | 4/4 |
| 2048 | 136.5 | 29.3ms | 15.15G | 4/4 |
| 1024 | 155.6 | 25.7ms | 14.36G | 4/4 |
| **512** | **198.3** | **20.2ms** | **13.99G** | **4/4** |

**chunk 越小，decode 越快。chunk=512 比 chunk=4096 快了 50%。**

### 原理

直觉上，大 chunk 应该让 prefill 更高效（更好的算术强度）。但实测相反。

关键在于 **GPU 缓存污染**：

```
大 chunk (4096 tokens):
  Prefill 产生海量 activation → 占满 GPU L2 cache
  → 紧接着 decode 需要读取 KV Cache + 模型参数
  → 但 cache 被 activation 占满了 → cache miss 暴增
  → decode 步延迟 30.4ms

小 chunk (512 tokens):
  Prefill 产生少量 activation → 占用少量 cache
  → decode 的 KV Cache + 模型参数仍在 cache 中
  → cache hit rate 高 → 内存访问紧凑
  → decode 步延迟 20.2ms (-33%)
```

**"慢炖"哲学**：不要用大火（大 chunk）一次性烧干缓存，用文火（小 chunk）慢炖，让 decode 和 prefill 的数据都能共存于缓存中。

---

## 完整性能数据

### 全 Context 长度对比（Batch=4, Qwen3.5-35B-A3B, M4 Pro 48GB）

| Context | 方案 | PP (tok/s) | TG (tok/s) | TTFT | 内存峰值 | 质量 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 4K | 社区版 | 947 | 150.5 | 17.0s | 24.63G | 4/4 |
| 4K | **FlashMLX** | 950 | **184.3** | **4.9s** | **12.60G** | 4/4 |
| 8K | 社区版 | 883 | 137.7 | 36.3s | 25.64G | 4/4 |
| 8K | **FlashMLX** | 851 | **177.4** | **9.7s** | **12.98G** | 4/4 |
| 16K | 社区版 | 779 | 115.8 | 82.0s | 28.01G | 4/4 |
| 16K | **FlashMLX** | 775 | **196.9** | **21.1s** | **13.78G** | 4/4 |

### 改进幅度

| Context | TG 提升 | TTFT 提升 | 内存峰值节省 |
|:---:|:---:|:---:|:---:|
| 4K | +22.5% | -70.9% | -12.03G (49%) |
| 8K | +28.8% | -73.3% | -12.66G (49%) |
| **16K** | **+70.0%** | **-74.3%** | **-14.23G (51%)** |

> Context 越长，FlashMLX 优势越大。16K 下社区版 TG 被 KV Cache 拖到 115 tok/s，FlashMLX 通过更紧凑的调度维持 197 tok/s。

### KV Cache 可插拔量化配置（单请求, Qwen3-8B, M4 Pro 48GB）

v1.1 新增 TurboQuant (PolarQuant PQ4, ICLR 2026) — 第 4 级 flat buffer 量化。

| 配置 | Ctx | PP tok/s | TG tok/s | TTOF | KV PP Peak | KV TG | KV 节省 |
|------|-----|----------|----------|------|------------|-------|---------|
| standard | 16K | 347.2 | 21.0 | 47.02s | 3,006 MB | 2,454 MB | — |
| scored+bf16 | 16K | 378.3 | 28.3 | 43.15s | 976 MB | 302 MB | -88% |
| **scored+q8_0** | **16K** | **406.7** | **25.9** | **40.14s** | **976 MB** | **153 MB** | **-94%** |
| scored+q4_0 | 16K | 422.1 | 18.7 | 38.68s | 976 MB | 85 MB | -97% |
| scored+tq | 16K | 431.8 | 16.5 | 37.81s | 976 MB | 80 MB | -97% |

**推荐**：`scored_pq + q8_0` — TG +23%、KV -94%、最佳性能/内存平衡。TurboQuant 仅在极端内存受限场景值得（多省 73 MB，换 36% TG 代价）。
 为什么 scored+q8_0 比 standard 更快                                                                 
                                                                                                     
  每步 TG 做的事：读模型参数 + 读 KV Cache + attention 计算。                                         
                                                                                                      
  16K context 下：                                                                                    
                                                                                                      
  standard:     读 2,454 MB KV Cache → attention 扫全部 16K tokens                                    
  scored+q8_0:  读   153 MB KV Cache → dequant → attention 扫 ~1,600 tokens                           
                                                                                                      
  速度快不是因为 q8_0 快，而是因为 scored_pq 把 16K tokens 淘汰到了 ~1,600 tokens。                   
                                                                                             
  AM eviction 干掉了 90% 的 tokens，所以：                                                            
  - 内存读取：2,454 → 153 MB（省了 2.3 GB 带宽）                                                      
  - attention 计算量：16K → ~1.6K（少了 10x）
  - Q8 dequant 代价：对 153 MB 做一次乘法 ≈ 微不足道                                                  
                                                                                                      
  数字说话                                                                                            
                                                                                                      
  看 bf16 vs q8_0 的对比就能隔离出纯 dequant 开销：                                                   
                                                                                                      
  ┌─────────────┬──────┬─────────┬──────────────────────────┐                                         
  │    配置     │  TG  │ KV 内存 │         差异来源         │                                         
  ├─────────────┼──────┼─────────┼──────────────────────────┤
  │ scored+bf16 │ 28.3 │ 302 MB  │ 零 dequant，但读 2x 数据 │
  ├─────────────┼──────┼─────────┼──────────────────────────┤
  │ scored+q8_0 │ 25.9 │ 153 MB  │ 有 dequant，但读一半数据 │
  └─────────────┴──────┴─────────┴──────────────────────────┘

  bf16 快 9%，说明 Q8 dequant 的代价约等于省掉一半带宽的收益 — 两者基本打平，Q8 略输。

  但 standard 慢是因为它要读 16x 的数据量（2,454 vs 153 MB），这个差距远大于 dequant 开销。

  一句话

  速度提升来自 scored_pq 的 token eviction（-90% tokens），不是来自量化。Q8 dequant
  只是一次乘法，代价几乎可忽略。

  TurboQuant 慢的原因也是同理 — 它的 dequant 涉及 Haar 逆旋转 + Lloyd-Max 查表 + 拆包，比 Q8
  的一次乘法重得多，所以 25.9 → 16.5。

---

## 架构全景

### 调度时序

```
Tick 1:  [Decode R1,R2,R3]        → [Prefill Prompt4 chunk 0-511]
Tick 2:  [Decode R1,R2,R3,R4]     → [Prefill Prompt4 chunk 512-1023]
Tick 3:  [Decode R1,R2,R3,R4]     → [Prefill Prompt4 chunk 1024-1535]
  ...
Tick N:  [Decode R1,R2,R3,R4]     → [Prefill Prompt4 完成! 开始 Prompt3...]
```

### 三级专家管理

```
请求队列                      调度器                          GPU
┌─────────┐              ┌──────────────┐             ┌──────────────┐
│ Prompt 1 │──────┐      │  每个 tick:    │             │  Hot Pool    │
│ Prompt 2 │──────┤      │  1. Decode ALL │────────────▶│  (~153 exp)  │
│ Prompt 3 │──────┤      │  2. Prefill 1  │             │  ~0ms 访问   │
│ Prompt 4 │──────┘      │     chunk(512) │             │              │
└─────────┘              └──────────────┘             └──────┬───────┘
                                                             │ miss
                              CPU                            │
                         ┌──────────────┐                    ▼
                         │  Warm Cache   │◄──── UMA memcpy ~0.1ms
                         │  (~145 exp)   │
                         │              │──── miss ──▶ SSD Cold Storage
                         └──────────────┘              (全部 256 exp)
                                                        pread ~2-5ms
```

### 四大智能守护

| 守护机制 | 触发条件 | 作用 |
|----------|----------|------|
| **P3 内存预算门控** | GPU headroom < 2GB | 自动跳过新 prefill，防 OOM，显存恢复后继续 |
| **P4 命中率监控** | 热池命中率 < 70% | 发出警告，帮助诊断 pool_size 配置 |
| **A4 动态剪枝** | Gate entropy 变化 | 低 entropy = 少用专家提速，高 entropy = 保持 top-k 保质 |
| **A3 Pipeline 维护** | 每 4 步自动执行 | 促进热门专家、驱逐冷门专家、更新预测 |

---

## v1.0 + v2.0：30 项技术创新

### v2.0 新增创新 (5)

| # | 创新 | 解决的问题 |
|---|------|-----------|
| 1 | **分块交错调度** (Chunked Interleaved Scheduling) | batch prefill 内存叠加 + TTFT 灾难 |
| 2 | **Turbo 参数优化** (chunk=512, maint=4) | GPU 缓存污染导致 decode 变慢 |
| 3 | **P3 内存预算门控** | 长 context 高并发下的 OOM 风险 |
| 4 | **P4 命中率实时监控** | 专家池配置不当导致的性能退化 |
| 5 | **A4 动态剪枝增强** | entropy-adaptive 的 k 调整 |

### v1.1 新增：TurboQuant 可插拔量化

| # | 创新 | 解决的问题 |
|---|------|-----------|
| 1 | **PolarQuant PQ4** (ICLR 2026) | Haar 旋转 + Lloyd-Max 4-bit 标量量化，KV 压缩 3.8x |
| 2 | **可插拔 flat buffer 架构** | bf16/Q8/Q4/TurboQuant 统一接口，一行切换 |
| 3 | **head_dim 自动降级** | head_dim < 128 自动回退 q4_0，CLT 收敛保护 |

### v1.0 设计创新 (10)

| # | 创新 | 路线 | 成果 |
|---|------|------|------|
| 1 | Two-Phase Compact Pool | 参数 | 参数内存 -46%，TG 零惩罚 |
| 2 | Speculative Execution (clamp, no sentinel) | 参数 | 消除 GPU→CPU 同步 (5.6→92.8 tok/s) |
| 3 | UMA-Aware CPU Cache | 参数 | numpy→mx 6us 传输 |
| 4 | Chunked Prefill + Streaming Eviction | PP | O(N^2) → O(1) 内存 |
| 5 | On-Policy 分阶段校准 | TG | 18/36 → 36/36 层全压缩 |
| 6 | Bounded Beta Optimization | TG | 修复论文隐含假设 |
| 7 | Scored P2 一次性 Promotion | TG | 避免 PP 内存翻倍 |
| 8 | Q8_0 Flat Buffer | TG | 6% 速度换 50% 内存 |
| 9 | 可插拔量化策略 | TG | Q4/Q8/PolarQuant/TurboQuant 统一接口 |
| 10 | 自动校准系统 | TG | 新模型零配置 |

### 系统工程优化 (15)

| # | 优化 | 去掉它会怎样 |
|---|------|------------|
| S1 | Gather-Sort Cache Locality | gather_qmm cache miss 暴增 |
| S2 | Three-Tier Hierarchical Cache | miss 直接打 SSD (240us vs 6us) |
| S3 | Telemetry-Driven Prediction | compact 选错专家，频繁 miss |
| S4 | Dynamic Pool Self-Optimization | 长对话 expert 漂移后性能退化 |
| S5 | Background Prefetch Engine | miss 恢复延迟 40x 退化 |
| S6 | Regime Auto-Detection | 用户手动选 streaming/three-tier/full-gpu |
| S7 | Identity Path Detection | PP 白做 mx.take remap |
| S8 | Async SSD→CPU Population | CPU cache 填充阻塞 GPU |
| S9 | Deferred PP Index Collection | PP 2560 次 GPU→CPU sync |
| S10 | Pool Miss Mini-Pool Fallback | non-clamp miss 无精确恢复 |
| S11-S15 | KV Cache 工程优化 | 量化感知 eviction、RoPE 修正等 |

---

## 踩坑墓地：9 个被放弃的方案

每个死掉的方案都教会了我们活下来的方案学不到的东西：

| 方案 | 死因 | 教训 |
|------|------|------|
| Pipeline L0→L1→L2 三层缓存 | PP 内存**翻倍** | 简单(一次性)胜过复杂(三层) |
| AM 压缩重建 | Beta 补偿不稳定，误差累积 | AM 适合**评分**，不适合**重建** |
| 无界 Beta solver | Beta 飞到 [-171, +221] | 数值优化必须 bound |
| Off-policy 校准 | 第 18 层以后分布漂移 | On-policy 是多阶段流水线的刚需 |
| `.item()` sentinel 检测 | 5.6 tok/s（40x GPU→CPU sync） | 把检查从热路径移到冷路径 |
| AM on 混合架构 (SSM+Attention) | 全面乱码 | 架构级不兼容，不是调参问题 |
| Q4_0 flat buffer | TG 速度 -39% | KV 只占 6% 带宽，省带宽没意义 |
| Discovery phase `.tolist()` | PP 阶段 2560 次 GPU→CPU sync | 全加载后裁剪 > 逐个发现 |
| 精确 miss 处理 | 为 0.1% 的 miss 惩罚 99.9% 的 hit | 热路径不做条件判断 |

---

## 快速开始

### 安装

```bash
git clone https://github.com/lisihao/FlashMLX.git
cd FlashMLX
pip install -e .
```

### 使用 FlashBatchGenerator（MoE 批量推理）

```python
from mlx_lm import load
from mlx_lm.models.expert_offload import patch_model_for_offload, FlashBatchGenerator

# 加载模型 + 启用专家卸载
model, tokenizer = load("your-moe-model-path")
ctx = patch_model_for_offload(model, "your-moe-model-path")

# 创建批处理生成器（分块交错调度默认开启）
gen = FlashBatchGenerator(
    model, ctx,
    max_tokens=512,
    completion_batch_size=4,
    # Turbo 默认值已内置：chunk=512, maint=4
)

# 插入多个请求
prompts = [tokenizer.encode(p) for p in your_prompts]
uids = gen.insert(prompts, max_tokens=[512] * len(prompts))

# 流式获取结果
while True:
    responses = gen.next()
    for r in responses:
        print(r.token, end="")
        if r.finish_reason is not None:
            print(f"\nRequest {r.uid} complete")
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `interleaved` | `True` | 启用分块交错调度 |
| `interleaved_chunk_size` | `512` | 每次 prefill 的 chunk 大小 |
| `maintenance_interval` | `4` | 专家池维护频率（每 N 步） |

### KV Cache 压缩（Dense Transformer 模型）

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-8B-Instruct")

# 一行代码启用 KV Cache 压缩
result = generate(model, tokenizer, prompt="Your long prompt...",
                  kv_cache="scored_pq", kv_flat_quant="q8_0")

# 可选量化级别：None (bf16), "q8_0" (推荐), "q4_0", "turboquant" (极限压缩)
```

### KV Cache 配置指南

| 配置 | KV 内存 | TG 速度 | PP 速度 | 适用场景 |
|------|---------|---------|---------|----------|
| `scored_pq` (bf16) | -88% @ 16K | +35% | +9% | 最大 TG 速度 |
| `scored_pq` + `q8_0` | -94% @ 16K | +23% | +17% | **推荐默认** |
| `scored_pq` + `q4_0` | -97% @ 16K | -11% | +22% | 内存优先 |
| `scored_pq` + `turboquant` | -97% @ 16K | -21% | +24% | 极限压缩 |

---

## 研发历程

```
Day 1-2:   AM 单层完美，36 层乱码。找到 rank-deficient beta 矩阵。
Day 3:     差点放弃。发现两个关键 bug（无界 beta、偏差采样）。
Day 3-4:   On-Policy 校准 -- 论文没提的关键洞察。
Day 5-6:   Triple-Layer Cache → 失败 → Scored P2 一次性 Promotion。
Day 7:     Chunked Prefill + Streaming Eviction = O(1) PP 内存。
Day 8-9:   Expert Offloading: .item() (5.6 tok/s) → speculative clamp (92.8 tok/s)。
Day 10-12: 三级专家管理 + 动态剪枝 + prefetch engine。
Day 13:    分块交错调度: batch=4 首次跑通，TTFT -74%。
Day 14:    Turbo Sweep: 发现 chunk=512 是 killer，TG +70%。
Day 15:    TurboQuant 移植: PolarQuant PQ4, KV 80 MB (-97%)。
Day 16:    v2.0 发布。30+ 项创新，9 个墓碑，0 个 mock。
```

详细技术文章（25 个创新 + 15 个工程优化 + 9 个墓碑的完整故事）：[从论文到生产](.solar/ARTICLE-week-in-review.md)

---

## 致谢

FlashMLX 基于 Apple 的 [mlx-lm](https://github.com/ml-explore/mlx-lm) 构建。感谢 MLX 团队提供的优秀底层框架。

---

# FlashMLX v2.0 (English)

**Cut 35B MoE inference memory in half, boost throughput by 70%, 4x faster first token -- on Apple Silicon**

### Core Innovation: Chunked Interleaved Scheduling

Each scheduler tick: decode ALL active requests (1 token each) -> prefill 1 chunk (512 tokens) of next pending prompt. Prefill memory never stacks across requests.

### Results (Qwen3.5-35B-A3B, M4 Pro 48GB, Batch=4)

| Context | Mode | PP tok/s | TG tok/s | TTFT | Peak Memory | Quality |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 4K | Community | 947 | 150.5 | 17.0s | 24.63G | 4/4 |
| 4K | **FlashMLX** | 950 | **184.3** | **4.9s** | **12.60G** | 4/4 |
| 8K | Community | 883 | 137.7 | 36.3s | 25.64G | 4/4 |
| 8K | **FlashMLX** | 851 | **177.4** | **9.7s** | **12.98G** | 4/4 |
| 16K | Community | 779 | 115.8 | 82.0s | 28.01G | 4/4 |
| 16K | **FlashMLX** | 775 | **196.9** | **21.1s** | **13.78G** | 4/4 |

**16K worst case**: TG **+70%** | TTFT **-74%** | Peak **-51%** | Quality: ALL PASS

### KV Cache Pluggable Quantization (Single Request, Qwen3-8B, 16K)

| Config | TG tok/s | KV TG | KV Save | Use Case |
|--------|----------|-------|---------|----------|
| scored+bf16 | 28.3 | 302 MB | -88% | Speed king |
| **scored+q8_0** | **25.9** | **153 MB** | **-94%** | **Recommended** |
| scored+q4_0 | 18.7 | 85 MB | -97% | Memory-critical |
| scored+turboquant | 16.5 | 80 MB | -97% | Max compression |

### Why chunk=512?

Large prefill chunks pollute GPU L2 cache with activation data, causing decode-phase cache misses. Smaller chunks leave room for KV cache and model parameters to stay resident. Sweet spot: 512 tokens.

### Quick Start

```python
from mlx_lm import load
from mlx_lm.models.expert_offload import patch_model_for_offload, FlashBatchGenerator

model, tokenizer = load("your-moe-model-path")
ctx = patch_model_for_offload(model, "your-moe-model-path")
gen = FlashBatchGenerator(model, ctx, max_tokens=512, completion_batch_size=4)

uids = gen.insert(encoded_prompts, max_tokens=[512] * 4)
while True:
    for r in gen.next():
        if r.finish_reason: print(f"Done: {r.uid}")
```

### 30+ Innovations, 9 Gravestones, 0 Mocks

Built on Apple's [mlx-lm](https://github.com/ml-explore/mlx-lm). Full technical article: [From Paper to Production](.solar/ARTICLE-week-in-review.md)

---

*FlashMLX v2.0 — Built for Apple Silicon, March 2026*
*Batch data: Qwen3.5-35B-A3B (6-bit) on Apple M4 Pro 48GB*
*KV Cache data: Qwen3-8B-MLX (Q8) on Apple M4 Pro 48GB, generate_step methodology*
*All benchmarks: serial execution, isolated subprocess*
