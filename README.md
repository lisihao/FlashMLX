# FlashMLX

**Apple Silicon 上的高性能 MoE 推理引擎** | High-Performance MoE Inference on Apple Silicon

基于 [mlx-lm](https://github.com/ml-explore/mlx-lm) 的 fork，专为 Mixture-of-Experts (MoE) 大模型在内存受限的 Apple Silicon 设备上实现高效推理而设计。

---

## 核心创新

### 分块交错调度 (Chunked Interleaved Scheduling)

传统批处理推理在 prefill 阶段一次性处理所有请求的 prompt，导致内存峰值随 batch size 线性叠加。FlashMLX 采用**分块交错调度**策略：

```
每个调度 tick: 解码所有活跃请求 → prefill 1 个 chunk(512 tokens)
```

- Prefill 内存峰值不再随请求数叠加（每次只处理 1 个 chunk）
- 首个请求的 TTFT 大幅缩短（无需等待所有请求 prefill 完成）
- 解码吞吐量显著提升（更小的 chunk 减少 GPU activation 残留，改善缓存局部性）

### 三级专家管理 (Three-Tier Expert Management)

针对 MoE 模型（如 Qwen3.5-35B-A3B，256 experts/layer x 40 layers）的智能专家调度：

| 层级 | 存储 | 延迟 | 容量 |
|------|------|------|------|
| GPU 热池 (Hot Pool) | 统一显存 | ~0ms | ~153 experts/layer |
| CPU 暖缓存 (Warm Cache) | 系统内存 | ~0.1ms | ~145 experts/layer |
| SSD 冷存储 (Cold Storage) | NVMe/USB | ~2-5ms | 全部 256 experts/layer |

### A4 动态剪枝 (Dynamic Pruning)

运行时根据 gate entropy 自适应调整每层激活的专家数量（effective top-k），在不影响质量的前提下减少计算量。

### 内存预算门控 (Memory Budget Gate)

Prefill 前检测 GPU headroom，当剩余显存低于安全阈值（2GB）时自动跳过新请求的 prefill，防止 OOM。显存恢复后自动继续。

### 专家命中率监控 (Expert Hit Rate Monitor)

实时监控 GPU 热池命中率，当低于 70% 阈值时发出警告，帮助诊断 pool size 配置问题。

---

## 性能基准

**测试环境**: Apple M4 Pro 48GB | Qwen3.5-35B-A3B (6-bit) | Batch=4

### FlashMLX vs 社区版 mlx-lm

| 指标 | 社区版 (16K) | FlashMLX (16K) | 提升 |
|------|:---:|:---:|:---:|
| **解码吞吐 (TG)** | 115.8 tok/s | **196.9 tok/s** | **+70.0%** |
| **首 Token 延迟 (TTFT)** | 82.0s | **21.1s** | **-74.3%** |
| **GPU 内存峰值** | 28.01 GB | **13.78 GB** | **-50.8%** |
| **模型显存占用** | 18.21 GB | **11.42 GB** | **-37.3%** |

### 全 Context 长度对比

| Context | 模式 | PP (tok/s) | TG (tok/s) | TTFT | 内存峰值 | 质量 |
|---------|------|:---:|:---:|:---:|:---:|:---:|
| 4K | 社区版 | 947 | 150.5 | 17.0s | 24.63G | 4/4 |
| 4K | **FlashMLX** | 950 | **184.3** | **4.9s** | **12.60G** | 4/4 |
| 8K | 社区版 | 883 | 137.7 | 36.3s | 25.64G | 4/4 |
| 8K | **FlashMLX** | 851 | **177.4** | **9.7s** | **12.98G** | 4/4 |
| 16K | 社区版 | 779 | 115.8 | 82.0s | 28.01G | 4/4 |
| 16K | **FlashMLX** | 775 | **196.9** | **21.1s** | **13.78G** | 4/4 |

> Context 越长，FlashMLX 的优势越大。16K 下解码速度提升 70%，因为更小的 prefill chunk 减少了 GPU activation 残留，使解码阶段的内存访问模式更紧凑。

---

## 快速开始

### 安装

```bash
git clone https://github.com/lisihao/FlashMLX.git
cd FlashMLX
pip install -e .
```

### 使用 FlashBatchGenerator

```python
from mlx_lm import load
from mlx_lm.models.expert_offload import patch_model_for_offload, FlashBatchGenerator

# 加载模型并启用专家卸载
model, tokenizer = load("your-moe-model-path")
ctx = patch_model_for_offload(model, "your-moe-model-path")

# 创建批处理生成器（自动启用分块交错调度）
gen = FlashBatchGenerator(
    model, ctx,
    max_tokens=512,
    completion_batch_size=4,
)

# 插入多个请求
prompts = [tokenizer.encode(p) for p in your_prompts]
uids = gen.insert(prompts, max_tokens=[512] * len(prompts))

# 流式获取结果
while True:
    responses = gen.next()
    for r in responses:
        if r.finish_reason is not None:
            print(f"Request {r.uid} complete")
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `interleaved` | `True` | 启用分块交错调度 |
| `interleaved_chunk_size` | `512` | 每次 prefill 的 chunk 大小（tokens） |
| `maintenance_interval` | `4` | 专家池维护频率（每 N 步） |

---

## 技术架构

```
请求队列                      调度器                          GPU
┌─────────┐              ┌──────────────┐             ┌──────────────┐
│ Prompt 1 │──────┐      │  每个 tick:    │             │  Hot Pool    │
│ Prompt 2 │──────┤      │  1. Decode ALL │────────────▶│  (153 exp)   │
│ Prompt 3 │──────┤      │  2. Prefill 1  │             │              │
│ Prompt 4 │──────┘      │     chunk(512) │             │  KV Cache    │
└─────────┘              └──────────────┘             └──────┬───────┘
                                                             │
                              CPU                            │ miss
                         ┌──────────────┐                    │
                         │  Warm Cache   │◀───────────────────┘
                         │  (145 exp)    │
                         │              │──── miss ──▶ SSD Cold Storage
                         └──────────────┘              (256 exp, all)
```

### 调度时序

```
Tick 1: [Decode R1,R2,R3] → [Prefill Prompt4 chunk 0-511]
Tick 2: [Decode R1,R2,R3,R4] → [Prefill Prompt4 chunk 512-1023]
Tick 3: [Decode R1,R2,R3,R4] → [Prefill Prompt4 chunk 1024-1535]
...
```

每个 tick 中，所有已激活的请求先进行一步解码（产出 1 token），然后对下一个待处理的 prompt 进行 1 个 chunk 的 prefill。这确保了：
- 解码延迟始终稳定（不被大 prefill 阻塞）
- 内存峰值可控（只有 1 个 chunk 的 activation 开销）
- 首请求几乎立刻开始产出（无需等待整个 batch prefill 完成）

---

## 致谢

FlashMLX 基于 Apple 的 [mlx-lm](https://github.com/ml-explore/mlx-lm) 构建。感谢 MLX 团队提供的优秀底层框架。

---

# FlashMLX (English)

**High-Performance MoE Inference Engine on Apple Silicon**

A fork of [mlx-lm](https://github.com/ml-explore/mlx-lm), purpose-built for efficient Mixture-of-Experts (MoE) large model inference on memory-constrained Apple Silicon devices.

---

## Key Innovations

### Chunked Interleaved Scheduling

Traditional batch inference processes all prompts at once during prefill, causing memory peaks to scale linearly with batch size. FlashMLX uses **chunked interleaved scheduling**:

```
Each scheduler tick: decode all active requests → prefill 1 chunk (512 tokens)
```

- Prefill memory peaks no longer stack across requests (only 1 chunk at a time)
- First request TTFT drastically reduced (no waiting for all requests to prefill)
- Decode throughput significantly improved (smaller chunks reduce GPU activation residue, improving cache locality)

### Three-Tier Expert Management

Intelligent expert scheduling for MoE models (e.g., Qwen3.5-35B-A3B, 256 experts/layer x 40 layers):

| Tier | Storage | Latency | Capacity |
|------|---------|---------|----------|
| GPU Hot Pool | Unified Memory | ~0ms | ~153 experts/layer |
| CPU Warm Cache | System RAM | ~0.1ms | ~145 experts/layer |
| SSD Cold Storage | NVMe/USB | ~2-5ms | All 256 experts/layer |

### A4 Dynamic Pruning

Adaptively adjusts effective top-k per layer at runtime based on gate entropy, reducing computation without quality loss.

### Memory Budget Gate

Checks GPU headroom before prefill. Automatically skips new request prefill when available memory drops below the safety threshold (2GB), preventing OOM. Resumes automatically when memory recovers.

### Expert Hit Rate Monitor

Real-time monitoring of GPU hot pool hit rate. Warns when hit rate drops below 70% threshold, helping diagnose pool size configuration issues.

---

## Benchmarks

**Test Environment**: Apple M4 Pro 48GB | Qwen3.5-35B-A3B (6-bit) | Batch=4

### FlashMLX vs Community mlx-lm

| Metric | Community (16K) | FlashMLX (16K) | Improvement |
|--------|:---:|:---:|:---:|
| **Decode Throughput (TG)** | 115.8 tok/s | **196.9 tok/s** | **+70.0%** |
| **Time to First Token (TTFT)** | 82.0s | **21.1s** | **-74.3%** |
| **GPU Peak Memory** | 28.01 GB | **13.78 GB** | **-50.8%** |
| **Model Memory Footprint** | 18.21 GB | **11.42 GB** | **-37.3%** |

### Full Context Length Comparison

| Context | Mode | PP (tok/s) | TG (tok/s) | TTFT | Peak Memory | Quality |
|---------|------|:---:|:---:|:---:|:---:|:---:|
| 4K | Community | 947 | 150.5 | 17.0s | 24.63G | 4/4 |
| 4K | **FlashMLX** | 950 | **184.3** | **4.9s** | **12.60G** | 4/4 |
| 8K | Community | 883 | 137.7 | 36.3s | 25.64G | 4/4 |
| 8K | **FlashMLX** | 851 | **177.4** | **9.7s** | **12.98G** | 4/4 |
| 16K | Community | 779 | 115.8 | 82.0s | 28.01G | 4/4 |
| 16K | **FlashMLX** | 775 | **196.9** | **21.1s** | **13.78G** | 4/4 |

> The longer the context, the greater FlashMLX's advantage. At 16K, decode speed improves by 70% because smaller prefill chunks reduce GPU activation residue, resulting in more compact memory access patterns during decode.

---

## Quick Start

### Installation

```bash
git clone https://github.com/lisihao/FlashMLX.git
cd FlashMLX
pip install -e .
```

### Using FlashBatchGenerator

```python
from mlx_lm import load
from mlx_lm.models.expert_offload import patch_model_for_offload, FlashBatchGenerator

# Load model with expert offloading
model, tokenizer = load("your-moe-model-path")
ctx = patch_model_for_offload(model, "your-moe-model-path")

# Create batch generator (chunked interleaved scheduling enabled by default)
gen = FlashBatchGenerator(
    model, ctx,
    max_tokens=512,
    completion_batch_size=4,
)

# Insert multiple requests
prompts = [tokenizer.encode(p) for p in your_prompts]
uids = gen.insert(prompts, max_tokens=[512] * len(prompts))

# Stream results
while True:
    responses = gen.next()
    for r in responses:
        if r.finish_reason is not None:
            print(f"Request {r.uid} complete")
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interleaved` | `True` | Enable chunked interleaved scheduling |
| `interleaved_chunk_size` | `512` | Prefill chunk size per tick (tokens) |
| `maintenance_interval` | `4` | Expert pool maintenance frequency (every N steps) |

---

## Architecture

```
Request Queue                  Scheduler                        GPU
┌─────────┐              ┌──────────────┐             ┌──────────────┐
│ Prompt 1 │──────┐      │  Each tick:    │             │  Hot Pool    │
│ Prompt 2 │──────┤      │  1. Decode ALL │────────────▶│  (153 exp)   │
│ Prompt 3 │──────┤      │  2. Prefill 1  │             │              │
│ Prompt 4 │──────┘      │     chunk(512) │             │  KV Cache    │
└─────────┘              └──────────────┘             └──────┬───────┘
                                                             │
                              CPU                            │ miss
                         ┌──────────────┐                    │
                         │  Warm Cache   │◀───────────────────┘
                         │  (145 exp)    │
                         │              │──── miss ──▶ SSD Cold Storage
                         └──────────────┘              (256 exp, all)
```

### Scheduling Timeline

```
Tick 1: [Decode R1,R2,R3] → [Prefill Prompt4 chunk 0-511]
Tick 2: [Decode R1,R2,R3,R4] → [Prefill Prompt4 chunk 512-1023]
Tick 3: [Decode R1,R2,R3,R4] → [Prefill Prompt4 chunk 1024-1535]
...
```

Each tick: all active requests first perform one decode step (producing 1 token), then the next pending prompt gets 1 chunk of prefill. This ensures:
- Stable decode latency (never blocked by large prefills)
- Controllable memory peaks (only 1 chunk of activation overhead)
- Near-instant first request output (no waiting for full batch prefill)

---

## Acknowledgments

FlashMLX is built on Apple's [mlx-lm](https://github.com/ml-explore/mlx-lm). Thanks to the MLX team for the excellent underlying framework.
