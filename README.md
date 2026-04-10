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
## 系统架构总览

FlashMLX 实现 5 条优化路线，不是互相替代，而是**层层组合**：

| Route | 名称 | 优化目标 | 核心技术 |
|:-----:|------|---------|---------|
| **0** | Density Router | 压缩控制面 | 离散压缩级别 + Model Card 模式切换 |
| **1** | Expert Offloading | 参数内存 | 三级 GPU/CPU/SSD 专家池 |
| **3** | Scored P2 + Flat Buffer | KV cache 内存 | AM 打分 + 可插拔量化 |
| **4** | Chunked Prefill | PP 峰值内存 | 流式淘汰 + 交错调度 |
| **5** | Context Recall (KV-Direct) | 压缩后召回 | h^(0) 存档 + 按需重建 |

```
Route 0 (控制面 — 选择压缩策略)
    ├─ Route 3 (KV 压缩基座)
    │   ├─ Route 4 (PP 阶段分块 prefill + 流式淘汰)
    │   └─ Route 5 (h^(0) 备份，召回时重建)
    └─ Route 1 (Expert Offloading，正交独立)
```

### 代码仓库结构

**FlashMLX SDK** (`/src/flashmlx/`)

| 文件 | 行数 | 职责 |
|------|:---:|------|
| `config.py` | 362 | `CacheConfig`, `FlashMLXConfig`, `DensityLevel`, `snap_to_nearest()` |
| `model_cards.py` | 267 | `ModelCard`, `ModeConfig` — 每模型 JSON 配置，单一数据源 |
| `reconstruction.py` | 525 | `ReconstructionController` — h^(0) → K/V 重建编程接口 |
| `capabilities.py` | ~80 | `detect_capabilities()`, `recommend_config()` |
| `integration/thunderomlx.py` | 159 | `setup_flashmlx()` 入口 + ThunderOMLX 适配器 |

**mlx-lm 核心引擎** (`/mlx-lm-source/mlx_lm/models/`)

| 文件 | 行数 | 职责 |
|------|:---:|------|
| **`cache_factory.py`** | 497 | 智能工厂：按策略自动创建 cache、检测混合架构 |
| **`triple_layer_cache.py`** | 2034 | **Route 3 核心**：Recent/Warm/Cold 三层 + Flat Buffer + Scored 模式 |
| **`kv_direct_cache.py`** | 863 | **Route 5 核心**：`H0Store`, `reconstruct_prefix_kv()`, monkey-patch |
| `quantization_strategies.py` | 1088 | Q4\_0, PolarQuant, TurboQuant 量化后端 |
| `expert_offload.py` | 3084 | **Route 1 核心**：三级专家管理 (GPU→CPU→SSD) |

---

## 端到端数据流

```
User Prompt (text)
    │
    ▼
tokenizer.encode() → token_ids
    │
    ▼
╔══════════════════════════════════════════════════════════════════╗
║  PREFILL (多 token 输入)                                        ║
║                                                                  ║
║  embed_tokens(token_ids) → h^(0)  ← [Route 5: 零成本存入 H0Store]║
║      │                                                           ║
║      ▼                                                           ║
║  Layer 0..N-1 forward (with TripleLayerKVCache)                  ║
║      │                                                           ║
║      ├─ 每层: Q, K, V = proj(h)                                  ║
║      ├─ Cache: K, V → TripleLayerKVCache.update_and_fetch()      ║
║      │   ├─ Recent (L0, bf16, 512 tokens)                       ║
║      │   ├─ 溢出 → Warm (L1, Q4_0)  ← [Route 4: 流式淘汰]       ║
║      │   └─ 溢出 → Cold (L2, AM 打分压缩)  ← [Route 3]          ║
║      │                                                           ║
║      └─ Attention(Q, K_all, V_all) → h_next                     ║
║                                                                  ║
║  [Route 4: chunk=512, 每块之间淘汰冷 token]                      ║
╚══════════════════════════════════════════════════════════════════╝
    │
    ▼
╔══════════════════════════════════════════════════════════════════╗
║  PP→TG 过渡 (首个 TG token)                                     ║
║                                                                  ║
║  Scored Fast Promotion (scored_mode):                            ║
║      AM 在 bf16 上打分 → 只保留热 token                           ║
║      分配 Flat Buffer (Q8_0/Q4_0/bf16)                           ║
║      热 token → flat buffer (量化写入)                            ║
║      释放 L0/L1/L2 暂存缓存                                      ║
║                                                                  ║
║  [可选 Route 5: auto_reconstruct → 注入 h^(0) 重建的 K/V]        ║
╚══════════════════════════════════════════════════════════════════╝
    │
    ▼
╔══════════════════════════════════════════════════════════════════╗
║  TOKEN GENERATION (每步)                                         ║
║                                                                  ║
║  Flat Fast Path:                                                 ║
║      写入新 K,V → flat[offset]       ← O(1) slice assignment    ║
║      读取 flat[0:offset+1]           ← 按需反量化               ║
║      Attention(Q_new, K_flat, V_flat) → logits → 采样           ║
║      offset++                                                    ║
╚══════════════════════════════════════════════════════════════════╝
    │
    ▼ (按需触发，冷路径)
╔══════════════════════════════════════════════════════════════════╗
║  RECONSTRUCTION (由 ReconstructionController 触发)               ║
║                                                                  ║
║  h0_store.get_range(0, N) → h^(0)                               ║
║      │                                                           ║
║      ▼ 分块重放 (64 tok/chunk, ~30ms/chunk)                      ║
║  for chunk in [0:64, 64:128, ...]:                               ║
║      for layer in model.layers:                                  ║
║          h = layer(h_chunk, mask, temp_cache)                    ║
║      mx.eval(h)  ← 每块间让出 GPU                                ║
║      │                                                           ║
║      ▼                                                           ║
║  精确 K/V → inject_reconstruction() (dedup 去重)                  ║
╚══════════════════════════════════════════════════════════════════╝
```

### 函数调用链

```
generate_step()                          # mlx_lm/generate.py
  └─ model(tokens, cache=cache_list)     # Model.__call__
      └─ embed_tokens(tokens) → h        # (Route 5: h0_store.append)
      └─ for layer in self.layers:
          └─ layer(h, mask, cache=cache[i])
              └─ self_attn(h, mask, cache)
                  └─ cache.update_and_fetch(keys, values)
                      ├─ [prefill] _update_slow_path()
                      │   ├─ append to Recent (L0)
                      │   ├─ _manage_aging() → Warm (L1, Q4)
                      │   └─ _manage_aging() → Cold (L2, AM)
                      │
                      └─ [decode] _update_flat_path()
                          ├─ _write_flat(k, v)   ← 量化写入
                          └─ _fetch_flat()        ← 反量化读取
```

---

## Route 3: Scored P2 + Flat Buffer — KV Cache 压缩引擎

> **核心命题：不是"压得更狠"，而是"在 PP 和 TG 两个阶段用完全不同的数据结构。"**

### TripleLayerKVCache 架构

```
┌─────────────────────────────────────────────────────────────┐
│ TripleLayerKVCache (每个 Attention 层一个实例)                 │
│                                                              │
│  PP 阶段: 三层缓存                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Recent (L0)  │→ │  Warm (L1)   │→ │    Cold (L2)      │  │
│  │ bf16, 精确   │  │ Q4_0 压缩    │  │ AM 打分压缩        │  │
│  │ 512 tokens   │  │ 2048 tokens  │  │ 可变长度           │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
│                                                              │
│  TG 阶段: Flat Buffer (替换上面三层)                          │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Flat Buffer                                          │    │
│  │ Q8_0 / Q4_0 / TurboQuant / bf16                     │    │
│  │ 预分配, O(1) 写入, 读时反量化                          │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**PP→TG 过渡** 是整个系统最关键的一步：在 bf16 精确数据上做一次性 AM 打分，把热 token 压入预分配的量化 flat buffer，释放三层暂存。从此 TG 走 O(1) slice assignment，和标准 KVCache 一样快。

### Flat Buffer 量化选项

| `flat_quant` | 存储方式 | 反量化成本 | 压缩率 | 推荐 |
|-------------|---------|:---:|:---:|:---:|
| `None` (bf16) | 原始浮点 | 0 | 0% | 最快速度 |
| **`q8_0`** | **int8 + per-token scales** | **低** | **~50%** | **默认推荐** |
| `q4_0` | uint8 nibble-packed + per-group scales | 中等 | ~75% | 极限内存 |
| `turboquant` | PolarQuant packed | 中等 | ~75% | 实验性 |

### AM 打分流程 (Attention Matching)

在 PP→TG 过渡时 (scored\_mode=True)：

1. 用最后一个 query 计算注意力权重: `W = softmax(Q @ K^T / sqrt(d))`
2. 按聚合注意力权重给每个 token 打分
3. 保留 top-K 热 token（基于 compression\_ratio）
4. 应用 pinned 保护（前 N 个 token 永不淘汰）
5. 热 token 量化后写入 flat buffer

### 量化后端类层次

```
QuantizationStrategy (ABC)
├── Q4_0Quantizer       # 对称 4-bit, group_size=32, nibble-packed
├── PolarQuantizer      # 2-4 bit, 随机旋转 + Lloyd-Max (数据无关)
└── TurboQuantizer      # PolarQuant + 阻尼 QJL 残差校正
```

---

## Route 5: Context Recall — 基于 KV-Direct 的可恢复压缩

> **核心命题：KV cache 压缩之后，丢掉的 token 真的就永远找不回来了吗？**

### 论文基础

Route 5 基于论文 [KV-Direct (arXiv: 2603.19664)](https://arxiv.org/abs/2603.19664) 的核心观察：

> Transformer 中，所有层的 K/V 都是同一个输入 h^(0) = embed\_tokens(x) 经过逐层 forward 得到的。
> 只要保存 h^(0)，就可以在任何时刻重新跑一遍 forward pass，精确重建出每一层的 K/V。

这意味着：

| 存储方式 | 每 token 代价 (Qwen3-8B) | 精确度 |
|---------|:---:|:---:|
| 完整 K/V (36层 × 8头 × 128维 × 2) | 147,456 字节 | 100% |
| **h^(0) 存档 (4096维 × bf16)** | **8,192 字节** | **100% 可重建** |
| 压缩比 | **18×** | — |

**一句话：花 1/18 的内存，换回"随时可精确恢复"的能力。**

---

### 为什么需要 Route 5

FlashMLX v1.x 的 scored\_pq 已经很强：32K 上下文 TG +34%、内存 -89%。但它有一个结构性局限：

**丢了就是丢了。**

AM scoring 淘汰的 token，永远不会回来。对于单轮问答，这不是问题。但在这些场景下会暴露：

- **多轮长对话**：第一轮的系统提示 token 被淘汰，第五轮的回答质量塌了
- **多 Agent 共享上下文**：Agent A 用过的 token，Agent B 需要但已经丢了
- **RAG + 长前缀**：前 8K 的检索文档被压缩淘汰，后面回答找不到依据

Route 5 的回答不是"压得少一点"，而是：

> **正常压缩，正常运行。但每个 token 的 h^(0) 都存着档。需要的时候，精确重建。**

---

### 设计决策：我们做了什么，没做什么

#### 决策 1：h^(0) 存档，不存完整 K/V

论文的核心 insight 是 h^(0) 足够重建一切。我们没有选择存完整 K/V 的 checkpoint（太贵），也没有选择存中间层的 hidden state（不通用）。

h^(0) = embed\_tokens(x)，一次计算、所有层共享、18× 压缩。这是唯一正确的存储粒度。

#### 决策 2：只做连续前缀重建，不做稀疏补洞

论文的重建依赖因果注意力机制：position 0 → position N 的 token 必须顺序经过每一层。这意味着：

- **[0:N] 前缀重建** — 完全正确。RoPE 位置、因果 mask、KV cache offset 全对。
- **[50:100] 稀疏补洞** — 不正确。跳过 [0:50] 会导致 RoPE 错位和注意力 mask 断裂。

我们的 API 直接在接口层强制这个约束：

```python
def reconstruct_prefix_kv(inner_model, h0_store, start, end, chunk_size=0):
    if start != 0:
        raise NotImplementedError("Only prefix reconstruction (start=0) supported.")
```

**不给未来的自己留坑。**

#### 决策 3：Monkey-patch，不改模型代码

h^(0) capture 需要拦截 `embed_tokens` 的输出。两种做法：

| 方案 | 优点 | 缺点 |
|------|------|------|
| 修改模型 forward 代码 | 干净 | 每个模型架构都要改，维护地狱 |
| **`__class__` 动态替换** | **零侵入** | 需要防御机制 |

我们选了 monkey-patch。代价是需要一套防御体系：

- **双重 patch 检测**：`_KV_DIRECT_PATCHED` 哨兵，第二次 patch 直接 raise
- **batch_size > 1 拦截**：h^(0) 是序列级存储，多 batch 会混淆 token 顺序
- **`unpatch_model()`**：完整恢复原始 `__class__`，benchmark 切策略时必须

#### 决策 4：Q8 h^(0) 是工程探索，不是论文承诺

论文用 bf16 h^(0)，重建是 exact 的。我们提供 Q8/Q4 量化选项：

| h^(0) 模式 | 精度 | 内存 (32K, d=4096) | 定位 |
|------------|:---:|:---:|------|
| bf16 | exact | 256 MB | **论文对齐，默认推荐** |
| q8 | near-exact | 128 MB | 工程折中，长上下文友好 |
| q4 | lossy | 64 MB | 实验性，不推荐 |

**量化后的 h^(0) 重建出来的 K/V 不再是 exact 的。** 我们在代码和文档中严格区分这一点。

#### 决策 5：与 scored\_pq 融合，不是替代

Route 5 不是一个独立策略。它是 scored\_pq 的**上层能力**：

```
scored_pq          = AM scoring + flat buffer Q8 → 快速推理，token 淘汰不可逆
scored_kv_direct   = scored_pq + h^(0) capture  → 同样快，但淘汰 token 可精确重建
```

`apply_h0_capture_only()` 在 scored\_pq 之上安装 h^(0) 采集，不影响任何热路径逻辑。h^(0) 只是安静地存着，直到有人需要。

---

### 超越论文：五个关键改进

#### 改进 1：N Crossover 验证 — 论文 4B 以下，我们验到 8B

论文在 135M–4B 模型上验证，预测 N ≈ 50 时重建比标准 prefill 更快。我们实测：

| 模型 | 参数量 | Crossover N | 含义 |
|------|:------:|:----------:|------|
| 论文范围 | 135M–4B | ~50 | 50 token 以上重建比 prefill 快 |
| **Qwen3-8B** | **8B** | **8** | **8 个 token 就值得重建** |
| Qwen3-1.7B | 1.7B | 100 | 小模型重建开销相对大 |

**8B 模型 36 层，重建时每层共享同一份 h^(0)，层数越多重建越划算。** 这个发现直接把 Route 5 从"大 N 才划算"变成"几乎任何 N 都划算"。

#### 改进 2：分块重建 — 让出 GPU 给其他 Agent

论文的重建是一次性跑完所有 token。在多 Agent 场景下，一次重建 500 个 token 可能锁住 GPU 500ms，其他 Agent 的 decode 全部等待。

我们引入 **chunked reconstruction**：

```python
RECON_CHUNK_SIZE = 64  # 每块 64 token，约 25-30ms

for chunk_start in range(0, end, chunk_size):
    h_chunk = h0_store.get_range(chunk_start, chunk_end)
    for layer in inner_model.layers:
        h = layer(h_chunk, mask, temp_cache)
    mx.eval(h)  # 强制 evaluate，让出 GPU
```

**效果：GPU 最大阻塞时间从 ~500ms 降到 ~30ms。** 多 Agent 调度器可以在 chunk 间插入其他 Agent 的 decode step。

#### 改进 3：Pinned Prefix — 系统提示永不被淘汰

论文没有讨论系统提示保护。但在实际使用中，前 N 个 token（系统提示）被 AM scoring 淘汰是灾难性的。

我们在 AM scoring 的三条路径中都加入了 pinned 保护：

```
_scored_compress_prefix:  [pinned | scorable | recent]  — pinned 无条件保留
_scored_prefill_evict:    [pinned | scorable | recent]  — pinned 不参与打分
_am_compress_prefix:      [pinned | scorable | recent]  — pinned 跳过压缩
```

从 `CacheConfig.pinned_tokens` 一路穿透到 `TripleLayerKVCache.__init__`，全链路支持。

#### 改进 4：Dedup — 重建注入不产生重复 token

当 recall 重建 [0:N] 并注入 cache 时，flat buffer 中已有的前缀时代 token（pinned + hot scored）会和重建结果重叠。重复 token 会稀释注意力权重。

我们追踪 `_flat_prefix_token_count`，注入时跳过重叠部分：

```
注入前: flat buffer = [pinned + hot scored | recent | TG appended]
注入后: attention sees = [reconstructed 0:N] + [non-prefix part of flat] + [recent + TG]
```

**不重不漏，精确拼接。**

#### 改进 5：h^(0) 采集零成本 — 实测验证

这是最重要的发现。很多人会问：每个 token 都存 h^(0)，不会拖慢推理吗？

| 指标 | scored\_pq | scored\_kv\_direct | 差异 |
|------|:---:|:---:|:---:|
| PP tok/s (32K) | 409.5 | 409.5 | **+0.0%** |
| TG tok/s (32K) | 21.6 | 21.6 | **-0.0%** |
| 输出 | ✅ | ✅ | **完全一致** |
| 额外内存 | — | +32M (bf16) | h^(0) 存储 |

**原因**：`embed_tokens(x)` 是 forward pass 的第一步，scored\_pq 本来就要算。h^(0) capture 只是把已经算好的结果 `.append()` 到一个列表里。**没有额外计算，只有一次内存拷贝。**

---

### 实测性能基线 (Official v2.0)

**Qwen3-8B-MLX (4-bit) / 32K context / Apple M4 Max 64GB**

| 指标 | Standard | scored\_pq + Q8 | 变化 |
|------|:---:|:---:|:---:|
| PP 吞吐 | 269.5 tok/s | **409.5 tok/s** | **+51.9%** |
| TG 吞吐 | 16.1 tok/s | **21.6 tok/s** | **+34.2%** |
| TTFT | 121.6s | **80.0s** | **-34.2%** |
| PP Peak 内存 | 4,840 MB | **526 MB** | **-89.1%** |
| TG 内存 | 4,647 MB | **529 MB** | **-88.6%** |
| 质量 | PASS | **PASS** | **无损** |

> 所有参数固化在 `model_cards/qwen3-8b-mlx-4bit.json`，一行代码加载：
> ```python
> card = load_card("/path/to/model")
> cache = make_prompt_cache(model, **card.to_cache_kwargs())
> ```

---

### Route 5 数据流

```
热路径 (零成本采集):                    冷路径 (按需触发):

embed_tokens(x) → h^(0)              ReconstructionController.reconstruct()
     │                                    │
     ├─ h0_store.append(h^(0))            ├─ h0_store.get_range(0, N)
     │  (只是内存拷贝，无额外计算)           │
     ▼                                    ▼
Forward 36 layers                     分块重放 (64 tok/chunk, ~30ms)
     │                                for chunk in [0:64, 64:128, ...]:
     ├─ scored_pq: 压缩、淘汰              for layer in model.layers:
     └─ TG decode: flat buffer 热路径         h = layer(h_chunk, mask, temp_cache)
                                          mx.eval(h)  # 让出 GPU
                                              │
                                              ▼
                                     inject_reconstruction() + dedup
                                     → 精确 K/V 注入 cache
```

---

### 墓碑：Route 5 中被杀掉的路线

| 路线 | 为什么被杀 |
|------|----------|
| 稀疏补洞重建 [50:100] | RoPE 位置错位 + 因果 mask 断裂，不是"暂时不支持"，是"物理上不成立" |
| Q4 h^(0) 作为默认 | 量化误差逐层放大，重建出来的 K/V 偏差不可控 |
| 每步 TG 都自动触发重建 | 重建是冷路径操作，塞进热路径会杀死 TG 吞吐 |
| h^(0) 存完整 hidden state 而非 embed\_tokens | 中间层 hidden state 不通用（每层不同），存储量 36× 爆炸 |
| Batch > 1 支持 | h^(0) 是序列级全局存储，多 batch token 顺序会混淆，暂未解决 |

---

### 设计哲学

**Route 5 不是在论文上加"功能"。它是把论文的数学保证，翻译成工程系统中的真实能力。**

论文给了 insight：h^(0) 可以重建一切。
我们加了约束：只重建前缀、分块让出 GPU、pinned 保护、dedup 去重、monkey-patch 防御。

结果是：

> **scored\_pq 的速度 + h^(0) 的记忆 = 压缩了也找得回来。**

这不是"压缩率更高"的故事。这是"有损变有底"的故事。

---

## Route 0: Product Modes — 一行代码切换压缩策略

> **核心命题：压缩不应该是一个开关，而是一组场景预设。**

Route 0 把 scored\_pq（Route 3）和 KV-Direct（Route 5）封装成三个**产品模式**，通过 Model Card 一行代码切换：

```python
from flashmlx import load_card_or_detect, make_prompt_cache

card = load_card_or_detect(model, model_path)

# 一行切换模式
cache = make_prompt_cache(model, **card.to_cache_kwargs(mode="balanced"))      # 日常
cache = make_prompt_cache(model, **card.to_cache_kwargs(mode="ultra_long"))    # 超长上下文
cache = make_prompt_cache(model, **card.to_cache_kwargs(mode="recall_first")) # 细节召回
```

### 三种模式

| 模式 | 压缩力度 | 策略 | h^(0) 备份 | 最佳场景 |
|------|:---:|:---|:---:|:---|
| `balanced` | 2-3x | scored\_pq | 无 | **日常使用**，最快 TG，最低延迟 |
| `ultra_long` | 5-10x | scored\_pq | 无 | **32K+ 超长上下文**，最低内存 |
| `recall_first` | 10x+ | scored\_kv\_direct | 有 + 按需重建 | **细节召回**，压得最狠但丢了能找回来 |

### 8K Context 吞吐矩阵 (Qwen3-8B / M4 Max 64GB)

| Mode | PP tok/s | TG tok/s | TG Memory | vs Standard |
|------|:---:|:---:|:---:|:---:|
| standard (baseline) | 395 | 23.9 | 1,193 MB | — |
| **balanced** (scored\_pq) | **417** (+6%) | **24.6** (+3%) | **242 MB** | **-80% mem** |
| **ultra\_long** (10x) | **409** (+4%) | **26.0** (+9%) | **152 MB** | **-87% mem** |
| **recall\_first** (10x+h0) | 338 (-14%) | 26.1 (+9%) | 217 MB | -82% mem |

### Recall Quality Scorecard (8K, 3 needles × 2 keywords each)

| Mode | early | mid | late | Total |
|------|:---:|:---:|:---:|:---:|
| standard | PASS | PASS | PASS | **6/6** |
| balanced (scored\_pq 3x) | PASS | PASS | PASS | **6/6** |
| ultra\_long (10x) | PASS | 0/2 | PASS | 4/6 |
| recall\_first (10x, no recon) | PASS | 0/2 | PASS | 4/6 |
| **recall\_first + RECON** | **PASS** | **PASS** | **PASS** | **6/6** |
| **recall\_first + TARGETED** | **PASS** | **PASS** | **PASS** | **6/6** |

> **关键发现**：10x 压缩会丢掉中间位置的细节（4/6）。触发 h^(0) 重建后，完整恢复到 6/6。这验证了"前端敢压、后端能救"的设计论点。

---

## ReconstructionController — 重建编程接口

> **核心命题：重建不应该是个哑开关。它应该由上层调度器按需触发。**

`auto_reconstruct=True` 是一个简单开关：prefill 结束后无条件全量重建。但真实场景中，重建应该是**按需的、可控的、有预算的**：

- 检测到乱码/质量下降 → 触发重建
- 上下文压缩后需要回补数据 → 按需部分重建
- 内存充裕时 → 后台批量重建

`ReconstructionController` 是面向 ThunderOMLX 等上层调度器的编程接口：

```python
from flashmlx import ReconstructionController

# 1. 从 cache 自动发现（零配置）
cache_list = make_prompt_cache(model, **cache_kwargs)
recon = ReconstructionController.from_cache(cache_list, model)

# 2. 查询能力和成本
if recon.available:
    stats = recon.stats       # h0_tokens, h0_bytes, probe_available, ...
    cost = recon.estimate_cost(n_tokens=4096)  # time_ms_est, memory_mb_est

# 3. 按需重建（三种策略）
result = recon.reconstruct()                                    # 全量重建
result = recon.reconstruct(strategy="partial", max_tokens=4096) # 前 4096 tokens
result = recon.reconstruct(strategy="targeted", coverage=0.95)  # 探针导向

# 4. 检查结果
if result.success:
    print(f"重建 {result.tokens_reconstructed} tokens, "
          f"{result.layers_injected} layers, {result.time_ms:.0f}ms")

# 5. 释放内存
recon.clear()
```

### 设计要点

| 设计 | 为什么 |
|------|--------|
| **Null Object 模式** | `from_cache()` 找不到 h^(0) 时返回 `NullReconstructionController`，所有操作安全空操作，调用方永不判空 |
| **非阻塞锁** | 重建进行中再次调用立即返回 `success=False`，不阻塞调度器 |
| **Frozen 结果** | `ReconStats`、`ReconResult` 都是冻结数据类，线程安全，可跨线程传递 |
| **不改 make\_prompt\_cache** | 通过 `from_cache()` 工厂方法发现 controller，零 API 破坏 |

> 详细架构文档见 `docs/ARCHITECTURE.md`

---

## Cache 工厂：策略选择

`make_optimized_cache()` 根据 strategy 参数自动创建对应的 cache 组合：

| 策略 | Warm 层 | Cold 层 | Flat Buffer | 内存节省 | 定位 |
|------|--------|--------|:---:|:---:|------|
| `standard` | 无 | 无 | bf16 | 0% | 基线 |
| `triple` | Q4\_0 | Q4\_0 | 无 | ~48% | 质量优先 |
| `triple_am` | Q4\_0 | AM 压缩 | 无 | ~50% | 平衡 |
| `triple_pq` | PolarQuant | Q4\_0 | 无 | ~72% | 无校准压缩 |
| **`scored_pq`** | **(跳过)** | **AM 打分** | **Q8\_0** | **~81%** | **生产推荐** |
| `scored_kv_direct` | (跳过) | AM 打分 | Q8\_0 | ~81% + h^(0) | Route 5: 极限压缩 + 重建 |

### Model Cards：每模型配置的单一数据源

```json
{
  "model_id": "qwen3-8b-mlx-4bit",
  "architecture": { "num_layers": 36, "head_dim": 128, "num_kv_heads": 8 },
  "optimal": { "strategy": "scored_pq", "flat_quant": "q8_0" },
  "modes": {
    "balanced":     { "density_scale": 0.0, "strategy": "scored_pq" },
    "ultra_long":   { "density_scale": 1.5 },
    "recall_first": { "density_scale": 2.5, "strategy": "scored_kv_direct", "probe_layers": 3 }
  }
}
```

一行加载：`cache = make_prompt_cache(model, **card.to_cache_kwargs(mode="balanced"))`

---

## 集成架构

```
┌──────────────────────────────────────────────────────┐
│ ThunderOMLX (调度器 / 编排器)                          │
│                                                       │
│  batched.py:   _apply_flashmlx() → FlashMLXConfig    │
│  scheduler.py: _make_flashmlx_cache() → cache_list   │
│  scheduler.py: _recover_from_cache_error()            │
│                                                       │
│  NEW: ReconstructionController.from_cache(cache, model)│
│       → recon.reconstruct(strategy="targeted")        │
└───────────────────────┬──────────────────────────────┘
                        │ imports
                        ▼
┌──────────────────────────────────────────────────────┐
│ FlashMLX SDK (/src/flashmlx/)                         │
│                                                       │
│  config.py        → CacheConfig, FlashMLXConfig      │
│  model_cards.py   → load_card(), to_cache_kwargs()   │
│  reconstruction.py → ReconstructionController        │
│  capabilities.py  → recommend_config()                │
└───────────────────────┬──────────────────────────────┘
                        │ imports
                        ▼
┌──────────────────────────────────────────────────────┐
│ mlx-lm-source (/mlx_lm/models/)                      │
│                                                       │
│  cache_factory.py      → make_optimized_cache()      │
│  triple_layer_cache.py → TripleLayerKVCache          │
│  kv_direct_cache.py    → H0Store, reconstruct_*     │
│  quantization_strategies.py → Q4/Q8/Polar/Turbo     │
│  expert_offload.py     → ExpertOffloadManager        │
└──────────────────────────────────────────────────────┘
```

---

## 关键架构决策

| 决策 | 选择 | 为什么 | 被否决的方案 |
|------|------|--------|-------------|
| TG 过渡时 flat buffer | 一次性晋升 | 避免 PP 双缓冲；TG O(1) 写入 | Pipeline L0→L1→L2 (PP 内存 2x) |
| Q8\_0 作为默认 flat 量化 | 速度/内存平衡 | Q4 nibble-unpack 开销 > 省下的带宽 | Q4\_0 (TG -45%) |
| Scored 模式在 bf16 上打分 | 打分质量 | 对量化后数据打分 = 噪声打分 | AM on Q4 (精度损失) |
| Monkey-patch 采集 h^(0) | 零模型改动 | 适配任意模型架构 | 修改模型代码 (维护地狱) |
| 只做前缀重建 | 因果正确性 | RoPE + 因果 mask 要求 start=0 | 稀疏 [50:100] (位置错乱) |
| 非阻塞重建锁 | 调度器自由度 | 20s 重建不阻塞其他 Agent | 阻塞锁 (stall 调度器) |
| NullController 空对象 | API 简洁 | ThunderOMLX 永不判空 | Optional 返回 (到处判空) |

---

## 快速开始

### KV Cache 压缩（Route 0/3/5）

```python
import mlx.core as mx
from mlx_lm import load
from flashmlx import load_card_or_detect, make_prompt_cache
from mlx_lm.generate import generate_step

model, tokenizer = load("your-model-path")
card = load_card_or_detect(model, "your-model-path")

# 一行加载优化配置
cache = make_prompt_cache(model, **card.to_cache_kwargs(mode="balanced"))

# 正常 generate（cache 内部自动压缩）
prompt = mx.array(tokenizer.encode("Your prompt here"))
for token, _ in generate_step(prompt, model, prompt_cache=cache):
    print(tokenizer.decode([token]), end="", flush=True)

# 需要召回细节时，触发重建
from flashmlx import ReconstructionController
recon = ReconstructionController.from_cache(cache, model)
if recon.available:
    result = recon.reconstruct(strategy="targeted", coverage=0.95)
```

### Expert Offloading（Route 1，MoE 模型）

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

## Vision-Language Models (VLM) 支持

FlashMLX 现在支持 **Vision-Language Models**（视觉-语言模型），将 FlashMLX 的 cache 优化能力扩展到多模态场景。

### 快速开始

```python
from flashmlx.vlm import load_vlm_components
from flashmlx.generation import VLMGenerator, create_vlm_cache

# 加载 VLM 组件
model, tokenizer, processor, config = load_vlm_components(
    "mlx-community/Qwen2-VL-2B-Instruct-bf16"
)

# 创建优化 cache
cache = create_vlm_cache(model, kv_cache="standard")

# 创建生成器
generator = VLMGenerator(model, tokenizer, config.image_token_id)

# 文本生成
response = generator.generate("What is MLX?", cache=cache)

# Vision+Text 生成
response = generator.generate(
    "What's in this image?",
    pixel_values=pixel_values,
    grid_thw=grid_thw,
    cache=cache
)
```

### VLM 性能结果

**Qwen2-VL-2B / M4 Max 64GB / bf16**

| 场景 | Standard Cache | Compressed Cache | 提升 |
|------|----------------|------------------|------|
| 文本生成 (短上下文) | 52.2 tok/s | 55.0 tok/s | **+5%** |
| Vision+Text (256 tokens) | 11.2 tok/s | 16.1 tok/s | **+43.6%** |
| 质量 | Perfect ✅ | Short: OK, Long: Degraded ⚠️ | - |

**Cache 策略建议**：
- **生产环境**：使用 `standard` cache（完美质量）
- **实验性**：`triple_pq` cache（需 calibration）

### 支持的模型

- ✅ Qwen2-VL-2B-Instruct
- ✅ Qwen2-VL-7B-Instruct
- ⏳ LLaVA (planned)
- ⏳ InternVL (planned)

### 示例和文档

| 资源 | 描述 |
|------|------|
| [`docs/VLM_GUIDE.md`](docs/VLM_GUIDE.md) | 完整 VLM 使用指南 |
| `examples/demo_vlm_simple.py` | 基础使用示例 |
| `examples/demo_vlm_advanced.py` | 高级功能（多轮对话、批处理） |
| `examples/bench_vlm_vision_cache.py` | Vision+Text 性能测试 |

---

## 文档

| 文档 | 内容 |
|------|------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 完整系统架构、数据流、文件结构、所有 Route 详解 |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | API 参考 |
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | 使用指南 |
| [`docs/VLM_GUIDE.md`](docs/VLM_GUIDE.md) | **Vision-Language Models 使用指南** |
| `model_cards/*.json` | 每模型的优化配置（单一数据源） |
| `benchmarks/` | 性能基准测试套件 |

---

## 最后一句话

**不是所有推理系统都该长得像 vLLM。**  
**也不是所有本地优化都只配当“小技巧”。**

FlashMLX 代表的是另一条路：

# **从 Apple Silicon 出发，重写本地 LLM 推理的执行秩序。**

如果你也相信这件事值得做，欢迎 star、benchmark、提 issue，或者直接一起把它打磨成真正的下一代本地推理 runtime。
