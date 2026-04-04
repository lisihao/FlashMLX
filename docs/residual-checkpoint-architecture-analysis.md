# Residual Checkpoint 架构适配性分析

> 基于论文 arxiv 2603.19664 "Residual Stream is All You Need"
> 日期: 2026-04-03

## 1. 核心概念

### 1.1 Residual Checkpoint = h^(0)

Residual checkpoint 指 embed_tokens 的输出 h^(0)，不是残差流 h^(l) 或残差连接。

```
输入 tokens → embed_tokens → h^(0) ← 这就是 residual checkpoint
                               ↓
                            layer_0 → layer_1 → ... → layer_L-1 → norm → logits
```

**Residual Markov Property**: H(K,V | h^(l)) = 0。给定任意层的残差，可以精确重建该层之后所有层的 KV。h^(0) 是最紧凑的选择——一个 checkpoint 替代所有 L 层的 KV。

### 1.2 数据量对比

| 概念 | 定义 | Per-token 大小 (Qwen3-8B) |
|------|------|--------------------------|
| h^(0) (residual checkpoint) | embed_tokens 输出 | d_model × 2B = **8,192 B** |
| h^(l) (residual stream) | 第 l 层输出 | d_model × 2B = **8,192 B** |
| 单层 KV | K+V for 1 layer | 2 × n_kv × head_dim × 2B = **4,096 B** |
| 全量 KV | 所有 L 层的 K+V | L × 4,096 = **147,456 B** |

关键观察: h^(0) 和 h^(l) 大小相同 (d_model × 2B)，但 h^(0) 能替代**所有** L 层 KV，h^(l) 只能替代层 l 以后的 KV。

---

## 2. 通用压缩公式

### 2.1 h^(0) Only 压缩比

```
compression = full_KV / h0 = (L × 2 × n_kv × head_dim × 2B) / (d_model × 2B)
```

由于 d_model = n_q × head_dim:

```
compression = (L × 4 × n_kv × head_dim) / (n_q × head_dim × 2)
            = 2L × (n_kv / n_q)
```

**对于 MHA (n_kv = n_q)**: compression = **2L** (最优，只与深度有关)
**对于 GQA r:1 (n_kv = n_q/r)**: compression = **2L/r**
**对于 MQA (n_kv = 1)**: compression = **2L/n_q**

### 2.2 稀疏 Checkpoint 压缩比

每 k 层存一个 checkpoint (h^(0), h^(k), h^(2k), ...):

```
checkpoints_per_token = ceil(L/k) × d_model × 2B
full_KV_per_token = L × 2 × n_kv × head_dim × 2B

compression = full_KV / checkpoints
            = (L × 4 × n_kv × head_dim) / (ceil(L/k) × n_q × head_dim × 2)
            ≈ 2k × (n_kv / n_q)
```

### 2.3 最小 Checkpoint 间隔

稀疏 checkpoint 要有正收益 (compression > 1):

```
2k × (n_kv / n_q) > 1
k > n_q / (2 × n_kv)
```

| 架构 | n_q/n_kv 比 | 最小间隔 k | 含义 |
|------|------------|-----------|------|
| MHA | 1:1 | k > 0.5 → **任意** | 每层都可以存 checkpoint |
| GQA 2:1 | 2:1 | k > 1 → **k ≥ 2** | 每 2 层起 |
| GQA 4:1 | 4:1 | k > 2 → **k ≥ 3** | 每 3 层起 |
| GQA 8:1 | 8:1 | k > 4 → **k ≥ 5** | 每 5 层起 |
| MQA (n_kv=1) | 32:1 | k > 16 → **k ≥ 17** | 几乎只能 h^(0) |

---

## 3. 架构大全：压缩比矩阵

### 3.1 标准 Attention 架构

| 模型 | 架构 | L | n_q | n_kv | d_model | head_dim | h^(0) 压缩比 | 稀疏@4L | 稀疏@8L | 最小 k |
|------|------|---|-----|------|---------|----------|-------------|---------|---------|--------|
| GPT-2 (1.5B) | MHA | 48 | 25 | 25 | 1600 | 64 | **96×** | 8× | 16× | 1 |
| Llama-2-7B | GQA 4:1 | 32 | 32 | 8 | 4096 | 128 | **16×** | 2× | 4× | 3 |
| Llama-3.1-8B | GQA 4:1 | 32 | 32 | 8 | 4096 | 128 | **16×** | 2× | 4× | 3 |
| Llama-3.1-70B | GQA 8:1 | 80 | 64 | 8 | 8192 | 128 | **20×** | 1× | 2× | 5 |
| Llama-3.1-405B | GQA 8:1 | 126 | 128 | 8 | 16384 | 128 | **15.75×** | 0.5× | 1× | 9 |
| Qwen3-0.6B | GQA 2:1 | 28 | 16 | 8 | 1024 | 64 | **28×** | 4× | 8× | 2 |
| Qwen3-1.7B | GQA 2:1 | 28 | 16 | 8 | 2048 | 128 | **28×** | 4× | 8× | 2 |
| **Qwen3-8B** | GQA 4:1 | 36 | 32 | 8 | 4096 | 128 | **18×** | 2× | 4× | 3 |
| Qwen3-32B | GQA 4:1 | 64 | 64 | 8 | 5120 | 128 | **16×** | 1× | 2× | 5 |
| Mistral-7B | GQA 4:1 | 32 | 32 | 8 | 4096 | 128 | **16×** | 2× | 4× | 3 |
| Mixtral-8x7B | GQA 4:1 | 32 | 32 | 8 | 4096 | 128 | **16×** | 2× | 4× | 3 |
| **Gemma-4-31B** | GQA 2:1 | 60 | 32 | 16 | 5376 | 256 | **60×** | 8× | 16× | 2 |
| Gemma-4-E4B | GQA 4:1 | 42 | 8 | 2 | 2560 | 256 | **21×** | 2× | 4× | 3 |
| Phi-3-mini | GQA 4:1 | 32 | 32 | 8 | 3072 | 96 | **16×** | 2× | 4× | 3 |

### 3.2 特殊架构

| 模型 | 架构 | L | KV per tok per layer | h^(0) per tok | h^(0) 压缩比 | 备注 |
|------|------|---|---------------------|--------------|-------------|------|
| DeepSeek-V3 | MLA | 61 | 1,024 B (latent=512) | 14,336 B | **4.4×** | MLA 天敌 |
| DeepSeek-V2 | MLA | 60 | 1,024 B | 10,240 B | **6.0×** | 稍好 |
| Falcon-180B | MQA | 80 | 512 B (1 head) | 28,672 B | **1.4×** | 勉强正收益 |
| Falcon-7B | MQA | 32 | 256 B (1 head) | 9,216 B | **0.9×** | **负收益!** |

### 3.3 Hybrid 架构 (Sliding Window + Global)

Gemma-4, Mistral-Nemo 等使用混合注意力:
- Global attention layers: 需要完整 KV → h^(0) 替代价值高
- Sliding window layers: KV 本身有上限 (window_size) → 替代价值低

| 模型 | Global 层数 | Window 层数 | Window 大小 | 有效压缩比 |
|------|------------|------------|------------|-----------|
| Gemma-4-31B | 30 (50%) | 30 (50%) | 1024 | 视 context 而定 |
| Mistral-Nemo | 16 (50%) | 16 (50%) | 4096 | 视 context 而定 |

Sliding window 层的 KV 被窗口截断，不随 context 线性增长。
h^(0) 对这些层的替代价值在 context > window_size 时为零 (KV 已是紧凑的)。

**有效压缩**:
```
effective_compression = (global_layers × full_kv_per_layer + window_layers × min(ctx, window) × kv_per_tok_per_layer)
                       / (d_model × 2)
```

---

## 4. 稀疏 Checkpoint 的收益/代价权衡

### 4.1 Memory vs Reconstruction Speed

| 方案 | Archive 大小 | 重建速度 (最坏) | 适用 |
|------|-------------|-----------------|------|
| h^(0) only | 1× checkpoint | L 层 forward | 内存优先 |
| h^(0) + h^(L/2) | 2× checkpoint | L/2 层 forward | 平衡 |
| h^(0) + every 4L | L/4 checkpoints | 4 层 forward | 速度优先 |
| h^(0) + every layer | L checkpoints | 1 层 forward | 极速 (但内存爆) |

### 4.2 Qwen3-8B 具体计算 (4K context, bf16)

| 方案 | Checkpoint 存储 | KV 等效 | 压缩比 | 重建时间 (估) |
|------|----------------|---------|--------|-------------|
| 全量 KV | - | 576 MB | 1× | 0 (不需要) |
| h^(0) only | 32 MB | 替代 576 MB | **18×** | ~5.2 s (36L) |
| h^(0) + h^(18) | 64 MB | 替代 576 MB | **9×** | ~2.6 s (18L) |
| h^(0) + h^(9) + h^(18) + h^(27) | 128 MB | 替代 576 MB | **4.5×** | ~1.3 s (9L) |
| h^(0) + every 4L (9 ckpts) | 288 MB | 替代 576 MB | **2×** | ~0.6 s (4L) |
| h^(0) + every 3L (12 ckpts) | 384 MB | 替代 576 MB | **1.5×** | ~0.4 s (3L) |
| h^(0) + every 2L (18 ckpts) | 576 MB | 替代 576 MB | **1×** (无收益) | ~0.3 s (2L) |

### 4.3 Gemma-4-31B 具体计算 (4K context, bf16)

Full KV per token per layer = 2 × 16 × 256 × 2 = 16,384 B
Full KV total = 60 × 16384 × 4096 = **3,840 MB**
h^(0) per token = 5376 × 2 = 10,752 B

| 方案 | Checkpoint 存储 | 压缩比 | 重建时间 (估) |
|------|----------------|--------|-------------|
| h^(0) only | 42 MB | **91×** | 全量 60L |
| h^(0) + h^(15) + h^(30) + h^(45) | 168 MB | **22.8×** | 15L |
| h^(0) + every 4L (15 ckpts) | 630 MB | **6.1×** | 4L |
| h^(0) + every 2L (30 ckpts) | 1,260 MB | **3.0×** | 2L |

Gemma-4-31B 有**巨大的压缩空间** (91×)，可以大方使用稀疏 checkpoint 换重建速度。

---

## 5. 架构适配性总结

### 5.1 Tier 分级

| Tier | h^(0) 压缩比 | 架构特征 | 代表模型 | 稀疏 checkpoint |
|------|-------------|---------|---------|-----------------|
| **S** (>50×) | 极优 | MHA 深模型 / GQA 2:1 深模型 | Gemma-4-31B (60×), GPT-2-1.5B (96×) | 大量空间，可换速度 |
| **A** (20-50×) | 优秀 | GQA 2:1 中等深度 | Qwen3-0.6B (28×), Qwen3-1.7B (28×), Gemma-4-E4B (21×) | 有空间做 2-4 个 checkpoint |
| **B** (10-20×) | 良好 | GQA 4:1 标准 | Qwen3-8B (18×), Llama-3.1-8B (16×), Mistral-7B (16×), Llama-3.1-70B (20×) | 谨慎，h^(0)+h^(L/2) 可考虑 |
| **C** (3-10×) | 可用 | GQA 8:1 极深 / MLA | DeepSeek-V2 (6×), DeepSeek-V3 (4.4×) | 仅 h^(0)，不加稀疏 |
| **D** (<3×) | 差 | MQA / MLA 浅模型 | Falcon-7B (0.9×), Falcon-180B (1.4×) | **不适合 h^(0)** |

### 5.2 决策流程

```
1. 计算 compression = 2L × (n_kv / n_q)
2. if compression < 3:  → 不使用 h^(0)，用其他 KV 压缩
3. if compression < 10: → 仅 h^(0)，不加稀疏
4. if compression 10-30: → h^(0) + 可选 1-2 个稀疏 checkpoint
5. if compression > 30: → h^(0) + 积极稀疏 checkpoint (每 4-8 层)
```

### 5.3 稀疏 Checkpoint 推荐配置

| 模型 | 推荐方案 | Checkpoint 层 | 重建加速 | 压缩比 |
|------|---------|--------------|---------|--------|
| Qwen3-8B (18×) | h^(0) only | [0] | 1× | 18× |
| Qwen3-8B (激进) | h^(0) + h^(18) | [0, 18] | 2× | 9× |
| Qwen3-1.7B (28×) | h^(0) + h^(14) | [0, 14] | 2× | 14× |
| Gemma-4-31B (91×) | h^(0) + every 15L | [0, 15, 30, 45] | 4× | 22.8× |
| Llama-3.1-70B (20×) | h^(0) + h^(40) | [0, 40] | 2× | 10× |
| DeepSeek-V3 (4.4×) | h^(0) only | [0] | 1× | 4.4× |

---

## 6. MoE 的影响

MoE 替换 FFN 不替换 Attention，因此:
- **KV per token 不变** (KV 只在 attention 层产生)
- **h^(0) 大小不变** (d_model 与 MoE 无关)
- **压缩比不变** (公式中无 MoE 相关项)

MoE 的间接影响:
- MoE 模型通常**更深** (更多层) → L 增大 → 压缩比提高
- MoE 的 FFN 计算不影响 KV → 重建时 MoE 路由增加重建开销
- Mixtral-8x7B vs Mistral-7B: 同样 32L GQA 4:1, 压缩比相同 (16×)
- 但 Mixtral 重建每层更慢 (8 expert FFN vs 1 FFN)

---

## 7. 与 FlashMLX 实现的对齐

### 7.1 当前实现 (h^(0) only)

```
H0Store: 存储 h^(0), 支持 bf16/q8/q4
KVDirectCache: per-layer cache, recent window + reconstructed evicted
apply_h0_capture: 拦截 embed_tokens → 存 h^(0) → eviction 时重建
reconstruct_prefix_kv: h^(0) → 全量 L 层 forward → 产出所有层 KV
```

### 7.2 需要的扩展 (稀疏 checkpoint)

```
ResidualCheckpointStore: 存储 h^(0), h^(k), h^(2k), ...
  - 每个 checkpoint: (B, T, d_model) bf16/q8/q4
  - 内存: (L/k + 1) × T × d_model × dtype_bytes

apply_h0_capture (扩展):
  - 在 checkpoint_layers 处拦截 h → 存入 ResidualCheckpointStore
  - Phase 2 的 layer loop 中增加 checkpoint capture hook

reconstruct_from_checkpoint(start_layer, end_layer):
  - 从最近的 checkpoint 开始，只跑 start_layer..end_layer-1
  - 重建速度: (end_layer - start_layer) / L × 原速度
```

### 7.3 向后兼容

```python
# h^(0) only (当前行为)
store = ResidualCheckpointStore(checkpoint_layers=[0])  # 等价于 H0Store

# 稀疏 checkpoint
store = ResidualCheckpointStore(checkpoint_layers=[0, 18])  # h^(0) + h^(18)

# 密集 checkpoint (Gemma-4 等高压缩比模型)
store = ResidualCheckpointStore(checkpoint_layers=[0, 15, 30, 45])
```

---

## 8. 端云协同场景分析

### 8.1 五种传输方案

| 方案 | 传输内容 | 端侧模型 | 端侧重建 |
|------|----------|---------|---------|
| **B) Pipeline@cut** | h^(cut) + KV[0:cut-1] int8 | 半模型 (layers cut..L-1) | L-cut 层 |
| **C) h^(0) only** | h^(0) | 全模型 | L 层 |
| **E) Full KV** | 全量 KV int8 | 全模型 | 0 层 |
| **G) [0,cut]+Pipe** | h^(0) + h^(cut) | 全模型 | L 层 (两组各 L/2) |
| **Pipeline Decode** | 每 token h^(cut) | 半模型 | 0 (但每步通信) |

### 8.2 TTFT 公式

```
TTFT = cloud_overhead + transfer_size / bandwidth + edge_recon_time
```

### 8.3 带宽交叉点 (以 Qwen3-1.7B @ 4K 为例)

```
G vs C:
  G 多传输 16 MB, 省重建 626 ms
  交叉点 = 16 MB / 0.626s = 25.6 MB/s ≈ 200 Mbps
  → < 200 Mbps: C (h^(0) only) 赢
  → > 200 Mbps: G ([0,14]+Pipe) 快 ~0.5s

B vs G:
  B 多传输 103 MB, 省重建 667 ms
  交叉点 = 103 MB / 0.667s = 154 MB/s ≈ 1.2 Gbps
  → < 1.2 Gbps: G 远优于 B
  → > 1.2 Gbps: B 略优 (差异 <0.2s)

E vs C:
  E 多传输 236 MB, 省重建 2,847 ms
  交叉点 = 236 MB / 2.847s = 82.9 MB/s ≈ 660 Mbps
  → < 660 Mbps: C 赢
  → > 660 Mbps: E 最优 (零重建)
```

### 8.4 端云方案决策树

```
端侧内存紧张? (装不下全模型)
  ├─ YES → B (Pipeline@cut)
  │         半模型, 传输量大但端侧省内存
  │
  └─ NO → 网络带宽?
           ├─ > 660 Mbps (LAN/光纤)
           │     → E (Full KV)
           │       零重建, TTFT 最低
           │
           ├─ 200-660 Mbps (WiFi6)
           │     → C ≈ G (打平)
           │       推荐 C (更简单)
           │
           └─ < 200 Mbps (5G/4G/3G)
                 → C (h^(0) only)
                   传输最少, TTFT 最优
```

### 8.5 关键结论

1. **h^(0) only (C) 是端云最优解**: 在 < 200 Mbps (绝大多数真实场景) 稳赢
2. **[0,cut]+Pipeline (G) 是有价值的中间地带**: 高速网络下快 ~0.5s，但需要全模型
3. **Pipeline@cut (B) 的独特价值是内存**: 唯一支持半模型方案
4. **Full KV (E) 只在 LAN/光纤场景有意义**: 高带宽下零重建最优
5. **通用公式**: `交叉点带宽 = Δ传输量 / Δ重建时间`

### 8.6 实测验证

| 模型 | 方案 | 5G TTFT | 4G TTFT | 验证 |
|------|------|---------|---------|------|
| Qwen3-8B | B) Pipeline@18 | ~13s | ~56s | REPORT_edge_cloud.md |
| Qwen3-1.7B | B) Pipeline@14 | 12.45s | 55.65s | REPORT_qwen3_1.7b_edge_cloud.md |
| Qwen3-1.7B | C) h^(0) only | **4.23s** | **9.35s** | REPORT_qwen3_1.7b_edge_cloud.md |
| Qwen3-1.7B | G) [0,14]+Pipe | 4.88s | 15.12s | REPORT_qwen3_1.7b_edge_cloud.md |

---

*实验环境: Apple Silicon, MLX, bf16 KV, greedy decode*
*参考论文: arxiv 2603.19664 "Residual Stream is All You Need"*
