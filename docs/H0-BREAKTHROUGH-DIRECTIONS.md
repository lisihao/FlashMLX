# h^(0) 突破方向 — 纯 GPU 双集群 vs GPU+LPU

> 基于 residual_delta_analysis 实验数据
> 日期: 2026-04-03

## 1. 实验证伪: 低秩残差近似不可用

### 假说

h^(l) = h^(0) + δ^(l)，若 δ 是低秩的，可以用 h^(0) + 压缩的 δ 替代完整 KV 传输。

### 实验结果 (Qwen3-1.7B, 2048 tokens)

**好消息: δ^(l) 确实低秩**
- rank@99% 平均仅 29 (max 64)
- rank=64 → 19.3 MB 传输，vs KV 197 MB = **10.2× 压缩**

**坏消息: 生成质量完全崩溃**

| rank | 传输量 | vs KV | h^(l) 误差 | 生成匹配 |
|------|--------|-------|-----------|---------|
| 64 | 19.3 MB | 10.2× | avg 5.0% | 0/50 |
| 128 | 31.7 MB | 6.2× | avg 2.8% | 8/50 |
| 256 | 56.3 MB | 3.5× | avg 1.2% | 9/50 |

### 根因

**Attention 的 softmax 是指数函数，对 K/V 扰动极度敏感:**
```
attention = softmax(Q @ K^T / √d)
         = exp(q·k / √d) / Σ exp(q·k_i / √d)

K 有 1% 误差 → q·k 有 ~1% 误差 → exp(q·k) 有 ~e^0.01 - 1 ≈ 1% 误差
但 softmax 归一化后，相对排序可能完全翻转 → attention pattern 崩溃
```

**结论: 即使 99.9% 的能量被捕获，attention 仍依赖剩余 0.1% 的方向信息。低秩近似这条路不通。**

---

## 2. 核心矛盾

```
h^(0) 小 (18× 压缩) ←→ 从 h^(0) 恢复 KV 需要完整 forward pass (= prefill)

传输小 + 计算大: h^(0) 传输
传输大 + 计算小: KV 传输
```

**没有免费的午餐**: 要么传大数据 (KV)，要么做大计算 (reconstruction)。

---

## 3. 真正的突破方向

### 方向 A: 模型架构创新 — h^(0)-Native Attention

**核心思想**: 训练一个模型，使所有层的 K/V 都从 h^(0) 投影，而非从逐层残差。

```python
# 标准 Transformer (无法压缩)
K_l = norm(h^(l)) @ W_K_l   # 每层依赖不同的 h^(l)

# h^(0)-Native (可压缩!)
K_l = norm(h^(0)) @ W_K_l   # 所有层都从 h^(0) 投影
# Q 仍然从 h^(l) 计算 (保持层间信息传递)
Q_l = norm(h^(l)) @ W_Q_l
```

**效果**:
- KV cache = N × d_model × 2 bytes (只存 h^(0), 所有层共享)
- 传输: 只传 h^(0) + W_K/W_V 权重
- 压缩比: = 标准 KV / h^(0) = **18× (8B) 到 28× (1.7B)**
- Decode 集群: 从 h^(0) 投影 KV ~120ms (vs 4.5s full recon)

**可行性**:
- 类似 Perceiver / encoder-decoder cross-attention
- 需要重新训练 (或微调 + LoRA adapter)
- 可能有质量损失: 深层的 K/V 不再包含上下文理解
- 需要论文级实验验证

**研究价值**: 高。如果质量可接受，这是真正的 10× 突破。

### 方向 B: KV Cache 量化压缩 (工程可行, 增量优化)

**不依赖 h^(0)，直接压缩 KV cache 传输:**

| 方案 | 压缩比 | 质量损失 | 成熟度 |
|------|--------|---------|--------|
| int8 KV | 2× | ~0 | 已有论文验证 |
| int4 KV (per-group) | 4× | ~0.5% PPL | 多篇论文 |
| int2 KV (outlier-aware) | 8× | ~1-2% PPL | 前沿研究 |
| MQA/GQA (架构级) | 4-8× | 训练时补偿 | Llama 3 已用 |
| Token eviction (H2O) | 2-10× | 取决于策略 | 活跃研究 |

**最实用组合: int4 KV + token eviction**
- 8B, 4K tokens: 576 MB → ~30 MB (19× 压缩)
- 不需要 h^(0)，不需要重训练
- 但: 这是增量优化，不是架构创新

### 方向 C: 投机重建 (Speculative KV Reconstruction)

**核心思想**: 不等完整 KV，先用近似 KV 开始生成，后台校正。

```
Prefill 集群:
  t=0    开始 prefill
  t=0.5s 发送 h^(0) (32 MB) → Decode 集群
  t=4.5s 发送精确 KV (576 MB) → Decode 集群

Decode 集群:
  t=0.5s 收到 h^(0) → 用 h^(0) @ W_K 生成近似 KV → 开始投机生成
  t=1.0s 已经生成了 ~20 个投机 token
  t=5.5s 收到精确 KV → 验证投机 token → 保留匹配的，丢弃不匹配的
```

**效果**:
- TTFT ≈ h^(0) 传输时间 (0.5s)，而非 KV 传输时间 (5.5s)
- 投机 token 如果 50% 被接受 → 白赚 10 个 token
- 如果 0% 被接受 → 浪费 0.5s 计算，但 TTFT 不受影响 (用户看到即时响应)

**可行性**: 高。不需要重训练，可以在现有架构上实现。

### 方向 D: 层间流水线 + h^(0) 预热 (Pipeline Parallelism)

**核心思想**: 不在 prefill/decode 之间切，而是在层之间切。

```
集群 P (前 18 层):
  Prefill + Decode 的 layers 0-17
  存 layers 0-17 的 KV cache (288 MB)

集群 D (后 18 层):
  Prefill + Decode 的 layers 18-35
  存 layers 18-35 的 KV cache (288 MB)

通信: 每层边界传递 h^(l) = d_model × 2 = 8 KB/token
```

**h^(0) 的角色**:
- 集群 P 内部: h^(0) 缓存 → 重复 prompt 跳过重新 embed
- 集群 P → D: 传 h^(18)，只有 8 KB/token
- 每个集群只需一半模型权重 + 一半 KV cache

**问题**: Decode 每个 token 需要跨集群 round-trip (延迟)
- InfiniBand 400G: ~5μs latency → 36 round-trips/token = 180μs
- 对 45 tok/s target (22ms/token)，通信占 0.8%

**可行性**: 中高。标准 pipeline parallelism，但需要低延迟互联。

### 方向 E: h^(0) 作为"上下文指纹" (Context Fingerprint)

**换个思路: h^(0) 不用于传输，而用于路由和缓存。**

```
架构:
  Prefill 集群: 计算 hash(h^(0)) 作为上下文指纹
  KV Cache Pool: 分布式 KV 缓存 (key = h^(0) fingerprint)
  Decode 集群: 用指纹查找 → 命中则零传输，未命中则走 KV 传输

h^(0) fingerprint 特性:
  - 语义感知: 相似 prompt → 相近的 h^(0) → 可做近似匹配
  - 增量友好: 追加 tokens → h^(0) 追加 → 可做前缀匹配
  - 极小: 整个 fingerprint 可以用 h^(0) 的 PCA 降到 256 bytes
```

**效果**:
- 热门 prompt (RAG 文档, system prompt): 100% 缓存命中 → 零传输
- 相似 prompt: 前缀匹配 → 只传增量 KV
- 冷启动: 降级到全量 KV 传输

---

## 4. 综合评估

| 方向 | 压缩比 | 需重训练 | 质量风险 | 实现复杂度 | 突破性 |
|------|--------|---------|---------|-----------|--------|
| A: h^(0)-Native | **18×** | 是 | 中 | 高 | **革命性** |
| B: KV 量化 | 4-8× | 否 | 低 | 低 | 增量 |
| C: 投机重建 | ∞ (延迟) | 否 | 低 | 中 | **高** |
| D: 层间流水线 | 2× (每节点) | 否 | 零 | 中 | 中 |
| E: 上下文指纹 | ∞ (命中时) | 否 | 零 | 中 | **高** |

### 推荐路径

```
立即可做 (0-2周):
  C + E: 投机重建 + h^(0) 缓存指纹
  → TTFT 降 10×, 热 prompt 零传输

中期 (1-3月):
  B + D: int4 KV + 层间流水线
  → 4-8× KV 压缩 + 分布式推理

长期研究 (3-6月):
  A: h^(0)-Native 模型架构
  → 如果质量可接受，这是真正的 game changer
```

---

## 5. vs GPU+LPU (NVIDIA AFD) 的竞争力

| 维度 | NVIDIA AFD | 纯 GPU 双集群 + h^(0) |
|------|-----------|----------------------|
| 硬件成本 | Groq 3 LPU + Vera Rubin | 任何两组 GPU |
| Decode 延迟 | ~3ms/token (SRAM 150 TB/s) | ~22ms/token (HBM 2-4 TB/s) |
| 长上下文扩展 | GPU 侧 HBM 扩展 | 同 (KV 在 GPU HBM) |
| Prefill 效率 | GPU batch | GPU batch + h^(0) 缓存 |
| 每 token 通信 | GPU↔LPU 每层 round-trip | 集群间仅 KV 传输一次 |
| 可用性 | 2027+ (Groq 3 产品化) | **今天** |

**纯 GPU 方案的核心优势: 可用性。** 不需要等 NVIDIA 的下一代硬件。

**h^(0) 的独特价值不在传输压缩，而在:**
1. **缓存指纹** — 语义感知的 KV 缓存查找
2. **投机重建** — 用近似 KV 立即开始生成
3. **Server 侧 KV 内存管理** — 128K ctx 省 11× HBM
4. **未来架构研究** — h^(0)-Native 模型的训练基础
