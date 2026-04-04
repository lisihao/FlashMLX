# DeepSeek-V3 — Residual Checkpoint 模型卡

## 基本参数

| 参数 | 值 |
|------|-----|
| 架构 | MLA (Multi-head Latent Attention) + MoE |
| 层数 (L) | 61 |
| d_model | 7168 |
| latent_kv_dim | 512 |
| n_q_heads | 128 |
| head_dim | 128 |
| MoE experts | 256 (routed 8) |

## h^(0) Residual Checkpoint

| 指标 | 值 |
|------|-----|
| h^(0) per token | 14,336 B (7168 × 2B) |
| MLA KV per token per layer | 1,024 B (latent=512 × 2B) |
| Full KV per token (61L) | 62,464 B |
| **h^(0) 压缩比** | **4.4×** |
| Tier | **C** (可用但低效) |

## 为什么 MLA 是 h^(0) 的天敌

MLA 将 KV 投影到低维 latent space (512 vs d_model=7168)。
KV 已经被 MLA "压缩"了 14×，留给 h^(0) 的空间很小。

```
标准 GQA KV: 2 × n_kv × head_dim × 2B = 大
MLA KV:      latent_dim × 2B = 1024 B    = 小 (已压缩)
h^(0):       d_model × 2B = 14,336 B     = 大 (无法低于 d_model)
```

h^(0) (14 KB) 替代 MLA KV (1 KB/layer × 61 layers = 61 KB) → 仅 4.4×。

## 稀疏 Checkpoint

**不推荐。** 任何额外 checkpoint 都会让压缩比 < 3×。

## 推荐替代方案

- MLA 本身已是高效的 KV 压缩方案
- 考虑 MLA latent cache 量化 (q8/q4) 而非 h^(0)

## FlashMLX 配置

```python
# 谨慎使用
checkpoint_layers = [0]  # 4.4× (仅在内存极紧张时值得)

# 不推荐
# checkpoint_layers = [0, 30]  # 2.2× (不值得)
```
