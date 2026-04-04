# Mistral-7B — Residual Checkpoint 模型卡

## 基本参数

| 参数 | 值 |
|------|-----|
| 架构 | GQA 4:1 |
| 层数 (L) | 32 |
| d_model | 4096 |
| n_q_heads | 32 |
| n_kv_heads | 8 |
| head_dim | 128 |
| sliding_window | 4096 (部分版本) |

## h^(0) Residual Checkpoint

| 指标 | 值 |
|------|-----|
| h^(0) per token | 8,192 B |
| KV per token per layer | 4,096 B |
| Full KV per token (32L) | 131,072 B |
| **h^(0) 压缩比** | **16×** |
| Tier | **B** (良好) |

## 稀疏 Checkpoint

| 指标 | 值 |
|------|-----|
| 最小间隔 k | **3** |
| 推荐方案 | **h^(0) only** |

同样适用于 Mixtral-8x7B (MoE, 但 attention 参数相同)。

## FlashMLX 配置

```python
checkpoint_layers = [0]  # 16×
```
