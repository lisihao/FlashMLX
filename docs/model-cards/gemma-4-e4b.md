# Gemma-4-E4B — Residual Checkpoint 模型卡

## 基本参数

| 参数 | 值 |
|------|-----|
| 架构 | GQA 4:1 + Hybrid + KV Shared Layers |
| 层数 (L) | 42 |
| d_model | 2560 |
| n_q_heads | 8 |
| n_kv_heads | 2 |
| head_dim | 256 |
| num_kv_shared_layers | 18 |
| max_context | 128K |

## h^(0) Residual Checkpoint

| 指标 | 值 |
|------|-----|
| h^(0) per token | 5,120 B (2560 × 2B) |
| KV per token per layer | 2,048 B (2 × 2 × 256 × 2B) |
| Full KV per token (42L) | 86,016 B |
| **h^(0) 压缩比** | **16.8×** |
| Tier | **B** (良好) |

注意: 18 个 KV shared layers 意味着实际独立 KV 层数 < 42。
有效压缩比可能更低 (KV shared layers 已经是"免费"压缩)。

## 稀疏 Checkpoint

| 指标 | 值 |
|------|-----|
| 最小间隔 k | **3** (k > 8/(2×2) = 2) |
| 推荐方案 | **h^(0) + h^(21)** |

## FlashMLX 配置

```python
checkpoint_layers = [0]       # 保守
checkpoint_layers = [0, 21]   # 推荐
```
