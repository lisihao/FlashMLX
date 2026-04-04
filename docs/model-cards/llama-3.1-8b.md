# Llama-3.1-8B — Residual Checkpoint 模型卡

## 基本参数

| 参数 | 值 |
|------|-----|
| 架构 | GQA 4:1 |
| 层数 (L) | 32 |
| d_model | 4096 |
| n_q_heads | 32 |
| n_kv_heads | 8 |
| head_dim | 128 |

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
| 最小间隔 k | **3** (k > 32/(2×8) = 2) |
| h^(0) + h^(16) 压缩比 | 8× (重建 2× 加速) |
| 推荐方案 | **h^(0) only** (16× 够用) |

## FlashMLX 配置

```python
checkpoint_layers = [0]       # 推荐: 16×
checkpoint_layers = [0, 16]   # 激进: 8×, 2× faster
```
