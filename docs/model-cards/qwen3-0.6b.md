# Qwen3-0.6B — Residual Checkpoint 模型卡

## 基本参数

| 参数 | 值 |
|------|-----|
| 架构 | GQA 2:1 |
| 层数 (L) | 28 |
| d_model | 1024 |
| n_q_heads | 16 |
| n_kv_heads | 8 |
| head_dim | 64 |
| 量化版本 | Qwen3-0.6B-MLX-4bit |

## h^(0) Residual Checkpoint

| 指标 | 值 |
|------|-----|
| h^(0) per token | 2,048 B |
| KV per token per layer | 2,048 B |
| Full KV per token (28L) | 57,344 B |
| **h^(0) 压缩比** | **28×** |
| Tier | **A** (优秀) |

## 稀疏 Checkpoint

| 指标 | 值 |
|------|-----|
| 最小间隔 k | **2** |
| 推荐方案 | **h^(0) + h^(14)** (14×, 2× 加速) |

## FlashMLX 配置

```python
checkpoint_layers = [0]       # 保守: 28×
checkpoint_layers = [0, 14]   # 推荐: 14×, 2× faster
```
