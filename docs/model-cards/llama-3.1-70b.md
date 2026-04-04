# Llama-3.1-70B — Residual Checkpoint 模型卡

## 基本参数

| 参数 | 值 |
|------|-----|
| 架构 | GQA 8:1 |
| 层数 (L) | 80 |
| d_model | 8192 |
| n_q_heads | 64 |
| n_kv_heads | 8 |
| head_dim | 128 |

## h^(0) Residual Checkpoint

| 指标 | 值 |
|------|-----|
| h^(0) per token | 16,384 B |
| KV per token per layer | 4,096 B |
| Full KV per token (80L) | 327,680 B |
| **h^(0) 压缩比** | **20×** |
| Tier | **B** (良好) |

## 稀疏 Checkpoint

| 指标 | 值 |
|------|-----|
| 最小间隔 k | **5** (k > 64/(2×8) = 4) |
| h^(0) + h^(40) 压缩比 | 10× (重建 2× 加速) |
| h^(0) + h^(20) + h^(40) + h^(60) 压缩比 | 5× (重建 4× 加速) |
| 推荐方案 | **h^(0) + h^(40)** (10× 压缩, 2× 加速) |

注意: 70B 模型重建时间很长 (80L forward)，稀疏 checkpoint 的重建加速价值更高。

## FlashMLX 配置

```python
checkpoint_layers = [0]           # 保守: 20×
checkpoint_layers = [0, 40]       # 推荐: 10×, 2× faster
checkpoint_layers = [0, 20, 40, 60]  # 激进: 5×, 4× faster
```
