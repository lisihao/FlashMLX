# Qwen3-1.7B — Residual Checkpoint 模型卡

## 基本参数

| 参数 | 值 |
|------|-----|
| 架构 | GQA 2:1 |
| 层数 (L) | 28 |
| d_model | 2048 |
| n_q_heads | 16 |
| n_kv_heads | 8 |
| head_dim | 128 |
| 量化版本 | Qwen3-1.7B-MLX-4bit |

## h^(0) Residual Checkpoint

| 指标 | 值 |
|------|-----|
| h^(0) per token | 4,096 B |
| KV per token per layer | 4,096 B |
| Full KV per token (28L) | 114,688 B |
| **h^(0) 压缩比** | **28×** |
| Tier | **A** (优秀) |

## 稀疏 Checkpoint

| 指标 | 值 |
|------|-----|
| 最小间隔 k | **2** (k > 16/(2×8) = 1) |
| h^(0) + h^(14) 压缩比 | 14× (重建 2× 加速) |
| h^(0) + h^(7) + h^(14) + h^(21) 压缩比 | 7× (重建 4× 加速) |
| 推荐方案 | **h^(0) + h^(14)** (14× 压缩 + 2× 加速) |

## FlashMLX 配置

```python
# 保守
checkpoint_layers = [0]  # 28× compression

# 推荐
checkpoint_layers = [0, 14]  # 14× compression, 2× faster recon

# 激进
checkpoint_layers = [0, 7, 14, 21]  # 7× compression, 4× faster recon
```

## 实测数据

来源: FlashMLX dual-instance h^(0) 实验

### 单设备基准

| N tokens | Recon Time | Recon/Prefill | TG Speed | Peak Mem Saving |
|----------|-----------|---------------|----------|----------------|
| 1,024 | 492 ms | 82% | 156 t/s (0% Δ) | -15% |
| 2,048 | 958 ms | 81% | 147 t/s (0% Δ) | -19% |
| 2,751 | 1,348 ms | 84% | 139 t/s (0% Δ) | -34% |

### Sparse Checkpoint 对比 (2K context)

| 配置 | 传输量 | 压缩比 | 重建耗时 | 质量 |
|------|--------|--------|---------|------|
| [0] | 8 MB | 28× | 935 ms | EXACT |
| [0,14] | 16 MB | 14× | 1,025 ms | EXACT |
| [0,7,14,21] | 32 MB | 7× | 989 ms | EXACT |
| [0,4,8,...,24] | 56 MB | 4× | 1,033 ms | EXACT |

> 单设备总重建时间几乎相同 (~935-1033ms)。Sparse 的价值在端云分组传输。

### 端云协同 TTFT (4K context)

| 方案 | 传输量 | 端侧模型 | 5G TTFT | 4G TTFT |
|------|--------|---------|---------|---------|
| B) Pipeline@14 | 135 MB | 458 MB | 12.45s | 55.65s |
| C) h^(0) only | 16 MB | 872 MB | **4.23s** | **9.35s** |
| G) [0,14]+Pipe@14 | 32 MB | 872 MB | 4.88s | 15.12s |
| E) Full KV | 252 MB | 872 MB | 22.26s | 102.9s |

> h^(0) only 在 < 200 Mbps 全赢。[0,14]+Pipe@14 仅在 > 200 Mbps 时快 ~0.5s。
