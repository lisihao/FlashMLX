# Qwen3-8B — Residual Checkpoint 模型卡

## 基本参数

| 参数 | 值 |
|------|-----|
| 架构 | GQA 4:1 |
| 层数 (L) | 36 |
| d_model | 4096 |
| n_q_heads | 32 |
| n_kv_heads | 8 |
| head_dim | 128 |
| 量化版本 | Qwen3-8B-MLX-4bit (4,394 MB) |

## h^(0) Residual Checkpoint

| 指标 | 值 |
|------|-----|
| h^(0) per token | 8,192 B (d_model × 2B bf16) |
| KV per token per layer | 4,096 B (2 × 8 × 128 × 2B) |
| Full KV per token (36L) | 147,456 B |
| **h^(0) 压缩比** | **18×** |
| Tier | **B** (良好) |

## 稀疏 Checkpoint

| 指标 | 值 |
|------|-----|
| 最小间隔 k | **3** (k > 32/(2×8) = 2) |
| h^(0) + h^(18) 压缩比 | 9× (重建 2× 加速) |
| h^(0) + h^(9) + h^(18) + h^(27) 压缩比 | 4.5× (重建 4× 加速) |
| 推荐方案 | **h^(0) only** (18× 够用，不值得换速度) |

## 4K Context 内存画像

| 方案 | Archive 存储 | 重建时间 |
|------|-------------|---------|
| 全量 KV | 576 MB | 0 |
| h^(0) only | 32 MB | ~5.2 s (36L) |
| h^(0) + h^(18) | 64 MB | ~2.6 s (18L) |

## FlashMLX 配置

```python
# 推荐 (默认)
checkpoint_layers = [0]  # h^(0) only, 18× compression

# 激进 (牺牲压缩换速度)
checkpoint_layers = [0, 18]  # 9× compression, 2× faster recon
```

## 实测数据

来源: FlashMLX dual-instance h^(0) 实验 (2026-04-03)

| N tokens | Recon Time | Recon/Prefill | TG Speed | Peak Mem Saving |
|----------|-----------|---------------|----------|----------------|
| 1,024 | 2,227 ms | 90% | 46 t/s (0% Δ) | -5% |
| 2,048 | 4,503 ms | 91% | 42 t/s (0% Δ) | -11% |
| 2,751 | 6,310 ms | 94% | 40 t/s (0% Δ) | -16% |

### 数据中心 PD v2 实测 (Q8 量化, M4 Pro)

#### Sparse Checkpoint 并行重建 (4K context)

| 配置 | 传输量 | 压缩比 | 单节点重建 | 并行重建 (2D) | 并行重建 (4D) | 质量 |
|------|--------|--------|-----------|--------------|--------------|------|
| [0] | 34 MB | 18× | 13.0 s | — | — | EXACT |
| [0,18] | 68 MB | 9× | 11.7 s | **5.9 s** (2.0×) | — | EXACT |
| [0,9,18,27] | 135 MB | 4.5× | 10.9 s | — | **2.8 s** (3.9×) | EXACT |

#### Batch 放大 (A100 80GB, bf16)

| Context | Full KV batch | h^(0) batch | 放大 |
|---------|--------------|-------------|------|
| 4K | 74 | 414 | 5.6× |
| 8K | 37 | 316 | 8.5× |
| 128K | 2 | 39 | 19.5× |

#### Roofline

| Phase | AI (FLOPs/Byte) | Gap |
|-------|----------------|-----|
| Prefill | 16,026 | — |
| Decode | 3.4 | **4,689×** |
