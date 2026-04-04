# Gemma-4-31B — Residual Checkpoint 模型卡

## 基本参数

| 参数 | 值 |
|------|-----|
| 架构 | GQA 2:1 + Hybrid (Global + Sliding Window) |
| 层数 (L) | 60 |
| d_model | 5376 |
| n_q_heads | 32 |
| n_kv_heads | 16 |
| head_dim | 256 |
| sliding_window | 1024 |
| max_context | 256K |

## h^(0) Residual Checkpoint

| 指标 | 值 |
|------|-----|
| h^(0) per token | 10,752 B (5376 × 2B) |
| KV per token per layer | 16,384 B (2 × 16 × 256 × 2B) |
| Full KV per token (60L) | 983,040 B |
| **h^(0) 压缩比** | **91×** |
| Tier | **S** (极优) |

## 稀疏 Checkpoint

| 指标 | 值 |
|------|-----|
| 最小间隔 k | **2** (k > 32/(2×16) = 1) |
| h^(0) + h^(30) 压缩比 | 46× (重建 2× 加速) |
| h^(0) + h^(15) + h^(30) + h^(45) 压缩比 | 22.8× (重建 4× 加速) |
| h^(0) + every 4L (15 ckpts) 压缩比 | 6.1× (重建 15× 加速) |
| 推荐方案 | **h^(0) + every 15L** (22.8×, 4× 加速) |

## 特殊: Hybrid Attention

Gemma-4 使用混合注意力:
- **Global layers** (~30L): 标准 attention, 全量 KV → h^(0) 替代价值高
- **Sliding window layers** (~30L): KV 上限 1024 tokens → context > 1K 后 KV 不再增长

有效压缩比在长 context 时低于理论值:
- 4K context: ~91× (window 层 KV < 全量)
- 16K context: ~60× (window 层 KV 被截断)
- 256K context: ~45× (window 层 KV 占比缩小)

## 为什么 Gemma-4 是 h^(0) 的理想目标

1. **GQA 2:1**: n_kv/n_q = 0.5, 每层 KV 大 (16,384 B vs h^(0) 10,752 B)
2. **60 层**: 深模型 → 高 L → 高压缩比
3. **head_dim=256**: 增加单层 KV 但同比增加 d_model, 比例不变
4. **256K context**: 长上下文 KV 爆炸 (57.6 GB!)，h^(0) 压缩价值极大

## FlashMLX 配置

```python
# 保守 (最大压缩)
checkpoint_layers = [0]  # 91×

# 推荐 (平衡)
checkpoint_layers = [0, 15, 30, 45]  # 22.8×, 4× faster recon

# 激进 (最快重建)
checkpoint_layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56]  # 6.1×
```
