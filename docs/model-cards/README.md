# FlashMLX Model Cards — Residual Checkpoint 参数

每张模型卡包含 h^(0) residual checkpoint 的关键参数和推荐配置。

## 速查表

| 模型 | Tier | h^(0) 压缩 | 最小间隔 k | 推荐方案 |
|------|------|-----------|-----------|---------|
| [Qwen3-8B](qwen3-8b.md) | B (18×) | 18× | 3 | h^(0) only |
| [Qwen3-1.7B](qwen3-1.7b.md) | A (28×) | 28× | 2 | h^(0) + h^(14) |
| [Qwen3-0.6B](qwen3-0.6b.md) | A (28×) | 28× | 2 | h^(0) + h^(14) |
| [Llama-3.1-8B](llama-3.1-8b.md) | B (16×) | 16× | 3 | h^(0) only |
| [Llama-3.1-70B](llama-3.1-70b.md) | B (20×) | 20× | 5 | h^(0) + h^(40) |
| [Gemma-4-31B](gemma-4-31b.md) | S (60×) | 60× | 2 | h^(0) + every 15L |
| [Gemma-4-E4B](gemma-4-e4b.md) | A (21×) | 21× | 3 | h^(0) + h^(21) |
| [Mistral-7B](mistral-7b.md) | B (16×) | 16× | 3 | h^(0) only |
| [DeepSeek-V3](deepseek-v3.md) | C (4.4×) | 4.4× | - | h^(0) only (marginal) |

## 公式

```
h^(0) 压缩比 = 2L × (n_kv / n_q)
最小间隔 k > n_q / (2 × n_kv)
稀疏压缩比(间隔k) = 2k × (n_kv / n_q)
```

## Tier 定义

- **S** (>50×): 极优，可大量使用稀疏 checkpoint
- **A** (20-50×): 优秀，可适度使用稀疏 checkpoint
- **B** (10-20×): 良好，谨慎使用稀疏 checkpoint
- **C** (3-10×): 可用，仅 h^(0)
- **D** (<3×): 不适合 h^(0)

## 实验报告

| 报告 | 模型 | 内容 |
|------|------|------|
| [端云协同 (Qwen3-8B)](../experiments/dual_instance_h0/REPORT_edge_cloud.md) | Qwen3-8B | Pipeline split + scored_pq |
| [数据中心 PD (Qwen3-8B)](../experiments/dual_instance_h0/REPORT_datacenter_pd.md) | Qwen3-8B | PD 分离 + roofline 分析 |
| [端云 + Sparse (Qwen3-1.7B)](../experiments/dual_instance_h0/REPORT_qwen3_1.7b_edge_cloud.md) | Qwen3-1.7B | h^(0) vs Pipeline vs [0,14]+Pipe |
| [数据中心 PD v2 (Qwen3-1.7B)](../experiments/dual_instance_h0/REPORT_datacenter_pd_v2.md) | Qwen3-1.7B | Batch吞吐 + 闲置compute + 并行重建 + KV爆炸 + Roofline |
| [数据中心 PD v2 (Qwen3-8B)](../experiments/dual_instance_h0/REPORT_datacenter_pd_v2_8b.md) | Qwen3-8B | Batch吞吐 + 闲置compute + 并行重建 + KV爆炸 + 跨模型对比 |
| [架构适配性分析](../residual-checkpoint-architecture-analysis.md) | 15+ 模型 | 通用公式 + tier 分级 + 端云决策树 |
| [**Rubin R100 投影分析**](../experiments/dual_instance_h0/REPORT_rubin_r100_projection.md) | Qwen3-8B + 70B | R100 PD池投影 + 扇出比 + session密度 + 跨机架部署 |
