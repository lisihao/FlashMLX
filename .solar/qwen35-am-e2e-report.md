# Qwen3.5 系列 AM 压缩端到端测试报告

**日期**: 2026-03-23 11:51:40

## 测试目的

验证 AM 压缩在 Qwen3.5 混合架构上是否真的能用于推理

**关键区别**：
- 之前的测试只计算压缩质量（cosine similarity）
- 本测试在**真实推理**中使用 AM 压缩，检查是否产生乱码

## 测试方法

- 在 Attention 层使用 CompactedKVCache (AM 压缩)
- 在 SSM 层使用标准 KVCache (无压缩)
- Compression ratio: 2.0
- Max tokens: 100

## 测试结果

### qwen3.5-0.8b-opus-distilled

❌ 测试失败: create_attention_mask() missing 1 required positional argument: 'window_size'

### qwen3.5-2b-opus-distilled

❌ 测试失败: create_attention_mask() missing 1 required positional argument: 'window_size'

### qwen3.5-35b-mlx

❌ 测试失败: create_attention_mask() missing 1 required positional argument: 'window_size'

## 总结

### 测试失败的模型

- qwen3.5-0.8b-opus-distilled
- qwen3.5-2b-opus-distilled
- qwen3.5-35b-mlx

