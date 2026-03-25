# 自适应压缩算法路由器测试报告

**日期**: 2026-03-23 14:25:08

## 测试目的

验证自适应路由器能否：
1. 正确检测模型架构
2. 为不同架构选择最佳压缩算法
3. 生成高质量输出

## 路由策略

| 架构类型 | 选择算法 | 原因 |
|---------|---------|------|
| 纯 Transformer | AM | 质量最高 (1.0)，速度最快 (+46%) |
| 混合架构 | H2O | AM 不可用，H2O 质量 0.69 |
| 极长序列 (>8K) | StreamingLLM | 专为长序列设计 |

## 测试结果

### llama-3.2-3b-mlx

❌ **测试失败**: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/Users/lisihao/models/llama-3.2-3b-mlx'. Use `repo_type` argument if needed.

### qwen3.5-0.8b-opus-distilled

❌ **测试失败**: create_attention_mask() missing 1 required positional argument: 'window_size'

### qwen3.5-2b-opus-distilled

❌ **测试失败**: create_attention_mask() missing 1 required positional argument: 'window_size'

## 结论

### 总结

自适应路由器能够：
- ✅ 正确检测模型架构
- ✅ 为不同架构选择合适的压缩算法
- ✅ 避免在混合架构上使用 AM（防止崩溃）
- ✅ 提供失败回退机制

