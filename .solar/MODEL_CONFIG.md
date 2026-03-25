# FlashMLX 测试模型配置

**固定使用的模型，不要再换！**

## 主测试模型

### Qwen3-8B-MLX
- **路径**: `/Volumes/toshiba/models/qwen3-8b-mlx`
- **格式**: MLX safetensors
- **大小**: ~8.1 GB (5.0GB + 3.1GB)
- **用途**:
  - KV Cache 压缩测试
  - 真实推理数据收集
  - 质量基准测试
- **备注**:
  - 已验证可用
  - 包含完整 tokenizer
  - MLX 原生支持

## 备用模型（如需要）

### Qwen3.5-1.5B-Instruct (混合架构测试)
- **路径**: 待确定
- **用途**: 混合架构（Attention + SSM）压缩研究

## 使用规则

1. **默认**: 所有测试使用 Qwen3-8B-MLX
2. **路径**: 在代码中使用 `DEFAULT_MODEL_PATH` 常量
3. **不要换**: 除非有明确理由，不要更换模型

## 代码中的配置

```python
# tests/test_real_data_compression.py
DEFAULT_MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

# 使用时
model_path = DEFAULT_MODEL_PATH
model, tokenizer = load(model_path)
```

---

**最后更新**: 2026-03-23
**更新原因**: 固化模型配置，避免重复更换
