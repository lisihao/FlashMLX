# FlashMLX 模型路径配置

## 本地模型路径（Toshiba 外接硬盘）

**重要：所有模型都已下载，禁止重复下载！**

### Qwen3.5-35B-A3B (MLX 格式)

- **路径**: `/Volumes/toshiba/models/qwen3.5-35b-mlx/`
- **格式**: MLX SafeTensors (4 个分片)
- **大小**: 约 19GB
- **用途**: FlashMLX 和 ThunderOMLX 的主要测试模型

### 其他可用模型

- **Qwen3.5-35B-A3B (GGUF)**: `/Volumes/toshiba/models/qwen3.5-35b-a3b-gguf/` (llama.cpp)
- **Qwen3-30B-A3B (GGUF)**: `/Volumes/toshiba/models/qwen3-30b-a3b-gguf/` (llama.cpp)
- **Qwen3-8B (MLX)**: `/Volumes/toshiba/models/qwen3-8b-mlx/`
- **Llama-3.2-3B (MLX)**: `/Volumes/toshiba/models/llama-3.2-3b-mlx/`

## 使用方式

### Python (MLX-LM)

```python
from mlx_lm import load

# 使用本地路径
model, tokenizer = load("/Volumes/toshiba/models/qwen3.5-35b-mlx/")
```

### 环境变量（可选）

```bash
export FLASHMLX_MODEL_PATH="/Volumes/toshiba/models/qwen3.5-35b-mlx/"
```

---

**最后更新**: 2026-03-21
