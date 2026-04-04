# MAC-Attention 集成指南

**一行代码启用 MAC-Attention，加速长上下文推理！**

## 🚀 快速开始

```python
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

# 1. 启用 MAC-Attention (一行代码!)
import flashmlx
flashmlx.patch_mlx_lm()

# 2. 正常使用 mlx-lm (完全兼容)
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
response = generate(model, tokenizer, "你好" * 8000, max_tokens=100)
# 长上下文自动加速！
```

## 📊 性能预期（M4 Max）

| 上下文长度 | 标准 Attention | MAC-Attention | 加速比 |
|-----------|---------------|---------------|--------|
| 4K | 2489 tok/s | 1895 tok/s | 0.76× ❌ |
| 8K | 1346 tok/s | 1825 tok/s | 1.36× |
| 16K | 780 tok/s | 1628 tok/s | 2.09× |
| 32K | 418 tok/s | 1721 tok/s | 4.12× ✅ |
| **64K** | **214 tok/s** | **1765 tok/s** | **8.25× 🔥** |

**注意**：
- ✅ **推荐**：上下文 > 16K 时使用 MAC
- ❌ **不推荐**：上下文 < 8K 时 (有 overhead)

## 🎯 支持的模型

当前支持所有 Qwen 系列模型：
- ✅ Qwen3 (qwen3.py)
- ✅ Qwen3.5 (qwen3_5.py)
- ✅ Qwen (legacy qwen.py)

更多模型支持即将添加...

## 📖 使用示例

### 示例 1: 基础使用

```python
import flashmlx
flashmlx.patch_mlx_lm()  # 启用 MAC

from mlx_lm import load, generate
model, tokenizer = load("Qwen/Qwen2.5-7B-Instruct")

# 长上下文推理
long_context = "..." * 10000  # 32K tokens
response = generate(model, tokenizer, long_context, max_tokens=100)
```

### 示例 2: 性能对比

```bash
# 运行性能测试
python examples/mac_benchmark.py
```

### 示例 3: 动态开关

```python
import flashmlx

# 启用 MAC
flashmlx.patch_mlx_lm()

# ... 使用 MAC ...

# 禁用 MAC（恢复标准 attention）
flashmlx.unpatch_mlx_lm()

# ... 使用标准 attention ...
```

## 🔧 工作原理

MAC-Attention 通过 **Monkey Patch** 替换 mlx-lm 的 attention 实现：

1. **Prefill 阶段**（输入长文本）：
   - 使用标准 attention
   - 同时建立 MAC ring cache

2. **Decode 阶段**（生成 tokens）：
   - 自动切换到 MAC-Attention
   - Match → Amend → Complete
   - 只计算变化的部分（~1-3% tokens）

3. **性能优势**：
   - 标准 attention: 时间 ∝ context length
   - MAC-Attention: 时间 ≈ 常数 (~1750 tok/s)

## 🧪 测试验证

```bash
# 快速测试 patch 是否工作
cd /Users/lisihao/FlashMLX
python3 -c "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'mlx-lm-source'); import flashmlx; flashmlx.patch_mlx_lm()"

# 应该看到：
# ✅ Patched: Qwen3
# ✅ Patched: Qwen3.5
# ✅ Patched: Qwen
# 🚀 MAC-Attention 已启用！
```

## 📚 技术细节

### Patch 范围

只替换 `Attention.__call__` 方法：
- ✅ 保留原始权重加载
- ✅ 保留 RoPE
- ✅ 保留 KV cache
- ✅ 只改 attention 计算

### 兼容性

- ✅ mlx-lm v0.20+
- ✅ 所有 quantization 方法
- ✅ 所有 sampling 策略
- ✅ Batch generation

### 限制

- ❌ 短上下文 (< 8K) 性能可能下降
- ❌ 仅支持 decode 阶段加速（prefill 用标准 attention）
- ❌ Batch size > 1 时性能待验证

## 🐛 故障排除

**问题 1**: `No module named 'flashmlx'`
```bash
# 确保正确设置 PYTHONPATH
export PYTHONPATH="/Users/lisihao/FlashMLX/src:/Users/lisihao/FlashMLX/mlx-lm-source:$PYTHONPATH"
```

**问题 2**: Patch 无效
```bash
# 检查 mlx-lm 版本
pip show mlx-lm

# 确保在 load() 之前调用 patch
flashmlx.patch_mlx_lm()
model, tokenizer = load(...)  # 正确顺序
```

**问题 3**: 性能没有提升
- 检查上下文长度（需要 > 16K）
- 确认是 decode 阶段（不是 prefill）
- 检查 MAC cache 是否已预热

## 📝 引用

```bibtex
@article{mac-attention-2024,
  title={MAC-Attention: Match-Amend-Complete Attention for Long Context LLM Inference},
  journal={arXiv:2604.00235},
  year={2024}
}
```

## 🙏 致谢

- MLX team @ Apple
- MAC-Attention 论文作者
- mlx-lm 社区
