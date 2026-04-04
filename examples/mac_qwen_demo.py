#!/usr/bin/env python3
"""
MAC-Attention + Qwen 演示

一行代码启用 MAC-Attention，加速长上下文推理！
"""

import sys
sys.path.insert(0, "src")

# 1. 启用 MAC-Attention (一行代码！)
import flashmlx
flashmlx.patch_mlx_lm()

# 2. 正常使用 mlx-lm (完全兼容)
from mlx_lm import load, generate
import time

print("=" * 80)
print("MAC-Attention + Qwen 演示")
print("=" * 80)
print()

# 加载模型
model_path = "mlx-community/Qwen2.5-7B-Instruct-4bit"
print(f"加载模型: {model_path}")
print("(第一次运行会下载模型，请耐心等待...)")
print()

model, tokenizer = load(model_path)

print("✅ 模型加载完成")
print()

# 测试场景 1: 短对话 (MAC 可能无优势)
print("=" * 80)
print("测试 1: 短对话 (4K tokens)")
print("=" * 80)
prompt_short = "你好，" * 1000  # ~4K tokens

print("生成中...")
t0 = time.time()
response = generate(
    model, tokenizer, 
    prompt=prompt_short,
    max_tokens=50,
    verbose=False
)
t_short = time.time() - t0

print(f"✅ 完成！耗时: {t_short:.2f}s")
print(f"响应: {response[:100]}...")
print()

# 测试场景 2: 长对话 (MAC 应该加速)
print("=" * 80)
print("测试 2: 长对话 (32K tokens)")
print("=" * 80)
prompt_long = "这是一个很长的上下文。" * 8000  # ~32K tokens

print("生成中...")
t0 = time.time()
response = generate(
    model, tokenizer,
    prompt=prompt_long,
    max_tokens=50,
    verbose=False
)
t_long = time.time() - t0

print(f"✅ 完成！耗时: {t_long:.2f}s")
print(f"响应: {response[:100]}...")
print()

# 总结
print("=" * 80)
print("性能总结")
print("=" * 80)
print(f"短对话 (4K):  {t_short:.2f}s")
print(f"长对话 (32K): {t_long:.2f}s")
print()
print("💡 提示:")
print("  - MAC-Attention 在长上下文 (>16K) 时效果最好")
print("  - 如果要禁用: flashmlx.unpatch_mlx_lm()")
print()

# 禁用演示
print("=" * 80)
print("禁用 MAC-Attention")
print("=" * 80)
flashmlx.unpatch_mlx_lm()
print("✅ 已恢复标准 attention")
print()
