#!/usr/bin/env python3
"""
测试标准 attention 的速度（不用MAC）
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

from mlx_lm import load, generate

print("加载模型...")
model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
print("✅ 模型加载完成\n")

# 不启用MAC
print("测试标准 attention (100 tokens prompt, 生成 10 tokens)...\n")

prompt = "你好" * 100
response = generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=10,
    verbose=True
)

print(f"\n响应: {response[:100]}...")
