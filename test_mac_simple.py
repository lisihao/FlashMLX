#!/usr/bin/env python3
"""
简单测试：验证 MAC 是否被调用
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

print("=" * 80)
print("MAC-Attention 简单验证")
print("=" * 80)

# 加载模型
from mlx_lm import load, generate
print("\n加载模型...")
model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
print("✅ 模型加载完成\n")

# 启用 MAC
import flashmlx
flashmlx.patch_mlx_lm()

# 短提示 - 触发 decode
prompt = "你好" * 100  # ~100 tokens
print(f"提示: {len(tokenizer.encode(prompt))} tokens")
print("生成 10 tokens...\n")

response = generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=10,
    verbose=True  # 看看输出
)

print(f"\n响应: {response[:200]}...")
print("\n检查是否看到 [MAC DEBUG] 输出")
