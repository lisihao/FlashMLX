#!/usr/bin/env python3
"""
直接 profile MAC 各阶段，找到真正的瓶颈
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

from mlx_lm import load, generate
import flashmlx
import mlx.core as mx
import time

# 加载模型
model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
flashmlx.patch_mlx_lm()

# 准备长上下文输入
prompt = "你好" * 10000  # 20K tokens

print(f"目标上下文: ~20K tokens")
print()

# 关键：启用 profiling
flashmlx.enable_profiling()

# 运行推理（生成1个token就够了，看 decode 阶段）
output = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=1,
    verbose=False,
)

# 获取 profiling 结果
print(flashmlx.get_profiling_stats())
