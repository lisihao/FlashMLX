#!/usr/bin/env python3
"""
测试标准 attention 的性能（不用 MAC）
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

from mlx_lm import load, generate
import flashmlx
import time

# 加载模型（不启用 MAC）
model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")

# 准备长上下文输入
prompt = "你好" * 10000  # 20K tokens

print(f"目标上下文: ~20K tokens")
print()

# 启用 profiling（虽然不用 MAC，但可以看其他组件）
flashmlx.enable_profiling()

# 运行推理
output = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=1,
    verbose=False,
)

# 获取 profiling 结果（只有非-MAC 的部分）
print(flashmlx.get_profiling_stats())
