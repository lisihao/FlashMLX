#!/usr/bin/env python3
"""
测试 MAC 在长上下文下的性能
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

from mlx_lm import load, generate
import flashmlx
import time

print("加载模型...")
model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
print("✅ 模型加载完成\n")

# 测试长上下文
prompt = "你好" * 5000  # ~5000 tokens
actual_len = len(tokenizer.encode(prompt))
print(f"提示长度: {actual_len:,} tokens\n")

# 测试1: 标准
print("[1/2] 标准 attention...")
flashmlx.unpatch_mlx_lm()
t0 = time.time()
_ = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
t_std = time.time() - t0
print(f"      耗时: {t_std:.2f}s, 生成速度: {10/t_std:.1f} tok/s\n")

# 测试2: MAC
print("[2/2] MAC-Attention...")
flashmlx.patch_mlx_lm()
t0 = time.time()
_ = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
t_mac = time.time() - t0
print(f"      耗时: {t_mac:.2f}s, 生成速度: {10/t_mac:.1f} tok/s\n")

print(f"加速比: {t_std/t_mac:.2f}×")
