#!/usr/bin/env python3
"""
抓取 MAC attention 的 Metal trace
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
import flashmlx

# 加载模型
print("加载模型...")
model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
flashmlx.patch_mlx_lm()

# 准备输入（短一点，方便分析）
prompt = "你好" * 1000  # ~2K tokens
tokens = mx.array([tokenizer.encode(prompt)])

print(f"Token 数量: {tokens.shape[1]}")
print()

# 使用 generate 进行 prefill + 几次 decode
print("Prefill 阶段（生成 3 个 token）...")
from mlx_lm import generate

# 先不捕获，让 cache 填充
_ = generate(model, tokenizer, prompt=prompt, max_tokens=3, verbose=False)
print("✅ Cache 已填充")
print()

# 现在捕获第二次生成（这次会用到 MAC）
print("开始 Metal trace 捕获...")
print("捕获第二次生成（应该有高 hit rate）...")
print("=" * 80)

trace_file = "/tmp/mac_trace.gputrace"

try:
    # 开始捕获
    mx.metal.start_capture(trace_file)

    # 再次生成（这次 MAC 应该能 hit）
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=2, verbose=False)

    # 停止捕获
    mx.metal.stop_capture()

    print("✅ Metal trace 已保存到:", trace_file)
    print("=" * 80)

except Exception as e:
    try:
        mx.metal.stop_capture()
    except:
        pass
    print(f"❌ 捕获失败: {e}")

print()
print("打开 trace:")
print(f"  open {trace_file}")
print()
print("或者在 Instruments.app 中打开")
