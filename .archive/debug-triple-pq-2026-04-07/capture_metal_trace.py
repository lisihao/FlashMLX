#!/usr/bin/env python3
"""
抓取 Metal trace - 确认 MLX attention 到底访问了什么
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load, generate
import flashmlx
import os

print("=" * 80)
print("Metal Trace 捕获")
print("=" * 80)
print()

# 检查环境变量
if os.environ.get('MTL_CAPTURE_ENABLED') != '1':
    print("⚠️  需要设置环境变量: MTL_CAPTURE_ENABLED=1")
    print("请运行: MTL_CAPTURE_ENABLED=1 python3 capture_metal_trace.py")
    sys.exit(1)

# 加载模型
print("加载模型...")
model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
flashmlx.patch_mlx_lm()

# 短上下文，方便分析
prompt = "你好" * 1000  # 2K tokens
print(f"上下文: ~2K tokens")
print()

# Prefill（让 cache 填充，建立 hit）
print("Prefill 阶段...")
_ = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
print("✅ Cache 已填充")
print()

# 重置 MAC 统计
mac = flashmlx.patch._global_mac_wrapper
if mac:
    mac.reset()

# 捕获一次 decode
print("开始捕获 Metal trace...")
trace_file = "/tmp/mac_decode_trace.gputrace"

try:
    mx.metal.start_capture(trace_file)

    # 单次生成（应该有高 hit rate）
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=2, verbose=False)

    mx.metal.stop_capture()

    print("✅ Trace 已保存:", trace_file)

    # 打印统计
    if mac and hasattr(mac, '_total_count') and mac._total_count > 0:
        hit_rate = mac._hit_count / (mac._total_count * mac.num_heads)
        avg_skip = mac._total_skip_ratio / mac._total_count
        print(f"   Hit rate: {hit_rate:.1%}")
        print(f"   Avg skip: {avg_skip:.1%}")

except Exception as e:
    try:
        mx.metal.stop_capture()
    except:
        pass
    print(f"❌ 捕获失败: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("打开 trace:")
print(f"  open {trace_file}")
print("或在 Xcode > Open Developer Tool > Instruments 中打开")
print("=" * 80)
