#!/usr/bin/env python3
"""
分析 MAC attention 的内存使用模式
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load, generate
import flashmlx

print("=" * 80)
print("MAC Attention 内存分析")
print("=" * 80)
print()

# 加载模型
model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
flashmlx.patch_mlx_lm()

# 测试不同长度下的内存使用
contexts = [
    ("5K", 2500),
    ("10K", 5000),
    ("20K", 10000),
]

print(f"{'Context':<10} {'Peak Mem (GB)':>15} {'Attn Time (ms)':>20} {'Hit Rate':>15} {'Avg Skip':>15}")
print("-" * 100)

for name, repeat in contexts:
    prompt = "你好" * repeat

    # 清空缓存和内存统计
    mx.clear_cache()  # 使用新 API
    mx.reset_peak_memory()  # 使用新 API

    # 重置 MAC 统计器（在生成前）
    mac = flashmlx.patch._global_mac_wrapper
    if mac:
        mac.reset()  # 完整重置

    # 生成（足够的 tokens 让 cache 填充）
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)

    # 获取峰值内存
    peak_mem = mx.get_peak_memory() / (1024**3)  # GB

    # 从 MAC wrapper 获取统计
    if mac and hasattr(mac, '_time_attn') and hasattr(mac, '_total_count') and mac._total_count > 0:
        avg_attn_time = mac._time_attn / mac._total_count
        hit_rate = mac._hit_count / (mac._total_count * mac.num_heads) if hasattr(mac, '_hit_count') else 0
        avg_skip = mac._total_skip_ratio / mac._total_count if hasattr(mac, '_total_skip_ratio') else 0
    else:
        avg_attn_time = 0
        hit_rate = 0
        avg_skip = 0

    print(f"{name:<10} {peak_mem:>14.2f}  {avg_attn_time:>19.2f}  {hit_rate:>13.1%}  {avg_skip:>13.1%}")

print()
print("=" * 80)
print("分析:")
print("如果 Peak Memory 随 context 线性增长，说明需要读取整个 KV cache")
print("如果 Attn Time 随 context 增长，说明计算时间不只取决于 partial 长度")
print("=" * 80)
