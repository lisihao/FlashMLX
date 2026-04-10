#!/usr/bin/env python3
"""
测试 MAC hit rate - 看 match 效果如何
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

from mlx_lm import load, generate
import flashmlx
import mlx.core as mx

# Patch and load
model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
flashmlx.patch_mlx_lm()

# 生成一些重复性的内容，增加 hit 概率
prompt = "今天天气真好，今天天气真好，" * 5000  # 重复内容

print("生成文本，查看 hit rate...")
output = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=20,  # 生成 20 个 token
    verbose=True,
)

# 查看 global MAC wrapper 的统计
mac = flashmlx.patch._global_mac_wrapper
if mac:
    print(f"\nMAC Statistics:")
    print(f"  Ring cache filled: {mac.ring_cache.filled} / {mac.cache_capacity}")

    # 手动测试一次 match
    q = mx.random.normal((32, 128)).astype(mx.bfloat16)
    q_norm = mac._normalize_query(q)
    hit, left_start = mac._match(q_norm)

    hit_rate = float(hit.mean())
    print(f"  Sample hit rate: {hit_rate:.1%}")
    print(f"  Sample left_start range: [{int(left_start.min())}, {int(left_start.max())}]")
