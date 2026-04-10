#!/usr/bin/env python3
"""
测试真实推理过程中的 hit rate
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

from mlx_lm import load, generate
import flashmlx

model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
flashmlx.patch_mlx_lm()

# 长上下文
prompt = "你好" * 5000  # ~10K tokens

print("生成 20 个 token，观察真实 hit rate...")
print()

output = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=20,
    verbose=False,
)

print()
print("=" * 80)
print("最终统计:")
mac = flashmlx.patch._global_mac_wrapper
if mac and hasattr(mac, '_hit_count'):
    final_hit_rate = mac._hit_count / (mac._total_count * mac.num_heads)
    print(f"Total decode calls: {mac._total_count}")
    print(f"Total hits (across all heads): {mac._hit_count}")
    print(f"Hit rate: {final_hit_rate:.2%}")
    print(f"Ring cache filled: {mac.ring_cache.filled} / {mac.cache_capacity}")
else:
    print("No hit statistics available")
