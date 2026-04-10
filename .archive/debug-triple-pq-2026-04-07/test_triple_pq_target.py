#!/usr/bin/env python3
"""测试triple_pq应该有的压缩率（理论值）"""

import sys
sys.path.insert(0, "mlx-lm-source")

# 理论计算：triple_pq with 4-bit PolarQuant
# KV cache at 16K tokens:
# - Standard (bf16): 16K × 8 heads × 128 dim × 36 layers × 2 (K+V) × 2 bytes = ~4.7 GB
# - PolarQuant 4-bit: 16K × 8 × 128 × 36 × 2 × 0.5 bytes = ~1.2 GB
# - Expected compression: ~75%

# But with Q8 flat buffer instead of 4-bit:
# - Q8 flat: 16K × 8 × 128 × 36 × 2 × 1 byte = ~2.4 GB
# - Expected compression: ~50%

print("triple_pq理论压缩率")
print("="*60)
print("\n假设16K tokens, 8 KV heads, 128 dim, 36 layers:")
print("\nBf16 KV cache: 16384 × 8 × 128 × 36 × 2 × 2 bytes = 4.7 GB")
print("Q8 flat:       16384 × 8 × 128 × 36 × 2 × 1 byte  = 2.4 GB")
print("Q4 (PolarQuant): 16384 × 8 × 128 × 36 × 2 × 0.5 = 1.2 GB")

print(f"\n期望压缩率:")
print(f"  triple_pq + Q8 flat:  ~50% compression")
print(f"  triple_pq + Q4 flat:  ~75% compression")

print(f"\n我的修复结果:")
print(f"  triple_pq + Q8 flat:  ~10% compression ❌")
print(f"\n差距: 应该50%，实际10% → 我的修复破坏了压缩！")
